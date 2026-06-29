"""All networking: packet parsing, ADB stream reader, UDP receive, Python -> Unity send.

The receive thread updates the gesture/gaze state in `state.py`, and ALSO
remembers the sender's IP so we can send VLM results back without manual config.
"""
import json
import socket
import subprocess
import threading
import time

import numpy as np

from . import config, state
from . import ridge
from . import vlm_client


# =================== PACKET PARSING ===================
def parse_packet(msg: str) -> dict:
    parts = msg.strip().split(",")
    if len(parts) < 4:
        raise ValueError(f"packet too short: {msg!r}")

    pt = parts[0]

    if pt == "GAZE":
        if len(parts) != 7:
            raise ValueError(f"GAZE expects 7 fields, got {len(parts)}")
        return {
            "type": "GAZE",
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "is_tracked": int(parts[3]),
            "gx": float(parts[4]),
            "gy": float(parts[5]),
            "gz": float(parts[6]),
        }

    if pt == "GESTURE_EVENT":
        if len(parts) != 5:
            raise ValueError(f"GESTURE_EVENT expects 5 fields, got {len(parts)}")
        return {
            "type": "GESTURE_EVENT",
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "gesture_name": parts[3],
            "event_type": parts[4],
        }

    if pt == "GESTURE":
        return {"type": "GESTURE", "raw": parts}

    if pt == "ASK_QUESTION":
        # ASK_QUESTION,<seq>,<time>,<question text> -- question may contain commas
        # so we re-join everything past index 2.
        if len(parts) < 4:
            raise ValueError(f"ASK_QUESTION expects at least 4 fields, got {len(parts)}")
        return {
            "type": "ASK_QUESTION",
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "question": ",".join(parts[3:]).strip(),
        }

    if pt == "VOICE_COMMAND":
        # Legacy fallback from Unity. New Android STT flow should prefer the
        # /voice_command HTTP JSON request so image + transcript share a request_id.
        if len(parts) < 4:
            raise ValueError(f"VOICE_COMMAND expects at least 4 fields, got {len(parts)}")
        return {
            "type": "VOICE_COMMAND",
            "seq": int(parts[1]),
            "sender_time": float(parts[2]),
            "transcript": ",".join(parts[3:]).strip(),
        }

    raise ValueError(f"unknown packet type: {pt}")


# =================== ADB STREAM READER THREAD ===================
def stream_reader_loop():
    frame_size = config.STREAM_W * config.STREAM_H * 3

    while not state.stop_event.is_set():
        adb_proc = None
        ffmpeg_proc = None
        try:
            adb_proc = subprocess.Popen(
                config.ADB_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**7,
            )
            ffmpeg_proc = subprocess.Popen(
                config.build_ffmpeg_cmd(config.STREAM_W, config.STREAM_H),
                stdin=adb_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**7,
            )
            print(f"[STREAM] adb|ffmpeg started ({config.STREAM_W}x{config.STREAM_H})")

            while not state.stop_event.is_set():
                raw = ffmpeg_proc.stdout.read(frame_size)
                if not raw or len(raw) != frame_size:
                    print("[STREAM] frame read failed -> restart")
                    break
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (config.STREAM_H, config.STREAM_W, 3)
                )
                with state.frame_lock:
                    state.latest_frame = arr
        except Exception as exc:
            print(f"[STREAM][ERROR] {exc}")
        finally:
            for p in (ffmpeg_proc, adb_proc):
                try:
                    if p and p.stdout is not None:
                        p.stdout.close()
                except Exception:
                    pass
                try:
                    if p:
                        p.kill()
                except Exception:
                    pass
            if not state.stop_event.is_set():
                time.sleep(0.5)  # back-off before restart


# =================== UDP RECEIVER THREAD ===================
def udp_receiver_loop(sock: socket.socket):
    """Drain every UDP packet; route GAZE -> buffer, GESTURE_EVENT -> state machine."""
    sock.settimeout(0.05)

    while not state.stop_event.is_set():
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break

        # Remember the Unity sender's address so we can send results back.
        with state.unity_addr_lock:
            state.last_unity_addr = addr

        try:
            pkt = parse_packet(data.decode("utf-8"))
        except Exception as exc:
            print(f"[UDP][WARN] parse failed: {exc}")
            continue

        ptype = pkt["type"]

        if ptype == "GAZE":
            tracked = pkt["is_tracked"] == 1
            mapped = (
                ridge.map_gaze_dir_to_norm(pkt["gx"], pkt["gy"], pkt["gz"])
                if tracked
                else None
            )
            with state.gaze_lock:
                state.latest_is_tracked = tracked
                state.latest_gaze_norm = mapped
                if (
                    state.is_gesture_active
                    and mapped is not None
                    and not state.gaze_logging_frozen
                ):
                    state.gesture_norm_points.append(mapped)

        elif ptype == "GESTURE_EVENT":
            evt = pkt["event_type"]
            pkt_name = pkt.get("gesture_name") or ""
            with state.gaze_lock:
                # Suppress "Pending" placeholder events whenever a real (non-Pending)
                # gesture is already in flight. PinchStrokeCapture on the Unity side
                # fires Pending START/END/FAIL on every tight pinch, which during a
                # Translate gesture (e.g. when the user re-pinches to close the area)
                # would otherwise reset gesture_norm_points and lose the accumulated
                # gaze trail for Translate.
                if (
                    pkt_name == "Pending"
                    and state.is_gesture_active
                    and state.gesture_name_active not in (None, "Pending")
                ):
                    print(
                        f"[GESTURE] DROP {evt} name=Pending while active="
                        f"{state.gesture_name_active!r} seq={pkt['seq']}"
                    )
                elif evt == "START":
                    state.is_gesture_active = True
                    state.gesture_name_active = pkt["gesture_name"]
                    state.gesture_norm_points = []
                    state.gaze_logging_frozen = False
                    print(
                        f"\n[GESTURE] START name={state.gesture_name_active} "
                        f"seq={pkt['seq']}"
                    )
                elif evt == "READY":
                    # Compare arming marker: freeze the gaze trail here so the
                    # "bring hands together" motion that follows is not logged.
                    # Translate's READY does the same gaze freeze AND triggers
                    # stage-1 OCR via the translate handler. END (after the
                    # confirming swipe) then runs the GPT translation.
                    state.gaze_logging_frozen = True
                    print(
                        f"\n[GESTURE] READY name={pkt_name} "
                        f"seq={pkt['seq']} pts_frozen={len(state.gesture_norm_points)}"
                    )
                    if pkt_name == "Translate":
                        # Snapshot the frame + gaze trail right now, then dispatch
                        # OCR on a background thread so the UDP loop keeps draining.
                        with state.frame_lock:
                            captured = None if state.latest_frame is None else state.latest_frame.copy()
                        snapshot_points = list(state.gesture_norm_points)
                        threading.Thread(
                            target=_run_translate_ocr_stage,
                            args=(captured, snapshot_points, pkt_name),
                            daemon=True,
                        ).start()
                elif evt == "END":
                    # Prefer END's gesture_name over the START-time placeholder
                    # (Unity GestureRouter sets START name to "Pending" and only
                    # finalizes the actual gesture name on END after classification).
                    end_name = pkt.get("gesture_name") or state.gesture_name_active
                    print(
                        f"\n[GESTURE] END   name={end_name} "
                        f"(start_name={state.gesture_name_active}) "
                        f"seq={pkt['seq']} pts={len(state.gesture_norm_points)}"
                    )
                    state.pending_gesture_end = {
                        "gesture_name": end_name,
                        "norm_points": list(state.gesture_norm_points),
                        "ready_at": time.time() + config.CAPTURE_DELAY_AFTER_END,
                    }
                    state.is_gesture_active = False
                    state.gesture_name_active = None
                    state.gesture_norm_points = []
                    state.gaze_logging_frozen = False
                elif evt == "FAIL":
                    failed_name = state.gesture_name_active or pkt.get("gesture_name")
                    print(
                        f"\n[GESTURE] FAIL  name={failed_name} "
                        f"seq={pkt['seq']} pts={len(state.gesture_norm_points)}"
                    )
                    state.pending_gesture_end = None
                    state.last_gesture_fail = {
                        "gesture_name": failed_name,
                        "reason": "Gesture failed or hand tracking lost",
                        "fail_time": time.time(),
                    }
                    state.is_gesture_active = False
                    state.gesture_name_active = None
                    state.gesture_norm_points = []
                    state.gaze_logging_frozen = False

        elif ptype == "ASK_QUESTION":
            question = pkt.get("question", "").strip()
            print(f"\n[ASK_QUESTION] received seq={pkt['seq']} question={question!r}")
            # Dispatch VLM call to a background thread so the UDP loop keeps draining.
            threading.Thread(
                target=process_ask_question,
                args=(question,),
                daemon=True,
            ).start()

        elif ptype == "VOICE_COMMAND":
            transcript = pkt.get("transcript", "").strip()
            print(
                f"\n[VOICE_COMMAND][LEGACY] received seq={pkt['seq']} "
                f"transcript={transcript!r}; routing through cached Ask fallback."
            )
            threading.Thread(
                target=process_ask_question,
                args=(transcript,),
                daemon=True,
            ).start()


# =================== ASK_QUESTION VLM PROCESSING ===================
# When Unity sends ASK_QUESTION, we pair it with the most recently cached Ask target
# (set by handlers/ask.py at gesture END), build a combined prompt with the user's
# follow-up question, call the VLM, and push the result back to Unity over the
# existing VLM_RESULT channel.
ASK_TARGET_TTL_SEC = 120.0  # cached Ask target expires this many seconds after gesture END


def _extract_answer_text(gpt_response) -> str:
    """Pull a usable answer string out of whatever shape GPT came back with."""
    if not isinstance(gpt_response, dict):
        return ""
    for key in ("answer", "response", "text", "raw"):
        v = gpt_response.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def process_ask_question(question: str):
    if not question:
        print("[ASK_QUESTION] empty question; ignoring.")
        return

    with state.ask_lock:
        cached = state.latest_ask_target

    if cached is None:
        print("[ASK_QUESTION] no cached Ask target. Did the user pinch '?' first?")
        _send_ask_error_to_unity(question, "No target. Please point at an object first.")
        return

    age = time.time() - cached.get("timestamp", 0)
    if age > ASK_TARGET_TTL_SEC:
        print(f"[ASK_QUESTION] cached target is stale ({age:.1f}s old); rejecting.")
        _send_ask_error_to_unity(question, "Target expired. Please point again.")
        return

    crop = cached["crop"]
    target_meta = dict(cached["target_meta"])
    match_meta = dict(cached.get("match_meta") or {})
    gesture_name = cached.get("gesture_name", "Ask")
    matched_object = cached.get("matched_object")

    target_meta["user_question"] = question

    base_prompt = getattr(config, "ASK_REFERENCE_PROMPT", "")
    if matched_object is not None:
        known_block = (
            "The object in the image has already been identified for you. "
            "Use this DB info as authoritative ground truth and only fall back "
            "to the image when the question is about something the DB does not "
            "cover (colour, condition, position, etc.).\n"
            f"- name: {matched_object.get('name', '')}\n"
            f"- info: {matched_object.get('result_search', '')}\n"
        )
    else:
        known_block = ""

    prompt = (
        base_prompt
        + ("\n\n" if base_prompt else "")
        + (known_block + "\n" if known_block else "")
        + f"User question about the object:\n{question}\n\n"
        + "Answer the user's question specifically. Respond in the JSON schema above."
    )

    gpt_response = vlm_client.call_vlm_on_crop(crop, prompt)

    # Build the *final* response Unity sees. Even if GPT misbehaves we still
    # surface the DB name so the user-facing card always says the right object.
    db_name = matched_object.get("name", "") if matched_object else ""

    if gpt_response is None:
        answer_text = ""
        ok = False
        error = "vlm_call_failed"
    elif isinstance(gpt_response, dict) and gpt_response.get("error"):
        answer_text = ""
        ok = False
        error = str(gpt_response.get("error"))
    else:
        answer_text = _extract_answer_text(gpt_response)
        ok = bool(answer_text)
        error = "" if ok else "empty_answer"

    final_response = {
        "name": db_name or (gpt_response.get("name", "") if isinstance(gpt_response, dict) else ""),
        "answer": answer_text,
        "user_question": question,
    }
    if not ok:
        final_response["error"] = error

    payload = {
        "timestamp": "",  # filled by save_vlm_response
        "gesture": gesture_name,
        "model": config.OPENAI_MODEL,
        "status": "ok" if ok else "fail",
        "stage": "answer",
        "target_meta": target_meta,
        "match_meta": match_meta,
        "response": final_response,
    }

    # Persist for the audit trail (writes timestamp + the raw GPT response too).
    log_response = dict(final_response)
    log_response["_gpt_raw"] = gpt_response
    saved = vlm_client.save_vlm_response(
        log_response, gesture_name, target_meta, crop, prompt
    )
    if saved is not None:
        payload["timestamp"] = saved.get("timestamp", "")

    send_vlm_result_to_unity(payload)
    print(
        f"[ASK_QUESTION] phase2 sent name={final_response['name']!r} "
        f"answer_len={len(answer_text)} status={payload['status']}"
    )


# =================== TRANSLATE OCR (stage 1, fired on Translate READY) ===================
def _run_translate_ocr_stage(captured_frame, norm_points, gesture_name):
    """Background worker for Translate stage 1. Calls the translate handler's
    do_ocr() which runs OCR, caches the result, and sends a partial VLM_RESULT
    so Unity can show the source text before the user confirms with a swipe."""
    try:
        from .handlers import translate as translate_handler  # lazy: handlers package imports network
        rendered = translate_handler.do_ocr(captured_frame, norm_points, gesture_name)
        with state.target_lock:
            state.target_canvas = rendered
    except Exception as exc:
        print(f"[TRANSLATE-OCR][ERROR] {exc}")


def _send_ask_error_to_unity(question: str, message: str):
    """Push a synthetic error payload back to Unity so the Ask card doesn't hang."""
    payload = {
        "timestamp": "",
        "gesture": "Ask",
        "model": "n/a",
        "status": "fail",
        "stage": "answer",
        "reason": message,
        "target_meta": {"user_question": question},
        "response": {"error": message},
    }
    send_vlm_result_to_unity(payload)


def send_gesture_fail_to_unity(gesture_name: str, reason: str, extra_meta: dict = None):
    """Tell Unity that the gesture handler concluded the gesture is unusable
    (e.g. CLIP couldn't identify the object). Same VLM_RESULT envelope as a
    success, but with status=fail set so the Unity side can show a fail UI.
    """
    from datetime import datetime as _dt
    payload = {
        "timestamp": _dt.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture_name,
        "model": "local",
        "status": "fail",
        "reason": reason,
        "target_meta": extra_meta or {},
        "response": {"error": reason},
    }
    send_vlm_result_to_unity(payload)


# =================== PYTHON -> UNITY SENDER ===================
def init_unity_sender_socket():
    try:
        state.unity_sender_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[UNITY-SEND] Sender socket ready (target port {config.UNITY_RESULT_PORT})")
    except Exception as exc:
        state.unity_sender_sock = None
        print(f"[UNITY-SEND][ERROR] socket creation failed: {exc}")


def _resolve_unity_host():
    if config.UNITY_HOST_OVERRIDE:
        return config.UNITY_HOST_OVERRIDE
    with state.unity_addr_lock:
        if state.last_unity_addr is not None:
            return state.last_unity_addr[0]
    return None


def send_vlm_result_to_unity(payload: dict):
    """Send the VLM result back to the Unity headset.

    Wire format (one UDP datagram, UTF-8):  VLM_RESULT|<json>
    """
    if state.unity_sender_sock is None:
        print("[UNITY-SEND][WARN] sender socket not initialized; skipping.")
        return

    host = _resolve_unity_host()
    if host is None:
        print("[UNITY-SEND][WARN] no Unity host known yet (no inbound UDP seen). Skipping.")
        return

    try:
        body = json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        print(f"[UNITY-SEND][ERROR] json encode failed: {exc}")
        return

    msg = f"{config.VLM_PACKET_PREFIX}|{body}"
    data = msg.encode("utf-8")

    if len(data) > 60000:
        print(
            f"[UNITY-SEND][WARN] packet size {len(data)} bytes exceeds typical UDP "
            "datagram limit (~65 KB). Truncating fields would be safer."
        )

    try:
        state.unity_sender_sock.sendto(data, (host, config.UNITY_RESULT_PORT))
        print(
            f"[UNITY-SEND] -> {host}:{config.UNITY_RESULT_PORT}  "
            f"prefix={config.VLM_PACKET_PREFIX} bytes={len(data)}"
        )
    except Exception as exc:
        print(f"[UNITY-SEND][ERROR] sendto failed: {exc}")
