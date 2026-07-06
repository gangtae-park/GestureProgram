"""HTTP server for Unity requests (ports the device pushes JSON to).

Paths handled here:
  1. /ask_voice      Android STT transcript JSON -> cached Ask target.
                     Body { "transcript": "...", "request_id": "..." }.
                     Dispatches process_ask_question(transcript) against the
                     cached Ask target from handlers/ask.py. Result is pushed
                     back to Unity over the UDP VLM_RESULT channel.
  2. /voice_command  Android STT transcript + voice-start camera pose ->
                     VLM (run against the latest ADB-stream frame) -> Unity
                     VLM_RESULT. Unity no longer ships a JPG because its own
                     ScreenCapture on Android XR misses the OS-composited
                     passthrough layer -- we use the ADB screenrecord frame
                     for the same reason /object_ui does.
  3. /object_ui      UI-interaction trigger. Delegates to vlm_pipeline.object_ui
                     which runs YOLO + Depth Anything V2 + inverse calibration
                     against the latest ADB-stream frame and ships per-detection
                     {gaze_dir, depth_meters} back to Unity.

Object-UI logic intentionally lives in a separate module so the depth/calibration
deps don't bleed into the voice path.
"""
import http.server
import json
import socketserver
import threading
import uuid
from datetime import datetime

from . import config, object_ui, state, vlm_client
from .network import process_ask_question, send_vlm_result_to_unity


VOICE_SERVER_PORT = 5007
# /voice_command no longer carries a JPG so the body is pure JSON metadata;
# 256 KB is generous headroom for pose + transcript.
MAX_VOICE_JSON_BYTES = 256 * 1024
# /object_ui payloads are also pure JSON metadata (server pulls the frame from
# the ADB stream), same budget.
MAX_OBJECT_UI_JSON_BYTES = 256 * 1024
MAX_ASK_TRANSCRIPT_JSON_BYTES = 64 * 1024  # transcripts are text; 64 KB is plenty


class _AskVoiceHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/ask_voice":
            self._handle_ask_voice()
            return
        if self.path == "/voice_command":
            self._handle_voice_command()
            return
        if self.path == "/object_ui":
            self._handle_object_ui()
            return

        self.send_response(404)
        self.end_headers()

    def _handle_ask_voice(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0 or length > MAX_ASK_TRANSCRIPT_JSON_BYTES:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"bad length {length}".encode("utf-8"))
            return

        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"[ASK_VOICE][ERROR] bad JSON body: {e}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"bad json")
            return

        request_id = str(payload.get("request_id") or payload.get("requestId") or "")
        transcript = str(payload.get("transcript") or "").strip()
        print(
            f"[ASK_VOICE] received request_id={request_id!r} "
            f"transcript={transcript!r} from={self.client_address[0]}"
        )

        _remember_unity_host(self.client_address[0])

        # Hand off to a worker so we ack the POST immediately.
        threading.Thread(
            target=_process_ask_transcript_async,
            args=(transcript, request_id),
            daemon=True,
        ).start()

        self.send_response(202)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        try:
            self.wfile.write(
                json.dumps({"ok": True, "request_id": request_id}, ensure_ascii=False).encode("utf-8")
            )
        except Exception:
            pass

    def _handle_voice_command(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0 or length > MAX_VOICE_JSON_BYTES:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"bad json length {length}".encode("utf-8"))
            return

        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"[VOICE-COMMAND][ERROR] bad JSON body: {e}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"bad json")
            return

        request_id = str(payload.get("request_id") or payload.get("requestId") or "")
        transcript = str(payload.get("transcript") or "").strip()
        print(
            f"[VOICE-COMMAND] received request_id={request_id!r} "
            f"transcript={transcript!r} from={self.client_address[0]}"
        )

        _remember_unity_host(self.client_address[0])

        threading.Thread(
            target=_process_voice_command_async,
            args=(payload,),
            daemon=True,
        ).start()

        self.send_response(202)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        try:
            self.wfile.write(json.dumps({"ok": True, "request_id": request_id}, ensure_ascii=False).encode("utf-8"))
        except Exception:
            pass

    def _handle_object_ui(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0

        if length <= 0 or length > MAX_OBJECT_UI_JSON_BYTES:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"bad json length {length}".encode("utf-8"))
            return

        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"[OBJECT_UI][ERROR] bad JSON body: {e}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"bad json")
            return

        request_id = str(payload.get("request_id") or payload.get("requestId") or uuid.uuid4().hex)
        print(f"[OBJECT_UI][RX] request_id={request_id} from={self.client_address[0]}")

        _remember_unity_host(self.client_address[0])

        threading.Thread(
            target=object_ui.process_request,
            args=(payload, request_id),
            daemon=True,
        ).start()

        self.send_response(202)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        try:
            self.wfile.write(json.dumps({"ok": True, "request_id": request_id}, ensure_ascii=False).encode("utf-8"))
        except Exception:
            pass

    def do_GET(self):
        # Tiny health check for adb-side debugging
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"voice-server alive")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy default access log
        return


def _remember_unity_host(host: str):
    if not host:
        return
    with state.unity_addr_lock:
        state.last_unity_addr = (host, 0)


def _process_ask_transcript_async(transcript: str, request_id: str):
    """Pair an on-device STT transcript with the cached Ask target."""
    text = (transcript or "").strip()
    if not text:
        print(f"[ASK_VOICE] empty transcript request_id={request_id!r}; aborting.")
        send_vlm_result_to_unity({
            "timestamp": "",
            "gesture": "Ask",
            "model": "android_stt",
            "status": "fail",
            "stage": "answer",
            "request_id": request_id,
            "requestId": request_id,
            "target_meta": {"user_question": ""},
            "response": {"error": "빈 음성 입력이 수신되었습니다. 다시 시도해주세요."},
        })
        return

    print(f"[ASK_VOICE] dispatching to Ask pipeline request_id={request_id!r} text={text!r}")
    process_ask_question(text)


def _describe_gaze_region(nx: float, ny_topleft: float) -> str:
    """Human-readable label for a normalised (top-left origin) point on the
    frame. Feeds the prompt so GPT can double-check its own resolution
    against a coarse spatial descriptor -- avoids "the model quoted a wrong
    pixel" failures we hit when the coordinate alone was ambiguous."""
    if nx < 0.33:
        col = "left"
    elif nx < 0.67:
        col = "center"
    else:
        col = "right"
    if ny_topleft < 0.33:
        row = "top"
    elif ny_topleft < 0.67:
        row = "middle"
    else:
        row = "bottom"
    if col == "center" and row == "middle":
        return "dead center of the frame"
    return f"{row}-{col} region of the frame"


def _build_gaze_info_block(payload: dict, frame_w: int, frame_h: int) -> str:
    """Translate Unity's gaze viewport (0..1, bottom-left origin) into a
    text block for the prompt. Mirrors GazePointAR Figure 8's "gaze data"
    section -- the reason we can disambiguate pronouns instead of guessing
    at scene center."""
    tracked = bool(payload.get("gaze_tracked"))
    if not tracked:
        return (
            "No eye-tracking data was available at listen-start. Fall back "
            "to the center of the frame as the presumed gaze target, but "
            "flag lower confidence if the answer depends on the target."
        )

    try:
        vp_x = float(payload.get("gaze_viewport_x") or 0.0)
        vp_y_bl = float(payload.get("gaze_viewport_y") or 0.0)  # Unity: bottom-left origin
    except (TypeError, ValueError):
        return (
            "Eye-tracking data was malformed. Fall back to the center of "
            "the frame as the presumed gaze target."
        )

    vp_x = max(0.0, min(1.0, vp_x))
    vp_y_bl = max(0.0, min(1.0, vp_y_bl))
    # Convert to image (top-left origin) convention used by cv2/OpenAI.
    ny_tl = 1.0 - vp_y_bl
    px = int(round(vp_x * max(1, frame_w - 1)))
    py = int(round(ny_tl * max(1, frame_h - 1)))
    region = _describe_gaze_region(vp_x, ny_tl)

    return (
        f"The user's eye gaze at listen-start was tracked at pixel "
        f"({px}, {py}) on the {frame_w}x{frame_h} frame -- normalised "
        f"({vp_x:.3f}, {ny_tl:.3f}), which lands in the {region}. Treat "
        f"whatever object occupies that pixel neighbourhood as the PRIMARY "
        f"referent when resolving pronouns; only override if a clearly "
        f"visible hand-pointing gesture aims somewhere else."
    )


def _extract_answer_text(gpt_response) -> str:
    if not isinstance(gpt_response, dict):
        return ""
    for key in ("answer", "response", "text", "raw"):
        value = gpt_response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _send_voice_command_error(request_id: str, transcript: str, message: str):
    payload = {
        "request_id": request_id,
        "requestId": request_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": "VoiceAsk",
        "model": "local",
        "status": "fail",
        "stage": "answer",
        "reason": message,
        "target_meta": {
            "source": "voice_adb_frame",
            "user_question": transcript,
        },
        "response": {
            "name": "Voice request",
            "answer": "",
            "error": message,
        },
    }
    send_vlm_result_to_unity(payload)


def _process_voice_command_async(payload: dict):
    request_id = str(payload.get("request_id") or payload.get("requestId") or "")
    transcript = str(payload.get("transcript") or "").strip()
    if not transcript:
        _send_voice_command_error(request_id, transcript, "Empty transcript.")
        return

    # Grab the latest ADB-stream frame (same source /object_ui uses). Unity's
    # own screen capture only sees its rendered overlays, not the OS-composited
    # passthrough layer, so the LLM was reasoning over a mostly empty scene.
    with state.frame_lock:
        image_bgr = None if state.latest_frame is None else state.latest_frame.copy()

    if image_bgr is None:
        reason = "no ADB stream frame yet (is adb screenrecord running?)"
        print(f"[VOICE-COMMAND][ERROR] request_id={request_id} {reason}")
        _send_voice_command_error(request_id, transcript, reason)
        return

    h, w = image_bgr.shape[:2]
    print(f"[VOICE-COMMAND] request_id={request_id} source=ADB_stream frame={w}x{h}")

    gaze_info_block = _build_gaze_info_block(payload, w, h)
    prompt = (
        config.VOICE_COMMAND_PROMPT
        .replace("{transcript}", transcript)
        .replace("{gaze_info}", gaze_info_block)
    )
    gpt_response = vlm_client.call_vlm_on_crop(image_bgr, prompt)

    referent_text = ""
    confidence_text = ""
    if gpt_response is None:
        answer_text = ""
        ok = False
        error = "vlm_call_failed"
        response_name = "Voice request"
    elif isinstance(gpt_response, dict) and gpt_response.get("error"):
        answer_text = ""
        ok = False
        error = str(gpt_response.get("error"))
        response_name = str(gpt_response.get("name") or "Voice request")
        referent_text = str(gpt_response.get("referent") or "")
        confidence_text = str(gpt_response.get("confidence") or "")
    else:
        answer_text = _extract_answer_text(gpt_response)
        ok = bool(answer_text)
        error = "" if ok else "empty_answer"
        if isinstance(gpt_response, dict):
            response_name = str(gpt_response.get("name") or "Voice request")
            referent_text = str(gpt_response.get("referent") or "")
            confidence_text = str(gpt_response.get("confidence") or "")
        else:
            response_name = "Voice request"

    target_meta = {
        "source": "voice_adb_frame",
        "user_question": transcript,
        "image_width": int(image_bgr.shape[1]),
        "image_height": int(image_bgr.shape[0]),
        "screen_width": int(payload.get("screen_width") or 0),
        "screen_height": int(payload.get("screen_height") or 0),
        # Preserve the gaze cue so vlm_outputs/ retains it alongside the
        # prompt/response -- important for post-hoc analysis of pronoun
        # disambiguation, per GazePointAR's Study 2 evaluation approach.
        "gaze_tracked": bool(payload.get("gaze_tracked")),
        "gaze_viewport_x": float(payload.get("gaze_viewport_x") or 0.0),
        "gaze_viewport_y_unity_bl": float(payload.get("gaze_viewport_y") or 0.0),
    }

    final_response = {
        "name": response_name,
        "answer": answer_text,
        "user_question": transcript,
        # GazePointAR-style extras: surfaces pronoun resolution + model
        # confidence for study logging. Unity's ResultCardSpawner only reads
        # `answer`/`name`; the rest is captured in vlm_outputs for analysis.
        "referent": referent_text,
        "confidence": confidence_text,
    }
    if not ok:
        final_response["error"] = error

    log_response = dict(final_response)
    log_response["_gpt_raw"] = gpt_response
    saved = vlm_client.save_vlm_response(
        log_response, "VoiceAsk", target_meta, image_bgr, prompt
    )

    payload_out = {
        "request_id": request_id,
        "requestId": request_id,
        "timestamp": saved.get("timestamp", "") if saved else "",
        "gesture": "VoiceAsk",
        "model": config.OPENAI_MODEL,
        "status": "ok" if ok else "fail",
        "stage": "answer",
        "target_meta": target_meta,
        "response": final_response,
    }
    if not ok:
        payload_out["reason"] = error

    send_vlm_result_to_unity(payload_out)
    print(
        f"[VOICE-COMMAND] result sent request_id={request_id} "
        f"status={payload_out['status']} answer_len={len(answer_text)}"
    )


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


_server_instance = None


def start_voice_server_thread() -> bool:
    """Spawn the HTTP server on a daemon thread. Safe to call once at startup."""
    global _server_instance
    if _server_instance is not None:
        return True
    try:
        _server_instance = _ThreadingHTTPServer(("0.0.0.0", VOICE_SERVER_PORT), _AskVoiceHandler)
    except OSError as e:
        print(f"[VOICE-SERVER][ERROR] could not bind port {VOICE_SERVER_PORT}: {e}")
        _server_instance = None
        return False

    t = threading.Thread(target=_server_instance.serve_forever, daemon=True)
    t.start()
    print(f"[VOICE-SERVER] listening on 0.0.0.0:{VOICE_SERVER_PORT} (paths: /ask_voice, /voice_command, /object_ui)")
    return True
