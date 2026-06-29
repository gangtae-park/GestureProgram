"""HTTP server for Unity voice requests.

It supports two paths:
  1. /ask_voice      legacy raw WAV upload -> Whisper -> cached Ask target.
  2. /voice_command  Android STT transcript + voice-start screenshot in one JSON
                     request -> VLM -> Unity VLM_RESULT.

Wire protocol (intentionally trivial):
  Unity POSTs raw WAV bytes to  http://<python_host>:5007/ask_voice
  Body Content-Type: audio/wav   (or anything; we only look at the bytes)
  Server returns 200 OK after accepting -- the actual VLM result is pushed back
  to Unity through the existing UDP VLM_RESULT channel (network.send_vlm_result_to_unity).

The cached Ask target (set by handlers/ask.py at gesture END) is what the
transcribed question gets paired with. If no cached target exists or it has
expired, the server still echoes an error payload back to Unity so the card
doesn't hang.
"""
import base64
import http.server
import json
import socketserver
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np

from . import config, segmentation, state, vlm_client
from .network import process_ask_question, send_vlm_result_to_unity


VOICE_SERVER_PORT = 5007
MAX_AUDIO_BYTES = 10 * 1024 * 1024   # 10 MB hard cap (~10 minutes at 16kHz mono PCM)
MAX_VOICE_JSON_BYTES = 12 * 1024 * 1024


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

        if length <= 0 or length > MAX_AUDIO_BYTES:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(f"bad length {length}".encode("utf-8"))
            return

        try:
            audio_bytes = self.rfile.read(length)
        except Exception as e:
            print(f"[VOICE-SERVER][ERROR] reading body: {e}")
            self.send_response(500)
            self.end_headers()
            return

        print(f"[VOICE-SERVER] received {len(audio_bytes)} bytes from {self.client_address[0]}")

        # Hand off to a worker so we ack the POST immediately.
        threading.Thread(
            target=_process_audio_async,
            args=(audio_bytes,),
            daemon=True,
        ).start()

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        try:
            self.wfile.write(b"OK")
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
        image_b64_len = len(str(payload.get("image_base64") or ""))
        print(
            f"[VOICE-COMMAND] received request_id={request_id!r} "
            f"transcript={transcript!r} image_b64_len={image_b64_len} "
            f"from={self.client_address[0]}"
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

        if length <= 0 or length > MAX_VOICE_JSON_BYTES:
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
        image_b64_len = len(str(payload.get("image_base64") or ""))
        print(f"[OBJECT_UI][RX] request_id={request_id} image_b64_len={image_b64_len} from={self.client_address[0]}")

        _remember_unity_host(self.client_address[0])

        threading.Thread(
            target=_process_object_ui_async,
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


def _process_audio_async(audio_bytes: bytes):
    text = vlm_client.transcribe_audio_bytes(audio_bytes, file_format="wav")
    if not text or not text.strip():
        print("[VOICE-SERVER] empty transcript; aborting.")
        from .network import send_vlm_result_to_unity
        send_vlm_result_to_unity({
            "timestamp": "",
            "gesture": "Ask",
            "model": "whisper-1",
            "status": "fail",
            "stage": "answer",
            "target_meta": {"user_question": ""},
            "response": {"error": "Couldn't understand audio. Please try again."},
        })
        return

    print(f"[VOICE-SERVER] dispatching to Ask pipeline: {text!r}")
    process_ask_question(text.strip())


def _decode_image_base64(image_base64: str):
    if not image_base64:
        return None, "missing image_base64"
    if "," in image_base64 and image_base64.strip().lower().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        return None, f"base64 decode failed: {exc}"
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None, "cv2.imdecode failed"
    return image, ""


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
            "source": "voice_snapshot",
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

    image_bgr, image_error = _decode_image_base64(str(payload.get("image_base64") or ""))
    if image_bgr is None:
        print(f"[VOICE-COMMAND][ERROR] request_id={request_id} image decode failed: {image_error}")
        _send_voice_command_error(request_id, transcript, f"Voice snapshot image unavailable: {image_error}")
        return

    prompt = config.VOICE_COMMAND_PROMPT.replace("{transcript}", transcript)
    gpt_response = vlm_client.call_vlm_on_crop(image_bgr, prompt)

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
    else:
        answer_text = _extract_answer_text(gpt_response)
        ok = bool(answer_text)
        error = "" if ok else "empty_answer"
        response_name = str(gpt_response.get("name") or "Voice request") if isinstance(gpt_response, dict) else "Voice request"

    target_meta = {
        "source": "voice_snapshot",
        "user_question": transcript,
        "image_width": int(payload.get("image_width") or image_bgr.shape[1]),
        "image_height": int(payload.get("image_height") or image_bgr.shape[0]),
        "screen_width": int(payload.get("screen_width") or 0),
        "screen_height": int(payload.get("screen_height") or 0),
    }

    final_response = {
        "name": response_name,
        "answer": answer_text,
        "user_question": transcript,
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


def _process_object_ui_async(payload: dict, request_id: str):
    # Galaxy XR's Unity ScreenCapture can only see Unity-rendered content;
    # passthrough is composited by the OS and never reaches Unity's frame
    # buffer. So the image Unity used to POST was effectively blank and YOLO
    # detected nothing. Instead we run YOLO against the latest ADB-streamed
    # frame (state.latest_frame) -- that's the same source the gesture
    # handlers (Search/Anchor/Save/etc.) already use successfully, because
    # the ADB screenrecord stream captures the OS-composited display.
    with state.frame_lock:
        image_bgr = None if state.latest_frame is None else state.latest_frame.copy()

    if image_bgr is None:
        reason = "no ADB stream frame yet (is adb screenrecord running?)"
        print(f"[OBJECT_UI][ERROR] {reason} request_id={request_id}")
        _send_object_ui_result(request_id, [], payload, status="fail", reason=reason)
        return

    h, w = image_bgr.shape[:2]
    print(f"[OBJECT_UI][RX] request_id={request_id} source=ADB_stream frame={w}x{h}")

    # DEBUG: dump the frame YOLO sees so we can verify content + tune later.
    try:
        import os, cv2 as _cv2
        os.makedirs("/tmp/object_ui_debug", exist_ok=True)
        _path = f"/tmp/object_ui_debug/{request_id}.jpg"
        _cv2.imwrite(_path, image_bgr)
        print(f"[OBJECT_UI][DEBUG] saved frame -> {_path}")
    except Exception as _e:
        print(f"[OBJECT_UI][DEBUG] save failed: {_e}")

    t0 = time.perf_counter()
    detections_raw = segmentation.run_yolo(image_bgr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[OBJECT_UI][YOLO] detections={len(detections_raw)} request_id={request_id} elapsed_ms={elapsed_ms:.0f}")

    # Override the dimension fields in the source payload so the response
    # advertises the ADB-stream resolution (which is what bboxes are in),
    # not whatever Unity put there.
    payload_for_response = dict(payload)
    payload_for_response["image_width"] = w
    payload_for_response["image_height"] = h

    detections = []
    for i, item in enumerate(detections_raw):
        bbox = [float(v) for v in item.get("bbox", (0, 0, 0, 0))]
        class_name = str(item.get("class_name") or item.get("class_id") or "detected_object")
        confidence = float(item.get("conf") or 0.0)
        det = {
            "request_id": request_id,
            "requestId": request_id,
            "label": class_name,
            "class_name": class_name,
            "confidence": confidence,
            "conf": confidence,
            "bbox": bbox,
            "image_width": w,
            "image_height": h,
            "imageWidth": w,
            "imageHeight": h,
        }
        detections.append(det)
        print(f"[OBJECT_UI][YOLO] det[{i}] class={class_name} conf={confidence:.3f} bbox={bbox}")

    _send_object_ui_result(request_id, detections, payload_for_response)


def _send_object_ui_result(request_id: str, detections: list, source_payload: dict, status: str = "ok", reason: str = ""):
    payload = {
        "request_id": request_id,
        "requestId": request_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": "ObjectUI",
        "model": f"YOLO({config.SEG_MODEL_PATH})",
        "status": status,
        "stage": "yolo",
        "reason": reason,
        "detections": detections,
        "image_width": int(source_payload.get("image_width") or 0),
        "image_height": int(source_payload.get("image_height") or 0),
        "imageWidth": int(source_payload.get("image_width") or 0),
        "imageHeight": int(source_payload.get("image_height") or 0),
        "target_meta": {
            "source": "unity_object_ui_capture",
            "mode": source_payload.get("mode", "object_ui"),
            "gaze_tracked": bool(source_payload.get("gaze_tracked", False)),
            "gaze_viewport": [
                float(source_payload.get("gaze_viewport_x") or 0.5),
                float(source_payload.get("gaze_viewport_y") or 0.5),
            ],
        },
        "response": {
            "name": "ObjectUI",
            "raw": f"detections={len(detections)}",
            "error": reason if status != "ok" else "",
        },
    }
    send_vlm_result_to_unity(payload)


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
    print(f"[VOICE-SERVER] listening on 0.0.0.0:{VOICE_SERVER_PORT}/ask_voice")
    return True
