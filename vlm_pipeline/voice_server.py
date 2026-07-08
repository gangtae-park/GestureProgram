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

from . import object_ui, state, voice_pipeline
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


def _extract_gaze_trail(payload: dict):
    """Pull the parallel gaze_trail_x / gaze_trail_y arrays Unity ships in
    the /voice_command body and zip them into a list of (nx, ny) tuples,
    still in Unity's bottom-left origin. voice_pipeline flips the y axis to
    the top-left origin the rest of the pipeline uses.

    Returns None (not [] ) when the trail is missing or empty, so callers
    can distinguish "user paused motionless" from "trail unavailable, fall
    back to synth"."""
    tx = payload.get("gaze_trail_x") or []
    ty = payload.get("gaze_trail_y") or []
    if not isinstance(tx, list) or not isinstance(ty, list):
        return None
    n = min(len(tx), len(ty))
    if n <= 0:
        return None
    out = []
    for i in range(n):
        try:
            out.append((float(tx[i]), float(ty[i])))
        except (TypeError, ValueError):
            continue
    return out or None


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

    # Route through the shared voice pipeline:
    # 1. GPT classifies the transcript into one of the 7 canonical referents
    #    (Ask is the fallback bucket for open-ended questions).
    # 2. The classified intent dispatches to the same YOLO+CLIP+DB handler
    #    that the gesture path uses, so Voice-triggered runs surface the
    #    same response cards (Search/Anchor/Save/etc.) as gestures do.
    # 3. Ask specifically falls back to the two-phase handlers/ask.py +
    #    network.process_ask_question chain so voice Ask == gesture Ask.
    gaze_viewport_bl = None
    if bool(payload.get("gaze_tracked")):
        try:
            gaze_viewport_bl = (
                float(payload.get("gaze_viewport_x") or 0.0),
                float(payload.get("gaze_viewport_y") or 0.0),
            )
        except (TypeError, ValueError):
            gaze_viewport_bl = None

    # Gaze trail buffered by Unity from listen-start through transcript-final.
    # When present it wholly replaces the synthesised norm_points cluster --
    # Python then sees the actual gaze span from the utterance, matching how
    # gesture flow accumulates gesture_norm_points.
    trail_bl = _extract_gaze_trail(payload)

    intent, confidence, rationale = voice_pipeline.dispatch(
        transcript=transcript,
        frame_bgr=image_bgr,
        gaze_viewport_bl=gaze_viewport_bl,
        gaze_tracked=bool(payload.get("gaze_tracked")),
        gaze_trail_bl=trail_bl,
        request_id=request_id,
    )
    print(
        f"[VOICE-COMMAND] dispatched request_id={request_id} intent={intent!r} "
        f"confidence={confidence} rationale={rationale!r}"
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
