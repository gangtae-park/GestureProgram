"""HTTP server that receives audio bytes from the Unity headset, runs them through
OpenAI Whisper, and feeds the transcribed text into the existing Ask handler.

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
import http.server
import socketserver
import threading

from . import state, vlm_client
from .network import process_ask_question


VOICE_SERVER_PORT = 5007
MAX_AUDIO_BYTES = 10 * 1024 * 1024   # 10 MB hard cap (~10 minutes at 16kHz mono PCM)


class _AskVoiceHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/ask_voice":
            self.send_response(404)
            self.end_headers()
            return

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
