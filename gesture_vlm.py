"""
Gesture-driven target selection pipeline -- entry point.

This file just orchestrates: load models, start threads, run the OpenCV display
loop, and dispatch each completed gesture to its registered handler.

Implementation details live in the `vlm_pipeline/` package:

  vlm_pipeline/
  ├── config.py         constants, prompts, paths
  ├── state.py          shared state + locks
  ├── geometry.py       projection + bbox helpers
  ├── ridge.py          gaze direction -> normalized screen coords
  ├── segmentation.py   YOLO segmentation backend
  ├── clip_matcher.py   CLIP image embedding + cosine match
  ├── object_db.py      fixed 3-object DB (metadata + cached embeddings)
  ├── ocr.py            EasyOCR wrapper (ROI + paragraph mode)
  ├── vlm_client.py     OpenAI client (translation, Ask follow-up, Whisper)
  ├── network.py        ADB stream, UDP receive/send, packet parsing
  ├── render.py         OpenCV drawing helpers
  └── handlers/
      ├── __init__.py            handler registry + dispatch_gesture()
      ├── search_find_info.py    Search/Find Info  (YOLO -> CLIP -> DB lookup)
      ├── ask.py                 Ask  (YOLO crop cache; GPT runs on ASK_QUESTION)
      └── translate.py           Translate (OCR paragraph -> GPT translate)

"""
import socket
import threading
import time

import cv2
import numpy as np

from vlm_pipeline import config, state
from vlm_pipeline.handlers import dispatch_gesture
from vlm_pipeline.network import (
    init_unity_sender_socket,
    stream_reader_loop,
    udp_receiver_loop,
)
from vlm_pipeline.render import placeholder_canvas
from vlm_pipeline.ridge import load_ridge_model
from vlm_pipeline.segmentation import load_yolo_model
from vlm_pipeline.object_db import load_object_db
from vlm_pipeline.clip_matcher import load_clip_model
from vlm_pipeline.vlm_client import init_openai_client
from vlm_pipeline.ocr import init_ocr_reader
from vlm_pipeline.voice_server import start_voice_server_thread


def _consume_pending_gesture():
    with state.gaze_lock:
        if (
            state.pending_gesture_end is not None
            and time.time() >= state.pending_gesture_end["ready_at"]
        ):
            pending = state.pending_gesture_end
            state.pending_gesture_end = None
            return pending
    return None


def _consume_pending_fail():
    with state.gaze_lock:
        if state.last_gesture_fail is not None:
            fail = state.last_gesture_fail
            state.last_gesture_fail = None
            return fail
    return None


def _draw_live_canvas():
    with state.frame_lock:
        live_src = None if state.latest_frame is None else state.latest_frame.copy()

    if live_src is None:
        return placeholder_canvas("Waiting for ADB stream...")

    canvas = live_src
    lh, lw = canvas.shape[:2]
    with state.gaze_lock:
        g = state.latest_gaze_norm
        tracked = state.latest_is_tracked
        active = state.is_gesture_active
        active_pts = list(state.gesture_norm_points) if active else []

    if tracked and g is not None:
        px = int(np.clip(g[0] * lw, 0, lw - 1))
        py = int(np.clip(g[1] * lh, 0, lh - 1))
        cv2.circle(canvas, (px, py), config.POINT_RADIUS, config.POINT_COLOR, -1)

    if active_pts:
        for nx, ny in active_pts:
            cv2.circle(
                canvas,
                (int(nx * lw), int(ny * lh)),
                3,
                config.TRAIL_COLOR,
                -1,
            )
        cv2.putText(
            canvas, "GESTURE ACTIVE",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )
    return canvas


def main():
    # ---- one-time init ----
    load_ridge_model()
    load_yolo_model()
    load_clip_model()
    load_object_db()
    init_openai_client()
    init_ocr_reader(languages=["en"])
    init_unity_sender_socket()
    start_voice_server_thread()

    # ---- UDP receive socket (Unity -> Python) ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((config.HOST, config.PORT))

    # ---- background threads ----
    threading.Thread(target=stream_reader_loop, daemon=True).start()
    threading.Thread(target=udp_receiver_loop, args=(sock,), daemon=True).start()

    # ---- OpenCV windows ----
    cv2.namedWindow(config.LIVE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.LIVE_WINDOW, config.CANVAS_W, config.CANVAS_H)
    cv2.namedWindow(config.TARGET_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.TARGET_WINDOW, config.CANVAS_W, config.CANVAS_H)
    state.target_canvas = placeholder_canvas("Waiting for first gesture END...")

    print(f"Listening on UDP {config.HOST}:{config.PORT}. Press ESC to quit.")

    try:
        while True:
            # 1) Drain pending gesture END -- snapshot frame + run handler.
            # Skip the dispatch when the END just carries the "Pending" placeholder
            # (Unity sends this on a Jackknife reject) so the previous successful
            # result stays on screen for debugging.
            pending = _consume_pending_gesture()
            if pending is not None:
                gname = pending["gesture_name"] or "Unknown"
                if gname in ("Pending", "Unknown"):
                    print(f"[Main] skipping dispatch for placeholder END name={gname!r}; keeping previous canvas")
                else:
                    with state.frame_lock:
                        captured = None if state.latest_frame is None else state.latest_frame.copy()
                    rendered = dispatch_gesture(
                        captured,
                        pending["norm_points"],
                        gname,
                    )
                    with state.target_lock:
                        state.target_canvas = rendered

            # 2) Drain pending gesture FAIL -- log but DO NOT overwrite the canvas.
            # Refreshing on every fail wipes the prior result mid-debugging.
            fail = _consume_pending_fail()
            if fail is not None:
                fail_name = fail.get("gesture_name") or "Unknown"
                print(f"[Main] FAIL drained name={fail_name}; keeping previous canvas")

            # 3) Render both windows
            live_canvas = _draw_live_canvas()
            with state.target_lock:
                tgt = state.target_canvas

            cv2.imshow(
                config.LIVE_WINDOW,
                cv2.resize(live_canvas, (config.CANVAS_W, config.CANVAS_H), interpolation=cv2.INTER_LINEAR),
            )
            cv2.imshow(
                config.TARGET_WINDOW,
                cv2.resize(tgt, (config.CANVAS_W, config.CANVAS_H), interpolation=cv2.INTER_LINEAR),
            )

            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break
    finally:
        state.stop_event.set()
        time.sleep(0.3)
        cv2.destroyAllWindows()
        try:
            sock.close()
        except Exception:
            pass
        try:
            if state.unity_sender_sock is not None:
                state.unity_sender_sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

































































































































































































































































































