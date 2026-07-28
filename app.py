
import socket
import threading
import time

import cv2
import numpy as np

from vlm_pipeline import config, state, study
from vlm_pipeline.handlers import dispatch_gesture
from vlm_pipeline.unity.network import init_unity_sender_socket, stream_reader_loop, udp_receiver_loop
from vlm_pipeline.ui.render import placeholder_canvas
from vlm_pipeline.gaze.ridge import load_ridge_model
from vlm_pipeline.vision.segmentation import load_yolo_model
from vlm_pipeline.vision.object_db import load_object_db
from vlm_pipeline.vision.clip_matcher import load_clip_model
from vlm_pipeline.llm.vlm_client import init_openai_client, warm_up_stream_client
from vlm_pipeline.vision.ocr import init_ocr_reader
from vlm_pipeline.voice.voice_server import start_voice_server_thread
from vlm_pipeline.vision import depth as depth_module


def _compose_split_view(live_canvas, target_canvas):
    """Build the combined two-pane image shown in the single MacProgram window.

    Both input canvases are first resized to (CANVAS_W, CANVAS_H) so the two
    panes line up regardless of what each source rendered at. A slim header bar
    with the pane label is drawn on top of each half, and a 1-pixel vertical
    divider separates them.
    """
    w = config.CANVAS_W
    h = config.CANVAS_H
    hdr = config.PANE_HEADER_H

    left = cv2.resize(live_canvas, (w, h), interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(target_canvas, (w, h), interpolation=cv2.INTER_LINEAR)

    # Header bars sit ABOVE the canvases so they never occlude gaze / overlay
    # rendering. Same fixed height for both so the two canvases stay aligned.
    header_bar = np.full((hdr, w, 3), 30, dtype=np.uint8)  # dark grey
    left_header = header_bar.copy()
    right_header = header_bar.copy()
    text_y = int(hdr * 0.68)
    cv2.putText(left_header, config.LIVE_WINDOW, (12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(right_header, config.TARGET_WINDOW, (12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    left_pane = np.vstack([left_header, left])
    right_pane = np.vstack([right_header, right])

    # 1-pixel light-grey divider between the two panes.
    divider = np.full((h + hdr, 1, 3), 90, dtype=np.uint8)
    return np.hstack([left_pane, divider, right_pane])


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
        # With head compensation the trail lives on the RIGHT pane (in the
        # start frame's coordinates); drawing it here would misalign whenever
        # the head moves. Legacy path (no head pose -> old 7-field sender)
        # keeps the old behaviour. gesture_start_head_R is the head-comp
        # marker: set for the whole gesture, even while the delay-aligned
        # start frame is still in flight.
        active_pts = (
            list(state.gesture_norm_points)
            if active and state.gesture_start_head_R is None
            else []
        )

    if tracked and g is not None:
        px = int(np.clip(g[0] * lw, 0, lw - 1))
        py = int(np.clip(g[1] * lh, 0, lh - 1))
        cv2.circle(canvas, (px, py), config.POINT_RADIUS, config.POINT_COLOR, -1)

    for nx, ny in active_pts:
        cv2.circle(
            canvas,
            (int(nx * lw), int(ny * lh)),
            3,
            config.TRAIL_COLOR,
            -1,
        )
    if active:
        cv2.putText(
            canvas, "GESTURE ACTIVE",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )
    return canvas


def _draw_gesture_canvas():
    """Right pane while a gesture is in flight: the frame frozen at gesture
    START with the head-compensated gaze trail accumulating on it. Returns
    None when no gesture is active or no snapshot exists (legacy path)."""
    with state.gaze_lock:
        if not state.is_gesture_active or state.gesture_start_frame is None:
            return None
        canvas = state.gesture_start_frame.copy()
        pts = list(state.gesture_norm_points)
        gname = state.gesture_name_active or ""

    h, w = canvas.shape[:2]
    for nx, ny in pts:
        cv2.circle(
            canvas,
            (int(np.clip(nx * w, 0, w - 1)), int(np.clip(ny * h, 0, h - 1))),
            3,
            config.TRAIL_COLOR,
            -1,
        )
    if pts:
        nx, ny = pts[-1]  # current head-compensated gaze
        cv2.circle(
            canvas,
            (int(np.clip(nx * w, 0, w - 1)), int(np.clip(ny * h, 0, h - 1))),
            config.POINT_RADIUS,
            config.TRAIL_COLOR,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas, f"START FRAME  {gname}",
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

    # Force-load the depth-anything weights here so the FIRST gesture doesn't
    # eat the ~1.5s transformers weights load. Cheap after this call; no-op on
    # subsequent invocations.
    depth_module.preload()

    # Warm the OpenAI streaming connection in the background (TLS handshake +
    # HTTP/2 setup + any server-side caches). Runs in a daemon thread so it
    # doesn't block the UI from appearing; the first real Ask/Translate stream
    # gets a hot connection.
    threading.Thread(
        target=warm_up_stream_client,
        name="vlm-warmup",
        daemon=True,
    ).start()

    # ---- UDP receive socket (Unity -> Python) ----
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((config.HOST, config.PORT))

    # ---- background threads ----
    threading.Thread(target=stream_reader_loop, daemon=True).start()
    threading.Thread(target=udp_receiver_loop, args=(sock,), daemon=True).start()

    # ---- OpenCV window (single, split horizontally into two panes) ----
    cv2.namedWindow(config.COMBINED_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        config.COMBINED_WINDOW,
        config.CANVAS_W * 2,
        config.CANVAS_H + config.PANE_HEADER_H,
    )
    state.target_canvas = placeholder_canvas("Waiting for first gesture END...")

    print(f"Listening on UDP {config.HOST}:{config.PORT}. Press ESC to quit.")

    # Right pane hold: after gesture END the frozen START frame (with its
    # trail) stays visible until the handler result actually replaces it,
    # instead of flashing the previous result during the dispatch delay.
    held_gesture_canvas = None

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
                    held_gesture_canvas = None  # no result coming; release the hold
                else:
                    # Head-comp targeting: prefer the frame frozen at gesture
                    # START (the trail was re-projected into ITS camera pose).
                    # Fall back to a fresh END-time capture when the snapshot
                    # is missing (legacy sender, startup edge).
                    captured = pending.get("start_frame")
                    if captured is None and pending.get("await_start_frame"):
                        # Gesture ended before the delay-aligned frame arrived;
                        # by ready_at (END+0.3s > START+delay) it has landed.
                        with state.gaze_lock:
                            captured = state.gesture_start_frame
                            state.gesture_start_frame = None
                            state.gesture_start_capture_due = None
                    if captured is not None:
                        print("[Main] dispatching on gesture-START frame (head-comp)")
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
                    held_gesture_canvas = None  # result is in; show it

            # 2) Drain pending gesture FAIL -- log but DO NOT overwrite the canvas.
            # Refreshing on every fail wipes the prior result mid-debugging.
            fail = _consume_pending_fail()
            if fail is not None:
                fail_name = fail.get("gesture_name") or "Unknown"
                print(f"[Main] FAIL drained name={fail_name}; keeping previous canvas")
                held_gesture_canvas = None  # gesture died; drop the held frame

            # 3) Render both panes into the single combined window. While a
            # gesture is in flight the right pane shows the frozen START frame
            # with the compensated trail; otherwise the last handler result.
            live_canvas = _draw_live_canvas()
            gesture_canvas = _draw_gesture_canvas()
            if gesture_canvas is not None:
                held_gesture_canvas = gesture_canvas  # remember for the END->dispatch gap
                tgt = gesture_canvas
            elif held_gesture_canvas is not None:
                tgt = held_gesture_canvas             # gesture ended; waiting for the handler
            else:
                with state.target_lock:
                    tgt = state.target_canvas
            cv2.imshow(config.COMBINED_WINDOW, _compose_split_view(live_canvas, tgt))

            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break
    finally:
        state.stop_event.set()
        time.sleep(0.3)
        cv2.destroyAllWindows()
        study.save()
        try:
            sock.close()
        except Exception:
            pass
        try:
            if state.unity_sender_sock is not None:
                state.unity_sender_sock.close()
        except Exception:
            pass


# Study logging is armed remotely: study_control.py (researcher panel) sends
# STUDY,PNUM,<n> / STUDY,START,...,<pnum> packets and the first pnum that
# arrives arms vlm_pipeline.study. Without the panel, app.py runs exactly as
# before and writes no study files.
if __name__ == "__main__":
    main()
