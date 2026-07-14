"""Voice-command routing.

Bridges the /voice_command HTTP endpoint to the existing gesture handler
registry. The flow:

  1. voice_intent.classify(transcript) picks one of the seven canonical
     referents. Ask is the fallback bucket for anything the six DB-backed
     intents don't cleanly claim.
  2. We synthesise a small cluster of gaze "points" around the user's
     tracked gaze pixel so the handlers' existing YOLO-overlap / CLIP-DB
     pipeline can run unchanged. Compare gets a wider cluster because it
     needs two YOLO segments in the same gaze bbox.
  3. We push the voice request context (request_id, gaze pixel) onto a
     thread-local so network.send_vlm_result_to_unity can stamp every
     outbound VLM_RESULT with the request_id -- this is what lets Unity's
     CaptureContextRegistry snap the response card to the pose registered
     at listen-start.
  4. For Ask specifically, we route through the existing
     handlers.ask + network.process_ask_question two-phase pipeline so
     Ask via voice behaves identically to Ask via gesture.

Rationale for reusing the gesture handlers instead of re-implementing per
intent: single source of truth for CLIP threshold behaviour, target_anchor
depth logic, and Unity payload schema. Any tweak to Search's payload
automatically applies to voice-Search as well.
"""
import math
import threading
import time
from typing import Optional, Tuple

import numpy as np

from . import config, handlers, state


# =========================================================================
# Active voice context (consumed by network._stamp_voice_request_id)
# =========================================================================
# We can't use threading.local() here because several gesture handlers spawn
# a daemon Thread for the CLIP worker (e.g. search_find_info._match_worker)
# and those child threads don't inherit the parent's thread-local storage.
# Instead we use a lock-guarded module singleton with a short TTL: the
# dispatch call publishes the context, and any send_vlm_result_to_unity that
# runs within TTL_SECONDS picks it up. For a single-user study the race
# window (two overlapping voice runs) is effectively impossible, so this
# stays simple.
_CONTEXT_TTL_SECONDS = 15.0
_context_lock = threading.Lock()
_current_context: Optional[dict] = None
_current_context_expiry: float = 0.0


def current_context() -> Optional[dict]:
    global _current_context, _current_context_expiry
    with _context_lock:
        if _current_context is None:
            return None
        if time.time() > _current_context_expiry:
            _current_context = None
            return None
        return dict(_current_context)


def _publish_context(request_id: str, gaze_pixel: Tuple[int, int], gaze_tracked: bool,
                     extras: Optional[dict] = None):
    global _current_context, _current_context_expiry
    payload = {
        "request_id": request_id,
        "gaze_pixel": gaze_pixel,
        "gaze_tracked": gaze_tracked,
    }
    if extras:
        # extras carries per-intent auxiliary data (e.g. Save's parsed
        # note_content) so downstream stamps can inject it into the payload
        # without changing the handler signatures.
        payload.update(extras)
    with _context_lock:
        _current_context = payload
        _current_context_expiry = time.time() + _CONTEXT_TTL_SECONDS


# =========================================================================
# Public entry point
# =========================================================================

def dispatch(
    transcript: str,
    frame_bgr: np.ndarray,
    gaze_viewport_bl: Tuple[float, float],
    gaze_tracked: bool,
    request_id: str,
    gaze_trail_bl: Optional[list] = None,
) -> Tuple[str, str, str]:
    """Route a /voice_command request. Returns (intent, confidence, rationale)
    for the caller to log. Sending the VLM_RESULT to Unity is handled inside;
    on any hard failure we still emit a fail payload so the Unity card doesn't
    hang.

    - transcript:         final STT text from Android SpeechRecognizer.
    - frame_bgr:          ADB screenrecord frame captured at request-arrival.
    - gaze_viewport_bl:   Single (nx, ny) snapshot at listen-start, Unity's
                          bottom-left origin convention (0..1). Used as the
                          fallback centre when gaze_trail_bl is empty.
    - gaze_tracked:       True iff the eye tracker had a valid sample at
                          listen-start.
    - request_id:         the id Unity registered its CapturePose with.
    - gaze_trail_bl:      Optional list of (nx, ny) samples buffered by
                          Unity while STT was listening. When present, these
                          become the norm_points fed to the YOLO overlap
                          check, matching how the gesture pipeline uses a
                          gaze trajectory rather than a snapshot.
    """
    if frame_bgr is None:
        _emit_voice_fail(request_id, transcript, "Ask", "no ADB frame available")
        return "Ask", "low", "no_frame"

    # Lazy import to avoid a circular voice_intent<->config load if the module
    # graph ever grows.
    from . import voice_intent

    intent, confidence, rationale, extras = voice_intent.classify(transcript)

    h, w = frame_bgr.shape[:2]
    gaze_pixel = _resolve_gaze_pixel(gaze_viewport_bl, gaze_tracked, w, h)

    norm_points = _build_norm_points(
        gaze_trail_bl=gaze_trail_bl,
        fallback_gaze_pixel=gaze_pixel,
        frame_w=w,
        frame_h=h,
        wider=(intent == "Compare"),
    )

    _publish_context(request_id, gaze_pixel, gaze_tracked, extras=extras)
    if intent == "Ask":
        _run_voice_ask(transcript, frame_bgr, norm_points, request_id)
    else:
        _run_gesture_handler_for_voice(intent, frame_bgr, norm_points)

    return intent, confidence, rationale


# =========================================================================
# Ask fallback -- reuse handlers/ask.py phase-1 + network.process_ask_question
# =========================================================================

def _run_voice_ask(transcript: str, frame_bgr, norm_points, request_id: str):
    """Voice Ask: run the same YOLO+CLIP target-selection as gesture Ask
    phase 1, populate state.latest_ask_target, then IMMEDIATELY call
    network.process_ask_question so only the final AskResultCard renders.

    We inline the target selection (rather than calling handlers.ask.handle)
    because gesture Ask phase 1 broadcasts an intermediate "object_recognized"
    payload that Unity turns into an AskQuestionCard prompting "I see X --
    what do you want to ask?". For voice we already have the transcript, so
    that intermediate card would flash for the ~2s of GPT latency and then
    swap to the answer -- confusing UX. Skipping the broadcast keeps the
    voice flow single-card, matching how gesture / UI methods surface a
    single result card.
    """
    # Local imports to keep the module-graph clean.
    from . import clip_matcher, geometry, network as network_mod, segmentation, target_anchor

    if frame_bgr is None:
        _emit_voice_fail(request_id, transcript, "Ask", "no_frame")
        return

    pixel_points = geometry.project_norm_points(norm_points, frame_bgr.shape)
    gaze_bbox = geometry.compute_gaze_bbox(pixel_points, frame_bgr.shape)
    if gaze_bbox is None:
        _emit_voice_fail(request_id, transcript, "Ask", "not_enough_gaze_points")
        return

    yolo_items = segmentation.run_yolo(frame_bgr)
    target = None
    if yolo_items:
        idx, overlap, iou = geometry.pick_best_overlap(gaze_bbox, yolo_items)
        if idx >= 0 and overlap > 0:
            target = dict(yolo_items[idx])
            target["best_overlap"] = overlap
            target["best_iou"] = iou

    if target is None:
        # No YOLO segment hit the gaze. Fall back to a whole-frame Ask so
        # the user still gets an answer instead of a hard fail (e.g. "how's
        # the weather?" or general-knowledge questions).
        _cache_open_frame_ask_target(frame_bgr, transcript, gaze_bbox)
        network_mod.process_ask_question(transcript)
        return

    try:
        clip_crop = clip_matcher.prepare_query_crop(target, frame_bgr)
    except Exception as exc:
        _emit_voice_fail(request_id, transcript, "Ask", f"crop_error: {exc}")
        return

    matched_obj, match_meta = clip_matcher.resolve_db_match(clip_crop)

    crop_x1, crop_y1, crop_x2, crop_y2 = geometry.expand_bbox_for_crop(
        target["bbox"], frame_bgr.shape, config.TARGET_CROP_PAD_RATIO
    )
    crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    target_meta = {
        "source": "YOLO",
        "bbox": list(target["bbox"]),
        "best_overlap": float(target.get("best_overlap", 0.0)),
        "best_iou": float(target.get("best_iou", 0.0)),
        "class_name": target.get("class_name"),
        "conf": float(target.get("conf", 0.0)) if "conf" in target else None,
        "crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
        "gaze_bbox": list(gaze_bbox),
        "clip_masked_crop": bool(
            config.CLIP_USE_MASKED_CROP and target.get("mask_bool") is not None
        ),
        "user_question": transcript,
        "voice_triggered": True,
    }

    anchor = target_anchor.compute(
        frame_bgr, target.get("bbox"), target.get("mask_bool")
    )

    with state.ask_lock:
        state.latest_ask_target = {
            "crop": crop,
            "target_meta": target_meta,
            "match_meta": match_meta,
            "matched_object": matched_obj,   # may be None if CLIP failed
            "gesture_name": "Ask",
            "timestamp": time.time(),
            "anchor": anchor,
        }

    # Phase 2: GPT with the crop + user question. process_ask_question reads
    # state.latest_ask_target we just wrote and re-enters
    # send_vlm_result_to_unity, which the request_id stamper picks up.
    network_mod.process_ask_question(transcript)


def _cache_open_frame_ask_target(frame_bgr, transcript: str, gaze_bbox):
    """Fallback when YOLO finds nothing at the gaze bbox: cache the WHOLE
    frame as the Ask target so process_ask_question can still answer
    general-knowledge questions. matched_object stays None, so the GPT call
    won't be biased by a DB entry."""
    target_meta = {
        "source": "voice_open_frame",
        "gaze_bbox": list(gaze_bbox) if gaze_bbox is not None else None,
        "user_question": transcript,
        "voice_triggered": True,
    }
    with state.ask_lock:
        state.latest_ask_target = {
            "crop": frame_bgr.copy(),
            "target_meta": target_meta,
            "match_meta": {},
            "matched_object": None,
            "gesture_name": "Ask",
            "timestamp": time.time(),
            "anchor": {},
        }


# =========================================================================
# The six DB-backed intents -- delegate to the exact gesture handler
# =========================================================================

# Voice classifier returns the canonical Unity name; some Python handlers are
# registered under a slightly longer key. Bridge them here.
_INTENT_TO_HANDLER = {
    "Search":           "Search",
    "Translate":        "Translate",
    "Compare":          "Compare",
    "Anchor":           "Anchor",
    "Save":             "Save",
    "Capture":          "Capture",
}


def _run_gesture_handler_for_voice(intent: str, frame_bgr, norm_points):
    handler_name = _INTENT_TO_HANDLER.get(intent)
    if handler_name is None:
        print(f"[VOICE_PIPELINE][WARN] no handler mapping for intent={intent!r}; skipping.")
        return

    handler = handlers.get_handler(handler_name)
    if handler is None:
        print(f"[VOICE_PIPELINE][WARN] handler {handler_name!r} not registered; skipping.")
        return

    try:
        overlay = handler(frame_bgr.copy(), norm_points, handler_name)
    except Exception as exc:
        print(f"[VOICE_PIPELINE][ERROR] handler {handler_name!r} raised: {exc}")
        return

    _publish_target_canvas(overlay)


# =========================================================================
# Helpers
# =========================================================================

def _resolve_gaze_pixel(gaze_viewport_bl, gaze_tracked, frame_w, frame_h):
    """Convert Unity's bottom-left viewport (0..1) to top-left pixel coords
    used by cv2 / handlers. Falls back to frame centre when the eye tracker
    didn't report a valid sample."""
    if not gaze_tracked or gaze_viewport_bl is None:
        return int(frame_w // 2), int(frame_h // 2)
    try:
        nx = float(gaze_viewport_bl[0])
        ny_bl = float(gaze_viewport_bl[1])
    except (TypeError, ValueError):
        return int(frame_w // 2), int(frame_h // 2)
    nx = max(0.0, min(1.0, nx))
    ny_bl = max(0.0, min(1.0, ny_bl))
    ny_tl = 1.0 - ny_bl
    px = int(round(nx * max(1, frame_w - 1)))
    py = int(round(ny_tl * max(1, frame_h - 1)))
    return px, py


def _build_norm_points(gaze_trail_bl, fallback_gaze_pixel, frame_w, frame_h, wider):
    """Pick the best available gaze representation for the YOLO overlap:

      1. If Unity shipped a live gaze trail (samples buffered while STT was
         listening), flip its y-axis to the top-left origin the pipeline
         expects and return it. This mirrors the gesture pipeline, which
         accumulates gaze DURING the interaction rather than at a single
         instant.
      2. Otherwise fall back to a synthesised ring around the listen-start
         snapshot (or frame centre if the eye tracker didn't report).

    `wider` widens the fallback ring for Compare, which needs two YOLO
    segments in the same gaze bbox.
    """
    if gaze_trail_bl:
        top_left_pts = []
        for nx, ny_bl in gaze_trail_bl:
            nx = max(0.0, min(1.0, float(nx)))
            ny_bl = max(0.0, min(1.0, float(ny_bl)))
            top_left_pts.append((nx, 1.0 - ny_bl))
        if len(top_left_pts) >= config.MIN_GAZE_POINTS_FOR_TARGET:
            return top_left_pts
        # Trail too short (e.g. STT finished before enough samples came in).
        # Fall through to synthesis so compute_gaze_bbox still succeeds.

    radius_frac = 0.20 if wider else 0.06
    return _synth_norm_points(fallback_gaze_pixel, frame_w, frame_h, radius_frac)


def _synth_norm_points(gaze_pixel, frame_w, frame_h, radius_frac):
    """Fallback when no live gaze trail is available. Builds a small ring
    around the given pixel so handlers see enough points to compute a
    gaze bbox and overlap at least one YOLO segment."""
    n = max(6, config.MIN_GAZE_POINTS_FOR_TARGET + 1)
    short_side = max(1, min(frame_w, frame_h))
    radius_px = max(4, int(round(short_side * radius_frac)))
    cx, cy = gaze_pixel
    pts = [(cx / max(1, frame_w), cy / max(1, frame_h))]  # centre point too
    for i in range(n - 1):
        theta = (2.0 * math.pi * i) / (n - 1)
        px = cx + int(round(radius_px * math.cos(theta)))
        py = cy + int(round(radius_px * math.sin(theta)))
        px = max(0, min(frame_w - 1, px))
        py = max(0, min(frame_h - 1, py))
        pts.append((px / max(1, frame_w), py / max(1, frame_h)))
    return pts


def _publish_target_canvas(canvas):
    """Handlers return a rendered overlay for the debug 'Target Result'
    window. Push it to state so the main render loop picks it up, same as
    gesture flow does."""
    if canvas is None:
        return
    try:
        with state.target_lock:
            state.target_canvas = canvas
    except Exception:
        pass


def _emit_voice_fail(request_id: str, transcript: str, intent: str, reason: str):
    """Push a synthetic fail VLM_RESULT so the Unity voice card resolves to
    an error state instead of hanging waiting for a response."""
    # Local import to avoid load-order cycles.
    from . import network as network_mod
    from datetime import datetime

    payload = {
        "request_id": request_id,
        "requestId": request_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": intent if intent != "Ask" else "VoiceAsk",
        "model": "voice_pipeline",
        "status": "fail",
        "stage": "answer",
        "reason": reason,
        "target_meta": {
            "source": "voice_pipeline",
            "user_question": transcript,
        },
        "response": {
            "name": "Voice request",
            "answer": "",
            "error": reason,
        },
    }
    network_mod.send_vlm_result_to_unity(payload)
