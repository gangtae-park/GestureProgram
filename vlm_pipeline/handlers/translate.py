"""Handler for the 'Translate' gesture (two-stage).

Stage 1 -- on Translate READY (fires when Jackknife recognises the pose,
before the user has confirmed):
  do_ocr() builds the gaze bbox, runs OCR over a padded ROI, picks the
  paragraph closest to / containing the gaze, and sends a partial VLM_RESULT
  to Unity with the recognised text (no translation yet). The chosen text +
  metadata is cached in state.latest_translate_pending for stage 2.

Stage 2 -- on Translate END (fires when the user confirms with a palm-forward
swipe):
  handle() pulls the cached OCR result and runs GPT translation. Sends the
  final VLM_RESULT with both the original text and the Korean translation.

Splitting like this lets Unity show the OCR'd text immediately so the user
can see WHAT will be translated before committing to the (slower) GPT call.
"""
import time
from datetime import datetime

import cv2
import numpy as np

from .. import config, geometry, network, ocr, render, state
from ..vlm_client import translate_texts_to_korean
from . import register


PENDING_TTL_SEC = 30.0  # cached OCR result expires this many seconds after READY


def _pick_block_for_gaze(blocks, gaze_bbox):
    """Return (index, "overlap" | "nearest", iou_or_distance). Falls back to
    the nearest block when nothing overlaps; returns (-1, None, 0) if blocks
    is empty.
    """
    if not blocks:
        return -1, None, 0.0

    best_overlap_idx, best_overlap = -1, 0.0
    for i, b in enumerate(blocks):
        iou = geometry.bbox_iou(gaze_bbox, b["bbox"])
        if iou > best_overlap:
            best_overlap = iou
            best_overlap_idx = i
    if best_overlap_idx >= 0:
        return best_overlap_idx, "overlap", best_overlap

    # No overlap -- pick whichever block's centre is closest to gaze centre.
    gcx = (gaze_bbox[0] + gaze_bbox[2]) / 2.0
    gcy = (gaze_bbox[1] + gaze_bbox[3]) / 2.0
    best_dist = float("inf")
    best_idx = -1
    for i, b in enumerate(blocks):
        bx1, by1, bx2, by2 = b["bbox"]
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        d = (bcx - gcx) ** 2 + (bcy - gcy) ** 2
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx, "nearest", float(best_dist ** 0.5)


def do_ocr(captured_frame: np.ndarray, norm_points, gesture_name: str) -> np.ndarray:
    """Stage 1 -- called from network.py on Translate READY. OCR only; caches
    the chosen text in state.latest_translate_pending for the END stage."""
    if captured_frame is None:
        print("[Translate][OCR] no captured frame at READY")
        with state.translate_lock: state.latest_translate_pending = None
        return render.placeholder_canvas("No frame at Translate READY")

    pixel_points = geometry.project_norm_points(norm_points, captured_frame.shape)
    gaze_bbox = geometry.compute_gaze_bbox(pixel_points, captured_frame.shape)

    h, w = captured_frame.shape[:2]
    print(
        f"[Translate][OCR] READY | frame={w}x{h} | "
        f"gaze_points={len(pixel_points)} | gaze_bbox={gaze_bbox}"
    )

    if gaze_bbox is None:
        with state.translate_lock: state.latest_translate_pending = None
        overlay = render.render_target_overlay(
            captured_frame, pixel_points, None, None, "NONE", [], gesture_name
        )
        cv2.putText(
            overlay, f"NOT ENOUGH GAZE POINTS ({len(pixel_points)})",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )
        _send_ocr_fail(gesture_name, "Not enough gaze points.")
        return overlay

    ocr_blocks = ocr.run_ocr_in_roi(captured_frame, gaze_bbox=gaze_bbox)
    print(f"[Translate][OCR] detected {len(ocr_blocks)} paragraph blocks")

    overlay = render.render_target_overlay(
        captured_frame, pixel_points, gaze_bbox,
        None, "NONE", [], gesture_name,
    )
    for b in ocr_blocks:
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (120, 120, 120), 1)

    idx, pick_mode, pick_score = _pick_block_for_gaze(ocr_blocks, gaze_bbox)
    if idx < 0:
        with state.translate_lock: state.latest_translate_pending = None
        cv2.putText(
            overlay, "TRANSLATE: no OCR text near gaze",
            (20, overlay.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA,
        )
        _send_ocr_fail(gesture_name, "No OCR text near gaze.")
        return overlay

    chosen = ocr_blocks[idx]
    print(
        f"[Translate][OCR] picked via {pick_mode} ({pick_score:.3f}) | "
        f"bbox={chosen['bbox']} text={chosen['text']!r}"
    )

    x1, y1, x2, y2 = chosen["bbox"]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = chosen["text"]
    if len(label) > 60:
        label = label[:57] + "..."
    cv2.putText(
        overlay, label, (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
    )

    # Cache for stage 2.
    with state.translate_lock:
        state.latest_translate_pending = {
            "text": chosen["text"],
            "bbox": list(chosen["bbox"]),
            "gaze_bbox": list(gaze_bbox),
            "pick_mode": pick_mode,
            "pick_score": float(pick_score),
            "timestamp": time.time(),
        }

    # Send partial result to Unity: source text, no translation yet.
    payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture_name,
        "stage": "ocr",
        "model": "EasyOCR",
        "status": "ok",
        "target_meta": {
            "source": "OCR",
            "bbox": list(chosen["bbox"]),
            "pick_mode": pick_mode,
            "pick_score": float(pick_score),
            "gaze_bbox": list(gaze_bbox),
        },
        "response": {
            "name": chosen["text"],
            "translation": "",
        },
    }
    network.send_vlm_result_to_unity(payload)

    cv2.putText(
        overlay, f"OCR ready [{pick_mode}] -- swipe to translate",
        (20, overlay.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA,
    )
    return overlay


@register("Translate")
def handle(captured_frame: np.ndarray, norm_points, gesture_name: str) -> np.ndarray:
    """Stage 2 -- called on Translate END (after the palm-forward swipe).
    Pulls the cached OCR text and runs GPT translation."""
    overlay = (
        render.placeholder_canvas("Translate END")
        if captured_frame is None else captured_frame.copy()
    )

    with state.translate_lock:
        cached = state.latest_translate_pending
        state.latest_translate_pending = None

    if cached is None:
        reason = "no cached OCR (did READY fire?)"
        print(f"[Translate] END but {reason}")
        network.send_gesture_fail_to_unity(gesture_name, reason, {"stage": "translation"})
        cv2.putText(
            overlay, f"TRANSLATE FAIL: {reason}",
            (20, overlay.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return overlay

    age = time.time() - cached.get("timestamp", 0)
    if age > PENDING_TTL_SEC:
        reason = f"cached OCR is stale ({age:.1f}s old)"
        print(f"[Translate] END but {reason}")
        network.send_gesture_fail_to_unity(gesture_name, reason, {"stage": "translation"})
        return overlay

    text = cached["text"]
    koreans = translate_texts_to_korean([text])
    ko = koreans[0] if koreans else ""

    print("[Translate] === translation result ===")
    print(f"  EN: {text}")
    print(f"  KO: {ko}")
    print("[Translate] ===========================")

    payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture_name,
        "stage": "translation",
        "model": config.OPENAI_MODEL,
        "status": "ok",
        "target_meta": {
            "source": "OCR",
            "bbox": cached["bbox"],
            "pick_mode": cached["pick_mode"],
            "pick_score": cached["pick_score"],
            "gaze_bbox": cached["gaze_bbox"],
        },
        "response": {
            "name": text,
            "translation": ko,
        },
    }
    network.send_vlm_result_to_unity(payload)

    cv2.putText(
        overlay, f"TRANSLATE done -> Unity (EN '{text[:30]}...' -> KO)",
        (20, overlay.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA,
    )
    return overlay


def _send_ocr_fail(gesture_name, reason):
    network.send_gesture_fail_to_unity(gesture_name, reason, {"stage": "ocr"})
