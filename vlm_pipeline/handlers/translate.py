"""Handler for the 'Translate' gesture.

Pipeline:
  1. Build gaze bbox from the gesture's gaze points.
  2. Run OCR in a padded ROI around the gaze bbox with paragraph=True so words
     near each other are already merged into multi-line blocks.
  3. Pick the OCR block that best contains the gaze: either the block whose
     IoU vs. gaze is highest, OR -- if none overlaps directly -- the block
     whose center is closest to the gaze center (so the user can look near a
     paragraph and still get the full thing).
  4. Translate that whole block to Korean via OpenAI.
"""
from datetime import datetime

import cv2
import numpy as np

from .. import config, geometry, network, ocr, render
from ..vlm_client import translate_texts_to_korean
from . import register


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


@register("Translate")
def handle(captured_frame: np.ndarray, norm_points, gesture_name: str) -> np.ndarray:
    if captured_frame is None:
        print("[Translate] no captured frame at gesture END")
        return render.placeholder_canvas("No frame at gesture END")

    pixel_points = geometry.project_norm_points(norm_points, captured_frame.shape)
    gaze_bbox = geometry.compute_gaze_bbox(pixel_points, captured_frame.shape)

    h, w = captured_frame.shape[:2]
    print(
        f"[Translate] gesture={gesture_name} | frame={w}x{h} | "
        f"gaze_points={len(pixel_points)} | gaze_bbox={gaze_bbox}"
    )

    if gaze_bbox is None:
        overlay = render.render_target_overlay(
            captured_frame, pixel_points, None, None, "NONE", [], gesture_name
        )
        cv2.putText(
            overlay, f"NOT ENOUGH GAZE POINTS ({len(pixel_points)})",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )
        return overlay

    # ---- Stage 1: OCR over a padded ROI around the gaze, paragraph mode on ----
    ocr_blocks = ocr.run_ocr_in_roi(captured_frame, gaze_bbox=gaze_bbox)
    print(f"[Translate][OCR] detected {len(ocr_blocks)} paragraph blocks")

    overlay = render.render_target_overlay(
        captured_frame, pixel_points, None,
        None, "NONE", [], gesture_name, gaze_bbox=gaze_bbox,
    )
    for b in ocr_blocks:
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (120, 120, 120), 1)

    # ---- Stage 2: pick the paragraph block tied to the gaze ----
    idx, pick_mode, pick_score = _pick_block_for_gaze(ocr_blocks, gaze_bbox)
    if idx < 0:
        print("[Translate] no OCR text found in ROI; skipping translation.")
        cv2.putText(
            overlay, "TRANSLATE: no OCR text near gaze",
            (20, overlay.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA,
        )
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

    # ---- Stage 3: translate the whole paragraph to Korean ----
    koreans = translate_texts_to_korean([chosen["text"]])
    ko = koreans[0] if koreans else ""

    print("[Translate] === translation result ===")
    print(f"  EN: {chosen['text']}")
    print(f"  KO: {ko}")
    print("[Translate] ===========================")

    # ---- Stage 4: push the result to Unity ----
    payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture_name,
        "model": config.OPENAI_MODEL,
        "target_meta": {
            "source": "OCR",
            "bbox": list(chosen["bbox"]),
            "pick_mode": pick_mode,
            "pick_score": float(pick_score),
            "gaze_bbox": list(gaze_bbox),
        },
        "response": {
            "name": chosen["text"],
            "translation": ko,
        },
    }
    network.send_vlm_result_to_unity(payload)

    cv2.putText(
        overlay, f"TRANSLATE [{pick_mode}] -> sent to Unity",
        (20, overlay.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA,
    )
    return overlay
