"""Handler for the 'Capture' gesture (camera-frame hand pose, both hands).

Pipeline (identical to Anchor / Save -- "identify the gazed object and ack"):
  1. Build the gaze bbox from the gesture-window gaze trail.
  2. Run YOLO; pick the segment whose bbox best overlaps the gaze.
  3. CLIP-embed the (optionally masked) crop and look it up in the 3-object DB.
  4. If the match clears CLIP_MATCH_MIN_SCORE, send an ack VLM_RESULT carrying
     just the matched object's name + id. Unity opens its Capture UI from
     there (framing rectangle preview, etc.).
  5. Below threshold (or no YOLO match) -> gesture fail.

NOTE: This intentionally targets a single object via gaze, matching Anchor/Save.
The eventual "framing rectangle defined by both hands' L-corners + two-hand
pinch shutter" pipeline can refine the target selection later -- when that
lands, this handler grows a 2D ROI argument and YOLO selection is filtered
by that ROI instead of (or in addition to) the gaze bbox.
"""
from datetime import datetime

import cv2
import numpy as np

from .. import clip_matcher, config, geometry, network, render, segmentation
from . import register


def _fail(overlay, gesture_name, reason, target_meta, match_meta=None):
    cv2.putText(
        overlay, f"CAPTURE FAIL: {reason}",
        (20, overlay.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
    )
    extra = dict(target_meta or {})
    if match_meta is not None:
        extra["match_meta"] = match_meta
    extra["stage"] = "ack"
    network.send_gesture_fail_to_unity(gesture_name, reason, extra)
    print(f"[Capture] gesture fail | {reason}")
    return overlay


@register("Capture")
def handle(captured_frame: np.ndarray, norm_points, gesture_name: str) -> np.ndarray:
    if captured_frame is None:
        return render.placeholder_canvas("No frame at gesture END")

    pixel_points = geometry.project_norm_points(norm_points, captured_frame.shape)
    gaze_bbox = geometry.compute_gaze_bbox(pixel_points, captured_frame.shape)

    if gaze_bbox is None:
        empty = render.render_target_overlay(
            captured_frame, pixel_points, None, None, "NONE", [], gesture_name
        )
        cv2.putText(
            empty, f"NOT ENOUGH GAZE POINTS ({len(pixel_points)})",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )
        _fail(empty, gesture_name, "Not enough gaze points for Capture.", {"gaze_bbox": None})
        return empty

    # ---- YOLO segment overlapping gaze ----
    yolo_items = segmentation.run_yolo(captured_frame)
    target = None
    target_source = "NONE"

    if yolo_items:
        idx, overlap, iou = geometry.pick_best_overlap(gaze_bbox, yolo_items)
        if idx >= 0 and overlap > 0:
            chosen = dict(yolo_items[idx])
            chosen["best_overlap"] = overlap
            chosen["best_iou"] = iou
            target = chosen
            target_source = "YOLO"
            print(
                f"[Capture][YOLO] target | class={chosen['class_name']} "
                f"conf={chosen['conf']:.2f} overlap={overlap:.2f} iou={iou:.2f} "
                f"bbox={chosen['bbox']}"
            )

    overlay = render.render_target_overlay(
        captured_frame, pixel_points, gaze_bbox,
        target, target_source, yolo_items, gesture_name,
    )

    if target is None:
        return _fail(
            overlay, gesture_name, "No YOLO segment overlaps gaze.",
            target_meta={"gaze_bbox": list(gaze_bbox)},
        )

    # ---- CLIP query crop (masked when possible) ----
    try:
        clip_crop = clip_matcher.prepare_query_crop(target, captured_frame)
    except Exception as exc:
        return _fail(
            overlay, gesture_name, f"CROP ERROR: {exc}",
            target_meta={"bbox": list(target["bbox"]), "gaze_bbox": list(gaze_bbox)},
        )

    matched_obj, match_meta = clip_matcher.resolve_db_match(clip_crop)

    target_meta = {
        "source": target_source,
        "bbox": list(target["bbox"]),
        "frame_size": [int(captured_frame.shape[1]), int(captured_frame.shape[0])],  # [width, height]
        "best_overlap": float(target.get("best_overlap", 0.0)),
        "best_iou": float(target.get("best_iou", 0.0)),
        "class_name": target.get("class_name"),
        "conf": float(target.get("conf", 0.0)) if "conf" in target else None,
        "gaze_bbox": list(gaze_bbox),
        "clip_masked_crop": bool(
            config.CLIP_USE_MASKED_CROP and target.get("mask_bool") is not None
        ),
    }

    if matched_obj is None:
        reason = clip_matcher.fail_reason_for(match_meta["status"], match_meta)
        return _fail(overlay, gesture_name, reason, target_meta, match_meta=match_meta)

    # ---- Success: ack Unity with just the matched object name + id ----
    network.send_vlm_result_to_unity({
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture_name,
        "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
        "status": "ok",
        "stage": "ack",
        "target_meta": target_meta,
        "match_meta": match_meta,
        "response": {
            "name": matched_obj.get("name", ""),
            "object_id": matched_obj.get("id", ""),
            "message": "Capture object recognised.",
        },
    })

    cv2.putText(
        overlay,
        f"CAPTURE: {matched_obj.get('name','?')} (score={match_meta['score']:.2f})",
        (20, overlay.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TARGET_COLOR, 2, cv2.LINE_AA,
    )
    print(
        f"[Capture] matched id={matched_obj['id']} "
        f"name={matched_obj.get('name')!r} score={match_meta['score']:.3f}"
    )
    return overlay
