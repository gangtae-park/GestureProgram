"""Handler for the 'Ask' gesture (question-mark shape).

Pipeline (two phases, two Unity messages):

  PHASE 1 -- triggered by gesture END:
    a. gaze bbox -> YOLO -> masked crop.
    b. CLIP match against the 3-object DB.
       * Below threshold       -> gesture fail (Unity gets fail VLM_RESULT).
    c. Cache crop + matched object in state.latest_ask_target.
    d. Send VLM_RESULT with stage='object_recognized' + the DB name so Unity
       can immediately prompt the user with "I see <name>, what do you want
       to ask?".

  PHASE 2 -- triggered later, when Unity POSTs the recorded audio:
    a. voice_server -> Whisper transcribe -> network.process_ask_question.
    b. process_ask_question pulls the cached match + crop, calls GPT with the
       DB info as ground truth, and sends VLM_RESULT with stage='answer'
       carrying both the DB name and the GPT answer.
"""
import time
from datetime import datetime

import cv2
import numpy as np

from .. import (
    clip_matcher,
    config,
    geometry,
    network,
    render,
    segmentation,
    state,
)
from . import register


def _fail(overlay, gesture_name, reason, target_meta, match_meta=None):
    """Render fail overlay + tell Unity + clear the Ask cache."""
    with state.ask_lock:
        state.latest_ask_target = None
    cv2.putText(
        overlay, f"ASK FAIL: {reason}",
        (20, overlay.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
    )
    extra = dict(target_meta or {})
    if match_meta is not None:
        extra["match_meta"] = match_meta
    extra["stage"] = "object_recognized"
    network.send_gesture_fail_to_unity(gesture_name, reason, extra)
    print(f"[Ask] gesture fail | {reason}")
    return overlay


@register("Ask")
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
        with state.ask_lock:
            state.latest_ask_target = None
        return empty

    # ---- YOLO segment ----
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
                f"[Ask][YOLO] target | class={chosen['class_name']} conf={chosen['conf']:.2f} "
                f"overlap={overlap:.2f} iou={iou:.2f} bbox={chosen['bbox']}"
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

    # ---- Crop we keep for the GPT call (uses padded bbox like before) ----
    crop_x1, crop_y1, crop_x2, crop_y2 = geometry.expand_bbox_for_crop(
        target["bbox"], captured_frame.shape, config.TARGET_CROP_PAD_RATIO
    )
    crop = captured_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    target_meta = {
        "source": target_source,
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
    }

    if matched_obj is None:
        reason = clip_matcher.fail_reason_for(match_meta["status"], match_meta)
        return _fail(overlay, gesture_name, reason, target_meta, match_meta=match_meta)

    # ---- Phase 1 success: cache + tell Unity which object was recognized ----
    with state.ask_lock:
        state.latest_ask_target = {
            "crop": crop,
            "target_meta": target_meta,
            "match_meta": match_meta,
            "matched_object": matched_obj,
            "gesture_name": gesture_name,
            "timestamp": time.time(),
        }

    recognized_payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture_name,
        "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
        "status": "ok",
        "stage": "object_recognized",
        "target_meta": target_meta,
        "match_meta": match_meta,
        "response": {
            "name": matched_obj.get("name", ""),
            "object_id": matched_obj.get("id", ""),
        },
    }
    network.send_vlm_result_to_unity(recognized_payload)

    cv2.putText(
        overlay,
        f"ASK: {matched_obj.get('name','?')} "
        f"(score={match_meta['score']:.2f}) -- waiting for question.",
        (20, overlay.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA,
    )
    print(
        f"[Ask] phase1 sent name={matched_obj.get('name')!r} "
        f"(id={matched_obj['id']}, score={match_meta['score']:.3f})"
    )
    return overlay
