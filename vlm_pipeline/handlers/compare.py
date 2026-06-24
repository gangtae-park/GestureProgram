"""Handler for the 'Compare' gesture (two-hand).

Interaction (driven by the Unity GestureRouter):
  1. Look at object A, right-hand pinch  -> gesture START (gaze logging begins).
  2. Look at object B, left-hand pinch   -> Compare READY marker. The Python
     network layer FREEZES gaze logging here, so the trail captured for Compare
     is exactly "A + B" up to this instant; the subsequent "bring both pinched
     hands together" motion adds no gaze points.
  3. Hands meet                          -> gesture END -> this handler runs.

Targeting mirrors the other handlers: build the gaze bbox from the (frozen)
trail, run YOLO, and pick the YOLO segments whose bbox best overlaps the gaze.
Compare just takes the TOP TWO instead of one, CLIP-matches each against the
3-object DB, and ships both objects' info to Unity for a side-by-side card.
"""
import json
import os
import threading
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
)
from . import register


def _persist(crops, target_meta, match_metas, payload):
    """Audit trail under vlm_outputs/: one JSON + one PNG per compared object."""
    os.makedirs(config.VLM_OUTPUT_DIR, exist_ok=True)
    timestamp = payload.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = os.path.join(config.VLM_OUTPUT_DIR, f"{timestamp}_Compare")
    log = {
        "timestamp": timestamp,
        "gesture": "Compare",
        "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
        "target_meta": target_meta,
        "match_meta": match_metas,
        "payload": payload,
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    for i, crop in enumerate(crops):
        try:
            cv2.imwrite(f"{base}_{i + 1}.png", crop)
        except Exception as exc:
            print(f"[COMPARE][WARN] crop #{i + 1} save failed: {exc}")
    print(f"[COMPARE] saved -> {base}.json")


def _fail(gesture_name, reason, target_meta):
    network.send_gesture_fail_to_unity(gesture_name, reason, target_meta)
    print(f"[COMPARE] gesture fail | {reason}")


def _match_worker(crops, target_meta, gesture_name):
    """CLIP-match both crops against the DB; send the pair to Unity, or fail if
    either object can't be recognised."""
    objects = []
    match_metas = []
    for i, crop in enumerate(crops):
        matched_obj, match_meta = clip_matcher.resolve_db_match(crop)
        match_metas.append(match_meta)
        if matched_obj is None:
            reason = clip_matcher.fail_reason_for(match_meta["status"], match_meta)
            fail_payload = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                "gesture": gesture_name,
                "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
                "status": "fail",
                "reason": f"object #{i + 1}: {reason}",
                "target_meta": target_meta,
                "match_meta": match_metas,
                "response": {"error": f"object #{i + 1}: {reason}"},
            }
            _persist(crops, target_meta, match_metas, fail_payload)
            network.send_vlm_result_to_unity(fail_payload)
            print(f"[COMPARE] fail | object #{i + 1} | {match_meta['status']} | {reason}")
            return

        objects.append({
            "name": matched_obj.get("name", ""),
            "object_id": matched_obj.get("id", ""),
            "result_search": matched_obj.get("result_search", ""),
        })
        print(
            f"[COMPARE] object #{i + 1} matched id={matched_obj['id']} "
            f"score={match_meta['score']:.3f} name={matched_obj.get('name')!r}"
        )

    success_payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture_name,
        "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
        "status": "ok",
        "target_meta": target_meta,
        "match_meta": match_metas,
        "response": {"objects": objects},
    }
    _persist(crops, target_meta, match_metas, success_payload)
    network.send_vlm_result_to_unity(success_payload)
    print(f"[COMPARE] sent pair -> {[o['name'] for o in objects]}")


@register("Compare")
def handle(captured_frame: np.ndarray, norm_points, gesture_name: str) -> np.ndarray:
    if captured_frame is None:
        return render.placeholder_canvas("No frame at gesture END")

    pixel_points = geometry.project_norm_points(norm_points, captured_frame.shape)
    gaze_bbox = geometry.compute_gaze_bbox(pixel_points, captured_frame.shape)

    if gaze_bbox is None:
        empty = render.render_compare_overlay(
            captured_frame, pixel_points, None, [], [], gesture_name
        )
        cv2.putText(
            empty, f"NOT ENOUGH GAZE POINTS ({len(pixel_points)})",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )
        _fail(gesture_name, "Not enough gaze points for Compare.", {"gaze_bbox": None})
        return empty

    # ---- YOLO: pick the two segments best overlapping the gaze bbox ----
    yolo_items = segmentation.run_yolo(captured_frame)
    top = (
        geometry.pick_top_overlaps(gaze_bbox, yolo_items, top_n=config.COMPARE_TOP_N)
        if yolo_items else []
    )

    targets = []
    for idx, overlap, iou in top:
        chosen = dict(yolo_items[idx])
        chosen["best_overlap"] = overlap
        chosen["best_iou"] = iou
        targets.append(chosen)
        print(
            f"[COMPARE][YOLO] target #{len(targets)} | class={chosen['class_name']} "
            f"conf={chosen['conf']:.2f} overlap={overlap:.2f} iou={iou:.2f} bbox={chosen['bbox']}"
        )

    overlay = render.render_compare_overlay(
        captured_frame, pixel_points, gaze_bbox, targets, yolo_items, gesture_name
    )

    if len(targets) < 2:
        cv2.putText(
            overlay, f"COMPARE NEEDS TWO OBJECTS (found {len(targets)})",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )
        _fail(
            gesture_name,
            f"Compare needs two overlapping objects; found {len(targets)}.",
            {"gaze_bbox": list(gaze_bbox)},
        )
        return overlay

    # ---- Build CLIP query crops (masked when available) for both targets ----
    crops = []
    target_meta_list = []
    for i, t in enumerate(targets):
        try:
            crops.append(clip_matcher.prepare_query_crop(t, captured_frame))
        except Exception as exc:
            cv2.putText(
                overlay, f"CROP ERROR #{i + 1}: {exc}",
                (20, overlay.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
            )
            _fail(gesture_name, f"object #{i + 1} crop error: {exc}", {"gaze_bbox": list(gaze_bbox)})
            return overlay
        target_meta_list.append({
            "source": "YOLO",
            "bbox": list(t["bbox"]),
            "best_overlap": float(t.get("best_overlap", 0.0)),
            "best_iou": float(t.get("best_iou", 0.0)),
            "class_name": t.get("class_name"),
            "conf": float(t.get("conf", 0.0)) if "conf" in t else None,
            "clip_masked_crop": bool(
                config.CLIP_USE_MASKED_CROP and t.get("mask_bool") is not None
            ),
        })

    target_meta = {"gaze_bbox": list(gaze_bbox), "targets": target_meta_list}

    threading.Thread(
        target=_match_worker,
        args=(crops, target_meta, gesture_name),
        daemon=True,
    ).start()

    return overlay
