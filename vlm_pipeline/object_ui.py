"""HTTP-triggered Object UI pipeline.

Flow (called by voice_server when a /object_ui POST lands):

  1. Grab the latest ADB-stream frame from state.latest_frame.
  2. Run YOLO segmentation on it to get bboxes + per-detection masks.
  3. For each YOLO segment, build a CLIP query crop and look it up in the
     fixed object_db. Drop any segment that doesn't match a registered object
     (study uses a closed set of 3 objects). Dedupe by matched object_id so
     each real-world object becomes at most one bubble (keep the best score).
  4. Run Depth Anything V2 once on the frame to get a metric depth map.
  5. For each surviving detection:
       - bbox center -> normalised ADB coords (norm_x, norm_y) in [0,1]
       - apply the inverse gaze calibration to get the head-space gaze direction
         that points at that bbox (gaze_dir_x/y/z, unit vector)
       - take median depth over the segmentation mask (fallback: bbox area)
         to get distance in metres
  6. Pack everything into a VLM_RESULT payload and send it to Unity over UDP.

Unity then anchors each bubble at  capture_camera_pos + R_capture * gaze_dir * depth.
No on-device raycast or scene mesh required.

This module used to live inside voice_server.py — it's split out for clarity
and so that the depth/calibration deps don't bleed into the voice path.
"""

import time
from datetime import datetime
from typing import Optional

import numpy as np

from . import clip_matcher, config, depth, gaze_calibration, segmentation, state
from .network import send_vlm_result_to_unity


def process_request(payload: dict, request_id: str):
    """Worker entry point. Always responds to Unity over UDP (success or fail)
    so the caller's request_id never hangs."""
    # Best-effort eager loads -- they're idempotent and avoid race conditions.
    gaze_calibration.load()
    # depth.estimate() will lazy-load itself on first call.

    with state.frame_lock:
        image_bgr = None if state.latest_frame is None else state.latest_frame.copy()

    if image_bgr is None:
        reason = "no ADB stream frame yet (is adb screenrecord running?)"
        print(f"[OBJECT_UI][ERROR] {reason} request_id={request_id}")
        _send_result(request_id, [], payload, status="fail", reason=reason)
        return

    h, w = image_bgr.shape[:2]
    print(f"[OBJECT_UI][RX] request_id={request_id} source=ADB_stream frame={w}x{h}")

    _debug_dump(image_bgr, request_id)

    # ---- 1) YOLO ----
    t_yolo = time.perf_counter()
    detections_raw = segmentation.run_yolo(image_bgr)
    yolo_ms = (time.perf_counter() - t_yolo) * 1000.0
    print(f"[OBJECT_UI][YOLO] detections={len(detections_raw)} request_id={request_id} elapsed_ms={yolo_ms:.0f}")

    # ---- 2) CLIP filter against the fixed object_db ----
    # Only YOLO segments that map to a registered object survive. Each surviving
    # logical object becomes at most one bubble (best CLIP score wins ties).
    t_clip = time.perf_counter()
    filtered_items = _clip_filter_detections(detections_raw, image_bgr)
    clip_ms = (time.perf_counter() - t_clip) * 1000.0
    print(
        f"[OBJECT_UI][CLIP] kept={len(filtered_items)} of {len(detections_raw)} "
        f"elapsed_ms={clip_ms:.0f}"
    )

    # If nothing survived, ship an empty (but ok) response so Unity can clear
    # the listening UI cleanly instead of waiting for a timeout.
    if not filtered_items:
        _send_result(request_id, [], _payload_for_response(payload, w, h))
        return

    # ---- 3) Depth (single forward pass for the whole frame) ----
    t_depth = time.perf_counter()
    depth_map = depth.estimate(image_bgr)
    depth_ms = (time.perf_counter() - t_depth) * 1000.0
    if depth_map is None:
        print(f"[OBJECT_UI][DEPTH] unavailable; depth_meters will be left at 0 (Unity should fall back). request_id={request_id} elapsed_ms={depth_ms:.0f}")
    else:
        print(f"[OBJECT_UI][DEPTH] map={depth_map.shape} dtype={depth_map.dtype} elapsed_ms={depth_ms:.0f}")

    cal_ready = gaze_calibration.is_loaded()
    if not cal_ready:
        print("[OBJECT_UI][CAL] inverse calibration unavailable -- gaze_dir will be zero; Unity will fall back to viewport ray.")

    # ---- 4) Per-detection enrichment ----
    payload_for_response = _payload_for_response(payload, w, h)

    detections_out = []
    for i, entry in enumerate(filtered_items):
        item = entry["item"]
        matched_obj = entry["object"]
        match_score = float(entry["score"])
        bbox_xyxy = [float(v) for v in item.get("bbox", (0, 0, 0, 0))]
        yolo_class = str(item.get("class_name") or item.get("class_id") or "detected_object")
        yolo_conf = float(item.get("conf") or 0.0)
        mask_bool = item.get("mask_bool")

        cx = (bbox_xyxy[0] + bbox_xyxy[2]) * 0.5
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) * 0.5
        norm_x = float(np.clip(cx / max(1, w), 0.0, 1.0))
        norm_y = float(np.clip(cy / max(1, h), 0.0, 1.0))

        gaze_dir_x = 0.0
        gaze_dir_y = 0.0
        gaze_dir_z = 0.0
        if cal_ready:
            inv = gaze_calibration.inverse(norm_x, norm_y)
            if inv is not None:
                gaze_dir_x, gaze_dir_y, gaze_dir_z = inv

        depth_meters = 0.0
        depth_source = "none"
        if depth_map is not None:
            d = depth.median_depth_in_mask(depth_map, mask_bool) if mask_bool is not None else None
            if d is None:
                d = depth.median_depth_in_bbox(depth_map, bbox_xyxy)
                depth_source = "bbox" if d is not None else "none"
            else:
                depth_source = "mask"
            if d is not None:
                depth_meters = float(d)

        # The label Unity shows is the DB name (e.g. "로지텍 M650"), not the
        # generic YOLO class. confidence becomes the CLIP score so downstream
        # filtering by confidence still makes sense.
        db_name = str(matched_obj.get("name") or matched_obj.get("id") or yolo_class)
        det = {
            "request_id": request_id,
            "requestId": request_id,
            "label": db_name,
            "class_name": db_name,
            "object_id": str(matched_obj.get("id") or ""),
            "yolo_class": yolo_class,
            "yolo_confidence": yolo_conf,
            "confidence": match_score,
            "conf": match_score,
            "bbox": bbox_xyxy,
            "image_width": w,
            "image_height": h,
            "imageWidth": w,
            "imageHeight": h,
            # World-space anchoring fields for the bubble spawner.
            "gaze_dir_x": float(gaze_dir_x),
            "gaze_dir_y": float(gaze_dir_y),
            "gaze_dir_z": float(gaze_dir_z),
            "depth_meters": float(depth_meters),
            "depth_source": depth_source,
            "norm_x": norm_x,
            "norm_y": norm_y,
        }
        detections_out.append(det)
        print(
            f"[OBJECT_UI][DET] #{i} db={db_name!r} clip={match_score:.3f} "
            f"yolo_class={yolo_class} yolo_conf={yolo_conf:.3f} bbox={bbox_xyxy} "
            f"norm=({norm_x:.3f},{norm_y:.3f}) "
            f"gaze=({gaze_dir_x:+.3f},{gaze_dir_y:+.3f},{gaze_dir_z:+.3f}) "
            f"depth={depth_meters:.2f}m src={depth_source}"
        )

    _send_result(request_id, detections_out, payload_for_response)


def _clip_filter_detections(detections_raw, image_bgr):
    """Run CLIP DB match per YOLO segment, drop misses, dedupe by object_id.

    Returns a list of dicts: [{"item": <yolo_segment>, "object": <db_object>,
    "score": <clip_score>}, ...] containing the best-scoring segment per
    matched object.
    """
    if not detections_raw:
        return []
    if not clip_matcher.is_ready():
        print("[OBJECT_UI][CLIP] CLIP not ready; cannot filter -- returning empty.")
        return []

    best_by_object = {}  # object_id -> entry
    for i, item in enumerate(detections_raw):
        try:
            crop = clip_matcher.prepare_query_crop(item, image_bgr)
        except Exception as exc:
            print(f"[OBJECT_UI][CLIP] #{i} prepare_query_crop failed: {exc}")
            continue

        matched, meta = clip_matcher.resolve_db_match(crop)
        status = meta.get("status", "unknown")
        if matched is None:
            print(
                f"[OBJECT_UI][CLIP] #{i} rejected status={status} "
                f"score={meta.get('score', 0.0):.3f} threshold={meta.get('threshold', 0.0):.3f}"
            )
            continue

        score = float(meta.get("score", 0.0))
        obj_id = str(matched.get("id") or "")
        existing = best_by_object.get(obj_id)
        if existing is None or score > existing["score"]:
            best_by_object[obj_id] = {"item": item, "object": matched, "score": score}
            kept_word = "kept" if existing is None else "replaced previous"
            print(
                f"[OBJECT_UI][CLIP] #{i} {kept_word} -> id={obj_id} "
                f"name={matched.get('name')!r} score={score:.3f}"
            )
        else:
            print(
                f"[OBJECT_UI][CLIP] #{i} duplicate id={obj_id} "
                f"score={score:.3f} <= existing {existing['score']:.3f}; skipped"
            )

    # Stable order: best score first so Unity's first bubble is the most confident.
    return sorted(best_by_object.values(), key=lambda e: e["score"], reverse=True)


def _payload_for_response(source_payload, w, h):
    out = dict(source_payload)
    out["image_width"] = w
    out["image_height"] = h
    return out


def _send_result(request_id: str, detections: list, source_payload: dict,
                 status: str = "ok", reason: str = ""):
    payload = {
        "request_id": request_id,
        "requestId": request_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": "ObjectUI",
        "model": f"YOLO({config.SEG_MODEL_PATH})+DepthAnythingV2",
        "status": status,
        "stage": "yolo",
        "reason": reason,
        "detections": detections,
        "image_width": int(source_payload.get("image_width") or 0),
        "image_height": int(source_payload.get("image_height") or 0),
        "imageWidth": int(source_payload.get("image_width") or 0),
        "imageHeight": int(source_payload.get("image_height") or 0),
        "target_meta": {
            "source": "unity_object_ui_capture",
            "mode": source_payload.get("mode", "object_ui"),
            "gaze_tracked": bool(source_payload.get("gaze_tracked", False)),
            "gaze_viewport": [
                float(source_payload.get("gaze_viewport_x") or 0.5),
                float(source_payload.get("gaze_viewport_y") or 0.5),
            ],
            "calibration_inverse_used": gaze_calibration.is_loaded(),
            "depth_used": depth.is_ready(),
        },
        "response": {
            "name": "ObjectUI",
            "raw": f"detections={len(detections)}",
            "error": reason if status != "ok" else "",
        },
    }
    send_vlm_result_to_unity(payload)


def _debug_dump(image_bgr: np.ndarray, request_id: str):
    try:
        import os
        import cv2
        os.makedirs("/tmp/object_ui_debug", exist_ok=True)
        path = f"/tmp/object_ui_debug/{request_id}.jpg"
        cv2.imwrite(path, image_bgr)
    except Exception as exc:
        print(f"[OBJECT_UI][DEBUG] dump failed: {exc}")
