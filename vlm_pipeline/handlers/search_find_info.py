"""Handler for the 'Search/Find Info' gesture.

CHI 2027 study version -- no GPT VLM. We have a fixed 3-object database; CLIP
picks the best match and we ship the pre-authored info card straight to Unity.

Pipeline:
  1. Compute gaze bbox from the buffered gesture trajectory.
  2. Run YOLO on the captured frame; pick the segment that overlaps the gaze.
  3. Build a (masked) crop of that segment, embed with CLIP, look up the best
     match in object_db.
  4. Save the crop + match metadata to disk for the log, push the stored info
     fields back to Unity.
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
    object_db,
    render,
    segmentation,
)
from . import register


def _persist(crop_bgr, target_meta, match_meta, payload):
    """Mirror the old vlm_outputs/ behaviour so we still have an audit trail."""
    os.makedirs(config.VLM_OUTPUT_DIR, exist_ok=True)
    timestamp = payload.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = os.path.join(config.VLM_OUTPUT_DIR, f"{timestamp}_Search_Find_Info")
    log = {
        "timestamp": timestamp,
        "gesture": "Search/Find Info",
        "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
        "target_meta": target_meta,
        "match_meta": match_meta,
        "payload": payload,
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    try:
        cv2.imwrite(base + ".png", crop_bgr)
    except Exception as exc:
        print(f"[SEARCH][WARN] crop save failed: {exc}")
    print(f"[SEARCH] saved -> {base}.json")


def _resolve_match(crop_bgr):
    """Run CLIP and return (object_dict_or_None, match_meta).
    object_dict is None when we should treat this as a gesture fail.
    """
    db = object_db.get_db()
    if db is None or db.embedding_matrix is None or len(db.embedding_matrix) == 0:
        return None, {"status": "db_empty"}
    if not clip_matcher.is_ready():
        return None, {"status": "clip_unavailable"}

    try:
        query_emb = clip_matcher.embed_image(crop_bgr)
    except Exception as exc:
        print(f"[SEARCH][ERROR] embed failed: {exc}")
        return None, {"status": "embed_error", "error": str(exc)}

    match = clip_matcher.match_against_db(query_emb, db)
    if match is None:
        return None, {"status": "no_candidates"}

    ranking = [(oid, float(s)) for oid, s in match["ranking"]]
    score = float(match["score"])
    if score < config.CLIP_MATCH_MIN_SCORE:
        return None, {
            "status": "below_threshold",
            "score": score,
            "ranking": ranking,
            "threshold": float(config.CLIP_MATCH_MIN_SCORE),
        }

    obj = match["object"]
    return obj, {
        "status": "matched",
        "object_id": obj["id"],
        "score": score,
        "ranking": ranking,
        "threshold": float(config.CLIP_MATCH_MIN_SCORE),
    }


def _match_worker(crop_bgr, target_meta, gesture_name):
    matched_obj, match_meta = _resolve_match(crop_bgr)

    if matched_obj is None:
        reason = {
            "db_empty": "Object DB is empty -- add reference images.",
            "clip_unavailable": "CLIP model is not loaded.",
            "embed_error": "CLIP embedding failed.",
            "no_candidates": "No reference embeddings available.",
            "below_threshold": (
                f"Below CLIP threshold "
                f"({match_meta.get('score', 0.0):.2f} < {config.CLIP_MATCH_MIN_SCORE:.2f})."
            ),
        }.get(match_meta["status"], "Object not recognised.")

        fail_payload = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
            "gesture": gesture_name,
            "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
            "status": "fail",
            "reason": reason,
            "target_meta": target_meta,
            "response": {"error": reason},
        }
        _persist(crop_bgr, target_meta, match_meta, fail_payload)
        network.send_vlm_result_to_unity(fail_payload)
        print(f"[SEARCH] gesture fail | {match_meta['status']} | {reason}")
        return

    response = {
        "name": matched_obj.get("name", ""),
        "result_search": matched_obj.get("result_search", ""),
    }
    print(
        f"[SEARCH] matched id={matched_obj['id']} score={match_meta['score']:.3f} "
        f"name={matched_obj.get('name')!r}"
    )
    success_payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture_name,
        "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
        "status": "ok",
        "target_meta": target_meta,
        "match_meta": match_meta,
        "response": response,
    }
    _persist(crop_bgr, target_meta, match_meta, success_payload)
    network.send_vlm_result_to_unity(success_payload)


@register("Search/Find Info")
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
        return empty

    # ---- YOLO segment selection ----
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
                f"[SEARCH][YOLO] target | class={chosen['class_name']} "
                f"conf={chosen['conf']:.2f} overlap={overlap:.2f} iou={iou:.2f} "
                f"bbox={chosen['bbox']}"
            )

    overlay = render.render_target_overlay(
        captured_frame, pixel_points, gaze_bbox,
        target, target_source, yolo_items, gesture_name,
    )

    if target is None:
        cv2.putText(
            overlay, "NO TARGET (no YOLO segment overlaps gaze)",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )
        return overlay

    # ---- CLIP query crop -- masked when available ----
    try:
        crop_for_clip = clip_matcher.prepare_query_crop(target, captured_frame)
    except Exception as exc:
        print(f"[SEARCH][ERROR] prepare_query_crop failed: {exc}")
        cv2.putText(
            overlay, f"CROP ERROR: {exc}",
            (20, overlay.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return overlay

    target_meta = {
        "source": target_source,
        "bbox": list(target["bbox"]),
        "best_overlap": float(target.get("best_overlap", 0.0)),
        "best_iou": float(target.get("best_iou", 0.0)),
        "class_name": target.get("class_name"),
        "conf": float(target.get("conf", 0.0)) if "conf" in target else None,
        "gaze_bbox": list(gaze_bbox),
        "clip_masked_crop": bool(
            config.CLIP_USE_MASKED_CROP and target.get("mask_bool") is not None
        ),
    }

    threading.Thread(
        target=_match_worker,
        args=(crop_for_clip, target_meta, gesture_name),
        daemon=True,
    ).start()

    return overlay
