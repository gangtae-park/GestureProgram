"""Handler for the 'Translate' gesture -- DB-based study version.

CHI 2027 study version -- no OCR, no GPT. Mirrors Search: YOLO picks the
document-like object under gaze (document / name card / ...), CLIP matches it
against object_db, and the pre-authored source text + translation stored in
objects.json ship straight to Unity. OCR + live GPT translation is a
variability confound the study design excludes; the old OCR+GPT handler
lives in git history if the real-world path is ever needed again.

objects.json fields consumed here (per object):
  "text_original":    the text printed on the physical document, verbatim.
  "result_translate": the pre-authored translation to display.

Unity contract is unchanged from the OCR version, so no Unity edits needed:
  packet 1: stage='ocr' VLM_RESULT with response.name = source text and an
            empty translation (Unity shows the source + "translating...")
  packet 2: one stream delta + a stream end carrying the full translation
            (Unity swaps the placeholder for the final text).
"""
import json
import os
import threading
import uuid
from datetime import datetime

import cv2
import numpy as np

from .. import config
from ..gaze import geometry
from ..ui import render
from ..unity import network
from ..vision import clip_matcher, segmentation, target_anchor
from . import register


def _persist(crop_bgr, target_meta, match_meta, payload):
    """Audit trail in vlm_outputs/, same shape as Search's."""
    os.makedirs(config.VLM_OUTPUT_DIR, exist_ok=True)
    timestamp = payload.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = os.path.join(config.VLM_OUTPUT_DIR, f"{timestamp}_Translate")
    log = {
        "timestamp": timestamp,
        "gesture": "Translate",
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
        print(f"[Translate][WARN] crop save failed: {exc}")
    print(f"[Translate] saved -> {base}.json")


def _fail(gesture_name, reason, extra=None):
    payload_extra = dict(extra or {})
    payload_extra["stage"] = "translation"
    network.send_gesture_fail_to_unity(gesture_name, reason, payload_extra)
    print(f"[Translate] gesture fail | {reason}")


def _match_worker(crop_bgr, target_meta, gesture_name, anchor=None):
    """CLIP match -> DB translate fields -> the two-packet Unity sequence."""
    matched_obj, match_meta = clip_matcher.resolve_db_match(crop_bgr)

    if matched_obj is None:
        reason = clip_matcher.fail_reason_for(match_meta["status"], match_meta)
        fail_payload = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
            "gesture": gesture_name,
            "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
            "status": "fail",
            "stage": "translation",
            "reason": reason,
            "target_meta": target_meta,
            "response": {"error": reason},
        }
        _persist(crop_bgr, target_meta, match_meta, fail_payload)
        network.send_vlm_result_to_unity(fail_payload)
        print(f"[Translate] gesture fail | {match_meta['status']} | {reason}")
        return

    text_original = str(matched_obj.get("text_original") or "").strip()
    translation = str(matched_obj.get("result_translate") or "").strip()
    if not text_original or not translation:
        _fail(
            gesture_name,
            f"DB object {matched_obj['id']!r} has no translate fields "
            f"(text_original / result_translate).",
        )
        return

    print(
        f"[Translate] matched id={matched_obj['id']} score={match_meta['score']:.3f} "
        f"name={matched_obj.get('name')!r}"
    )

    # ---- packet 1: source text (Unity shows it with a placeholder) ----
    ocr_response = {"name": text_original, "translation": ""}
    target_anchor.merge_into_response(ocr_response, anchor or {})
    ocr_stage_payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture_name,
        "stage": "ocr",
        "model": f"YOLO+CLIP({config.CLIP_MODEL_NAME})",
        "status": "ok",
        "target_meta": target_meta,
        "match_meta": match_meta,
        "response": ocr_response,
    }
    network.send_vlm_result_to_unity(ocr_stage_payload)

    # ---- packet 2: the stored translation via the stream channel Unity
    # already understands (one delta with the full text, then the end) ----
    stream_id = uuid.uuid4().hex
    network.send_stream_delta_to_unity(
        stream_id=stream_id,
        gesture=gesture_name,
        delta=translation,
        stage="translation",
        seq=0,
        target_meta=target_meta,
    )
    translation_response = {"name": text_original, "translation": translation}
    target_anchor.merge_into_response(translation_response, anchor or {})
    network.send_stream_end_to_unity(
        stream_id=stream_id,
        gesture=gesture_name,
        stage="translation",
        status="ok",
        response=translation_response,
        target_meta=target_meta,
        error="",
    )

    final_payload = dict(ocr_stage_payload)
    final_payload["stage"] = "translation"
    final_payload["response"] = translation_response
    _persist(crop_bgr, target_meta, match_meta, final_payload)


@register("Translate")
def handle(captured_frame: np.ndarray, norm_points, gesture_name: str) -> np.ndarray:
    """Called on Translate END. Same target selection as Search: gaze bbox ->
    YOLO overlap -> masked CLIP crop -> DB match, then the stored translation
    ships to Unity."""
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
        _fail(gesture_name, "Not enough gaze points.")
        return empty

    # ---- YOLO segment selection (documents come in as e.g. 'document') ----
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
                f"[Translate][YOLO] target | class={chosen['class_name']} "
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
        _fail(gesture_name, "No YOLO segment overlaps gaze.")
        return overlay

    # ---- CLIP query crop -- masked when available ----
    try:
        crop_for_clip = clip_matcher.prepare_query_crop(target, captured_frame)
    except Exception as exc:
        print(f"[Translate][ERROR] prepare_query_crop failed: {exc}")
        cv2.putText(
            overlay, f"CROP ERROR: {exc}",
            (20, overlay.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )
        _fail(gesture_name, f"CROP ERROR: {exc}")
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

    anchor = target_anchor.compute(captured_frame, target.get("bbox"), target.get("mask_bool"))

    threading.Thread(
        target=_match_worker,
        args=(crop_for_clip, target_meta, gesture_name, anchor),
        daemon=True,
    ).start()

    cv2.putText(
        overlay, "TRANSLATE: matching against DB...",
        (20, overlay.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA,
    )
    return overlay
