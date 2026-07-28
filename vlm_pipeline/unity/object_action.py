"""Handle OBJECT_ACTION packets coming from the Unity bubble menu.

The bubble flow already CLIP-matched each YOLO segment to one of the registered
DB objects when it shipped detections to Unity. When the user picks a wedge
(Search / Ask / Translate / Compare / Anchor / Save / Capture) on a particular
bubble, Unity sends OBJECT_ACTION with the matched object_id (plus a second id
for Compare). Python skips detection entirely and builds the same VlmResultPayload
shape the corresponding gesture handler would produce -- so the existing
Unity-side card spawners render the result with no changes.

Position handling on the Unity side: ObjectActionCommandBridge calls
ResultCardSpawner.OverrideNextSpawnPosition(bubbleWorldPosition) before sending
the packet, so whatever Python ships back lands exactly on (or between) the
clicked bubble(s). Per-action depth/gaze_dir in the response is therefore
optional -- we still fill them with the request's cached anchor when we can,
but the spawner override is the source of truth.

For Translate and Capture (which inherently need pixel-space data), we run a
quick YOLO+CLIP pass on the latest ADB frame to locate the target object's
current bbox/mask. Other actions just need the DB record.
"""
from datetime import datetime
from typing import Optional

import numpy as np

from .. import config, state
from . import network
from ..vision import clip_matcher, object_db, segmentation, target_anchor


# ---------- public entry ----------

def process(packet: dict):
    action = (packet.get("action") or "").strip()
    object_id = (packet.get("object_id") or "").strip()
    second_object_id = (packet.get("second_object_id") or "").strip()
    request_id = (packet.get("request_id") or "").strip()

    if action == "Cancel":
        print("[OBJECT_ACTION] Cancel received (no-op on Python side).")
        return

    if not object_id:
        _send_fail(action, request_id, "Missing object_id.")
        return

    obj = _lookup_object(object_id)
    if obj is None:
        _send_fail(action, request_id, f"Unknown object_id '{object_id}'.")
        return

    print(f"[OBJECT_ACTION] action={action} obj={obj.get('name')!r} id={object_id} request_id={request_id}")

    # Response-latency equalisation for the user study: DB-only actions
    # answer in ~10 ms, so sleeping first pins the UI method's system time to
    # the configured floor (comparable to Gesture's YOLO+CLIP cost). Capture
    # and Ask are EXEMPT -- they run a real YOLO+CLIP pass of their own, so
    # adding the floor on top would overshoot it. We run on a dispatch worker
    # thread, so this never blocks the UDP receive loop.
    if action not in ("Capture", "Ask") and config.UI_RESULT_DELAY_S > 0:
        import time
        time.sleep(config.UI_RESULT_DELAY_S)

    if action == "Search":
        _do_search(obj, request_id)
    elif action == "Ask":
        _do_ask(obj, request_id)
    elif action == "Compare":
        second = _lookup_object(second_object_id) if second_object_id else None
        if second is None:
            _send_fail(action, request_id, "Compare needs a valid second object.")
            return
        _do_compare(obj, second, request_id)
    elif action == "Anchor":
        _do_anchor(obj, request_id)
    elif action == "Save":
        _do_save(obj, request_id)
    elif action == "Capture":
        _do_capture(obj, request_id)
    elif action == "Translate":
        _do_translate(obj, request_id)
    else:
        _send_fail(action, request_id, f"Unsupported action '{action}'.")


# ---------- per-action handlers ----------

def _do_search(obj: dict, request_id: str):
    response = {
        "name": obj.get("name", ""),
        "object_id": obj.get("id", ""),
        "result_search": obj.get("result_search", ""),
    }
    payload = _base_payload(
        gesture="Search",
        request_id=request_id,
        model_tag=f"OBJECT_ACTION+DB({obj.get('id')})",
        response=response,
    )
    network.send_vlm_result_to_unity(payload)
    print(f"[OBJECT_ACTION][Search] sent name={response['name']!r}")


def _do_ask(obj: dict, request_id: str):
    """Phase 1 (object_recognized). The user's spoken question still arrives via
    the existing /ask_voice HTTP endpoint and triggers process_ask_question for
    the final answer. We also stash the matched object in state.latest_ask_target
    so phase 2 has everything it needs."""
    anchor = _fresh_anchor_for_object(obj)
    response = {
        "name": obj.get("name", ""),
        "object_id": obj.get("id", ""),
    }
    target_anchor.merge_into_response(response, anchor or {})

    # Stash for the Android STT follow-up (mimics handlers/ask.py at gesture END).
    crop = anchor.get("_clip_crop") if anchor else None
    matched_meta = anchor.get("_match_meta") if anchor else None
    with state.ask_lock:
        state.latest_ask_target = {
            "crop": crop if crop is not None else np.zeros((1, 1, 3), dtype=np.uint8),
            "target_meta": {"source": "OBJECT_ACTION", "object_id": obj.get("id")},
            "match_meta": matched_meta or {"status": "matched", "score": 1.0, "object_id": obj.get("id")},
            "gesture_name": "Ask",
            "matched_object": obj,
            "timestamp": __import__("time").time(),
            "anchor": _clean_anchor(anchor),
        }

    payload = _base_payload(
        gesture="Ask",
        request_id=request_id,
        model_tag=f"OBJECT_ACTION+DB({obj.get('id')})",
        response=response,
        stage="object_recognized",
    )
    network.send_vlm_result_to_unity(payload)
    print(f"[OBJECT_ACTION][Ask] phase1 sent name={response['name']!r}; awaiting /ask_voice transcript.")


def _do_compare(obj_a: dict, obj_b: dict, request_id: str):
    a_id, b_id = obj_a.get("id", ""), obj_b.get("id", "")
    db = object_db.get_db()
    compare_rows = db.lookup_comparison(a_id, b_id) if db is not None else None

    name_a = obj_a.get("name", a_id)
    name_b = obj_b.get("name", b_id)
    pair_name = f"{name_a} vs {name_b}"
    pair_id = "_vs_".join(sorted([a_id, b_id]))

    if not compare_rows:
        print(f"[OBJECT_ACTION][Compare] no comparison registered for ({a_id}, {b_id}); placeholder.")
        compare_rows = [{
            "category": "",
            "value_a": "비교 정보가 등록되어 있지 않습니다.",
            "value_b": "",
        }]

    response = {
        "name": pair_name,
        "object_id": pair_id,
        "name_a": name_a,
        "name_b": name_b,
        "compare_rows": compare_rows,
        "result_search": _compare_rows_to_text(name_a, name_b, compare_rows),
        "objects": [
            {"name": name_a, "object_id": a_id, "result_search": obj_a.get("result_search", "")},
            {"name": name_b, "object_id": b_id, "result_search": obj_b.get("result_search", "")},
        ],
    }
    payload = _base_payload(
        gesture="Compare",
        request_id=request_id,
        model_tag=f"OBJECT_ACTION+DB({a_id},{b_id})",
        response=response,
    )
    network.send_vlm_result_to_unity(payload)
    print(f"[OBJECT_ACTION][Compare] sent {pair_name} ({len(compare_rows)} rows)")


def _compare_rows_to_text(name_a: str, name_b: str, rows: list) -> str:
    if not rows:
        return ""
    lines = []
    for row in rows:
        cat = (row.get("category") or "").strip()
        va = (row.get("value_a") or "").strip()
        vb = (row.get("value_b") or "").strip()
        prefix = f"•{cat}: " if cat else "•"
        lines.append(f"{prefix}{name_a} {va} / {name_b} {vb}")
    return "\n".join(lines)


def _do_anchor(obj: dict, request_id: str):
    # Fast-path: Anchor only needs DB metadata. Unity already knows the world
    # position from the clicked bubble (OverrideNextSpawnPosition was set by
    # ObjectActionCommandBridge before this packet was sent), so we don't
    # re-run YOLO + CLIP + Depth Anything just to recompute gaze_dir/depth
    # that the spawner will ignore. depth_meters is left at 0 so the spawner
    # consumes the override unchanged. Round-trip is now milliseconds, not
    # seconds.
    response = {
        "name": obj.get("name", ""),
        "object_id": obj.get("id", ""),
        "message": "Anchor object recognised.",
    }
    payload = _base_payload(
        gesture="Anchor",
        request_id=request_id,
        model_tag=f"OBJECT_ACTION+DB({obj.get('id')})",
        response=response,
        stage="ack",
    )
    network.send_vlm_result_to_unity(payload)
    print(f"[OBJECT_ACTION][Anchor] fast-path sent name={response['name']!r}")


def _do_save(obj: dict, request_id: str):
    # Same fast-path reasoning as Anchor: NoteManager just needs the DB record
    # and a world position. Unity's override supplies the position.
    response = {
        "name": obj.get("name", ""),
        "object_id": obj.get("id", ""),
        "message": "Save object recognised.",
    }
    payload = _base_payload(
        gesture="Save",
        request_id=request_id,
        model_tag=f"OBJECT_ACTION+DB({obj.get('id')})",
        response=response,
        stage="ack",
    )
    network.send_vlm_result_to_unity(payload)
    print(f"[OBJECT_ACTION][Save] fast-path sent name={response['name']!r}")


def _do_capture(obj: dict, request_id: str):
    """Capture needs a bbox so the CaptureControlCard can size its frame. We
    run a fresh YOLO+CLIP pass on the latest ADB frame to find this object's
    current bbox; if it isn't visible right now, fall back to a default sizer."""
    frame, item = _find_segment_for_object(obj)
    target_meta = {"source": "OBJECT_ACTION", "object_id": obj.get("id", "")}
    if frame is not None and item is not None:
        bbox = [int(v) for v in item.get("bbox", (0, 0, 0, 0))]
        target_meta["bbox"] = bbox
        target_meta["frame_size"] = [int(frame.shape[1]), int(frame.shape[0])]
        anchor = target_anchor.compute(frame, item.get("bbox"), item.get("mask_bool"))
    else:
        anchor = {}
        print("[OBJECT_ACTION][Capture] target not visible in current frame; sending without bbox.")

    response = {
        "name": obj.get("name", ""),
        "object_id": obj.get("id", ""),
        "message": "Capture object recognised.",
    }
    target_anchor.merge_into_response(response, anchor)
    payload = _base_payload(
        gesture="Capture",
        request_id=request_id,
        model_tag=f"OBJECT_ACTION+DB({obj.get('id')})",
        response=response,
        stage="ack",
        target_meta=target_meta,
    )
    network.send_vlm_result_to_unity(payload)
    print(f"[OBJECT_ACTION][Capture] sent name={response['name']!r} bbox={target_meta.get('bbox')}")


def _do_translate(obj: dict, request_id: str):
    """DB-based study version: ship the pre-authored text_original /
    result_translate straight from objects.json -- no OCR, no GPT. The bubble
    already CLIP-identified the object, so translation is a pure DB lookup;
    the card position comes from Unity's spawn override, so no anchor pass is
    needed either."""
    text = str(obj.get("text_original") or "").strip()
    translation = str(obj.get("result_translate") or "").strip()
    if not text or not translation:
        _send_fail("Translate", request_id,
                   f"{obj.get('name', '')}에 등록된 번역 데이터가 없습니다 "
                   "(objects.json의 text_original / result_translate 확인).",
                   gesture="Translate", stage="translation")
        return

    response = {
        "name": text,
        "object_id": obj.get("id", ""),
        "translation": translation,
    }
    payload = _base_payload(
        gesture="Translate",
        request_id=request_id,
        model_tag=f"OBJECT_ACTION+DB({obj.get('id')})",
        response=response,
        stage="translation",
        target_meta={"source": "OBJECT_ACTION", "object_id": obj.get("id", "")},
    )
    network.send_vlm_result_to_unity(payload)
    print(f"[OBJECT_ACTION][Translate] sent text={text!r} ko={translation!r}")


# ---------- helpers ----------

def _lookup_object(object_id: str) -> Optional[dict]:
    """Resolve a DB object by id; fall back to a name match (CLIP filter used to
    ship the DB name in `label`, which is what the user's UI label shows)."""
    if not object_id:
        return None
    db = object_db.get_db()
    if db is None:
        return None
    direct = db.get_object(object_id)
    if direct is not None:
        return direct
    # Name fallback (Korean labels may arrive without their DB id).
    for o in (db.objects or []):
        if str(o.get("name", "")) == object_id:
            return o
    return None


def _find_segment_for_object(obj: dict):
    """Run a fresh YOLO+CLIP pass on the latest ADB frame and return
    (frame_bgr, yolo_item) for the target object_id. (None, None) on miss."""
    with state.frame_lock:
        frame = None if state.latest_frame is None else state.latest_frame.copy()
    if frame is None:
        return None, None

    items = segmentation.run_yolo(frame)
    if not items:
        return frame, None

    target_id = obj.get("id")
    best = None
    best_score = -1.0
    for item in items:
        try:
            crop = clip_matcher.prepare_query_crop(item, frame)
        except Exception:
            continue
        matched, meta = clip_matcher.resolve_db_match(crop, log_study=False)
        if matched is None or str(matched.get("id", "")) != str(target_id):
            continue
        score = float(meta.get("score", 0.0))
        if score > best_score:
            best_score = score
            best = item
    return frame, best


def _fresh_anchor_for_object(obj: dict) -> Optional[dict]:
    """Compute (gaze_dir, depth) for an object by locating it in the latest
    frame. Returns None if not visible -- callers should treat that as "no
    anchor info" (the Unity spawner override still places the card correctly)."""
    frame, item = _find_segment_for_object(obj)
    if frame is None or item is None:
        return None
    anchor = target_anchor.compute(frame, item.get("bbox"), item.get("mask_bool"))
    # Stash crop + meta for callers that want them (Ask phase 2 reuses the crop).
    try:
        anchor["_clip_crop"] = clip_matcher.prepare_query_crop(item, frame)
        anchor["_match_meta"] = {"status": "matched", "object_id": obj.get("id"), "score": 1.0}
    except Exception:
        pass
    return anchor


def _clean_anchor(anchor: Optional[dict]) -> Optional[dict]:
    if not anchor:
        return anchor
    return {k: v for k, v in anchor.items() if not k.startswith("_")}


def _base_payload(*, gesture: str, request_id: str, model_tag: str, response: dict,
                  stage: str = "answer", target_meta: Optional[dict] = None) -> dict:
    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture,
        "model": model_tag,
        "status": "ok",
        "stage": stage,
        "request_id": request_id,
        "requestId": request_id,
        "target_meta": target_meta or {"source": "OBJECT_ACTION"},
        "response": response,
    }


def _send_fail(action: str, request_id: str, reason: str,
               gesture: Optional[str] = None, stage: str = "answer"):
    payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": gesture or action or "ObjectAction",
        "model": "OBJECT_ACTION",
        "status": "fail",
        "stage": stage,
        "reason": reason,
        "request_id": request_id,
        "requestId": request_id,
        "target_meta": {"source": "OBJECT_ACTION"},
        "response": {"name": "", "error": reason},
    }
    network.send_vlm_result_to_unity(payload)
    print(f"[OBJECT_ACTION][FAIL] action={action} reason={reason}")
