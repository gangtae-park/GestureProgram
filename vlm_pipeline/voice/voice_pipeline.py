"""Voice-command routing -- GazePointAR-style call grounded in the object DB.

Bridges the /voice_command HTTP endpoint to one GPT call per utterance:

  1. Question end -> the frame captured at request arrival and THE single
     capture-moment gaze pixel: state.latest_gaze_norm read at the same
     instant, which is delay-aligned to the adb frame content (no head
     compensation -- one instantaneous point on that exact frame). The
     Unity-sent trail / listen-start snapshot are fallbacks only.
  2. YOLO picks the segment at the gaze pixel and CLIP matches it against
     object_db -- same target selection the gesture handlers use. The DB
     fields of the match (name / result_search / text_original /
     result_translate) become the {db_info} ground-truth block; a miss
     becomes an explicit "no match, use the image" note.
  3. ONE GPT call with the full frame + gaze text + DB block + verbatim
     transcript (config.VOICE_COMMAND_PROMPT). The model resolves the
     pronoun, classifies the seven-way intent, and answers grounded in the
     DB fields.
  4. The result is sent to Unity as one VLM_RESULT whose `gesture` field is
     the classified intent, so ResultCardSpawner shows the matching card.
  5. The voice request context (request_id, gaze pixel) is published to a
     module singleton so network.send_vlm_result_to_unity can stamp every
     outbound VLM_RESULT with the request_id -- this is what lets Unity's
     CaptureContextRegistry snap the response card to the pose registered
     at listen-start.
"""
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

from .. import config, state


# Canonical intents Unity's ResultCardSpawner switches on.
CANONICAL_INTENTS = (
    "Search", "Translate", "Compare", "Anchor", "Save", "Capture", "Ask",
)


# =========================================================================
# Active voice context (consumed by network._stamp_voice_request_id)
# =========================================================================
# We can't use threading.local() here because downstream sends may happen on
# other threads. Instead we use a lock-guarded module singleton with a short
# TTL: dispatch publishes the context, and any send_vlm_result_to_unity that
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
        # extras carries auxiliary data (e.g. Save's parsed note_content) so
        # the network stamper can inject it into outbound payloads.
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
    live_gaze_norm: Optional[Tuple[float, float]] = None,
    live_gaze_tracked: bool = False,
) -> Tuple[str, str]:
    """Route a /voice_command request. Returns (intent, status) for the
    caller to log -- status is "ok" or a short failure tag. Sending the
    VLM_RESULT to Unity is handled inside; on any hard failure we still emit
    a fail payload so the Unity card doesn't hang.

    - transcript:         final STT text from Android SpeechRecognizer.
    - frame_bgr:          ADB screenrecord frame captured at request-arrival,
                          i.e. the moment the spoken question ended.
    - gaze_viewport_bl:   Single (nx, ny) snapshot at listen-start, Unity's
                          bottom-left origin convention (0..1). Last-resort
                          fallback.
    - gaze_tracked:       True iff the eye tracker had a valid sample at
                          listen-start.
    - request_id:         the id Unity registered its CapturePose with.
    - gaze_trail_bl:      Optional Unity-buffered (nx, ny) samples; fallback
                          when no live gaze is available.
    - live_gaze_norm:     PRIMARY gaze source: state.latest_gaze_norm read at
                          the same instant the frame was copied. Already
                          top-left normalized AND delay-aligned to the adb
                          frame content, so it is the single capture-moment
                          gaze point on this exact frame -- no head-motion
                          compensation involved.
    - live_gaze_tracked:  tracked flag of that same delayed sample.
    """
    # Local import: vlm_client sits in llm/, keep module load light.
    from ..llm import vlm_client

    if frame_bgr is None:
        _emit_voice_fail(request_id, transcript, "Ask", "no ADB frame available")
        return "Ask", "no_frame"

    h, w = frame_bgr.shape[:2]
    gaze_pixel, gaze_source = _resolve_question_end_gaze(
        live_gaze_norm, live_gaze_tracked,
        gaze_trail_bl, gaze_viewport_bl, gaze_tracked, w, h
    )
    _publish_context(request_id, gaze_pixel, gaze_tracked)

    # YOLO + CLIP at the gaze pixel -> DB ground truth for the prompt.
    # (Runs BEFORE the GPT call on purpose: a parallel variant was tried and
    # reverted -- without the DB block in the prompt, Ask answers lose their
    # grounding and drift.)
    db_info, db_meta, target_bbox, matched_obj, yolo_items = _lookup_db_target(
        frame_bgr, gaze_pixel
    )

    # Latency: GPT sees a downscaled copy (fewer image tokens); YOLO/CLIP
    # above already ran on the full frame. gaze_info cites coordinates in the
    # DOWNSCALED image so the pixel the model reads matches what it sees.
    send_frame, scale = _downscale_for_vlm(frame_bgr)
    sh, sw = send_frame.shape[:2]
    scaled_gaze = (int(round(gaze_pixel[0] * scale)), int(round(gaze_pixel[1] * scale)))
    gaze_info = _build_gaze_info(scaled_gaze, gaze_tracked, gaze_source, sw, sh)
    prompt = (
        config.VOICE_COMMAND_PROMPT
        .replace("{transcript}", transcript)
        .replace("{gaze_info}", gaze_info)
        .replace("{db_info}", db_info)
    )

    t0 = time.perf_counter()
    result = vlm_client.call_vlm_on_crop(
        send_frame, prompt,
        model=config.VOICE_OPENAI_MODEL,
        reasoning_effort=None,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if result is None:
        _emit_voice_fail(request_id, transcript, "Ask", "GPT request failed")
        return "Ask", "vlm_error"
    if "raw" in result and "answer" not in result:
        _emit_voice_fail(request_id, transcript, "Ask", "GPT returned unparseable output")
        return "Ask", "vlm_unparseable"

    intent = str(result.get("intent") or "").strip()
    if intent not in CANONICAL_INTENTS:
        # Never surface a made-up card type; Ask is the study's fallback bucket.
        print(f"[VOICE_PIPELINE][WARN] non-canonical intent {intent!r} -> Ask")
        intent = "Ask"
    name = str(result.get("name") or "").strip()
    answer = str(result.get("answer") or "").strip()
    note_content = str(result.get("note_content") or "").strip()

    from .. import study
    study.log_event("intent_classified", f"{intent} vlm_ms={elapsed_ms:.0f}")

    # Re-publish with Save's note body so network's stamper can forward it
    # (mirrors the old classifier-extras path Unity's NoteManager expects).
    if intent == "Save" and note_content:
        _publish_context(request_id, gaze_pixel, gaze_tracked,
                         extras={"note_content": note_content})

    response = {
        "name": name or "Voice request",
        "answer": answer,
    }
    if intent == "Save" and note_content:
        response["note_text"] = note_content

    # Unity's cards read intent-specific fields (SearchResultCard shows
    # response.result_search, TranslateResultCard shows response.name +
    # response.translation, CompareResultCard shows name_a/name_b +
    # compare_rows) -- fill them from the DB so the voice path surfaces the
    # same pre-authored content as the gesture path.
    if matched_obj is not None:
        if matched_obj.get("name"):
            response["name"] = matched_obj["name"]
        if intent == "Search" and matched_obj.get("result_search"):
            response["result_search"] = matched_obj["result_search"]
        elif intent == "Translate" and matched_obj.get("result_translate"):
            response["name"] = matched_obj.get("text_original") or response["name"]
            response["translation"] = matched_obj["result_translate"]

    if intent == "Compare":
        # Two-target case: resolve the second object from the remaining YOLO
        # segments (lazy -- only paid when the query actually is a Compare).
        second_obj, second_bbox = _resolve_second_object(
            frame_bgr, yolo_items, target_bbox, matched_obj, gaze_pixel
        )
        _fill_compare_response(response, matched_obj, second_obj, answer)
        if second_bbox is not None:
            db_meta["second_bbox"] = list(second_bbox)
            db_meta["second_object_id"] = second_obj.get("id")

    # Anchor the card to the ACTUAL object (gaze_dir + metric depth), same as
    # the gesture handlers do. Without this the payload has no depth_meters,
    # and Unity falls back to a fixed distance along the listen-start gaze --
    # which is why voice cards used to drift off-object.
    if target_bbox is not None:
        from ..vision import target_anchor
        target_mask = None
        for it in yolo_items:
            if it.get("bbox") == target_bbox:
                target_mask = it.get("mask_bool")
                break
        anchor = target_anchor.compute(frame_bgr, target_bbox, target_mask)
        target_anchor.merge_into_response(response, anchor)

    payload = {
        "request_id": request_id,
        "requestId": request_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
        "gesture": intent,
        "model": f"YOLO+CLIP+GazePointAR({config.VOICE_OPENAI_MODEL})",
        "status": "ok",
        "stage": "answer",
        "target_meta": {
            "source": "voice_gazepointar",
            "user_question": transcript,
            "gaze_pixel": [int(gaze_pixel[0]), int(gaze_pixel[1])],
            "gaze_tracked": bool(gaze_tracked),
            "gaze_source": gaze_source,
            "frame_size": [int(w), int(h)],
            "vlm_ms": round(elapsed_ms, 1),
            "db_lookup": db_meta,
        },
        "response": response,
    }

    from ..unity import network as network_mod
    network_mod.send_vlm_result_to_unity(payload)

    _publish_target_canvas(_render_debug_canvas(
        frame_bgr, gaze_pixel, intent, name, answer, target_bbox
    ))
    print(
        f"[VOICE_PIPELINE] single-call ok intent={intent!r} name={name!r} "
        f"gaze={gaze_pixel}({gaze_source}) {elapsed_ms:.0f}ms"
    )
    return intent, "ok"


# =========================================================================
# Helpers
# =========================================================================

def _lookup_db_target(frame_bgr, gaze_pixel):
    """YOLO segment at the gaze pixel -> CLIP -> object_db.

    Returns (db_info_text, meta, target_bbox, matched_obj, yolo_items):
      - db_info_text : the {db_info} block for VOICE_COMMAND_PROMPT -- DB
                       fields as authoritative ground truth on a match, an
                       explicit "no match" note otherwise.
      - meta         : lookup provenance for target_meta / logs.
      - target_bbox  : matched YOLO bbox for the debug overlay, or None.
      - matched_obj  : the objects.json entry on a match, else None --
                       dispatch copies its intent-specific fields into the
                       Unity response.
      - yolo_items   : ALL segments from this frame, so Compare can resolve
                       a second object without re-running YOLO.
    """
    from ..gaze import geometry
    from ..vision import clip_matcher, segmentation

    h, w = frame_bgr.shape[:2]
    px, py = gaze_pixel
    r = max(24, int(min(w, h) * 0.06))
    gaze_bbox = (max(0, px - r), max(0, py - r), min(w - 1, px + r), min(h - 1, py + r))
    meta = {"gaze_bbox": list(gaze_bbox)}

    yolo_items = []

    def _no_match(reason):
        meta["status"] = reason
        return (
            f"No database match for the object at the gaze pixel ({reason}). "
            "Resolve the referent from the image alone.",
            meta,
            None,
            None,
            yolo_items,
        )

    try:
        yolo_items = segmentation.run_yolo(frame_bgr)
    except Exception as exc:
        return _no_match(f"yolo_error: {exc}")
    if not yolo_items:
        return _no_match("no YOLO segment detected in the frame")

    idx, overlap, iou = geometry.pick_best_overlap(gaze_bbox, yolo_items)
    if idx < 0 or overlap <= 0:
        return _no_match("no YOLO segment overlaps the gaze")

    target = dict(yolo_items[idx])
    meta.update({
        "bbox": list(target["bbox"]),
        "class_name": target.get("class_name"),
        "conf": float(target.get("conf", 0.0)),
        "best_overlap": float(overlap),
        "best_iou": float(iou),
    })

    try:
        crop = clip_matcher.prepare_query_crop(target, frame_bgr)
        matched_obj, match_meta = clip_matcher.resolve_db_match(crop)
    except Exception as exc:
        return _no_match(f"clip_error: {exc}")

    meta["clip_status"] = match_meta.get("status")
    if "score" in match_meta:
        meta["clip_score"] = float(match_meta["score"])
    if matched_obj is None:
        return _no_match(f"CLIP {match_meta.get('status', 'no_match')}")

    meta["status"] = "matched"
    meta["matched_object_id"] = matched_obj.get("id")
    meta["db_name"] = matched_obj.get("name")

    lines = [
        "A YOLO+CLIP lookup already identified the object at the gaze pixel "
        f"against the study's object database (CLIP score "
        f"{match_meta.get('score', 0.0):.2f}):",
        f"- name: {matched_obj.get('name', '')}",
    ]
    if matched_obj.get("result_search"):
        lines.append(f"- info: {matched_obj['result_search']}")
    if matched_obj.get("text_original"):
        lines.append(f"- printed text on the object (verbatim): {matched_obj['text_original']}")
    if matched_obj.get("result_translate"):
        lines.append(f"- pre-authored translation of that text: {matched_obj['result_translate']}")
    lines.append(
        "These fields are AUTHORITATIVE ground truth for the gaze target."
    )
    print(
        f"[VOICE_PIPELINE][DB] matched id={matched_obj.get('id')} "
        f"score={match_meta.get('score', 0.0):.3f} class={target.get('class_name')}"
    )
    return "\n".join(lines), meta, target.get("bbox"), matched_obj, yolo_items


def _downscale_for_vlm(frame_bgr):
    """Return (frame_for_gpt, scale). Downscales so the longest side is
    config.VOICE_IMAGE_MAX_SIDE -- cuts image tokens (and therefore latency)
    roughly in half vs the full 1100px frame. scale maps full-res pixel
    coords into the downscaled image."""
    h, w = frame_bgr.shape[:2]
    max_side = int(getattr(config, "VOICE_IMAGE_MAX_SIDE", 0) or 0)
    if max_side <= 0 or max(h, w) <= max_side:
        return frame_bgr, 1.0
    scale = max_side / float(max(h, w))
    resized = cv2.resize(
        frame_bgr, (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _resolve_second_object(frame_bgr, yolo_items, primary_bbox, primary_obj,
                           gaze_pixel):
    """Compare needs TWO targets. The primary is the gaze target; pick the
    second from the remaining YOLO segments: closest to the gaze first, CLIP
    matched against the DB, first hit with a DIFFERENT object id wins. Only
    the top few candidates are tried to bound the added CLIP latency.

    Returns (obj_dict, bbox) or (None, None).
    """
    from ..vision import clip_matcher

    if not yolo_items:
        return None, None

    def _centre_dist(item):
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (cx - gaze_pixel[0]) ** 2 + (cy - gaze_pixel[1]) ** 2

    primary_id = (primary_obj or {}).get("id")
    candidates = [
        it for it in yolo_items
        if primary_bbox is None or tuple(it["bbox"]) != tuple(primary_bbox)
    ]
    candidates.sort(key=_centre_dist)

    for item in candidates[:3]:
        try:
            crop = clip_matcher.prepare_query_crop(item, frame_bgr)
            obj, match_meta = clip_matcher.resolve_db_match(crop)
        except Exception as exc:
            print(f"[VOICE_PIPELINE][Compare] candidate match failed: {exc}")
            continue
        if obj is None or obj.get("id") == primary_id:
            continue
        print(
            f"[VOICE_PIPELINE][Compare] second object id={obj.get('id')} "
            f"score={match_meta.get('score', 0.0):.3f}"
        )
        return obj, item.get("bbox")
    return None, None


def _fill_compare_response(response, primary_obj, second_obj, answer):
    """Shape the response like the gesture Compare handler so Unity's
    CompareResultCard renders (name_a/name_b + compare_rows). Falls back to a
    placeholder row carrying the GPT answer when a proper pair isn't
    available."""
    from ..unity.object_action import _compare_rows_to_text
    from ..vision import object_db

    name_a = (primary_obj or {}).get("name") or "대상 1"
    name_b = (second_obj or {}).get("name") or "대상 2"

    compare_rows = None
    if primary_obj is not None and second_obj is not None:
        db = object_db.get_db()
        if db is not None:
            compare_rows = db.lookup_comparison(primary_obj["id"], second_obj["id"])
    if not compare_rows:
        compare_rows = [{
            "category": "",
            "value_a": answer or "비교 정보가 등록되어 있지 않습니다.",
            "value_b": "",
        }]

    id_a = (primary_obj or {}).get("id", "")
    id_b = (second_obj or {}).get("id", "")
    response["name"] = f"{name_a} vs {name_b}"
    response["object_id"] = "_vs_".join(sorted([id_a, id_b])) if id_a and id_b else id_a
    response["name_a"] = name_a
    response["name_b"] = name_b
    response["compare_rows"] = compare_rows
    response["result_search"] = _compare_rows_to_text(name_a, name_b, compare_rows)


def _resolve_question_end_gaze(live_gaze_norm, live_gaze_tracked,
                               gaze_trail_bl, gaze_viewport_bl, gaze_tracked,
                               frame_w, frame_h):
    """The single capture-moment gaze pixel, top-left origin. Preference:

      1. Live delay-aligned sample (state.latest_gaze_norm read together
         with the frame). Because that stream is delayed to match the adb
         frame content, this pixel and the frame describe the SAME instant --
         one instantaneous gaze point, no head-motion compensation.
      2. Last sample of the Unity gaze trail (real-time coords; can be
         ~GAZE_SCREEN_DELAY_S ahead of the frame -- fallback only).
      3. Listen-start snapshot (old senders / trail dropped).
      4. Frame centre (eye tracker not tracking).

    Returns (pixel, source_tag) so logs / payloads record which one won.
    """
    if live_gaze_tracked and live_gaze_norm is not None:
        try:
            nx = max(0.0, min(1.0, float(live_gaze_norm[0])))
            ny = max(0.0, min(1.0, float(live_gaze_norm[1])))  # already top-left
            px = int(round(nx * max(1, frame_w - 1)))
            py = int(round(ny * max(1, frame_h - 1)))
            return (px, py), "capture_moment"
        except (TypeError, ValueError, IndexError):
            pass
    if gaze_trail_bl:
        try:
            nx, ny_bl = gaze_trail_bl[-1]
            return _viewport_bl_to_pixel(nx, ny_bl, frame_w, frame_h), "trail_end"
        except (TypeError, ValueError, IndexError):
            pass
    if gaze_tracked and gaze_viewport_bl is not None:
        try:
            nx, ny_bl = float(gaze_viewport_bl[0]), float(gaze_viewport_bl[1])
            return _viewport_bl_to_pixel(nx, ny_bl, frame_w, frame_h), "listen_start"
        except (TypeError, ValueError):
            pass
    return (int(frame_w // 2), int(frame_h // 2)), "centre_fallback"


def _viewport_bl_to_pixel(nx, ny_bl, frame_w, frame_h):
    """Unity bottom-left viewport (0..1) -> cv2 top-left pixel coords."""
    nx = max(0.0, min(1.0, float(nx)))
    ny_bl = max(0.0, min(1.0, float(ny_bl)))
    px = int(round(nx * max(1, frame_w - 1)))
    py = int(round((1.0 - ny_bl) * max(1, frame_h - 1)))
    return px, py


def _build_gaze_info(gaze_pixel, gaze_tracked, gaze_source, frame_w, frame_h):
    """Text block injected into the prompt's gaze section (GazePointAR-style
    explicit coordinate injection)."""
    px, py = gaze_pixel
    if not gaze_tracked and gaze_source == "centre_fallback":
        return (
            "The eye tracker did NOT report a valid gaze sample for this "
            "query. Fall back to the centre of the image and to any pointing "
            "gesture (section 3) when resolving pronouns."
        )
    nx = px / max(1, frame_w - 1)
    ny = py / max(1, frame_h - 1)
    horiz = "left" if nx < 0.33 else ("right" if nx > 0.67 else "centre")
    vert = "top" if ny < 0.33 else ("bottom" if ny > 0.67 else "middle")
    return (
        f"The image is {frame_w}x{frame_h} pixels. When the question ended, "
        f"the user's gaze was at pixel (x={px}, y={py}) counting from the "
        f"TOP-LEFT corner -- i.e. the {vert}-{horiz} region "
        f"(normalized x={nx:.2f}, y={ny:.2f}). The object AT or CLOSEST TO "
        f"this pixel is the referent for any pronoun in the query."
    )


def _render_debug_canvas(frame_bgr, gaze_pixel, intent, name, answer,
                         target_bbox=None):
    """Right-pane debug overlay: gaze marker + DB target bbox + intent + answer."""
    canvas = frame_bgr.copy()
    if target_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in target_bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(canvas, tuple(gaze_pixel), 14, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(gaze_pixel), 3, (0, 255, 255), -1)
    cv2.putText(
        canvas, f"VOICE {intent}: {name}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, answer[:90],
        (20, canvas.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2, cv2.LINE_AA,
    )
    return canvas


def _publish_target_canvas(canvas):
    """Push the debug overlay to state so the main render loop shows it in
    the 'Target Result' pane, same as the gesture flow does."""
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
    from ..unity import network as network_mod

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
