"""Voice-command intent classifier.

Maps a user transcript to one of seven canonical referent gestures. Ask is
the open-ended fallback (per the study's design: "let Ask absorb anything the
six DB-backed intents don't cleanly claim").

Text-only call to gpt-5 so it stays fast and cheap; the image doesn't help
distinguish "translate this" from "how much is this?" and both intents will
run the same visual CLIP pipeline downstream anyway.
"""
import json
import time
from typing import Tuple

from . import config
from .vlm_client import _openai_client  # reuse the shared OpenAI client


CANONICAL_INTENTS = (
    "Search",
    "Ask",
    "Translate",
    "Compare",
    "Anchor",
    "Save",
    "Capture",
)


def classify(transcript: str) -> Tuple[str, str, str, dict]:
    """Classify the transcript into one of CANONICAL_INTENTS.

    Returns (intent, confidence, rationale, extras) where `extras` is a
    dict of per-intent auxiliary fields the classifier chose to surface.
    Today the only extra is `note_content` when intent="Save" (the note
    body extracted from phrases like "~~라고 노트 저장해줘"), but the shape
    is open for future intents.

    On any failure (empty transcript, missing OpenAI client, malformed
    response) defaults to ("Ask", "low", "<reason>", {}). Ask is the safe
    fallback because its downstream pipeline can handle arbitrary
    open-ended questions.
    """
    if not transcript or not transcript.strip():
        return "Ask", "low", "empty transcript", {}

    if _openai_client is None:
        print("[VOICE_INTENT][WARN] OpenAI client not initialised; defaulting to Ask.")
        return "Ask", "low", "openai_unavailable", {}

    prompt = config.VOICE_INTENT_PROMPT.replace("{transcript}", transcript)

    try:
        request_kwargs = dict(
            model=config.OPENAI_MODEL,
            # Classification needs almost no reasoning + one short JSON blob.
            # Keeping the ceiling tight makes this a sub-second call so we
            # don't add noticeable latency on top of the downstream YOLO/CLIP.
            max_completion_tokens=256,
            response_format={"type": "json_object"},
            reasoning_effort="minimal",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        t0 = time.perf_counter()
        try:
            completion = _openai_client.chat.completions.create(**request_kwargs)
        except TypeError:
            # Older / non-reasoning models don't accept reasoning_effort.
            request_kwargs.pop("reasoning_effort", None)
            completion = _openai_client.chat.completions.create(**request_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw = (completion.choices[0].message.content or "").strip()
        if not raw:
            print(f"[VOICE_INTENT][WARN] empty response in {elapsed_ms:.0f}ms; defaulting to Ask.")
            return "Ask", "low", "empty_response", {}

        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[VOICE_INTENT][ERROR] JSON decode failed: {exc}; defaulting to Ask.")
        return "Ask", "low", f"json_error: {exc}", {}
    except Exception as exc:
        print(f"[VOICE_INTENT][ERROR] {exc}; defaulting to Ask.")
        return "Ask", "low", f"exception: {exc}", {}

    intent = str(parsed.get("intent") or "").strip()
    confidence = str(parsed.get("confidence") or "low").strip().lower()
    rationale = str(parsed.get("rationale") or "").strip()

    # If the classifier hallucinates a label outside our set (or omits it),
    # snap to Ask -- it's the fallback bucket by design.
    if intent not in CANONICAL_INTENTS:
        print(
            f"[VOICE_INTENT][WARN] unrecognised intent {intent!r} "
            f"(rationale={rationale!r}); coercing to Ask."
        )
        intent = "Ask"
        confidence = "low"

    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    extras: dict = {}
    # Save-specific: capture the note body so voice_pipeline can inject it
    # into the outbound VLM_RESULT and Unity's NoteManager commits a
    # StickyNote directly, skipping the manual SaveNoteCard input UI.
    if intent == "Save":
        note_content = str(parsed.get("note_content") or "").strip()
        if note_content:
            extras["note_content"] = note_content

    print(
        f"[VOICE_INTENT] transcript={transcript!r} -> intent={intent!r} "
        f"confidence={confidence} elapsed_ms={elapsed_ms:.0f} rationale={rationale!r} "
        f"extras={extras!r}"
    )
    return intent, confidence, rationale, extras
