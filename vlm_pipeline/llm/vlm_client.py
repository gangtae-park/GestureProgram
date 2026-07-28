"""OpenAI multimodal VLM client + on-disk persistence.

Handler modules call call_vlm_on_crop() with a (crop_bgr, prompt) pair, then
save_vlm_response() to write the result + crop image to vlm_outputs/.
"""
import base64
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

try:
    from openai import OpenAI
except Exception as _openai_exc:
    OpenAI = None
    print(f"[VLM][IMPORT][WARN] {_openai_exc}")

from .. import config


# Ask pipeline: gpt-4o via the Responses API with the web_search_preview tool.
# Bigger world knowledge than the mini models AND can fetch fresh info when the
# question is beyond both the DB and the model's training data. First-token
# ~500ms, ~40 tokens/sec streaming; web-search invocations add ~1-3s on top,
# but only when the model decides it needs external info.
ASK_STREAM_MODEL = "gpt-4o"

# Translate pipeline: gpt-4o-mini via plain Chat Completions streaming.
# Translation needs no world knowledge, so we prioritise raw speed here (first
# token ~250ms, ~70 tokens/sec).
TRANSLATE_STREAM_MODEL = "gpt-4o-mini"


_openai_client = None


def init_openai_client() -> bool:
    global _openai_client
    if OpenAI is None:
        print("[VLM][ERROR] openai package not installed. `pip install openai`")
        _openai_client = None
        return False
    api_key = os.environ.get(config.OPENAI_API_KEY_ENV)
    if not api_key:
        print(f"[VLM][ERROR] {config.OPENAI_API_KEY_ENV} env var not set.")
        _openai_client = None
        return False
    try:
        _openai_client = OpenAI(api_key=api_key, timeout=config.VLM_REQUEST_TIMEOUT_SEC)
        print(f"[VLM] OpenAI client ready, model={config.OPENAI_MODEL}")
        return True
    except Exception as exc:
        _openai_client = None
        print(f"[VLM][ERROR] OpenAI init failed: {exc}")
        return False






def warm_up_stream_client() -> bool:
    """Prime BOTH streaming paths (Ask via Responses API, Translate via Chat
    Completions) with tiny dummy requests so the FIRST real gesture doesn't
    eat the TLS handshake + cold-cache penalty. Runs from startup in a
    background thread.

    Returns True if AT LEAST ONE warm-up round-tripped. Failures per-model
    are logged and swallowed so a broken tool config on one path can't stop
    the other from warming.
    """
    if _openai_client is None:
        return False
    any_ok = False

    # Translate path: Chat Completions.
    try:
        t0 = time.perf_counter()
        stream = _openai_client.chat.completions.create(
            model=TRANSLATE_STREAM_MODEL,
            stream=True,
            max_completion_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        for _ in stream:
            pass
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[VLM][WARMUP] {TRANSLATE_STREAM_MODEL} (chat) handshake+ping in {elapsed_ms:.0f}ms")
        any_ok = True
    except Exception as exc:
        print(f"[VLM][WARMUP][WARN] {TRANSLATE_STREAM_MODEL}: {exc}")

    # Ask path: Responses API. No tools attached -- warm-up shouldn't invoke
    # web_search; we just want the connection primed and the Responses API
    # code path exercised so the SDK's internal caches are ready.
    try:
        t0 = time.perf_counter()
        stream = _openai_client.responses.create(
            model=ASK_STREAM_MODEL,
            stream=True,
            max_output_tokens=1,
            input="hi",
        )
        for _ in stream:
            pass
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[VLM][WARMUP] {ASK_STREAM_MODEL} (responses) handshake+ping in {elapsed_ms:.0f}ms")
        any_ok = True
    except Exception as exc:
        print(f"[VLM][WARMUP][WARN] {ASK_STREAM_MODEL}: {exc}")

    return any_ok


def _encode_image_to_data_uri(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


_REASONING_FROM_CONFIG = object()  # sentinel: "use config.VLM_REASONING_EFFORT"


def call_vlm_on_crop(crop_bgr: np.ndarray, prompt: str, model: str = None,
                     reasoning_effort=_REASONING_FROM_CONFIG):
    """Send crop + prompt to GPT, return parsed dict or None on hard error.

    - model:            override config.OPENAI_MODEL (e.g. the fast voice model).
    - reasoning_effort: override config.VLM_REASONING_EFFORT; pass None to
                        omit the param entirely (required for non-reasoning
                        models like gpt-4o, which reject it server-side).

    Return shapes:
      - {dict from JSON}             on a clean JSON response
      - {"raw": "<text>", ...}       if model produced text but it didn't parse
      - {"raw": "", ...}             if model produced nothing (token budget exhausted etc.)
      - None                         if the request itself failed
    """
    if _openai_client is None:
        return None
    if reasoning_effort is _REASONING_FROM_CONFIG:
        reasoning_effort = config.VLM_REASONING_EFFORT
    try:
        data_uri = _encode_image_to_data_uri(crop_bgr)

        request_kwargs = dict(
            model=model or config.OPENAI_MODEL,
            max_completion_tokens=config.VLM_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        )
        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort

        t0 = time.perf_counter()
        try:
            completion = _openai_client.chat.completions.create(**request_kwargs)
        except TypeError:
            request_kwargs.pop("reasoning_effort", None)
            completion = _openai_client.chat.completions.create(**request_kwargs)
        elapsed = time.perf_counter() - t0

        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(completion, "usage", None)
        usage_str = ""
        if usage is not None:
            try:
                usage_str = (
                    f" prompt_tokens={getattr(usage, 'prompt_tokens', '?')} "
                    f"completion_tokens={getattr(usage, 'completion_tokens', '?')} "
                    f"total={getattr(usage, 'total_tokens', '?')}"
                )
                details = getattr(usage, "completion_tokens_details", None)
                if details is not None:
                    rt = getattr(details, "reasoning_tokens", None)
                    if rt is not None:
                        usage_str += f" reasoning_tokens={rt}"
            except Exception:
                pass

        print(
            f"[VLM] {config.OPENAI_MODEL} responded in {elapsed*1000:.0f} ms "
            f"finish_reason={finish_reason}{usage_str}"
        )

        if not text:
            print(
                "[VLM][WARN] empty content. Likely token budget exhausted by reasoning. "
                "Increase VLM_MAX_OUTPUT_TOKENS or set VLM_REASONING_EFFORT='minimal'."
            )
            refusal = getattr(choice.message, "refusal", None)
            return {"raw": "", "finish_reason": finish_reason, "refusal": refusal}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text, "finish_reason": finish_reason}
    except Exception as exc:
        print(f"[VLM][ERROR] {exc}")
        return None


def call_vlm_on_crop_stream(crop_bgr: np.ndarray, prompt: str, on_delta) -> str:
    """Streaming vision Q&A over a crop + prompt, backed by the Responses API.

    Uses the ``web_search_preview`` tool so the model can fetch external info
    when the DB context + its training knowledge aren't enough (e.g. product
    specs, recall status, current pricing). Web-search hops add ~1-3s but the
    model only invokes the tool when it decides it needs to -- typical Q&A
    stays on the fast path.

    The prompt should ask for a PLAIN-TEXT answer (no JSON schema) because we
    push deltas as they arrive; buffering a whole JSON would defeat streaming.

    on_delta(chunk_text) fires synchronously for each text delta.
    Returns the concatenated full text ("" on failure).
    """
    if _openai_client is None:
        print("[VLM][STREAM][ERROR] OpenAI client not initialized.")
        return ""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    try:
        data_uri = _encode_image_to_data_uri(crop_bgr)
    except Exception as exc:
        print(f"[VLM][STREAM][ERROR] image encode failed: {exc}")
        return ""

    try:
        t0 = time.perf_counter()
        stream = _openai_client.responses.create(
            model=ASK_STREAM_MODEL,
            stream=True,
            max_output_tokens=800,
            tools=[{"type": "web_search_preview"}],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_uri},
                    ],
                }
            ],
        )
        parts = []
        first_ms = None
        web_search_calls = 0
        for event in stream:
            etype = getattr(event, "type", "") or ""
            # Text deltas -- the primary path we care about.
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if not delta:
                    continue
                if first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000
                    print(f"[VLM][STREAM] first token in {first_ms:.0f}ms")
                parts.append(delta)
                try:
                    on_delta(delta)
                except Exception as cb_exc:
                    print(f"[VLM][STREAM][WARN] on_delta raised: {cb_exc}")
            # Web search invocation logs -- useful for judging if the model is
            # actually consulting external sources or answering from priors.
            elif etype in ("response.web_search_call.in_progress",
                           "response.web_search_call.searching"):
                web_search_calls += 1
                print(f"[VLM][STREAM] web_search invoked (#{web_search_calls})")
            elif etype == "response.error":
                err = getattr(event, "error", None)
                print(f"[VLM][STREAM][ERROR] stream error event: {err!r}")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        full = "".join(parts)
        print(
            f"[VLM][STREAM] done in {elapsed_ms:.0f}ms "
            f"first={first_ms:.0f}ms len={len(full)} web_search={web_search_calls}"
        )
        return full
    except Exception as exc:
        print(f"[VLM][STREAM][ERROR] {exc}")
        return ""


def save_vlm_response(
    response: dict,
    gesture_name: str,
    target_meta: dict,
    crop_bgr: np.ndarray,
    prompt: str,
):
    """Write the response + crop image to disk and return the on-disk payload dict
    (so the same object can be forwarded to Unity)."""
    os.makedirs(config.VLM_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_gesture = gesture_name.replace("/", "_").replace(" ", "_")
    base_name = f"{timestamp}_{safe_gesture}"

    payload = {
        "timestamp": timestamp,
        "gesture": gesture_name,
        "model": config.OPENAI_MODEL,
        "prompt": prompt,
        "target_meta": target_meta,
        "response": response,
    }

    json_path = os.path.join(config.VLM_OUTPUT_DIR, base_name + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    crop_path = os.path.join(config.VLM_OUTPUT_DIR, base_name + ".png")
    try:
        cv2.imwrite(crop_path, crop_bgr)
    except Exception as exc:
        print(f"[VLM][WARN] failed to save crop image: {exc}")

    print(f"[VLM] Saved -> {json_path}")
    if isinstance(response, dict):
        for k, v in response.items():
            print(f"        {k}: {v}")

    return payload
