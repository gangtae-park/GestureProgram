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

from . import config


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


def translate_texts_to_korean(texts: list) -> list:
    """Translate a list of source-language strings to Korean in a single call.

    Returns a list of Korean strings, one per input, preserving order. On any
    failure each entry falls back to an empty string so the caller can still
    log the original side-by-side.
    """
    if not texts:
        return []
    if _openai_client is None:
        print("[TRANSLATE][ERROR] OpenAI client not initialized.")
        return ["" for _ in texts]

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    system_msg = (
        "You are a translation engine. Translate every input line into natural Korean. "
        "Return ONLY a JSON object of the form "
        '{"translations": ["<line 1 ko>", "<line 2 ko>", ...]} '
        "with exactly the same number of entries as the input, in the same order. "
        "Preserve proper nouns when no Korean equivalent exists."
    )
    user_msg = f"Translate the following lines to Korean:\n{numbered}"

    try:
        # gpt-5 is a reasoning model: max_completion_tokens covers BOTH reasoning
        # and output tokens. With the global VLM_REASONING_EFFORT ("low") and a
        # tight token budget, reasoning consumes the whole budget and the output
        # comes back empty. Translation needs essentially no reasoning, so force
        # "minimal" here and give the output a generous floor.
        request_kwargs = dict(
            model=config.OPENAI_MODEL,
            max_completion_tokens=max(1024, 200 * len(texts)),
            response_format={"type": "json_object"},
            reasoning_effort="minimal",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        t0 = time.perf_counter()
        try:
            completion = _openai_client.chat.completions.create(**request_kwargs)
        except TypeError:
            # Older / non-reasoning models don't accept reasoning_effort.
            request_kwargs.pop("reasoning_effort", None)
            completion = _openai_client.chat.completions.create(**request_kwargs)
        elapsed = time.perf_counter() - t0

        choice = completion.choices[0]
        raw = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(completion, "usage", None)
        usage_str = ""
        if usage is not None:
            try:
                usage_str = (
                    f" prompt={getattr(usage, 'prompt_tokens', '?')} "
                    f"completion={getattr(usage, 'completion_tokens', '?')} "
                    f"total={getattr(usage, 'total_tokens', '?')}"
                )
                details = getattr(usage, "completion_tokens_details", None)
                if details is not None:
                    rt = getattr(details, "reasoning_tokens", None)
                    if rt is not None:
                        usage_str += f" reasoning={rt}"
            except Exception:
                pass
        print(
            f"[TRANSLATE] {config.OPENAI_MODEL} responded in {elapsed*1000:.0f} ms "
            f"finish_reason={finish_reason}{usage_str}"
        )

        if not raw:
            print(
                "[TRANSLATE][WARN] empty content. "
                "If finish_reason=length, raise max_completion_tokens or lower reasoning_effort further."
            )
            return ["" for _ in texts]

        parsed = json.loads(raw)
        out = parsed.get("translations", [])
        if not isinstance(out, list):
            print(f"[TRANSLATE][WARN] unexpected response shape: {parsed!r}")
            return ["" for _ in texts]
        if len(out) != len(texts):
            print(
                f"[TRANSLATE][WARN] count mismatch: got {len(out)}, expected {len(texts)}. "
                "Padding / truncating."
            )
            if len(out) < len(texts):
                out = list(out) + [""] * (len(texts) - len(out))
            else:
                out = out[: len(texts)]
        return [str(x) if x is not None else "" for x in out]
    except json.JSONDecodeError as exc:
        print(f"[TRANSLATE][ERROR] JSON decode failed: {exc}; raw={raw!r}")
        return ["" for _ in texts]
    except Exception as exc:
        print(f"[TRANSLATE][ERROR] {exc}")
        return ["" for _ in texts]


def transcribe_audio_bytes(audio_bytes: bytes, file_format: str = "wav") -> str:
    """Send raw audio bytes to OpenAI Whisper. Returns transcribed text or '' on failure."""
    if _openai_client is None:
        print("[VOICE][ERROR] OpenAI client not initialized.")
        return ""
    if not audio_bytes:
        return ""

    try:
        from io import BytesIO
        buf = BytesIO(audio_bytes)
        # OpenAI SDK needs the file-like to have a name attribute that hints the format.
        buf.name = f"recording.{file_format}"

        t0 = time.perf_counter()
        result = _openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=buf,
        )
        elapsed = time.perf_counter() - t0
        text = getattr(result, "text", "") or ""
        print(f"[VOICE] whisper ok in {elapsed*1000:.0f}ms, {len(audio_bytes)} bytes -> {text!r}")
        return text
    except Exception as e:
        print(f"[VOICE][ERROR] Whisper failed: {e}")
        return ""


def _encode_image_to_data_uri(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def call_vlm_on_crop(crop_bgr: np.ndarray, prompt: str):
    """Send crop + prompt to GPT, return parsed dict or None on hard error.

    Return shapes:
      - {dict from JSON}             on a clean JSON response
      - {"raw": "<text>", ...}       if model produced text but it didn't parse
      - {"raw": "", ...}             if model produced nothing (token budget exhausted etc.)
      - None                         if the request itself failed
    """
    if _openai_client is None:
        return None
    try:
        data_uri = _encode_image_to_data_uri(crop_bgr)

        request_kwargs = dict(
            model=config.OPENAI_MODEL,
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
        if config.VLM_REASONING_EFFORT is not None:
            request_kwargs["reasoning_effort"] = config.VLM_REASONING_EFFORT

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
