"""CLIP image embedding + cosine-similarity matching against object_db.

Public API:
  load_clip_model()         -> bool      one-time init
  embed_image(bgr)          -> np.ndarray (D,) unit-norm float32
  match_against_db(bgr, db) -> dict      best match info + score, or None
  prepare_query_crop(seg, frame) -> np.ndarray   build the BGR crop we feed CLIP

The CLIP backbone is loaded on the same device as the rest of the pipeline
(MPS on Apple Silicon, CUDA if available, else CPU).
"""
import cv2
import numpy as np

try:
    import torch
except Exception as _torch_exc:
    torch = None
    print(f"[CLIP][IMPORT][WARN] torch missing: {_torch_exc}")

try:
    import open_clip
except Exception as _clip_exc:
    open_clip = None
    print(f"[CLIP][IMPORT][WARN] open_clip missing: {_clip_exc}")

from . import config


_model = None
_preprocess = None
_tokenizer = None
_device = "cpu"


def _resolve_device() -> str:
    if torch is None:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_clip_model() -> bool:
    """Build the CLIP model + preprocessor once. Returns True on success."""
    global _model, _preprocess, _tokenizer, _device

    if open_clip is None or torch is None:
        print("[CLIP][ERROR] open_clip / torch not importable; CLIP matching disabled.")
        return False

    _device = _resolve_device()
    try:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME,
            pretrained=config.CLIP_PRETRAINED,
        )
        _model = _model.to(_device)
        _model.eval()
        _tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
        print(
            f"[CLIP] Loaded {config.CLIP_MODEL_NAME}/{config.CLIP_PRETRAINED} "
            f"on device={_device}"
        )
        return True
    except Exception as exc:
        _model = None
        print(f"[CLIP][ERROR] {exc}")
        return False


def is_ready() -> bool:
    return _model is not None


def embed_image(bgr_image: np.ndarray) -> np.ndarray:
    """Return a unit-norm CLIP image embedding for a single BGR frame/crop."""
    if _model is None or _preprocess is None:
        raise RuntimeError("CLIP model not loaded; call load_clip_model() first.")
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("empty image")

    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    from PIL import Image  # lazy: open_clip already pulls Pillow in
    pil = Image.fromarray(rgb)
    tensor = _preprocess(pil).unsqueeze(0).to(_device)

    with torch.inference_mode():
        feats = _model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].detach().cpu().numpy().astype(np.float32)


def prepare_query_crop(target_segment: dict, frame_bgr: np.ndarray) -> np.ndarray:
    """Build the BGR crop we feed CLIP.

    If config.CLIP_USE_MASKED_CROP and the YOLO segment has a mask, the
    background pixels inside the bbox are replaced by mid-grey so CLIP focuses
    on the object. Otherwise we return a plain bbox crop.
    """
    x1, y1, x2, y2 = target_segment["bbox"]
    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate bbox: {target_segment['bbox']}")

    crop = frame_bgr[y1:y2, x1:x2].copy()

    if config.CLIP_USE_MASKED_CROP and target_segment.get("mask_bool") is not None:
        mask_full = target_segment["mask_bool"]
        if mask_full.shape == frame_bgr.shape[:2]:
            mask_crop = mask_full[y1:y2, x1:x2]
            grey = np.full_like(crop, 127, dtype=np.uint8)
            crop = np.where(mask_crop[..., None], crop, grey)
    return crop


def match_against_db(query_emb: np.ndarray, db) -> dict:
    """Cosine similarity match against an ObjectDB. Returns:

      { "object": <object info dict>,
        "score": float,
        "ranking": [(object_id, score), ...] }

    or None if the DB is empty.
    """
    if db is None or db.embedding_matrix is None or len(db.embedding_matrix) == 0:
        return None

    sims = db.embedding_matrix @ query_emb  # (N,)
    # Take the per-object max across that object's reference images.
    per_object_best = {}
    for sim, obj_id in zip(sims, db.embedding_object_ids):
        prev = per_object_best.get(obj_id, -1.0)
        if sim > prev:
            per_object_best[obj_id] = float(sim)

    ranking = sorted(per_object_best.items(), key=lambda kv: kv[1], reverse=True)
    if not ranking:
        return None

    best_id, best_score = ranking[0]
    obj_info = db.get_object(best_id)
    return {
        "object": obj_info,
        "score": best_score,
        "ranking": ranking,
    }
