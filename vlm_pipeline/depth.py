"""Monocular metric depth estimation via Depth Anything V2.

Wraps the HuggingFace `transformers` checkpoint
  depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
which returns per-pixel depth in metres for indoor scenes (the study setup).

The module imports torch/transformers lazily so the rest of the pipeline keeps
working in environments where the deps aren't installed yet — `is_ready()`
reflects whether depth estimation is actually usable.

Typical usage from a handler:

    depth_map = depth.estimate(frame_bgr)
    if depth_map is None:
        return None
    d = depth.median_depth_in_mask(depth_map, mask_bool)  # metres
"""

import threading
import time
from typing import Optional

import numpy as np

# Lazy state -- populated on first successful load.
_lock = threading.Lock()
_model = None
_processor = None
_torch = None
_device = None
_dtype = None
_initialised = False
_ready = False
_model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


def configure(model_id: Optional[str] = None) -> None:
    """Change the checkpoint id (must be called before the first estimate())."""
    global _model_id
    if model_id:
        _model_id = model_id


def is_ready() -> bool:
    return _ready


def _ensure_loaded() -> bool:
    global _model, _processor, _torch, _device, _dtype, _initialised, _ready
    if _ready:
        return True
    with _lock:
        if _ready:
            return True
        if _initialised:
            return False  # we tried earlier and failed; don't keep retrying every frame
        _initialised = True

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            print(
                "[DEPTH][WARN] torch / transformers unavailable "
                f"({type(exc).__name__}: {exc}). Install with "
                "`pip install transformers torch pillow` to enable Depth "
                "Anything V2; pipeline will fall back to no-depth path."
            )
            return False

        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float32  # mps doesn't love fp16 for this model size
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        try:
            t0 = time.perf_counter()
            processor = AutoImageProcessor.from_pretrained(_model_id)
            model = AutoModelForDepthEstimation.from_pretrained(_model_id, torch_dtype=dtype)
            model.to(device)
            model.eval()
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"[DEPTH][ERROR] failed to load {_model_id}: {exc}")
            return False

        _torch = torch
        _model = model
        _processor = processor
        _device = device
        _dtype = dtype
        _ready = True
        print(f"[DEPTH] loaded {_model_id} on {device} dtype={dtype} in {elapsed*1000:.0f} ms")
        return True


def estimate(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Run Depth Anything V2 on a BGR uint8 frame. Returns an (H, W) float32
    metric depth map (metres), or None if the model isn't available / fails."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    if not _ensure_loaded():
        return None

    try:
        # Lazy import again (cheap; cached) so we don't reference cv2 just to flip channels.
        import cv2
    except Exception:
        cv2 = None

    if cv2 is not None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame_bgr[..., ::-1].copy()

    h, w = frame_rgb.shape[:2]

    try:
        from PIL import Image
        pil_image = Image.fromarray(frame_rgb)
        inputs = _processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(_device, dtype=_dtype if v.dtype.is_floating_point else None) for k, v in inputs.items()}

        with _torch.no_grad():
            outputs = _model(**inputs)
            predicted = outputs.predicted_depth  # (1, h_pred, w_pred)

        depth = _torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).squeeze(0)
        depth_np = depth.detach().to(_torch.float32).cpu().numpy()
        return depth_np
    except Exception as exc:
        print(f"[DEPTH][ERROR] inference failed: {exc}")
        return None


def median_depth_in_mask(depth_map: np.ndarray, mask_bool: np.ndarray) -> Optional[float]:
    """Median depth (in metres) over the True pixels of `mask_bool`. Returns
    None if mask is empty or sizes mismatch."""
    if depth_map is None or mask_bool is None:
        return None
    if depth_map.shape[:2] != mask_bool.shape[:2]:
        # Best-effort: resize mask to depth shape using nearest neighbour.
        try:
            import cv2
            mask_resized = cv2.resize(
                mask_bool.astype(np.uint8),
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        except Exception:
            return None
        mask_bool = mask_resized

    values = depth_map[mask_bool]
    if values.size == 0:
        return None
    return float(np.median(values))


def median_depth_in_bbox(depth_map: np.ndarray, bbox_xyxy) -> Optional[float]:
    """Median depth (metres) over the bbox area. Fallback when no segmentation
    mask is available."""
    if depth_map is None or bbox_xyxy is None or len(bbox_xyxy) < 4:
        return None
    x1, y1, x2, y2 = bbox_xyxy[:4]
    h, w = depth_map.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth_map[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return float(np.median(patch))
