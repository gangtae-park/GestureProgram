"""Runtime accessor for the gaze calibration model.

`calibration.py` learns two quadratic ridge regressions from the same set of
samples:

  - forward  f : gaze_dir(3D unit vec) -> norm_xy(ADB-frame 2D in [0,1])
  - inverse  g : norm_xy(2D)            -> gaze_dir(3D unit vec, z>=0)

Both weight matrices live side-by-side in `calibration_ridge_model.json`. This
module loads that file once and exposes:

  - forward(gaze_dir_xyz)  -> (norm_x, norm_y)
  - inverse(norm_x, norm_y) -> (gaze_dir_x, gaze_dir_y, gaze_dir_z)

The inverse reconstructs z under the assumption that the user is facing the
display (z > 0 across the calibration grid; samples in the JSON show z in
0.94..1.0). If learned x^2 + y^2 ever reaches >= 1 because of extrapolation,
we clamp to the unit-sphere boundary so the caller still gets a valid direction.
"""

import json
import math
import os
import threading
from typing import Optional, Tuple

import numpy as np

from . import config


_lock = threading.Lock()
_model = None  # type: Optional[dict]
_loaded_path = None  # type: Optional[str]


def _resolve_default_path() -> str:
    # First try the absolute path baked into config; fall back to the relative
    # working-directory variant if it doesn't exist (e.g. dev launches).
    abs_path = getattr(config, "RIDGE_MODEL_ABS_PATH", None)
    if abs_path and os.path.isfile(abs_path):
        return abs_path
    return config.RIDGE_MODEL_PATH


def load(path: Optional[str] = None, force: bool = False) -> bool:
    """Load the calibration JSON into the module-level cache. Returns True on
    success."""
    global _model, _loaded_path

    target_path = path or _resolve_default_path()
    with _lock:
        if not force and _model is not None and _loaded_path == target_path:
            return True

        if not os.path.isfile(target_path):
            print(f"[GAZECAL][ERROR] calibration file not found: {target_path}")
            _model = None
            _loaded_path = None
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as exc:
            print(f"[GAZECAL][ERROR] failed to parse {target_path}: {exc}")
            _model = None
            _loaded_path = None
            return False

        # Required keys for forward/inverse usage.
        forward_w = np.asarray(doc.get("weights") or [], dtype=np.float64)
        inverse_w = np.asarray(doc.get("inverse_weights") or [], dtype=np.float64)

        if forward_w.size == 0 or inverse_w.size == 0:
            print(
                f"[GAZECAL][ERROR] calibration JSON missing weights/inverse_weights "
                f"forward_shape={forward_w.shape} inverse_shape={inverse_w.shape}; "
                "re-run calibration.py to regenerate."
            )
            _model = None
            _loaded_path = None
            return False

        _model = {
            "forward_weights": forward_w,         # shape (10, 2)
            "inverse_weights": inverse_w,         # shape (6, 2)
            "forward_mse": float(doc.get("training_mse", -1.0)),
            "inverse_mse": float(doc.get("inverse_training_mse", -1.0)),
            "raw": doc,
        }
        _loaded_path = target_path
        print(
            f"[GAZECAL] loaded {target_path} "
            f"forward_mse={_model['forward_mse']:.2e} "
            f"inverse_mse={_model['inverse_mse']:.2e}"
        )
        return True


def is_loaded() -> bool:
    return _model is not None


def _ensure_loaded() -> bool:
    if _model is not None:
        return True
    return load()


def _forward_features(gx: float, gy: float, gz: float) -> np.ndarray:
    return np.array([
        1.0,
        gx,
        gy,
        gz,
        gx * gx,
        gy * gy,
        gz * gz,
        gx * gy,
        gx * gz,
        gy * gz,
    ], dtype=np.float64)


def _inverse_features(nx: float, ny: float) -> np.ndarray:
    return np.array([
        1.0,
        nx,
        ny,
        nx * nx,
        ny * ny,
        nx * ny,
    ], dtype=np.float64)


def forward(gaze_dir_xyz: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
    """Map a unit-length head-space gaze direction to (norm_x, norm_y) in [0,1].
    Returns None if the calibration model isn't available."""
    if not _ensure_loaded():
        return None
    gx, gy, gz = float(gaze_dir_xyz[0]), float(gaze_dir_xyz[1]), float(gaze_dir_xyz[2])
    feats = _forward_features(gx, gy, gz)
    out = feats @ _model["forward_weights"]
    return float(out[0]), float(out[1])


def inverse(norm_x: float, norm_y: float) -> Optional[Tuple[float, float, float]]:
    """Map an ADB-frame normalised point to a unit-length head-space gaze
    direction. Returns None if the calibration model isn't available.

    z is reconstructed as sqrt(max(0, 1 - x^2 - y^2)); if the learned (x,y)
    lies outside the unit disc (extrapolation), we project it back onto the
    boundary and set z=0 instead of returning NaN.
    """
    if not _ensure_loaded():
        return None
    feats = _inverse_features(float(norm_x), float(norm_y))
    out = feats @ _model["inverse_weights"]
    gx = float(out[0])
    gy = float(out[1])
    r_sq = gx * gx + gy * gy
    if r_sq >= 1.0:
        # Outside the unit disc -> project to boundary, z=0.
        r = math.sqrt(r_sq)
        gx /= r
        gy /= r
        gz = 0.0
    else:
        gz = math.sqrt(1.0 - r_sq)
    return gx, gy, gz
