"""Ridge regression model that maps a 3D gaze direction to normalized screen coords.

Trained by calibration.py and saved to calibration_ridge_model.json.
"""
import json

import numpy as np

from . import config


# Module-level state. Populated by load_ridge_model() at startup.
_model_weights = None


def build_feature_vector(x: float, y: float, z: float) -> np.ndarray:
    """Polynomial expansion of (gx, gy, gz) -- must match calibration.py."""
    return np.array(
        [1.0, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z],
        dtype=np.float64,
    )


def load_ridge_model() -> bool:
    """Read JSON calibration weights from disk. Returns True on success."""
    global _model_weights
    try:
        with open(config.RIDGE_MODEL_PATH, "r") as f:
            data = json.load(f)
            _model_weights = np.array(data["weights"], dtype=np.float64)
            print(f"[MODEL] Loaded ridge model: {config.RIDGE_MODEL_PATH}")
            return True
    except Exception as exc:
        _model_weights = None
        print(f"[MODEL][ERROR] Failed to load {config.RIDGE_MODEL_PATH}: {exc}")
        return False


def map_gaze_dir_to_norm(gx: float, gy: float, gz: float):
    """Apply the ridge model to a 3D gaze direction. Returns (norm_x, norm_y) or None."""
    if _model_weights is None:
        return None
    feat = build_feature_vector(gx, gy, gz)
    pred = feat @ _model_weights
    return float(pred[0]), float(pred[1])


def is_loaded() -> bool:
    return _model_weights is not None
