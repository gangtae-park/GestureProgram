"""Compute the (gaze_dir, depth_meters) anchor for a single gesture target.

Used by the per-gesture handlers (Search/Ask/Translate/Anchor/Save/Capture) so
the Unity ResultCardSpawner can place the resulting card at the same world
position as the actual object instead of the legacy fixed 1.2 m fallback.

  - gaze_dir comes from running the inverse gaze calibration on the target's
    YOLO bbox centre (normalised ADB-frame coords). Head-space unit vector.
  - depth_meters comes from Depth Anything V2 — median over the mask if
    present, otherwise median over the bbox.

Returns a dict with the values + a tiny diagnostic blob; callers usually splat
the result into the response payload they're already building.
"""

from typing import Optional

import numpy as np

from . import depth, gaze_calibration


def compute(frame_bgr: Optional[np.ndarray],
            bbox_xyxy,
            mask_bool: Optional[np.ndarray] = None) -> dict:
    """Build the anchor block. Always returns a dict, even on failure — fields
    default to zeros so the JSON-serialisable shape is stable for Unity."""
    out = {
        "gaze_dir_x": 0.0,
        "gaze_dir_y": 0.0,
        "gaze_dir_z": 0.0,
        "depth_meters": 0.0,
        "depth_source": "none",
        "norm_x": 0.0,
        "norm_y": 0.0,
        "anchor_ok": False,
    }

    if frame_bgr is None or bbox_xyxy is None or len(bbox_xyxy) < 4:
        return out

    h, w = frame_bgr.shape[:2]
    if w <= 0 or h <= 0:
        return out

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    norm_x = float(np.clip(cx / max(1, w), 0.0, 1.0))
    norm_y = float(np.clip(cy / max(1, h), 0.0, 1.0))
    out["norm_x"] = norm_x
    out["norm_y"] = norm_y

    # ---- Inverse gaze calibration ----
    if gaze_calibration.is_loaded() or gaze_calibration.load():
        inv = gaze_calibration.inverse(norm_x, norm_y)
        if inv is not None:
            out["gaze_dir_x"], out["gaze_dir_y"], out["gaze_dir_z"] = inv

    # ---- Depth (single forward pass; depth module caches model load) ----
    depth_map = depth.estimate(frame_bgr)
    if depth_map is not None:
        if mask_bool is not None:
            d = depth.median_depth_in_mask(depth_map, mask_bool)
            if d is not None:
                out["depth_meters"] = float(d)
                out["depth_source"] = "mask"
        if out["depth_source"] == "none":
            d = depth.median_depth_in_bbox(depth_map, [x1, y1, x2, y2])
            if d is not None:
                out["depth_meters"] = float(d)
                out["depth_source"] = "bbox"

    out["anchor_ok"] = out["depth_meters"] > 0.0
    return out


def merge_into_response(response: dict, anchor: dict) -> dict:
    """Convenience: copy anchor fields into a VlmResponse dict, leaving other
    keys intact. Returns the same response for chaining."""
    if response is None:
        return response
    for key in ("gaze_dir_x", "gaze_dir_y", "gaze_dir_z",
                "depth_meters", "depth_source"):
        response[key] = anchor.get(key, response.get(key, 0.0))
    return response
