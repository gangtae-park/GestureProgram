"""Head-rotation compensation for the gesture gaze trail.

Motivation: targeting used to capture the frame at gesture END, but the gaze
trail was accumulated while the head could be moving -- points collected
earlier in the gesture land on the wrong pixels of that frame. The new scheme
freezes the frame at gesture START instead, and re-projects every subsequent
gaze sample into that frame's camera pose:

    ridge(norm) --inv pinhole--> camera ray --R_sample--> world ray
              --R_start^T--> start-frame ray --pinhole--> start-frame norm

Unity sends the head (camera) world rotation with every GAZE packet, so both
rotations are known per sample. Translation is ignored (rotation-only), which
is accurate for targets much farther than the head moves within a gesture.

The pinhole intrinsics of the 1100x1000 mirrored stream are derived from the
9-dot calibration geometry: the dots sit on a head-locked grid 1 m ahead with
known tangent coordinates, and their fixed screen-norm targets give
fx=729, fy=-665 (image y grows down while camera y is up), cx=575, cy=501.
Same values as compare_bbox_analysis.py's --anchor-intrinsics.
"""
import numpy as np

from .. import config


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Unity xyzw quaternion -> 3x3 rotation (camera-local -> world)."""
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def norm_to_ray(norm_x: float, norm_y: float) -> np.ndarray:
    """Normalized stream coords -> unit camera-local ray (x right, y up, z fwd)."""
    px = norm_x * config.STREAM_W
    py = norm_y * config.STREAM_H
    v = np.array(
        [
            (px - config.HEADCOMP_CX) / config.HEADCOMP_FX,
            (py - config.HEADCOMP_CY) / config.HEADCOMP_FY,
            1.0,
        ],
        dtype=np.float64,
    )
    return v / np.linalg.norm(v)


def ray_to_norm(v: np.ndarray):
    """Camera-local ray -> normalized stream coords, or None when behind."""
    if v[2] <= 1e-6:
        return None
    px = config.HEADCOMP_FX * v[0] / v[2] + config.HEADCOMP_CX
    py = config.HEADCOMP_FY * v[1] / v[2] + config.HEADCOMP_CY
    return px / config.STREAM_W, py / config.STREAM_H


def reproject_norm(norm, r_sample: np.ndarray, r_ref: np.ndarray):
    """Move a ridge-mapped screen point from its own camera pose into the
    reference (gesture-start) camera pose. Returns norm coords or None."""
    ray_world = r_sample @ norm_to_ray(norm[0], norm[1])
    return ray_to_norm(r_ref.T @ ray_world)
