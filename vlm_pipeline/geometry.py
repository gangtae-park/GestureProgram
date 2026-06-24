"""Pure geometry helpers: projection, bbox math, IoU, gaze overlap."""
import numpy as np

from . import config


def project_norm_points(norm_points, frame_shape):
    """Convert normalized (0..1, 0..1) gaze points to integer pixel coords."""
    h, w = frame_shape[:2]
    return [
        (
            int(np.clip(nx * w, 0, w - 1)),
            int(np.clip(ny * h, 0, h - 1)),
        )
        for nx, ny in norm_points
    ]


def compute_gaze_bbox(pixel_points, frame_shape):
    """Tight axis-aligned bbox around the gesture-window gaze trail (with padding)."""
    if len(pixel_points) < config.MIN_GAZE_POINTS_FOR_TARGET:
        return None
    h, w = frame_shape[:2]
    xs = [p[0] for p in pixel_points]
    ys = [p[1] for p in pixel_points]
    pad = config.GAZE_BBOX_PADDING
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w - 1, max(xs) + pad)
    y2 = min(h - 1, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def gaze_overlap_ratio(gaze_bbox, obj_bbox):
    """Fraction of the gaze bbox that lies inside obj_bbox."""
    gx1, gy1, gx2, gy2 = gaze_bbox
    ox1, oy1, ox2, oy2 = obj_bbox
    ix1, iy1 = max(gx1, ox1), max(gy1, oy1)
    ix2, iy2 = min(gx2, ox2), min(gy2, oy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    gaze_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    return inter / gaze_area if gaze_area > 0 else 0.0


def mask_to_bbox(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox_for_crop(bbox, frame_shape, pad_ratio: float):
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w - 1, x2 + pad_x)
    ny2 = min(h - 1, y2 + pad_y)
    return nx1, ny1, nx2, ny2


def pick_highest_iou(gaze_bbox, candidates):
    """Return (best_index, best_iou) for the candidate dict (with 'bbox') that
    maximises IoU against gaze_bbox. Candidates with zero overlap are skipped;
    if nothing overlaps, returns (-1, 0.0).
    """
    best_idx, best_iou = -1, 0.0
    for i, c in enumerate(candidates):
        iou = bbox_iou(gaze_bbox, c["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx, best_iou


def inside_ratio(inner_bbox, outer_bbox) -> float:
    """Fraction of inner_bbox that lies inside outer_bbox.

    Useful for OCR-vs-gaze checks: an OCR word is typically much smaller than
    the gaze bbox, so inside_ratio(word, gaze) > 0.5 means most of the word is
    being looked at.
    """
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    cx1, cy1 = max(ix1, ox1), max(iy1, oy1)
    cx2, cy2 = min(ix2, ox2), min(iy2, oy2)
    cw, ch = max(0, cx2 - cx1), max(0, cy2 - cy1)
    inter = cw * ch
    inner_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / inner_area if inner_area > 0 else 0.0


def pick_best_overlap(gaze_bbox, candidates):
    """Score every candidate dict (with 'bbox') against gaze_bbox, return
    (best_index, best_overlap, best_iou). Candidates with overlap below
    config.TARGET_MIN_OVERLAP are skipped entirely.
    """
    best_idx = -1
    best_score = -1.0
    best_overlap = 0.0
    best_iou = 0.0
    for i, c in enumerate(candidates):
        overlap = gaze_overlap_ratio(gaze_bbox, c["bbox"])
        if overlap < config.TARGET_MIN_OVERLAP:
            continue
        iou = bbox_iou(gaze_bbox, c["bbox"])
        score = (1 - config.TARGET_SCORE_IOU_WEIGHT) * overlap + config.TARGET_SCORE_IOU_WEIGHT * iou
        if score > best_score:
            best_score = score
            best_idx = i
            best_overlap = overlap
            best_iou = iou
    return best_idx, best_overlap, best_iou


def pick_top_overlaps(gaze_bbox, candidates, top_n=2):
    """Like pick_best_overlap but returns the top-N candidates instead of one,
    for multi-object targeting (e.g. Compare needs two).

    Returns a list of (index, overlap, iou) sorted by score descending. Only
    candidates clearing config.TARGET_MIN_OVERLAP are included, so the list may
    be shorter than top_n (or empty) when fewer objects overlap the gaze.
    """
    scored = []
    for i, c in enumerate(candidates):
        overlap = gaze_overlap_ratio(gaze_bbox, c["bbox"])
        if overlap < config.TARGET_MIN_OVERLAP:
            continue
        iou = bbox_iou(gaze_bbox, c["bbox"])
        score = (1 - config.TARGET_SCORE_IOU_WEIGHT) * overlap + config.TARGET_SCORE_IOU_WEIGHT * iou
        scored.append((i, overlap, iou, score))
    scored.sort(key=lambda t: t[3], reverse=True)
    return [(i, ov, iou) for (i, ov, iou, _s) in scored[:top_n]]
