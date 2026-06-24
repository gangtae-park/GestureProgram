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

    Legacy bbox-vs-bbox targeting. Object handlers now prefer the Gaussian
    field + mask path below; this is kept for non-object callers.
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


# =========================================================================
# Gaze Gaussian field  +  mask soft-IoU targeting
# =========================================================================
# The gaze trail is turned into a soft 2D field (a fuzzy union of Gaussian
# circles centred on each gaze point). A YOLO object is scored by the soft IoU
# between this field and the object's segmentation MASK -- i.e. against the
# object's real outline, not its bounding box. Because the field decays away
# from where the user actually looked, the empty gap between two separated
# objects contributes almost nothing, which is what Compare needs.


def build_gaze_gaussian_field(pixel_points, frame_shape, sigma_px=None):
    """Return an HxW float32 gaze field in [0, 1].

    Each gaze sample paints a Gaussian blob and the blobs are ACCUMULATED
    (summed), so this is a fixation/dwell heatmap: places the user dwelled on
    (many overlapping samples) build up high weight, while a brief saccade
    grazing across the gap deposits very little. The result is normalised by
    its peak so values land in [0, 1] and the soft IoU against an object mask
    stays comparable in scale. Each blob is only evaluated inside a
    +/-TRUNCATE*sigma window for speed.
    """
    h, w = frame_shape[:2]
    field = np.zeros((h, w), dtype=np.float32)
    if not pixel_points:
        return field

    if sigma_px is None:
        sigma_px = max(config.GAZE_GAUSSIAN_MIN_SIGMA_PX,
                       config.GAZE_GAUSSIAN_SIGMA_FRAC * min(h, w))
    sigma = float(sigma_px)
    if sigma <= 0:
        return field
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    radius = int(np.ceil(config.GAZE_GAUSSIAN_TRUNCATE * sigma))

    for px, py in pixel_points:
        x0 = max(0, px - radius); x1 = min(w, px + radius + 1)
        y0 = max(0, py - radius); y1 = min(h, py + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        dx = np.arange(x0, x1, dtype=np.float32) - px
        dy = np.arange(y0, y1, dtype=np.float32) - py
        blob = np.exp(-(np.outer(dy * dy, np.ones_like(dx))
                        + np.outer(np.ones_like(dy), dx * dx)) * inv_2s2)
        field[y0:y1, x0:x1] += blob

    peak = float(field.max())
    if peak > 0:
        field /= peak
    return field


def _candidate_mask(candidate, frame_shape):
    """Boolean HxW mask for a YOLO candidate: its segmentation mask when
    available, otherwise a filled rectangle from its bbox (so box-only
    detections still participate)."""
    mask = candidate.get("mask_bool")
    if mask is not None:
        return mask
    h, w = frame_shape[:2]
    rect = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = candidate["bbox"]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 > x1 and y2 > y1:
        rect[y1:y2, x1:x2] = True
    return rect


def gaussian_mask_iou(gaze_field, mask_bool):
    """Soft IoU (fuzzy Jaccard) between the gaze field (values 0..1) and a
    binary mask:  inter = sum(field over mask),  union = mask_area + field_sum
    - inter."""
    inter = float(gaze_field[mask_bool].sum())
    field_sum = float(gaze_field.sum())
    mask_area = float(np.count_nonzero(mask_bool))
    union = mask_area + field_sum - inter
    return inter / union if union > 0 else 0.0


def gaussian_mask_overlap(gaze_field, mask_bool):
    """Fraction of total gaze weight that falls on the mask. Used as the
    inclusion gate (cheap, scale-free)."""
    field_sum = float(gaze_field.sum())
    if field_sum <= 0:
        return 0.0
    return float(gaze_field[mask_bool].sum()) / field_sum


def _score_candidates_by_mask(gaze_field, candidates, frame_shape):
    """Yield (index, overlap, iou) for every candidate clearing the overlap
    gate, scored against the gaze field via its mask."""
    scored = []
    for i, c in enumerate(candidates):
        mask = _candidate_mask(c, frame_shape)
        overlap = gaussian_mask_overlap(gaze_field, mask)
        if overlap < config.TARGET_MIN_OVERLAP:
            continue
        iou = gaussian_mask_iou(gaze_field, mask)
        scored.append((i, overlap, iou))
    return scored


def pick_best_mask_target(gaze_field, candidates, frame_shape):
    """Single best object for the gaze field. Returns (best_index, overlap,
    iou); (-1, 0, 0) when nothing clears the gate. Mirrors pick_best_overlap's
    overlap-gate + IoU-weighted score, but on masks instead of boxes."""
    best_idx, best_score, best_overlap, best_iou = -1, -1.0, 0.0, 0.0
    for i, overlap, iou in _score_candidates_by_mask(gaze_field, candidates, frame_shape):
        score = (1 - config.TARGET_SCORE_IOU_WEIGHT) * overlap + config.TARGET_SCORE_IOU_WEIGHT * iou
        if score > best_score:
            best_score, best_idx, best_overlap, best_iou = score, i, overlap, iou
    return best_idx, best_overlap, best_iou


def pick_top_mask_targets(gaze_field, candidates, frame_shape, top_n=None):
    """Top-N distinct objects by soft IoU, for Compare-style multi-targeting.
    Returns a list of (index, overlap, iou) sorted by IoU desc (highest first),
    each clearing config.TARGET_MIN_OVERLAP."""
    if top_n is None:
        top_n = config.TARGET_MULTI_TOP_N
    scored = _score_candidates_by_mask(gaze_field, candidates, frame_shape)
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:top_n]
