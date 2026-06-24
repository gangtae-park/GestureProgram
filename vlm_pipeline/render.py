"""Drawing utilities for the OpenCV target window.

These are pure functions: input frame + state -> output frame. Handlers call
them; the main loop just imshow()s the result.
"""
import cv2
import numpy as np

from . import config


def placeholder_canvas(text: str, h: int = None, w: int = None):
    h = h if h is not None else config.STREAM_H
    w = w if w is not None else config.STREAM_W
    img = np.full((h, w, 3), config.BG_COLOR, dtype=np.uint8)
    cv2.putText(
        img, text, (40, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA,
    )
    return img


def _draw_mask_contour(overlay, mask_bool, color, thickness=2):
    """Trace the outline of a boolean mask onto overlay."""
    if mask_bool is None:
        return
    m = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness, cv2.LINE_AA)


def _blend_gaze_field(overlay, gaze_field, alpha):
    """Blend the gaze Gaussian field onto overlay as a per-pixel-weighted
    heatmap (hotter = looked at more). Returns the blended image."""
    if gaze_field is None:
        return overlay
    f = np.clip(gaze_field, 0.0, 1.0).astype(np.float32)
    if float(f.max()) <= 0.0:
        return overlay
    heat = cv2.applyColorMap((f * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (f * float(alpha))[..., None]
    return (overlay.astype(np.float32) * (1.0 - a) + heat.astype(np.float32) * a).astype(np.uint8)


def render_target_overlay(captured_frame, pixel_points, gaze_field, target,
                           target_source, candidates, gesture_name, gaze_bbox=None):
    """Draw the gaze representation (Gaussian heatmap, or a bbox for the legacy
    OCR path), candidate object outlines, and the target mask outline + label.

    gaze_field : HxW float field from geometry.build_gaze_gaussian_field, or None.
    gaze_bbox  : legacy rectangle (Translate/OCR), drawn only when gaze_field is None.
    """
    overlay = captured_frame.copy()

    # Gaze field heatmap first so later outlines/points stay crisp on top.
    overlay = _blend_gaze_field(overlay, gaze_field, config.GAZE_FIELD_HEATMAP_ALPHA)

    # Candidate objects: trace their YOLO outline (fall back to bbox if no mask).
    for c in candidates[:config.MAX_SEGMENTS_TO_RENDER]:
        if c.get("mask_bool") is not None:
            _draw_mask_contour(overlay, c["mask_bool"], (160, 160, 160), 1)
        else:
            x1, y1, x2, y2 = c["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (160, 160, 160), 1)

    # Legacy rectangular gaze bbox (only when no Gaussian field is supplied).
    if gaze_field is None and gaze_bbox is not None:
        gx1, gy1, gx2, gy2 = gaze_bbox
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), config.TRAIL_COLOR, 3)
        cv2.putText(
            overlay, f"gaze bbox ({len(pixel_points)} pts)",
            (gx1, max(20, gy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )

    # Raw gaze points on top for reference.
    for p in pixel_points:
        cv2.circle(overlay, p, 3, config.TRAIL_COLOR, -1)

    if target is not None:
        color_rgb = config.TARGET_COLOR
        if target.get("mask_bool") is not None:
            color = np.zeros_like(overlay)
            color[target["mask_bool"]] = color_rgb
            overlay = cv2.addWeighted(overlay, 1.0, color, 0.45, 0)
            _draw_mask_contour(overlay, target["mask_bool"], color_rgb, 3)
        else:
            tx1, ty1, tx2, ty2 = target["bbox"]
            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color_rgb, 3)

        label_bits = [f"TARGET[{target_source}]"]
        if "class_name" in target:
            label_bits.append(f"{target.get('class_name', '?')}")
        if "conf" in target:
            label_bits.append(f"conf={target.get('conf', 0.0):.2f}")
        label_bits.append(f"ov={target.get('best_overlap', 0.0):.2f}")
        label_bits.append(f"iou={target.get('best_iou', 0.0):.2f}")
        cv2.putText(
            overlay, "  ".join(label_bits),
            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rgb, 2, cv2.LINE_AA,
        )

    # Gesture label last so the heatmap never washes it out.
    cv2.putText(
        overlay, f"GESTURE: {gesture_name}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    return overlay


