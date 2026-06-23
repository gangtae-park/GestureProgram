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


def render_target_overlay(captured_frame, pixel_points, gaze_bbox, target,
                           target_source, candidates, gesture_name):
    """Draw gaze trail, gaze bbox, candidate boxes (light), target mask + bbox + label."""
    overlay = captured_frame.copy()

    cv2.putText(
        overlay, f"GESTURE: {gesture_name}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    for c in candidates[:config.MAX_SEGMENTS_TO_RENDER]:
        x1, y1, x2, y2 = c["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (160, 160, 160), 1)

    for p in pixel_points:
        cv2.circle(overlay, p, 3, config.TRAIL_COLOR, -1)

    if gaze_bbox is not None:
        gx1, gy1, gx2, gy2 = gaze_bbox
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), config.TRAIL_COLOR, 3)
        cv2.putText(
            overlay, f"gaze bbox ({len(pixel_points)} pts)",
            (gx1, max(20, gy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TRAIL_COLOR, 2, cv2.LINE_AA,
        )

    if target is not None:
        tx1, ty1, tx2, ty2 = target["bbox"]
        color_rgb = config.TARGET_COLOR
        if target.get("mask_bool") is not None:
            color = np.zeros_like(overlay)
            color[target["mask_bool"]] = color_rgb
            overlay = cv2.addWeighted(overlay, 1.0, color, 0.45, 0)
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

    return overlay


