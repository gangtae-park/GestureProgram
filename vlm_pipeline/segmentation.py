"""YOLO segmentation backend.

YOLO-only: if ultralytics or the model file is missing, run_yolo() returns []
so handlers degrade to a "no target" overlay rather than crashing.
"""
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as _yolo_exc:
    YOLO = None
    print(f"[YOLO][IMPORT][WARN] {_yolo_exc}")

from . import config


_yolo_model = None


def load_yolo_model() -> bool:
    global _yolo_model
    if YOLO is None:
        print("[YOLO][ERROR] ultralytics not importable.")
        return False
    try:
        _yolo_model = YOLO(config.SEG_MODEL_PATH)
        print(f"[YOLO] Loaded model: {config.SEG_MODEL_PATH}")
        return True
    except Exception as exc:
        _yolo_model = None
        print(f"[YOLO][ERROR] {exc}")
        return False


def run_yolo(captured_frame: np.ndarray):
    """Returns list of segment dicts:
       [{ 'bbox': (x1,y1,x2,y2), 'mask_bool': HxW or None,
          'class_id': int, 'class_name': str, 'conf': float }, ...]
    Empty list on failure / no detections.
    """
    if _yolo_model is None:
        return []
    try:
        results = _yolo_model.predict(
            source=captured_frame,
            conf=config.SEG_CONF,
            iou=config.SEG_IOU_THRESH,
            verbose=False,
            retina_masks=True,
        )
    except Exception as exc:
        print(f"[YOLO][ERROR] predict failed: {exc}")
        return []

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    r0 = results[0]
    boxes = r0.boxes.xyxy.cpu().numpy()
    classes = r0.boxes.cls.cpu().numpy().astype(int)
    confs = r0.boxes.conf.cpu().numpy()
    names = r0.names
    masks = r0.masks.data.cpu().numpy() if r0.masks is not None else None

    items = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b.astype(int)
        m = None
        if masks is not None and i < len(masks):
            mask_arr = masks[i]
            if mask_arr.shape != captured_frame.shape[:2]:
                mask_arr = cv2.resize(
                    mask_arr,
                    (captured_frame.shape[1], captured_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            m = mask_arr > 0.5
        cn = int(classes[i])
        items.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "mask_bool": m,
            "class_id": cn,
            "class_name": (names.get(cn, str(cn)) if isinstance(names, dict) else str(cn)),
            "conf": float(confs[i]),
        })
    return items
