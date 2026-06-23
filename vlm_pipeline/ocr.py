"""EasyOCR wrapper tuned for the gaze-driven translate pipeline.

Speed tricks:
  - run_ocr_in_roi() restricts EasyOCR to a padded crop around the gaze bbox
    instead of the entire 1100x1000 frame. EasyOCR's runtime grows roughly
    with the pixel count, so this is by far the biggest win.
  - The ROI is optionally downscaled to config.OCR_MAX_SIDE before being fed
    to the reader; results are rescaled back to original-frame coords.
  - paragraph=True groups nearby word boxes into sentences/paragraphs in one
    pass, which is also what the Translate handler wants to ship to Unity.

Return shape stays the same regardless of mode:
    [
      {"text": str, "bbox": (x1,y1,x2,y2), "conf": float, "polygon": [(x,y), ...]},
      ...
    ]
with coordinates always expressed in the ORIGINAL frame's pixel space.
"""
import time

import cv2
import numpy as np

try:
    import easyocr
except Exception as _easyocr_exc:
    easyocr = None
    print(f"[OCR][IMPORT][WARN] {_easyocr_exc}")

from . import config


_reader = None
_reader_languages: list = []


def init_ocr_reader(languages=None) -> bool:
    """Build the EasyOCR Reader once on CPU. Returns True on success.

    languages: list of language codes (EasyOCR codes), e.g. ['en'], ['en', 'ko'].
    Defaults to ['en'] since the translate handler targets English -> Korean.
    """
    global _reader, _reader_languages
    if easyocr is None:
        print("[OCR][ERROR] easyocr package not importable. `pip install easyocr`")
        _reader = None
        return False

    languages = languages or ["en"]
    try:
        t0 = time.perf_counter()
        _reader = easyocr.Reader(languages, gpu=False, verbose=False)
        _reader_languages = list(languages)
        elapsed = time.perf_counter() - t0
        print(f"[OCR] EasyOCR ready in {elapsed*1000:.0f} ms, langs={languages}")
        return True
    except Exception as exc:
        _reader = None
        print(f"[OCR][ERROR] init failed: {exc}")
        return False


def _expand_roi_around_gaze(gaze_bbox, frame_shape) -> tuple:
    """Pad gaze_bbox by OCR_ROI_PAD_PX on every side and enforce OCR_ROI_MIN_SIDE.
    Returns (x1, y1, x2, y2) clipped to the frame. If gaze_bbox is None, returns
    the whole frame.
    """
    h, w = frame_shape[:2]
    if gaze_bbox is None:
        return 0, 0, w, h

    gx1, gy1, gx2, gy2 = gaze_bbox
    pad = max(int(config.OCR_ROI_PAD_PX), 0)
    x1 = gx1 - pad
    y1 = gy1 - pad
    x2 = gx2 + pad
    y2 = gy2 + pad

    # Inflate to the minimum side length while staying centred.
    min_side = max(int(config.OCR_ROI_MIN_SIDE), 0)
    if min_side > 0:
        if (x2 - x1) < min_side:
            cx = (x1 + x2) // 2
            x1, x2 = cx - min_side // 2, cx + min_side // 2
        if (y2 - y1) < min_side:
            cy = (y1 + y2) // 2
            y1, y2 = cy - min_side // 2, cy + min_side // 2

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    return x1, y1, x2, y2


def run_ocr_in_roi(frame_bgr: np.ndarray, gaze_bbox=None,
                   min_conf: float = None, paragraph: bool = None) -> list:
    """Run OCR on a padded ROI around gaze_bbox and return results in original
    frame coordinates. When gaze_bbox is None, falls back to the whole frame.
    """
    if _reader is None:
        return []
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    min_conf = config.OCR_MIN_CONF if min_conf is None else min_conf
    paragraph = config.OCR_PARAGRAPH if paragraph is None else paragraph

    x1, y1, x2, y2 = _expand_roi_around_gaze(gaze_bbox, frame_bgr.shape)
    if x2 <= x1 or y2 <= y1:
        return []
    roi = frame_bgr[y1:y2, x1:x2]

    # Optional downscale for speed. The mapping back to original frame coords
    # multiplies by inv_scale and then adds the ROI offset (x1, y1).
    long_side = max(roi.shape[0], roi.shape[1])
    if config.OCR_MAX_SIDE and long_side > config.OCR_MAX_SIDE:
        scale = config.OCR_MAX_SIDE / float(long_side)
        new_w = max(1, int(round(roi.shape[1] * scale)))
        new_h = max(1, int(round(roi.shape[0] * scale)))
        roi_small = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        roi_small = roi

    inv_scale = 1.0 / scale if scale > 0 else 1.0

    t0 = time.perf_counter()
    try:
        raw = _reader.readtext(roi_small, paragraph=paragraph)
    except Exception as exc:
        print(f"[OCR][ERROR] readtext failed: {exc}")
        return []
    elapsed = time.perf_counter() - t0
    print(
        f"[OCR] roi={x2-x1}x{y2-y1} scaled={roi_small.shape[1]}x{roi_small.shape[0]} "
        f"paragraph={paragraph} -> {len(raw)} blocks in {elapsed*1000:.0f} ms"
    )

    results = []
    for entry in raw:
        # paragraph=True returns (polygon, text) WITHOUT a confidence.
        # paragraph=False returns (polygon, text, conf).
        try:
            if len(entry) == 3:
                polygon, text, conf = entry
            elif len(entry) == 2:
                polygon, text = entry
                conf = 1.0
            else:
                continue
        except (ValueError, TypeError):
            continue

        if conf is None:
            conf = 1.0
        if float(conf) < min_conf:
            continue
        if not text or not text.strip():
            continue

        # Remap polygon to original frame coords.
        mapped = [
            (int(round(p[0] * inv_scale)) + x1, int(round(p[1] * inv_scale)) + y1)
            for p in polygon
        ]
        xs = [p[0] for p in mapped]
        ys = [p[1] for p in mapped]
        ox1, oy1 = int(min(xs)), int(min(ys))
        ox2, oy2 = int(max(xs)), int(max(ys))
        if ox2 <= ox1 or oy2 <= oy1:
            continue

        results.append({
            "text": text.strip(),
            "bbox": (ox1, oy1, ox2, oy2),
            "conf": float(conf),
            "polygon": mapped,
        })

    return results


# Back-compat for any caller that still wants the old "OCR the whole frame" path.
def run_ocr(frame_bgr: np.ndarray, min_conf: float = None) -> list:
    return run_ocr_in_roi(frame_bgr, gaze_bbox=None, min_conf=min_conf)
