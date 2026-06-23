"""Gesture-driven VLM pipeline package.

Public modules (import as `from vlm_pipeline import <module>`):
  - config       : all tunable constants (network, models, prompt)
  - state        : process-wide shared state + locks
  - geometry     : projection / bbox / IoU helpers
  - ridge        : gaze direction -> normalized screen coords
  - segmentation : YOLO segmentation backend
  - clip_matcher : CLIP image embedding + cosine match against object DB
  - object_db    : fixed 3-object DB (metadata + cached CLIP embeddings)
  - ocr          : EasyOCR wrapper with ROI + paragraph mode
  - vlm_client   : OpenAI client (translation + Ask follow-ups + Whisper)
  - network      : packet parsing, ADB stream thread, UDP receive thread,
                   Python -> Unity UDP sender
  - render       : drawing utilities (overlays, placeholder canvas)
  - handlers     : per-gesture handler registry + dispatch_gesture()

Entry point lives at /MacProgram/gesture_vlm.py and just wires these together.
"""
