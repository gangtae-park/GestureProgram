"""Gesture-driven VLM pipeline package, organized by role:

  - config, state : tunable constants / process-wide shared state (top level,
                    imported by every subpackage)
  - gaze/         : gaze direction -> screen coords, calibration, head comp
  - vision/       : YOLO/YOLOE segmentation, CLIP matching, object DB,
                    metric depth, target anchors, OCR
  - llm/          : OpenAI client (translation, Ask follow-ups, streaming)
  - voice/        : voice-command HTTP ingress + GazePointAR-style single-call
                    routing (frame + gaze + query -> one multimodal GPT call)
  - unity/        : ADB stream + UDP comms with Unity, Unity-triggered actions
  - ui/           : OpenCV drawing helpers for the Mac window
  - handlers/     : per-gesture handler registry + dispatch_gesture()

Import as `from vlm_pipeline.<subpackage> import <module>`, e.g.
`from vlm_pipeline.vision import segmentation`.

Entry point lives at /MacProgram/app.py and just wires these together.
"""
