Gesture-driven target selection pipeline -- entry point.

This file just orchestrates: load models, start threads, run the OpenCV display
loop, and dispatch each completed gesture to its registered handler.

Implementation details live in the vlm_pipeline package:

```
  vlm_pipeline/
  ├── config.py    constants, prompts, paths
  ├── state.py     process-wide shared state + locks
  ├── gaze/        gaze processing
  │   ├── geometry.py          projection + bbox helpers
  │   ├── ridge.py             gaze direction -> normalized screen coords
  │   ├── gaze_calibration.py  runtime accessor for the calibration model
  │   └── headcomp.py          head-rotation compensation for the gaze trail
  ├── vision/      perception
  │   ├── segmentation.py      YOLO/YOLOE segmentation backend
  │   ├── clip_matcher.py      CLIP image embedding + cosine match
  │   ├── object_db.py         fixed object DB (metadata + cached embeddings)
  │   ├── depth.py             monocular metric depth (Depth Anything V2)
  │   ├── target_anchor.py     (gaze_dir, depth) anchor for gesture targets
  │   └── ocr.py               EasyOCR wrapper (ROI + paragraph mode)
  ├── llm/         OpenAI client
  │   └── vlm_client.py        translation, Ask follow-ups, streaming
  ├── voice/       voice commands
  │   ├── voice_server.py      HTTP server for headset STT transcripts
  │   └── voice_pipeline.py    GazePointAR-style single-call voice routing
  ├── unity/       Unity comms + Unity-triggered features
  │   ├── network.py           ADB stream, UDP receive/send, packet parsing
  │   ├── object_action.py     OBJECT_ACTION packets from the Unity bubble menu
  │   └── object_ui.py         HTTP-triggered Object UI pipeline
  ├── ui/          Mac window drawing
  │   └── render.py            OpenCV drawing helpers
  └── handlers/    one file per gesture + registry
      ├── __init__.py          handler registry + dispatch_gesture()
      ├── anchor.py  ask.py  capture.py  compare.py
      └── save.py  search.py  translate.py
```