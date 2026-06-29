"""Gesture handler registry + dispatcher.

To add a new gesture:
  1. Create a new file in this folder (e.g. `compare.py`).
  2. Define a function with signature
        def handle(captured_frame, norm_points, gesture_name) -> np.ndarray
     where the return value is the rendered overlay BGR for the target window.
     Long-running work (segmentation, VLM, network) should run synchronously
     here OR be dispatched to a daemon thread that updates state.target_canvas.
  3. Decorate it with @register("Gesture/Name") -- the name must match the
     `gestureName` Unity sends in GESTURE_EVENT.
  4. Import the new module from this __init__.py so its decorator runs.

The main loop calls dispatch_gesture(...) which routes by gesture name.
Unknown names fall through to a default placeholder so the system keeps running.
"""
from typing import Callable

import numpy as np

from .. import render

# Registry: gesture_name -> handler function
_HANDLERS: dict = {}


def register(name: str) -> Callable:
    """Decorator: registers a handler for the given gesture name."""
    def _decorator(fn: Callable) -> Callable:
        if name in _HANDLERS:
            print(f"[HANDLER][WARN] overriding existing handler for '{name}'")
        _HANDLERS[name] = fn
        return fn
    return _decorator


def get_handler(gesture_name: str):
    return _HANDLERS.get(gesture_name)


def list_registered() -> list:
    return sorted(_HANDLERS.keys())


def _default_handler(captured_frame, norm_points, gesture_name):
    """Fallback when a gesture comes in for which no handler is registered."""
    print(f"[HANDLER][WARN] no handler for gesture '{gesture_name}'. Registered: {list_registered()}")
    if captured_frame is None:
        return render.placeholder_canvas(f"Unsupported gesture: {gesture_name}")
    overlay = captured_frame.copy()
    import cv2
    cv2.putText(
        overlay, f"UNSUPPORTED GESTURE: {gesture_name}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
    )
    return overlay


def dispatch_gesture(captured_frame, norm_points, gesture_name: str) -> np.ndarray:
    """Route a finished gesture to its handler. Always returns a renderable BGR overlay."""
    handler = _HANDLERS.get(gesture_name) or _default_handler
    return handler(captured_frame, norm_points, gesture_name)


# Import every handler module so its @register(...) calls execute on package load.
# Add new handler imports here as you add gestures.
from . import search_find_info  # noqa: E402, F401
from . import ask     # noqa: E402, F401
from . import translate  # noqa: E402, F401
from . import anchor  # noqa: E402, F401
from . import compare  # noqa: E402, F401
from . import save  # noqa: E402, F401
from . import capture  # noqa: E402, F401
