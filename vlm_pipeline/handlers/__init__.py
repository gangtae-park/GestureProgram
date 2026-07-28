"""
Gesture handler registry + dispatcher.
The main loop calls dispatch_gesture which routes by gesture name.
"""
from typing import Callable

import numpy as np

from ..ui import render

_HANDLERS: dict = {}

def register(name: str) -> Callable:
    """Registers a handler for the given gesture name."""
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


# Import every handler module so its @register calls execute on package load.
from . import search
from . import ask
from . import translate
from . import anchor
from . import compare
from . import save
from . import capture
