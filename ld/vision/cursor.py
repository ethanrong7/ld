"""Pointer handling for the LD solver.

- ``find_cursor``: locate the green GT crosshair (evaluation only).
- ``strip_pointer``: remove pointer pixels from frames before tracking / motion
  analysis so the solver never sees a moving cursor beacon (labelled green GT
  or a live mouse at a known screen position).
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ld.config import (
    GREEN_LOWER,
    GREEN_MIN_AREA,
    GREEN_UPPER,
    POINTER_INPAINT_RADIUS,
    POINTER_RADIUS,
)

# Re-export for callers that only need detection
__all__ = [
    "find_cursor",
    "green_mask",
    "strip_pointer",
    "PointerAt",
]

PointerAt = Callable[[int], tuple[float, float] | None]


def find_cursor(frame: np.ndarray) -> tuple[float, float] | None:
    """Return (x, y) centroid of the green cursor, or None if absent."""
    mask = green_mask(frame)
    if cv2.countNonZero(mask) < GREEN_MIN_AREA:
        return None
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def green_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask of bright-green GT cursor pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, k)


def _disk_mask(h: int, w: int, x: float, y: float, radius: float) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(round(x)), int(round(y))), int(round(radius)), 255, -1)
    return mask


def strip_pointer(
    frame: np.ndarray,
    *,
    mouse_xy: tuple[float, float] | None = None,
    mouse_radius: float | None = None,
    strip_green: bool = True,
) -> np.ndarray:
    """Return a copy of ``frame`` with pointer pixels inpainted away.

    Applies before grayscale / motion residual / tracking on every pathway.

    - ``strip_green``: inpaint the labelled GT green crosshair (t* clips).
    - ``mouse_xy``: inpaint a disk at the last known live-mouse position; use
      this in live mode where the OS cursor is not green.
    """
    h, w = frame.shape[:2]
    combined = np.zeros((h, w), np.uint8)
    if mouse_xy is not None:
        r = mouse_radius if mouse_radius is not None else POINTER_RADIUS
        combined = cv2.bitwise_or(combined, _disk_mask(h, w, mouse_xy[0], mouse_xy[1], r))
    if strip_green:
        combined = cv2.bitwise_or(combined, green_mask(frame))
    if cv2.countNonZero(combined) == 0:
        return frame
    return cv2.inpaint(frame, combined, POINTER_INPAINT_RADIUS, cv2.INPAINT_TELEA)
