"""Green-cursor detector.

The bright-green crosshair marks the correct answer in the labelled t*-clips.
It is used ONLY as ground truth for evaluation and is never consumed by the
solver itself (real samples will not contain it).
"""
from __future__ import annotations

import cv2
import numpy as np

from ld.config import GREEN_LOWER, GREEN_MIN_AREA, GREEN_UPPER


def find_cursor(frame: np.ndarray) -> tuple[float, float] | None:
    """Return (x, y) centroid of the green cursor, or None if absent."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
    if cv2.countNonZero(mask) < GREEN_MIN_AREA:
        return None
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])
