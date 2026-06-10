"""Countdown acquisition.

Before the round starts the real shape is rendered solid white in the centre.
That gives us, with no reliance on the green crosshair, both an initial
position and a rough scale (radius) to seed the tracker once the shape blends
into the sheet. The cyan countdown numbers are bright but saturated, so an
upper-saturation bound rejects them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ld.config import WHITE_MIN_AREA, WHITE_S_MAX, WHITE_V_MIN

__all__ = ["WhiteShape", "detect_white_shape", "acquire_seed"]


@dataclass
class WhiteShape:
    cx: float
    cy: float
    radius: float


def detect_white_shape(frame: np.ndarray) -> WhiteShape | None:
    """Locate the bright, desaturated countdown shape, or None if absent."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 0, WHITE_V_MIN)), np.array((180, WHITE_S_MAX, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < WHITE_MIN_AREA:
        return None
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return WhiteShape(cx, cy, math.sqrt(area / math.pi))


@dataclass
class Seed:
    cx: float
    cy: float
    radius: float
    start_frame: int  # first frame after the white shape has blended away


def acquire_seed(
    frames: list[np.ndarray],
    *,
    miss_to_confirm: int = 3,
) -> Seed | None:
    """Track the white shape through the countdown; return its last pose.

    ``start_frame`` is where the shape has faded (acquisition handoff point).
    """
    last: WhiteShape | None = None
    last_idx = -1
    radii: list[float] = []
    misses = 0
    for idx, frame in enumerate(frames):
        ws = detect_white_shape(frame)
        if ws is not None:
            last, last_idx = ws, idx
            radii.append(ws.radius)
            misses = 0
        elif last is not None:
            misses += 1
            if misses >= miss_to_confirm:
                break
    if last is None:
        return None
    return Seed(last.cx, last.cy, float(np.median(radii)), last_idx + 1)
