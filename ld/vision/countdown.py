"""Countdown-phase analysis.

During the countdown the target is a solid near-white filled shape, centered
and rotating at a constant rate. This module segments that white blob so we can
(a) detect the countdown phase, (b) capture the target outline/template, and
(c) measure its centroid and orientation over time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ld.config import WHITE_MIN_AREA, WHITE_S_MAX, WHITE_V_MIN


@dataclass
class WhiteShape:
    cx: float
    cy: float
    area: float
    angle: float  # radians, from 2nd-order moments (mod pi); NaN if degenerate
    mask: np.ndarray  # full-frame uint8 mask of the chosen blob
    bbox: tuple[int, int, int, int]  # x, y, w, h


def white_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s <= WHITE_S_MAX) & (v >= WHITE_V_MIN)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def _moment_angle(blob: np.ndarray) -> float:
    m = cv2.moments(blob, binaryImage=True)
    denom = m["mu20"] - m["mu02"]
    if m["m00"] == 0 or (abs(2 * m["mu11"]) < 1e-6 and abs(denom) < 1e-6):
        return math.nan
    return 0.5 * math.atan2(2 * m["mu11"], denom)


def detect_white_shape(frame: np.ndarray) -> WhiteShape | None:
    """Largest near-white blob (the countdown target), or None."""
    mask = white_mask(frame)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    # pick largest non-background component above area floor
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    best = None
    for k in order:
        idx = k + 1
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < WHITE_MIN_AREA:
            break
        best = idx
        break
    if best is None:
        return None
    blob = (labels == best).astype(np.uint8) * 255
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    cx, cy = centroids[best]
    return WhiteShape(
        cx=float(cx),
        cy=float(cy),
        area=float(stats[best, cv2.CC_STAT_AREA]),
        angle=_moment_angle(blob),
        mask=blob,
        bbox=(x, y, w, h),
    )
