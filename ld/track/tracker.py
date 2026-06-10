"""Recursive tracker over the independent-motion saliency.

A constant-velocity estimate is gated to a search radius around the predicted
position; the gated saliency peak updates it. When the shape slows it stops
shedding outliers and the saliency fades, so the tracker coasts on its motion
model; after sustained low confidence it re-acquires from the global saliency
peak. The green crosshair is never consulted.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ld.config import (
    GATE_RADIUS,
    REACQUIRE_FRAC,
    REACQUIRE_PATIENCE,
    SALIENCY_EMA,
    SALIENCY_FLOOR,
    UPDATE_ALPHA,
    VEL_DAMP,
    VEL_MAX,
)

__all__ = ["OutlierTracker", "TrackPoint"]


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float
    confidence: float
    state: str  # "track" | "coast" | "reacquire"


class OutlierTracker:
    def __init__(self, x: float, y: float, gate_radius: float | None = None):
        self.pos = np.array([x, y], np.float32)
        self.vel = np.zeros(2, np.float32)
        self.gate = float(gate_radius) if gate_radius else GATE_RADIUS
        self.sal_ema: np.ndarray | None = None
        self.lost = 0
        self._grid: tuple[np.ndarray, np.ndarray] | None = None

    def _gate_mask(self, shape: tuple[int, int], center: np.ndarray) -> np.ndarray:
        h, w = shape
        if self._grid is None or self._grid[0].shape != (h, w):
            ys, xs = np.mgrid[0:h, 0:w]
            self._grid = (xs.astype(np.float32), ys.astype(np.float32))
        xs, ys = self._grid
        d2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
        return np.exp(-d2 / (2.0 * self.gate * self.gate)).astype(np.float32)

    def update(self, frame_idx: int, saliency: np.ndarray) -> TrackPoint:
        if self.sal_ema is None:
            self.sal_ema = saliency.copy()
        else:
            self.sal_ema = SALIENCY_EMA * self.sal_ema + (1.0 - SALIENCY_EMA) * saliency

        pred = self.pos + self.vel
        _, wval, _, wloc = cv2.minMaxLoc(self.sal_ema)
        gated = self.sal_ema * self._gate_mask(self.sal_ema.shape, pred)
        _, gval, _, gloc = cv2.minMaxLoc(gated)

        trusted = wval >= SALIENCY_FLOOR and gval >= REACQUIRE_FRAC * wval
        reacquired = False
        if trusted:
            meas = np.array(gloc, np.float32)
            new = (1.0 - UPDATE_ALPHA) * pred + UPDATE_ALPHA * meas
            self.lost = 0
            state = "track"
        else:
            new = pred
            self.lost += 1
            state = "coast"
            if self.lost >= REACQUIRE_PATIENCE and wval >= SALIENCY_FLOOR:
                new = np.array(wloc, np.float32)
                self.lost = 0
                state = "reacquire"
                reacquired = True

        if reacquired:
            self.vel[:] = 0.0
        else:
            self.vel = VEL_DAMP * self.vel + (1.0 - VEL_DAMP) * (new - self.pos)
            speed = float(np.hypot(*self.vel))
            if speed > VEL_MAX:
                self.vel *= VEL_MAX / speed

        h, w = self.sal_ema.shape
        self.pos = np.array([float(np.clip(new[0], 0, w - 1)),
                             float(np.clip(new[1], 0, h - 1))], np.float32)
        return TrackPoint(frame_idx, float(self.pos[0]), float(self.pos[1]),
                          float(gval), state)
