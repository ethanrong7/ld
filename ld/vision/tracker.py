"""Step 7: seeded target tracker.

Initialized at the handoff seed, the tracker predicts with a constant-velocity
(+ constant-omega) model, then corrects using the motion residual measured ONLY
within a gate around the prediction. Gating removes the global "arc centroid"
outliers seen with naive residual argmax: we trust the predicted location and
look for residual mass nearby.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ld.vision.motion import estimate_translation, residual_map
from ld.vision.template import RoundInit


@dataclass
class TrackState:
    x: float
    y: float
    theta: float
    vx: float = 0.0
    vy: float = 0.0
    measured: bool = True   # False when coasting (no residual support)


class TargetTracker:
    def __init__(self, init: RoundInit, init_gray: np.ndarray,
                 search_radius: float | None = None,
                 resid_thresh: int = 50,
                 pos_alpha: float = 0.5,
                 vel_beta: float = 0.7):
        self.x, self.y = init.center_seed
        self.theta = 0.0
        self.omega = init.omega_deg_per_frame
        self.vx = self.vy = 0.0
        self.radius = init.template_radius
        self.search_radius = search_radius or max(45.0, init.template_radius)
        self.resid_thresh = resid_thresh
        self.pos_alpha = pos_alpha
        self.vel_beta = vel_beta
        h, w = init_gray.shape[:2]
        self.window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self._yy, self._xx = np.mgrid[0:h, 0:w]

    def _gated_measurement(self, resid: np.ndarray, px: float, py: float):
        gate = (self._xx - px) ** 2 + (self._yy - py) ** 2 <= self.search_radius ** 2
        weights = resid.astype(np.float32)
        weights[~gate] = 0
        weights[weights < self.resid_thresh] = 0
        total = float(weights.sum())
        if total < 1e-3:
            return None
        mx = float((weights * self._xx).sum() / total)
        my = float((weights * self._yy).sum() / total)
        return mx, my

    def step(self, prev_gray: np.ndarray, cur_gray: np.ndarray) -> TrackState:
        # predict
        px, py = self.x + self.vx, self.y + self.vy
        ptheta = self.theta + self.omega

        dx, dy, _ = estimate_translation(prev_gray, cur_gray, self.window)
        resid = residual_map(prev_gray, cur_gray, dx, dy)
        meas = self._gated_measurement(resid, px, py)

        old_x, old_y = self.x, self.y
        if meas is not None:
            mx, my = meas
            nx = px * (1 - self.pos_alpha) + mx * self.pos_alpha
            ny = py * (1 - self.pos_alpha) + my * self.pos_alpha
            measured = True
        else:
            nx, ny = px, py   # coast
            measured = False

        # velocity update (smoothed)
        self.vx = self.vel_beta * self.vx + (1 - self.vel_beta) * (nx - old_x)
        self.vy = self.vel_beta * self.vy + (1 - self.vel_beta) * (ny - old_y)
        self.x, self.y, self.theta = nx, ny, ptheta
        return TrackState(self.x, self.y, self.theta, self.vx, self.vy, measured)
