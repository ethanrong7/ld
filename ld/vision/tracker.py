"""Seeded target tracker: template match (primary) + motion residual (fallback).

Initialized at the handoff seed. Each frame predicts with constant-velocity
(+ constant-ω for asymmetric shapes), estimates background translation (with
the predicted target region masked), then fuses a gated template match with a
gated residual centroid.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ld.config import TEMPLATE_MATCH_MIN
from ld.vision.match import _extract_patch, match_in_gate
from ld.vision.motion import estimate_translation, residual_map
from ld.vision.template import RoundInit


@dataclass
class TrackState:
    x: float
    y: float
    theta: float
    vx: float = 0.0
    vy: float = 0.0
    measured: bool = True


class TargetTracker:
    def __init__(
        self,
        init: RoundInit,
        init_gray: np.ndarray,
        appearance_gray: np.ndarray | None = None,
        search_radius: float | None = None,
        resid_thresh: int = 50,
        pos_alpha: float = 0.5,
        vel_beta: float = 0.7,
    ):
        self.init = init
        self.x, self.y = init.center_seed
        self.theta = 0.0
        self.omega = init.omega_deg_per_frame
        self.symmetric = init.symmetric
        self.vx = self.vy = 0.0
        self.radius = init.template_radius
        self.search_radius = search_radius or max(45.0, init.template_radius)
        self.resid_thresh = resid_thresh
        self.pos_alpha = pos_alpha
        self.vel_beta = vel_beta
        h, w = init_gray.shape[:2]
        self.window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self._yy, self._xx = np.mgrid[0:h, 0:w]
        half = int(round(max(24, self.radius * 1.1)))
        ag = appearance_gray if appearance_gray is not None else init_gray
        self._appearance = _extract_patch(ag, self.x, self.y, half)

    def _mask_for_bg(self, gray: np.ndarray) -> np.ndarray:
        """Reduce target bias on global phase-correlation shift estimate."""
        out = gray.copy()
        fill = float(np.median(gray))
        cv2.circle(
            out,
            (int(round(self.x)), int(round(self.y))),
            int(round(self.radius * 1.2)),
            fill,
            -1,
        )
        return out

    def _gated_residual(self, resid: np.ndarray, px: float, py: float):
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

    def _fuse(
        self,
        px: float,
        py: float,
        tmpl: tuple[float, float, float] | None,
        resid: tuple[float, float] | None,
    ) -> tuple[float, float, bool]:
        if tmpl is not None:
            tx, ty, tscore = tmpl
            if resid is not None:
                rx, ry = resid
                span = max(1e-3, 1.0 - TEMPLATE_MATCH_MIN)
                w = min(0.88, 0.5 + 0.45 * (tscore - TEMPLATE_MATCH_MIN) / span)
                if self.symmetric:
                    w = min(0.92, w + 0.08)
                mx = w * tx + (1.0 - w) * rx
                my = w * ty + (1.0 - w) * ry
                return mx, my, True
            return tx, ty, True
        if resid is not None:
            return resid[0], resid[1], True
        return px, py, False

    def step(self, prev_gray: np.ndarray, cur_gray: np.ndarray) -> TrackState:
        px, py = self.x + self.vx, self.y + self.vy
        ptheta = self.theta + self.omega

        prev_bg = self._mask_for_bg(prev_gray)
        dx, dy, _ = estimate_translation(prev_bg, cur_gray, self.window)
        resid = residual_map(prev_gray, cur_gray, dx, dy)

        tmpl = match_in_gate(
            cur_gray,
            self.init.template,
            px,
            py,
            ptheta,
            self.search_radius,
            symmetric=self.symmetric,
            appearance=self._appearance,
        )
        resid_meas = self._gated_residual(resid, px, py)
        mx, my, measured = self._fuse(px, py, tmpl, resid_meas)

        old_x, old_y = self.x, self.y
        if measured:
            nx = px * (1 - self.pos_alpha) + mx * self.pos_alpha
            ny = py * (1 - self.pos_alpha) + my * self.pos_alpha
        else:
            nx, ny = px, py

        self.vx = self.vel_beta * self.vx + (1 - self.vel_beta) * (nx - old_x)
        self.vy = self.vel_beta * self.vy + (1 - self.vel_beta) * (ny - old_y)
        self.x, self.y, self.theta = nx, ny, ptheta
        return TrackState(self.x, self.y, self.theta, self.vx, self.vy, measured)
