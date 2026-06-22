"""Human-cursor output dynamics (plan.md 2026-06-21 — HUMAN-LIKE CURSOR PHYSICS).

A *standalone, mode-agnostic* output-dynamics layer that reshapes the EMITTED
(x, y) point stream so it reads like a hand instead of a detector readout. It
consumes the already-chosen per-frame position (the centroid run through the
trellis + decode-layer freezes) and re-emits a physically-plausible cursor
trajectory. It does NOT touch identity, the trellis, or the decode-layer freezes
-- exactly the project's "decode-layer only" discipline (cf. fpath_hedge /
fpath_freeze, which transform the output, not the identity state).

Two pieces, both strictly causal:

  * ``HumanCursor`` -- a stateful filter; ``.update(target_xy) -> smoothed_xy``,
    called once per frame with the raw chosen position. Three stacked layers,
    each independently switchable (the probe keeps the simplest config that
    clears the bars -- fewer params overfit less under LOO):
      1. **1-Euro filter** (Casiez, Roussel & Vogel 2012) -- speed-adaptive
         low-pass: heavy smoothing at low speed (kills the lock-wobble), light
         smoothing on fast bursts (little lag). The single highest-EV piece.
      2. **Deadband** -- hold position when the (smoothed) target moves < eps px;
         a hand does not chase sub-pixel detector noise.
      3. **Bounded-velocity, critically-damped PD steering** -- give the dot
         inertia so it *glides* to a new location instead of teleporting:
         ``a = k*(target-pos) - c*vel``, ``c = 2*sqrt(k)`` (critical damping, no
         overshoot), ``|vel| <= v_max``, ``|a| <= a_max``. Turns the 200-400px
         freeze-snaps into smooth eases while following legitimate bursts.

  * ``humanize_track(points, fps, **params) -> points`` -- pure function that
    replays a fresh ``HumanCursor`` over a recorded stream, for the offline gate
    (cursor_physics_probe.py) and the optional render-only fallback. Supports a
    bounded fixed-lag ``lag`` (the project's online constraint permits ~10-15
    frames): a centered Gaussian over the causal-smoothed stream, emitted delayed
    by ``lag`` frames (so output t uses inputs up to t, never beyond).

GT is the *target* and GT-only: the filter is driven ONLY by the emitted point
stream -- never feed gt_x/gt_y in.
"""
from __future__ import annotations

import math

from ld.config import (
    HUMAN_MIN_CUTOFF, HUMAN_BETA, HUMAN_DCUTOFF,
    HUMAN_DEADBAND, HUMAN_PD_K, HUMAN_V_MAX, HUMAN_A_MAX, HUMAN_LAG,
)


def _alpha(cutoff: float, dt: float) -> float:
    """1-Euro smoothing factor for a given cutoff frequency (Hz) and timestep."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class HumanCursor:
    """Strictly-causal human-cursor filter. ``update`` sees only past frames.

    Layers are enabled by their params: 1-Euro is always on (set ``beta=0`` for a
    fixed-cutoff low-pass); deadband off when ``deadband<=0``; PD steering off when
    ``use_pd=False`` (then the 1-Euro/deadband output is emitted directly).
    """

    def __init__(self, fps: float = 60.0, *,
                 min_cutoff: float = HUMAN_MIN_CUTOFF,
                 beta: float = HUMAN_BETA,
                 dcutoff: float = HUMAN_DCUTOFF,
                 deadband: float = HUMAN_DEADBAND,
                 use_pd: bool = False,
                 k: float = HUMAN_PD_K,
                 v_max: float = HUMAN_V_MAX,
                 a_max: float = HUMAN_A_MAX):
        self.dt = 1.0 / float(fps)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.deadband = float(deadband)
        self.use_pd = bool(use_pd)
        self.k = float(k)
        self.c = 2.0 * math.sqrt(max(self.k, 0.0))   # critical damping
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        # 1-Euro state
        self._xhat: tuple[float, float] | None = None   # last filtered target
        self._raw_prev: tuple[float, float] | None = None
        self._dhat: float = 0.0                          # filtered speed (px/s)
        # output / PD state
        self._pos: tuple[float, float] | None = None
        self._vx = 0.0
        self._vy = 0.0

    def _one_euro(self, x: float, y: float) -> tuple[float, float]:
        if self._xhat is None:
            self._xhat = (x, y)
            self._raw_prev = (x, y)
            self._dhat = 0.0
            return self._xhat
        # speed = magnitude of the raw derivative (isotropic -> one cutoff for both axes)
        rpx, rpy = self._raw_prev
        dx = (x - rpx) / self.dt
        dy = (y - rpy) / self.dt
        speed = math.hypot(dx, dy)
        a_d = _alpha(self.dcutoff, self.dt)
        self._dhat = a_d * speed + (1.0 - a_d) * self._dhat
        cutoff = self.min_cutoff + self.beta * self._dhat
        a = _alpha(cutoff, self.dt)
        hx = a * x + (1.0 - a) * self._xhat[0]
        hy = a * y + (1.0 - a) * self._xhat[1]
        self._xhat = (hx, hy)
        self._raw_prev = (x, y)
        return self._xhat

    def update(self, target: tuple[float, float]) -> tuple[float, float]:
        tx, ty = self._one_euro(float(target[0]), float(target[1]))

        if self._pos is None:
            self._pos = (tx, ty)
            return self._pos
        px, py = self._pos

        # Deadband: ignore sub-eps moves of the (smoothed) target.
        if self.deadband > 0.0 and math.hypot(tx - px, ty - py) < self.deadband:
            tx, ty = px, py

        if not self.use_pd:
            self._pos = (tx, ty)
            return self._pos

        # Critically-damped PD steer with velocity/accel clamps.
        ax = self.k * (tx - px) - self.c * self._vx
        ay = self.k * (ty - py) - self.c * self._vy
        amag = math.hypot(ax, ay)
        if self.a_max > 0.0 and amag > self.a_max:
            s = self.a_max / amag
            ax, ay = ax * s, ay * s
        self._vx += ax
        self._vy += ay
        vmag = math.hypot(self._vx, self._vy)
        if self.v_max > 0.0 and vmag > self.v_max:
            s = self.v_max / vmag
            self._vx, self._vy = self._vx * s, self._vy * s
        self._pos = (px + self._vx, py + self._vy)
        return self._pos


def _gaussian_kernel(half: int) -> list[float]:
    """Symmetric Gaussian weights of total width 2*half+1 (sigma = half/2)."""
    if half <= 0:
        return [1.0]
    sigma = max(half / 2.0, 1e-6)
    w = [math.exp(-(i * i) / (2.0 * sigma * sigma)) for i in range(-half, half + 1)]
    s = sum(w)
    return [v / s for v in w]


def humanize_track(points: list[tuple[float, float]], fps: float = 60.0, *,
                   lag: int = HUMAN_LAG, **params) -> list[tuple[float, float]]:
    """Replay a fresh ``HumanCursor`` over a recorded ``(x, y)`` stream.

    Pure (no shared state). With ``lag>0``, applies a centered Gaussian (half-width
    ``lag``) on top of the causal-smoothed stream and shifts the output forward by
    ``lag`` frames, so emitted point t depends only on inputs up to frame t (a
    bounded fixed-lag buffer, permitted by the online constraint). Output length
    equals input length; NaN points pass through unchanged and reset the filter.
    """
    cur = HumanCursor(fps=fps, **params)
    smoothed: list[tuple[float, float]] = []
    for p in points:
        if p is None or (isinstance(p[0], float) and math.isnan(p[0])):
            smoothed.append(p)
            cur = HumanCursor(fps=fps, **params)   # reset across gaps
            continue
        smoothed.append(cur.update(p))

    if lag <= 0:
        return smoothed

    ker = _gaussian_kernel(lag)
    n = len(smoothed)
    out: list[tuple[float, float]] = []
    for t in range(n):
        # Emit the centered smoothing of frame (t-lag): window [t-2*lag, t], all <= t.
        c = t - lag
        if c < 0 or smoothed[c] is None or (isinstance(smoothed[c][0], float)
                                            and math.isnan(smoothed[c][0])):
            out.append(smoothed[c] if 0 <= c < n else smoothed[0])
            continue
        sx = sy = wsum = 0.0
        for j, w in enumerate(ker):
            k = c - lag + j
            if k < 0 or k >= n or smoothed[k] is None:
                continue
            q = smoothed[k]
            if isinstance(q[0], float) and math.isnan(q[0]):
                continue
            sx += w * q[0]
            sy += w * q[1]
            wsum += w
        out.append((sx / wsum, sy / wsum) if wsum > 1e-9 else smoothed[c])
    return out
