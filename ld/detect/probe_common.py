"""Shared helpers for the surviving tuning probes.

These small utilities were previously scattered across one-off experiment probes that
have since been deleted (sheet_residual_probe, resid_override_probe, localize_probe).
They are kept here because the LOO tuning probes for the shipped method still use them:

  - resid_freeze_probe.py  -> _inv, _nearest, _resid_cache  (tunes fpath_freeze tau/lag)
  - cursor_physics_probe.py -> _add_clips                    (tunes fpath_human dynamics)

Affine helpers operate on the cached per-frame prev->cur RANSAC affines (see
hedge_probe._affines). The residual machinery is the EXP-Q1 cumulative sheet-frame
back-walk that fpath_freeze ships as identity._cumulative_residual.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ld.config import DATA_DIR

# Held-out validation clips (additional_evidence board); a03/a07 excluded (bad GT).
ADD_DIR = DATA_DIR / "additional_evidence"
ADD_EXCLUDE = {"a03", "a07"}


def _inv(T):
    """2x3 inverse affine (cur->prev) of the cached prev->cur affine, or None."""
    if T is None:
        return None
    return cv2.invertAffineTransform(np.asarray(T, np.float64))


def _ap(T, p):
    """Map point p by 2x3 affine T (identity if None)."""
    if T is None:
        return np.asarray(p, np.float64)
    return T[:, :2] @ np.asarray(p, np.float64) + T[:, 2]


def _nearest(p, cloud):
    """(index, distance) of the nearest centroid in `cloud` (list of np arrays) to p."""
    best_i, best_d = -1, float("inf")
    for j, c in enumerate(cloud):
        d = math.hypot(c[0] - p[0], c[1] - p[1])
        if d < best_d:
            best_i, best_d = j, d
    return best_i, best_d


def _residual_mag(t, i, cents, invaff, radius, N, snap_frac=1.0):
    """Cumulative sheet-frame residual MAGNITUDE of box `i` (frame `t`) at horizon N, or None
    if the affine-prediction chain breaks before reaching N (early frame, or association lost).
    Same back-walk shipped as identity._cumulative_residual."""
    p_ref = np.asarray(cents[t][i], np.float64)   # rigid back-transport of frame-t centroid
    chain = np.asarray(cents[t][i], np.float64)   # actual detected ancestor (snap each step)
    for s in range(1, N + 1):
        f = t - s + 1
        Tinv = invaff.get(f)
        prev = f - 1
        if Tinv is None or prev not in cents or not cents[prev]:
            return None
        p_ref = _ap(Tinv, p_ref)
        pred = _ap(Tinv, chain)
        j, d = _nearest(pred, cents[prev])
        if d >= snap_frac * radius:
            return None
        chain = cents[prev][j]
    return math.hypot(p_ref[0] - chain[0], p_ref[1] - chain[1])


def _resid_cache(rows, cents, invaff, radius, N):
    """{idx: [residual or None per box]} for every scored frame -- the expensive pass,
    depends only on N so it is reused across a sweep."""
    out = {}
    for r in rows:
        t = r["idx"]
        boxes = cents.get(t)
        if not boxes:
            continue
        out[t] = [_residual_mag(t, i, cents, invaff, radius, N) for i in range(len(boxes))]
    return out


def _add_clips() -> list[Path]:
    """Valid held-out additional_evidence clips (a*.mp4, excluding a03/a07)."""
    return sorted(p for p in ADD_DIR.glob("a*.mp4")
                  if p.stem.split("_")[0] not in ADD_EXCLUDE)
