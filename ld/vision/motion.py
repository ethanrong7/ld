"""Rigid paper-motion estimation and independent-motion saliency.

The Lie Detector background ("sheet of paper") moves as a single near-rigid
body, so every fake shape shares one global transform. The real shape moves
independently. We therefore track features between two frames, fit the global
rigid motion with RANSAC, and treat features that disagree with it (the
outliers) as evidence for the real shape. The spatial density of those
outliers is the per-frame saliency the tracker consumes.

This operates on sparse feature points, which makes it immune to the
edge/resampling noise that defeats pixel-level background subtraction.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ld.config import (
    BORDER_MARGIN,
    FEAT_MAX,
    FEAT_MIN_DIST,
    FEAT_QUALITY,
    LK_LEVELS,
    LK_WIN,
    OUTLIER_DOT_RADIUS,
    OUTLIER_RESID_CAP,
    OUTLIER_RESID_MIN,
    RANSAC_THRESH,
    SALIENCY_SIGMA,
)

__all__ = ["MotionField", "estimate_motion", "saliency_map"]


@dataclass
class MotionField:
    """Result of comparing two consecutive (cursor-stripped) gray frames."""

    affine: np.ndarray | None              # 2x3 global rigid sheet motion (prev->cur)
    inliers: np.ndarray                    # (N,2) features moving with the paper
    outliers: np.ndarray                   # (M,2) features moving independently
    outlier_weights: np.ndarray            # (M,) clamped residual magnitude per outlier

    @property
    def ok(self) -> bool:
        return self.affine is not None


def estimate_motion(prev_gray: np.ndarray, cur_gray: np.ndarray) -> MotionField:
    """Track features prev->cur and split them by agreement with rigid motion."""
    empty = np.empty((0, 2), np.float32)
    p0 = cv2.goodFeaturesToTrack(prev_gray, FEAT_MAX, FEAT_QUALITY, FEAT_MIN_DIST)
    if p0 is None:
        return MotionField(None, empty, empty, np.empty((0,), np.float32))

    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, p0, None,
        winSize=(LK_WIN, LK_WIN), maxLevel=LK_LEVELS,
    )
    keep = status.ravel() == 1
    a = p0.reshape(-1, 2)[keep]
    b = p1.reshape(-1, 2)[keep]
    if len(a) < 30:
        return MotionField(None, empty, empty, np.empty((0,), np.float32))

    affine, _ = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH,
    )
    if affine is None:
        return MotionField(None, empty, empty, np.empty((0,), np.float32))

    pred = (a @ affine[:, :2].T) + affine[:, 2]
    resid = np.linalg.norm(pred - b, axis=1)
    out = resid > OUTLIER_RESID_MIN
    return MotionField(
        affine=affine,
        inliers=a[~out],
        outliers=a[out],
        outlier_weights=np.clip(resid[out], 0.0, OUTLIER_RESID_CAP),
    )


def saliency_map(field: MotionField, shape: tuple[int, int]) -> np.ndarray:
    """Blurred spatial density of independently-moving features."""
    h, w = shape
    vote = np.zeros((h, w), np.float32)
    m = BORDER_MARGIN
    for (x, y), weight in zip(field.outliers, field.outlier_weights):
        ix, iy = int(x), int(y)
        if m < ix < w - m and m < iy < h - m:
            cv2.circle(vote, (ix, iy), OUTLIER_DOT_RADIUS, float(weight), -1)
    if vote.max() <= 0:
        return vote
    return cv2.GaussianBlur(vote, (0, 0), SALIENCY_SIGMA)
