"""Global background-motion estimation and motion residual.

The background is a rigid sheet that only translates frame-to-frame, so its
dominant motion is a single (dx, dy). Phase correlation recovers that shift
robustly (the target is a small minority of pixels). After compensating the
background shift, pixels that still differ are where motion disagreed with the
background -- i.e. the independently-moving (and rotating) target.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_gray_f32(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def estimate_translation(prev_gray: np.ndarray, cur_gray: np.ndarray,
                         window: np.ndarray | None = None) -> tuple[float, float, float]:
    """Dominant (background) shift mapping prev -> cur, plus response.

    Returns (dx, dy, response). (dx, dy) is how much `prev` content moved to
    align with `cur`.
    """
    if window is None:
        window = cv2.createHanningWindow((prev_gray.shape[1], prev_gray.shape[0]), cv2.CV_32F)
    (sx, sy), response = cv2.phaseCorrelate(prev_gray, cur_gray, window)
    return sx, sy, response


def warp_translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def residual_map(prev_gray: np.ndarray, cur_gray: np.ndarray,
                 dx: float, dy: float, border: int = 16,
                 blur: int = 21) -> np.ndarray:
    """Background-compensated residual (uint8). High where motion != background."""
    warped = warp_translate(prev_gray, dx, dy)
    diff = cv2.absdiff(warped, cur_gray)
    diff = cv2.GaussianBlur(diff, (blur, blur), 0)
    # kill border artifacts from the shift / replicate fill
    if border > 0:
        diff[:border, :] = 0
        diff[-border:, :] = 0
        diff[:, :border] = 0
        diff[:, -border:] = 0
    m = diff.max()
    if m > 0:
        diff = (diff / m * 255.0)
    return diff.astype(np.uint8)
