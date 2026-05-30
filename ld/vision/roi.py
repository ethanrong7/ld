"""Crop / normalize frame to LD panel region."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def apply_roi(frame: np.ndarray, *, pre_cropped: bool = True) -> np.ndarray:
    """
    Return LD panel region.

    Pre-cropped test clips (t*_cropped_trimmed.mp4) use the full frame as the panel.
    """
    if pre_cropped:
        return frame
    # Full-screen gameplay: tune fractional crop in a later phase.
    h, w = frame.shape[:2]
    y1, y2 = int(h * 0.14), int(h * 0.70)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    return frame[y1:y2, x1:x2]
