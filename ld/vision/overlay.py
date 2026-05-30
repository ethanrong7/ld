"""Debug drawing on frames."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from ld.config import OVERLAY_FONT_SCALE, OVERLAY_FONT_THICKNESS

if TYPE_CHECKING:
    import numpy as np


def draw_hud(
    frame: np.ndarray,
    *,
    frame_idx: int,
    timestamp: float,
    total_frames: int | None = None,
    processing_fps: float | None = None,
    label: str = "Phase 1",
) -> np.ndarray:
    """Draw frame index, timestamp, and optional processing FPS."""
    out = frame.copy()
    h, w = out.shape[:2]
    lines = [
        label,
        f"frame {frame_idx}" + (f" / {total_frames}" if total_frames else ""),
        f"t={timestamp:.3f}s",
    ]
    if processing_fps is not None:
        lines.append(f"proc {processing_fps:.1f} fps")

    y = 24
    for line in lines:
        cv2.putText(
            out,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            OVERLAY_FONT_SCALE,
            (255, 255, 255),
            OVERLAY_FONT_THICKNESS + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            OVERLAY_FONT_SCALE,
            (0, 0, 0),
            OVERLAY_FONT_THICKNESS,
            cv2.LINE_AA,
        )
        y += int(22 * OVERLAY_FONT_SCALE + 12)

    # subtle border so HUD is visible on sandy background
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (40, 40, 40), 1)
    return out
