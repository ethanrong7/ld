"""End-to-end solver: acquire from the countdown, then track the real shape.

Pipeline (causal, so it ports to live):
  1. Acquisition - follow the white countdown shape to get a seed position and
     scale; the handoff is when it blends into the sheet.
  2. Tracking    - per frame, estimate the rigid sheet motion, build the
     independent-motion saliency, and update the recursive tracker.

The green crosshair is stripped from every frame (``strip_pointer``) and is
never an input. In live use, pass ``mouse_at`` to inpaint the real OS cursor
instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ld.capture.video_source import VideoMeta, VideoSource
from ld.config import GATE_RADIUS
from ld.track import OutlierTracker, TrackPoint
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import strip_pointer
from ld.vision.motion import estimate_motion, saliency_map

__all__ = ["SolveResult", "solve_clip", "write_track_csv"]

MouseAt = Callable[[int], "tuple[float, float] | None"]


@dataclass
class SolveResult:
    track: list[TrackPoint]
    meta: VideoMeta
    start_frame: int
    seed_radius: float


def solve_clip(
    path: str | Path,
    *,
    strip_green: bool = True,
    mouse_at: MouseAt | None = None,
    miss_to_confirm: int = 3,
) -> SolveResult:
    src = VideoSource(path)
    meta = src.meta

    track: list[TrackPoint] = []
    prev_gray: np.ndarray | None = None
    tracker: OutlierTracker | None = None
    last_white = None
    radii: list[float] = []
    misses = 0
    started = False
    start_frame = 0
    seed_radius = 0.0

    for idx, raw in src.frames():
        mouse = mouse_at(idx) if mouse_at else None
        frame = strip_pointer(raw, strip_green=strip_green, mouse_xy=mouse)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not started:
            ws = detect_white_shape(frame)
            if ws is not None:
                last_white = ws
                radii.append(ws.radius)
                misses = 0
                track.append(TrackPoint(idx, ws.cx, ws.cy, 1.0, "acquire"))
            elif last_white is not None:
                misses += 1
                track.append(TrackPoint(idx, last_white.cx, last_white.cy, 0.0, "acquire"))
                if misses >= miss_to_confirm:
                    seed_radius = float(np.median(radii)) if radii else 0.0
                    gate = max(GATE_RADIUS, 1.3 * seed_radius)
                    tracker = OutlierTracker(last_white.cx, last_white.cy, gate_radius=gate)
                    started = True
                    start_frame = idx + 1
            else:
                track.append(TrackPoint(idx, float("nan"), float("nan"), 0.0, "search"))
            prev_gray = gray
            continue

        if prev_gray is None:
            track.append(TrackPoint(idx, tracker.pos[0], tracker.pos[1], 0.0, "coast"))
            prev_gray = gray
            continue

        field = estimate_motion(prev_gray, gray)
        sal = saliency_map(field, gray.shape)
        track.append(tracker.update(idx, sal))
        prev_gray = gray

    src.release()
    return SolveResult(track, meta, start_frame, seed_radius)


def write_track_csv(result: SolveResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("frame,x,y,confidence,state\n")
        for tp in result.track:
            f.write(f"{tp.frame},{tp.x:.2f},{tp.y:.2f},{tp.confidence:.3f},{tp.state}\n")
    return path
