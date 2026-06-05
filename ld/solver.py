"""End-to-end LD solver: countdown init -> seeded target tracking.

The solver is pointer-blind: frames are preprocessed with ``strip_pointer`` before
grayscale (green GT on labelled clips; optional live mouse disk). The green cursor
is never used as a measurement — only for scoring in debug callbacks on the raw
frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ld.capture.video_source import VideoSource
from ld.vision.cursor import PointerAt, strip_pointer
from ld.vision.motion import to_gray_f32
from ld.vision.template import RoundInit, analyze_round
from ld.vision.tracker import TargetTracker, TrackState

OnFrame = Callable[[int, np.ndarray, TrackState], None]


@dataclass
class SolveResult:
    init: RoundInit
    states: list[tuple[int, TrackState]]   # (frame_idx, state) for tracked frames


def track_video(
    input_path: str,
    on_frame: OnFrame | None = None,
    init: RoundInit | None = None,
    *,
    strip_pointer_from_frames: bool = True,
    mouse_at: PointerAt | None = None,
) -> SolveResult:
    """Run the solver on a video file.

    ``strip_pointer_from_frames``: inpaint green GT / live mouse before tracking
    (default True for all pathways). Set False only for ablation comparisons.

    ``mouse_at``: per-frame live mouse ``(x, y)`` in frame coordinates; when
    provided, that disk is inpainted in addition to any green GT pixels.
    """
    ri = init or analyze_round(
        input_path, strip_pointer_from_frames=strip_pointer_from_frames
    )
    src = VideoSource(input_path)
    tracker: TargetTracker | None = None
    prev_gray = None
    start = ri.handoff_frame + 1
    states: list[tuple[int, TrackState]] = []

    for idx, frame in src.frames():
        raw = frame
        if strip_pointer_from_frames:
            xy = mouse_at(idx) if mouse_at is not None else None
            frame = strip_pointer(frame, mouse_xy=xy)
        if idx < start:
            prev_gray = to_gray_f32(frame)
            continue
        cur_gray = to_gray_f32(frame)
        if prev_gray is None:
            prev_gray = cur_gray
            continue
        if tracker is None:
            tracker = TargetTracker(ri, prev_gray, appearance_gray=cur_gray)

        st = tracker.step(prev_gray, cur_gray)
        states.append((idx, st))
        if on_frame is not None:
            on_frame(idx, raw, st)
        prev_gray = cur_gray

    src.release()
    return SolveResult(init=ri, states=states)
