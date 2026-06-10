"""Score a solver track against the green crosshair ground truth.

The green crosshair is read here, and ONLY here, to evaluate accuracy. It is
never fed back into the solver. Accuracy is reported relative to the shape
radius (a hit means the cursor would land on the shape).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ld.capture.video_source import VideoSource
from ld.solve import SolveResult, solve_clip
from ld.vision.cursor import find_cursor

__all__ = ["ScoreReport", "score_clip"]


@dataclass
class ScoreReport:
    clip: str
    n: int
    median_px: float
    mean_px: float
    within_radius: float
    within_1p5_radius: float
    radius: float

    def __str__(self) -> str:
        return (f"{self.clip:28s} n={self.n:4d} median={self.median_px:6.1f}px "
                f"mean={self.mean_px:6.1f}px  within_r={self.within_radius:.2f} "
                f"within_1.5r={self.within_1p5_radius:.2f}  (r={self.radius:.0f}px)")


def _gt_track(path: str | Path) -> dict[int, tuple[float, float]]:
    src = VideoSource(path)
    gt: dict[int, tuple[float, float]] = {}
    for idx, raw in src.frames():
        c = find_cursor(raw)
        if c is not None:
            gt[idx] = c
    src.release()
    return gt


def score_clip(
    path: str | Path,
    result: SolveResult | None = None,
    *,
    radius: float | None = None,
) -> ScoreReport:
    if result is None:
        result = solve_clip(path)
    gt = _gt_track(path)
    r = radius if radius else (result.seed_radius or 55.0)

    errs: list[float] = []
    for tp in result.track:
        if tp.frame < result.start_frame:
            continue  # acquisition phase, position known from white shape
        g = gt.get(tp.frame)
        if g is None or math.isnan(tp.x):
            continue
        errs.append(math.hypot(tp.x - g[0], tp.y - g[1]))

    if not errs:
        return ScoreReport(Path(path).name, 0, float("nan"), float("nan"), 0.0, 0.0, r)

    errs.sort()
    n = len(errs)
    median = errs[n // 2]
    mean = sum(errs) / n
    within_r = sum(e < r for e in errs) / n
    within_1p5 = sum(e < 1.5 * r for e in errs) / n
    return ScoreReport(Path(path).name, n, median, mean, within_r, within_1p5, r)
