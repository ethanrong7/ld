"""Step 6: capture target template + rotation rate from the countdown.

During the countdown the target is a clean white blob. We scan the opening
white-present run, reject frames contaminated by the blue digit's glow (area
spikes / off-center blob), build a canonical centered template from the best
clean mask, and fit a constant rotation rate (deg/frame) from the unwrapped
moment-angle. The result seeds the post-countdown tracker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ld.capture.video_source import VideoSource
from ld.vision.countdown import detect_white_shape


@dataclass
class RoundInit:
    template: np.ndarray        # centered binary uint8 mask of the shape
    template_radius: float      # equiv. circle radius of the shape (px)
    center_seed: tuple[float, float]   # last clean white centroid (handoff pos)
    omega_deg_per_frame: float  # fitted constant rotation rate
    handoff_frame: int          # last frame of opening white-present run
    n_clean: int                # how many clean frames contributed


def _unwrap_deg(angles: list[float], max_step: float = 90.0) -> list[float]:
    """Unwrap a sequence of mod-180 degree angles into a continuous curve."""
    out = [angles[0]]
    for a in angles[1:]:
        prev = out[-1]
        d = a - (prev % 180.0)
        while d > max_step:
            d -= 180.0
        while d < -max_step:
            d += 180.0
        out.append(prev + d)
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (e.g. the green-cursor cutout) of a solid shape."""
    filled = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _center_template(mask: np.ndarray, size: int = 160) -> tuple[np.ndarray, float]:
    mask = _fill_holes(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((size, size), np.uint8), 0.0
    cx, cy = xs.mean(), ys.mean()
    area = float(len(xs))
    radius = math.sqrt(area / math.pi)
    canvas = np.zeros((size, size), np.uint8)
    dx = int(round(size / 2 - cx))
    dy = int(round(size / 2 - cy))
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    shifted = cv2.warpAffine(mask, M, (size, size))
    return shifted, radius


def analyze_round(input_path: str, max_scan: int | None = 400) -> RoundInit:
    src = VideoSource(input_path)
    records: list[dict] = []
    handoff = None
    started = False
    for idx, frame in src.frames(max_scan):
        ws = detect_white_shape(frame)
        if ws is not None:
            started = True
            handoff = idx
            records.append({"idx": idx, "ws": ws})
        elif started:
            break
    src.release()

    if not records:
        raise RuntimeError("No countdown white shape found.")

    areas = np.array([r["ws"].area for r in records], float)
    med_area = float(np.median(areas))
    cxs = np.array([r["ws"].cx for r in records])
    cys = np.array([r["ws"].cy for r in records])
    med_cx, med_cy = float(np.median(cxs)), float(np.median(cys))

    # clean = area near median (no digit-glow spike) AND blob near the run center
    clean = [
        r for r in records
        if 0.7 * med_area <= r["ws"].area <= 1.3 * med_area
        and math.hypot(r["ws"].cx - med_cx, r["ws"].cy - med_cy) < 60
        and not math.isnan(r["ws"].angle)
    ]
    if len(clean) < 5:
        clean = records  # fall back

    # rotation fit on the clean window
    idxs = [r["idx"] for r in clean]
    angs = [math.degrees(r["ws"].angle) for r in clean]
    unwrapped = _unwrap_deg(angs)
    slope = float(np.polyfit(idxs, unwrapped, 1)[0]) if len(idxs) >= 2 else 0.0

    # template from the clean mask whose area is closest to median
    best = min(clean, key=lambda r: abs(r["ws"].area - med_area))
    template, radius = _center_template(best["ws"].mask)

    # seed = last clean centroid (closest to handoff)
    seed_rec = clean[-1]
    seed = (seed_rec["ws"].cx, seed_rec["ws"].cy)

    return RoundInit(
        template=template,
        template_radius=radius,
        center_seed=seed,
        omega_deg_per_frame=slope,
        handoff_frame=handoff if handoff is not None else records[-1]["idx"],
        n_clean=len(clean),
    )
