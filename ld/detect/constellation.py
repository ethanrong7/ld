"""Constellation-tracking prototype: detections -> real-shape track -> score.

This is the *kill-shot* test for the detector pivot. A detector that boxes
shapes is worthless unless those boxes can be turned into the real shape's
position. This module does exactly that and scores it against the green GT, so
we learn -- cheaply, no new labels -- whether the approach can actually localize
the real shape better than the ~50% motion-saliency baseline.

Pipeline (causal; ports to live):
  1. Per-frame pass (cached): strip the pointer, run YOLO to get shape-box
     centres, read the green GT (scoring only), and detect the white countdown
     shape (identity seed). Cached to JSON so re-runs of the tracker are instant.
  2. Seed: the real shape's start position is the last countdown white-shape
     centroid (no crosshair involved), handed off when it blends.
  3. Track: among detections, the fakes move as one rigid sheet; the real shape
     moves independently. Each frame we (a) fit the rigid prev->curr transform
     of the detection cloud (RANSAC; the fakes dominate), (b) gate detections
     around the constant-velocity prediction, and (c) pick the gated detection
     that best combines proximity-to-prediction with *deviation from the rigid
     transform* (the independent-motion cue, now at the clean detection level
     instead of noisy pixel flow). Coast on the motion model when nothing gates.
  4. Score: inferred position vs green GT, within-radius like eval/score.

Usage:
    python -m ld.detect.constellation --weights data/detect/runs/yolov8n_probe/weights/best.pt
    python -m ld.detect.constellation --weights .../best.pt --inputs data/t1_cropped_trimmed.mp4
    python -m ld.detect.constellation --weights .../best.pt --evidence   # overlay videos
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.config import DATA_DIR, DETECT_DIR, GATE_RADIUS
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import find_cursor, strip_pointer

__all__ = ["FramePack", "detect_clip", "track_clip", "score_track", "run_clip"]

CACHE_DIR = DETECT_DIR / "cache"

# Tracker tunables (prototype; deliberately simple to read the signal cleanly).
UPDATE_ALPHA = 0.5      # blend prediction toward the chosen detection
VEL_DAMP = 0.7
VEL_MAX = 8.0
RIGID_RANSAC_THRESH = 3.0   # px; inlier tol for the sheet (fake) transform
OUTLIER_BONUS = 0.5         # weight on independent-motion deviation in scoring
COAST_PATIENCE = 12         # frames coasting before trusting raw proximity again


@dataclass
class FramePack:
    """Cached per-frame facts for one clip."""

    idx: int
    gt: tuple[float, float] | None              # green crosshair (scoring only)
    white: tuple[float, float, float] | None    # countdown shape (cx,cy,r)
    dets: list[tuple[float, float, float]]       # detection centres (x,y,conf)


# --------------------------------------------------------------------------- #
# 1. Per-frame detection pass (cached)
# --------------------------------------------------------------------------- #
def _weights_tag(weights: Path) -> str:
    st = weights.stat()
    h = hashlib.md5(f"{weights.resolve()}:{st.st_mtime_ns}".encode()).hexdigest()[:10]
    return h


def _cache_path(clip: Path, weights: Path, conf: float, imgsz: int) -> Path:
    return CACHE_DIR / f"{clip.stem}_{_weights_tag(weights)}_c{conf}_s{imgsz}.json"


def detect_clip(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
                imgsz: int = 768, use_cache: bool = True) -> list[FramePack]:
    weights, clip = Path(weights), Path(clip)
    cache = _cache_path(clip, weights, conf, imgsz)
    if use_cache and cache.exists():
        raw = json.loads(cache.read_text())
        return [FramePack(p["idx"],
                          tuple(p["gt"]) if p["gt"] else None,
                          tuple(p["white"]) if p["white"] else None,
                          [tuple(d) for d in p["dets"]]) for p in raw]

    from ultralytics import YOLO  # lazy heavy dep

    model = YOLO(str(weights))
    packs: list[FramePack] = []
    src = VideoSource(clip)
    for idx, frame in src.frames():
        gt = find_cursor(frame)
        stripped = strip_pointer(frame, strip_green=True)
        ws = detect_white_shape(stripped)
        res = model.predict(stripped, conf=conf, imgsz=imgsz, verbose=False)[0]
        dets: list[tuple[float, float, float]] = []
        if res.boxes is not None and len(res.boxes):
            xywh = res.boxes.xywh.cpu().numpy()
            cfs = res.boxes.conf.cpu().numpy()
            for (cx, cy, _w, _h), cf in zip(xywh, cfs):
                dets.append((float(cx), float(cy), float(cf)))
        packs.append(FramePack(idx, gt,
                               (ws.cx, ws.cy, ws.radius) if ws else None, dets))
    src.release()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([
        {"idx": p.idx, "gt": list(p.gt) if p.gt else None,
         "white": list(p.white) if p.white else None,
         "dets": [list(d) for d in p.dets]} for p in packs]))
    return packs


# --------------------------------------------------------------------------- #
# 2/3. Seed + constellation tracker
# --------------------------------------------------------------------------- #
@dataclass
class TrackPoint:
    idx: int
    x: float
    y: float
    state: str  # "acquire" | "track" | "coast"
    n_dets: int


def _seed(packs: list[FramePack], miss_to_confirm: int = 3) -> tuple[float, float, float, int]:
    """Return (cx, cy, radius, start_frame) from the countdown white shape."""
    last = None
    last_idx = -1
    radii: list[float] = []
    misses = 0
    for p in packs:
        if p.white is not None:
            last = p.white
            last_idx = p.idx
            radii.append(p.white[2])
            misses = 0
        elif last is not None:
            misses += 1
            if misses >= miss_to_confirm:
                break
    if last is None:
        return (float("nan"), float("nan"), 55.0, 0)
    return (last[0], last[1], float(np.median(radii)), last_idx + 1)


def _rigid_residuals(prev: list[tuple], curr: list[tuple]
                     ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit similarity transform prev->curr over NN matches; return (T, curr_pts).

    The fakes dominate the cloud, so RANSAC locks onto the sheet motion and the
    real shape surfaces as a high-residual point.
    """
    if len(prev) < 3 or len(curr) < 3:
        return None, None
    p = np.array([[x, y] for x, y, _ in prev], np.float32)
    c = np.array([[x, y] for x, y, _ in curr], np.float32)
    # Nearest-neighbour correspondence prev->curr (greedy, good enough for a probe).
    d2 = ((p[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    src_pts = p
    dst_pts = c[j]
    T, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                       ransacReprojThreshold=RIGID_RANSAC_THRESH)
    return T, c


def _deviation(T: np.ndarray | None, prev_pos: np.ndarray,
               det_xy: np.ndarray) -> float:
    """How far a detection sits from where the rigid sheet would carry prev_pos."""
    if T is None:
        return 0.0
    mapped = T[:, :2] @ prev_pos + T[:, 2]
    return float(np.hypot(*(det_xy - mapped)))


def track_clip(packs: list[FramePack], *, gate_radius: float | None = None,
               frame_wh: tuple[int, int] | None = None
               ) -> tuple[list[TrackPoint], int, float]:
    cx, cy, radius, start = _seed(packs)
    gate = gate_radius or max(GATE_RADIUS, 1.3 * radius)
    pos = np.array([cx, cy], np.float32)
    vel = np.zeros(2, np.float32)
    lost = 0
    prev_dets: list[tuple] | None = None
    w, h = frame_wh if frame_wh else (None, None)

    track: list[TrackPoint] = []
    for p in packs:
        if p.idx < start or math.isnan(pos[0]):
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]),
                                    "acquire", len(p.dets)))
            prev_dets = p.dets
            continue

        pred = pos + vel
        T, _ = _rigid_residuals(prev_dets or [], p.dets)

        best = None
        best_score = -1e9
        for (dx, dy, _cf) in p.dets:
            det = np.array([dx, dy], np.float32)
            dist = float(np.hypot(*(det - pred)))
            if dist > gate:
                continue
            prox = 1.0 - dist / gate                    # in [0,1]
            dev = _deviation(T, pos, det)               # px; real shape => larger
            score = prox + OUTLIER_BONUS * math.tanh(dev / max(radius, 1.0))
            if score > best_score:
                best_score, best = score, det

        if best is not None:
            new = (1.0 - UPDATE_ALPHA) * pred + UPDATE_ALPHA * best
            lost = 0
            state = "track"
        else:
            new = pred
            lost += 1
            state = "coast"
            if lost >= COAST_PATIENCE:
                vel[:] = 0.0   # stop coasting blindly once the target is long gone

        vel = VEL_DAMP * vel + (1.0 - VEL_DAMP) * (new - pos)
        sp = float(np.hypot(*vel))
        if sp > VEL_MAX:
            vel *= VEL_MAX / sp
        pos = new
        if w is not None:           # keep the estimate on-frame
            pos = np.array([float(np.clip(pos[0], 0, w - 1)),
                            float(np.clip(pos[1], 0, h - 1))], np.float32)
        track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), state, len(p.dets)))
        prev_dets = p.dets

    return track, start, radius


# --------------------------------------------------------------------------- #
# 4. Scoring vs green GT
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    clip: str
    n: int
    median_px: float
    within_r: float
    within_1p5r: float
    radius: float
    mean_dets: float
    oracle_within_r: float   # ceiling: nearest detection to GT (perfect association)

    def __str__(self) -> str:
        return (f"{self.clip:10s} n={self.n:4d} median={self.median_px:6.1f}px "
                f"within_r={self.within_r:.2f} within_1.5r={self.within_1p5r:.2f} "
                f"| oracle_within_r={self.oracle_within_r:.2f} "
                f"(r={self.radius:.0f}, dets/frame={self.mean_dets:.1f})")


def score_track(packs: list[FramePack], track: list[TrackPoint], start: int,
                radius: float, clip_name: str) -> Report:
    gt = {p.idx: p.gt for p in packs if p.gt is not None}
    errs: list[float] = []
    for tp in track:
        if tp.idx < start:
            continue
        g = gt.get(tp.idx)
        if g is None or math.isnan(tp.x):
            continue
        errs.append(math.hypot(tp.x - g[0], tp.y - g[1]))
    mean_dets = float(np.mean([len(p.dets) for p in packs])) if packs else 0.0

    # Oracle ceiling: if association were perfect, how often is *some* detection
    # within radius of the real shape? Decouples detector coverage from tracking.
    oracle: list[float] = []
    for p in packs:
        if p.idx < start or p.gt is None or not p.dets:
            continue
        oracle.append(min(math.hypot(dx - p.gt[0], dy - p.gt[1]) for dx, dy, _ in p.dets))
    oracle_wr = (sum(e < radius for e in oracle) / len(oracle)) if oracle else 0.0

    if not errs:
        return Report(clip_name, 0, float("nan"), 0.0, 0.0, radius, mean_dets, oracle_wr)
    errs.sort()
    n = len(errs)
    return Report(clip_name, n, errs[n // 2],
                  sum(e < radius for e in errs) / n,
                  sum(e < 1.5 * radius for e in errs) / n, radius, mean_dets, oracle_wr)


def _render_evidence(clip: Path, packs: list[FramePack], track: list[TrackPoint],
                     out: Path) -> None:
    tp_by_idx = {t.idx: t for t in track}
    src = VideoSource(clip)
    writer = open_writer(out, src.meta.width, src.meta.height, src.meta.fps or 30.0)
    for idx, frame in src.frames():
        p = packs[idx] if idx < len(packs) else None
        if p is not None:
            for (dx, dy, _c) in p.dets:
                cv2.circle(frame, (int(dx), int(dy)), 4, (255, 180, 0), -1)
            if p.gt is not None:
                cv2.drawMarker(frame, (int(p.gt[0]), int(p.gt[1])), (0, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 18, 2)
        tp = tp_by_idx.get(idx)
        if tp is not None and not math.isnan(tp.x):
            col = {"track": (0, 0, 255), "coast": (0, 140, 255)}.get(tp.state, (200, 200, 200))
            cv2.circle(frame, (int(tp.x), int(tp.y)), 10, col, 2)
        writer.write(frame)
    writer.release()
    src.release()


def run_clip(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
             imgsz: int = 768, use_cache: bool = True, evidence: bool = False) -> Report:
    clip = Path(clip)
    packs = detect_clip(weights, clip, conf=conf, imgsz=imgsz, use_cache=use_cache)
    src = VideoSource(clip)
    frame_wh = (src.meta.width, src.meta.height)
    src.release()
    track, start, radius = track_clip(packs, frame_wh=frame_wh)
    report = score_track(packs, track, start, radius, clip.stem.replace("_cropped_trimmed", ""))
    if evidence:
        out = DETECT_DIR / "evidence" / f"{clip.stem}_constellation.mp4"
        _render_evidence(clip, packs, track, out)
        print(f"  evidence -> {out}")
    return report


def _default_inputs() -> list[Path]:
    return sorted(DATA_DIR.glob("t*_cropped_trimmed.mp4"),
                  key=lambda p: int(p.stem.split("_")[0][1:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Constellation-tracking prototype + scoring")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--evidence", action="store_true", help="write overlay videos")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs] if args.inputs else _default_inputs()
    if not inputs:
        raise SystemExit("No input clips found.")

    reports: list[Report] = []
    for clip in inputs:
        print(f"[{clip.stem}] detecting + tracking ...")
        rep = run_clip(args.weights, clip, conf=args.conf, imgsz=args.imgsz,
                       use_cache=not args.no_cache, evidence=args.evidence)
        print("  ", rep)
        reports.append(rep)

    valid = [r for r in reports if r.n > 0]
    if valid:
        wr = sum(r.within_r for r in valid) / len(valid)
        w15 = sum(r.within_1p5r for r in valid) / len(valid)
        orc = sum(r.oracle_within_r for r in valid) / len(valid)
        med = float(np.median([r.median_px for r in valid]))
        print(f"\nMEAN within_r={wr:.3f}  within_1.5r={w15:.3f}  median≈{med:.1f}px"
              f"\nMEAN oracle_within_r={orc:.3f}  <- detector ceiling; if this is low,"
              f" it's a detector/data problem, not a tracking one"
              f"\n(baseline motion-saliency ≈0.50 within_r)")


if __name__ == "__main__":
    main()
