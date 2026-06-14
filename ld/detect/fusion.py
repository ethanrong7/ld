"""YOLO + classical motion fusion experiment.

Tier 1: per-box independent-motion saliency -- score saliency inside each YOLO
        ROI; pick the box with the strongest signal.
Tier 2: focus mask -- zero saliency outside the union of detection boxes.
Tier 3: suppress rigid inliers -- drop boxes whose centroids move with the
        fitted sheet transform (locked fakes).

Compares three causal pipelines on the same clip:
  A) classical motion only (ld.solve)
  B) YOLO constellation only (centroid rigid residual)
  C) fusion (per-box saliency among YOLO candidates)

Reports within_r, oracle (detector ceiling), and conditional_within_r
(fusion accuracy on frames where *some* detection is already on the real shape).

Usage:
    python -m ld.detect.fusion --weights data/detect/runs/yolov8n_probe/weights/best.pt
    python -m ld.detect.fusion --weights .../best.pt --inputs data/t1_cropped_trimmed.mp4
    python -m ld.detect.fusion --weights .../best.pt --evidence
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.config import DATA_DIR, DETECT_DIR, GATE_RADIUS
from ld.detect.constellation import (
    CACHE_DIR,
    TrackPoint,
    _seed,
    detect_clip,
    score_track,
    track_clip,
)
from ld.eval.score import score_clip
from ld.solve import solve_clip
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import find_cursor, strip_pointer
from ld.vision.motion import estimate_motion, saliency_map

__all__ = ["FusionPack", "detect_fusion_clip", "track_fusion", "run_scoreboard"]

FUSION_CACHE_DIR = DETECT_DIR / "cache" / "fusion"

# Tracker (shared spirit with constellation).
UPDATE_ALPHA = 0.5
VEL_DAMP = 0.7
VEL_MAX = 8.0
COAST_PATIENCE = 12
RIGID_RANSAC_THRESH = 3.0
INLIER_RESID_MAX = 2.5       # px; centroid moves with sheet => suppress


@dataclass
class FusionPack:
    idx: int
    gt: tuple[float, float] | None
    white: tuple[float, float, float] | None
    boxes: list[tuple[float, float, float, float, float]]  # x1,y1,x2,y2,conf

    @property
    def dets(self) -> list[tuple[float, float, float]]:
        return [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[4]) for b in self.boxes]


@dataclass
class FusionReport:
    clip: str
    within_r: float
    within_1p5r: float
    median_px: float
    oracle_within_r: float
    conditional_within_r: float   # within_r on oracle-hit frames only
    n: int
    n_oracle_hit: int

    def __str__(self) -> str:
        return (f"within_r={self.within_r:.3f}  within_1.5r={self.within_1p5r:.3f}  "
                f"median={self.median_px:.1f}px  oracle={self.oracle_within_r:.3f}  "
                f"conditional={self.conditional_within_r:.3f} "
                f"(n={self.n}, oracle_frames={self.n_oracle_hit})")


def _weights_tag(weights: Path) -> str:
    st = weights.stat()
    return hashlib.md5(f"{weights.resolve()}:{st.st_mtime_ns}".encode()).hexdigest()[:10]


def _fusion_cache(clip: Path, weights: Path, conf: float, imgsz: int) -> Path:
    return FUSION_CACHE_DIR / f"{clip.stem}_{_weights_tag(weights)}_c{conf}_s{imgsz}.json"


def detect_fusion_clip(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
                       imgsz: int = 768, use_cache: bool = True) -> list[FusionPack]:
    weights, clip = Path(weights), Path(clip)
    cache = _fusion_cache(clip, weights, conf, imgsz)
    if use_cache and cache.exists():
        raw = json.loads(cache.read_text())
        return [FusionPack(p["idx"],
                           tuple(p["gt"]) if p["gt"] else None,
                           tuple(p["white"]) if p["white"] else None,
                           [tuple(b) for b in p["boxes"]]) for p in raw]

    from ultralytics import YOLO

    model = YOLO(str(weights))
    packs: list[FusionPack] = []
    src = VideoSource(clip)
    for idx, frame in src.frames():
        gt = find_cursor(frame)
        stripped = strip_pointer(frame, strip_green=True)
        ws = detect_white_shape(stripped)
        res = model.predict(stripped, conf=conf, imgsz=imgsz, verbose=False)[0]
        boxes: list[tuple[float, float, float, float, float]] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cfs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cf in zip(xyxy, cfs):
                boxes.append((float(x1), float(y1), float(x2), float(y2), float(cf)))
        packs.append(FusionPack(idx, gt,
                                (ws.cx, ws.cy, ws.radius) if ws else None, boxes))
    src.release()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([
        {"idx": p.idx, "gt": list(p.gt) if p.gt else None,
         "white": list(p.white) if p.white else None,
         "boxes": [list(b) for b in p.boxes]} for p in packs]))
    return packs


def _box_saliency(sal: np.ndarray, box: tuple[float, float, float, float, float]) -> float:
    """Mean independent-motion saliency inside a detection ROI."""
    h, w = sal.shape
    x1 = int(max(0, math.floor(box[0])))
    y1 = int(max(0, math.floor(box[1])))
    x2 = int(min(w, math.ceil(box[2])))
    y2 = int(min(h, math.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = sal[y1:y2, x1:x2]
    return float(patch.mean()) if patch.size else 0.0


def _focus_mask(boxes: list[tuple], shape: tuple[int, int]) -> np.ndarray:
    """Tier 2: binary mask = union of detection boxes."""
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2, _ in boxes:
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, -1)
    return mask


def _rigid_inlier_flags(prev: list[tuple], curr: list[tuple]) -> list[bool]:
    """Tier 3: True = centroid moves with the rigid sheet (locked fake)."""
    if len(prev) < 3 or len(curr) < 3:
        return [False] * len(curr)
    p = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in prev], np.float32)
    c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in curr], np.float32)
    d2 = ((p[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    T, inlier = cv2.estimateAffinePartial2D(
        p, c[j], method=cv2.RANSAC, ransacReprojThreshold=RIGID_RANSAC_THRESH)
    if T is None:
        return [False] * len(curr)
    # Per-curr-box residual from nearest prev match under T.
    flags: list[bool] = []
    for i, pt in enumerate(c):
        src_i = int(np.argmin(((p - pt) ** 2).sum(1)))
        mapped = T[:, :2] @ p[src_i] + T[:, 2]
        resid = float(np.hypot(*(pt - mapped)))
        flags.append(resid < INLIER_RESID_MAX)
    return flags


def track_fusion(clip: str | Path, packs: list[FusionPack], *,
                 frame_wh: tuple[int, int] | None = None,
                 gate_radius: float | None = None) -> tuple[list[TrackPoint], int, float]:
    """Tier 1+2+3 fusion tracker over cached detection boxes + live motion."""
    cx, cy, radius, start = _seed(packs)  # type: ignore[arg-type]
    gate = gate_radius or max(GATE_RADIUS, 1.3 * radius)
    pos = np.array([cx, cy], np.float32)
    vel = np.zeros(2, np.float32)
    lost = 0
    prev_boxes: list[tuple] | None = None
    fw, fh = frame_wh if frame_wh else (None, None)

    track: list[TrackPoint] = []
    prev_gray: np.ndarray | None = None
    src = VideoSource(clip)

    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        stripped = strip_pointer(raw, strip_green=True)
        gray = cv2.cvtColor(stripped, cv2.COLOR_BGR2GRAY)

        if p.idx < start or math.isnan(pos[0]) or prev_gray is None:
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]),
                                    "acquire", len(p.boxes)))
            prev_gray = gray
            prev_boxes = p.boxes
            continue

        field = estimate_motion(prev_gray, gray)
        sal = saliency_map(field, gray.shape)

        # Tier 2: only saliency on/near embossed shapes.
        if p.boxes:
            sal = sal * _focus_mask(p.boxes, gray.shape)

        pred = pos + vel
        inlier_flags = _rigid_inlier_flags(prev_boxes or [], p.boxes)

        best_xy = None
        best_score = -1e9
        for box, is_inlier in zip(p.boxes, inlier_flags):
            if is_inlier:          # Tier 3: skip locked fakes
                continue
            cx_b = (box[0] + box[2]) / 2
            cy_b = (box[1] + box[3]) / 2
            det = np.array([cx_b, cy_b], np.float32)
            dist = float(np.hypot(*(det - pred)))
            if dist > gate:
                continue
            sal_score = _box_saliency(sal, box)
            prox = 1.0 - dist / gate
            score = sal_score + 0.25 * prox
            if score > best_score:
                best_score, best_xy = score, det

        if best_xy is not None and best_score > 0:
            new = (1.0 - UPDATE_ALPHA) * pred + UPDATE_ALPHA * best_xy
            lost = 0
            state = "track"
        else:
            new = pred
            lost += 1
            state = "coast"
            if lost >= COAST_PATIENCE:
                vel[:] = 0.0

        vel = VEL_DAMP * vel + (1.0 - VEL_DAMP) * (new - pos)
        sp = float(np.hypot(*vel))
        if sp > VEL_MAX:
            vel *= VEL_MAX / sp
        pos = new
        if fw is not None:
            pos = np.array([float(np.clip(pos[0], 0, fw - 1)),
                            float(np.clip(pos[1], 0, fh - 1))], np.float32)
        track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), state, len(p.boxes)))
        prev_gray = gray
        prev_boxes = p.boxes

    src.release()
    return track, start, radius


def score_fusion(packs: list[FusionPack], track: list[TrackPoint], start: int,
                 radius: float, clip_name: str) -> FusionReport:
    gt = {p.idx: p.gt for p in packs if p.gt is not None}
    errs: list[float] = []
    oracle_hit_frames: list[float] = []

    for tp in track:
        if tp.idx < start:
            continue
        g = gt.get(tp.idx)
        if g is None or math.isnan(tp.x):
            continue
        err = math.hypot(tp.x - g[0], tp.y - g[1])
        errs.append(err)

        p = packs[tp.idx]
        if p.boxes:
            oracle_err = min(
                math.hypot((b[0] + b[2]) / 2 - g[0], (b[1] + b[3]) / 2 - g[1])
                for b in p.boxes)
            if oracle_err < radius:
                oracle_hit_frames.append(err)

    oracle: list[float] = []
    for p in packs:
        if p.idx < start or p.gt is None or not p.boxes:
            continue
        oracle.append(min(
            math.hypot((b[0] + b[2]) / 2 - p.gt[0], (b[1] + b[3]) / 2 - p.gt[1])
            for b in p.boxes))
    oracle_wr = (sum(e < radius for e in oracle) / len(oracle)) if oracle else 0.0

    if not errs:
        return FusionReport(clip_name, 0.0, 0.0, float("nan"), oracle_wr, 0.0, 0, 0)
    errs.sort()
    n = len(errs)
    cond = (sum(e < radius for e in oracle_hit_frames) / len(oracle_hit_frames)
            if oracle_hit_frames else 0.0)
    return FusionReport(
        clip_name,
        sum(e < radius for e in errs) / n,
        sum(e < 1.5 * radius for e in errs) / n,
        errs[n // 2],
        oracle_wr,
        cond,
        n,
        len(oracle_hit_frames),
    )


def _render_evidence(clip: Path, packs: list[FusionPack], track: list[TrackPoint],
                     out: Path) -> None:
    tp_by_idx = {t.idx: t for t in track}
    src = VideoSource(clip)
    writer = open_writer(out, src.meta.width, src.meta.height, src.meta.fps or 30.0)
    prev_gray: np.ndarray | None = None
    prev_boxes: list[tuple] | None = None

    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        frame = strip_pointer(raw, strip_green=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        inlier_flags: list[bool] = []
        if prev_gray is not None and p.boxes:
            field = estimate_motion(prev_gray, gray)
            sal = saliency_map(field, gray.shape) * _focus_mask(p.boxes, gray.shape)
            sal_vis = (sal / sal.max() * 255).astype(np.uint8) if sal.max() > 0 else sal.astype(np.uint8)
            frame = cv2.addWeighted(frame, 0.65, cv2.applyColorMap(sal_vis, cv2.COLORMAP_HOT), 0.35, 0)
            inlier_flags = _rigid_inlier_flags(prev_boxes or [], p.boxes)

        for box, inl in zip(p.boxes, inlier_flags or [False] * len(p.boxes)):
            col = (80, 80, 255) if inl else (255, 180, 0)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 2)
        if p.gt is not None:
            cv2.drawMarker(frame, (int(p.gt[0]), int(p.gt[1])), (0, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 18, 2)
        tp = tp_by_idx.get(idx)
        if tp is not None and not math.isnan(tp.x):
            col = {"track": (0, 0, 255), "coast": (0, 140, 255)}.get(tp.state, (180, 180, 180))
            cv2.circle(frame, (int(tp.x), int(tp.y)), 10, col, 2)
        writer.write(frame)
        prev_gray = gray
        prev_boxes = p.boxes

    writer.release()
    src.release()


@dataclass
class Scoreboard:
    clip: str
    classical: FusionReport
    constellation: FusionReport
    fusion: FusionReport


def run_scoreboard(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
                   imgsz: int = 768, use_cache: bool = True,
                   evidence: bool = False) -> Scoreboard:
    clip = Path(clip)
    name = clip.stem.replace("_cropped_trimmed", "")

    # A) classical motion
    print(f"  [{name}] A) classical motion ...")
    c_score = score_clip(clip)
    classical = FusionReport(
        name, c_score.within_radius, c_score.within_1p5_radius, c_score.median_px,
        float("nan"), float("nan"), c_score.n, 0)

    # B) constellation (reuse FramePack cache)
    print(f"  [{name}] B) YOLO constellation ...")
    packs_b = detect_clip(weights, clip, conf=conf, imgsz=imgsz, use_cache=use_cache)
    src = VideoSource(clip)
    wh = (src.meta.width, src.meta.height)
    src.release()
    track_b, start_b, radius_b = track_clip(packs_b, frame_wh=wh)
    rep_b = score_track(packs_b, track_b, start_b, radius_b, name)
    cond_b = _conditional_from_packs(packs_b, track_b, start_b, radius_b)
    constellation = FusionReport(
        name, rep_b.within_r, rep_b.within_1p5r, rep_b.median_px,
        rep_b.oracle_within_r, cond_b, rep_b.n, _n_oracle(packs_b, start_b, radius_b))

    # C) fusion
    print(f"  [{name}] C) YOLO + motion fusion ...")
    packs_c = detect_fusion_clip(weights, clip, conf=conf, imgsz=imgsz, use_cache=use_cache)
    track_c, start_c, radius_c = track_fusion(clip, packs_c, frame_wh=wh)
    fusion = score_fusion(packs_c, track_c, start_c, radius_c, name)
    if evidence:
        out = DETECT_DIR / "evidence" / f"{clip.stem}_fusion.mp4"
        _render_evidence(clip, packs_c, track_c, out)
        print(f"    evidence -> {out}")

    return Scoreboard(name, classical, constellation, fusion)


def _n_oracle(packs, start: int, radius: float) -> int:
    n = 0
    for p in packs:
        if p.idx < start or p.gt is None or not p.dets:
            continue
        if min(math.hypot(dx - p.gt[0], dy - p.gt[1]) for dx, dy, _ in p.dets) < radius:
            n += 1
    return n


def _conditional_from_packs(packs, track: list[TrackPoint], start: int, radius: float) -> float:
    gt = {p.idx: p.gt for p in packs if p.gt is not None}
    tp_by = {t.idx: t for t in track}
    hits: list[float] = []
    for p in packs:
        if p.idx < start or p.gt is None or not p.dets:
            continue
        oracle_err = min(math.hypot(dx - p.gt[0], dy - p.gt[1]) for dx, dy, _ in p.dets)
        if oracle_err >= radius:
            continue
        tp = tp_by.get(p.idx)
        if tp is None or math.isnan(tp.x):
            continue
        hits.append(math.hypot(tp.x - p.gt[0], tp.y - p.gt[1]))
    if not hits:
        return 0.0
    return sum(e < radius for e in hits) / len(hits)


def _print_scoreboard(sb: Scoreboard) -> None:
    print(f"\n{'='*72}")
    print(f"SCOREBOARD: {sb.clip}")
    print(f"{'='*72}")
    print(f"  A) Classical motion     {sb.classical}")
    print(f"  B) YOLO constellation    {sb.constellation}")
    print(f"  C) YOLO + motion fusion  {sb.fusion}")
    print()
    print("  conditional = within_r on frames where some detection is already on GT")
    print("  (measures association quality, decoupled from detector recall)")


def _default_inputs() -> list[Path]:
    return sorted(DATA_DIR.glob("t*_cropped_trimmed.mp4"),
                  key=lambda p: int(p.stem.split("_")[0][1:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-way fusion experiment scoreboard")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--evidence", action="store_true")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs] if args.inputs else _default_inputs()
    if not inputs:
        raise SystemExit("No input clips found.")

    boards: list[Scoreboard] = []
    for clip in inputs:
        print(f"\n[{clip.stem}]")
        boards.append(run_scoreboard(
            args.weights, clip, conf=args.conf, imgsz=args.imgsz,
            use_cache=not args.no_cache, evidence=args.evidence))
        _print_scoreboard(boards[-1])

    if len(boards) > 1:
        def mean_attr(attr: str, sub: str) -> float:
            vals = [getattr(getattr(b, sub), attr) for b in boards
                    if not math.isnan(getattr(getattr(b, sub), attr))]
            return sum(vals) / len(vals) if vals else float("nan")

        print(f"\n{'='*72}")
        print(f"MEAN ACROSS {len(boards)} CLIPS")
        print(f"  A) classical     within_r={mean_attr('within_r', 'classical'):.3f}")
        print(f"  B) constellation within_r={mean_attr('within_r', 'constellation'):.3f}  "
              f"oracle={mean_attr('oracle_within_r', 'constellation'):.3f}  "
              f"conditional={mean_attr('conditional_within_r', 'constellation'):.3f}")
        print(f"  C) fusion        within_r={mean_attr('within_r', 'fusion'):.3f}  "
              f"oracle={mean_attr('oracle_within_r', 'fusion'):.3f}  "
              f"conditional={mean_attr('conditional_within_r', 'fusion'):.3f}")


if __name__ == "__main__":
    main()
