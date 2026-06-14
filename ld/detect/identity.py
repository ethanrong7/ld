"""Identity tracker: countdown lock -> box-ID association -> rigid outlier.

Phase 1 of the detector pivot. YOLO localizes shape candidates; this module
decides *which* box is the real shape:

  1. Countdown lock -- on the last white-countdown frame, pick the YOLO box with
     the highest overlap against the white-shape mask (not nearest centroid).
  2. Box-ID tracks -- associate detections from the lock frame forward by IoU
     (same track ID through blend; no handoff re-match).
  3. Rigid outlier -- when the locked track is lost, fall back to highest-
     residual box among candidates.

Usage:
    python -m ld.detect.identity --weights data/detect/runs/yolov8n_probe/weights/best.pt
    python -m ld.detect.identity --weights .../best.pt --inputs data/t1_cropped_trimmed.mp4
    python -m ld.detect.identity --weights .../best.pt --evidence
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.config import DATA_DIR, DETECT_DIR, GATE_RADIUS, WHITE_S_MAX, WHITE_V_MIN
from ld.detect.constellation import TrackPoint, _seed
from ld.detect.fusion import FusionPack, detect_fusion_clip
from ld.vision.cursor import strip_pointer

__all__ = ["LockInfo", "IdentityReport", "compute_countdown_lock", "track_identity",
           "score_identity", "run_clip"]


@dataclass
class LockInfo:
    """Box locked on the last white-countdown frame."""

    box: tuple[float, float, float, float, float]
    cx: float
    cy: float
    frame: int

# Tracker tunables.
UPDATE_ALPHA = 0.55
VEL_DAMP = 0.7
VEL_MAX = 8.0
COAST_PATIENCE = 15
RESID_EMA = 0.6
LOCK_BONUS = 2.0
RIGID_RANSAC_THRESH = 3.0
IOU_MATCH_MIN = 0.15
TRACK_MISS_MAX = 8


@dataclass
class BoxTrack:
    tid: int
    box: tuple[float, float, float, float, float]
    cx: float
    cy: float
    residual_ema: float = 0.0
    misses: int = 0


@dataclass
class IdentityReport:
    clip: str
    within_r: float
    within_1p5r: float
    median_px: float
    oracle_within_r: float
    conditional_within_r: float
    n: int
    n_oracle_hit: int

    def __str__(self) -> str:
        return (f"within_r={self.within_r:.3f}  within_1.5r={self.within_1p5r:.3f}  "
                f"median={self.median_px:.1f}px  oracle={self.oracle_within_r:.3f}  "
                f"conditional={self.conditional_within_r:.3f}  "
                f"(n={self.n}, oracle_frames={self.n_oracle_hit})")


def _centroid(box: tuple[float, float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _associate(tracks: list[BoxTrack], boxes: list[tuple],
               next_tid: int) -> tuple[list[BoxTrack], int]:
    """Greedy IoU matching: update tracks, spawn new, drop stale."""
    if not boxes:
        for t in tracks:
            t.misses += 1
        return [t for t in tracks if t.misses <= TRACK_MISS_MAX], next_tid

    pairs: list[tuple[float, int, int]] = []
    for ti, tr in enumerate(tracks):
        for bi, box in enumerate(boxes):
            v = _iou(tr.box, box)
            if v >= IOU_MATCH_MIN:
                pairs.append((v, ti, bi))
    pairs.sort(reverse=True)

    used_t: set[int] = set()
    used_b: set[int] = set()
    updated = [BoxTrack(t.tid, t.box, t.cx, t.cy, t.residual_ema, t.misses) for t in tracks]

    for _, ti, bi in pairs:
        if ti in used_t or bi in used_b:
            continue
        box = boxes[bi]
        cx, cy = _centroid(box)
        updated[ti].box = box
        updated[ti].cx, updated[ti].cy = cx, cy
        updated[ti].misses = 0
        used_t.add(ti)
        used_b.add(bi)

    for ti, tr in enumerate(updated):
        if ti not in used_t:
            tr.misses += 1

    alive = [t for t in updated if t.misses <= TRACK_MISS_MAX]
    for bi, box in enumerate(boxes):
        if bi not in used_b:
            cx, cy = _centroid(box)
            alive.append(BoxTrack(next_tid, box, cx, cy))
            next_tid += 1
    return alive, next_tid


def _rigid_residuals(prev_tracks: list[BoxTrack],
                     curr_tracks: list[BoxTrack]) -> dict[int, float]:
    """Per track-id residual from fitted sheet motion (prev -> curr)."""
    prev_by = {t.tid: np.array([t.cx, t.cy], np.float32) for t in prev_tracks}
    out: dict[int, float] = {t.tid: 0.0 for t in curr_tracks}
    common = [t for t in curr_tracks if t.tid in prev_by]
    if len(common) < 3:
        return out

    src = np.array([prev_by[t.tid] for t in common], np.float32)
    dst = np.array([[t.cx, t.cy] for t in common], np.float32)
    T, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=RIGID_RANSAC_THRESH)
    if T is None:
        return out

    for t in common:
        mapped = T[:, :2] @ prev_by[t.tid] + T[:, 2]
        out[t.tid] = float(np.hypot(*(np.array([t.cx, t.cy]) - mapped)))
    return out


def _white_mask(frame: np.ndarray) -> np.ndarray:
    """Mask of the largest bright desaturated blob (countdown shape, not START text)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, np.array((0, 0, WHITE_V_MIN)), np.array((180, WHITE_S_MAX, 255)))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return raw
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(raw)
    cv2.drawContours(mask, [c], -1, 255, -1)
    return mask


def _box_mask_overlap(box: tuple, mask: np.ndarray) -> float:
    """Fraction of box pixels covered by the white-shape mask."""
    h, w = mask.shape
    x1 = int(max(0, math.floor(box[0])))
    y1 = int(max(0, math.floor(box[1])))
    x2 = int(min(w, math.ceil(box[2])))
    y2 = int(min(h, math.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = mask[y1:y2, x1:x2]
    return float(cv2.countNonZero(patch)) / float(patch.size)


def _box_white_circle_overlap(box: tuple, white: tuple[float, float, float]) -> float:
    """Fallback: overlap of box with white-shape circle from detect_white_shape."""
    cx, cy, r = white
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    local = np.zeros((bh, bw), np.uint8)
    cv2.circle(local, (int(round(cx - x1)), int(round(cy - y1))), int(round(r)), 255, -1)
    return float(cv2.countNonZero(local)) / float(bw * bh)


def _countdown_white_pack(packs: list[FusionPack], miss_to_confirm: int = 3) -> FusionPack | None:
    """Last frame with visible countdown white shape (stops when countdown ends)."""
    last: FusionPack | None = None
    misses = 0
    for p in packs:
        if p.white is not None:
            last = p
            misses = 0
        elif last is not None:
            misses += 1
            if misses >= miss_to_confirm:
                break
    return last


def compute_countdown_lock(packs: list[FusionPack], clip: str | Path) -> LockInfo | None:
    """Pick the YOLO box that best overlaps the white countdown blob."""
    lock_pack = _countdown_white_pack(packs)
    if lock_pack is None or not lock_pack.boxes or lock_pack.white is None:
        return None

    mask: np.ndarray | None = None
    src = VideoSource(clip)
    for idx, frame in src.frames():
        if idx == lock_pack.idx:
            mask = _white_mask(strip_pointer(frame, strip_green=True))
            break
    src.release()

    best_i, best_s = 0, -1.0
    for i, box in enumerate(lock_pack.boxes):
        s = _box_mask_overlap(box, mask) if mask is not None else 0.0
        if s <= 0:
            s = _box_white_circle_overlap(box, lock_pack.white)
        if s > best_s:
            best_s, best_i = s, i

    box = lock_pack.boxes[best_i]
    cx, cy = _centroid(box)
    return LockInfo(box, cx, cy, lock_pack.idx)


def _locked_tid_on_frame(boxes: list[tuple], lock: LockInfo,
                         next_tid: int) -> tuple[list[BoxTrack], int, int]:
    """Spawn tracks on the lock frame; return (tracks, next_tid, locked_tid)."""
    tracks, next_tid = _associate([], boxes, next_tid)
    lock_i = max(range(len(boxes)), key=lambda i: _iou(boxes[i], lock.box))
    return tracks, next_tid, tracks[lock_i].tid


def _handoff_track_id(tracks: list[BoxTrack], boxes: list[tuple],
                      lock: LockInfo) -> int | None:
    """Match lock-frame box to a track at blend handoff (IoU + centroid proximity)."""
    best_i, best_s = 0, -1e9
    for i, box in enumerate(boxes):
        cx, cy = _centroid(box)
        iou_lock = _iou(box, lock.box)
        dist = math.hypot(cx - lock.cx, cy - lock.cy)
        score = iou_lock * 4.0 - dist / 80.0
        if score > best_s:
            best_s, best_i = score, i
    if best_i < len(tracks):
        return tracks[best_i].tid
    return tracks[0].tid if tracks else None


def _find_locked(tracks: list[BoxTrack], locked_tid: int | None) -> BoxTrack | None:
    if locked_tid is None:
        return None
    return next((t for t in tracks if t.tid == locked_tid), None)


def track_identity(packs: list[FusionPack], lock: LockInfo | None = None, *,
                   gate_radius: float | None = None,
                   frame_wh: tuple[int, int] | None = None
                   ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """Return (track, locked_tid_per_frame, start_frame, radius)."""
    seed_x, seed_y, radius, start = _seed(packs)  # type: ignore[arg-type]
    lock_frame = lock.frame if lock is not None else start
    gate = gate_radius or max(GATE_RADIUS, 1.3 * radius)
    pos = np.array([seed_x, seed_y], np.float32)
    vel = np.zeros(2, np.float32)
    lost = 0
    locked_tid: int | None = None
    tracks: list[BoxTrack] = []
    prev_tracks: list[BoxTrack] = []
    next_tid = 0
    fw, fh = frame_wh if frame_wh else (None, None)

    track: list[TrackPoint] = []
    locked_hist: list[int | None] = []

    for p in packs:
        # Before countdown lock: follow white shape centroid.
        if p.idx < lock_frame or math.isnan(pos[0]):
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(None)
            if p.white is not None:
                pos = np.array([p.white[0], p.white[1]], np.float32)
            continue

        # Lock frame: spawn tracks and lock identity on this frame's boxes.
        if p.idx == lock_frame and locked_tid is None:
            if lock is not None and p.boxes:
                tracks, next_tid, locked_tid = _locked_tid_on_frame(p.boxes, lock, next_tid)
                pos = np.array([lock.cx, lock.cy], np.float32)
                vel[:] = 0.0
            elif p.boxes:
                tracks, next_tid = _associate([], p.boxes, next_tid)
                locked_tid = tracks[0].tid
                pos = np.array([tracks[0].cx, tracks[0].cy], np.float32)
                vel[:] = 0.0
            prev_tracks = [BoxTrack(t.tid, t.box, t.cx, t.cy, t.residual_ema, t.misses) for t in tracks]
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(locked_tid)
            continue

        # Propagate box IDs from lock frame forward.
        tracks, next_tid = _associate(tracks, p.boxes, next_tid)
        if p.idx == start and lock is not None and tracks:
            locked_tid = _handoff_track_id(tracks, p.boxes, lock)
        locked_t = _find_locked(tracks, locked_tid)

        # Bridge frames between lock and blend: follow locked box.
        if p.idx < start:
            if locked_t is not None:
                pos = np.array([locked_t.cx, locked_t.cy], np.float32)
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(locked_tid)
            prev_tracks = [BoxTrack(t.tid, t.box, t.cx, t.cy, t.residual_ema, t.misses) for t in tracks]
            continue

        # Tracking phase: residual scoring + lock bonus.
        resid = _rigid_residuals(prev_tracks, tracks)
        for t in tracks:
            r = resid.get(t.tid, 0.0)
            t.residual_ema = RESID_EMA * t.residual_ema + (1.0 - RESID_EMA) * r

        pred = pos + vel
        best_xy = None
        best_score = -1e9
        for t in tracks:
            dist = float(np.hypot(t.cx - pred[0], t.cy - pred[1]))
            if dist > gate:
                continue
            prox = 1.0 - dist / gate
            score = t.residual_ema + 0.3 * prox
            if t.tid == locked_tid:
                score += LOCK_BONUS
            if score > best_score:
                best_score, best_xy = score, np.array([t.cx, t.cy], np.float32)

        if best_xy is not None:
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
        locked_hist.append(locked_tid)
        prev_tracks = [BoxTrack(t.tid, t.box, t.cx, t.cy, t.residual_ema, t.misses) for t in tracks]

    return track, locked_hist, start, radius


def score_identity(packs: list[FusionPack], track: list[TrackPoint], start: int,
                   radius: float, clip_name: str) -> IdentityReport:
    gt = {p.idx: p.gt for p in packs if p.gt is not None}
    errs: list[float] = []
    oracle_hit_errs: list[float] = []

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
            oracle_err = min(math.hypot(_centroid(b)[0] - g[0], _centroid(b)[1] - g[1]) for b in p.boxes)
            if oracle_err < radius:
                oracle_hit_errs.append(err)

    oracle: list[float] = []
    for p in packs:
        if p.idx < start or p.gt is None or not p.boxes:
            continue
        oracle.append(min(
            math.hypot(_centroid(b)[0] - p.gt[0], _centroid(b)[1] - p.gt[1]) for b in p.boxes))
    oracle_wr = (sum(e < radius for e in oracle) / len(oracle)) if oracle else 0.0

    if not errs:
        return IdentityReport(clip_name, 0.0, 0.0, float("nan"), oracle_wr, 0.0, 0, 0)
    errs.sort()
    n = len(errs)
    cond = (sum(e < radius for e in oracle_hit_errs) / len(oracle_hit_errs)
            if oracle_hit_errs else 0.0)
    return IdentityReport(
        clip_name, sum(e < radius for e in errs) / n,
        sum(e < 1.5 * radius for e in errs) / n, errs[n // 2],
        oracle_wr, cond, n, len(oracle_hit_errs))


def _render_evidence(clip: Path, packs: list[FusionPack], track: list[TrackPoint],
                     locked_hist: list[int | None], lock: LockInfo | None, out: Path) -> None:
    tp_by = {t.idx: t for t in track}
    src = VideoSource(clip)
    writer = open_writer(out, src.meta.width, src.meta.height, src.meta.fps or 30.0)
    tracks: list[BoxTrack] = []
    next_tid = 0
    locked_tid: int | None = None
    _, _, _, start = _seed(packs)  # type: ignore[arg-type]
    lock_frame = lock.frame if lock else start

    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        frame = strip_pointer(raw, strip_green=True)

        if lock is not None and p.idx >= lock_frame:
            if p.idx == lock_frame and locked_tid is None and p.boxes:
                tracks, next_tid, locked_tid = _locked_tid_on_frame(p.boxes, lock, next_tid)
            elif locked_tid is not None:
                tracks, next_tid = _associate(tracks, p.boxes, next_tid)
                if p.idx == start:
                    locked_tid = _handoff_track_id(tracks, p.boxes, lock)

        chosen_tid = locked_hist[idx] if idx < len(locked_hist) else locked_tid
        for t in tracks:
            is_locked = t.tid == chosen_tid
            col = (0, 255, 0) if is_locked else (255, 180, 0)
            thick = 3 if is_locked else 1
            b = t.box
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, thick)

        if p.gt is not None:
            cv2.drawMarker(frame, (int(p.gt[0]), int(p.gt[1])), (0, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 18, 2)
        tp = tp_by.get(idx)
        if tp is not None and not math.isnan(tp.x):
            col = (0, 0, 255) if tp.state == "track" else (0, 140, 255)
            cv2.circle(frame, (int(tp.x), int(tp.y)), 10, col, 2)
        hud = f"f{idx} dets={len(p.boxes)} lock={chosen_tid}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()
    src.release()


def run_clip(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
             imgsz: int = 768, use_cache: bool = True, evidence: bool = False) -> IdentityReport:
    clip = Path(clip)
    name = clip.stem.replace("_cropped_trimmed", "")
    packs = detect_fusion_clip(weights, clip, conf=conf, imgsz=imgsz, use_cache=use_cache)
    lock = compute_countdown_lock(packs, clip)
    src = VideoSource(clip)
    wh = (src.meta.width, src.meta.height)
    src.release()
    track, locked_hist, start, radius = track_identity(packs, lock, frame_wh=wh)
    report = score_identity(packs, track, start, radius, name)
    if evidence:
        out = DETECT_DIR / "evidence" / f"{clip.stem}_identity.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        _render_evidence(clip, packs, track, locked_hist, lock, out)
        print(f"  evidence -> {out}")
    return report


def _default_inputs() -> list[Path]:
    return sorted(DATA_DIR.glob("t*_cropped_trimmed.mp4"),
                  key=lambda p: int(p.stem.split("_")[0][1:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Countdown lock + box-ID identity tracker")
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

    reports: list[IdentityReport] = []
    for clip in inputs:
        print(f"[{clip.stem}] identity tracking ...")
        rep = run_clip(args.weights, clip, conf=args.conf, imgsz=args.imgsz,
                       use_cache=not args.no_cache, evidence=args.evidence)
        print(f"  {rep}")
        reports.append(rep)

    valid = [r for r in reports if r.n > 0]
    if valid:
        print(f"\n{'='*60}")
        print(f"MEAN across {len(valid)} clips:")
        print(f"  within_r       = {sum(r.within_r for r in valid)/len(valid):.3f}")
        print(f"  oracle         = {sum(r.oracle_within_r for r in valid)/len(valid):.3f}")
        print(f"  conditional    = {sum(r.conditional_within_r for r in valid)/len(valid):.3f}")
        print(f"  median px      = {float(np.median([r.median_px for r in valid])):.1f}")


if __name__ == "__main__":
    main()
