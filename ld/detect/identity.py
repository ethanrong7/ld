"""Identity tracker: countdown lock -> per-frame box chaining -> rigid outlier.

Phase 1 of the detector pivot. YOLO localizes shape candidates; this module
decides *which* box is the real shape:

  1. Countdown lock -- stable white centroid (teleport guard for START false
     positive) -> YOLO box via mask overlap or anchor-nearest fallback.
  2. Paper motion (default) -- track the sheet from background features (YOLO
     boxes masked out), refine from fake boxes, pick the YOLO box with highest
     temporally-smoothed deviation from paper motion.
  3. Multi-hypothesis (--mode hypothesis) -- top-K parallel box tracks scored
     over a sliding window of paper residuals (no per-frame proximity commit).
  4. Trajectory graph (--mode trajectory) -- Viterbi over IoU-linked detection
     paths; pick the path with highest cumulative paper residual (delayed commit).
  5. Hybrid (--mode hybrid) -- paper residual + IoU chain fallback on disagreement.
  4. Hybrid unified (--mode hybrid_unified) -- fused paper+chain score every frame.
  5. Box chaining (--mode chain) -- IoU continuity + rigid residual fallback.

Usage:
    python -m ld.detect.identity --weights data/detect/runs/yolov8n_probe/weights/best.pt
    python -m ld.detect.identity --weights .../best.pt --inputs data/t1_cropped_trimmed.mp4
    python -m ld.detect.identity --weights .../best.pt --evidence
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.config import (DATA_DIR, DETECT_DIR, FEAT_MAX, FEAT_MIN_DIST, FEAT_QUALITY,
                       FIELD_LAG_CONFIRM, FIELD_LAG_K,
                       GATE_RADIUS, LK_LEVELS, LK_WIN, RANSAC_THRESH,
                       WHITE_S_MAX, WHITE_V_MIN)
from ld.detect.constellation import TrackPoint, _seed
from ld.detect.fusion import FusionPack, detect_fusion_clip
from ld.vision.cursor import strip_pointer

__all__ = ["LockInfo", "IdentityReport", "compute_countdown_lock",
           "track_identity", "track_paper_identity", "score_identity", "run_clip"]


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
PAPER_RESID_EMA = 0.45      # temporal smooth; lower = more responsive
PAPER_RESID_BLEND = 0.65    # weight on EMA when frame is ambiguous (instant top-2 close)
PAPER_RESID_AMBIG = 1.2     # px; apply EMA blend only if top-2 instant residuals within this
PAPER_PROX_W = 0.30         # weight on pred proximity in default paper pick
PAPER_INLIER_MAX = 2.5   # px; fake boxes used to refine sheet fit
# Causal accumulating-evidence tracker (mode "accum"). The real shape is the
# persistent motion outlier; integrate per-tracklet residual over time and only
# switch the locked identity with hysteresis, so a momentary pause (residual ->0)
# cannot lose it and per-frame residual noise cannot make the pick jitter.
ACCUM_RESID_FLOOR = 1.5    # px; residual below this is paper noise -> no evidence
ACCUM_DECAY = 0.97         # leaky-sum memory (~33-frame half-life; pause-robust)
ACCUM_CLAMP = 40.0         # cap accumulated evidence so one frame can't dominate
ACCUM_SWITCH_MARGIN = 12.0  # challenger must lead the incumbent's evidence by this...
ACCUM_SWITCH_K = 18         # ...for this many consecutive frames before switching
ACCUM_REBIND_GATE = 1.6    # x gate: radius to re-bind incumbent when IoU assoc breaks
# Independent-rotation evidence (Phase 3). The real shape rotates on its own; fakes
# only inherit the sheet's global rotation. Measure each box's frame-to-frame
# rotation by log-polar phase correlation, subtract the global sheet rotation
# (from the paper affine), and accumulate the net per tracklet as a second
# evidence channel fused with translation. ROT_WEIGHT=0 disables it (== plain accum).
ROT_WEIGHT = 5.0         # weight of rotation evidence vs translation (0 = off)
ROT_FLOOR = 0.06         # rad (~3.4 deg); net rotation below this is noise
ROT_MIN_RESPONSE = 0.35  # min phase-correlation peak to trust a rotation reading
ROT_ROI_N = 48           # log-polar ROI resample size (px)
ROT_MIN_SIDE = 24        # px; skip boxes smaller than this (rotation unreliable)
LOCK_MASK_MIN = 0.01     # min mask overlap before anchor fallback
WHITE_TELEPORT_PX = 80.0 # START-text false positive jumps farther than this
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


def _box_rigid_residuals(prev_boxes: list[tuple], curr_boxes: list[tuple]) -> list[float]:
    """Per-box residual from fitted sheet motion (prev frame -> curr frame)."""
    n = len(curr_boxes)
    if len(prev_boxes) < 3 or n == 0:
        return [0.0] * n

    pairs: list[tuple[float, int, int]] = []
    for pi, pb in enumerate(prev_boxes):
        pcx, pcy = _centroid(pb)
        for ci, cb in enumerate(curr_boxes):
            if _iou(pb, cb) >= IOU_MATCH_MIN:
                pairs.append((_iou(pb, cb), pi, ci))
    if len(pairs) < 3:
        return [0.0] * n
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_c: set[int] = set()
    src_pts: list[np.ndarray] = []
    dst_pts: list[np.ndarray] = []
    for _, pi, ci in pairs:
        if pi in used_p or ci in used_c:
            continue
        src_pts.append(np.array(_centroid(prev_boxes[pi]), np.float32))
        dst_pts.append(np.array(_centroid(curr_boxes[ci]), np.float32))
        used_p.add(pi)
        used_c.add(ci)
        if len(src_pts) >= max(3, len(curr_boxes) // 2):
            break
    if len(src_pts) < 3:
        return [0.0] * n

    src = np.array(src_pts, np.float32)
    dst = np.array(dst_pts, np.float32)
    T, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=RIGID_RANSAC_THRESH)
    if T is None:
        return [0.0] * n

    out = [0.0] * n
    for ci, cb in enumerate(curr_boxes):
        cc = np.array(_centroid(cb), np.float32)
        best_prev: np.ndarray | None = None
        best_iou = IOU_MATCH_MIN
        for pb in prev_boxes:
            v = _iou(pb, cb)
            if v > best_iou:
                best_iou, best_prev = v, np.array(_centroid(pb), np.float32)
        if best_prev is None:
            continue
        mapped = T[:, :2] @ best_prev + T[:, 2]
        out[ci] = float(np.hypot(*(cc - mapped)))
    return out


def _shape_exclude_mask(shape: tuple[int, int], boxes: list[tuple]) -> np.ndarray:
    """Feature mask: paper/background only (YOLO shape ROIs zeroed)."""
    mask = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2, _ in boxes:
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
    return mask


def estimate_paper_motion(prev_gray: np.ndarray, cur_gray: np.ndarray,
                          boxes: list[tuple]) -> np.ndarray | None:
    """Rigid sheet transform from background features (shape boxes masked out)."""
    mask = _shape_exclude_mask(prev_gray.shape, boxes)
    p0 = cv2.goodFeaturesToTrack(prev_gray, FEAT_MAX, FEAT_QUALITY, FEAT_MIN_DIST, mask=mask)
    if p0 is None or len(p0) < 30:
        p0 = cv2.goodFeaturesToTrack(prev_gray, FEAT_MAX, FEAT_QUALITY, FEAT_MIN_DIST)
    if p0 is None:
        return None

    p1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, p0, None, winSize=(LK_WIN, LK_WIN), maxLevel=LK_LEVELS)
    keep = status.ravel() == 1
    a = p0.reshape(-1, 2)[keep]
    b = p1.reshape(-1, 2)[keep]
    if len(a) < 30:
        return None

    T, _ = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH)
    return T


def _refine_paper_from_fakes(prev_boxes: list[tuple], curr_boxes: list[tuple],
                             paper_T: np.ndarray | None) -> np.ndarray | None:
    """Second-pass sheet fit using only box centroids that move with the paper."""
    if paper_T is None or len(prev_boxes) < 4 or len(curr_boxes) < 4:
        return paper_T

    src_pts: list[np.ndarray] = []
    dst_pts: list[np.ndarray] = []
    for cb in curr_boxes:
        cc = np.array(_centroid(cb), np.float32)
        best_prev: np.ndarray | None = None
        best_iou = IOU_MATCH_MIN
        for pb in prev_boxes:
            v = _iou(pb, cb)
            if v > best_iou:
                best_iou, best_prev = v, np.array(_centroid(pb), np.float32)
        if best_prev is None:
            continue
        mapped = paper_T[:, :2] @ best_prev + paper_T[:, 2]
        if float(np.hypot(*(cc - mapped))) <= PAPER_INLIER_MAX:
            src_pts.append(best_prev)
            dst_pts.append(cc)
    if len(src_pts) < 4:
        return paper_T

    T, _ = cv2.estimateAffinePartial2D(
        np.array(src_pts, np.float32), np.array(dst_pts, np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=RIGID_RANSAC_THRESH)
    return T if T is not None else paper_T


def _paper_box_residuals(prev_boxes: list[tuple], curr_boxes: list[tuple],
                         paper_T: np.ndarray | None) -> list[float]:
    """Per-box deviation from where the tracked paper would carry it."""
    n = len(curr_boxes)
    if paper_T is None or not prev_boxes:
        return [0.0] * n
    out = [0.0] * n
    for ci, cb in enumerate(curr_boxes):
        cc = np.array(_centroid(cb), np.float32)
        best_prev: np.ndarray | None = None
        best_iou = IOU_MATCH_MIN
        for pb in prev_boxes:
            v = _iou(pb, cb)
            if v > best_iou:
                best_iou, best_prev = v, np.array(_centroid(pb), np.float32)
        if best_prev is None:
            continue
        mapped = paper_T[:, :2] @ best_prev + paper_T[:, 2]
        out[ci] = float(np.hypot(*(cc - mapped)))
    return out


def _ema_paper_residuals(prev_boxes: list[tuple], prev_ema: list[float] | None,
                         curr_boxes: list[tuple], instant: list[float]) -> list[float]:
    """Smooth per-box paper residuals over time (IoU-matched to previous frame)."""
    if not prev_ema or len(prev_boxes) != len(prev_ema) or not curr_boxes:
        return list(instant)
    out = list(instant)
    for ci, cb in enumerate(curr_boxes):
        best_pi, best_iou = -1, IOU_MATCH_MIN
        for pi, pb in enumerate(prev_boxes):
            v = _iou(pb, cb)
            if v > best_iou:
                best_iou, best_pi = v, pi
        if best_pi >= 0:
            out[ci] = (PAPER_RESID_EMA * prev_ema[best_pi]
                       + (1.0 - PAPER_RESID_EMA) * instant[ci])
    return out


def _resid_for_pick(instant: list[float], ema: list[float] | None,
                    *, use_ema: bool) -> list[float]:
    """Instant residual by default; blend with EMA only on ambiguous frames."""
    if not use_ema or not ema or len(ema) != len(instant):
        return instant
    ranked = sorted(instant, reverse=True)
    if len(ranked) < 2 or ranked[0] - ranked[1] >= PAPER_RESID_AMBIG:
        return instant
    a = PAPER_RESID_BLEND
    return [(1.0 - a) * instant[i] + a * ema[i] for i in range(len(instant))]


def _pick_paper_outlier_box(boxes: list[tuple], prev_box: tuple | None,
                            pred: np.ndarray, gate: float,
                            paper_resid: list[float]) -> tuple | None:
    """Fake boxes ride the paper (low residual); real shape is the outlier."""
    if not boxes:
        return None
    if prev_box is None:
        best_i = max(range(len(boxes)), key=lambda i: paper_resid[i])
        return boxes[best_i]

    best_box: tuple | None = None
    best_score = -1e9
    for i, box in enumerate(boxes):
        iou = _iou(box, prev_box)
        cx, cy = _centroid(box)
        dist = float(np.hypot(cx - pred[0], cy - pred[1]))
        if dist > gate and iou < IOU_MATCH_MIN:
            continue
        prox = 1.0 - min(dist / gate, 1.0)
        score = paper_resid[i] + 0.45 * iou + PAPER_PROX_W * prox
        if score > best_score:
            best_score, best_box = score, box
    if best_box is not None:
        return best_box
    return max(boxes, key=lambda b: _iou(b, prev_box))


def _chain_locked_box(boxes: list[tuple], prev_box: tuple | None,
                      pred: np.ndarray, gate: float,
                      residuals: list[float] | None = None) -> tuple | None:
    """Pick the box that best continues prev_box (IoU + proximity + residual)."""
    if not boxes:
        return None
    if prev_box is None:
        return boxes[0]

    best_box: tuple | None = None
    best_score = -1e9
    for i, box in enumerate(boxes):
        iou = _iou(box, prev_box)
        cx, cy = _centroid(box)
        dist = float(np.hypot(cx - pred[0], cy - pred[1]))
        if dist > gate and iou < IOU_MATCH_MIN:
            continue
        prox = 1.0 - min(dist / gate, 1.0)
        resid = residuals[i] if residuals and i < len(residuals) else 0.0
        score = iou * 4.0 + 0.4 * prox + 0.15 * resid
        if score > best_score:
            best_score, best_box = score, box

    if best_box is not None:
        chained = best_box
    else:
        chained = max(boxes, key=lambda b: _iou(b, prev_box))
    return chained


def _pick_locked_box(boxes: list[tuple], prev_box: tuple | None,
                     pred: np.ndarray, gate: float,
                     residuals: list[float] | None, *, tracking: bool) -> tuple | None:
    """Bridge: IoU chain. Tracking: rigid residual + continuity."""
    if not boxes:
        return None
    if prev_box is None:
        return boxes[0]
    if not tracking or not residuals:
        return _chain_locked_box(boxes, prev_box, pred, gate, residuals)

    best_box: tuple | None = None
    best_score = -1e9
    for i, box in enumerate(boxes):
        iou = _iou(box, prev_box)
        cx, cy = _centroid(box)
        dist = float(np.hypot(cx - pred[0], cy - pred[1]))
        if dist > gate and iou < IOU_MATCH_MIN:
            continue
        prox = 1.0 - min(dist / gate, 1.0)
        resid = residuals[i]
        score = resid + 0.5 * iou + 0.35 * prox
        if score > best_score:
            best_score, best_box = score, box
    if best_box is not None:
        return best_box
    return _chain_locked_box(boxes, prev_box, pred, gate, residuals)


def _tid_for_box(tracks: list[BoxTrack], box: tuple) -> int | None:
    """Map a chained box to the nearest IoU track (for evidence overlay only)."""
    if not tracks:
        return None
    best = max(tracks, key=lambda t: _iou(t.box, box))
    return best.tid if _iou(best.box, box) >= IOU_MATCH_MIN else None


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


def _stable_white_anchor(packs: list[FusionPack]) -> tuple[float, float, float, int] | None:
    """Last white centroid before a large teleport (e.g. detector latches onto START)."""
    prev: tuple[float, float] | None = None
    stable: tuple[float, float, float, int] | None = None
    for p in packs:
        if p.white is None:
            continue
        cx, cy, r = p.white
        if prev is not None and math.hypot(cx - prev[0], cy - prev[1]) > WHITE_TELEPORT_PX:
            break
        stable = (cx, cy, r, p.idx)
        prev = (cx, cy)
    return stable


def _pick_lock_box(boxes: list[tuple], mask: np.ndarray | None,
                   white: tuple[float, float, float],
                   anchor: tuple[float, float]) -> int:
    """Mask overlap when strong; else nearest box to stable countdown anchor."""
    best_i, best_s = 0, -1.0
    for i, box in enumerate(boxes):
        s = _box_mask_overlap(box, mask) if mask is not None else 0.0
        if s <= 0:
            s = _box_white_circle_overlap(box, white)
        if s > best_s:
            best_s, best_i = s, i
    if best_s >= LOCK_MASK_MIN:
        return best_i
    ax, ay = anchor
    return min(range(len(boxes)),
               key=lambda i: math.hypot(_centroid(boxes[i])[0] - ax,
                                        _centroid(boxes[i])[1] - ay))


def compute_countdown_lock(packs: list[FusionPack], clip: str | Path) -> LockInfo | None:
    """Pick the YOLO box for the real shape at countdown handoff."""
    lock_pack = _countdown_white_pack(packs)
    if lock_pack is None or not lock_pack.boxes or lock_pack.white is None:
        return None

    anchor_info = _stable_white_anchor(packs)
    if anchor_info is not None:
        anchor = (anchor_info[0], anchor_info[1])
    else:
        anchor = (lock_pack.white[0], lock_pack.white[1])

    mask: np.ndarray | None = None
    src = VideoSource(clip)
    for idx, frame in src.frames():
        if idx == lock_pack.idx:
            mask = _white_mask(strip_pointer(frame, strip_green=True))
            break
    src.release()

    best_i = _pick_lock_box(lock_pack.boxes, mask, lock_pack.white, anchor)
    box = lock_pack.boxes[best_i]
    cx, cy = _centroid(box)
    return LockInfo(box, cx, cy, lock_pack.idx)


def _locked_tid_on_frame(boxes: list[tuple], lock: LockInfo,
                         next_tid: int) -> tuple[list[BoxTrack], int, int]:
    """Spawn tracks on the lock frame; return (tracks, next_tid, locked_tid)."""
    tracks, next_tid = _associate([], boxes, next_tid)
    lock_i = max(range(len(boxes)), key=lambda i: _iou(boxes[i], lock.box))
    return tracks, next_tid, tracks[lock_i].tid


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
    next_tid = 0
    prev_locked_box: tuple | None = None
    prev_boxes: list[tuple] = []
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
                prev_locked_box = lock.box
                pos = np.array([lock.cx, lock.cy], np.float32)
                vel[:] = 0.0
            elif p.boxes:
                tracks, next_tid = _associate([], p.boxes, next_tid)
                locked_tid = tracks[0].tid
                prev_locked_box = tracks[0].box
                pos = np.array([tracks[0].cx, tracks[0].cy], np.float32)
                vel[:] = 0.0
            prev_boxes = list(p.boxes)
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(locked_tid)
            continue

        # Per-frame box chaining (IoU continuity + motion prediction).
        tracks, next_tid = _associate(tracks, p.boxes, next_tid)
        pos_prev = pos.copy()
        pred = pos + vel
        chain_prev = prev_locked_box
        if p.idx == start and lock is not None and p.idx - lock_frame <= 1:
            chain_prev = lock.box
        resid = _box_rigid_residuals(prev_boxes, p.boxes) if prev_boxes else None
        chained = _pick_locked_box(p.boxes, chain_prev, pred, gate, resid,
                                   tracking=p.idx >= start)
        if chained is not None:
            cx, cy = _centroid(chained)
            target = np.array([cx, cy], np.float32)
            if p.idx < start:
                pos = target
                state = "acquire"
            else:
                pos = (1.0 - UPDATE_ALPHA) * pred + UPDATE_ALPHA * target
                lost = 0
                state = "track"
            prev_locked_box = chained
            locked_tid = _tid_for_box(tracks, chained)
        elif p.idx >= start:
            pos = pred
            lost += 1
            state = "coast"
            if lost >= COAST_PATIENCE:
                vel[:] = 0.0
        else:
            state = "acquire"

        if p.idx >= start:
            vel = VEL_DAMP * vel + (1.0 - VEL_DAMP) * (pos - pos_prev)
            sp = float(np.hypot(*vel))
            if sp > VEL_MAX:
                vel *= VEL_MAX / sp

        if fw is not None:
            pos = np.array([float(np.clip(pos[0], 0, fw - 1)),
                            float(np.clip(pos[1], 0, fh - 1))], np.float32)

        track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), state, len(p.boxes)))
        locked_hist.append(locked_tid)
        prev_boxes = list(p.boxes)

    return track, locked_hist, start, radius


def track_paper_identity(clip: str | Path, packs: list[FusionPack],
                           lock: LockInfo | None = None, *,
                           gate_radius: float | None = None,
                           frame_wh: tuple[int, int] | None = None,
                           pick_mode: str = "paper"
                           ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """Track real shape by deviation of each YOLO box from estimated paper motion."""
    clip = Path(clip)
    seed_x, seed_y, radius, start = _seed(packs)  # type: ignore[arg-type]
    lock_frame = lock.frame if lock is not None else start
    gate = gate_radius or max(GATE_RADIUS, 1.3 * radius)
    pos = np.array([seed_x, seed_y], np.float32)
    vel = np.zeros(2, np.float32)
    lost = 0
    locked_tid: int | None = None
    tracks: list[BoxTrack] = []
    next_tid = 0
    prev_locked_box: tuple | None = None
    prev_boxes: list[tuple] = []
    paper_resid_ema: list[float] | None = None
    prev_gray: np.ndarray | None = None
    ol_tracks: list = []
    ol_next_tid = 0
    ol_lock_tid: int | None = None
    low_rank_streak = 0
    fw, fh = frame_wh if frame_wh else (None, None)

    track: list[TrackPoint] = []
    locked_hist: list[int | None] = []

    src = VideoSource(clip)
    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        stripped = strip_pointer(raw, strip_green=True)
        gray = cv2.cvtColor(stripped, cv2.COLOR_BGR2GRAY)

        if p.idx < lock_frame or math.isnan(pos[0]):
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(None)
            if p.white is not None:
                pos = np.array([p.white[0], p.white[1]], np.float32)
            prev_gray = gray
            prev_boxes = list(p.boxes)
            continue

        if p.idx == lock_frame and locked_tid is None:
            if lock is not None and p.boxes:
                tracks, next_tid, locked_tid = _locked_tid_on_frame(p.boxes, lock, next_tid)
                prev_locked_box = lock.box
                pos = np.array([lock.cx, lock.cy], np.float32)
                vel[:] = 0.0
            elif p.boxes:
                tracks, next_tid = _associate([], p.boxes, next_tid)
                locked_tid = tracks[0].tid
                prev_locked_box = tracks[0].box
                pos = np.array([tracks[0].cx, tracks[0].cy], np.float32)
                vel[:] = 0.0
            if pick_mode in ("paper_outlier", "paper_outlier_rank") and lock is not None and p.boxes:
                from ld.detect.outlier_track import init_outlier_tracks
                ol_tracks, ol_next_tid, ol_lock_tid = init_outlier_tracks(p.boxes, lock)
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(locked_tid)
            prev_boxes = list(p.boxes)
            prev_gray = gray
            continue

        tracks, next_tid = _associate(tracks, p.boxes, next_tid)
        pos_prev = pos.copy()
        pred = pos + vel

        paper_T = None
        paper_resid = [0.0] * len(p.boxes)
        if prev_gray is not None and prev_boxes and p.boxes:
            paper_T = estimate_paper_motion(prev_gray, gray, p.boxes)
            paper_T = _refine_paper_from_fakes(prev_boxes, p.boxes, paper_T)
            paper_resid = _paper_box_residuals(prev_boxes, p.boxes, paper_T)
            paper_resid_ema = _ema_paper_residuals(
                prev_boxes, paper_resid_ema, p.boxes, paper_resid)

        chain_prev = prev_locked_box
        if p.idx == start and lock is not None and p.idx - lock_frame <= 1:
            chain_prev = lock.box

        resid_pick = _resid_for_pick(paper_resid, paper_resid_ema, use_ema=p.idx >= start)
        if pick_mode == "paper_outlier":
            from ld.detect.outlier_track import advance_outlier_tracks, outlier_switch_box
            assoc_gate = max(gate, 1.3 * radius * 1.15)
            ol_tracks, ol_next_tid = advance_outlier_tracks(
                ol_tracks, ol_next_tid, prev_boxes, prev_gray, gray, p.boxes, assoc_gate)
            paper_chosen = _pick_paper_outlier_box(
                p.boxes, chain_prev, pred, gate, resid_pick)
            alt = outlier_switch_box(ol_tracks, ol_lock_tid, paper_chosen, lost=lost)
            chosen = alt if alt is not None else paper_chosen
        elif pick_mode == "paper_outlier_rank":
            from ld.detect.outlier_track import (advance_outlier_tracks,
                                                 outlier_switch_rank_box,
                                                 residual_rank,
                                                 update_low_rank_streak)
            assoc_gate = max(gate, 1.3 * radius * 1.15)
            ol_tracks, ol_next_tid = advance_outlier_tracks(
                ol_tracks, ol_next_tid, prev_boxes, prev_gray, gray, p.boxes, assoc_gate)
            paper_chosen = _pick_paper_outlier_box(
                p.boxes, chain_prev, pred, gate, resid_pick)
            rank = residual_rank(p.boxes, paper_chosen, resid_pick)
            low_rank_streak = update_low_rank_streak(rank, low_rank_streak)
            alt = outlier_switch_rank_box(
                ol_tracks, p.boxes, resid_pick, paper_chosen,
                low_rank_streak=low_rank_streak)
            if alt is not None:
                chosen = alt
                low_rank_streak = 0
            else:
                chosen = paper_chosen
        else:
            chosen = _pick_paper_outlier_box(p.boxes, chain_prev, pred, gate, resid_pick)
        if chosen is not None:
            cx, cy = _centroid(chosen)
            target = np.array([cx, cy], np.float32)
            if p.idx < start:
                pos = target
                state = "acquire"
            else:
                pos = (1.0 - UPDATE_ALPHA) * pred + UPDATE_ALPHA * target
                lost = 0
                state = "track"
            prev_locked_box = chosen
            locked_tid = _tid_for_box(tracks, chosen)
        elif p.idx >= start:
            pos = pred
            lost += 1
            state = "coast"
            if lost >= COAST_PATIENCE:
                vel[:] = 0.0
        else:
            state = "acquire"

        if p.idx >= start:
            vel = VEL_DAMP * vel + (1.0 - VEL_DAMP) * (pos - pos_prev)
            sp = float(np.hypot(*vel))
            if sp > VEL_MAX:
                vel *= VEL_MAX / sp

        if fw is not None:
            pos = np.array([float(np.clip(pos[0], 0, fw - 1)),
                            float(np.clip(pos[1], 0, fh - 1))], np.float32)

        track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), state, len(p.boxes)))
        locked_hist.append(locked_tid)
        prev_boxes = list(p.boxes)
        prev_gray = gray

    src.release()
    return track, locked_hist, start, radius


_ROT_HAN: np.ndarray | None = None


def _logpolar_roi(gray: np.ndarray, box: tuple, n: int) -> np.ndarray | None:
    """Resampled log-polar view of a box ROI; rotation -> vertical shift."""
    h, w = gray.shape
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    if x2 - x1 < ROT_MIN_SIDE or y2 - y1 < ROT_MIN_SIDE:
        return None
    roi = cv2.resize(gray[y1:y2, x1:x2], (n, n)).astype(np.float32)
    return cv2.warpPolar(roi, (n, n), (n / 2.0, n / 2.0), n / 2.0, cv2.WARP_POLAR_LOG)


def _box_net_rotations(prev_boxes: list[tuple], prev_gray: np.ndarray,
                       curr_boxes: list[tuple], cur_gray: np.ndarray,
                       theta_global: float) -> list[float | None]:
    """Per-current-box net rotation (rad) vs the global sheet rotation.

    Each box is matched to its best-IoU box in the previous frame (the real shape
    rotating in place keeps high IoU, so it matches); log-polar phase correlation
    recovers its absolute rotation, from which the sheet rotation is subtracted.
    None where unmatched, ROI too small, or correlation unconfident.
    """
    global _ROT_HAN
    if _ROT_HAN is None:
        _ROT_HAN = cv2.createHanningWindow((ROT_ROI_N, ROT_ROI_N), cv2.CV_32F)
    out: list[float | None] = []
    for cb in curr_boxes:
        best_prev: tuple | None = None
        best_iou = IOU_MATCH_MIN
        for pb in prev_boxes:
            v = _iou(pb, cb)
            if v > best_iou:
                best_iou, best_prev = v, pb
        if best_prev is None:
            out.append(None)
            continue
        lp0 = _logpolar_roi(prev_gray, best_prev, ROT_ROI_N)
        lp1 = _logpolar_roi(cur_gray, cb, ROT_ROI_N)
        if lp0 is None or lp1 is None:
            out.append(None)
            continue
        (_dx, dy), resp = cv2.phaseCorrelate(lp0, lp1, _ROT_HAN)
        if resp < ROT_MIN_RESPONSE:
            out.append(None)
            continue
        theta_box = dy / ROT_ROI_N * 2.0 * math.pi
        d = (theta_box - theta_global + math.pi) % (2.0 * math.pi) - math.pi
        out.append(abs(d))
    return out


def track_accum_identity(clip: str | Path, packs: list[FusionPack],
                         lock: LockInfo | None = None, *,
                         gate_radius: float | None = None,
                         frame_wh: tuple[int, int] | None = None
                         ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """Causal accumulating-evidence identity tracker.

    Every detected shape is a persistent tracklet accumulating an
    independent-motion evidence score (leaky integrator of its residual from the
    fitted paper motion). The locked identity is *followed* frame-to-frame (by
    IoU, falling back to motion-predicted proximity when the real shape's
    independent motion breaks IoU) rather than re-picked, so the per-frame
    position does not jitter. The lock only switches to a challenger tracklet
    when that challenger's accumulated evidence leads the incumbent's by
    ACCUM_SWITCH_MARGIN for ACCUM_SWITCH_K consecutive frames.
    """
    clip = Path(clip)
    seed_x, seed_y, radius, start = _seed(packs)  # type: ignore[arg-type]
    lock_frame = lock.frame if lock is not None else start
    gate = gate_radius or max(GATE_RADIUS, 1.3 * radius)
    rebind_gate = ACCUM_REBIND_GATE * gate
    pos = np.array([seed_x, seed_y], np.float32)
    vel = np.zeros(2, np.float32)
    lost = 0
    incumbent_tid: int | None = None
    tracks: list[BoxTrack] = []
    next_tid = 0
    prev_boxes: list[tuple] = []
    prev_gray: np.ndarray | None = None
    evidence: dict[int, float] = {}
    rot_evidence: dict[int, float] = {}
    switch_streak = 0
    fw, fh = frame_wh if frame_wh else (None, None)

    def total_ev(tid: int | None) -> float:
        """Fused translation + rotation evidence for a tracklet."""
        if tid is None:
            return -1.0
        return evidence.get(tid, 0.0) + ROT_WEIGHT * rot_evidence.get(tid, 0.0)

    track: list[TrackPoint] = []
    locked_hist: list[int | None] = []

    src = VideoSource(clip)
    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        stripped = strip_pointer(raw, strip_green=True)
        gray = cv2.cvtColor(stripped, cv2.COLOR_BGR2GRAY)

        # --- pre-lock: follow the white countdown shape, no identity yet ---
        if p.idx < lock_frame or math.isnan(pos[0]):
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(None)
            if p.white is not None:
                pos = np.array([p.white[0], p.white[1]], np.float32)
            prev_gray = gray
            prev_boxes = list(p.boxes)
            continue

        # --- lock frame: seed the incumbent identity from the countdown lock ---
        if p.idx == lock_frame and incumbent_tid is None:
            if lock is not None and p.boxes:
                tracks, next_tid, incumbent_tid = _locked_tid_on_frame(p.boxes, lock, next_tid)
                pos = np.array([lock.cx, lock.cy], np.float32)
            elif p.boxes:
                tracks, next_tid = _associate([], p.boxes, next_tid)
                incumbent_tid = tracks[0].tid
                pos = np.array([tracks[0].cx, tracks[0].cy], np.float32)
            vel[:] = 0.0
            evidence = {t.tid: 0.0 for t in tracks}
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(incumbent_tid)
            prev_boxes = list(p.boxes)
            prev_gray = gray
            continue

        tracks, next_tid = _associate(tracks, p.boxes, next_tid)
        pos_prev = pos.copy()
        pred = pos + vel

        # --- paper motion + per-box residual from the rigid sheet ---
        paper_resid = [0.0] * len(p.boxes)
        rot_nets: list[float | None] = [None] * len(p.boxes)
        if prev_gray is not None and prev_boxes and p.boxes:
            paper_T = estimate_paper_motion(prev_gray, gray, p.boxes)
            paper_T = _refine_paper_from_fakes(prev_boxes, p.boxes, paper_T)
            paper_resid = _paper_box_residuals(prev_boxes, p.boxes, paper_T)
            if ROT_WEIGHT > 0.0:
                theta_global = (math.atan2(paper_T[1, 0], paper_T[0, 0])
                                if paper_T is not None else 0.0)
                rot_nets = _box_net_rotations(
                    prev_boxes, prev_gray, p.boxes, gray, theta_global)

        # --- accumulate translation + rotation evidence per tracklet ---
        seen: set[int] = set()
        for i, box in enumerate(p.boxes):
            tid = _tid_for_box(tracks, box)
            if tid is None:
                continue
            e = max(0.0, paper_resid[i] - ACCUM_RESID_FLOOR)
            evidence[tid] = min(ACCUM_CLAMP, ACCUM_DECAY * evidence.get(tid, 0.0) + e)
            if rot_nets[i] is not None:
                re = max(0.0, rot_nets[i] - ROT_FLOOR)
                rot_evidence[tid] = min(ACCUM_CLAMP, ACCUM_DECAY * rot_evidence.get(tid, 0.0) + re)
            seen.add(tid)
        for tid in list(evidence):
            if tid not in seen:
                evidence[tid] *= ACCUM_DECAY
        for tid in list(rot_evidence):
            if tid not in seen:
                rot_evidence[tid] *= ACCUM_DECAY

        # --- follow the incumbent identity (IoU assoc; re-bind on break) ---
        inc = next((t for t in tracks if t.tid == incumbent_tid), None)
        if inc is None and p.boxes:
            # The real shape's independent motion likely broke IoU continuity.
            # Re-bind among boxes near the prediction, preferring the one with the
            # most accumulated evidence (the integrated outlier), not whichever
            # fake happens to sit nearest the prediction this frame.
            cands = [b for b in p.boxes
                     if math.hypot(_centroid(b)[0] - pred[0], _centroid(b)[1] - pred[1]) <= rebind_gate]
            if cands:
                def _rebind_key(b: tuple) -> tuple[float, float]:
                    tid = _tid_for_box(tracks, b)
                    dist = math.hypot(_centroid(b)[0] - pred[0], _centroid(b)[1] - pred[1])
                    return (total_ev(tid), -dist)
                box = max(cands, key=_rebind_key)
                incumbent_tid = _tid_for_box(tracks, box)
                inc = next((t for t in tracks if t.tid == incumbent_tid), None)
                switch_streak = 0

        # --- hysteresis switch to a higher-evidence challenger ---
        inc_ev = total_ev(incumbent_tid)
        challengers = [(t.tid, total_ev(t.tid)) for t in tracks if t.tid != incumbent_tid]
        if challengers:
            ch_tid, ch_ev = max(challengers, key=lambda kv: kv[1])
            if ch_ev - inc_ev >= ACCUM_SWITCH_MARGIN:
                switch_streak += 1
            else:
                switch_streak = 0
            if switch_streak >= ACCUM_SWITCH_K:
                incumbent_tid = ch_tid
                inc = next((t for t in tracks if t.tid == incumbent_tid), inc)
                switch_streak = 0

        chosen = inc.box if inc is not None else None
        if chosen is not None:
            cx, cy = _centroid(chosen)
            target = np.array([cx, cy], np.float32)
            pos = (1.0 - UPDATE_ALPHA) * pred + UPDATE_ALPHA * target
            lost = 0
            state = "track"
        else:
            pos = pred
            lost += 1
            state = "coast"
            if lost >= COAST_PATIENCE:
                vel[:] = 0.0

        vel = VEL_DAMP * vel + (1.0 - VEL_DAMP) * (pos - pos_prev)
        sp = float(np.hypot(*vel))
        if sp > VEL_MAX:
            vel *= VEL_MAX / sp

        if fw is not None:
            pos = np.array([float(np.clip(pos[0], 0, fw - 1)),
                            float(np.clip(pos[1], 0, fh - 1))], np.float32)

        track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), state, len(p.boxes)))
        locked_hist.append(incumbent_tid)
        prev_boxes = list(p.boxes)
        prev_gray = gray

    src.release()
    return track, locked_hist, start, radius


FIELD_SNAP_FRAC = 1.0   # snap field output to nearest YOLO box within this x gate

# Coherence snap-weight blend (coherence emission integrator). The snap normally picks
# argmax saliency-MASS; this reweights it by the directional/temporal COHERENCE of the
# outlier residual vectors inside each box, accumulated over a causal window. coherence =
# ||sum vecs|| / sum||vecs|| in [0,1]: ~1 when the box's independent motion all points one
# way (the real shape drifting rigidly), ~0 when it's incoherent noise/fakes. The probe
# (coh_gate) showed the coherent box already sits on GT 50% of field's still-wrong frames.
FIELD_COH_LAMBDA = 0.0  # 0 = OFF (field/field_lag byte-identical to pre-coherence leader)
FIELD_COH_WIN = 16      # causal window (frames) the coherent-mass is accumulated over

# Coherence FAR-JUMP override (mode "field_coh"). The snap-weight blend above is confined
# to the snap radius and CANNOT reach the recoverable signal (measured: on t5's wrong frames
# the coherent on-GT box is OUTSIDE the snap radius 190x vs 4x inside). The signal is an
# escape-from-lock-in cue: when a coherent-mass challenger persistently disagrees with the
# committed pick, is FAR away, and is confident (margin>=tau), override to it. Validated via
# coh_gate at LOO 0.7442 (+0.023 over field_lag, no clip regresses). Config = the gate's
# LOO-robust corner (single conservative cfg, do-no-harm on every clip incl. t1).
FIELD_COH_OVR_WIN = 16       # accumulation window for coherent-mass challenger
FIELD_COH_OVR_TAU = 0.30     # min margin (top-runnerup)/top to trust the challenger [causal key]
FIELD_COH_OVR_C = 8          # consecutive frames of far+confident disagreement before override
FIELD_COH_OVR_FAR = 1.5      # challenger must be > this many radii from the committed pick
FIELD_COH_CURL_W = 0.0       # weight of rotational-curl vs translational coherence in the fused
#   challenger. 0 = SHIPPED (coherence-only). Rotational curl DOES localize the camouflaged real
#   shape (curl_probe: t5 top-3 0.70, t8 0.64 — falsifies the old "rotation weakest on laggards"
#   claim) and its top-3 MISSES are disjoint from coherence's (fuse_probe union lift +0.14..+0.25).
#   BUT fusing it into the argmax far-jump override does NOT help: curl's top-1 is far weaker than
#   its top-3 (t5 0.35 vs 0.70), so its argmax often points at a fake, and max-fusion lets curl's
#   confident-but-wrong frames trigger wrong overrides (curl_w=1.0 -> -0.016 mean, t5 -0.047, the
#   clip it should rescue; curl_w=0.5 is inert). The disjoint coverage is real in LOCALIZATION but
#   not exploitable via argmax+margin — curl's margin doesn't mark its correctness (causal-key wall
#   one level down). A future mechanism using curl's TOP-3 (e.g. restrict coherence search to curl's
#   top-3 boxes) is unbuilt; do not re-enable curl_w in the override without it.


def _box_coherent_mass(box: tuple, ovecs_win) -> float:
    """Resultant of outlier residual VECTORS inside `box`, weighted by directional
    coherence, accumulated over a causal window. `ovecs_win` is a list of (pix, vecs)
    pairs (newest-first or any order; order-independent) where pix is (M,2) feature
    positions and vecs is (M,2) residual vectors for one past frame."""
    acc = 0.0
    for pix, vecs in ovecs_win:
        if pix.shape[0] == 0:
            continue
        inb = ((pix[:, 0] >= box[0]) & (pix[:, 0] <= box[2])
               & (pix[:, 1] >= box[1]) & (pix[:, 1] <= box[3]))
        if inb.sum() < 2:
            continue
        v = vecs[inb]
        net = float(np.linalg.norm(v.sum(0)))
        tot = float(np.linalg.norm(v, axis=1).sum())
        if tot > 1e-6:
            acc += net * (net / tot)
    return acc


def _box_rotational_curl(box: tuple, ovecs_win) -> float:
    """Coherent ROTATIONAL curl of the residual vectors inside `box`, accumulated over a
    causal window. Orthogonal to `_box_coherent_mass`: translational coherence sums
    circulating vectors to ~0 (blind to pure rotation), this measures the signed angular
    momentum Σ(r×v) of vectors about their centroid — large when the box's independent
    motion is a coherent spin (the real shape rotating), ~0 for rigid fakes / incoherent
    noise. Probe (curl_probe/fuse_probe): top-3-localizes the camouflaged real shape on
    t5 0.70 / t8 0.64 where appearance-based rotation fails, and its MISSES are disjoint
    from coherence's (union lift +0.14..+0.25 on laggard miss frames)."""
    acc = 0.0
    for pix, vecs in ovecs_win:
        if pix.shape[0] < 3:
            continue
        inb = ((pix[:, 0] >= box[0]) & (pix[:, 0] <= box[2])
               & (pix[:, 1] >= box[1]) & (pix[:, 1] <= box[3]))
        if inb.sum() < 3:
            continue
        P = pix[inb]
        V = vecs[inb]
        c = P.mean(0)
        R = P - c
        cross = R[:, 0] * V[:, 1] - R[:, 1] * V[:, 0]   # z of r×v per feature
        den = float((np.linalg.norm(R, axis=1) * np.linalg.norm(V, axis=1)).sum())
        s = float(cross.sum())
        if den > 1e-6:
            acc += abs(s) * (abs(s) / den)
    return acc


def _box_saliency_mass(box: tuple, sal: np.ndarray) -> float:
    h, w = sal.shape
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(sal[y1:y2, x1:x2].sum())


def track_field_identity(clip: str | Path, packs: list[FusionPack],
                         lock: LockInfo | None = None, *,
                         gate_radius: float | None = None,
                         frame_wh: tuple[int, int] | None = None,
                         yolo_snap: bool = True,
                         snap_frac: float = FIELD_SNAP_FRAC,
                         snap_mode: str = "mass",
                         snap_feedback: bool = True,
                         coh_lambda: float = FIELD_COH_LAMBDA,
                         coh_win: int = FIELD_COH_WIN,
                         chosen_centroids: list | None = None,
                         ovecs_out: list | None = None,
                         ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """Position-field tracker: identity-free, fragmentation-immune.

    Independent-motion evidence is accumulated *spatially* (per-frame outlier
    saliency from `motion.estimate_motion`, EMA-smoothed and gated by a
    constant-velocity model in `OutlierTracker`) rather than per tracklet — so the
    real shape's identity fragmenting across YOLO boxes (the diagnosed limiter)
    cannot lose it. The tracked peak is snapped to a YOLO box each frame, fusing the
    healthy detector for precise localization.

    snap_mode: "nearest" = box centroid nearest the tracked peak; "mass" = box with
    the most saliency inside it among boxes within the snap radius (field-native).
    snap_feedback: write the snapped position back into the tracker so it stays
    box-anchored between frames (reduces drift).
    """
    from ld.track.tracker import OutlierTracker
    from ld.vision.motion import estimate_motion, saliency_map

    clip = Path(clip)
    seed_x, seed_y, radius, start = _seed(packs)  # type: ignore[arg-type]
    lock_frame = lock.frame if lock is not None else start
    gate = gate_radius or max(GATE_RADIUS, 1.3 * radius)
    snap_dist = snap_frac * gate
    pos = np.array([seed_x, seed_y], np.float32)
    tracker: OutlierTracker | None = None
    prev_gray: np.ndarray | None = None
    fw, fh = frame_wh if frame_wh else (None, None)

    track: list[TrackPoint] = []
    locked_hist: list[int | None] = []

    # Rolling causal window of (positions, vectors) for the coherence snap-weight.
    from collections import deque
    ovecs_hist: deque = deque(maxlen=coh_win) if coh_lambda > 0 else None

    src = VideoSource(clip)
    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        gray = cv2.cvtColor(strip_pointer(raw, strip_green=True), cv2.COLOR_BGR2GRAY)

        if p.idx < lock_frame or math.isnan(pos[0]):
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(None)
            if p.white is not None:
                pos = np.array([p.white[0], p.white[1]], np.float32)
            prev_gray = gray
            continue

        if p.idx == lock_frame and tracker is None:
            cx, cy = (lock.cx, lock.cy) if lock is not None else (float(pos[0]), float(pos[1]))
            pos = np.array([cx, cy], np.float32)
            tracker = OutlierTracker(cx, cy, gate)
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(None)
            prev_gray = gray
            continue

        state = "coast"
        x, y = float(pos[0]), float(pos[1])
        sal = None
        if prev_gray is not None and tracker is not None:
            field = estimate_motion(prev_gray, gray)
            sal = saliency_map(field, gray.shape)
            tp = tracker.update(p.idx, sal)
            x, y, state = tp.x, tp.y, tp.state
            if ovecs_hist is not None:
                ovecs_hist.append((field.outliers, field.outlier_vectors))
            if ovecs_out is not None:
                ovecs_out.append((p.idx, field.outliers, field.outlier_vectors))

        if yolo_snap and p.boxes:
            near = [b for b in p.boxes
                    if math.hypot(_centroid(b)[0] - x, _centroid(b)[1] - y) <= snap_dist]
            chosen = None
            if snap_mode == "mass" and near and sal is not None and sal.max() > 0:
                if coh_lambda > 0 and ovecs_hist:
                    win = list(ovecs_hist)
                    cohs = [_box_coherent_mass(b, win) for b in near]
                    cmax = max(cohs)
                    cn = [c / cmax if cmax > 1e-9 else 0.0 for c in cohs]
                    chosen = max(range(len(near)),
                                 key=lambda i: _box_saliency_mass(near[i], sal)
                                 * (1.0 + coh_lambda * cn[i]))
                    chosen = near[chosen]
                else:
                    chosen = max(near, key=lambda b: _box_saliency_mass(b, sal))
            elif near:
                chosen = min(near, key=lambda b: math.hypot(_centroid(b)[0] - x, _centroid(b)[1] - y))
            if chosen is not None:
                x, y = _centroid(chosen)
                if snap_feedback and tracker is not None:
                    tracker.pos = np.array([x, y], np.float32)
                if chosen_centroids is not None:
                    chosen_centroids.append((p.idx, float(x), float(y)))
            elif chosen_centroids is not None:
                chosen_centroids.append((p.idx, None))

        if fw is not None:
            x = float(np.clip(x, 0, fw - 1))
            y = float(np.clip(y, 0, fh - 1))

        pos = np.array([x, y], np.float32)
        track.append(TrackPoint(p.idx, x, y, state, len(p.boxes)))
        locked_hist.append(None)
        prev_gray = gray

    src.release()
    return track, locked_hist, start, radius


def track_field_lag_identity(clip: str | Path, packs: list[FusionPack],
                             lock: LockInfo | None = None, *,
                             gate_radius: float | None = None,
                             frame_wh: tuple[int, int] | None = None,
                             lag_k: int = FIELD_LAG_K,
                             confirm: float = FIELD_LAG_CONFIRM,
                             coh_lambda: float = FIELD_COH_LAMBDA,
                             coh_win: int = FIELD_COH_WIN,
                             ovecs_out: list | None = None,
                             ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """Fixed-lag confirmation over field's box-pick sequence.

    Defers committing a pick until `lag_k` future frames confirm the same box is
    field's choice in >= `confirm` fraction of the window. Overrules transient
    single-frame creep onto an adjacent fake before it poisons the CV velocity and
    locks in (the diagnosed failure mode). Emits frame t-lag_k; legitimately online.

    It is a confirmation filter on the pick sequence, NOT a velocity model and NOT a
    new leader (avoids the trajectory-led collapse) and NOT a speed cap (avoids the
    failed smoother). It only delays commitment so a brief creep is outvoted.
    """
    centroids: list = []
    track, locked_hist, start, radius = track_field_identity(
        clip, packs, lock, gate_radius=gate_radius, frame_wh=frame_wh,
        coh_lambda=coh_lambda, coh_win=coh_win,
        chosen_centroids=centroids, ovecs_out=ovecs_out)
    # idx -> (cx, cy) field box-pick (only frames with a snap)
    pick = {c[0]: (c[1], c[2]) for c in centroids if c is not None and c[1] is not None}
    tp_by = {tp.idx: tp for tp in track}
    order = [tp.idx for tp in track]

    out: list[TrackPoint] = []
    for i, idx in enumerate(order):
        tp = tp_by[idx]
        # lookahead window [idx, idx+lag_k] of field box-picks
        win = [pick[j] for j in order[i:i + lag_k + 1] if j in pick]
        committed = None
        if win and idx in pick:
            base = pick[idx]
            agree = [w for w in win
                     if math.hypot(w[0] - base[0], w[1] - base[1]) < radius]
            if len(agree) / len(win) >= confirm:
                cx = sum(w[0] for w in agree) / len(agree)
                cy = sum(w[1] for w in agree) / len(agree)
                committed = (cx, cy)
        x, y = committed if committed is not None else (tp.x, tp.y)
        # mirror identity.py's constructor: TrackPoint(idx, x, y, state, n_dets)
        out.append(TrackPoint(idx, float(x), float(y), tp.state, tp.n_dets))
    return out, locked_hist, start, radius


def _coh_challenger_per_frame(packs, ovecs, order, win, curl_w=FIELD_COH_CURL_W):
    """Per frame: (challenger_centroid, margin) by accumulated coherent-mass over the last
    `win` frames at each current box's location. `ovecs` is {idx: (positions, vectors)} from
    the production tracker (no second flow pass). margin = (top-runnerup)/top is the causal
    confidence key.

    When `curl_w` > 0, FUSES translational coherence with rotational curl: each signal is
    normalized to [0,1] across the frame's boxes and combined as max(coh_n, curl_w*curl_n).
    The two signals fail disjoint frames (coherence is blind to pure rotation, curl to pure
    translation), so the fused challenger fires where EITHER points confidently — covering
    coherence's blind spots on the laggards (probe: union lift +0.14..+0.25 on miss frames)."""
    pos_of = {idx: i for i, idx in enumerate(order)}
    out = {}
    for idx in order:
        if idx >= len(packs):
            continue
        p = packs[idx]
        if not p.boxes:
            out[idx] = None
            continue
        i0 = pos_of[idx]
        window = []
        for back in range(win):
            j = i0 - back
            if j < 0:
                break
            ov = ovecs.get(order[j])
            if ov is not None:
                window.append(ov)
        coh = [_box_coherent_mass(b, window) for b in p.boxes]
        if curl_w > 0:
            curl = [_box_rotational_curl(b, window) for b in p.boxes]
            cmax = max(coh) or 1.0
            rmax = max(curl) or 1.0
            scores = [max(coh[i] / cmax, curl_w * curl[i] / rmax)
                      for i in range(len(p.boxes))]
        else:
            scores = coh
        bi = max(range(len(scores)), key=lambda i: scores[i])
        top = scores[bi]
        run = max((scores[i] for i in range(len(scores)) if i != bi), default=0.0)
        margin = (top - run) / (top + 1e-6) if top > 0 else 0.0
        out[idx] = (_centroid(p.boxes[bi]), margin)
    return out


def track_field_coh_identity(clip: str | Path, packs: list[FusionPack],
                             lock: LockInfo | None = None, *,
                             gate_radius: float | None = None,
                             frame_wh: tuple[int, int] | None = None,
                             win: int = FIELD_COH_OVR_WIN,
                             tau: float = FIELD_COH_OVR_TAU,
                             c_consec: int = FIELD_COH_OVR_C,
                             far_r: float = FIELD_COH_OVR_FAR,
                             curl_w: float = FIELD_COH_CURL_W,
                             ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """field_lag + coherence FAR-JUMP override (the shipped coherence mode).

    Runs the field_lag tracker, then overrides its emitted (x,y) to a coherent-mass
    challenger when that challenger persistently (>=c_consec frames) disagrees with the
    committed pick, is >far_r radii away, and is confident (margin>=tau). The challenger
    is the box maximizing the directional/temporal COHERENCE of the outlier residual
    vectors (`_box_coherent_mass`) accumulated over a causal window -- the real shape
    drifting rigidly produces coherent vectors; incoherent noise/fakes do not.

    Captures the escape-from-lock-in failure (field_lag drifts onto a fake; the real shape
    is far away with a coherent signal) that the local snap-weight cannot reach. Strictly
    causal (backward-only window + streak). Validated LOO 0.7442 (+0.023 over field_lag)."""
    ovecs_out: list = []
    lag_track, locked_hist, start, radius = track_field_lag_identity(
        clip, packs, lock, gate_radius=gate_radius, frame_wh=frame_wh,
        ovecs_out=ovecs_out)

    ovecs = {idx: (pos, vec) for idx, pos, vec in ovecs_out}
    order = [tp.idx for tp in lag_track]
    chal = _coh_challenger_per_frame(packs, ovecs, order, win, curl_w=curl_w)

    far_px = far_r * radius
    streak = 0
    out: list[TrackPoint] = []
    for tp in lag_track:
        x, y = tp.x, tp.y
        ch = chal.get(tp.idx)
        if ch is not None:
            (chx, chy), margin = ch
            if math.hypot(chx - tp.x, chy - tp.y) > far_px and margin >= tau:
                streak += 1
            else:
                streak = 0
            if streak >= c_consec:
                x, y = chx, chy
        else:
            streak = 0
        out.append(TrackPoint(tp.idx, float(x), float(y), tp.state, tp.n_dets))
    return out, locked_hist, start, radius


# Causal fused-path integrator (mode "fpath"). The Viterbi-ceiling diagnostic showed
# a causal (K=0) cumulative-path decode over the field's saliency-mass signal beats
# greedy `field` by up to +0.24 on strong-signal clips (t10 0.70->0.94) but LOSES on
# weak/misleading-signal clips because mass-only emission drops field's positional
# (CV-velocity) prior. This integrator restores that prior: the per-frame emission
# FUSES saliency mass + proximity to a running constant-velocity prediction, and a
# spatial-continuity transition penalty (physics: true shape never moves >~1 radius/
# frame) accumulates over a forward trellis. Fully causal (no lookahead) so it ships
# live. Decode = argmax cumulative score each frame -> that box's centroid.
FPATH_TRANS_W = 1.0       # transition penalty weight (jump in radii, squared)
FPATH_PROX_W = 0.0        # weight on CV-proximity emission (0 = pure mass; prox HURTS, see CLAUDE.md)
FPATH_PROX_SIGMA = 0.6    # proximity gaussian width in radii
FPATH_VEL_DAMP = 0.7      # CV prediction velocity damping (matches tracker VEL_DAMP)
FPATH_MASS_EMA = 0.98     # running-scale EMA for absolute mass normalization
FPATH_REACQUIRE_PATIENCE = 8    # frames off-track before teleport to global mass peak
FPATH_REACQUIRE_MASS_FRAC = 0.3 # min normalised mass for reacquire target (do-no-harm gate)
FPATH_TRANS_CAP = 8.0     # cap on transition penalty in radii² — limits lock-in depth
FPATH_COH_W = 0.0         # coherence emission weight; 0 = pure mass (current default)
FPATH_COH_W_DEFAULT = 1.8 # used by fpath_coh mode; swept on all 10 clips, peak at 1.8


def track_fused_path_identity(clip: str | Path, packs: list[FusionPack],
                              lock: LockInfo | None = None, *,
                              gate_radius: float | None = None,
                              frame_wh: tuple[int, int] | None = None,
                              trans_w: float = FPATH_TRANS_W,
                              prox_w: float = FPATH_PROX_W,
                              prox_sigma: float = FPATH_PROX_SIGMA,
                              vel_damp: float = FPATH_VEL_DAMP,
                              mass_ema: float = FPATH_MASS_EMA,
                              reacquire: bool = False,
                              trans_cap: float | None = None,
                              coh_w: float = FPATH_COH_W,
                              ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """Causal Viterbi-style integrator fusing saliency mass + CV-proximity prior.

    Forward trellis over YOLO boxes; state = which box is the real shape this frame.
      emission e_t(i)   = norm_mass(i) + prox_w * exp(-(d_pred(i)/(sigma*r))^2)
                          where d_pred = |centroid_i - CV-predicted position|
      transition c(i,j) = trans_w * (|centroid_i - centroid_j| / r)^2
    alpha[t][i] = e_t(i) + max_j (alpha[t-1][j] - c(i,j)); decode argmax each frame.
    The chosen box's centroid feeds the CV predictor (its own positional memory), so
    the path stays anchored the way field's snap_feedback does -- but globally, not
    greedily.
    """
    from ld.vision.motion import estimate_motion, saliency_map, box_coherence

    clip = Path(clip)
    seed_x, seed_y, radius, start = _seed(packs)  # type: ignore[arg-type]
    lock_frame = lock.frame if lock is not None else start
    fw, fh = frame_wh if frame_wh else (None, None)
    pos = np.array([seed_x, seed_y], np.float32)
    vel = np.zeros(2, np.float32)
    prev_cents: list[tuple[float, float]] | None = None
    prev_alpha: np.ndarray | None = None
    prev_gray: np.ndarray | None = None
    active = False
    lost = 0        # frames since last trusted Viterbi pick
    mass_scale = 0.0   # running EMA of per-frame peak absolute mass (cross-frame conf)

    track: list[TrackPoint] = []
    locked_hist: list[int | None] = []

    src = VideoSource(clip)
    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        gray = cv2.cvtColor(strip_pointer(raw, strip_green=True), cv2.COLOR_BGR2GRAY)

        if p.idx < lock_frame or math.isnan(pos[0]):
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(None)
            if p.white is not None:
                pos = np.array([p.white[0], p.white[1]], np.float32)
            prev_gray = gray
            continue

        if p.idx == lock_frame and not active:
            cx, cy = (lock.cx, lock.cy) if lock is not None else (float(pos[0]), float(pos[1]))
            pos = np.array([cx, cy], np.float32)
            vel[:] = 0.0
            active = True
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(None)
            prev_gray = gray
            continue

        x, y, state = float(pos[0]), float(pos[1]), "coast"
        if prev_gray is not None and active and p.boxes:
            fld = estimate_motion(prev_gray, gray)
            sal = saliency_map(fld, gray.shape)
            cents = [_centroid(b) for b in p.boxes]
            mass = np.array([_box_saliency_mass(b, sal) for b in p.boxes], np.float32)
            # cross-frame confidence: scale by a running EMA of peak mass, NOT the
            # per-frame max (which erases the weak/strong-frame distinction). A
            # genuinely low-signal frame -> low emission everywhere -> the prior +
            # transition carry the path (field's coast behavior, globally optimal).
            fmax = float(mass.max())
            mass_scale = fmax if mass_scale == 0.0 else max(fmax, mass_ema * mass_scale + (1 - mass_ema) * fmax)
            mass = mass / mass_scale if mass_scale > 0 else mass
            if coh_w > 0:
                coh = np.array([box_coherence(b, fld) for b in p.boxes], np.float32)
            else:
                coh = np.zeros(len(p.boxes), np.float32)
            pred = pos + vel
            emis = np.empty(len(cents), np.float32)
            for i, (cx, cy) in enumerate(cents):
                d = math.hypot(cx - pred[0], cy - pred[1])
                prox = math.exp(-(d / (prox_sigma * radius)) ** 2)
                emis[i] = mass[i] * (1.0 + coh_w * coh[i]) + prox_w * prox
            if prev_alpha is None or prev_cents is None:
                alpha = emis.copy()
            else:
                alpha = np.empty(len(cents), np.float32)
                for i, (cx, cy) in enumerate(cents):
                    best = -1e18
                    for j, (px, py) in enumerate(prev_cents):
                        dd = math.hypot(cx - px, cy - py) / radius
                        cost = trans_w * dd * dd
                        if trans_cap is not None:
                            cost = min(cost, trans_cap)
                        v = prev_alpha[j] - cost
                        if v > best:
                            best = v
                    alpha[i] = best + emis[i]
            # global re-centering: keep alpha bounded (subtract max), no decision change
            alpha = alpha - float(alpha.max())
            choice = int(np.argmax(alpha))
            # Reacquire: if off-track for long enough, override Viterbi with global
            # mass peak (same teleport logic as OutlierTracker). Only fires when a
            # trustworthy target exists (mass gate). Resets path memory on teleport.
            if reacquire and lost >= FPATH_REACQUIRE_PATIENCE:
                best_mass_idx = int(np.argmax(mass))
                if mass_scale > 0 and mass[best_mass_idx] >= FPATH_REACQUIRE_MASS_FRAC:
                    choice = best_mass_idx
                    prev_alpha, prev_cents = None, None
                    vel[:] = 0.0
            nx, ny = cents[choice]
            vel = vel_damp * vel + (1.0 - vel_damp) * (np.array([nx, ny], np.float32) - pos)
            x, y, state = nx, ny, "track"
            prev_alpha, prev_cents = alpha, cents
            lost = 0
        else:
            # Coast: advance via CV prediction, wipe trellis (no boxes to anchor)
            pos = pos + vel
            vel = vel * vel_damp
            prev_alpha, prev_cents = None, None
            lost += 1

        if fw is not None:
            x = float(np.clip(x, 0, fw - 1))
            y = float(np.clip(y, 0, fh - 1))
        pos = np.array([x, y], np.float32)
        track.append(TrackPoint(p.idx, x, y, state, len(p.boxes)))
        locked_hist.append(None)
        prev_gray = gray

    src.release()
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
    prev_locked_box: tuple | None = None
    prev_boxes: list[tuple] = []
    pos = np.array([float("nan"), float("nan")], np.float32)
    vel = np.zeros(2, np.float32)
    _, _, _, start = _seed(packs)  # type: ignore[arg-type]
    lock_frame = lock.frame if lock else start
    gate = GATE_RADIUS

    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        frame = strip_pointer(raw, strip_green=True)

        if lock is not None and p.idx >= lock_frame and p.boxes:
            if p.idx == lock_frame and prev_locked_box is None:
                tracks, next_tid, locked_tid = _locked_tid_on_frame(p.boxes, lock, next_tid)
                prev_locked_box = lock.box
                pos = np.array([lock.cx, lock.cy], np.float32)
                prev_boxes = list(p.boxes)
            else:
                tracks, next_tid = _associate(tracks, p.boxes, next_tid)
                pred = pos + vel
                resid = _box_rigid_residuals(prev_boxes, p.boxes) if prev_boxes else None
                chained = _chain_locked_box(p.boxes, prev_locked_box, pred, gate, resid)
                if chained is not None:
                    prev_locked_box = chained
                    locked_tid = _tid_for_box(tracks, chained)
                    cx, cy = _centroid(chained)
                    pos = np.array([cx, cy], np.float32)
                prev_boxes = list(p.boxes)

        chosen_tid = locked_hist[idx] if idx < len(locked_hist) else locked_tid
        tp = tp_by.get(idx)
        for t in tracks:
            is_locked = t.tid == chosen_tid
            if tp is not None and not math.isnan(tp.x) and p.boxes:
                # Highlight box nearest the tracked position (mode-agnostic).
                nearest = min(p.boxes,
                              key=lambda b: math.hypot(_centroid(b)[0] - tp.x,
                                                       _centroid(b)[1] - tp.y))
                is_locked = _iou(t.box, nearest) >= IOU_MATCH_MIN
            col = (0, 255, 0) if is_locked else (255, 180, 0)
            thick = 3 if is_locked else 1
            b = t.box
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, thick)

        if p.gt is not None:
            cv2.drawMarker(frame, (int(p.gt[0]), int(p.gt[1])), (0, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 18, 2)
        if tp is not None and not math.isnan(tp.x):
            col = (0, 0, 255) if tp.state == "track" else (0, 140, 255)
            cv2.circle(frame, (int(tp.x), int(tp.y)), 10, col, 2)
        hud = f"f{idx} dets={len(p.boxes)} lock={chosen_tid}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()
    src.release()


# Modes routed through track_paper_identity (pick_mode == mode). Everything else
# is dispatched specially in _dispatch_mode (outlier tracker, or chain fallback).
_PAPER_PICK_MODES = (
    "paper", "paper_outlier", "paper_outlier_rank",
)

# All selectable modes, in leaderboard order. Single source of truth shared by
# run_clip's argparse and the eval_modes harness so the two never drift.
ALL_MODES = ("chain", *_PAPER_PICK_MODES, "outlier", "accum", "field", "field_lag", "field_coh", "fpath", "fpath_coh", "fpath_reacq")


def _dispatch_mode(clip: Path, packs: list[FusionPack], lock: LockInfo | None,
                   frame_wh: tuple[int, int], mode: str
                   ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """Run one identity mode -> (track, locked_hist, start, radius).

    Single dispatch point for every pick mode, reused by both run_clip and the
    eval_modes leaderboard so a mode behaves identically in both.
    """
    if mode in _PAPER_PICK_MODES:
        return track_paper_identity(clip, packs, lock, frame_wh=frame_wh, pick_mode=mode)
    if mode == "accum":
        return track_accum_identity(clip, packs, lock, frame_wh=frame_wh)
    if mode == "field":
        return track_field_identity(clip, packs, lock, frame_wh=frame_wh)
    if mode == "field_lag":
        return track_field_lag_identity(clip, packs, lock, frame_wh=frame_wh)
    if mode == "field_coh":
        return track_field_coh_identity(clip, packs, lock, frame_wh=frame_wh)
    if mode == "fpath":
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh)
    if mode == "fpath_coh":
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh,
                                         coh_w=FPATH_COH_W_DEFAULT)
    if mode == "fpath_reacq":
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh,
                                         reacquire=True,
                                         trans_cap=FPATH_TRANS_CAP)
        # NOTE: trans_cap=8.0 helps t4/t8 (+0.018/+0.007) but hurts t3/t5 (-0.040/-0.084)
        # net -0.010 on all 10 clips. Same regime-coupling as prior fixes. Dead end.
    if mode == "outlier":
        from ld.detect.outlier_track import track_outlier_identity
        return track_outlier_identity(clip, packs, lock, frame_wh=frame_wh)
    return track_identity(packs, lock, frame_wh=frame_wh)


def run_clip(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
             imgsz: int = 768, use_cache: bool = True, evidence: bool = False,
             mode: str = "field_coh") -> IdentityReport:
    clip = Path(clip)
    name = clip.stem.replace("_cropped_trimmed", "")
    packs = detect_fusion_clip(weights, clip, conf=conf, imgsz=imgsz, use_cache=use_cache)
    lock = compute_countdown_lock(packs, clip)
    src = VideoSource(clip)
    wh = (src.meta.width, src.meta.height)
    src.release()
    track, locked_hist, start, radius = _dispatch_mode(clip, packs, lock, wh, mode)
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
    ap.add_argument("--mode",
                    choices=ALL_MODES,
                    default="field_coh",
                    help="field_coh=default, leader (field_lag + coherence far-jump "
                         "override); field_lag=fixed-lag confirmation smoother over field; "
                         "field=underlying position-field saliency + YOLO snap; "
                         "accum=causal per-tracklet evidence + rotation; "
                         "paper_outlier_rank/paper/etc=older per-frame baselines; "
                         "see ld/detect/LEADERBOARD.md for the full ranking")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs] if args.inputs else _default_inputs()
    if not inputs:
        raise SystemExit("No input clips found.")

    reports: list[IdentityReport] = []
    for clip in inputs:
        print(f"[{clip.stem}] identity tracking ...")
        rep = run_clip(args.weights, clip, conf=args.conf, imgsz=args.imgsz,
                       use_cache=not args.no_cache, evidence=args.evidence,
                       mode=args.mode)
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
