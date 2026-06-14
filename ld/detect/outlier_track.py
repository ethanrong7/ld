"""Outlier-only persistent identity tracker.

Pipeline (causal):
  1. Estimate global paper translation per frame pair (phaseCorrelate + box median).
  2. Hungarian-match detections to existing tracks.
  3. Classify each track: inlier (rides paper) vs outlier (independent motion).
  4. Pick the real shape among *persistent outlier* tracks only (min streak/hits).
  5. Kalman-style coast when the chosen track is briefly missed.

Usage:
    python -m ld.detect.outlier_track --weights data/detect/runs/yolov8n_probe/weights/best.pt
    python -m ld.detect.identity --weights .../best.pt --mode outlier
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ld.capture.video_source import VideoSource
from ld.config import DATA_DIR, DETECT_DIR, GATE_RADIUS
from ld.detect.constellation import TrackPoint, _seed
from ld.detect.fusion import FusionPack, detect_fusion_clip
from ld.detect.identity import LockInfo, compute_countdown_lock, score_identity
from ld.vision.cursor import strip_pointer

__all__ = ["track_outlier_identity", "run_clip", "advance_outlier_tracks",
           "outlier_switch_box", "outlier_switch_rank_box", "init_outlier_tracks",
           "residual_rank", "update_low_rank_streak"]

# Tunables
UPDATE_ALPHA = 0.55
VEL_DAMP = 0.7
VEL_MAX = 8.0
COAST_PATIENCE = 15
IOU_MATCH_MIN = 0.15
TRACK_MISS_MAX = 8
OUTLIER_THRESH = 2.5       # px; used for inlier-refined shift only
OUTLIER_TOP_FRAC = 0.18    # top fraction by residual classified outlier this frame
MIN_OUTLIER_STREAK = 3
MIN_OUTLIER_HITS = 4
MIN_SWITCH_STREAK = 8     # alt track needs this to steal from paper
MIN_SWITCH_HITS = 8
PAPER_SWITCH_LOST = 3     # coast frames only — no split-streak switching
OUTLIER_SCORE_EMA = 0.35
LOCK_TRACK_BONUS = 6.0
SWITCH_MARGIN = 4.0
RANK_TOP_K = 2            # paper pick should land in top-K by residual
RANK_LOW_STREAK = 9       # consecutive frames outside top-K before switch
RANK_SWITCH_MIN_STREAK = 4
RANK_SWITCH_MIN_HITS = 4
RANK_RESID_MARGIN = 0.4   # top residual must beat paper pick by this (px)
SHIFT_REFINE_ITERS = 2
HUNGARIAN_COST_MAX = 1e6


@dataclass
class OutlierTrack:
    tid: int
    box: tuple[float, float, float, float, float]
    cx: float
    cy: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    hits: int = 1
    streak: int = 1
    outlier_hits: int = 0
    outlier_streak: int = 0
    outlier_score_ema: float = 0.0
    misses: int = 0


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


def _phase_shift(prev_gray: np.ndarray, cur_gray: np.ndarray) -> tuple[float, float]:
    prev_f = prev_gray.astype(np.float64)
    cur_f = cur_gray.astype(np.float64)
    (dx, dy), _ = cv2.phaseCorrelate(prev_f, cur_f)
    return float(dx), float(dy)


def _box_median_shift(prev_boxes: list[tuple], curr_boxes: list[tuple],
                      matches: dict[int, int]) -> tuple[float, float] | None:
    if len(matches) < 3:
        return None
    shifts: list[tuple[float, float]] = []
    for ti, bi in matches.items():
        px, py = _centroid(prev_boxes[ti])
        cx, cy = _centroid(curr_boxes[bi])
        shifts.append((cx - px, cy - py))
    return float(np.median([s[0] for s in shifts])), float(np.median([s[1] for s in shifts]))


def _refine_paper_shift(prev_boxes: list[tuple], curr_boxes: list[tuple],
                        tracks: list[OutlierTrack], matches: dict[int, int],
                        dx: float, dy: float) -> tuple[float, float]:
    """Re-fit shift using only low-residual (paper-following) matches."""
    for _ in range(SHIFT_REFINE_ITERS):
        inlier_shifts: list[tuple[float, float]] = []
        for ti, bi in matches.items():
            if ti >= len(tracks):
                continue
            tr = tracks[ti]
            resid = _motion_residual(tr, curr_boxes[bi], dx, dy)
            if resid <= OUTLIER_THRESH:
                px, py = tr.cx, tr.cy
                cx, cy = _centroid(curr_boxes[bi])
                inlier_shifts.append((cx - px, cy - py))
        if len(inlier_shifts) >= 4:
            dx = float(np.median([s[0] for s in inlier_shifts]))
            dy = float(np.median([s[1] for s in inlier_shifts]))
    return dx, dy


def _outlier_flags(residuals: list[float]) -> list[bool]:
    """Relative outlier: top OUTLIER_TOP_FRAC by residual this frame."""
    n = len(residuals)
    if n == 0:
        return []
    top_k = max(1, int(math.ceil(n * OUTLIER_TOP_FRAC)))
    order = sorted(range(n), key=lambda i: residuals[i], reverse=True)
    top = set(order[:top_k])
    med = float(np.median(residuals))
    mad = float(np.median([abs(r - med) for r in residuals])) + 1e-6
    hard = med + max(3.0, 2.5 * mad)
    return [i in top or residuals[i] >= hard for i in range(n)]


def _estimate_paper_shift(prev_gray: np.ndarray | None, cur_gray: np.ndarray,
                          prev_boxes: list[tuple], curr_boxes: list[tuple],
                          matches: dict[int, int]) -> tuple[float, float]:
    box_shift = _box_median_shift(prev_boxes, curr_boxes, matches)
    if prev_gray is not None:
        phase_dx, phase_dy = _phase_shift(prev_gray, cur_gray)
        if box_shift is not None:
            w = min(1.0, len(matches) / 8.0)
            return ((1.0 - w) * phase_dx + w * box_shift[0],
                    (1.0 - w) * phase_dy + w * box_shift[1])
        return phase_dx, phase_dy
    if box_shift is not None:
        return box_shift
    return 0.0, 0.0


def _hungarian_match(tracks: list[OutlierTrack], boxes: list[tuple],
                     max_dist: float, *, use_pred: bool = True) -> tuple[dict[int, int], set[int], set[int]]:
    """Map track index -> box index (distance + IoU, gated)."""
    if not tracks or not boxes:
        return {}, set(), set()
    n_t, n_b = len(tracks), len(boxes)
    cost = np.full((n_t, n_b), HUNGARIAN_COST_MAX, np.float64)
    for i, tr in enumerate(tracks):
        px = float(tr.cx + tr.vel[0]) if use_pred else tr.cx
        py = float(tr.cy + tr.vel[1]) if use_pred else tr.cy
        for j, box in enumerate(boxes):
            cx, cy = _centroid(box)
            dist = math.hypot(cx - px, cy - py)
            iou = _iou(tr.box, box)
            if dist > max_dist and iou < IOU_MATCH_MIN:
                continue
            cost[i, j] = dist - 3.0 * iou
    row, col = linear_sum_assignment(cost)
    matches: dict[int, int] = {}
    matched_t: set[int] = set()
    matched_b: set[int] = set()
    for r, c in zip(row, col):
        if cost[r, c] < HUNGARIAN_COST_MAX / 2:
            matches[r] = c
            matched_t.add(r)
            matched_b.add(c)
    return matches, matched_t, matched_b


def _motion_residual(tr: OutlierTrack, box: tuple, paper_dx: float, paper_dy: float) -> float:
    cx, cy = _centroid(box)
    disp_x = cx - tr.cx
    disp_y = cy - tr.cy
    return math.hypot(disp_x - paper_dx, disp_y - paper_dy)


def _update_outlier_track(tr: OutlierTrack, box: tuple, paper_dx: float, paper_dy: float,
                          *, is_outlier: bool, resid: float) -> None:
    cx, cy = _centroid(box)
    tr.vel = VEL_DAMP * tr.vel + (1.0 - VEL_DAMP) * np.array([cx - tr.cx, cy - tr.cy], np.float32)
    sp = float(np.hypot(*tr.vel))
    if sp > VEL_MAX:
        tr.vel *= VEL_MAX / sp
    tr.box = box
    tr.cx, tr.cy = cx, cy
    tr.hits += 1
    tr.streak += 1
    tr.misses = 0
    if is_outlier:
        tr.outlier_hits += 1
        tr.outlier_streak += 1
        tr.outlier_score_ema = (OUTLIER_SCORE_EMA * tr.outlier_score_ema
                                + (1.0 - OUTLIER_SCORE_EMA) * resid)
    else:
        tr.outlier_streak = 0


def _spawn_track(box: tuple, tid: int, paper_dx: float, paper_dy: float,
                 prev_tr: OutlierTrack | None) -> OutlierTrack:
    cx, cy = _centroid(box)
    tr = OutlierTrack(tid, box, cx, cy)
    if prev_tr is not None:
        resid = _motion_residual(prev_tr, box, paper_dx, paper_dy)
        if resid >= OUTLIER_THRESH:
            tr.outlier_hits = 1
            tr.outlier_streak = 1
            tr.outlier_score_ema = resid
    return tr


def _track_outlier_score(tr: OutlierTrack, lock_tid: int | None, *, qualified_only: bool = True) -> float:
    if qualified_only and tr.outlier_streak < MIN_OUTLIER_STREAK and tr.outlier_hits < MIN_OUTLIER_HITS:
        return -1e9
    score = tr.outlier_score_ema + 0.6 * tr.outlier_streak + 0.15 * tr.outlier_hits
    if lock_tid is not None and tr.tid == lock_tid:
        score += LOCK_TRACK_BONUS
    return score


def _pick_real_track(tracks: list[OutlierTrack], lock_tid: int | None,
                     lock_box: tuple | None) -> OutlierTrack | None:
    if not tracks:
        return None
    lock_tr = next((tr for tr in tracks if lock_tid is not None and tr.tid == lock_tid), None)

    best_alt: OutlierTrack | None = None
    best_alt_score = -1e9
    for tr in tracks:
        if lock_tr is not None and tr.tid == lock_tr.tid:
            continue
        s = _track_outlier_score(tr, lock_tid)
        if s > best_alt_score:
            best_alt_score, best_alt = s, tr

    if lock_tr is None:
        if best_alt is not None:
            return best_alt
        if lock_box is not None:
            return max(tracks, key=lambda tr: _iou(tr.box, lock_box))
        return max(tracks, key=lambda tr: (tr.outlier_streak, tr.outlier_score_ema))

    lock_score = _track_outlier_score(lock_tr, lock_tid, qualified_only=False)
    if (best_alt is not None and best_alt_score >= lock_score + SWITCH_MARGIN
            and (best_alt.outlier_streak >= MIN_OUTLIER_STREAK
                 or best_alt.outlier_hits >= MIN_OUTLIER_HITS)):
        return best_alt
    return lock_tr


def _locked_tid_on_frame(boxes: list[tuple], lock: LockInfo, next_tid: int
                         ) -> tuple[list[OutlierTrack], int, int]:
    tracks: list[OutlierTrack] = []
    for box in boxes:
        cx, cy = _centroid(box)
        tracks.append(OutlierTrack(next_tid, box, cx, cy))
        next_tid += 1
    lock_i = max(range(len(boxes)), key=lambda i: _iou(boxes[i], lock.box))
    return tracks, next_tid, tracks[lock_i].tid


def init_outlier_tracks(boxes: list[tuple], lock: LockInfo,
                        next_tid: int = 0) -> tuple[list[OutlierTrack], int, int]:
    return _locked_tid_on_frame(boxes, lock, next_tid)


def advance_outlier_tracks(tracks: list[OutlierTrack], next_tid: int,
                           prev_boxes: list[tuple], prev_gray: np.ndarray | None,
                           gray: np.ndarray, boxes: list[tuple],
                           assoc_gate: float) -> tuple[list[OutlierTrack], int]:
    """Hungarian tracks + relative outlier scoring for one frame."""
    if not boxes:
        for tr in tracks:
            tr.misses += 1
        return [tr for tr in tracks if tr.misses <= TRACK_MISS_MAX], next_tid

    paper_dx, paper_dy = 0.0, 0.0
    if prev_boxes:
        tmp_tracks = [OutlierTrack(-1, pb, *_centroid(pb)) for pb in prev_boxes]
        pre_match, _, _ = _hungarian_match(tmp_tracks, boxes, assoc_gate)
        paper_dx, paper_dy = _estimate_paper_shift(prev_gray, gray, prev_boxes, boxes, pre_match)

    matches, _, matched_b = _hungarian_match(tracks, boxes, assoc_gate)
    if matches:
        paper_dx, paper_dy = _refine_paper_shift(prev_boxes, boxes, tracks, matches, paper_dx, paper_dy)

    match_residuals: list[float] = []
    match_ti: list[int] = []
    for ti, bi in matches.items():
        match_ti.append(ti)
        match_residuals.append(_motion_residual(tracks[ti], boxes[bi], paper_dx, paper_dy))
    outlier_by_ti = {
        match_ti[i]: flag for i, flag in enumerate(_outlier_flags(match_residuals))
    }

    updated: list[OutlierTrack] = []
    for ti, tr in enumerate(tracks):
        if ti in matches:
            box = boxes[matches[ti]]
            resid = _motion_residual(tr, box, paper_dx, paper_dy)
            _update_outlier_track(tr, box, paper_dx, paper_dy,
                                  is_outlier=outlier_by_ti.get(ti, False), resid=resid)
            updated.append(tr)
        else:
            tr.misses += 1
            tr.streak = 0
            tr.outlier_streak = 0
            if tr.misses <= TRACK_MISS_MAX:
                updated.append(tr)

    for bi, box in enumerate(boxes):
        if bi not in matched_b:
            prev_near = max(updated, key=lambda tr: _iou(tr.box, box)) if updated else None
            updated.append(_spawn_track(
                box, next_tid, paper_dx, paper_dy,
                prev_near if prev_near and _iou(prev_near.box, box) >= IOU_MATCH_MIN else None))
            next_tid += 1

    return updated, next_tid


def _best_switch_candidate(tracks: list[OutlierTrack], lock_tid: int | None,
                           exclude_box: tuple | None) -> OutlierTrack | None:
    best: OutlierTrack | None = None
    best_score = -1e9
    for tr in tracks:
        if exclude_box is not None and _iou(tr.box, exclude_box) >= IOU_MATCH_MIN:
            continue
        if tr.outlier_streak < MIN_SWITCH_STREAK and tr.outlier_hits < MIN_SWITCH_HITS:
            continue
        s = tr.outlier_score_ema + 0.6 * tr.outlier_streak + 0.15 * tr.outlier_hits
        if s > best_score:
            best_score, best = s, tr
    return best


def outlier_switch_box(tracks: list[OutlierTrack], lock_tid: int | None,
                       paper_box: tuple | None, *, lost: int) -> tuple | None:
    """Return alt box when coasting and a persistent outlier track wins."""
    if not tracks or lost < PAPER_SWITCH_LOST:
        return None

    paper_tr = None
    paper_score = -1e9
    if paper_box is not None:
        paper_tr = max(tracks, key=lambda tr: _iou(tr.box, paper_box))
        if _iou(paper_tr.box, paper_box) >= IOU_MATCH_MIN:
            paper_score = (paper_tr.outlier_score_ema + 0.6 * paper_tr.outlier_streak
                           + 0.15 * paper_tr.outlier_hits)

    alt = _best_switch_candidate(tracks, lock_tid, paper_box)
    if alt is None:
        return None
    alt_score = alt.outlier_score_ema + 0.6 * alt.outlier_streak + 0.15 * alt.outlier_hits
    if alt_score < paper_score + SWITCH_MARGIN:
        return None
    return alt.box


def residual_rank(boxes: list[tuple], chosen_box: tuple | None,
                  resid: list[float]) -> int:
    """1 = highest residual; len(boxes)+1 if chosen not among boxes."""
    if not boxes or chosen_box is None or not resid:
        return 1
    ci = max(range(len(boxes)), key=lambda i: _iou(boxes[i], chosen_box))
    if _iou(boxes[ci], chosen_box) < IOU_MATCH_MIN:
        return len(boxes) + 1
    r = resid[ci]
    return 1 + sum(1 for x in resid if x > r + 1e-9)


def update_low_rank_streak(rank: int, streak: int) -> int:
    """Increment when paper pick is outside top-K residual rank."""
    if rank > RANK_TOP_K:
        return streak + 1
    return 0


def _best_rank_switch_candidate(tracks: list[OutlierTrack],
                              exclude_box: tuple | None) -> OutlierTrack | None:
    best: OutlierTrack | None = None
    best_score = -1e9
    for tr in tracks:
        if exclude_box is not None and _iou(tr.box, exclude_box) >= IOU_MATCH_MIN:
            continue
        if (tr.outlier_streak < RANK_SWITCH_MIN_STREAK
                and tr.outlier_hits < RANK_SWITCH_MIN_HITS):
            continue
        s = tr.outlier_score_ema + 0.6 * tr.outlier_streak + 0.15 * tr.outlier_hits
        if s > best_score:
            best_score, best = s, tr
    return best


def outlier_switch_rank_box(tracks: list[OutlierTrack], boxes: list[tuple],
                            resid_pick: list[float], paper_box: tuple | None,
                            *, low_rank_streak: int) -> tuple | None:
    """Switch when paper pick ranks poorly vs residuals for several frames."""
    if (low_rank_streak < RANK_LOW_STREAK or not boxes or not resid_pick
            or paper_box is None):
        return None

    ci = max(range(len(boxes)), key=lambda i: _iou(boxes[i], paper_box))
    if _iou(boxes[ci], paper_box) < IOU_MATCH_MIN:
        return None
    paper_resid = resid_pick[ci]

    top_i = max(range(len(boxes)), key=lambda i: resid_pick[i])
    if resid_pick[top_i] < paper_resid + RANK_RESID_MARGIN:
        return None
    top_box = boxes[top_i]
    if _iou(top_box, paper_box) >= IOU_MATCH_MIN:
        return None

    alt = _best_rank_switch_candidate(tracks, paper_box)
    if alt is not None:
        alt_resid_i = max(range(len(boxes)), key=lambda i: _iou(boxes[i], alt.box))
        if (_iou(boxes[alt_resid_i], alt.box) >= IOU_MATCH_MIN
                and resid_pick[alt_resid_i] >= paper_resid + RANK_RESID_MARGIN):
            return alt.box

    top_tr = max(tracks, key=lambda tr: _iou(tr.box, top_box)) if tracks else None
    if (top_tr is not None and _iou(top_tr.box, top_box) >= IOU_MATCH_MIN
            and (top_tr.outlier_streak >= 2 or top_tr.outlier_hits >= 3)):
        return top_box

    if low_rank_streak >= RANK_LOW_STREAK + 2:
        return top_box
    return None


def update_split_streak(tracks: list[OutlierTrack], paper_box: tuple | None,
                        split_streak: int) -> int:
    """Count consecutive frames paper pick disagrees with best persistent outlier."""
    if paper_box is None or not tracks:
        return 0
    alt = _best_switch_candidate(tracks, None, paper_box)
    if alt is not None and _iou(alt.box, paper_box) < IOU_MATCH_MIN:
        return split_streak + 1
    return 0


def track_outlier_identity(clip: str | Path, packs: list[FusionPack],
                           lock: LockInfo | None = None, *,
                           gate_radius: float | None = None,
                           frame_wh: tuple[int, int] | None = None
                           ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    clip = Path(clip)
    seed_x, seed_y, radius, start = _seed(packs)  # type: ignore[arg-type]
    lock_frame = lock.frame if lock is not None else start
    gate = gate_radius or max(GATE_RADIUS, 1.3 * radius)
    assoc_gate = max(gate, 1.5 * radius)

    pos = np.array([seed_x, seed_y], np.float32)
    vel = np.zeros(2, np.float32)
    lost = 0
    lock_tid: int | None = None
    lock_box: tuple | None = lock.box if lock is not None else None
    tracks: list[OutlierTrack] = []
    next_tid = 0
    chosen_tid: int | None = None
    prev_boxes: list[tuple] = []
    prev_gray: np.ndarray | None = None
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

        if p.idx == lock_frame and lock_tid is None:
            if lock is not None and p.boxes:
                tracks, next_tid, lock_tid = _locked_tid_on_frame(p.boxes, lock, next_tid)
                lock_box = lock.box
                pos = np.array([lock.cx, lock.cy], np.float32)
                vel[:] = 0.0
                chosen_tid = lock_tid
            elif p.boxes:
                for box in p.boxes:
                    cx, cy = _centroid(box)
                    tracks.append(OutlierTrack(next_tid, box, cx, cy))
                    next_tid += 1
                lock_tid = tracks[0].tid
                chosen_tid = lock_tid
                lock_box = tracks[0].box
                pos = np.array([tracks[0].cx, tracks[0].cy], np.float32)
            track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), "acquire", len(p.boxes)))
            locked_hist.append(lock_tid)
            prev_boxes = list(p.boxes)
            prev_gray = gray
            continue

        pos_prev = pos.copy()
        paper_dx, paper_dy = 0.0, 0.0
        if prev_boxes and p.boxes:
            tmp_tracks = [
                OutlierTrack(-1, pb, *_centroid(pb)) for pb in prev_boxes
            ]
            pre_match, _, _ = _hungarian_match(tmp_tracks, p.boxes, assoc_gate)
            paper_dx, paper_dy = _estimate_paper_shift(
                prev_gray, gray, prev_boxes, p.boxes, pre_match)

        matches, matched_t, matched_b = _hungarian_match(tracks, p.boxes, assoc_gate)
        if matches:
            paper_dx, paper_dy = _refine_paper_shift(
                prev_boxes, p.boxes, tracks, matches, paper_dx, paper_dy)

        match_residuals: list[float] = []
        match_ti: list[int] = []
        for ti, bi in matches.items():
            match_ti.append(ti)
            match_residuals.append(_motion_residual(tracks[ti], p.boxes[bi], paper_dx, paper_dy))
        outlier_by_ti = {
            match_ti[i]: flag
            for i, flag in enumerate(_outlier_flags(match_residuals))
        }

        updated: list[OutlierTrack] = []
        for ti, tr in enumerate(tracks):
            if ti in matches:
                box = p.boxes[matches[ti]]
                resid = _motion_residual(tr, box, paper_dx, paper_dy)
                is_out = outlier_by_ti.get(ti, False)
                _update_outlier_track(tr, box, paper_dx, paper_dy,
                                      is_outlier=is_out, resid=resid)
                updated.append(tr)
            else:
                tr.misses += 1
                tr.streak = 0
                tr.outlier_streak = 0
                if tr.misses <= TRACK_MISS_MAX:
                    updated.append(tr)

        for bi, box in enumerate(p.boxes):
            if bi not in matched_b:
                prev_near = max(updated, key=lambda tr: _iou(tr.box, box)) if updated else None
                updated.append(_spawn_track(
                    box, next_tid, paper_dx, paper_dy,
                    prev_near if prev_near and _iou(prev_near.box, box) >= IOU_MATCH_MIN else None))
                next_tid += 1

        tracks = updated
        real = _pick_real_track(tracks, lock_tid, lock_box)

        if real is not None:
            chosen_tid = real.tid
            target = np.array([real.cx, real.cy], np.float32)
            if p.idx < start:
                pos = target
                state = "acquire"
            else:
                pred = pos + vel
                pos = (1.0 - UPDATE_ALPHA) * pred + UPDATE_ALPHA * target
                vel = VEL_DAMP * vel + (1.0 - VEL_DAMP) * (pos - pos_prev)
                sp = float(np.hypot(*vel))
                if sp > VEL_MAX:
                    vel *= VEL_MAX / sp
                lost = 0
                state = "track"
        elif p.idx >= start:
            pos = pos + vel
            lost += 1
            state = "coast"
            if lost >= COAST_PATIENCE:
                vel[:] = 0.0
        else:
            state = "acquire"

        if fw is not None:
            pos = np.array([float(np.clip(pos[0], 0, fw - 1)),
                            float(np.clip(pos[1], 0, fh - 1))], np.float32)

        track.append(TrackPoint(p.idx, float(pos[0]), float(pos[1]), state, len(p.boxes)))
        locked_hist.append(chosen_tid)
        prev_boxes = list(p.boxes)
        prev_gray = gray

    src.release()
    return track, locked_hist, start, radius


def run_clip(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
             imgsz: int = 768, use_cache: bool = True) -> object:
    clip = Path(clip)
    name = clip.stem.replace("_cropped_trimmed", "")
    packs = detect_fusion_clip(weights, clip, conf=conf, imgsz=imgsz, use_cache=use_cache)
    lock = compute_countdown_lock(packs, clip)
    src = VideoSource(clip)
    wh = (src.meta.width, src.meta.height)
    src.release()
    track, _, start, radius = track_outlier_identity(clip, packs, lock, frame_wh=wh)
    return score_identity(packs, track, start, radius, name)


def _default_inputs() -> list[Path]:
    return sorted(DATA_DIR.glob("t*_cropped_trimmed.mp4"),
                  key=lambda p: int(p.stem.split("_")[0][1:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Outlier-only persistent identity tracker")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs] if args.inputs else _default_inputs()
    reports = []
    for clip in inputs:
        print(f"[{clip.stem}] outlier tracking ...")
        rep = run_clip(args.weights, clip, conf=args.conf, imgsz=args.imgsz,
                       use_cache=not args.no_cache)
        print(f"  {rep}")
        reports.append(rep)

    valid = [r for r in reports if r.n > 0]
    if valid:
        print(f"\nMEAN within_r = {sum(r.within_r for r in valid) / len(valid):.3f}")
        print(f"MEAN oracle   = {sum(r.oracle_within_r for r in valid) / len(valid):.3f}")


if __name__ == "__main__":
    main()
