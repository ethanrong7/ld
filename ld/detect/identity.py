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
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.config import (DATA_DIR, DETECT_DIR, FEAT_MAX, FEAT_MIN_DIST, FEAT_QUALITY,
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
HYBRID_CHAIN_IOU_MIN = 0.30 # prefer IoU chain over paper when signals disagree
PAPER_FREE_IOU_W = 0.25     # continuity to prev box only; no pred proximity gate
PAPER_PROX_W = 0.30         # weight on pred proximity in default paper pick
PAPER_REACQ_LOST = 5        # global paper pick after this many coast frames
HYP_K = 5
HYP_WINDOW = 25
HYP_SCORE_EMA = 0.40          # per-hypothesis residual smooth
GRAPH_WINDOW = 30               # Viterbi lookback (frames)
GRAPH_IOU_W = 0.35              # continuity bonus on graph edges (light)
GRAPH_DIST_W = 0.15             # weak edge when centroid within gate but IoU low
GRAPH_LOCK_BONUS = 3.0          # seed path at countdown lock box
GRAPH_DECAY = 0.94              # recent frames weigh more in path score
GRAPH_SWITCH_MARGIN = 2.0       # trajectory wins over paper only above this gap
TRAJ_REACQ_LOST = 3             # coast frames before Viterbi re-acquire
PAPER_INLIER_MAX = 2.5   # px; fake boxes used to refine sheet fit
LOCK_MASK_MIN = 0.01     # min mask overlap before anchor fallback
WHITE_TELEPORT_PX = 80.0 # START-text false positive jumps farther than this
RIGID_RANSAC_THRESH = 3.0
IOU_MATCH_MIN = 0.15
TRACK_MISS_MAX = 8


@dataclass
class Hypothesis:
    """One candidate real-shape track accumulated over time."""

    box: tuple[float, float, float, float, float]
    score_ema: float = 0.0
    window: list[float] = field(default_factory=list)
    misses: int = 0


@dataclass
class GraphFrame:
    """One frame in the trajectory-graph buffer."""

    boxes: list[tuple[float, float, float, float, float]]
    resid: list[float]


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


def _paper_resid_margin(resid: list[float]) -> float:
    ranked = sorted(resid, reverse=True)
    if len(ranked) < 2:
        return ranked[0] if ranked else 0.0
    return ranked[0] - ranked[1]


def _hypothesis_score(h: Hypothesis) -> float:
    """Rolling paper-residual sum + EMA; penalize stale tracks."""
    return sum(h.window[-HYP_WINDOW:]) + 0.4 * h.score_ema - 0.15 * h.misses


def _hypothesis_claimed(hyps: list[Hypothesis], box: tuple) -> bool:
    return any(_iou(h.box, box) >= IOU_MATCH_MIN for h in hyps)


def _init_hypotheses(boxes: list[tuple], resid: list[float],
                     seed_box: tuple | None) -> list[Hypothesis]:
    hyps: list[Hypothesis] = []
    if seed_box is not None:
        hyps.append(Hypothesis(seed_box))
    order = sorted(range(len(boxes)), key=lambda i: resid[i], reverse=True)
    for i in order:
        if len(hyps) >= HYP_K:
            break
        if _hypothesis_claimed(hyps, boxes[i]):
            continue
        hyps.append(Hypothesis(boxes[i], score_ema=resid[i], window=[resid[i]]))
    return hyps


def _update_hypotheses(hyps: list[Hypothesis], boxes: list[tuple],
                       resid: list[float],
                       seed_box: tuple | None) -> tuple[tuple | None, list[Hypothesis]]:
    """Propagate top-K box tracks; pick the highest cumulative paper-residual score."""
    if not boxes:
        for h in hyps:
            h.misses += 1
        hyps = [h for h in hyps if h.misses <= TRACK_MISS_MAX]
        if not hyps:
            return None, hyps
        best = max(hyps, key=_hypothesis_score)
        return best.box, hyps

    if not hyps:
        hyps = _init_hypotheses(boxes, resid, seed_box)
        if not hyps:
            return None, hyps
        return max(hyps, key=_hypothesis_score).box, hyps

    pairs: list[tuple[float, int, int]] = []
    for hi, h in enumerate(hyps):
        for bi, box in enumerate(boxes):
            v = _iou(h.box, box)
            if v >= IOU_MATCH_MIN:
                pairs.append((v, hi, bi))
    pairs.sort(reverse=True)

    used_h: set[int] = set()
    used_b: set[int] = set()
    for _, hi, bi in pairs:
        if hi in used_h or bi in used_b:
            continue
        h = hyps[hi]
        r = resid[bi]
        h.box = boxes[bi]
        h.misses = 0
        h.score_ema = HYP_SCORE_EMA * h.score_ema + (1.0 - HYP_SCORE_EMA) * r
        h.window.append(r)
        if len(h.window) > HYP_WINDOW:
            h.window.pop(0)
        used_h.add(hi)
        used_b.add(bi)

    for hi, h in enumerate(hyps):
        if hi not in used_h:
            h.misses += 1

    hyps = [h for h in hyps if h.misses <= TRACK_MISS_MAX]

    order = sorted(range(len(boxes)), key=lambda i: resid[i], reverse=True)
    for i in order:
        if len(hyps) >= HYP_K:
            break
        if i in used_b or _hypothesis_claimed(hyps, boxes[i]):
            continue
        hyps.append(Hypothesis(boxes[i], score_ema=resid[i], window=[resid[i]]))

    if not hyps:
        return None, hyps
    return max(hyps, key=_hypothesis_score).box, hyps


def _pick_hypothesis_paper_box(hyps: list[Hypothesis], boxes: list[tuple],
                               prev_box: tuple | None, pred: np.ndarray,
                               gate: float, resid_pick: list[float],
                               seed_box: tuple | None
                               ) -> tuple[tuple | None, list[Hypothesis]]:
    """Hypothesis on disagreement only when trajectory score has clear margin."""
    hyp_box, hyps = _update_hypotheses(hyps, boxes, resid_pick, seed_box)
    paper_box = _pick_paper_outlier_box(boxes, prev_box, pred, gate, resid_pick)
    if hyp_box is None:
        return paper_box, hyps
    if paper_box is None:
        return hyp_box, hyps
    if _iou(hyp_box, paper_box) >= IOU_MATCH_MIN:
        return paper_box, hyps
    ranked = sorted(hyps, key=_hypothesis_score, reverse=True)
    if (len(ranked) >= 2
            and _hypothesis_score(ranked[0]) - _hypothesis_score(ranked[1]) >= 1.5):
        return ranked[0].box, hyps
    return paper_box, hyps


def _graph_edge_bonus(prev_box: tuple, curr_box: tuple, gate: float) -> float | None:
    """IoU-linked edge weight, or a weak centroid link when motion is fast."""
    iou = _iou(prev_box, curr_box)
    if iou >= IOU_MATCH_MIN:
        return GRAPH_IOU_W * iou
    cx1, cy1 = _centroid(prev_box)
    cx2, cy2 = _centroid(curr_box)
    dist = float(np.hypot(cx2 - cx1, cy2 - cy1))
    if dist <= gate:
        return GRAPH_DIST_W * (1.0 - min(dist / gate, 1.0))
    return None


def _neg_inf() -> float:
    return -1e9


def _viterbi_trajectory_path(history: list[GraphFrame], *, gate: float,
                             lock_box: tuple | None = None
                             ) -> tuple[list[int], float, list[float]]:
    """Best box-index path, its score, and all end-state scores at the last frame."""
    w = len(history)
    if w == 0:
        return [], _neg_inf(), []
    if not history[-1].boxes:
        return [], _neg_inf(), []

    neg = -1e9
    scores: list[list[float]] = []
    back: list[list[int]] = []

    n0 = len(history[0].boxes)
    if n0 == 0:
        return [], neg, []
    decay = [GRAPH_DECAY ** (w - 1 - f) for f in range(w)]
    s0 = [neg] * n0
    for i, r in enumerate(history[0].resid):
        s0[i] = decay[0] * r
        if lock_box is not None and _iou(history[0].boxes[i], lock_box) >= IOU_MATCH_MIN:
            s0[i] += GRAPH_LOCK_BONUS
    scores.append(s0)
    back.append([-1] * n0)

    for f in range(1, w):
        prev = history[f - 1]
        curr = history[f]
        nf = len(curr.boxes)
        sf = [neg] * max(nf, 1)
        bf = [-1] * max(nf, 1)
        if nf == 0 or not prev.boxes:
            scores.append(sf)
            back.append(bf)
            continue
        for j in range(nf):
            best_s, best_i = neg, -1
            for i, pb in enumerate(prev.boxes):
                if scores[f - 1][i] <= neg / 2:
                    continue
                edge = _graph_edge_bonus(pb, curr.boxes[j], gate)
                if edge is None:
                    continue
                cand = scores[f - 1][i] + decay[f] * curr.resid[j] + edge
                if cand > best_s:
                    best_s, best_i = cand, i
            sf[j] = best_s
            bf[j] = best_i
        scores.append(sf)
        back.append(bf)

    last_f = w - 1
    end_scores = list(scores[last_f])
    if not history[last_f].boxes:
        return [], neg, end_scores
    end_i = max(range(len(history[last_f].boxes)), key=lambda i: scores[last_f][i])
    if scores[last_f][end_i] <= neg / 2:
        end_i = max(range(len(history[last_f].boxes)),
                    key=lambda i: history[last_f].resid[i])
    best_score = scores[last_f][end_i]

    path = [0] * w
    path[last_f] = end_i
    for f in range(last_f - 1, -1, -1):
        nxt = path[f + 1]
        if nxt < len(back[f + 1]) and back[f + 1][nxt] >= 0:
            path[f] = back[f + 1][nxt]
        elif history[f].boxes:
            path[f] = max(range(len(history[f].boxes)), key=lambda i: history[f].resid[i])
    return path, best_score, end_scores


def _trajectory_path_scores(history: list[GraphFrame], *, gate: float,
                            lock_box: tuple | None = None) -> tuple[list[int], float, float]:
    """Return (path, best_score, second_best_score) at the newest frame."""
    path, best, end_scores = _viterbi_trajectory_path(history, gate=gate, lock_box=lock_box)
    if not end_scores:
        return path, best, _neg_inf()
    ranked = sorted(end_scores, reverse=True)
    second = ranked[1] if len(ranked) >= 2 else ranked[0]
    return path, best, second


def _pick_trajectory_graph(history: list[GraphFrame], *, gate: float,
                           lock_box: tuple | None = None) -> tuple | None:
    """Pick the box at the newest buffered frame from the best Viterbi path."""
    if not history or not history[-1].boxes:
        return None
    path, _, _ = _viterbi_trajectory_path(history, gate=gate, lock_box=lock_box)
    if not path:
        return history[-1].boxes[max(range(len(history[-1].resid)),
                                     key=lambda i: history[-1].resid[i])]
    idx = path[-1]
    if 0 <= idx < len(history[-1].boxes):
        return history[-1].boxes[idx]
    return history[-1].boxes[max(range(len(history[-1].resid)),
                                 key=lambda i: history[-1].resid[i])]


def _pick_trajectory_paper_box(history: list[GraphFrame], boxes: list[tuple],
                               prev_box: tuple | None, pred: np.ndarray,
                               gate: float, resid_pick: list[float],
                               lock_box: tuple | None) -> tuple | None:
    """Trajectory when it disagrees with paper and has a clear path margin."""
    traj = _pick_trajectory_graph(history, gate=gate, lock_box=lock_box)
    paper = _pick_paper_outlier_box(boxes, prev_box, pred, gate, resid_pick)
    if traj is None:
        return paper
    if paper is None:
        return traj
    if _iou(traj, paper) >= IOU_MATCH_MIN:
        return paper
    _, best, second = _trajectory_path_scores(history, gate=gate, lock_box=lock_box)
    if best - second >= GRAPH_SWITCH_MARGIN:
        return traj
    return paper


def _pick_trajectory_reacq_box(history: list[GraphFrame], boxes: list[tuple],
                               prev_box: tuple | None, pred: np.ndarray,
                               gate: float, resid_pick: list[float],
                               lock_box: tuple | None, *, lost: int) -> tuple | None:
    """Paper pick by default; Viterbi re-acquire after sustained coast."""
    if lost >= TRAJ_REACQ_LOST and history:
        chosen = _pick_trajectory_graph(history, gate=gate, lock_box=lock_box)
        if chosen is not None:
            return chosen
    return _pick_paper_outlier_box(boxes, prev_box, pred, gate, resid_pick)


def _pick_paper_reacq_box(boxes: list[tuple], prev_box: tuple | None,
                          pred: np.ndarray, gate: float,
                          paper_resid: list[float], *, lost: int) -> tuple | None:
    """Default gated paper pick; global re-acquire when margin clear or coasting."""
    margin = _paper_resid_margin(paper_resid)
    if (prev_box is None or lost >= PAPER_REACQ_LOST
            or margin >= PAPER_RESID_AMBIG):
        if boxes:
            return boxes[max(range(len(boxes)), key=lambda i: paper_resid[i])]
        return None
    return _pick_paper_outlier_box(boxes, prev_box, pred, gate, paper_resid)


def _pick_paper_free_box(boxes: list[tuple], prev_box: tuple | None,
                         paper_resid: list[float]) -> tuple | None:
    """Paper residual pick without pred gate/proximity; re-acquire on clear margin."""
    if not boxes:
        return None
    margin = _paper_resid_margin(paper_resid)
    reacquire = prev_box is None or margin >= PAPER_RESID_AMBIG
    if not reacquire and prev_box is not None:
        if max(_iou(b, prev_box) for b in boxes) < IOU_MATCH_MIN:
            reacquire = True
        else:
            prev_i = max(range(len(boxes)), key=lambda i: _iou(boxes[i], prev_box))
            if paper_resid[prev_i] < max(paper_resid) - margin:
                reacquire = margin >= PAPER_RESID_AMBIG * 0.5
    if reacquire:
        return boxes[max(range(len(boxes)), key=lambda i: paper_resid[i])]

    best_box: tuple | None = None
    best_score = -1e9
    for i, box in enumerate(boxes):
        score = paper_resid[i] + PAPER_FREE_IOU_W * _iou(box, prev_box)
        if score > best_score:
            best_score, best_box = score, box
    return best_box


def _pick_hybrid_unified(boxes: list[tuple], prev_box: tuple | None,
                         pred: np.ndarray, gate: float,
                         paper_resid: list[float],
                         rigid_resid: list[float] | None) -> tuple | None:
    """Fused paper + IoU chain score every frame."""
    if not boxes:
        return None
    if prev_box is None:
        return boxes[max(range(len(boxes)), key=lambda i: paper_resid[i])]
    best_box, best_score = None, -1e9
    for i, box in enumerate(boxes):
        iou = _iou(box, prev_box)
        cx, cy = _centroid(box)
        dist = float(np.hypot(cx - pred[0], cy - pred[1]))
        if dist > gate and iou < IOU_MATCH_MIN:
            continue
        prox = 1.0 - min(dist / gate, 1.0)
        chain = iou * 4.0 + 0.4 * prox
        rigid = rigid_resid[i] if rigid_resid and i < len(rigid_resid) else 0.0
        score = paper_resid[i] + 0.45 * iou + 0.3 * prox + 0.3 * chain + 0.1 * rigid
        if score > best_score:
            best_score, best_box = score, box
    if best_box is not None:
        return best_box
    return max(boxes, key=lambda b: _iou(b, prev_box))


def _pick_hybrid_box(boxes: list[tuple], prev_box: tuple | None,
                     pred: np.ndarray, gate: float,
                     paper_resid: list[float],
                     rigid_resid: list[float] | None) -> tuple | None:
    """Paper outlier + IoU chain: chain wins on disagreement when continuity is strong."""
    if not boxes:
        return None
    paper = _pick_paper_outlier_box(boxes, prev_box, pred, gate, paper_resid)
    if prev_box is None:
        return paper
    chain = _chain_locked_box(boxes, prev_box, pred, gate, rigid_resid)
    if paper is None:
        return chain
    if chain is None:
        return paper
    if _iou(paper, chain) >= IOU_MATCH_MIN:
        return paper
    ranked = sorted(paper_resid, reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) >= 2 else ranked[0]
    if _iou(chain, prev_box) >= HYBRID_CHAIN_IOU_MIN and margin < PAPER_RESID_AMBIG:
        return chain
    return paper


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
    hypotheses: list[Hypothesis] = []
    graph_history: list[GraphFrame] = []
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
            if pick_mode in ("hypothesis",) and prev_locked_box is not None:
                hypotheses = [Hypothesis(prev_locked_box)]
            if pick_mode in ("trajectory", "trajectory_paper", "trajectory_reacq") and prev_locked_box is not None and p.boxes:
                n = len(p.boxes)
                resid_seed = [0.0] * n
                lock_i = max(range(n), key=lambda i: _iou(p.boxes[i], prev_locked_box))
                resid_seed[lock_i] = GRAPH_LOCK_BONUS
                graph_history = [GraphFrame(list(p.boxes), resid_seed)]
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
        rigid = _box_rigid_residuals(prev_boxes, p.boxes) if prev_boxes else None
        if pick_mode == "trajectory":
            graph_history.append(GraphFrame(list(p.boxes), list(resid_pick)))
            if len(graph_history) > GRAPH_WINDOW:
                graph_history.pop(0)
            lock_seed = lock.box if lock is not None and p.idx - lock_frame <= GRAPH_WINDOW else None
            chosen = _pick_trajectory_graph(graph_history, gate=gate, lock_box=lock_seed)
        elif pick_mode == "trajectory_paper":
            graph_history.append(GraphFrame(list(p.boxes), list(resid_pick)))
            if len(graph_history) > GRAPH_WINDOW:
                graph_history.pop(0)
            lock_seed = lock.box if lock is not None and p.idx - lock_frame <= GRAPH_WINDOW else None
            chosen = _pick_trajectory_paper_box(
                graph_history, p.boxes, chain_prev, pred, gate, resid_pick, lock_seed)
        elif pick_mode == "trajectory_reacq":
            graph_history.append(GraphFrame(list(p.boxes), list(resid_pick)))
            if len(graph_history) > GRAPH_WINDOW:
                graph_history.pop(0)
            lock_seed = lock.box if lock is not None and p.idx - lock_frame <= GRAPH_WINDOW else None
            chosen = _pick_trajectory_reacq_box(
                graph_history, p.boxes, chain_prev, pred, gate, resid_pick, lock_seed, lost=lost)
        elif pick_mode == "hypothesis":
            seed = lock.box if lock is not None and p.idx - lock_frame <= 2 else None
            chosen, hypotheses = _update_hypotheses(
                hypotheses, p.boxes, resid_pick, seed)
        elif pick_mode == "hypothesis_paper":
            seed = lock.box if lock is not None and p.idx - lock_frame <= 2 else None
            chosen, hypotheses = _pick_hypothesis_paper_box(
                hypotheses, p.boxes, chain_prev, pred, gate, resid_pick, seed)
        elif pick_mode == "hybrid_unified":
            chosen = _pick_hybrid_unified(p.boxes, chain_prev, pred, gate, resid_pick, rigid)
        elif pick_mode == "hybrid":
            chosen = _pick_hybrid_box(p.boxes, chain_prev, pred, gate, resid_pick, rigid)
        elif pick_mode == "paper_free":
            chosen = _pick_paper_free_box(p.boxes, chain_prev, resid_pick)
        elif pick_mode == "paper_reacq":
            chosen = _pick_paper_reacq_box(
                p.boxes, chain_prev, pred, gate, resid_pick, lost=lost)
        elif pick_mode == "paper_outlier":
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
    "paper", "hypothesis", "hypothesis_paper", "hybrid", "hybrid_unified",
    "paper_free", "paper_reacq", "trajectory", "trajectory_paper",
    "trajectory_reacq", "paper_outlier", "paper_outlier_rank",
)

# All selectable modes, in leaderboard order. Single source of truth shared by
# run_clip's argparse and the eval_modes harness so the two never drift.
ALL_MODES = ("chain", *_PAPER_PICK_MODES, "outlier")


def _dispatch_mode(clip: Path, packs: list[FusionPack], lock: LockInfo | None,
                   frame_wh: tuple[int, int], mode: str
                   ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """Run one identity mode -> (track, locked_hist, start, radius).

    Single dispatch point for every pick mode, reused by both run_clip and the
    eval_modes leaderboard so a mode behaves identically in both.
    """
    if mode in _PAPER_PICK_MODES:
        return track_paper_identity(clip, packs, lock, frame_wh=frame_wh, pick_mode=mode)
    if mode == "outlier":
        from ld.detect.outlier_track import track_outlier_identity
        return track_outlier_identity(clip, packs, lock, frame_wh=frame_wh)
    return track_identity(packs, lock, frame_wh=frame_wh)


def run_clip(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
             imgsz: int = 768, use_cache: bool = True, evidence: bool = False,
             mode: str = "paper") -> IdentityReport:
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
                    default="paper",
                    help="paper=default; paper_outlier_rank=switch on low residual rank; "
                         "paper_outlier=paper + coast outlier switch; "
                         "outlier=persistent outlier tracks; "
                         "trajectory=Viterbi graph; trajectory_paper=gated blend; "
                         "hypothesis_paper=multi-hyp + gated paper; "
                         "hypothesis=multi-hyp only; hybrid=paper+chain; chain=IoU")
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
