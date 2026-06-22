"""Identity tracker: countdown lock -> causal Viterbi trellis over YOLO boxes.

YOLO localizes shape candidates per frame; this module decides *which* box is the
real shape over time. The shipped method is the ``fpath`` lineage (see ALL_MODES):

  1. Countdown lock -- stable white centroid (teleport guard for START false
     positive) -> YOLO box via mask overlap or anchor-nearest fallback.
  2. ``fpath`` -- a causal forward trellis over the boxes: emission = a weighted
     sum of motion-evidence channels (saliency mass, windowed coherent-mass,
     rotational curl), transition penalty = (jump_in_radii)^2; argmax cumulative
     score -> chosen-box centroid feeds a CV predictor.
  3. ``fpath_human`` (leader) -- ``fpath`` + hysteresis + decode-layer churn-hedge
     + residual-gated freeze + a human-cursor 1-Euro/deadband output filter. Box
     decisions are identical to ``fpath_freeze``; only the emitted point's motion
     differs. See CLAUDE.md for the full lineage and the dead-ends behind it.

Usage:
    python -m ld.detect.identity --weights data/detect/runs/yolov8n_single_combined/weights/best.pt
    python -m ld.detect.identity --weights .../best.pt --inputs data/t1_cropped_trimmed.mp4 --mode fpath_human

For overlay evidence videos use ``python -m ld.detect.render_evidence`` instead.
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
from ld.track.humanize import HumanCursor
from ld.vision.cursor import strip_pointer

__all__ = ["LockInfo", "IdentityReport", "compute_countdown_lock",
           "score_identity", "run_clip", "ALL_MODES", "BOARD_MODES"]


@dataclass
class LockInfo:
    """Box locked on the last white-countdown frame."""

    box: tuple[float, float, float, float, float]
    cx: float
    cy: float
    frame: int

# Tracker tunables.
LOCK_MASK_MIN = 0.01     # min mask overlap before anchor fallback
WHITE_TELEPORT_PX = 80.0 # START-text false positive jumps farther than this


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


# ---------------------------------------------------------------------------
# fpath trellis tunables (emission channels, transition prior, hysteresis,
# decode-layer hedge + residual-freeze). All are LOO-tuned; see CLAUDE.md and
# the surviving probes (fuse_sweep / exp3_sweep / hedge_probe / resid_freeze_probe).
# ---------------------------------------------------------------------------
FPATH_TRANS_W = 1.0       # transition penalty weight (jump in radii, squared)
FPATH_PROX_W = 0.0        # weight on CV-proximity emission (0 = pure mass; prox HURTS, see CLAUDE.md)
FPATH_PROX_SIGMA = 0.6    # proximity gaussian width in radii
FPATH_VEL_DAMP = 0.7      # CV prediction velocity damping (matches tracker VEL_DAMP)
FPATH_MASS_EMA = 0.98     # running-scale EMA for absolute mass normalization
FPATH_REACQUIRE_PATIENCE = 8    # frames off-track before teleport to global mass peak
FPATH_REACQUIRE_MASS_FRAC = 0.3 # min normalised mass for reacquire target (do-no-harm gate)
FPATH_TRANS_CAP = 8.0     # cap on transition penalty in radiiÂ² â€” limits lock-in depth
FPATH_COH_W = 0.0         # coherence emission weight; 0 = pure mass (current default)
FPATH_COH_W_DEFAULT = 1.8 # used by fpath_coh mode; swept on all 10 clips, peak at 1.8
# Additive multi-channel emission (mode "fpath_fuse"). fuse_probe (2026-06-19) showed
# t4/t5 are emission-channel failures the mass-only emission can't rank #1: the WINDOWED
# coherent-mass ranks the real box #1 on 60% of t4 misses, the rotational curl on 50% of
# t5 misses (vs mass ~43%), and coherence top-3 recall on miss frames is 0.842 (mass 0.803).
# Max-fusion is a dead end (probe top-1 0.22 -- whichever channel spikes on a fake wins), so
# the channels are combined ADDITIVELY: emission = norm(mass) + cmass_w*norm(cohmass) +
# curl_w*norm(curl). Each channel is normalized cross-frame by its own running peak EMA (NOT
# per-frame max) so a genuinely weak frame still yields low emission everywhere and the
# transition prior carries the path (preserves fpath's coast behavior, the load-bearing trick).
FPATH_CMASS_W = 0.0       # additive windowed coherent-mass weight (0 = off)
FPATH_CURL_W = 0.0        # additive windowed rotational-curl weight (0 = off)
FPATH_FUSE_WIN = 12       # causal window (frames) coherent-mass / curl accumulate over
# Tuned defaults for the fpath_fuse mode. LOO sweep (fuse_sweep.py, 2026-06-19) over
# cmass_w x curl_w: this is the no-per-clip-regression corner the LOO selector chose for
# 7/10 folds. In-sample mean 0.876 (pure-mass fpath 0.797; fpath_coh 0.825), worst-clip
# delta +0.004 vs pure mass. cmass is the big lever (t4 +0.11, t5 +0.14); curl a small
# complement that removes the t2 regression cmass-alone would cause.
FPATH_FUSE_CMASS_W = 1.5
FPATH_FUSE_CURL_W = 0.5

# EMA-coherent-mass hysteresis override (mode "fpath_hyst"). EXP-3 (exp3_switch_probe /
# exp3_sweep, 2026-06-19): a leaky EMA of per-box instantaneous coherent-mass, carried to
# each box's CURRENT location by nearest-centroid association, sits on GT 0.51 of t1's miss
# frames -- the only laggard clearing 0.5 (fixed windows stay 0.3-0.4; the carry is what
# tracks moving boxes). A DISTANCE-AGNOSTIC switch that fires when a challenger leads the
# path box's EMA by >margin for >=consec frames catches t1's ADJACENT creep that the dead
# far-jump reacquire structurally cannot, while the margin+consec gate keeps it silent on
# the strong lock-in clips. LOO sweep: +0.0022 mean, worst-clip +0.000 -- the first lock-in
# escape with NO strong-clip regression; all 10 folds select this same conservative corner.
FPATH_HYST_ALPHA = 0.92    # leaky-EMA retention of past coherent-mass (0 = override off)
FPATH_HYST_MARGIN = 0.30   # challenger must lead the path box's EMA by this fraction
FPATH_HYST_CONSEC = 10     # ...for this many consecutive frames before the switch fires

# Churn-gated freeze hedge (mode "fpath_hedge"). hedge_probe (2026-06-19): the catastrophic
# misses are SWEPT-LOCKS -- the path hops among adjacent fakes and rides the rigid sheet
# 200-400px from the (near-stationary) real shape. A pure freeze recovers 100% of those miss
# frames, but the MAGNITUDE of independent motion can't trigger it -- |d_indep| is HIGHER on
# laggard hops (9.8-10.3 px/fr) than on the correctly-tracked slow real shape (strong clips
# 3.3-7.4). The separator is DIRECTIONAL COHERENCE: churn = mean|d_indep| * (1 - R), with
# R = |sum d_indep| / sum|d_indep| in [0,1]. A coherent burst (real shape, one direction, same
# box) -> R~1 -> churn~0 -> commit; an incoherent box-hop (directions cancel) -> R~0 -> churn
# high -> freeze. DECODE-LAYER ONLY: output = w*chosen_centroid + (1-w)*prev_output, with
# w = clamp(1 - churn/churn_hi, 0, 1); the trellis/identity state is byte-identical to
# fpath_hyst (the hedge never enters the emission -- dodges the dead proximity-prior). Gate:
# every clip flat-or-up, mean within_r +0.021 (0.878->~0.899), worst-clip +0.000.
FPATH_HEDGE_CHURN_HI = 8.0   # churn (px/frame) at which trust w hits 0 (full freeze)
FPATH_HEDGE_WIN = 8          # causal window the churn coherence/magnitude accumulate over


# Residual-gated freeze (mode "fpath_freeze"). resid_freeze_probe (2026-06-20): the churn hedge
# only catches SWEPT locks (incoherent box-hops); it cannot fix the COHERENT identity-locks where
# the path creeps onto an adjacent fake and rides it smoothly -- there the hedge freezes at the
# wrong (fake) location, preserving the error. EXP-Q1's cumulative sheet-frame residual answers a
# 1-box BINARY the override (EXP-Q2b, dead) could not answer as a 15-box ranking: "is the box I am
# HOLDING a rigid fake?". A fake's N=30 residual is just detector jitter (~9-15px, near a clip- and
# radius-independent floor); the real shape's is much higher (~45-91px, its accumulated independent
# drift). When the chosen box's residual falls below the floor (tau) for `consec` frames we have
# locked onto a fake -> FREEZE the output toward a LAGGED pre-creep anchor (the output `lag` frames
# ago, before the creep started) and hold until the residual recovers (the trellis re-acquired the
# real shape). Works because the real shape barely moves (median 1.3 px/fr), so an onset-anchored
# freeze stays within radius for 20+ frame runs. DECODE-LAYER ONLY (identity state untouched), and
# runs BEFORE the churn hedge (which then passes the frozen position through: a held position reads
# as coherent sheet motion -> churn~0 -> w~1 -> commit). Gate: LOO 0.899->0.936, worst-clip +0.000,
# all 10 folds independently select tau=15/lag=6/consec=1 (t8 +0.133, t1 +0.074, t5 +0.049).
FPATH_FREEZE_TAU = 15.0      # residual (px) below which the chosen box reads as a rigid fake
FPATH_FREEZE_LAG = 6         # frames to rewind the freeze anchor toward the lock onset
FPATH_FREEZE_CONSEC = 1      # consecutive below-floor frames required before freezing
FPATH_FREEZE_N = 30          # horizon (frames) the cumulative sheet-frame residual integrates over


def _cumulative_residual(rhist, chosen_cent, n, radius):
    """Cumulative sheet-frame residual magnitude of the chosen box over horizon `n`, via a causal
    affine-prediction back-walk (mirrors resid_override_probe._residual_mag, validated by the gate).

    `rhist` is a deque of (cents, inv_affine) per recent frame, rhist[-1] = current frame; `cents`
    is the list of box centroids that frame (or None on a coast/no-box frame), `inv_affine` inverts
    that frame's prev->cur sheet affine (or None). Walk back n frames: at each step map the running
    points by the inverse affine (rigid back-transport) and snap the chain to the nearest actual
    centroid in the previous frame; the residual is how far the rigid back-transport of the frame-t
    centroid ends up from its actual n-frames-ago ancestor. Returns None if the chain breaks -- a
    gap/coast in the window, a missing affine, or a snap beyond the radius -- so the freeze then
    conservatively does not fire (a fake whose chain breaks is simply not frozen that frame)."""
    if len(rhist) < n + 1:
        return None
    p_ref = np.asarray(chosen_cent, np.float64)
    chain = np.asarray(chosen_cent, np.float64)
    r2 = radius * radius
    for s in range(1, n + 1):
        _cents_cur, invaff = rhist[-s]            # frame t-s+1: its affine maps (t-s)->(t-s+1)
        cents_prev, _inv_prev = rhist[-s - 1]     # frame t-s: the actual ancestor centroids
        if invaff is None or not cents_prev:
            return None
        p_ref = invaff[:, :2] @ p_ref + invaff[:, 2]
        pred = invaff[:, :2] @ chain + invaff[:, 2]
        bj, bd = -1, 1e18
        for j, (cx, cy) in enumerate(cents_prev):
            d = (cx - pred[0]) ** 2 + (cy - pred[1]) ** 2
            if d < bd:
                bd, bj = d, j
        if bd >= r2:                              # nearest ancestor beyond radius -> chain broke
            return None
        chain = np.asarray(cents_prev[bj], np.float64)
    return float(math.hypot(p_ref[0] - chain[0], p_ref[1] - chain[1]))


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
                              cmass_w: float = FPATH_CMASS_W,
                              curl_w: float = FPATH_CURL_W,
                              fuse_win: int = FPATH_FUSE_WIN,
                              hyst_alpha: float = 0.0,
                              hyst_margin: float = 0.0,
                              hyst_consec: int = 0,
                              hedge: bool = False,
                              hedge_churn_hi: float = FPATH_HEDGE_CHURN_HI,
                              hedge_win: int = FPATH_HEDGE_WIN,
                              freeze: bool = False,
                              freeze_tau: float = FPATH_FREEZE_TAU,
                              freeze_lag: int = FPATH_FREEZE_LAG,
                              freeze_consec: int = FPATH_FREEZE_CONSEC,
                              freeze_n: int = FPATH_FREEZE_N,
                              humanize: bool = False,
                              choice_sink: dict | None = None,
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
    cmass_scale = 0.0  # same, for the additive windowed coherent-mass channel
    curl_scale = 0.0   # same, for the additive windowed rotational-curl channel
    from collections import deque
    use_fuse = cmass_w > 0 or curl_w > 0
    ovecs_hist: deque = deque(maxlen=fuse_win) if use_fuse else None
    use_hyst = hyst_alpha > 0 and hyst_consec > 0
    ema_cm: np.ndarray | None = None      # leaky EMA of instantaneous coherent-mass
    ema_cents: list[tuple[float, float]] | None = None
    hyst_streak = 0
    # Churn-gated freeze hedge (mode fpath_hedge): a decode-layer output blend that holds
    # position when the pick is churning (incoherent box-hops); identity state untouched.
    hpos: np.ndarray | None = None        # hedged OUTPUT position (what gets emitted)
    hwin: deque = deque(maxlen=hedge_win)  # recent chosen-box independent-motion vectors
    hprev_cent: np.ndarray | None = None   # chosen-box centroid at the last track frame
    # Residual-gated freeze (mode fpath_freeze): runs BEFORE the hedge; transforms the committed
    # output to a held pre-onset anchor when the chosen box's N-residual collapses to the fake floor.
    use_freeze = freeze
    rhist: deque | None = deque(maxlen=freeze_n + 1) if use_freeze else None  # (cents, inv_aff)/frame
    freeze_buf: deque | None = deque(maxlen=max(freeze_lag, 1)) if use_freeze else None  # past outputs
    fpos: np.ndarray | None = None   # the held (frozen) output position
    frozen = False        # currently holding the anchor
    low_streak = 0        # consecutive below-floor (on-a-fake) frames
    # Human-cursor output dynamics (mode fpath_human): a strictly-causal 1-Euro + deadband
    # filter applied as the LAST transform on the emitted point (decode-layer only; identity
    # state untouched). Started fresh at the first scored frame (idx >= start) so it is
    # byte-faithful to the offline gate (cursor_physics_probe). Smooths the lock-wobble and
    # eases the freeze-snaps into reach-like glides; within_r holds flat-or-up on both boards.
    hcursor: HumanCursor | None = None

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
        cur_affine = None  # this frame's global sheet affine (set in the track branch)
        frame_cents = None  # this frame's box centroids (set in the track branch; None on coast)
        if prev_gray is not None and active and p.boxes:
            fld = estimate_motion(prev_gray, gray)
            cur_affine = fld.affine
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
            # Additive windowed coherent-mass + curl channels (fpath_fuse). Each is
            # normalized cross-frame by its own running peak EMA (same trick as mass) so
            # weak frames stay low-emission everywhere and the transition prior carries.
            cmass = np.zeros(len(p.boxes), np.float32)
            curl = np.zeros(len(p.boxes), np.float32)
            if ovecs_hist is not None:
                ovecs_hist.append((fld.outliers, fld.outlier_vectors))
                win = list(ovecs_hist)
                if cmass_w > 0:
                    cm = np.array([_box_coherent_mass(b, win) for b in p.boxes], np.float32)
                    cpk = float(cm.max())
                    cmass_scale = cpk if cmass_scale == 0.0 else max(cpk, mass_ema * cmass_scale + (1 - mass_ema) * cpk)
                    cmass = cm / cmass_scale if cmass_scale > 0 else cm
                if curl_w > 0:
                    cu = np.array([_box_rotational_curl(b, win) for b in p.boxes], np.float32)
                    upk = float(cu.max())
                    curl_scale = upk if curl_scale == 0.0 else max(upk, mass_ema * curl_scale + (1 - mass_ema) * upk)
                    curl = cu / curl_scale if curl_scale > 0 else cu
            # Leaky EMA of instantaneous (single-frame) coherent-mass, carried to each
            # current box by nearest-centroid association (mode fpath_hyst, EXP-3).
            if use_hyst:
                inst_cm = np.array(
                    [_box_coherent_mass(b, [(fld.outliers, fld.outlier_vectors)])
                     for b in p.boxes], np.float32)
                if ema_cm is not None and ema_cents is not None:
                    new_ema = inst_cm.copy()
                    for i, (cx, cy) in enumerate(cents):
                        bd, bj = 1e18, -1
                        for pj, (px, py) in enumerate(ema_cents):
                            dd = (px - cx) ** 2 + (py - cy) ** 2
                            if dd < bd:
                                bd, bj = dd, pj
                        if bj >= 0:
                            new_ema[i] = hyst_alpha * ema_cm[bj] + (1 - hyst_alpha) * inst_cm[i]
                    ema_cm = new_ema
                else:
                    ema_cm = inst_cm
                ema_cents = cents
            pred = pos + vel
            emis = np.empty(len(cents), np.float32)
            for i, (cx, cy) in enumerate(cents):
                d = math.hypot(cx - pred[0], cy - pred[1])
                prox = math.exp(-(d / (prox_sigma * radius)) ** 2)
                emis[i] = (mass[i] * (1.0 + coh_w * coh[i])
                           + cmass_w * cmass[i] + curl_w * curl[i] + prox_w * prox)
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
            # EMA-coherent-mass hysteresis override (fpath_hyst, EXP-3): switch the path
            # to a box that PERSISTENTLY dominates the leaky-integrated coherent-mass,
            # regardless of distance. Resets path memory so the transition prior doesn't
            # immediately drag the path back onto the box it had locked.
            hyst_fired = False
            if use_hyst and ema_cm is not None:
                chal = int(np.argmax(ema_cm))
                if chal != choice and ema_cm[chal] > (1.0 + hyst_margin) * max(float(ema_cm[choice]), 1e-9):
                    hyst_streak += 1
                else:
                    hyst_streak = 0
                if hyst_streak >= hyst_consec:
                    choice = chal
                    hyst_fired = True
                    hyst_streak = 0
                    vel[:] = 0.0
            nx, ny = cents[choice]
            vel = vel_damp * vel + (1.0 - vel_damp) * (np.array([nx, ny], np.float32) - pos)
            x, y, state = nx, ny, "track"
            frame_cents = cents  # for the residual back-walk (mode fpath_freeze)
            if choice_sink is not None:
                # Read-only instrumentation (Phase-1 localization gate): record the
                # CHOSEN box + two in-box localizer candidates from the same motion
                # field the trellis used. Active only when a sink is passed; the
                # shipped emission path (nx,ny == centroid) is untouched.
                cb = p.boxes[choice]
                hh, ww = sal.shape
                xa, ya = max(0, int(cb[0])), max(0, int(cb[1]))
                xb, yb = min(ww, int(cb[2])), min(hh, int(cb[3]))
                sal_pt = None
                if xb > xa and yb > ya:
                    sub = sal[ya:yb, xa:xb]
                    if float(sub.max()) > 0.0:
                        iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
                        sal_pt = (float(xa + ix), float(ya + iy))
                owc_pt = None
                ox = fld.outliers
                if len(ox):
                    m = ((ox[:, 0] >= cb[0]) & (ox[:, 0] < cb[2]) &
                         (ox[:, 1] >= cb[1]) & (ox[:, 1] < cb[3]))
                    if bool(m.any()):
                        wts = fld.outlier_weights[m]
                        pts = ox[m]
                        wsum = float(wts.sum())
                        if wsum > 1e-9:
                            owc_pt = (float((pts[:, 0] * wts).sum() / wsum),
                                      float((pts[:, 1] * wts).sum() / wsum))
                choice_sink[p.idx] = dict(box=tuple(cb), cent=(float(nx), float(ny)),
                                          sal=sal_pt, owc=owc_pt)
            prev_alpha, prev_cents = (None, None) if hyst_fired else (alpha, cents)
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
        if use_freeze:
            # Residual-gated freeze (mode fpath_freeze), BEFORE the hedge. Push this frame's
            # (centroids, inverse sheet affine) for the back-walk; coast frames push None and
            # break the chain. `committed` == the chosen-box output (decode-layer only: `pos`
            # above keeps the committed value, so the trellis/CV state is untouched). When the
            # chosen box's cumulative residual collapses to the fake floor for `freeze_consec`
            # frames, hold the output at the lagged pre-creep anchor (freeze_buf[0]); release to
            # the live pick when it recovers. A broken chain (rv None) leaves low False -> commit.
            invaff = (cv2.invertAffineTransform(np.asarray(cur_affine, np.float64))
                      if cur_affine is not None else None)
            rhist.append((frame_cents, invaff))
            committed = np.array([x, y], np.float64)
            rv = (_cumulative_residual(rhist, committed, freeze_n, radius)
                  if state == "track" else None)
            low = rv is not None and rv < freeze_tau
            low_streak = low_streak + 1 if low else 0
            if low_streak >= freeze_consec:
                if not frozen:                       # onset: rewind to the pre-creep anchor
                    fpos = (np.asarray(freeze_buf[0], np.float64)
                            if (freeze_lag > 0 and len(freeze_buf) > 0) else committed.copy())
                    frozen = True
                # else: hold fpos
            else:
                frozen = False
                fpos = committed.copy()              # commit to the live pick
            freeze_buf.append(committed.copy())
            x, y = float(fpos[0]), float(fpos[1])
        out_x, out_y = x, y
        if hedge:
            # Decode-layer churn-gated freeze. c_t = the committed output (== chosen box
            # centroid on track frames). Accumulate the chosen box's per-frame independent
            # motion (sheet-removed) over a causal window; freeze the output toward its last
            # value when that motion is large AND directionally incoherent (a box-hop), keep
            # committing when it is coherent (a real-shape burst) or small (stable lock).
            c_t = np.array([x, y], np.float64)
            if state == "track" and cur_affine is not None and hprev_cent is not None:
                T = cur_affine
                hwin.append(c_t - (T[:, :2] @ hprev_cent + T[:, 2]))
            if state == "track":
                hprev_cent = c_t.copy()
            if hwin:
                mags = [float(math.hypot(v[0], v[1])) for v in hwin]
                s = np.sum(hwin, axis=0)
                tot_m = sum(mags)
                R = float(math.hypot(s[0], s[1]) / tot_m) if tot_m > 1e-9 else 0.0
                churn = (tot_m / len(mags)) * (1.0 - R)
                w = min(max(1.0 - churn / hedge_churn_hi, 0.0), 1.0)
            else:
                w = 1.0
            hpos = c_t.copy() if hpos is None else w * c_t + (1.0 - w) * hpos
            out_x, out_y = float(hpos[0]), float(hpos[1])
        if humanize and p.idx >= start:
            # LAST transform: glide the emitted point like a hand. Started at the first
            # scored frame so it matches the offline gate exactly (60 fps clips).
            if hcursor is None:
                hcursor = HumanCursor(fps=60.0)
            out_x, out_y = hcursor.update((out_x, out_y))
        track.append(TrackPoint(p.idx, out_x, out_y, state, len(p.boxes)))
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



# Full dispatch registry: every mode `_dispatch_mode` can run. The fpath lineage is the
# only surviving family -- pure-mass baseline -> coh -> fuse -> hyst -> hedge -> freeze ->
# human (the leader). The intermediate stages are kept runnable (`--modes <name>`) so the
# tuning probes can regenerate their per-mode eval CSVs (hedge/resid_freeze read fpath_hyst,
# cursor_physics reads fpath_freeze). The retired field/paper/accum/chain/reacq families were
# removed in the 2026-06-22 cleanup (all permanently dominated; see CLAUDE.md dead-ends).
ALL_MODES = ("fpath", "fpath_coh", "fpath_fuse", "fpath_hyst", "fpath_hedge", "fpath_freeze", "fpath_human")

# Default leaderboard set (eval_modes `--modes` default): the shipped leader plus the pure
# ablation base. eval_modes also reports the per-clip oracle ceiling alongside these.
BOARD_MODES = ("fpath", "fpath_human")


def _dispatch_mode(clip: Path, packs: list[FusionPack], lock: LockInfo | None,
                   frame_wh: tuple[int, int], mode: str
                   ) -> tuple[list[TrackPoint], list[int | None], int, float]:
    """Run one identity mode -> (track, locked_hist, start, radius).

    Single dispatch point for every pick mode, reused by both run_clip and the
    eval_modes leaderboard so a mode behaves identically in both.
    """
    if mode == "fpath":
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh)
    if mode == "fpath_coh":
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh,
                                         coh_w=FPATH_COH_W_DEFAULT)
    if mode == "fpath_fuse":
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh,
                                         cmass_w=FPATH_FUSE_CMASS_W,
                                         curl_w=FPATH_FUSE_CURL_W)
    if mode == "fpath_hyst":
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh,
                                         cmass_w=FPATH_FUSE_CMASS_W,
                                         curl_w=FPATH_FUSE_CURL_W,
                                         hyst_alpha=FPATH_HYST_ALPHA,
                                         hyst_margin=FPATH_HYST_MARGIN,
                                         hyst_consec=FPATH_HYST_CONSEC)
    if mode == "fpath_hedge":
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh,
                                         cmass_w=FPATH_FUSE_CMASS_W,
                                         curl_w=FPATH_FUSE_CURL_W,
                                         hyst_alpha=FPATH_HYST_ALPHA,
                                         hyst_margin=FPATH_HYST_MARGIN,
                                         hyst_consec=FPATH_HYST_CONSEC,
                                         hedge=True)
    if mode == "fpath_freeze":
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh,
                                         cmass_w=FPATH_FUSE_CMASS_W,
                                         curl_w=FPATH_FUSE_CURL_W,
                                         hyst_alpha=FPATH_HYST_ALPHA,
                                         hyst_margin=FPATH_HYST_MARGIN,
                                         hyst_consec=FPATH_HYST_CONSEC,
                                         hedge=True, freeze=True)
    if mode == "fpath_human":
        # fpath_freeze + human-cursor output dynamics (1-Euro + deadband). Identical box
        # decisions; only HOW the emitted point moves between them changes (decode-layer).
        return track_fused_path_identity(clip, packs, lock, frame_wh=frame_wh,
                                         cmass_w=FPATH_FUSE_CMASS_W,
                                         curl_w=FPATH_FUSE_CURL_W,
                                         hyst_alpha=FPATH_HYST_ALPHA,
                                         hyst_margin=FPATH_HYST_MARGIN,
                                         hyst_consec=FPATH_HYST_CONSEC,
                                         hedge=True, freeze=True, humanize=True)
    raise ValueError(f"unknown identity mode: {mode!r} (valid: {', '.join(ALL_MODES)})")


def run_clip(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
             imgsz: int = 768, use_cache: bool = True,
             mode: str = "fpath_human") -> IdentityReport:
    clip = Path(clip)
    name = clip.stem.replace("_cropped_trimmed", "")
    packs = detect_fusion_clip(weights, clip, conf=conf, imgsz=imgsz, use_cache=use_cache)
    lock = compute_countdown_lock(packs, clip)
    src = VideoSource(clip)
    wh = (src.meta.width, src.meta.height)
    src.release()
    track, locked_hist, start, radius = _dispatch_mode(clip, packs, lock, wh, mode)
    return score_identity(packs, track, start, radius, name)


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
    ap.add_argument("--mode",
                    choices=ALL_MODES,
                    default="fpath_human",
                    help="fpath_human=default leader (fpath + hysteresis + churn-hedge + "
                         "residual-freeze + human-cursor 1-Euro/deadband output); "
                         "fpath_freeze=same box decisions, raw centroid output; "
                         "fpath=pure-mass Viterbi ablation base; the intermediate "
                         "fpath_coh/fuse/hyst/hedge stages are runnable for probe retuning; "
                         "see ld/detect/LEADERBOARD.md for the ranking")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs] if args.inputs else _default_inputs()
    if not inputs:
        raise SystemExit("No input clips found.")

    reports: list[IdentityReport] = []
    for clip in inputs:
        print(f"[{clip.stem}] identity tracking ...")
        rep = run_clip(args.weights, clip, conf=args.conf, imgsz=args.imgsz,
                       use_cache=not args.no_cache, mode=args.mode)
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
