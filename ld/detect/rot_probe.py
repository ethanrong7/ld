"""EXP-R1 GATE (read-only): does DIRECT OUTLINE-ROTATION rate rank the real-shape box
#1 on the laggard (t8/t5/t1) frames the trellis currently MISSES -- more often than
saliency mass and more often than the already-shipped `curl` channel?

HYPOTHESIS (plan.md "OUTLINE ROTATION"). A human spots the real tile because its
BORDER slowly spins (~1-1.25 deg/frame on t7) while every fake stays axis-aligned with
the rigid sheet (which itself rotates only ~0.006 deg/frame). Orientation lives in the
BOUNDARY, not the interior fill -- the fill is white specular blobs that slosh with no
fixed orientation. Every prior rotation/appearance attempt measured the wrong thing:

  - EXP-1 de-rotated by the GLOBAL SHEET angle (a no-op) and compared INTERIOR-fill NCC.
  - NCC / log-polar trackers ran on the WHOLE patch (fill included) as trackers.
  - the live `curl` channel reads Sum(r x v) of sparse LK flow -- an INDIRECT, noisy
    proxy for a slow boundary spin.

This probe measures boundary orientation DIRECTLY and ONLY: build an edge map of each
box patch, MASK OUT the interior (keep a boundary annulus) so the sloshing fill is
excluded, then estimate the inter-frame rotation of that edge ring. Two estimators run
side by side:

  rot_bank : argmax over a +/-MAXDEG rotation bank of NCC(rotate(prev_edge, deg), cur_edge)
             -- bounded by construction, robust to sparse edges.            [primary]
  rot_lp   : log-polar + phase-correlation on the edge ring (rotation -> angular shift).

Both are accumulated as |rotation rate| over a causal window, the box tracked backward
by nearest-prev-centroid (mirrors EXP-1's association). Motion baselines mass / coherent
-mass / curl are computed on the SAME frames for an apples-to-apples comparison.

GATE (plan.md EXP-R1): on the t8/t5/t1 MISS frames, a rot_* channel's top-1 must beat
BOTH (a) instantaneous mass (t8 ~0.32) AND (b) the existing curl. If it only matches
curl, it's curl rediscovered -> stop. If it beats curl on the laggard miss frames it's a
new orthogonal channel -> proceed to EXP-R2 (additive emission). t7 is included as a
SANITY clip: if the estimator can't see t7's obvious spin, the estimator is broken, not
the hypothesis.

Read-only. Reuses coh_gate's cached outlier vectors for the motion baselines and EXP-1's
gray-loading + patch helpers.

    python -m ld.detect.rot_probe --weights data/detect/runs/.../best.pt --clips t8 t5 t1 t7
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from ld.detect.coh_gate import _compute_ov, _box_coherent_mass
from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.exp1_appearance_probe import _patch, _ncc, _nearest_prev_box, _load_grays
from ld.detect.fuse_probe import (
    _box_curl, _windowed, _gt_box_idx, _rank_of, _load_miss_frames,
)
from ld.detect.identity import _centroid, compute_countdown_lock, _seed


# ---- boundary-orientation estimators ----------------------------------------

RS = 64  # edge-map working size (patch resampled to RS x RS before edge/rotation)


def _edge(patch):
    """Sobel gradient-magnitude edge map of a patch, resampled to RS x RS. float32.
    The boundary band carries the orientation; the interior fill is masked out by
    `_annulus` before any rotation is estimated."""
    p = cv2.resize(patch, (RS, RS), interpolation=cv2.INTER_LINEAR)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _annulus(r_in, r_out):
    """Boolean->float32 ring mask on the RS x RS grid: keep radius in [r_in,r_out]*(RS/2).
    Kills the central specular fill (no orientation) and the box-corner clutter, leaving
    the rounded-frame border band. This masking IS the experiment -- a full-patch estimate
    repeats the dead log-polar/NCC mistake."""
    cy = cx = (RS - 1) / 2.0
    ys, xs = np.mgrid[0:RS, 0:RS]
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    rmax = RS / 2.0
    m = (r >= r_in * rmax) & (r <= r_out * rmax)
    return m.astype(np.float32)


def _est_bank(prev_patch, cur_patch, mask, max_deg, step):
    """Inter-frame |rotation| (deg) of the boundary: rotate the prev edge ring across a
    +/-max_deg bank and take the angle whose rotated ring best matches the cur edge ring
    (NCC). Bounded, so a noisy fake can't post a huge accumulated value. Returns
    (|best_deg|, best_ncc)."""
    ep = _edge(prev_patch) * mask
    ec = _edge(cur_patch) * mask
    center = (RS / 2.0, RS / 2.0)
    best_a, best_v = 0.0, -2.0
    degs = np.arange(-max_deg, max_deg + 1e-9, step)
    for deg in degs:
        M = cv2.getRotationMatrix2D(center, float(deg), 1.0)
        rot = cv2.warpAffine(ep, M, (RS, RS), flags=cv2.INTER_LINEAR)
        v = _ncc(rot, ec)
        if v > best_v:
            best_v, best_a = v, float(deg)
    return abs(best_a), best_v


def _est_logpolar(prev_patch, cur_patch, mask, max_deg):
    """Inter-frame |rotation| (deg) via log-polar + phase correlation on the edge ring.
    In log-polar coords a rotation becomes a shift along the angular (row) axis; the
    phase-correlation row shift -> degrees. Clamped to max_deg (sparse-edge phase-corr can
    spike). Returns (|deg|, response)."""
    ep = _edge(prev_patch) * mask
    ec = _edge(cur_patch) * mask
    center = (RS / 2.0, RS / 2.0)
    Mlp = RS / math.log(RS / 2.0)
    lp_p = cv2.logPolar(ep, center, Mlp, cv2.WARP_FILL_OUTLIERS).astype(np.float32)
    lp_c = cv2.logPolar(ec, center, Mlp, cv2.WARP_FILL_OUTLIERS).astype(np.float32)
    try:
        (_dx, dy), resp = cv2.phaseCorrelate(lp_p, lp_c)
    except cv2.error:
        return 0.0, 0.0
    angle = dy / RS * 360.0
    angle = (angle + 180.0) % 360.0 - 180.0      # wrap to [-180,180]
    return min(abs(angle), max_deg), float(resp)


def _box_rot_window(grays, packs, idxs, pos, box, half, mask, win, est):
    """Accumulated |rotation rate| over a causal window: walk the box backward by
    nearest-prev-centroid (identical tiles + small motion make this reliable -- same
    association EXP-1 uses), measuring the boundary rotation of each consecutive pair and
    summing. Returns the windowed sum (>=0; bigger => spinning more independently)."""
    total = 0.0
    c = _centroid(box)
    p = pos
    for _ in range(win):
        if p <= 0:
            break
        cur_idx, prev_idx = idxs[p], idxs[p - 1]
        if cur_idx not in grays or prev_idx not in grays:
            break
        pc = _nearest_prev_box(packs[prev_idx].boxes, c)
        if pc is None:
            break
        cur_patch = _patch(grays[cur_idx], c[0], c[1], half, 0.0)
        prev_patch = _patch(grays[prev_idx], pc[0], pc[1], half, 0.0)
        rate, _conf = est(prev_patch, cur_patch, mask)
        total += rate
        c = pc
        p -= 1
    return total


# ---- driver -----------------------------------------------------------------

ROT_CHANS = ["rot_bank", "rot_lp"]
MOT_CHANS = ["mass", "coh", "curl"]


def run(weights, clips, win, miss_mode, max_deg, step, r_in, r_out):
    chans = MOT_CHANS + ROT_CHANS
    agg = {c: [0, 0, 0, 0, 0, 0] for c in chans}  # t1_all,t3_all,n_all,t1_miss,t3_miss,n_miss
    mask = _annulus(r_in, r_out)
    print(f"window={win}  bank=+/-{max_deg}deg@{step}  annulus=[{r_in},{r_out}]*R"
          f"  miss frames from `{miss_mode}`\n")
    est_bank = lambda a, b, m: _est_bank(a, b, m, max_deg, step)
    est_lp = lambda a, b, m: _est_logpolar(a, b, m, max_deg)

    for clip in clips:
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        compute_countdown_lock(packs, clip)
        _, _, radius, start = _seed(packs)
        idxs, ov = _compute_ov(clip, packs)
        grays = _load_grays(clip, packs)
        half = int(round(radius))
        name = clip.stem.replace("_cropped_trimmed", "")
        miss = _load_miss_frames(clip.stem, miss_mode)
        pos_of = {ix: pos for pos, ix in enumerate(idxs)}
        loc = {c: [0, 0, 0, 0, 0, 0] for c in chans}
        for idx in idxs:
            if idx < start or idx >= len(packs):
                continue
            p = packs[idx]
            if p.gt is None or not p.boxes:
                continue
            gi = _gt_box_idx(p.boxes, p.gt, radius)
            if gi is None:
                continue  # oracle miss; identity can't win
            pos = pos_of[idx]
            if pos == 0:
                continue
            # motion baselines (same as fuse_probe)
            coh = _windowed(idxs, ov, p.boxes, idx, win, _box_coherent_mass)
            curl = _windowed(idxs, ov, p.boxes, idx, win, _box_curl)
            o0 = ov.get(idx)
            mass = []
            for b in p.boxes:
                if o0 is None:
                    mass.append(0.0)
                    continue
                pix, _r, magv = o0
                inb = ((pix[:, 0] >= b[0]) & (pix[:, 0] <= b[2])
                       & (pix[:, 1] >= b[1]) & (pix[:, 1] <= b[3]))
                mass.append(float(magv[inb].sum()))
            # rotation channels (boundary-only, windowed)
            rot_bank = [_box_rot_window(grays, packs, idxs, pos, b, half, mask, win, est_bank)
                        for b in p.boxes]
            rot_lp = [_box_rot_window(grays, packs, idxs, pos, b, half, mask, win, est_lp)
                      for b in p.boxes]
            scores = {"mass": mass, "coh": coh, "curl": curl,
                      "rot_bank": rot_bank, "rot_lp": rot_lp}
            in_miss = idx in miss
            for c in chans:
                rk = _rank_of(scores[c], gi)
                loc[c][2] += 1
                if rk == 1: loc[c][0] += 1
                if rk <= 3: loc[c][1] += 1
                if in_miss:
                    loc[c][5] += 1
                    if rk == 1: loc[c][3] += 1
                    if rk <= 3: loc[c][4] += 1
        print(f"[{name}]  radius={radius:.0f}  half={half}  miss_frames={len(miss)}")
        for c in chans:
            t1, t3, n, mt1, mt3, mn_ = loc[c]
            for k in range(6):
                agg[c][k] += loc[c][k]
            f1 = t1 / n if n else 0
            f3 = t3 / n if n else 0
            mf1 = mt1 / mn_ if mn_ else 0
            mf3 = mt3 / mn_ if mn_ else 0
            tag = " <-rot" if c in ROT_CHANS else ""
            print(f"   {c:9} all: top1={f1:.3f} top3={f3:.3f} (n={n})   "
                  f"MISS: top1={mf1:.3f} top3={mf3:.3f} (n={mn_}){tag}")
        print()
    print("=" * 74)
    print("OVERALL (all probed clips)")
    for c in chans:
        t1, t3, n, mt1, mt3, mn_ = agg[c]
        tag = " <-rot" if c in ROT_CHANS else ""
        print(f"   {c:9} all: top1={t1/n if n else 0:.3f} top3={t3/n if n else 0:.3f}   "
              f"MISS: top1={mt1/mn_ if mn_ else 0:.3f} top3={mt3/mn_ if mn_ else 0:.3f} "
              f"(miss_n={mn_}){tag}")
    print("\nGATE: a rot_* channel whose MISS-top1 beats BOTH mass AND curl on the laggard")
    print("clips => a genuinely new orthogonal emission term; build EXP-R2 (additive rot).")
    print("If rot ~= curl, it's curl rediscovered. If rot << mass, the outline is buried")
    print("by t8/t5 crumpling -> joins appearance/sub-box/detection as signal-limited.")
    print("(t7 sanity: rot_* should DOMINATE there; if not, the estimator is broken.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=["t8", "t5", "t1", "t7"])
    ap.add_argument("--win", type=int, default=10)
    ap.add_argument("--miss-mode", default="fpath_hyst")
    ap.add_argument("--max-deg", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--r-in", type=float, default=0.35)
    ap.add_argument("--r-out", type=float, default=1.0)
    args = ap.parse_args()
    clips = [c for c in _default_clips()
             if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.win, args.miss_mode, args.max_deg, args.step,
        args.r_in, args.r_out)


if __name__ == "__main__":
    main()
