"""EXP-S1 GATE (read-only): does the per-box AABB SIZE-PULSE rank the real-shape box
#1 on the laggard (t8/t5/t1) frames the trellis currently MISSES -- more often than
saliency mass AND more often than the already-shipped `curl` channel? And does it
DOMINATE on t7, where the spin is visually obvious (the measurement sanity)?

HYPOTHESIS (plan.md "BOX-DIMENSION ROTATION PULSE"). The real tile slowly rotates
independently of the rigid sheet; a rotating square's tight axis-aligned bounding box
PULSES in size -- side `= s*(|cos th| + |sin th|)`, `s` at 0deg, `s*sqrt2` at 45deg, area
up to 2x. A fake (sheet rotates ~0.006 deg/frame) has a dead-flat box. So the per-box
AABB size-oscillation, read STRAIGHT OFF THE DETECTOR over a long window, is a rotation
readout that lives ABOVE the pixel noise floor that killed every prior rotation attempt
(EXP-1 interior NCC, EXP-R1 outline rotation): the box is a detector aggregate of
thousands of pixels, so its geometry reads the ACCUMULATED rotation state far below the
per-pixel sub-pixel floor. Crucially this taps a DIFFERENT observable (rotation) than the
saturated translation channels (mass/coherent-mass), and rotation does NOT stop when
translation stalls -- the structural reason it might survive exactly on the drift-locks.

This probe reads ONLY `packs[idx].boxes` dims -- NO pixel pass. For each YOLO box, each
frame, walk it backward by nearest-prev-centroid over a causal window `W` and collect the
matched box's side `s = sqrt(w*h)` and area `a = w*h`. Then compute several pulse stats so
the gate picks the cleanest (per the hedge lesson, STRUCTURE beats MAGNITUDE):

  area_cv    : std(a)/mean(a) -- pulse MAGNITUDE (baseline; detector jitter pollutes this).
  side_ac1   : lag-1 autocorrelation of `s_t` -- a smooth slow oscillation -> ac1~1, white
               detector jitter -> ac1~0.  **Primary** (mirrors the hedge's coherence win).
  side_smooth: 1 - var(diff2 s)/var(diff s) -- smoothness of the pulse.
  side_trend : |slope|*R^2 of a linear fit `s_t` vs t -- a rotating shape has a locally
               monotonic side trend over the arc; jitter does not.

Motion baselines mass / coherent-mass / curl are computed on the SAME frames (reusing the
cached outlier vectors) for an apples-to-apples comparison, identical to rot_probe.

GATE (plan.md EXP-S1): on the t8/t5/t1 MISS frames a pulse stat's top-1 must beat BOTH
(a) instantaneous mass (t8 ~0.32) AND (b) the existing curl, AND DOMINATE on t7. If it
only matches curl -> curl rediscovered, stop. If it beats both on the laggard miss frames
AND passes t7 -> a new orthogonal channel -> proceed to EXP-S2 (additive emission).

    python -m ld.detect.box_pulse_probe --weights data/detect/runs/.../best.pt --clips t8 t5 t1 t7
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ld.detect.coh_gate import _compute_ov, _box_coherent_mass
from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.fuse_probe import (
    _box_curl, _windowed, _gt_box_idx, _rank_of, _load_miss_frames,
)
from ld.detect.identity import _centroid, compute_countdown_lock, _seed


# ---- box-dimension pulse statistics -----------------------------------------

def _nearest_prev_box_full(prev_boxes, c):
    """The prev-frame BOX nearest to point c (the box's own prior location; identical
    tiles + small frame-to-frame motion make this reliable -- same association EXP-1's
    `_nearest_prev_box` uses, but returns the full box so we can read its dims)."""
    if not prev_boxes:
        return None
    best, bd = None, 1e18
    for b in prev_boxes:
        pc = _centroid(b)
        d = (pc[0] - c[0]) ** 2 + (pc[1] - c[1]) ** 2
        if d < bd:
            bd, best = d, b
    return best


def _box_dims_window(packs, idxs, pos, box, win):
    """Side series `s_t = sqrt(w*h)` over a causal window, the box tracked backward by
    nearest-prev-centroid. Returns the series in CHRONOLOGICAL order (oldest..current);
    each element is sqrt(w*h) of the matched box. Length <= win+1; >=1 always."""
    sides = []
    c = _centroid(box)
    b = box
    p = pos
    for _ in range(win + 1):
        w = float(b[2] - b[0])
        h = float(b[3] - b[1])
        sides.append(math.sqrt(max(w * h, 0.0)))
        if p <= 0:
            break
        prev_idx = idxs[p - 1]
        pb = _nearest_prev_box_full(packs[prev_idx].boxes, c)
        if pb is None:
            break
        b = pb
        c = _centroid(pb)
        p -= 1
    sides.reverse()
    return np.asarray(sides, dtype=np.float64)


def _pulse_stats(sides):
    """The four pulse statistics from a side series (chronological). Higher => more
    likely the independently-rotating real shape. Returns dict of floats (0.0 on a series
    too short / degenerate to support the statistic -- a flat fake reads ~0, as desired)."""
    n = len(sides)
    area = sides ** 2
    # area_cv : pulse MAGNITUDE (the obvious-but-jitter-polluted baseline)
    am = float(area.mean())
    area_cv = float(area.std() / am) if am > 1e-6 else 0.0
    # side_ac1 : lag-1 autocorrelation -- STRUCTURE (smooth oscillation ~1, jitter ~0)
    side_ac1 = 0.0
    if n >= 3:
        sm = sides - sides.mean()
        den = float((sm * sm).sum())
        if den > 1e-9:
            side_ac1 = float((sm[1:] * sm[:-1]).sum() / den)
    # side_smooth : 1 - var(diff2)/var(diff) -- smoothness of the pulse
    side_smooth = 0.0
    if n >= 4:
        d1 = np.diff(sides)
        d2 = np.diff(sides, 2)
        v1 = float(d1.var())
        if v1 > 1e-9:
            side_smooth = max(0.0, 1.0 - float(d2.var()) / v1)
    # side_trend : |slope|*R^2 of a linear fit s_t vs t -- locally monotonic arc trend
    side_trend = 0.0
    if n >= 3:
        t = np.arange(n, dtype=np.float64)
        tm = t - t.mean()
        sm = sides - sides.mean()
        stt = float((tm * tm).sum())
        sss = float((sm * sm).sum())
        if stt > 1e-9 and sss > 1e-9:
            sxy = float((tm * sm).sum())
            slope = sxy / stt
            r2 = (sxy * sxy) / (stt * sss)
            side_trend = abs(slope) * r2
    return {"area_cv": area_cv, "side_ac1": side_ac1,
            "side_smooth": side_smooth, "side_trend": side_trend}


# ---- driver -----------------------------------------------------------------

PULSE_CHANS = ["area_cv", "side_ac1", "side_smooth", "side_trend"]
MOT_CHANS = ["mass", "coh", "curl"]


def run(weights, clips, win, miss_mode):
    chans = MOT_CHANS + PULSE_CHANS
    agg = {c: [0, 0, 0, 0, 0, 0] for c in chans}  # t1_all,t3_all,n_all,t1_miss,t3_miss,n_miss
    print(f"window={win}  miss frames from `{miss_mode}`  (box-dims only, no pixel pass)\n")

    for clip in clips:
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        compute_countdown_lock(packs, clip)
        _, _, radius, start = _seed(packs)
        idxs, ov = _compute_ov(clip, packs)
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
            # motion baselines (same as fuse_probe / rot_probe)
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
            # pulse channels (box-dims only, windowed)
            pulse = {c: [] for c in PULSE_CHANS}
            for b in p.boxes:
                sides = _box_dims_window(packs, idxs, pos, b, win)
                st = _pulse_stats(sides)
                for c in PULSE_CHANS:
                    pulse[c].append(st[c])
            scores = {"mass": mass, "coh": coh, "curl": curl, **pulse}
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
        print(f"[{name}]  radius={radius:.0f}  miss_frames={len(miss)}")
        for c in chans:
            t1, t3, n, mt1, mt3, mn_ = loc[c]
            for k in range(6):
                agg[c][k] += loc[c][k]
            f1 = t1 / n if n else 0
            f3 = t3 / n if n else 0
            mf1 = mt1 / mn_ if mn_ else 0
            mf3 = mt3 / mn_ if mn_ else 0
            tag = " <-pulse" if c in PULSE_CHANS else ""
            print(f"   {c:11} all: top1={f1:.3f} top3={f3:.3f} (n={n})   "
                  f"MISS: top1={mf1:.3f} top3={mf3:.3f} (n={mn_}){tag}")
        print()
    print("=" * 78)
    print("OVERALL (all probed clips)")
    for c in chans:
        t1, t3, n, mt1, mt3, mn_ = agg[c]
        tag = " <-pulse" if c in PULSE_CHANS else ""
        print(f"   {c:11} all: top1={t1/n if n else 0:.3f} top3={t3/n if n else 0:.3f}   "
              f"MISS: top1={mt1/mn_ if mn_ else 0:.3f} top3={mt3/mn_ if mn_ else 0:.3f} "
              f"(miss_n={mn_}){tag}")
    print("\nGATE: a pulse stat whose MISS-top1 beats BOTH mass AND curl on the laggard")
    print("clips => a genuinely new orthogonal emission term; build EXP-S2 (additive pulse).")
    print("If pulse ~= curl, it's curl rediscovered. If pulse << mass, the box doesn't")
    print("carry the spin (t8/t5 crumpling) -> joins appearance/sub-box as signal-limited.")
    print("(t7 sanity: a pulse stat should DOMINATE there; if not, detector boxes don't")
    print("track the pulse and Signal 1 is dead regardless of t8 -- same verdict as EXP-R1.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=["t8", "t5", "t1", "t7"])
    ap.add_argument("--win", type=int, default=15)
    ap.add_argument("--miss-mode", default="fpath_hyst")
    args = ap.parse_args()
    clips = [c for c in _default_clips()
             if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.win, args.miss_mode)


if __name__ == "__main__":
    main()
