"""STEP-1 GATE: does a fused mass+coherence+curl channel rank the real-shape box
highly on the frames the current leader (`fpath_coh`) gets WRONG?

The fpath Viterbi integrator emits the box maximizing saliency MASS (fpath_coh adds
a weak coherence bump). The diagnosed wall: on laggards (t1/t4/t5/t8) the real box is
in YOLO's output (oracle ~0.95) but mass ranks a FAKE #1, so the path locks onto it.
Probes claimed coherence/curl TOP-3-localize the real shape but argmax+margin can't
convert that to a top-1 pick (the "causal-key wall").

Viterbi path integration IS the tool to convert top-3 localization into a top-1
track. But that only pays off if a better per-frame EMISSION ranks the real box
higher than mass does. This probe measures exactly that, channel by channel:

  mass(box)  = saliency mass inside box (fpath's current emission)
  coh(box)   = directional coherent-mass, accumulated over window W
  curl(box)  = rotational curl of residual vectors, accumulated over window W
  fuse(box)  = max of the three, each normalized to [0,1] across the frame's boxes

For each channel we report, over scored oracle-hit frames AND over the subset that
`fpath_coh` currently misses, how often the real-shape box (the YOLO box nearest GT
and within radius) is ranked #1 and within top-3.

GATE: if `fuse` top-1 on the fpath_coh MISS frames is materially above `mass` top-1,
a fused-emission Viterbi integrator (Step 2) has signal to convert. If not, the main
lever is dead and we pivot. Read-only; reuses coh_gate's cached outlier vectors.

    python -m ld.detect.fuse_probe --weights .../best.pt --clips t1 t4 t5 t8
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from ld.config import DATA_DIR, DETECT_DIR
from ld.detect.coh_gate import _compute_ov, _box_coherent_mass
from ld.detect.eval_modes import _frame_wh, _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _centroid, compute_countdown_lock


def _box_curl(box, ov):
    """Rotational curl of residual vectors inside box (signed angular momentum,
    coherence-weighted). Orthogonal to coherent-mass: detects in-place spin."""
    if ov is None:
        return 0.0
    pix, resid, mag = ov
    if pix.shape[0] < 3:
        return 0.0
    inb = ((pix[:, 0] >= box[0]) & (pix[:, 0] <= box[2])
           & (pix[:, 1] >= box[1]) & (pix[:, 1] <= box[3]))
    if inb.sum() < 3:
        return 0.0
    P = pix[inb]
    V = resid[inb]
    c = P.mean(0)
    R = P - c
    cross = R[:, 0] * V[:, 1] - R[:, 1] * V[:, 0]
    den = float((np.linalg.norm(R, axis=1) * np.linalg.norm(V, axis=1)).sum())
    s = float(cross.sum())
    return abs(s) * (abs(s) / den) if den > 1e-6 else 0.0


def _windowed(idxs, ov, boxes, idx, win, fn):
    """Accumulate per-box score over the last `win` frames of outlier vectors."""
    pos_of = {ix: pos for pos, ix in enumerate(idxs)}
    pos = pos_of[idx]
    out = [0.0] * len(boxes)
    for back in range(win):
        j = pos - back
        if j < 0:
            break
        o = ov.get(idxs[j])
        if o is None:
            continue
        for bi, b in enumerate(boxes):
            out[bi] += fn(b, o)
    return out


def _gt_box_idx(boxes, gt, radius):
    """Index of the YOLO box nearest GT; None if none within radius (oracle miss)."""
    if not boxes or gt is None:
        return None
    dists = [math.hypot(_centroid(b)[0] - gt[0], _centroid(b)[1] - gt[1]) for b in boxes]
    bi = int(np.argmin(dists))
    return bi if dists[bi] < radius else None


def _rank_of(scores, target_i):
    """1-based rank of target_i when scores sorted descending (ties: worst case)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order.index(target_i) + 1


def _load_miss_frames(clip_stem, mode):
    """idx set the given mode currently MISSES while GT is in a box (identity creep)."""
    p = DETECT_DIR / "eval" / f"{clip_stem}__{mode}.csv"
    miss = set()
    if not p.exists():
        return miss
    for r in csv.DictReader(open(p)):
        if r["within_r"] == "0" and r["oracle_hit"] == "1":
            miss.add(int(r["idx"]))
    return miss


def run(weights, clips, win, miss_mode):
    chans = ["mass", "coh", "curl", "fuse"]
    # accumulators: per channel -> [top1_all, top3_all, n_all, top1_miss, top3_miss, n_miss]
    agg = {c: [0, 0, 0, 0, 0, 0] for c in chans}
    print(f"window={win}  miss frames from `{miss_mode}`\n")
    for clip in clips:
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        lock = compute_countdown_lock(packs, clip)
        from ld.detect.identity import _seed
        _, _, radius, start = _seed(packs)
        idxs, ov = _compute_ov(clip, packs)
        name = clip.stem.replace("_cropped_trimmed", "")
        miss = _load_miss_frames(clip.stem, miss_mode)
        # per-clip counters
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
            coh = _windowed(idxs, ov, p.boxes, idx, win, _box_coherent_mass)
            curl = _windowed(idxs, ov, p.boxes, idx, win, _box_curl)
            # saliency-mass proxy: summed outlier-residual magnitude in box, this frame
            # (what fpath's emission ranks on). Instantaneous, no window.
            o0 = ov.get(idx)
            mass = []
            for b in p.boxes:
                if o0 is None:
                    mass.append(0.0)
                    continue
                pix, _resid, magv = o0
                inb = ((pix[:, 0] >= b[0]) & (pix[:, 0] <= b[2])
                       & (pix[:, 1] >= b[1]) & (pix[:, 1] <= b[3]))
                mass.append(float(magv[inb].sum()))
            def norm(v):
                m = max(v) or 1.0
                return [x / m for x in v]
            mn, cn, rn = norm(mass), norm(coh), norm(curl)
            fuse = [max(mn[i], cn[i], rn[i]) for i in range(len(p.boxes))]
            chan_scores = {"mass": mass, "coh": coh, "curl": curl, "fuse": fuse}
            in_miss = idx in miss
            for c in chans:
                rk = _rank_of(chan_scores[c], gi)
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
            a = agg[c]
            for k in range(6):
                a[k] += loc[c][k]
            f1 = t1 / n if n else 0
            f3 = t3 / n if n else 0
            mf1 = mt1 / mn_ if mn_ else 0
            mf3 = mt3 / mn_ if mn_ else 0
            print(f"   {c:5}  all: top1={f1:.3f} top3={f3:.3f} (n={n})   "
                  f"MISS: top1={mf1:.3f} top3={mf3:.3f} (n={mn_})")
        print()
    print("=" * 70)
    print("OVERALL (all laggard clips)")
    for c in chans:
        t1, t3, n, mt1, mt3, mn_ = agg[c]
        print(f"   {c:5}  all: top1={t1/n if n else 0:.3f} top3={t3/n if n else 0:.3f}   "
              f"MISS: top1={mt1/mn_ if mn_ else 0:.3f} top3={mt3/mn_ if mn_ else 0:.3f} "
              f"(miss_n={mn_})")
    print("\nGATE: if `fuse` MISS-top1 >> `mass` MISS-top1, a fused-emission Viterbi")
    print("integrator has signal to convert. If fuse~=mass, the lever is weak -> pivot.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=["t1", "t4", "t5", "t8"])
    ap.add_argument("--win", type=int, default=12)
    ap.add_argument("--miss-mode", default="fpath_coh")
    args = ap.parse_args()
    clips = [c for c in _default_clips()
             if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.win, args.miss_mode)


if __name__ == "__main__":
    main()
