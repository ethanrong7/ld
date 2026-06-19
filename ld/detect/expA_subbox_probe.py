"""EXP-A GATE (read-only): does coherent-mass measured over a re-localized SUB-CLUSTER
inside each YOLO box rank the real-shape box #1 more often than whole-box coherent-mass?

Diagnosed wall (plan.md / Step-0): on the laggard lock frames the real box is present
(oracle ~0.95) but the windowed coherent-mass channel ranks a fake #1 (on-GT <0.30 on t8).
Hypothesis: oversized / merged GT boxes (t8 f656-719 up to 1.97x, t5 undersized) scoop up
peripheral INCOHERENT noise outliers, inflating the coherence denominator Sum||v|| without
raising the resultant ||Sum v|| -> the real shape's coherent signal is diluted. Restricting
the measurement window to the tight cluster where the coherent vectors actually concentrate
should recover the rank.

Channels compared (each accumulated over window W, same as fpath_fuse's cmass):
  whole      = _box_coherent_mass over the full box (current fpath_fuse channel; BASELINE)
  wcentroid  = coherent-mass over outliers within sub_frac*radius of the alignment-weighted
               centroid of in-box outliers
  topk       = coherent-mass over the top `keep_frac` in-box outliers by projection onto the
               in-box resultant direction (drops vectors that disagree with the majority)
  grid       = coherent-mass over the densest cell of a coarse grid partition of the box

Metric: over scored oracle-hit frames AND over the fpath_fuse MISS subset, how often the
real box (YOLO box nearest GT, within radius) is ranked #1 / top-3 by each channel.

GATE: on t8/t5 MISS frames, a sub-cluster channel must lift on-GT top1 materially above
`whole` (t8 baseline ~0.29, target ~0.5) WITHOUT lowering top1 on the strong clips
(t2/t6/t7/t9/t10). If yes -> build it into the trellis (Step 1b). If no -> EXP-A dead.

    python -m ld.detect.expA_subbox_probe --weights .../best.pt
    python -m ld.detect.expA_subbox_probe --weights .../best.pt --clips t8 t5 --miss-mode fpath_fuse
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ld.detect.coh_gate import _compute_ov, _box_coherent_mass
from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _centroid, _seed, compute_countdown_lock
from ld.detect.fuse_probe import _gt_box_idx, _rank_of, _load_miss_frames

STRONG = {"t2", "t6", "t7", "t9", "t10"}
LAGGARD = {"t1", "t5", "t8"}


def _coh_from_subset(pix_sub, resid_sub):
    """Coherent-mass (||Sum v||^2 / Sum||v||) over an already-selected subset."""
    if pix_sub.shape[0] < 2:
        return 0.0
    net = float(np.linalg.norm(resid_sub.sum(0)))
    tot = float(np.linalg.norm(resid_sub, axis=1).sum())
    return net * (net / tot) if tot > 1e-6 else 0.0


def _subbox_coherent_mass(box, ov, mode, *, radius, sub_frac=0.5, keep_frac=0.6,
                          min_pts=4):
    """Coherent-mass over a re-localized sub-cluster of the in-box outlier vectors.
    Falls back to whole-box coherent-mass when too few points to re-localize."""
    if ov is None:
        return 0.0
    pix, resid, mag = ov
    if pix.shape[0] == 0:
        return 0.0
    inb = ((pix[:, 0] >= box[0]) & (pix[:, 0] <= box[2])
           & (pix[:, 1] >= box[1]) & (pix[:, 1] <= box[3]))
    n = int(inb.sum())
    if n < 2:
        return 0.0
    P = pix[inb]
    V = resid[inb]
    if mode == "whole" or n < min_pts:
        return _coh_from_subset(P, V)

    res = V.sum(0)
    rn = float(np.linalg.norm(res))
    if rn < 1e-6:
        return _coh_from_subset(P, V)
    dhat = res / rn

    if mode == "wcentroid":
        # alignment-weighted centroid: where the vectors agreeing with the resultant sit
        w = np.clip(V @ dhat, 0.0, None)
        if w.sum() < 1e-6:
            return _coh_from_subset(P, V)
        c = (P * w[:, None]).sum(0) / w.sum()
        d = np.linalg.norm(P - c, axis=1)
        sel = d <= sub_frac * radius
        if sel.sum() < 2:
            sel = d <= np.median(d)
        return _coh_from_subset(P[sel], V[sel])

    if mode == "topk":
        proj = V @ dhat
        k = max(2, int(round(keep_frac * n)))
        order = np.argsort(proj)[::-1][:k]
        return _coh_from_subset(P[order], V[order])

    if mode == "grid":
        # densest 3x3 cell of the box, by outlier count
        gx = np.clip(((P[:, 0] - box[0]) / max(box[2] - box[0], 1e-6) * 3).astype(int), 0, 2)
        gy = np.clip(((P[:, 1] - box[1]) / max(box[3] - box[1], 1e-6) * 3).astype(int), 0, 2)
        cell = gy * 3 + gx
        counts = np.bincount(cell, minlength=9)
        best = int(np.argmax(counts))
        sel = cell == best
        if sel.sum() < 2:
            return _coh_from_subset(P, V)
        return _coh_from_subset(P[sel], V[sel])

    raise ValueError(mode)


def _windowed_sub(idxs, ov, boxes, idx, win, mode, radius, sub_frac, keep_frac):
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
            out[bi] += _subbox_coherent_mass(b, o, mode, radius=radius,
                                             sub_frac=sub_frac, keep_frac=keep_frac)
    return out


def run(weights, clips, win, miss_mode, sub_frac, keep_frac):
    modes = ["whole", "wcentroid", "topk", "grid"]
    # per channel -> [top1_all, top3_all, n_all, top1_miss, top3_miss, n_miss]
    agg_strong = {m: [0, 0, 0, 0, 0, 0] for m in modes}
    agg_lag = {m: [0, 0, 0, 0, 0, 0] for m in modes}
    print(f"window={win}  miss frames from `{miss_mode}`  sub_frac={sub_frac} keep_frac={keep_frac}\n")
    for clip in clips:
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        compute_countdown_lock(packs, clip)
        _, _, radius, start = _seed(packs)
        idxs, ov = _compute_ov(clip, packs)
        name = clip.stem.replace("_cropped_trimmed", "")
        miss = _load_miss_frames(clip.stem, miss_mode)
        loc = {m: [0, 0, 0, 0, 0, 0] for m in modes}
        for idx in idxs:
            if idx < start or idx >= len(packs):
                continue
            p = packs[idx]
            if p.gt is None or not p.boxes:
                continue
            gi = _gt_box_idx(p.boxes, p.gt, radius)
            if gi is None:
                continue
            in_miss = idx in miss
            for m in modes:
                sc = _windowed_sub(idxs, ov, p.boxes, idx, win, m, radius,
                                   sub_frac, keep_frac)
                rk = _rank_of(sc, gi)
                loc[m][2] += 1
                if rk == 1: loc[m][0] += 1
                if rk <= 3: loc[m][1] += 1
                if in_miss:
                    loc[m][5] += 1
                    if rk == 1: loc[m][3] += 1
                    if rk <= 3: loc[m][4] += 1
        bucket = "LAGGARD" if name in LAGGARD else ("STRONG" if name in STRONG else "other")
        print(f"[{name}] ({bucket})  radius={radius:.0f}  miss_frames={len(miss)}")
        for m in modes:
            t1, t3, n, mt1, mt3, mn_ = loc[m]
            tgt = agg_lag if name in LAGGARD else (agg_strong if name in STRONG else None)
            if tgt is not None:
                for k in range(6):
                    tgt[m][k] += loc[m][k]
            f1 = t1 / n if n else 0
            mf1 = mt1 / mn_ if mn_ else 0
            mf3 = mt3 / mn_ if mn_ else 0
            print(f"   {m:10} all-top1={f1:.3f}   MISS top1={mf1:.3f} top3={mf3:.3f} (n={mn_})")
        print()
    print("=" * 72)
    for label, agg in (("LAGGARD (t1/t5/t8)", agg_lag), ("STRONG (t2/t6/t7/t9/t10)", agg_strong)):
        print(f"OVERALL {label}")
        for m in modes:
            t1, t3, n, mt1, mt3, mn_ = agg[m]
            print(f"   {m:10} all-top1={t1/n if n else 0:.3f}   "
                  f"MISS top1={mt1/mn_ if mn_ else 0:.3f} top3={mt3/mn_ if mn_ else 0:.3f} (miss_n={mn_})")
        print()
    print("GATE: a sub-cluster channel must lift LAGGARD MISS-top1 well above `whole`")
    print("WITHOUT lowering STRONG all-top1 below `whole`.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*",
                    default=["t1", "t2", "t5", "t6", "t7", "t8", "t9", "t10"])
    ap.add_argument("--win", type=int, default=12)
    ap.add_argument("--miss-mode", default="fpath_fuse")
    ap.add_argument("--sub-frac", type=float, default=0.5)
    ap.add_argument("--keep-frac", type=float, default=0.6)
    args = ap.parse_args()
    clips = [c for c in _default_clips()
             if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.win, args.miss_mode, args.sub_frac, args.keep_frac)


if __name__ == "__main__":
    main()
