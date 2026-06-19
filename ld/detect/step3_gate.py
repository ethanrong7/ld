"""STEP-3 GATE: does a coherence-driven reacquire have a TARGET on fpath_fuse's
residual lock-in misses?

fpath_fuse (0.876) still loses t1/t8/t5 to path lock-in: the Viterbi path commits to a
fake and the anti-jump transition penalty holds it there. A reacquire can only help if,
on those locked-wrong frames, a coherent-mass challenger that is FAR from the path's
(wrong) position actually sits on GT -- otherwise there is nothing to jump to and the
lever is dead (pivot to fixed-lag decode instead).

This measures exactly that. For each clip, run fpath_fuse, find its miss frames (path
outside radius while GT is in a box = identity creep), and on those frames compute the
windowed coherent-mass argmax challenger (coh_gate's, the same signal the emission uses).
Report, of the miss frames:
  on_gt        : challenger centroid within radius of GT (a correct target exists)
  far          : challenger > far_r radii from the path's current (wrong) position
  recoverable  : on_gt AND far AND confident (margin>=tau) -- a gated far-jump would fix it

GATE: recoverable >~0.3 on t1/t8 => build the coherence reacquire. <~0.15 => pivot.
Read-only; reuses coh_gate's cached outlier vectors.

    python -m ld.detect.step3_gate --weights .../best.pt --clips t1 t8 t5
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from ld.detect.coh_gate import _compute_ov, _challenger_per_frame
from ld.detect.eval_modes import _default_clips, _frame_wh
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _centroid, _dispatch_mode, compute_countdown_lock


def run(weights, clips, win, tau, far_r):
    print(f"window={win}  tau={tau}  far_r={far_r}\n")
    tot = {"miss": 0, "on_gt": 0, "far": 0, "recov": 0}
    for clip in clips:
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        lock = compute_countdown_lock(packs, clip)
        wh = _frame_wh(clip)
        name = clip.stem.replace("_cropped_trimmed", "")
        track, _h, start, radius = _dispatch_mode(clip, packs, lock, wh, "fpath_fuse")
        tp_by = {tp.idx: tp for tp in track}
        idxs, ov = _compute_ov(clip, packs)
        chal = _challenger_per_frame(packs, idxs, ov, win)
        far_px = far_r * radius

        c = {"miss": 0, "on_gt": 0, "far": 0, "recov": 0}
        for idx in idxs:
            if idx < start or idx >= len(packs):
                continue
            p = packs[idx]
            tp = tp_by.get(idx)
            if p.gt is None or not p.boxes or tp is None:
                continue
            # identity creep: path wrong AND GT is in some box (oracle hit)
            path_err = math.hypot(tp.x - p.gt[0], tp.y - p.gt[1])
            oracle_err = min(math.hypot(_centroid(b)[0] - p.gt[0], _centroid(b)[1] - p.gt[1])
                             for b in p.boxes)
            if path_err < radius or oracle_err >= radius:
                continue
            c["miss"] += 1
            ch = chal.get(idx)
            if ch is None:
                continue
            (chx, chy), margin, _top = ch
            on_gt = math.hypot(chx - p.gt[0], chy - p.gt[1]) < radius
            far = math.hypot(chx - tp.x, chy - tp.y) > far_px
            if on_gt:
                c["on_gt"] += 1
            if far:
                c["far"] += 1
            if on_gt and far and margin >= tau:
                c["recov"] += 1
        for k in tot:
            tot[k] += c[k]
        m = c["miss"] or 1
        print(f"[{name:>4}] miss={c['miss']:>4}  on_gt={c['on_gt']/m:.3f}  "
              f"far={c['far']/m:.3f}  recoverable={c['recov']/m:.3f}  "
              f"(recov_n={c['recov']})")
    m = tot["miss"] or 1
    print(f"\nOVERALL miss={tot['miss']}  on_gt={tot['on_gt']/m:.3f}  "
          f"far={tot['far']/m:.3f}  recoverable={tot['recov']/m:.3f}")
    print("GATE: recoverable >~0.30 on t1/t8 => build the coherence reacquire;"
          " <~0.15 => pivot to fixed-lag decode.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=["t1", "t8", "t5"])
    ap.add_argument("--win", type=int, default=12)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--far-r", type=float, default=1.5)
    args = ap.parse_args()
    clips = [c for c in _default_clips()
             if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.win, args.tau, args.far_r)


if __name__ == "__main__":
    main()
