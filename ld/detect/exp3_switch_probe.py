"""EXP-3 GATE: would an evidence-hysteresis switch WITHOUT the far-distance
requirement catch the trellis's lock-in misses?

step3_gate killed the coherence FAR-JUMP reacquire because (a) t1's creep is onto an
ADJACENT box (far filter never fires) and (b) on t5/t8 the far coherent challenger is
the WRONG box. But that test gated on far+confident+coherent. plan.md EXP-3 asks the
narrower question that a hysteresis switch actually needs answered:

  On the `fpath_fuse` MISS frames, DROPPING the far filter entirely, does the box with
  the highest accumulated coherent-mass actually SIT ON GT (within radius)? And does a
  LONGER accumulation window (24-40 frames) or a leaky EMA raise that on-GT fraction
  materially above step3_gate's win12 numbers (t1 0.39 / t5 0.40 / t8 0.29)?

If on-GT clears ~0.5 on at least one laggard, a distance-agnostic accumulated-evidence
override has a real target -> build the hysteresis switch (EXP-3 build step) and LOO-tune
it. If every window stays ~0.3-0.4, the channel cannot point at the real box often enough
for ANY switch (far or near) to win, and EXP-3 is dead for the same signal-limit reason.

We report, per clip and per window, over the MISS frames:
  on-GT        = argmax(accumulated coh-mass) box centroid within radius of GT
  on-GT(EMA)   = same, but leaky-integrated (alpha) instead of a hard window
  margin@hit   = mean top-vs-runnerup margin on the frames where argmax IS on GT
                 (a switch needs a usable causal key, not just occasional correctness)

Read-only; reuses the cohgate outlier-vector cache.

    python -m ld.detect.exp3_switch_probe --weights .../best.pt --clips t1 t5 t8
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ld.detect.coh_gate import _compute_ov, _box_coherent_mass
from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.fuse_probe import _load_miss_frames
from ld.detect.identity import _centroid, compute_countdown_lock, _seed


def _acc_window(idxs, ov, boxes, pos, win):
    """Accumulated coherent-mass per box over the last `win` frames (current box
    location vs past outlier vectors), mirroring coh_gate's challenger."""
    out = [0.0] * len(boxes)
    for back in range(win):
        j = pos - back
        if j < 0:
            break
        o = ov.get(idxs[j])
        if o is None:
            continue
        for bi, b in enumerate(boxes):
            out[bi] += _box_coherent_mass(b, o)
    return out


def _argmax_margin(scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top = scores[order[0]]
    run = scores[order[1]] if len(order) > 1 else 0.0
    margin = (top - run) / (top + 1e-6) if top > 0 else 0.0
    return order[0], margin


def _on_gt(boxes, bi, gt, radius):
    c = _centroid(boxes[bi])
    return math.hypot(c[0] - gt[0], c[1] - gt[1]) <= radius


def run(weights, clips, windows, alpha, miss_mode):
    wins = sorted(windows)
    print(f"miss frames from `{miss_mode}`  windows={wins}  ema_alpha={alpha}\n")
    overall = {w: [0, 0] for w in wins}
    overall_ema = [0, 0]
    overall_margin = {w: [] for w in wins}
    for clip in clips:
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        compute_countdown_lock(packs, clip)
        _, _, radius, start = _seed(packs)
        idxs, ov = _compute_ov(clip, packs)
        miss = _load_miss_frames(clip.stem, miss_mode)
        name = clip.stem.replace("_cropped_trimmed", "")
        pos_of = {ix: pos for pos, ix in enumerate(idxs)}

        # leaky EMA of per-box coh-mass, carried at each box's CURRENT location
        # (re-seeded per frame from boxes; we accumulate by nearest-box association).
        per_win = {w: [0, 0] for w in wins}  # [on_gt, n]
        per_win_margin = {w: [] for w in wins}
        ema_hit = [0, 0]
        # maintain EMA keyed by box centroid via nearest carry-over
        prev_centroids = None
        prev_ema = None
        for idx in idxs:
            if idx < start or idx >= len(packs):
                continue
            p = packs[idx]
            if not p.boxes:
                prev_centroids, prev_ema = None, None
                continue
            pos = pos_of[idx]
            o = ov.get(idx)
            inst = [_box_coherent_mass(b, o) for b in p.boxes]
            cents = [_centroid(b) for b in p.boxes]
            # EMA update: carry prev EMA to nearest current box
            ema = list(inst)
            if prev_ema is not None:
                for bi, c in enumerate(cents):
                    # nearest prev centroid
                    bd, bj = 1e18, -1
                    for pj, pc in enumerate(prev_centroids):
                        d = (pc[0] - c[0]) ** 2 + (pc[1] - c[1]) ** 2
                        if d < bd:
                            bd, bj = d, pj
                    carry = prev_ema[bj] if bj >= 0 else 0.0
                    ema[bi] = alpha * carry + (1 - alpha) * inst[bi]
            prev_centroids, prev_ema = cents, ema

            if idx not in miss or p.gt is None:
                continue
            for w in wins:
                sc = _acc_window(idxs, ov, p.boxes, pos, w)
                bi, margin = _argmax_margin(sc)
                hit = _on_gt(p.boxes, bi, p.gt, radius)
                per_win[w][0] += int(hit)
                per_win[w][1] += 1
                if hit:
                    per_win_margin[w].append(margin)
            bi_e, _ = _argmax_margin(ema)
            ema_hit[0] += int(_on_gt(p.boxes, bi_e, p.gt, radius))
            ema_hit[1] += 1

        print(f"[{name}]  radius={radius:.0f}  miss_frames={len(miss)}")
        for w in wins:
            h, n = per_win[w]
            mg = np.mean(per_win_margin[w]) if per_win_margin[w] else 0.0
            print(f"   win{w:>2}: on-GT={h/n if n else 0:.3f} (n={n})  margin@hit={mg:.2f}")
            overall[w][0] += h
            overall[w][1] += n
            overall_margin[w].extend(per_win_margin[w])
        h, n = ema_hit
        print(f"   EMA a={alpha}: on-GT={h/n if n else 0:.3f} (n={n})")
        overall_ema[0] += h
        overall_ema[1] += n
        print()

    print("=" * 64)
    print("OVERALL (miss frames, all probed clips)")
    for w in wins:
        h, n = overall[w]
        mg = np.mean(overall_margin[w]) if overall_margin[w] else 0.0
        print(f"   win{w:>2}: on-GT={h/n if n else 0:.3f} (n={n})  margin@hit={mg:.2f}")
    h, n = overall_ema
    print(f"   EMA a={alpha}: on-GT={h/n if n else 0:.3f} (n={n})")
    print("\nGATE: build the hysteresis switch only if on-GT clears ~0.5 on a laggard")
    print("AND margin@hit is usable. Else the channel can't point at the real box -> dead.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=["t1", "t5", "t8"])
    ap.add_argument("--windows", nargs="+", type=int, default=[12, 24, 32, 40])
    ap.add_argument("--alpha", type=float, default=0.92)
    ap.add_argument("--miss-mode", default="fpath_fuse")
    args = ap.parse_args()
    clips = [c for c in _default_clips()
             if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.windows, args.alpha, args.miss_mode)


if __name__ == "__main__":
    main()
