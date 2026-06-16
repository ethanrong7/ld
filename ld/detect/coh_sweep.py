"""Validate the coherence snap-weight blend through the REAL field_lag tracker.

Unlike coh_gate (a standalone post-hoc override with its own scorer), this runs the
production `track_field_lag_identity` with coh_lambda>0 wired into the snap, scored by
score_identity. Sweeps coh_lambda over a window, reports per-clip + honest LOO with the
project's no-regression bar (worst-clip delta >= -0.004 vs the coh_lambda=0 baseline).

  python -m ld.detect.coh_sweep --weights .../best.pt
  python -m ld.detect.coh_sweep --weights .../best.pt --lambdas 0.5 1.0 2.0 --wins 16
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import (
    compute_countdown_lock, score_identity, track_field_lag_identity,
)
from ld.detect.eval_modes import _frame_wh, _default_clips


def _score(weights, clip: Path, coh_lambda, coh_win):
    packs = detect_fusion_clip(weights, clip, use_cache=True)
    lock = compute_countdown_lock(packs, clip)
    wh = _frame_wh(clip)
    name = clip.stem.replace("_cropped_trimmed", "")
    track, _h, start, radius = track_field_lag_identity(
        clip, packs, lock, frame_wh=wh, coh_lambda=coh_lambda, coh_win=coh_win)
    return name, score_identity(packs, track, start, radius, name).within_r


def run(weights, clips, lambdas, wins):
    names = [c.stem.replace("_cropped_trimmed", "") for c in clips]
    # baseline coh_lambda=0 (== shipped field_lag)
    base = {}
    for c in clips:
        n, wr = _score(weights, c, 0.0, wins[0])
        base[n] = wr
        print(f"  base {n}: field_lag within_r={wr:.3f}")
    base_mean = statistics.mean(base.values())
    print(f"\nfield_lag baseline mean within_r = {base_mean:.4f}\n")

    # scored[(lam,win)] = {clip: wr}
    scored = {}
    cfgs = [(lam, w) for lam in lambdas for w in wins if lam > 0]
    print("=== in-sample grid (mean within_r, delta vs field_lag) ===")
    for cfg in cfgs:
        wr = {}
        for c in clips:
            n, v = _score(weights, c, cfg[0], cfg[1])
            wr[n] = v
        scored[cfg] = wr
        m = statistics.mean(wr.values())
        worst = min(wr[n] - base[n] for n in names)
        print(f"  lam={cfg[0]:.2f} win={cfg[1]:>2} | mean={m:.4f} "
              f"({m-base_mean:+.4f})  worst_clip={worst:+.4f}")

    # Honest LOO: pick the config maximizing worst-fold delta then mean, no regression.
    print("\n=== leave-one-clip-out (honest, robustness-first) ===")
    loo = []
    for hi, held in enumerate(names):
        best_cfg, best_key = None, None
        for cfg in cfgs:
            wr = scored[cfg]
            others = [wr[n] - base[n] for n in names if n != held]
            worst_d = min(others)
            if worst_d < -0.004:
                continue
            key = (round(worst_d, 4), round(statistics.mean(others), 4))
            if best_key is None or key > best_key:
                best_key, best_cfg = key, cfg
        if best_cfg is None:
            loo.append(base[held])
            print(f"  {held:>4}: no admissible cfg -> field_lag {base[held]:.3f}")
        else:
            v = scored[best_cfg][held]
            loo.append(v)
            print(f"  {held:>4}: {v:.3f} (base {base[held]:.3f}, {v-base[held]:+.3f})  "
                  f"cfg=lam{best_cfg[0]} win{best_cfg[1]}")
    loo_mean = statistics.mean(loo)
    worst = min(loo[i] - base[names[i]] for i in range(len(names)))
    print(f"\nLOO mean within_r = {loo_mean:.4f}  (field_lag {base_mean:.4f}, "
          f"{loo_mean-base_mean:+.4f})  worst_clip={worst:+.4f}")
    print("VERDICT:", "PASS" if (loo_mean > base_mean and worst >= -0.004) else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--wins", nargs="+", type=int, default=[16])
    args = ap.parse_args()
    clips = _default_clips()
    if args.clips:
        clips = [c for c in clips
                 if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.lambdas, args.wins)


if __name__ == "__main__":
    main()
