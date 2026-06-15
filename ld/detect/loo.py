"""Leave-one-clip-out validation for the position-field tracker.

The field tracker's snap config was tuned on the same t1-t10 set we score on, so
its in-sample mean (~0.71) is optimistic. This runs honest leave-one-out: for each
held-out clip, pick the best field config by mean within_r on the *other* 9 clips,
apply it to the held-out clip, and record that score. The mean of held-out scores
is the generalization estimate; the per-fold winning config shows whether the
choice is stable.

The grid below is the search space — held-out numbers are only honest relative to
it. All runs are cache-backed (detections) but read frames for the saliency pass.

Usage:
    python -m ld.detect.loo --weights data/detect/runs/yolov8n_combined/weights/best.pt
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from ld.config import DATA_DIR
from ld.detect.eval_modes import _default_clips, _frame_wh
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import (compute_countdown_lock, score_identity,
                                track_field_identity, track_field_lag_identity)

# Search space for the field tracker. Held-out numbers are honest only w.r.t. this.
GRID = [
    {"snap_mode": "mass", "snap_feedback": True, "snap_frac": 1.0},
    {"snap_mode": "mass", "snap_feedback": True, "snap_frac": 1.5},
    {"snap_mode": "mass", "snap_feedback": True, "snap_frac": 0.7},
    {"snap_mode": "mass", "snap_feedback": False, "snap_frac": 1.0},
    {"snap_mode": "nearest", "snap_feedback": True, "snap_frac": 1.0},
    {"snap_mode": "nearest", "snap_feedback": False, "snap_frac": 1.0},
]

# Search space for the fixed-lag confirmation smoother (mode "field_lag").
FIELD_LAG_GRID = [
    {"lag_k": k, "confirm": c}
    for k in (8, 12, 15)
    for c in (0.5, 0.6, 0.75)
]


@dataclass
class Ctx:
    clip: Path
    packs: list
    lock: object
    wh: tuple


def _score(ctx: Ctx, cfg: dict, mode: str = "field") -> float:
    name = ctx.clip.stem.replace("_cropped_trimmed", "")
    if mode == "field_lag":
        tr, _, s, r = track_field_lag_identity(ctx.clip, ctx.packs, ctx.lock,
                                               frame_wh=ctx.wh, **cfg)
    else:
        tr, _, s, r = track_field_identity(ctx.clip, ctx.packs, ctx.lock,
                                           frame_wh=ctx.wh, **cfg)
    return score_identity(ctx.packs, tr, s, r, name).within_r


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-one-clip-out for the field tracker")
    ap.add_argument("--weights", default="data/detect/runs/yolov8n_combined/weights/best.pt")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--mode", default="field", choices=("field", "field_lag"),
                    help="which tracker's grid to LOO-validate")
    args = ap.parse_args()
    grid = FIELD_LAG_GRID if args.mode == "field_lag" else GRID
    clip_paths = ([Path(c) if Path(c).exists() else DATA_DIR / f"{c}_cropped_trimmed.mp4"
                   for c in args.clips] if args.clips else _default_clips())

    # Precompute detections/lock/frame-size, and the full score matrix (cfg x clip).
    ctxs: dict[str, Ctx] = {}
    for cp in clip_paths:
        packs = detect_fusion_clip(args.weights, cp, use_cache=True)
        ctxs[cp.stem] = Ctx(cp, packs, compute_countdown_lock(packs, cp), _frame_wh(cp))
    names = list(ctxs)
    print(f"scoring {len(grid)} configs x {len(names)} clips (mode={args.mode}) ...")
    matrix: dict[tuple[int, str], float] = {}
    for gi, cfg in enumerate(grid):
        for n in names:
            matrix[(gi, n)] = _score(ctxs[n], cfg, mode=args.mode)
        print(f"  cfg{gi} {cfg} done")

    # Baseline field per-clip scores for regression checking (best in-sample field cfg).
    base = {n: _score(ctxs[n], GRID[0], mode="field") for n in names}

    # In-sample best (single config, best mean over all clips).
    in_means = [(gi, statistics.mean(matrix[(gi, n)] for n in names)) for gi in range(len(grid))]
    best_gi, best_in = max(in_means, key=lambda kv: kv[1])

    # Leave-one-out: per held-out clip, choose best cfg on the other clips.
    held = []
    print(f"\n{'clip':>5}  held-out  base(field)  delta  picked-cfg")
    for n in names:
        others = [m for m in names if m != n]
        gi = max(range(len(grid)),
                 key=lambda g: statistics.mean(matrix[(g, m)] for m in others))
        held.append(matrix[(gi, n)])
        nm = n.split('_')[0]
        print(f"{nm:>5}  {matrix[(gi, n)]:.3f}     {base[n]:.3f}     "
              f"{matrix[(gi, n)] - base[n]:+.3f}  {grid[gi]}")

    print(f"\n{'='*60}")
    print(f"in-sample best config = cfg{best_gi} {grid[best_gi]}  mean={best_in:.3f}")
    print(f"leave-one-out mean within_r = {statistics.mean(held):.3f}   "
          f"(honest generalization estimate)")
    print(f"baseline field LOO ref = 0.693; field in-sample mean here = "
          f"{statistics.mean(base.values()):.3f}")
    print(f"optimism gap (in-sample - LOO) = {best_in - statistics.mean(held):.3f}")


if __name__ == "__main__":
    main()
