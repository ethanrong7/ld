"""GATE (read-only) for HUMAN-LIKE CURSOR PHYSICS (plan.md 2026-06-21).

The pipeline now emits a single point per frame (rendered as a filled red dot).
Visually it is correct but JITTERY and machine-like: it wobbles a few px every
frame while locked and TELEPORTS when the identity pick hops boxes or the
residual-freeze snaps to / releases an anchor. A human traces with inertia and
bounded dynamics. ``ld/track/humanize.py`` (1-Euro + deadband + bounded-velocity
PD) reshapes the EMITTED stream to read like a hand WITHOUT regressing within_r.

This probe is the offline gate (no detector / trellis re-run): the per-frame
emitted (x,y) and GT already live in the eval CSVs
``data/detect/eval/<stem>__fpath_freeze.csv``. For each clip on BOTH boards it:

  Phase 0 -- MEASURE the baseline smoothness metrics on the raw emitted stream
    (RMS accel, RMS jerk, velocity-reversal rate, p99/max per-frame jump) and
    DECOMPOSE the jitter into (a) high-frequency lock-wobble vs (b) discrete
    hops/teleports (share of total jerk from each). This is the number to beat.

  Phase 2 -- GATE: replay ``humanize_track`` over each stream for a grid of
    params; recompute (i) the smoothness metrics and (ii) within_r against the GT
    with the clip radius. Report per-clip + mean + worst-clip delta; find the best
    config that holds the guardrail (within_r mean flat-or-up, NO per-clip
    regression) AND clears the smoothness bars on EVERY clip, on BOTH boards;
    confirm with an honest LOO over the param grid.

Radius per clip via cached packs + ``identity._seed`` (the fusion cache is JSON,
so this is detector-free and fast). GT is GT-only -- never fed into the filter.

    python -m ld.detect.cursor_physics_probe --weights data/detect/runs/.../best.pt
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from ld.config import DATA_DIR
from ld.detect.eval_modes import _default_clips, _frame_wh
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _seed
from ld.detect.probe_common import _add_clips
from ld.track.humanize import humanize_track

MODE = "fpath_freeze"
EVAL_DIR = DATA_DIR / "detect" / "eval"
FPS = 60.0
# A "hop" = a per-frame jump above the real-shape p99 (CLAUDE.md Physics: p99 17.8,
# max 44.7 px/fr). Jumps above this are decode-layer snaps/box-hops, not real motion.
HOP_THRESH = 18.0
REG_BAR = -0.004                # per-clip within_r regression tolerance (mirror hedge/freeze probes)
# Smoothness acceptance bars (plan.md Phase 0 suggested), enforced on EVERY clip:
JERK_DROP_BAR = 0.40            # RMS jerk must fall >= 40%
REV_DROP_BAR = 0.20            # velocity-reversal rate must fall >= 20% (materially)


# ---------------------------------------------------------------------------
# IO

def _read_stream(stem: str):
    """Ordered per-frame (p=(x,y), gt=(x,y)|None) from the fpath_freeze eval CSV."""
    path = EVAL_DIR / f"{stem}__{MODE}.csv"
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            gt = None
            if r["gt_x"] != "" and r["gt_y"] != "":
                gt = (float(r["gt_x"]), float(r["gt_y"]))
            rows.append((int(r["idx"]), (float(r["x"]), float(r["y"])), gt,
                         int(float(r["within_r"])) if r["within_r"] != "" else 0))
    rows.sort(key=lambda d: d[0])
    return rows


def _radius(weights: str, clip: Path) -> float:
    packs = detect_fusion_clip(weights, clip, use_cache=True)
    return _seed(packs)[2]


def _name(clip: Path) -> str:
    return clip.stem.split("_")[0]


# ---------------------------------------------------------------------------
# Metrics

def _within_r(pts, gts, radius) -> float:
    hit = tot = 0
    for p, g in zip(pts, gts):
        if g is None:
            continue
        tot += 1
        if math.hypot(p[0] - g[0], p[1] - g[1]) < radius:
            hit += 1
    return hit / tot if tot else float("nan")


def _smoothness(pts) -> dict:
    """RMS accel / RMS jerk / velocity-reversal rate / p99 & max jump, plus the
    jerk-share from hop frames (jump > HOP_THRESH) vs wobble frames."""
    n = len(pts)
    jumps = [math.hypot(pts[t][0] - pts[t - 1][0], pts[t][1] - pts[t - 1][1])
             for t in range(1, n)]
    accel = [math.hypot(pts[t][0] - 2 * pts[t - 1][0] + pts[t - 2][0],
                        pts[t][1] - 2 * pts[t - 1][1] + pts[t - 2][1])
             for t in range(2, n)]
    jerk = [math.hypot(pts[t][0] - 3 * pts[t - 1][0] + 3 * pts[t - 2][0] - pts[t - 3][0],
                       pts[t][1] - 3 * pts[t - 1][1] + 3 * pts[t - 2][1] - pts[t - 3][1])
            for t in range(3, n)]
    # velocity reversals: sign flip of successive displacement vectors (dot < 0)
    rev = 0
    for t in range(2, n):
        ax, ay = pts[t][0] - pts[t - 1][0], pts[t][1] - pts[t - 1][1]
        bx, by = pts[t - 1][0] - pts[t - 2][0], pts[t - 1][1] - pts[t - 2][1]
        if ax * bx + ay * by < 0:
            rev += 1
    # jerk decomposition: a jerk sample at t touches frames t-3..t; attribute it to
    # "hop" if any jump in that window exceeds HOP_THRESH, else "wobble".
    hop_jerk = wob_jerk = 0.0
    for k, t in enumerate(range(3, n)):
        win_hop = any(jumps[i - 1] > HOP_THRESH for i in (t - 2, t - 1, t))
        if win_hop:
            hop_jerk += jerk[k] ** 2
        else:
            wob_jerk += jerk[k] ** 2
    tot_jerk = hop_jerk + wob_jerk

    def _rms(xs):
        return math.sqrt(sum(v * v for v in xs) / len(xs)) if xs else 0.0

    sj = sorted(jumps)
    p99 = sj[min(len(sj) - 1, int(0.99 * len(sj)))] if sj else 0.0
    return dict(
        rms_accel=_rms(accel), rms_jerk=_rms(jerk),
        rev_rate=rev / max(n - 2, 1),
        p99_jump=p99, max_jump=max(jumps) if jumps else 0.0,
        hop_jerk_frac=(hop_jerk / tot_jerk) if tot_jerk > 1e-9 else 0.0,
    )


# ---------------------------------------------------------------------------
# Param grid

def _grid() -> list[tuple[str, dict]]:
    """(label, humanize_track kwargs). Built simplest-first (1-Euro -> +deadband
    -> +PD -> +lag) so the gate can prefer the fewest params (LOO overfit)."""
    cfgs: list[tuple[str, dict]] = []
    # 1-Euro only
    for mc in (0.5, 1.0, 2.0):
        for beta in (0.003, 0.007, 0.015):
            cfgs.append((f"euro mc{mc} b{beta}",
                         dict(min_cutoff=mc, beta=beta)))
    # 1-Euro + deadband
    for mc in (0.5, 1.0):
        for db in (1.0, 2.0):
            cfgs.append((f"euro mc{mc} b0.007 db{db}",
                         dict(min_cutoff=mc, beta=0.007, deadband=db)))
    # 1-Euro + PD steering
    for k in (0.2, 0.4):
        cfgs.append((f"euro+pd k{k} v25 a8",
                     dict(min_cutoff=1.0, beta=0.007, use_pd=True,
                          k=k, v_max=25.0, a_max=8.0)))
    # 1-Euro + small fixed lag
    for lag in (8, 12):
        cfgs.append((f"euro mc1.0 b0.007 lag{lag}",
                     dict(min_cutoff=1.0, beta=0.007, lag=lag)))
    return cfgs


# ---------------------------------------------------------------------------
# Per-board run

def _load_board(weights: str, clips: list[Path]):
    """-> list of dicts {name, pts, gts, radius, base_wr, base_sm}."""
    out = []
    for clip in clips:
        rows = _read_stream(clip.stem)
        pts = [r[1] for r in rows]
        gts = [r[2] for r in rows]
        radius = _radius(weights, clip)
        out.append(dict(name=_name(clip), pts=pts, gts=gts, radius=radius,
                        base_wr=_within_r(pts, gts, radius),
                        base_sm=_smoothness(pts)))
    return out


def _bars_met(base_sm, sm) -> bool:
    jd = 1.0 - sm["rms_jerk"] / base_sm["rms_jerk"] if base_sm["rms_jerk"] > 1e-9 else 0.0
    rd = (1.0 - sm["rev_rate"] / base_sm["rev_rate"]) if base_sm["rev_rate"] > 1e-9 else 1.0
    return jd >= JERK_DROP_BAR and rd >= REV_DROP_BAR


def run_board(weights: str, clips: list[Path], label: str):
    print(f"\n{'='*78}\nBOARD: {label}\n{'='*78}")
    data = _load_board(weights, clips)
    names = [d["name"] for d in data]

    # ---- Phase 0: baseline table ----
    print("\n--- Phase 0: baseline smoothness (raw fpath_freeze emitted stream) ---")
    print(f"{'clip':>5} {'wr':>6} {'rmsAcc':>7} {'rmsJerk':>8} {'revRate':>8} "
          f"{'p99jmp':>7} {'maxjmp':>7} {'hopJerk%':>8}")
    for d in data:
        s = d["base_sm"]
        print(f"{d['name']:>5} {d['base_wr']:6.3f} {s['rms_accel']:7.2f} "
              f"{s['rms_jerk']:8.2f} {s['rev_rate']:8.3f} {s['p99_jump']:7.1f} "
              f"{s['max_jump']:7.1f} {s['hop_jerk_frac']*100:7.1f}%")
    print(f"{'MEAN':>5} {statistics.mean(d['base_wr'] for d in data):6.3f} "
          f"{statistics.mean(d['base_sm']['rms_accel'] for d in data):7.2f} "
          f"{statistics.mean(d['base_sm']['rms_jerk'] for d in data):8.2f} "
          f"{statistics.mean(d['base_sm']['rev_rate'] for d in data):8.3f}")

    # ---- Phase 2: evaluate the grid ----
    grid = _grid()
    # results[label][name] = (wr, delta, bars_met, sm)
    results: dict[str, dict] = {}
    for clabel, kw in grid:
        res = {}
        for d in data:
            sm_pts = humanize_track(d["pts"], FPS, **kw)
            wr = _within_r(sm_pts, d["gts"], d["radius"])
            res[d["name"]] = (wr, wr - d["base_wr"],
                              _bars_met(d["base_sm"], _smoothness(sm_pts)),
                              _smoothness(sm_pts))
        results[clabel] = res

    base = {d["name"]: d["base_wr"] for d in data}
    base_mean = statistics.mean(base.values())

    print(f"\n--- Phase 2: grid (within_r guardrail + smoothness bars), base mean {base_mean:.4f} ---")
    print(f"{'config':>26} {'wrMean':>7} {'dMean':>7} {'worst':>7} {'bars':>5} {'guard':>6}")
    admissible = []
    for clabel, _kw in grid:
        res = results[clabel]
        wrs = [res[n][0] for n in names]
        deltas = [res[n][1] for n in names]
        bars_all = all(res[n][2] for n in names)
        worst = min(deltas)
        guard = worst >= REG_BAR and statistics.mean(deltas) >= 0.0
        flag = "OK" if (bars_all and guard) else ""
        if bars_all and guard:
            admissible.append(clabel)
        print(f"{clabel:>26} {statistics.mean(wrs):7.4f} {statistics.mean(deltas):+7.4f} "
              f"{worst:+7.4f} {'Y' if bars_all else 'n':>5} {'Y' if guard else 'n':>6} {flag}")

    return data, names, base, results, grid, admissible


def loo(names, base, results, grid):
    """Honest LOO over the param grid: per held-out clip pick the best config that
    holds the guardrail + smoothness bars on the OTHER folds, score the held-out."""
    print("\n--- honest LOO over the param grid (no per-clip regression on train folds) ---")
    loo_vals = []
    for held in names:
        best_cfg, best_key = None, None
        for clabel, _kw in grid:
            res = results[clabel]
            train = [n for n in names if n != held]
            deltas = [res[n][1] for n in train]
            bars_all = all(res[n][2] for n in train)
            worst, mean_d = min(deltas), statistics.mean(deltas)
            if not bars_all or worst < REG_BAR or mean_d < 0.0:
                continue
            # worst-clip first (mirror hedge_probe): prefer the config that REGRESSES
            # LEAST on the train folds -> the most generalizing, not the most aggressive.
            key = (round(worst, 4), round(mean_d, 4))
            if best_key is None or key > best_key:
                best_key, best_cfg = key, clabel
        if best_cfg is None:
            loo_vals.append(base[held])
            print(f"   {held:>5}: no admissible config -> base {base[held]:.3f}")
        else:
            hv = results[best_cfg][held][0]
            loo_vals.append(hv)
            print(f"   {held:>5}: {hv:.3f} (base {base[held]:.3f}, {hv-base[held]:+.3f})  [{best_cfg}]")
    loo_mean = statistics.mean(loo_vals)
    base_mean = statistics.mean(base.values())
    worst = min(loo_vals[i] - base[names[i]] for i in range(len(names)))
    print(f"\n  LOO mean within_r = {loo_mean:.4f}  (base {base_mean:.4f}, "
          f"{loo_mean-base_mean:+.4f})  worst_clip={worst:+.4f}")
    return loo_mean, base_mean, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--board", choices=["t", "add", "both"], default="both")
    args = ap.parse_args()

    boards = []
    if args.board in ("t", "both"):
        boards.append((_default_clips(), "t1-t10"))
    if args.board in ("add", "both"):
        boards.append((_add_clips(), "additional_evidence (valid)"))

    summary = []
    per_board = {}
    for clips, label in boards:
        data, names, base, results, grid, adm = run_board(args.weights, clips, label)
        lm, bm, worst = loo(names, base, results, grid)
        summary.append((label, lm, bm, worst, adm))
        per_board[label] = (names, base, results, grid, set(adm))

    print(f"\n{'='*78}\nVERDICT\n{'='*78}")
    for label, lm, bm, worst, adm in summary:
        print(f"  {label:>30}: LOO {lm:.4f} (base {bm:.4f}, {lm-bm:+.4f}), "
              f"worst {worst:+.4f}, {len(adm)} admissible configs")

    # ---- single fixed config admissible (guardrail + bars) on BOTH full boards ----
    if len(per_board) == 2:
        labels = list(per_board)
        common = per_board[labels[0]][4] & per_board[labels[1]][4]
        print(f"\n  Configs admissible on BOTH full boards (the SHIP candidates): "
              f"{len(common)}")
        ranked = []
        for clabel in (c for c, _ in _grid() if c in common):
            mean_ds = []
            for lab in labels:
                _n, base, results, _g, _a = per_board[lab]
                res = results[clabel]
                mean_ds.append(statistics.mean(res[n][1] for n in _n))
            ranked.append((statistics.mean(mean_ds), clabel, mean_ds))
        ranked.sort(reverse=True)
        for combined, clabel, mean_ds in ranked:
            print(f"    {clabel:>26}  meanD per board: "
                  + ", ".join(f"{lab.split()[0]} {d:+.4f}" for lab, d in zip(labels, mean_ds)))
        if ranked:
            print(f"\n  RECOMMENDED single config: {ranked[0][1]}")
    print("\n  SHIP only if BOTH boards: a single fixed config holds within_r (mean up,")
    print("  worst_clip >= -0.004) AND clears the smoothness bars on every clip. Else")
    print("  report the within_r<->smoothness tradeoff and fall back to render-only.")


if __name__ == "__main__":
    main()
