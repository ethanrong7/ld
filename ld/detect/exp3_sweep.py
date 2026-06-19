"""EXP-3 build+LOO: add an EMA-coherent-mass HYSTERESIS override to the `fpath_fuse`
trellis and tune (alpha, margin, consec) leave-one-clip-out, no per-clip regression.

The EXP-3 gate (exp3_switch_probe) showed a LEAKY-EMA (nearest-box carried) of per-box
coherent-mass sits on GT 0.512 of t1's miss frames -- the only laggard clearing ~0.5,
and only via the EMA (fixed windows stay ~0.3-0.4). This harness tests whether that
target is convertible into a net win WITHOUT regressing the strong lock-in clips
(t2/t6/t7/t9/t10), the wall every prior escape (fpath_reacq, far-jump, trans-cap) hit.

Mechanism (inside the trellis, distance-AGNOSTIC -- the key difference from the dead
far-jump reacquire): maintain a leaky EMA of each box's instantaneous coherent-mass,
carried to the box's current location by nearest-centroid association. If a challenger
box leads the trellis's currently-chosen box by margin M in EMA for >=K consecutive
frames, SWITCH to it and reset path memory (prev_alpha=None), so the transition prior
doesn't immediately drag the path back to the old lock.

Reuses fuse_sweep's one-flow-pass precompute; the override is a pure re-decode, so the
(alpha, margin, consec) grid costs no extra flow passes. Accept only if LOO mean beats
the fpath_fuse baseline with worst-clip delta >= -0.004.

    python -m ld.detect.exp3_sweep --weights .../best.pt
"""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import numpy as np

from ld.detect.eval_modes import _default_clips
from ld.detect.fuse_sweep import _precompute_channels
from ld.detect.identity import (FPATH_TRANS_W, FPATH_MASS_EMA,
                                FPATH_FUSE_CMASS_W, FPATH_FUSE_CURL_W, FPATH_FUSE_WIN)


def _windowed_with_inst(clipdata, win):
    """Like fuse_sweep._windowed but ALSO emits the raw lag-0 instantaneous coherent-mass
    per box (cm[:,0]) for the EMA hysteresis channel. Returns frames of
    (idx, cents, mass_n, cmass_n, curl_n, cm0, gt, active)."""
    cmass_scale = curl_scale = 0.0
    out = []
    for (idx, cents, mass_n, cm, cu, gt, active) in clipdata["frames"]:
        if active and cents is not None:
            cmass_raw = cm[:, :win].sum(axis=1)
            curl_raw = cu[:, :win].sum(axis=1)
            cpk = float(cmass_raw.max())
            cmass_scale = cpk if cmass_scale == 0.0 else max(cpk, FPATH_MASS_EMA * cmass_scale + (1 - FPATH_MASS_EMA) * cpk)
            cmass_n = cmass_raw / cmass_scale if cmass_scale > 0 else cmass_raw
            upk = float(curl_raw.max())
            curl_scale = upk if curl_scale == 0.0 else max(upk, FPATH_MASS_EMA * curl_scale + (1 - FPATH_MASS_EMA) * upk)
            curl_n = curl_raw / curl_scale if curl_scale > 0 else curl_raw
            cm0 = cm[:, 0].astype(np.float32)  # raw single-frame coherent-mass
            out.append((idx, cents, mass_n, cmass_n, curl_n, cm0, gt, True))
        else:
            out.append((idx, None, None, None, None, None, gt, False))
    return out


def _nearest(prev_cents, c):
    bd, bj = 1e18, -1
    for pj, pc in enumerate(prev_cents):
        d = (pc[0] - c[0]) ** 2 + (pc[1] - c[1]) ** 2
        if d < bd:
            bd, bj = d, pj
    return bj


def _decode_hyst(frames, radius, start, cmass_w, curl_w, alpha_ema, margin, consec,
                 trans_w=FPATH_TRANS_W):
    """fpath_fuse trellis + EMA-coherent-mass hysteresis override. consec<=0 => override
    off (== plain fpath_fuse). Returns within_r over scored frames."""
    prev_alpha = prev_cents = None
    ema = ema_cents = None
    streak = 0
    last_xy = None
    hits = tot = 0
    use_hyst = consec > 0 and alpha_ema > 0
    for (idx, cents, mass_n, cmass_n, curl_n, cm0, gt, active) in frames:
        if active and cents is not None:
            emis = mass_n + cmass_w * cmass_n + curl_w * curl_n
            if prev_alpha is None:
                a = emis.copy()
            else:
                a = np.empty(len(cents), np.float32)
                for i, (cx, cy) in enumerate(cents):
                    best = -1e18
                    for j, (px, py) in enumerate(prev_cents):
                        dd = math.hypot(cx - px, cy - py) / radius
                        v = prev_alpha[j] - trans_w * dd * dd
                        if v > best:
                            best = v
                    a[i] = best + emis[i]
            a = a - float(a.max())
            choice = int(np.argmax(a))
            prev_alpha, prev_cents = a, cents
            # EMA hysteresis override (distance-agnostic)
            if use_hyst:
                e = cm0.copy()
                if ema is not None:
                    for i, c in enumerate(cents):
                        j = _nearest(ema_cents, c)
                        if j >= 0:
                            e[i] = alpha_ema * ema[j] + (1 - alpha_ema) * cm0[i]
                ema, ema_cents = e, cents
                chal = int(np.argmax(e))
                if chal != choice and e[chal] > (1 + margin) * max(float(e[choice]), 1e-9):
                    streak += 1
                else:
                    streak = 0
                if streak >= consec:
                    choice = chal
                    prev_alpha = prev_cents = None
                    streak = 0
            last_xy = cents[choice]
        else:
            prev_alpha = prev_cents = None
        if idx >= start and gt is not None and last_xy is not None:
            tot += 1
            if math.hypot(last_xy[0] - gt[0], last_xy[1] - gt[1]) < radius:
                hits += 1
    return hits / tot if tot else 0.0


def run(weights, clips, alphas, margins, consecs):
    win, cmw, cuw = FPATH_FUSE_WIN, FPATH_FUSE_CMASS_W, FPATH_FUSE_CURL_W
    data = [_precompute_channels(weights, c, win) for c in clips]
    names = [d["name"] for d in data]
    wf = {d["name"]: _windowed_with_inst(d, win) for d in data}
    rad = {d["name"]: d["radius"] for d in data}
    st = {d["name"]: d["start"] for d in data}

    def dec(name, al, mg, cs):
        return _decode_hyst(wf[name], rad[name], st[name], cmw, cuw, al, mg, cs)

    base = {n: dec(n, 0.0, 0.0, 0) for n in names}  # override off == fpath_fuse
    base_mean = statistics.mean(base.values())
    print(f"fpath_fuse baseline (win{win} cmass{cmw} curl{cuw}) per clip:")
    for n in names:
        print(f"   {n:>4}: {base[n]:.3f}")
    print(f"   MEAN: {base_mean:.4f}\n")

    cfgs = [(a, m, c) for a in alphas for m in margins for c in consecs]
    scored = {}
    print(f"{'alpha':>5} {'marg':>5} {'cons':>4} | "
          + " ".join(f"{n:>5}" for n in names) + " |   MEAN  worst_d")
    for cfg in cfgs:
        wr = {n: dec(n, *cfg) for n in names}
        scored[cfg] = wr
        mean = statistics.mean(wr.values())
        worst_d = min(wr[n] - base[n] for n in names)
        print(f"{cfg[0]:5.2f} {cfg[1]:5.2f} {cfg[2]:4d} | "
              + " ".join(f"{wr[n]:5.3f}" for n in names)
              + f" | {mean:6.4f}  {worst_d:+.3f}")

    print("\n=== leave-one-clip-out (no per-clip regression on train folds) ===")
    loo = []
    for hi, held in enumerate(names):
        best_cfg, best_key = None, None
        for cfg in cfgs:
            wr = scored[cfg]
            deltas = [wr[n] - base[n] for k, n in enumerate(names) if k != hi]
            worst_d = min(deltas)
            mean_d = statistics.mean(deltas)
            if worst_d < -0.004:
                continue
            key = (round(worst_d, 4), round(mean_d, 4), -cfg[2], -cfg[1])
            if best_key is None or key > best_key:
                best_key, best_cfg = key, cfg
        if best_cfg is None:
            loo.append(base[held])
            print(f"   {held:>4}: no admissible cfg -> base {base[held]:.3f}")
        else:
            hv = scored[best_cfg][held]
            loo.append(hv)
            print(f"   {held:>4}: {hv:.3f} (base {base[held]:.3f}, {hv-base[held]:+.3f})"
                  f"  cfg=a{best_cfg[0]} m{best_cfg[1]} c{best_cfg[2]}")
    loo_mean = statistics.mean(loo)
    worst = min(loo[i] - base[names[i]] for i in range(len(names)))
    print(f"\nLOO mean within_r = {loo_mean:.4f}  (base {base_mean:.4f}, {loo_mean-base_mean:+.4f})"
          f"  worst_clip={worst:+.3f}")
    adm = [c for c in cfgs if min(scored[c][n] - base[n] for n in names) >= -0.004]
    if adm:
        bc = max(adm, key=lambda c: statistics.mean(scored[c].values()))
        print(f"best NO-REGRESSION cfg = a{bc[0]} m{bc[1]} c{bc[2]} "
              f"mean={statistics.mean(scored[bc].values()):.4f}")
    else:
        print("NO admissible (no-per-clip-regression) cfg exists -> EXP-3 dead.")
    bc2 = max(cfgs, key=lambda c: statistics.mean(scored[c].values()))
    print(f"best in-sample cfg      = a{bc2[0]} m{bc2[1]} c{bc2[2]} "
          f"mean={statistics.mean(scored[bc2].values()):.4f}  "
          f"(worst_d {min(scored[bc2][n]-base[n] for n in names):+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.85, 0.90, 0.95])
    ap.add_argument("--margins", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    ap.add_argument("--consecs", nargs="+", type=int, default=[5, 8, 12])
    args = ap.parse_args()
    clips = _default_clips()
    if args.clips:
        clips = [c for c in clips
                 if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.alphas, args.margins, args.consecs)


if __name__ == "__main__":
    main()
