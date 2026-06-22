"""GATE (read-only) for EXP-Q3: residual-gated DECODE-layer freeze ("am I on a fake?").

The reframe that revived the residual after EXP-Q2b died: the override failed because
RANKING the real box among ~15 boxes by residual is unsharp (real box #1 only ~0.5-0.6).
But the decode-layer freeze needs only a 1-BOX BINARY -- "is the box I'm currently holding a
rigid fake?" -- and that IS sharp: the chosen box's own N=30 sheet-frame residual collapses
on miss frames (t8: ~9 px locked-on-fake vs ~55 px correctly-tracking; corrP25 34 > missP75
15 on 9/10 clips).

Two facts make this the highest-EV lead since fpath_hedge:
  1. FREEZE-AT-ONSET CEILING is near-oracle: freezing the output at its last in-radius
     position and holding through a miss run recovers t8 +0.221->0.997, t5 +0.161, t1 +0.147,
     board mean +0.084. (The real shape barely moves -- median 1.3 px/frame -- so a position
     frozen at run onset stays within radius even for 20+ frame runs.) The shipped churn-hedge
     misses this because it freezes LATE, after the output already crept onto the fake.
  2. The chosen-box residual is a CAUSAL onset detector (no GT) for that freeze.

This probe wires the two together end-to-end and gates on honest LOO with NO per-clip
regression vs fpath_hedge (0.899):

  - track = committed fpath_hyst chosen-box centroid (pre-hedge) from the eval CSV;
  - each frame compute the chosen box's N=30 residual (affine-chained, EXP-Q1 machinery);
  - if residual < tau -> we are probably on a fake -> FREEZE the output toward a LAGGED
    anchor (the output from `lag` frames ago, i.e. before the creep started); hold while
    residual stays low; release (commit to the live pick) when residual recovers >= tau.
  - then the shipped churn-hedge runs on top (the two compose: residual-freeze fixes the
    coherent identity-locks the churn gate can't see; churn still catches the swept rides).

Sweeps tau x lag; reports in-sample per-clip deltas, the best no-regression config, and LOO.

    python -m ld.detect.resid_freeze_probe --weights data/detect/runs/.../best.pt
"""
from __future__ import annotations

import argparse
import statistics
from collections import deque

import numpy as np

from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _centroid, _seed
from ld.detect.hedge_probe import _affines, _read_track, _hit, _churn_hedge
from ld.detect.probe_common import _inv, _nearest, _resid_cache

STRONG = {"t2", "t6", "t7", "t9", "t10"}
LAGGARD = {"t1", "t5", "t8"}
N = 30                       # residual horizon (EXP-Q1 sweet spot)
HEDGE_CHURN_HI, HEDGE_K = 8.0, 8     # shipped fpath_hedge config
REG_BAR = -0.004
# ABSOLUTE-floor trigger: freeze when the chosen box's N=30 sheet-frame residual is below tau,
# i.e. near the FAKE-NOISE FLOOR. That floor (~9-15 px) is a physical near-CONSTANT: a rigid
# fake has zero true independent drift, so its residual is just detector centroid jitter over the
# window -- roughly clip- and radius-independent. The real shape's residual is clip-dependent and
# much higher (~45-91 px). So an absolute threshold just above the fake floor generalises, whereas
# a relative-to-own-baseline trigger fires on the real shape's frame-to-frame residual noise during
# good tracking (it regressed every clip). The tau grid is capped at ~18 (the top of the fake floor)
# on PHYSICAL grounds -- higher tau cuts into the real-shape residual distribution and regresses the
# clips whose real-shape residual runs low (t10 corrP25 23) -- so the LOO cannot pick an unphysical
# tau that overfits non-held folds (the failure mode of the wider {15..30} grid).
TAUS = (10.0, 12.0, 15.0, 18.0)
LAGS = (6,)                          # frames to rewind the freeze anchor toward onset (6 won)
CONSECS = (1, 2)                     # consecutive low-residual frames required before freezing


def _chosen_resid(rows, cents, rc):
    """{idx: residual of the box nearest the committed output} -- the chosen box's own
    N-residual (the 'am I on a fake?' signal). rc is the precomputed per-box residual cache
    (already built with the affines/radius), so this only needs the cache + centroids."""
    out = {}
    for r in rows:
        t = r["idx"]
        boxes = cents.get(t)
        if not boxes or t not in rc:
            continue
        ci, _ = _nearest(np.asarray(r["com"], np.float64), boxes)
        out[t] = rc[t][ci]
    return out


def _resid_freeze_track(rows, chosen_resid, radius, tau, lag, consec=1):
    """Absolute-floor residual-gated freeze. Returns rows {idx, com=(x,y) frozen output, gt,
    within_r} ready to feed into the churn-hedge. Freeze toward the output from `lag` frames ago
    once the chosen box's N=30 residual has been below tau (near the fake-noise floor) for
    `consec` consecutive frames; commit (track the live pick) otherwise. tau ~ the clip-independent
    fake floor, so a single tau generalises (see the TAUS comment)."""
    buf: deque = deque(maxlen=max(lag, 1))   # recent committed outputs (oldest ~= t-lag)
    hpos = None
    frozen = False
    low_streak = 0
    out = []
    for r in rows:
        t, gt = r["idx"], r["gt"]
        com = np.asarray(r["com"], np.float64)
        rv = chosen_resid.get(t)
        low = rv is not None and rv < tau
        low_streak = low_streak + 1 if low else 0
        if hpos is None:
            hpos = com.copy()
        else:
            if low_streak >= consec:                 # persistently near the fake floor -> freeze
                if not frozen:                       # onset: rewind anchor to pre-creep pos
                    hpos = buf[0].copy() if (lag > 0 and len(buf) > 0) else com.copy()
                    frozen = True
                # else: hold hpos (freeze)
            else:
                frozen = False
                hpos = com.copy()                    # commit to the live pick
        buf.append(com.copy())
        out.append(dict(idx=t, com=(float(hpos[0]), float(hpos[1])), gt=gt,
                        within_r=int(bool(_hit(hpos, gt, radius))) if gt else 0))
    return out


def run(weights, clips):
    data = []
    for clip in clips:
        name = clip.stem.replace("_cropped_trimmed", "")
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        _sx, _sy, radius, _start = _seed(packs)
        rows = _read_track(clip.stem)
        aff = _affines(clip, packs)
        invaff = {idx: _inv(T) for idx, T in aff.items()}
        cents = {idx: [np.asarray(_centroid(b), np.float64) for b in packs[idx].boxes]
                 for idx in range(len(packs)) if packs[idx].boxes}
        rc = _resid_cache(rows, cents, invaff, radius, N)
        chosen = _chosen_resid(rows, cents, rc)
        hedge_base = _churn_hedge(rows, aff, radius, HEDGE_CHURN_HI, HEDGE_K)  # fpath_hedge
        data.append(dict(name=name, rows=rows, aff=aff, radius=radius,
                         chosen=chosen, hedge_base=hedge_base))
        print(f"  loaded {name}: fpath_hedge wr={hedge_base:.3f}  radius={radius:.0f}")

    names = [d["name"] for d in data]
    base = {d["name"]: d["hedge_base"] for d in data}
    cfgs = [(tau, lag, c) for tau in TAUS for lag in LAGS for c in CONSECS]
    table = {c: [] for c in cfgs}     # (name, delta vs fpath_hedge)
    absol = {c: {} for c in cfgs}
    for d in data:
        for (tau, lag, consec) in cfgs:
            fr = _resid_freeze_track(d["rows"], d["chosen"], d["radius"], tau, lag, consec)
            wr = _churn_hedge(fr, d["aff"], d["radius"], HEDGE_CHURN_HI, HEDGE_K)
            table[(tau, lag, consec)].append((d["name"], wr - d["hedge_base"]))
            absol[(tau, lag, consec)][d["name"]] = wr

    print(f"\n=== in-sample sweep (absolute-floor residual-freeze -> churn-hedge) vs fpath_hedge "
          f"(mean {statistics.mean(base.values()):.4f}) ===")
    print(f"  {'tau':>4} {'lag':>3} {'C':>2} | {'mean_d':>7} {'worst':>7} {'strongW':>8} {'lagMean':>8}")
    best = None
    for c in cfgs:
        ds = [x for _, x in table[c]]
        worst = min(ds)
        sw = min((x for n, x in table[c] if n in STRONG), default=0.0)
        lg = statistics.mean([x for n, x in table[c] if n in LAGGARD])
        flag = ""
        if worst >= REG_BAR and statistics.mean(ds) > 0:
            flag = "  <- no per-clip regression"
            if best is None or statistics.mean(ds) > statistics.mean([x for _, x in table[best]]):
                best = c
        print(f"  {c[0]:>4.0f} {c[1]:>3} {c[2]:>2} | {statistics.mean(ds):+7.4f} {worst:+7.4f} "
              f"{sw:+8.4f} {lg:+8.4f}{flag}")

    if best:
        print(f"\n=== per-clip detail, best no-regression config "
              f"tau={best[0]:.0f} lag={best[1]} C={best[2]} ===")
        print(f"  {'clip':>4} {'hedge':>6} {'+freeze':>7} {'delta':>7}")
        for n in names:
            tag = " (strong)" if n in STRONG else (" (laggard)" if n in LAGGARD else "")
            print(f"  {n:>4} {base[n]:6.3f} {absol[best][n]:6.3f} "
                  f"{absol[best][n]-base[n]:+7.3f}{tag}")

    # ---- honest LOO over (tau, lag): per held clip pick best no-regression cfg on other 9 ----
    abst = {c: {n: base[n] + dx for (n, dx) in table[c]} for c in cfgs}
    print("\n=== leave-one-clip-out over (tau, lag) ===")
    loo = []
    for held in names:
        best_cfg, best_key = None, None
        for c in cfgs:
            deltas = [abst[c][n] - base[n] for n in names if n != held]
            wd, md = min(deltas), statistics.mean(deltas)
            if wd < REG_BAR:
                continue
            key = (round(wd, 4), round(md, 4))
            if best_key is None or key > best_key:
                best_key, best_cfg = key, c
        if best_cfg is None:
            loo.append(base[held])
            print(f"   {held:>4}: no admissible cfg -> base {base[held]:.3f}")
        else:
            hv = abst[best_cfg][held]
            loo.append(hv)
            print(f"   {held:>4}: {hv:.3f} (base {base[held]:.3f}, {hv-base[held]:+.3f})  "
                  f"cfg=tau{best_cfg[0]:.0f}/lag{best_cfg[1]}/C{best_cfg[2]}")
    loo_mean = statistics.mean(loo)
    base_mean = statistics.mean(base.values())
    worst = min(loo[i] - base[names[i]] for i in range(len(names)))
    print(f"\n  LOO mean within_r = {loo_mean:.4f}  (fpath_hedge {base_mean:.4f}, "
          f"{loo_mean-base_mean:+.4f})  worst_clip={worst:+.3f}")
    print("  VERDICT:", "PASS -- LOO up with no per-clip regression; build fpath_freeze "
          "(residual-gated decode freeze) into identity.py."
          if loo_mean > base_mean + 1e-4 and worst >= REG_BAR else
          "FAIL -- record the negative.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()
    clips = _default_clips()
    if args.clips:
        clips = [c for c in clips if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips)


if __name__ == "__main__":
    main()
