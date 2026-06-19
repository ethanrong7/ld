"""GATE (read-only) for EXP-Q2b: residual-keyed lock-escape OVERRIDE.

EXP-Q1 (`sheet_residual_probe.py`) established the new fact: the cumulative sheet-frame
residual at a MODERATE horizon (N~=30) ranks the real box #1 on the laggard MISS frames at
0.65-0.68 -- far above the long-standing mass MISS-top1 wall of 0.32-0.41 that CLAUDE.md's
triple-confirmed "identity information limit" said nothing could beat. But EXP-A taught the
hard lesson that strong ISOLATED top-1 != track: feeding `topk` into the trellis as an
additive emission REGRESSED every clip (the causal-key wall). So EXP-Q2b does NOT add an
emission channel. It mirrors `fpath_hyst`'s distance-agnostic HYSTERESIS OVERRIDE
(identity.py:1663-1678) -- a persistence-gated switch -- but keys it on integrated-residual
dominance at N~=30 instead of EMA-coherent-mass.

This probe gates that override BEFORE building it into identity.py, exactly as `hedge_probe`
gated `fpath_hedge`. It simulates the override LAYERED ON the committed `fpath_hyst` track:

  - carry a "captured" box frame-to-frame by affine prediction (the EXP-Q1 association);
  - each frame compute every box's N-residual; the reference box is the captured box (or, if
    not captured, the trellis pick = box nearest the committed output);
  - if some challenger's residual PERSISTENTLY (K consecutive frames) and SUBSTANTIALLY
    ((1+margin)x) exceeds the reference box's residual -> CAPTURE it (switch emission to it);
  - hold the captured box until the trellis pick rejoins it (release) or its chain breaks.

It scores two ways, against the two relevant baselines:
  RAW    : override output vs committed `fpath_hyst` (within_r 0.878) -- does re-picking by
           residual recover lock frames at all?
  HEDGED : (override output -> the shipped churn-gated freeze-blend) vs `fpath_hedge`
           (within_r 0.899) -- the real ship comparison, since the override would sit
           BEFORE the decode hedge in the pipeline.

ACCEPT bar (same as every prior lock-in escape): LOO mean up with NO per-clip regression.
The override targets the PURE-IDENTITY locks the hedge structurally cannot fix (it freezes
the output at the wrong fake's location): t8 both long runs, t5 f521-545, t1 f135-154.

Read-only: committed track + gt + within_r come from data/detect/eval/<stem>__fpath_hyst.csv;
affines from the _hedge_aff_*.pkl cache; boxes from detect_fusion_clip's cache. No video
decode, no mode built here.

    python -m ld.detect.resid_override_probe --weights data/detect/runs/.../best.pt
"""
from __future__ import annotations

import argparse
import math
import statistics

import numpy as np

from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _centroid, _seed
from ld.detect.hedge_probe import _affines, _read_track, _apply, _hit, _churn_hedge
from ld.detect.sheet_residual_probe import _inv, _ap, _nearest

STRONG = {"t2", "t6", "t7", "t9", "t10"}
LAGGARD = {"t1", "t5", "t8"}
HEDGE_CHURN_HI = 8.0   # the shipped fpath_hedge config (hedge_probe LOO winner)
HEDGE_K = 8
REG_BAR = -0.004       # per-clip regression tolerance (matches hedge_probe / exp3_sweep)

# Sweep grid. N is the residual horizon (EXP-Q1 sweet spot ~30); margin = dominance ratio
# the challenger must clear; K = consecutive frames it must clear it (persistence gate).
NS = (30, 45)
MARGINS = (0.15, 0.30, 0.50)
KS = (5, 8, 12)


# ----------------------------------------------------------------- residual core

def _residual_mag(t, i, cents, invaff, radius, N, snap_frac=1.0):
    """Cumulative sheet-frame residual MAGNITUDE of box `i` (frame `t`) at horizon N, or None
    if the affine-prediction chain breaks before reaching N (early frame, or association lost
    -- a box with no residual is ineligible as challenger and not challengeable as reference).
    Same back-walk as sheet_residual_probe._box_residuals, single fixed N."""
    p_ref = np.asarray(cents[t][i], np.float64)   # rigid back-transport of frame-t centroid
    chain = np.asarray(cents[t][i], np.float64)   # actual detected ancestor (snap each step)
    for s in range(1, N + 1):
        f = t - s + 1
        Tinv = invaff.get(f)
        prev = f - 1
        if Tinv is None or prev not in cents or not cents[prev]:
            return None
        p_ref = _ap(Tinv, p_ref)
        pred = _ap(Tinv, chain)
        j, d = _nearest(pred, cents[prev])
        if d >= snap_frac * radius:
            return None
        chain = cents[prev][j]
    return math.hypot(p_ref[0] - chain[0], p_ref[1] - chain[1])


def _resid_cache(rows, cents, invaff, radius, N):
    """{idx: [residual or None per box]} for every scored frame -- the expensive pass,
    depends only on N so it is reused across the (margin, K) sweep."""
    out = {}
    for r in rows:
        t = r["idx"]
        boxes = cents.get(t)
        if not boxes:
            continue
        out[t] = [_residual_mag(t, i, cents, invaff, radius, N) for i in range(len(boxes))]
    return out


# ----------------------------------------------------------------- override sim

def _override_track(rows, cents, aff, resid, radius, margin, K, release=True):
    """Simulate the residual-dominance override layered on the committed track.
    Returns a list of dicts {idx, com=(x,y) overridden output, gt, within_r} -- same shape
    `_churn_hedge`/`_read_track` consume, so the output can be fed straight into the hedge."""
    out = []
    captured = False
    cap_cent = None        # captured box centroid at the previous frame (affine-carried)
    streak = 0
    for r in rows:
        t, gt, com = r["idx"], r["gt"], np.asarray(r["com"], np.float64)
        boxes = cents.get(t)
        rcs = resid.get(t)
        if not boxes or rcs is None:
            out.append(dict(idx=t, com=(float(com[0]), float(com[1])), gt=gt,
                            within_r=int(bool(_hit(com, gt, radius))) if gt else 0))
            captured = False
            continue
        # trellis pick = box nearest the committed output this frame
        ci, _ = _nearest(com, boxes)
        # advance the captured box by affine prediction (drop if it leaves the radius)
        cap_idx = -1
        if captured and cap_cent is not None:
            pred = _apply(aff.get(t), cap_cent)
            j, d = _nearest(pred, boxes)
            if d < radius:
                cap_idx = j
            else:
                captured = False
        ref = cap_idx if (captured and cap_idx >= 0) else ci
        ref_v = rcs[ref] if 0 <= ref < len(rcs) else None
        # best challenger by residual among boxes != ref with a defined chain
        best_j, best_v = -1, -1.0
        for j, v in enumerate(rcs):
            if j == ref or v is None:
                continue
            if v > best_v:
                best_v, best_j = v, j
        # persistence gate: dominance only counts when the reference residual is defined
        if ref_v is not None and best_j >= 0 and best_v > (1.0 + margin) * max(ref_v, 1e-9):
            streak += 1
        else:
            streak = 0
        if streak >= K:
            captured, cap_idx, streak = True, best_j, 0
        # emit
        if captured and cap_idx >= 0:
            ex, ey = boxes[cap_idx]
            cap_cent = np.asarray(boxes[cap_idx], np.float64)
            if release and ci == cap_idx:     # trellis caught up -> hand back
                captured = False
        else:
            ex, ey = float(com[0]), float(com[1])
        out.append(dict(idx=t, com=(float(ex), float(ey)), gt=gt,
                        within_r=int(bool(_hit((ex, ey), gt, radius))) if gt else 0))
    return out


def _within_r(rows):
    scored = [r for r in rows if r["gt"] is not None]
    return statistics.mean(r["within_r"] for r in scored) if scored else 0.0


# --------------------------------------------------------------------------- run

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
        com_wr = _within_r(rows)
        # hedge baseline = the shipped churn-hedge over the UN-overridden committed track
        # (self-consistent fpath_hedge; ~0.899). Treatment = hedge over the overridden track.
        hedge_base = _churn_hedge(rows, aff, radius, HEDGE_CHURN_HI, HEDGE_K)
        rc = {N: _resid_cache(rows, cents, invaff, radius, N) for N in NS}
        data.append(dict(name=name, rows=rows, aff=aff, cents=cents, radius=radius,
                         com_wr=com_wr, hedge_base=hedge_base, rc=rc))
        print(f"  loaded {name}: fpath_hyst wr={com_wr:.3f}  hedge_base wr={hedge_base:.3f}  "
              f"radius={radius:.0f}")

    names = [d["name"] for d in data]

    # ---- sweep: per (N, margin, K) the per-clip RAW and HEDGED within_r deltas ----
    cfgs = [(N, m, K) for N in NS for m in MARGINS for K in KS]
    raw = {c: [] for c in cfgs}     # (name, delta vs fpath_hyst)
    hed = {c: [] for c in cfgs}     # (name, delta vs hedge_base)
    raw_abs = {c: {} for c in cfgs}
    hed_abs = {c: {} for c in cfgs}
    for d in data:
        for (N, m, K) in cfgs:
            ov = _override_track(d["rows"], d["cents"], d["aff"], d["rc"][N],
                                 d["radius"], m, K)
            wr_raw = _within_r(ov)
            wr_hed = _churn_hedge(ov, d["aff"], d["radius"], HEDGE_CHURN_HI, HEDGE_K)
            raw[(N, m, K)].append((d["name"], wr_raw - d["com_wr"]))
            hed[(N, m, K)].append((d["name"], wr_hed - d["hedge_base"]))
            raw_abs[(N, m, K)][d["name"]] = wr_raw
            hed_abs[(N, m, K)][d["name"]] = wr_hed

    # ---- in-sample summary per config (both baselines) ----
    def _summ(table):
        rows_ = []
        for c in cfgs:
            ds = [x for _, x in table[c]]
            worst = min(ds)
            lag = statistics.mean([x for n, x in table[c] if n in LAGGARD])
            strong_worst = min((x for n, x in table[c] if n in STRONG), default=0.0)
            rows_.append((c, statistics.mean(ds), worst, strong_worst, lag))
        return rows_

    for label, table, base_mean in (
            ("RAW vs fpath_hyst (0.878)", raw,
             statistics.mean(d["com_wr"] for d in data)),
            ("HEDGED vs fpath_hedge (hedge_base)", hed,
             statistics.mean(d["hedge_base"] for d in data))):
        print(f"\n=== in-sample sweep -- {label} ===")
        print(f"  base mean within_r = {base_mean:.4f}")
        print(f"  {'N':>3} {'marg':>5} {'K':>3} | {'mean_d':>7} {'worst':>7} "
              f"{'strongW':>8} {'lagMean':>8}")
        for (c, mean_d, worst, sw, lag) in _summ(table):
            flag = "  <- no per-clip regression" if worst >= REG_BAR and mean_d > 0 else ""
            print(f"  {c[0]:>3} {c[1]:>5.2f} {c[2]:>3} | {mean_d:+7.4f} {worst:+7.4f} "
                  f"{sw:+8.4f} {lag:+8.4f}{flag}")

    # ---- per-clip detail for the best no-regression HEDGED config (if any) ----
    def _best_noreg(table):
        best = None
        for c in cfgs:
            ds = [x for _, x in table[c]]
            if min(ds) >= REG_BAR and statistics.mean(ds) > 0:
                if best is None or statistics.mean(ds) > statistics.mean([x for _, x in table[best]]):
                    best = c
        return best

    for label, table, abst, base_key in (
            ("RAW", raw, raw_abs, "com_wr"),
            ("HEDGED", hed, hed_abs, "hedge_base")):
        bc = _best_noreg(table)
        print(f"\n=== best no-regression {label} config: "
              f"{bc if bc else 'NONE (every config regresses some clip)'} ===")
        if bc:
            base = {d["name"]: d[base_key] for d in data}
            print(f"  {'clip':>4} {'base':>6} {'over':>6} {'delta':>7}")
            for n in names:
                tag = " (strong)" if n in STRONG else (" (laggard)" if n in LAGGARD else "")
                print(f"  {n:>4} {base[n]:6.3f} {abst[bc][n]:6.3f} "
                      f"{abst[bc][n]-base[n]:+7.3f}{tag}")

    # ---- honest LOO over the config grid, both baselines (mirror hedge_probe) ----
    for label, table, base_vals in (
            ("RAW (vs fpath_hyst)", raw, {d["name"]: d["com_wr"] for d in data}),
            ("HEDGED (vs fpath_hedge)", hed, {d["name"]: d["hedge_base"] for d in data})):
        abst = {c: {n: base_vals[n] + dx for (n, dx) in table[c]} for c in cfgs}
        print(f"\n=== leave-one-clip-out over (N,margin,K) -- {label} ===")
        loo = []
        for held in names:
            best_cfg, best_key = None, None
            for c in cfgs:
                deltas = [abst[c][n] - base_vals[n] for n in names if n != held]
                worst_d, mean_d = min(deltas), statistics.mean(deltas)
                if worst_d < REG_BAR:
                    continue
                key = (round(worst_d, 4), round(mean_d, 4))
                if best_key is None or key > best_key:
                    best_key, best_cfg = key, c
            if best_cfg is None:
                loo.append(base_vals[held])
                print(f"   {held:>4}: no admissible config -> base {base_vals[held]:.3f}")
            else:
                hv = abst[best_cfg][held]
                loo.append(hv)
                print(f"   {held:>4}: {hv:.3f} (base {base_vals[held]:.3f}, "
                      f"{hv-base_vals[held]:+.3f})  cfg={best_cfg}")
        loo_mean = statistics.mean(loo)
        base_mean = statistics.mean(base_vals.values())
        worst = min(loo[i] - base_vals[names[i]] for i in range(len(names)))
        print(f"\n  LOO mean within_r = {loo_mean:.4f}  (base {base_mean:.4f}, "
              f"{loo_mean-base_mean:+.4f})  worst_clip={worst:+.3f}")
        print("  VERDICT:", "PASS -- LOO up with no per-clip regression; wire the residual"
              " override into identity.py." if loo_mean > base_mean + 1e-4 and worst >= REG_BAR
              else "FAIL -- no LOO gain without a per-clip regression (isolated top-1 did not"
              " survive as a track; record the negative).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()
    clips = _default_clips()
    if args.clips:
        clips = [c for c in clips
                 if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips)


if __name__ == "__main__":
    main()
