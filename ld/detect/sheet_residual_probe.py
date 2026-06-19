"""GATE (read-only) for the CUMULATIVE SHEET-FRAME RESIDUAL identity channel
(plan.md EXP-Q1 / "long-horizon drift integrator").

The bet: every identity attempt measured the real shape's independent translation
PER-FRAME or over 8-10-frame windows -- exactly where ~1.3 px/frame sits UNDER the
affine/detector noise floor. So the "information limit" is really an SNR limit, and the
standard fix is to INTEGRATE: transport every box's centroid into the sheet's reference
frame via the cumulative inverse RANSAC affine and accumulate its divergence from rigid
prediction over a long causal horizon (N in {15,30,45,60,90}). A fake's residual is
identically ~0 at all N (it moves rigidly with the sheet by construction); the real
shape's DIRECTIONAL drift grows ~N*v while zero-mean noise grows only ~sqrt(N)*sigma.

For one box i present at frame t, over horizon N:
  1. Build a backward correspondence chain by AFFINE PREDICTION (not raw proximity --
     that is the dead EMA-association failure): predict the box's position at k-1 by
     T_k^{-1}, snap to the nearest actual centroid at k-1, DROP the chain if the snap
     exceeds the radius. Yields actual ancestor centroids p_t .. p_{t-N}.
  2. Transport p_t rigidly back to frame t-N via the composed inverse affine -> p_ref.
     The cumulative residual is |p_ref - p_{t-N}| : how far the box ended up from where
     pure sheet motion predicts its frame-t self should have come from. Fake ~0 (+noise);
     real shape ~ accumulated independent drift.

The GATE (the make-or-break, answered on the laggard MISS frames t8/t5/t1):
  - does the oracle box's cumulative residual rank it #1 MATERIALLY more often than mass
    (CLAUDE.md failure-taxonomy MISS-top1 0.32-0.41) / curl / coherence do, AND
  - does the separation ratio resid(oracle)/p90(resid over fakes) RISE as N goes 15->90?
A rising curve + oracle box rising to #1 -> proceed to EXP-Q2 (wire fpath_resid). A flat
or falling curve, or the oracle box still not #1 -> the translation signal is sub-noise
even integrated -> the information limit is real (that itself proves ~0.92-0.93 is this
stack's ceiling). t7 is the SANITY clip: if the oracle residual does not out-rank fakes
even there, the readout is broken (cf. EXP-R1/EXP-S1 t7-sanity failures).

Read-only: committed track + gt + within_r + oracle_hit come straight from the existing
data/detect/eval/<stem>__fpath_hyst.csv; the per-frame sheet affines load from the
already-built _hedge_aff_*.pkl cache; boxes from detect_fusion_clip's cache. No video
decode, no pixel pass, no re-detection, no mode built here.

    python -m ld.detect.sheet_residual_probe --weights data/detect/runs/.../best.pt
"""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _centroid, _seed
from ld.detect.hedge_probe import _affines, _read_track

NS = (15, 30, 45, 60, 90)          # horizon sweep; the SNR-vs-N curve IS the gate
MAX_N = max(NS)
LAGGARD = ("t8", "t5", "t1")       # the priority MISS-frame targets (ordered worst-first)
STRONG = ("t2", "t6", "t7", "t9", "t10")
SNAP_FRAC = 1.0                    # break a chain when the affine-predicted snap exceeds
                                   # SNAP_FRAC * radius (drop, don't mis-associate)
MASS_REF = "mass MISS-top1 0.32-0.41 (CLAUDE.md taxonomy)"   # the bar residual must beat


# ----------------------------------------------------------------- geometry helpers

def _inv(T):
    """2x3 inverse affine (cur->prev) of the cached prev->cur affine, or None."""
    if T is None:
        return None
    return cv2.invertAffineTransform(np.asarray(T, np.float64))


def _ap(T, p):
    """Map point p by 2x3 affine T (identity if None)."""
    if T is None:
        return np.asarray(p, np.float64)
    return T[:, :2] @ np.asarray(p, np.float64) + T[:, 2]


def _nearest(p, cloud):
    """(index, distance) of the nearest centroid in `cloud` (list of np arrays) to p."""
    best_i, best_d = -1, float("inf")
    for j, c in enumerate(cloud):
        d = math.hypot(c[0] - p[0], c[1] - p[1])
        if d < best_d:
            best_i, best_d = j, d
    return best_i, best_d


def _pctl(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, int(q * len(s)))
    return s[i]


# ----------------------------------------------------------------- residual core

def _box_residuals(t, i, cents, invaff, radius):
    """Cumulative sheet-frame residual VECTOR of box `i` (at frame `t`) at every horizon
    in NS, all expressed in the box's own t-N reference frame.

    Returns {N: (dx, dy)} for the horizons the chain survives to (a broken chain -- snap
    exceeds the radius, or a missing affine/centroid frame -- simply omits that N and every
    longer one). p_ref is the PURE rigid back-transport of the frame-t centroid (no snap);
    the chain endpoint is the actual detected ancestor; their gap is the drift vector. The
    caller reads its magnitude (raw EXP-Q1) and/or its common-mode-rejected magnitude
    (EXP-Q1b: subtract the per-frame across-box median residual vector -- the shared affine
    bias the fakes all carry -- so the fake cloud stops growing ~N).
    """
    p_t = cents[t][i]
    p_ref = np.asarray(p_t, np.float64)      # rigid back-transport (composition only)
    chain = np.asarray(p_t, np.float64)      # actual detected ancestor (snap each step)
    ns = set(NS)
    out = {}
    for s in range(1, MAX_N + 1):
        f = t - s + 1                        # invert T_f (maps f-1 -> f) to step f -> f-1
        Tinv = invaff.get(f)
        prev = f - 1
        if Tinv is None or prev not in cents or not cents[prev]:
            break
        p_ref = _ap(Tinv, p_ref)             # where rigid motion sends frame-t self at f-1
        pred = _ap(Tinv, chain)              # rigid prediction of this box's ancestor
        j, d = _nearest(pred, cents[prev])
        if d >= SNAP_FRAC * radius:          # chain broke -> drop (no mis-association)
            break
        chain = cents[prev][j]               # snap to the actual ancestor centroid
        if s in ns:
            out[s] = (p_ref[0] - chain[0], p_ref[1] - chain[1])   # residual VECTOR
    return out


def _oracle_index(t, cents, gt, radius):
    """Index of the box whose centroid is nearest GT and within radius (the oracle box),
    or -1 if none (no oracle hit this frame)."""
    if gt is None or t not in cents or not cents[t]:
        return -1
    j, d = _nearest(np.asarray(gt, np.float64), cents[t])
    return j if d < radius else -1


# --------------------------------------------------------------------------- run

def _scan_clip(name, rows, cents, invaff, radius, miss_only, debias=False):
    """Per-N stats over the scored frames of one clip.

    `miss_only` restricts to fpath_hyst MISS frames (within_r==0 & oracle_hit==1) -- the
    laggard target set; False = all oracle-hit frames (the t7-style sanity set).
    `debias` toggles EXP-Q1b common-mode rejection: subtract the per-frame across-box median
    residual VECTOR (the shared affine bias the rigid fakes carry) before taking magnitudes.
    Returns {N: dict(n, cover, top1, top3, sep, orc_mean, fake_p90)}."""
    by_idx = {r["idx"]: r for r in rows}
    acc = {N: dict(orc=[], fake_p90=[], fakes=[], rank1=0, rank3=0, n=0, frames=0)
           for N in NS}
    for r in rows:
        t, gt = r["idx"], r["gt"]
        if gt is None or r["oh"] != 1:
            continue
        if miss_only and r["within_r"] != 0:
            continue
        oi = _oracle_index(t, cents, gt, radius)
        if oi < 0 or t not in cents or len(cents[t]) < 2:
            continue
        # residual VECTOR of every box at frame t, at every surviving horizon
        per_box = {i: _box_residuals(t, i, cents, invaff, radius)
                   for i in range(len(cents[t]))}
        for N in NS:
            vecs = {i: per_box[i][N] for i in per_box if N in per_box[i]}
            if oi not in vecs or len(vecs) < 2:
                continue
            if debias:
                # common-mode = robust across-box centre of the residual vectors (the
                # shared affine bias); subtract it so rigid fakes collapse toward 0.
                mx = statistics.median([v[0] for v in vecs.values()])
                my = statistics.median([v[1] for v in vecs.values()])
                resids = {i: math.hypot(v[0] - mx, v[1] - my) for i, v in vecs.items()}
            else:
                resids = {i: math.hypot(v[0], v[1]) for i, v in vecs.items()}
            a = acc[N]
            a["frames"] += 1
            orc = resids[oi]
            fakes = [resids[i] for i in resids if i != oi]
            # rank of oracle box by DESCENDING residual (1 = largest drift = best)
            rank = 1 + sum(1 for v in fakes if v > orc)
            a["n"] += 1
            a["rank1"] += int(rank == 1)
            a["rank3"] += int(rank <= 3)
            a["orc"].append(orc)
            a["fakes"].extend(fakes)
            p90 = _pctl(fakes, 0.9)
            if p90 > 1e-6:
                a["fake_p90"].append(orc / p90)
    res = {}
    for N in NS:
        a = acc[N]
        if a["n"] == 0:
            res[N] = dict(n=0, top1=float("nan"), top3=float("nan"), sep=float("nan"),
                          orc_mean=float("nan"), fake_p90=float("nan"),
                          fake_p99=float("nan"))
            continue
        res[N] = dict(
            n=a["n"],
            top1=a["rank1"] / a["n"],
            top3=a["rank3"] / a["n"],
            sep=statistics.mean(a["fake_p90"]) if a["fake_p90"] else float("nan"),
            orc_mean=statistics.mean(a["orc"]),
            fake_p90=_pctl(a["fakes"], 0.9),
            fake_p99=_pctl(a["fakes"], 0.99),
        )
    return res


def _print_block(title, per_clip, order):
    print(f"\n=== {title} ===")
    print(f"  {'clip':>4} {'N':>3} {'n':>4} | {'orcResid':>8} {'fakeP90':>7} {'fakeP99':>7} "
          f"| {'sepRatio':>8} {'top1':>5} {'top3':>5}")
    for name in order:
        res = per_clip.get(name)
        if res is None:
            continue
        for N in NS:
            d = res[N]
            print(f"  {name:>4} {N:>3} {d['n']:>4} | {d['orc_mean']:8.2f} {d['fake_p90']:7.2f} "
                  f"{d['fake_p99']:7.2f} | {d['sep']:8.2f} {d['top1']:5.2f} {d['top3']:5.2f}")
        print(f"  {'':>4} {'':>3} {'':>4} | {'-'*8} {'-'*7} {'-'*7} | {'-'*8} {'-'*5} {'-'*5}")


def _rising(vals):
    """Crude monotonicity readout over the N sweep (ignores NaNs)."""
    xs = [v for v in vals if not math.isnan(v)]
    if len(xs) < 2:
        return "n/a"
    ups = sum(1 for a, b in zip(xs, xs[1:]) if b > a + 1e-9)
    downs = sum(1 for a, b in zip(xs, xs[1:]) if b < a - 1e-9)
    net = xs[-1] - xs[0]
    arrow = "RISES" if net > 1e-6 else ("FALLS" if net < -1e-6 else "flat")
    return f"{arrow} ({xs[0]:.2f}->{xs[-1]:.2f}, +{ups}/-{downs})"


def run(weights, clips):
    miss, sanity, miss_db = {}, {}, {}
    order = []
    for clip in clips:
        name = clip.stem.replace("_cropped_trimmed", "")
        order.append(name)
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        _sx, _sy, radius, _start = _seed(packs)
        rows = _read_track(clip.stem)
        aff = _affines(clip, packs)                          # cached prev->cur affines
        invaff = {idx: _inv(T) for idx, T in aff.items()}    # cur->prev for the back-walk
        cents = {idx: [np.asarray(_centroid(b), np.float64) for b in packs[idx].boxes]
                 for idx in range(len(packs)) if packs[idx].boxes}
        miss[name] = _scan_clip(name, rows, cents, invaff, radius, miss_only=True)
        miss_db[name] = _scan_clip(name, rows, cents, invaff, radius, miss_only=True,
                                   debias=True)
        sanity[name] = _scan_clip(name, rows, cents, invaff, radius, miss_only=False)
        n_miss = miss[name][NS[0]]["n"]
        print(f"  loaded {name}: radius={radius:.0f}  miss-frames(scored,N={NS[0]})={n_miss}")

    # ---- t7-style SANITY (all oracle-hit frames): readout must work on a clean clip ----
    _print_block("SANITY -- all oracle-hit frames (t7 is the key clean clip)",
                 sanity, order)
    print("  SANITY pass: on t7 the oracle residual out-ranks fakes (top1 well above"
          " chance, sep>1). If not, the readout itself is broken -> stop.")

    # ---- the GATE: laggard MISS frames (within_r==0 & oracle_hit==1) ----
    lag = [n for n in LAGGARD if n in miss]
    _print_block("GATE -- laggard MISS frames (within_r==0 & oracle_hit==1)",
                 miss, lag)

    # ---- the make-or-break SNR-vs-N curves on the laggard MISS frames ----
    print("\n=== SNR-vs-N (the make-or-break) -- laggard MISS frames ===")
    print(f"  channel bar to beat: {MASS_REF}")
    for name in lag:
        res = miss[name]
        print(f"  {name:>4}  sepRatio {_rising([res[N]['sep'] for N in NS])}")
        print(f"  {name:>4}  top1     {_rising([res[N]['top1'] for N in NS])}")
    # pooled laggard MISS aggregate (frame-weighted) per N
    print("\n  pooled laggard MISS aggregate:")
    print(f"  {'N':>3} {'n':>5} | {'sepRatio':>8} {'top1':>5} {'top3':>5}  (top1 bar ~0.41)")
    pooled_sep, pooled_top1 = [], []
    for N in NS:
        ns_tot = sum(miss[n][N]["n"] for n in lag)
        if ns_tot == 0:
            continue
        sep = statistics.mean([miss[n][N]["sep"] for n in lag
                               if not math.isnan(miss[n][N]["sep"])])
        # frame-weighted top1/top3 across the laggards
        t1 = sum(miss[n][N]["top1"] * miss[n][N]["n"] for n in lag) / ns_tot
        t3 = sum(miss[n][N]["top3"] * miss[n][N]["n"] for n in lag) / ns_tot
        pooled_sep.append(sep)
        pooled_top1.append(t1)
        print(f"  {N:>3} {ns_tot:>5} | {sep:8.2f} {t1:5.2f} {t3:5.2f}")
    print(f"\n  pooled sepRatio {_rising(pooled_sep)}")
    print(f"  pooled top1     {_rising(pooled_top1)}")

    # ---- EXP-Q1b: common-mode-rejected residual (does debiasing make SNR RISE with N?) ----
    _print_block("EXP-Q1b GATE -- common-mode-rejected residual, laggard MISS frames",
                 miss_db, lag)
    print("\n=== EXP-Q1b SNR-vs-N (debiased) -- laggard MISS frames ===")
    for name in lag:
        res = miss_db[name]
        print(f"  {name:>4}  sepRatio {_rising([res[N]['sep'] for N in NS])}")
        print(f"  {name:>4}  top1     {_rising([res[N]['top1'] for N in NS])}")
    print("\n  pooled debiased laggard MISS aggregate:")
    print(f"  {'N':>3} {'n':>5} | {'sepRatio':>8} {'top1':>5} {'top3':>5}  (top1 bar ~0.41)")
    db_sep, db_top1 = [], []
    for N in NS:
        ns_tot = sum(miss_db[n][N]["n"] for n in lag)
        if ns_tot == 0:
            continue
        sep = statistics.mean([miss_db[n][N]["sep"] for n in lag
                               if not math.isnan(miss_db[n][N]["sep"])])
        t1 = sum(miss_db[n][N]["top1"] * miss_db[n][N]["n"] for n in lag) / ns_tot
        t3 = sum(miss_db[n][N]["top3"] * miss_db[n][N]["n"] for n in lag) / ns_tot
        db_sep.append(sep)
        db_top1.append(t1)
        print(f"  {N:>3} {ns_tot:>5} | {sep:8.2f} {t1:5.2f} {t3:5.2f}")
    print(f"\n  pooled debiased sepRatio {_rising(db_sep)}")
    print(f"  pooled debiased top1     {_rising(db_top1)}")
    print("  EXP-Q1b PASS if debiasing makes sepRatio RISE with N where raw EXP-Q1 fell -> the")
    print("  fake cloud's ~N growth was the shared affine bias, and CMR converts it to a clean win.")

    # ---- verdict ----
    sep_rise = (len([v for v in pooled_sep if not math.isnan(v)]) >= 2
                and pooled_sep[-1] > pooled_sep[0] + 1e-6)
    top1_beats = any(v >= 0.41 for v in pooled_top1)
    print("\n  VERDICT:",
          "GATE PASSES -- separation ratio RISES with N and the oracle box reaches #1 "
          "more than mass (>=0.41) on the laggard MISS frames. Proceed to EXP-Q2 "
          "(wire fpath_resid as an additive emission channel)."
          if (sep_rise and top1_beats) else
          "GATE FAILS -- the cumulative residual does not separate the real shape on the "
          "laggard MISS frames (flat/falling SNR or oracle box still not #1 above mass). "
          "Integrated translation is still sub-noise -> the identity information-limit is "
          "real; record the negative (this establishes ~0.92-0.93 as the stack ceiling). "
          f"[sep_rises={sep_rise}, top1_beats_mass={top1_beats}]")


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
