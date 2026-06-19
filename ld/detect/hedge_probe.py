"""GATE (read-only) for `fpath_hedge` — confidence-weighted sheet-motion damping at
the DECODE layer (plan.md "HEDGE UNDER UNCERTAINTY").

The reframe: `within_r` rewards landing within ~radius of GT, NOT naming the correct
box. The catastrophic misses are SWEPT-LOCKS — the track creeps onto an adjacent fake
(~12px) then rides that fake as the sheet translates it 200-400px away. The real shape
moves independently of the rigid sheet (median 1.3 px/frame screen speed), so it stays
roughly put while a fake-locked track inherits the sheet's sweep. The hedge drops the
SHEET component of the chosen box's motion when we don't trust the pick:

    d_total = c_t - p_{t-1}                 # committed-box displacement
    d_sheet = T_t(p_{t-1}) - p_{t-1}        # how the sheet carries our last position
    d_indep = d_total - d_sheet             # the independent (real-shape-like) part
    p_t     = p_{t-1} + d_indep + w*d_sheet # w=1 -> commit (today); w=0 -> drop sweep

This probe gates the idea BEFORE building the mode. It must establish three things:

  1. FOUNDATIONAL: over the swept-lock runs, GT screen displacement << committed-track
     screen displacement (the real shape really did stay put while the track was swept).
     If GT moves nearly as fast as the track, the hedge has no purchase -> dead.
  2. RECOVERY: on swept-lock miss frames, do `freeze` / `coast` (w=0) / `cluster`
     (coast + nearby-YOLO-centroid pull) land within radius far more than `committed`?
  3. SAFETY: a self-validating trust-gated hedge (w from the chosen box's recent
     independent-motion magnitude -- plan's alternative trust signal, fully derivable
     from the committed track + affines) must NOT lower within_r on the strong clips
     (t2/t6/t7/t9/t10), where w~1 because the track is on the independently-moving shape.

Read-only: committed track + gt + within_r come straight from the existing
data/detect/eval/<stem>__fpath_hyst.csv; the only new compute is a per-frame sheet
affine flow pass (cached to disk). No mode is built here.

    python -m ld.detect.hedge_probe --weights data/detect/runs/.../best.pt
"""
from __future__ import annotations

import argparse
import csv
import math
import pickle
import statistics
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ld.config import DATA_DIR
from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _centroid, _seed
from ld.vision.cursor import strip_pointer
from ld.vision.motion import estimate_motion
from ld.capture.video_source import VideoSource

MODE = "fpath_hyst"
MIN_RUN = 3                # min consecutive miss frames to count as a sustained run
GROW_FACTOR = 1.8          # last-third err must exceed first-third by this -> "swept"
GROW_FLOOR = 150.0         # ...and reach at least this many px -> a real ride, not jitter
STRONG = {"t2", "t6", "t7", "t9", "t10"}
LAGGARD = {"t1", "t5", "t8"}
CLUSTER_R = 1.0            # cluster strategy: pull toward YOLO boxes within this*radius
CLUSTER_BETA = 0.5         # blend weight toward the local cluster centroid


# --------------------------------------------------------------------------- io

def _read_track(stem: str):
    """Per-frame committed fpath_hyst output straight from the eval CSV.
    Returns ordered list of dicts with idx, com=(x,y), gt=(x,y)|None, within_r, oh, err."""
    path = DATA_DIR / "detect" / "eval" / f"{stem}__{MODE}.csv"
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            gt = None
            if r["gt_x"] != "" and r["gt_y"] != "":
                gt = (float(r["gt_x"]), float(r["gt_y"]))
            rows.append(dict(
                idx=int(r["idx"]),
                com=(float(r["x"]), float(r["y"])),
                gt=gt,
                within_r=int(float(r["within_r"])) if r["within_r"] != "" else 0,
                oh=int(float(r["oracle_hit"])) if r["oracle_hit"] != "" else 0,
                err=float(r["err_px"]) if r["err_px"] != "" else float("nan"),
            ))
    rows.sort(key=lambda d: d["idx"])
    return rows


def _affines(clip: Path, packs):
    """Per-frame global sheet affine (prev->cur), keyed by current idx. Cached to disk —
    the only new compute the gate needs. estimate_motion(prev,cur).affine is exactly the
    T_t the hedge applies to p_{t-1}."""
    cache = DATA_DIR / "detect" / "cache" / f"_hedge_aff_{clip.stem}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    src = VideoSource(clip)
    grays = {}
    for idx, frame in src.frames():
        grays[idx] = cv2.cvtColor(strip_pointer(frame, strip_green=True), cv2.COLOR_BGR2GRAY)
    src.release()
    idxs = sorted(grays)
    aff = {}
    for pos, idx in enumerate(idxs):
        if pos == 0 or idx >= len(packs):
            continue
        aff[idx] = estimate_motion(grays[idxs[pos - 1]], grays[idx]).affine
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(aff, f)
    return aff


# ----------------------------------------------------------------- geometry helpers

def _apply(T, p):
    """Map point p by 2x3 affine T. Identity (no sheet motion) if T is None."""
    if T is None:
        return np.asarray(p, np.float64)
    p = np.asarray(p, np.float64)
    return T[:, :2] @ p + T[:, 2]


def _hit(p, gt, radius):
    return gt is not None and math.hypot(p[0] - gt[0], p[1] - gt[1]) < radius


# ----------------------------------------------------------------- swept-lock runs

def _swept_runs(rows):
    """Yield maximal within_r==0 runs (>=MIN_RUN) whose err grows into a real ride.
    Mirrors step0_failchar's 'C coast-runaway' / growing-'A' classifier."""
    run = []
    for r in rows:
        if r["within_r"] == 0:
            run.append(r)
        else:
            if len(run) >= MIN_RUN:
                yield run
            run = []
    if len(run) >= MIN_RUN:
        yield run


def _grows(run):
    n = len(run)
    err = [r["err"] for r in run if not math.isnan(r["err"])]
    if len(err) < 3:
        return False
    k = max(1, len(err) // 3)
    first = sum(err[:k]) / k
    last = sum(err[-k:]) / k
    return last > first * GROW_FACTOR and last > GROW_FLOOR


# ----------------------------------------------------------------- strategies

def _strategy_hits(run, p0, c_prev0, aff, cents_by_idx, radius):
    """within_r hit counts over a single swept-lock run for each strategy.
    All start from p0 = last in-radius committed position before the run; c_prev0 is the
    chosen-box centroid at that same pre-run frame.

    Correct sheet-damped coast (w=0): p_t = p_{t-1} + (c_t - T_t(c_{t-1})). The term is the
    chosen BOX's OWN independent motion (box now minus sheet-carried box-prev), ~0 for a
    fake riding the sheet -> coast freezes a fake but still follows a genuinely independent
    shape. (The plan's c_t - T(p_{t-1}) form is wrong: it tracks a swept fake one frame
    late instead of freezing it -- this gate caught that.)"""
    h = dict(committed=0, freeze=0, coast=0, cluster=0)
    p_coast = np.asarray(p0, np.float64)
    p_clu = np.asarray(p0, np.float64)
    p0 = np.asarray(p0, np.float64)
    c_prev = np.asarray(c_prev0, np.float64)
    indep_mags = []
    for r in run:
        idx, gt, c = r["idx"], r["gt"], np.asarray(r["com"], np.float64)
        T = aff.get(idx)
        d_indep = c - _apply(T, c_prev)            # chosen box's own independent motion
        indep_mags.append(float(math.hypot(d_indep[0], d_indep[1])))
        # committed (today's output)
        h["committed"] += r["within_r"]
        # freeze: hold the pre-run position
        h["freeze"] += _hit(p0, gt, radius)
        # coast: add only the box's independent motion (sheet sweep dropped)
        p_coast = p_coast + d_indep
        h["coast"] += _hit(p_coast, gt, radius)
        # cluster: same coast, then pull toward the centroid of nearby YOLO boxes
        p_clu = p_clu + d_indep
        near = [np.asarray(cc, np.float64) for cc in cents_by_idx.get(idx, [])
                if math.hypot(cc[0] - p_clu[0], cc[1] - p_clu[1]) < CLUSTER_R * radius]
        if near:
            cc = np.mean(near, axis=0)
            p_clu = (1 - CLUSTER_BETA) * p_clu + CLUSTER_BETA * cc
        h["cluster"] += _hit(p_clu, gt, radius)
        c_prev = c
    return h, (statistics.mean(indep_mags) if indep_mags else 0.0)


# ------------------------------------------------- trust-gated end-to-end hedge

def _trust_hedge(rows, aff, radius, trust_hi, k):
    """Self-validating trust-gated hedge over a WHOLE clip (the realistic shippable form,
    used for the strong-clip safety check). Trust w rises with the chosen box's recent
    independent-motion magnitude: a box moving only with the sheet (a fake) -> |d_indep|~0
    -> w->0 -> the estimate freezes near the last real position; a box with genuine
    independent motion -> w->1 -> ordinary commit. Causal (window is past frames only)."""
    win = deque(maxlen=k)
    p = None
    prevc = None
    hits = tot = 0
    for r in rows:
        idx, gt, c = r["idx"], r["gt"], np.asarray(r["com"], np.float64)
        T = aff.get(idx)
        if p is None or prevc is None or T is None:
            p = c.copy()
        else:
            # chosen box's own independent motion (~0 for a swept fake)
            d_indep = c - _apply(T, prevc)
            d_sheet_self = _apply(T, p) - p          # sheet carry of OUR position
            w = min(max(statistics.mean(win) / trust_hi, 0.0), 1.0) if win else 1.0
            p = p + d_indep + w * d_sheet_self
            win.append(float(math.hypot(d_indep[0], d_indep[1])))
        prevc = c
        if gt is not None:
            tot += 1
            hits += _hit(p, gt, radius)
    return hits / tot if tot else 0.0


def _freeze_hedge(rows, aff, radius, trust_hi, k):
    """Freeze-blend variant: instead of sheet-decomposition (which integrates the chosen
    box's jumpy per-frame motion), blend the committed pick toward HOLDING the last output
    as trust falls: p_t = w*c_t + (1-w)*p_{t-1}. w=1 commit, w=0 freeze. Trust w rises with
    the chosen box's recent independent-motion magnitude. This is the primitive the recovery
    test showed actually works (freeze recovered 100% of swept frames)."""
    win = deque(maxlen=k)
    p = None
    prevc = None
    hits = tot = 0
    for r in rows:
        idx, gt, c = r["idx"], r["gt"], np.asarray(r["com"], np.float64)
        T = aff.get(idx)
        if p is None or prevc is None or T is None:
            p = c.copy()
        else:
            d_indep = c - _apply(T, prevc)
            win.append(float(math.hypot(d_indep[0], d_indep[1])))
            w = min(max(statistics.mean(win) / trust_hi, 0.0), 1.0)
            p = w * c + (1 - w) * p
        prevc = c
        if gt is not None:
            tot += 1
            hits += _hit(p, gt, radius)
    return hits / tot if tot else 0.0


def _indep_mag(rows, aff):
    """Mean magnitude of the chosen box's per-frame independent motion |c_t - T(c_{t-1})| --
    the trust signal. For the hedge to work this must be markedly LOWER on swept-lock fakes
    than on the real shape (strong clips). If not, no causal trust gate can separate them."""
    prevc = None
    mags = []
    for r in rows:
        c, T = np.asarray(r["com"], np.float64), aff.get(r["idx"])
        if prevc is not None and T is not None:
            d = c - _apply(T, prevc)
            mags.append(float(math.hypot(d[0], d[1])))
        prevc = c
    return statistics.mean(mags) if mags else 0.0


def _churn_per_frame(rows, aff, k):
    """Per-frame CHURN score (causal, window k): mean|d_indep| * (1 - R), where R is the
    directional coherence of the recent independent-motion vectors,
    R = |sum(d_indep)| / sum(|d_indep|) in [0,1].

    The magnitude signal failed because a box-hopping lock and a fast real-shape burst BOTH
    have large |d_indep|. (1-R) is the new separator: a coherent burst (real shape moving one
    direction, same box) -> R~1 -> churn~0 -> commit; a churn (output hopping among scattered
    fakes) -> directions cancel -> R~0 -> churn large -> freeze. Returns {idx: churn}."""
    win: deque = deque(maxlen=k)
    prevc = None
    out = {}
    for r in rows:
        idx, c, T = r["idx"], np.asarray(r["com"], np.float64), aff.get(r["idx"])
        if prevc is not None and T is not None:
            win.append(c - _apply(T, prevc))
        prevc = c
        if win:
            mags = [float(math.hypot(v[0], v[1])) for v in win]
            s = np.sum(win, axis=0)
            tot = sum(mags)
            R = float(math.hypot(s[0], s[1]) / tot) if tot > 1e-9 else 0.0
            out[idx] = statistics.mean(mags) * (1.0 - R)
        else:
            out[idx] = 0.0
    return out


def _churn_hedge(rows, aff, radius, churn_hi, k):
    """Freeze-blend gated on CHURN instead of magnitude: w = clamp(1 - churn/churn_hi, 0, 1),
    p_t = w*c_t + (1-w)*p_{t-1}. Coherent burst or stable track -> churn low -> w~1 commit;
    incoherent box-hopping -> churn high -> w~0 freeze (the 100%-recovering primitive)."""
    churn = _churn_per_frame(rows, aff, k)
    p = None
    hits = tot = 0
    for r in rows:
        idx, gt, c = r["idx"], r["gt"], np.asarray(r["com"], np.float64)
        if p is None:
            p = c.copy()
        else:
            w = min(max(1.0 - churn.get(idx, 0.0) / churn_hi, 0.0), 1.0)
            p = w * c + (1 - w) * p
        if gt is not None:
            tot += 1
            hits += _hit(p, gt, radius)
    return hits / tot if tot else 0.0


def _pctl(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, int(q * len(s)))
    return s[i]


# --------------------------------------------------------------------------- run

def run(weights, clips):
    data = []
    for clip in clips:
        name = clip.stem.replace("_cropped_trimmed", "")
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        _sx, _sy, radius, _start = _seed(packs)
        rows = _read_track(clip.stem)
        aff = _affines(clip, packs)
        cents = {idx: [_centroid(b) for b in packs[idx].boxes]
                 for idx in range(len(packs)) if packs[idx].boxes}
        com_wr = statistics.mean(r["within_r"] for r in rows if r["gt"] is not None)
        data.append(dict(name=name, rows=rows, aff=aff, cents=cents,
                         radius=radius, com_wr=com_wr))
        print(f"  loaded {name}: committed within_r={com_wr:.3f}  radius={radius:.0f}")

    by_name = {d["name"]: d for d in data}

    # ---------- 1+2. foundational hypothesis + recovery on swept-lock runs ----------
    print("\n=== swept-lock runs (laggards) -- foundational + recovery ===")
    print(f"  {'clip':>4} {'frames':>11} {'n':>3} | {'gtDisp':>6} {'comDisp':>7} {'ratio':>6} "
          f"{'indepM':>6} | {'commit':>6} {'freeze':>6} {'coast':>6} {'clust':>6}")
    tot = dict(committed=0, freeze=0, coast=0, cluster=0, n=0)
    for d in data:
        if d["name"] not in LAGGARD:
            continue
        rows, aff, radius = d["rows"], d["aff"], d["radius"]
        by_idx = {r["idx"]: r for r in rows}
        for run_rows in _swept_runs(rows):
            if not _grows(run_rows):
                continue
            f0, f1 = run_rows[0]["idx"], run_rows[-1]["idx"]
            # p0 = last in-radius committed pos before the run (else first run frame)
            prev = by_idx.get(f0 - 1)
            p0 = prev["com"] if prev is not None else run_rows[0]["com"]
            # foundational: per-frame GT vs committed screen displacement over the run
            gt_d, com_d = [], []
            for a, b in zip(run_rows[:-1], run_rows[1:]):
                if a["gt"] and b["gt"]:
                    gt_d.append(math.hypot(b["gt"][0] - a["gt"][0], b["gt"][1] - a["gt"][1]))
                com_d.append(math.hypot(b["com"][0] - a["com"][0], b["com"][1] - a["com"][1]))
            gtm = statistics.mean(gt_d) if gt_d else float("nan")
            comm = statistics.mean(com_d) if com_d else float("nan")
            ratio = comm / gtm if gtm and gtm > 1e-6 else float("inf")
            # c_prev0 == committed centroid at the pre-run frame (== p0, since the
            # committed output IS the chosen box's centroid each frame)
            h, indepm = _strategy_hits(run_rows, p0, p0, aff, d["cents"], radius)
            n = len(run_rows)
            for kk in ("committed", "freeze", "coast", "cluster"):
                tot[kk] += h[kk]
            tot["n"] += n
            print(f"  {d['name']:>4} {f0:>5}-{f1:<5} {n:>3} | {gtm:6.1f} {comm:7.1f} "
                  f"{ratio:6.1f} {indepm:6.1f} | {h['committed']/n:6.2f} {h['freeze']/n:6.2f} "
                  f"{h['coast']/n:6.2f} {h['cluster']/n:6.2f}")
    if tot["n"]:
        print(f"  {'ALL':>4} {'':>11} {tot['n']:>3} | {'':>6} {'':>7} {'':>6} {'':>6} | "
              f"{tot['committed']/tot['n']:6.2f} {tot['freeze']/tot['n']:6.2f} "
              f"{tot['coast']/tot['n']:6.2f} {tot['cluster']/tot['n']:6.2f}")
    print("  FOUNDATIONAL pass: ratio (comDisp/gtDisp) >> 1 on the runs (track swept, GT put).")
    print("  RECOVERY pass: freeze/coast/cluster within_r >> committed (~0) on these frames.")
    print("  TRUST-SEPARATION: indepM here (chosen box |d_indep| on swept frames) must be")
    print("  markedly BELOW the strong-clip mean below, else no causal trust gate can fire.")

    # ----- trust-signal separation: chosen-box |d_indep| per clip (whole clip) -----
    print("\n=== trust-signal separation (mean |d_indep| of committed box, whole clip) ===")
    for d in sorted(data, key=lambda d: d["name"] not in STRONG):
        tag = "strong " if d["name"] in STRONG else ("laggard" if d["name"] in LAGGARD else "       ")
        print(f"  {d['name']:>4} ({tag}): {_indep_mag(d['rows'], d['aff']):6.2f} px/frame")
    print("  If laggard whole-clip |d_indep| ~ strong-clip |d_indep|, the trust signal does")
    print("  not separate on-real from on-fake -> the gate's central wall.")

    # ---------------- 3. trust-gated end-to-end hedge: strong-clip SAFETY -------------
    print("\n=== trust-gated hedge end-to-end (SAFETY: strong clips must not regress) ===")
    K = 10
    his = [1.0, 2.0, 4.0, 8.0]
    for variant, fn in (("sheet-decomp", _trust_hedge), ("freeze-blend", _freeze_hedge)):
        print(f"\n  --- {variant} (window K={K}; w=clamp(mean|d_indep|/trust_hi,0,1)) ---")
        header = "  " + f"{'clip':>4} {'commit':>7} | " + " ".join(f"hi{h:g}".rjust(7) for h in his)
        print(header)
        deltas = {h: [] for h in his}
        for d in data:
            cells = []
            for h in his:
                wr = fn(d["rows"], d["aff"], d["radius"], h, K)
                deltas[h].append((d["name"], wr - d["com_wr"]))
                cells.append(f"{wr:7.3f}")
            tag = " (strong)" if d["name"] in STRONG else (" (laggard)" if d["name"] in LAGGARD else "")
            print(f"  {d['name']:>4} {d['com_wr']:7.3f} | " + " ".join(cells) + tag)
        for h in his:
            all_d = [x for _, x in deltas[h]]
            strong_worst = min((x for n, x in deltas[h] if n in STRONG), default=0.0)
            lag_gain = statistics.mean([x for n, x in deltas[h] if n in LAGGARD]) if any(
                n in LAGGARD for n, _ in deltas[h]) else 0.0
            print(f"    hi={h:>4g}: mean d={statistics.mean(all_d):+.4f}  "
                  f"strong-worst d={strong_worst:+.4f}  laggard-mean d={lag_gain:+.4f}")
    print("\n  SAFETY pass: a (variant, trust_hi) exists with strong-worst d >= -0.004 AND")
    print("  laggard-mean d > 0. If the strong clips always regress, the trust signal can't")
    print("  separate -> the decode-layer hedge is dead.")

    # ============ 4. CHURN discriminator: coherence (not magnitude) as the trigger ============
    # The magnitude trust failed because box-hops and real bursts both have large |d_indep|.
    # churn = mean|d_indep| * (1 - directional_coherence). The question this section answers:
    # does churn on swept-lock frames sit ABOVE strong-clip churn (incl. their bursts)?
    CK = 8
    print(f"\n=== CHURN separation (window K={CK}; churn = mean|d_indep| * (1 - R)) ===")
    churn_pf = {d["name"]: _churn_per_frame(d["rows"], d["aff"], CK) for d in data}
    # swept-lock frame churn (laggards, grown runs) vs strong-clip churn (whole clip)
    swept_ch, strong_ch = [], []
    for d in data:
        ch = churn_pf[d["name"]]
        if d["name"] in LAGGARD:
            by_idx = {r["idx"]: r for r in d["rows"]}
            for run_rows in _swept_runs(d["rows"]):
                if not _grows(run_rows):
                    continue
                swept_ch += [ch[r["idx"]] for r in run_rows if r["idx"] in ch]
        if d["name"] in STRONG:
            strong_ch += [ch[r["idx"]] for r in d["rows"] if r["idx"] in ch]
    print(f"  swept-lock frames : mean={statistics.mean(swept_ch):6.2f}  "
          f"p50={_pctl(swept_ch,0.5):6.2f}  p90={_pctl(swept_ch,0.9):6.2f}  (n={len(swept_ch)})")
    print(f"  strong-clip frames: mean={statistics.mean(strong_ch):6.2f}  "
          f"p50={_pctl(strong_ch,0.5):6.2f}  p90={_pctl(strong_ch,0.9):6.2f}  (n={len(strong_ch)})")
    print("  SEPARATION: swept p50 should sit ABOVE strong p90 for a clean threshold to exist.")

    print("\n=== churn-gated freeze-blend end-to-end (the actual test) ===")
    chis = [4.0, 8.0, 16.0, 32.0]
    header = "  " + f"{'clip':>4} {'commit':>7} | " + " ".join(f"ch{h:g}".rjust(7) for h in chis)
    print(header)
    cdeltas = {h: [] for h in chis}
    for d in data:
        cells = []
        for h in chis:
            wr = _churn_hedge(d["rows"], d["aff"], d["radius"], h, CK)
            cdeltas[h].append((d["name"], wr - d["com_wr"]))
            cells.append(f"{wr:7.3f}")
        tag = " (strong)" if d["name"] in STRONG else (" (laggard)" if d["name"] in LAGGARD else "")
        print(f"  {d['name']:>4} {d['com_wr']:7.3f} | " + " ".join(cells) + tag)
    best = None
    for h in chis:
        all_d = [x for _, x in cdeltas[h]]
        worst = min(all_d)                       # project bar: NO per-clip regression
        lag_gain = statistics.mean([x for n, x in cdeltas[h] if n in LAGGARD])
        mean_d = statistics.mean(all_d)
        print(f"    churn_hi={h:>4g}: mean d={mean_d:+.4f}  "
              f"worst-clip d={worst:+.4f}  laggard-mean d={lag_gain:+.4f}")
        if worst >= -0.004 and mean_d > 0:        # no per-clip regression + net gain
            if best is None or mean_d > best[1]:
                best = (h, mean_d)
    print("\n  VERDICT:", f"PASS - churn_hi={best[0]:g} clears the NO-PER-CLIP-REGRESSION bar "
          f"(mean d={best[1]:+.4f}); build fpath_hedge as a churn-gated freeze-blend, then "
          "verify with an honest LOO sweep." if best else
          "FAIL - no churn_hi gains net without regressing some clip.")

    # ---- honest LOO over churn_hi (mirror exp3_sweep: per held-out clip pick the best
    # no-per-clip-regression churn_hi on the OTHER 9, score the held-out clip) ----
    names = [dd["name"] for dd in data]
    base = {dd["name"]: dd["com_wr"] for dd in data}
    wr_by = {h: {n: base[n] + dx for (n, dx) in cdeltas[h]} for h in chis}
    print("\n=== leave-one-clip-out over churn_hi (no per-clip regression on train folds) ===")
    loo = []
    for hi, held in enumerate(names):
        best_cfg, best_key = None, None
        for h in chis:
            deltas = [wr_by[h][n] - base[n] for n in names if n != held]
            worst_d, mean_d = min(deltas), statistics.mean(deltas)
            if worst_d < -0.004:
                continue
            key = (round(worst_d, 4), round(mean_d, 4))
            if best_key is None or key > best_key:
                best_key, best_cfg = key, h
        if best_cfg is None:
            loo.append(base[held])
            print(f"   {held:>4}: no admissible churn_hi -> base {base[held]:.3f}")
        else:
            hv = wr_by[best_cfg][held]
            loo.append(hv)
            print(f"   {held:>4}: {hv:.3f} (base {base[held]:.3f}, {hv-base[held]:+.3f})  churn_hi={best_cfg:g}")
    loo_mean = statistics.mean(loo)
    base_mean = statistics.mean(base.values())
    worst = min(loo[i] - base[names[i]] for i in range(len(names)))
    print(f"\n  LOO mean within_r = {loo_mean:.4f}  (base {base_mean:.4f}, {loo_mean-base_mean:+.4f})"
          f"  worst_clip={worst:+.3f}")


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
