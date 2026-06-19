"""GATE (read-only) for `fpath_lag` -- a bounded fixed-lag (bidirectional) position
smoother at the DECODE layer (plan.md "BOUNDED FIXED-LAG DECODE SMOOTHER", 2026-06-20).

The bet: the leader (`fpath_hedge`) emits position STRICTLY CAUSALLY and leaves the
~10-15-frame fixed-lag buffer the project explicitly permits completely UNUSED at the
decode layer. With L frames of lookahead, the output for frame t is decided AFTER seeing
t+1..t+L, so a swept-lock excursion -- the output riding a fake 200-400px off the
near-stationary real shape -- is an obvious physics-violating OUTLIER (real shape p99
17.8 / max 44.7 px/frame) and can be replaced by a radius-bounded robust estimate. The
causal hedge can only guess at this from backward churn; the lag pass SEES the excursion.

This probe gates the idea BEFORE building the mode. It is read-only: the committed
`fpath_hyst` chosen-box centroid + gt + within_r + oracle come straight from
data/detect/eval/<stem>__fpath_hyst.csv (the identity layer of the leader). NO pixel
pass, NO re-detection, NO new compute -- it operates on the cached centroid sequence.

It must answer, in order:

  0. DURATION (the decisive cheap measurement). The robust estimate is a median /
     Theil-Sen over the CHOSEN-box positions in [t-L, t+L]. During a SUSTAINED swept
     lock the whole window sits on the fake -> the median is ALSO a fake -> bidirectional
     buys nothing over the causal freeze. So bidirectional only helps if the excursion is
     SHORTER than ~2L (track leaves GT and returns within the window). Report the swept-run
     duration distribution vs 2L: the fraction of miss-frames in runs shorter than 2L is
     the optimistic ceiling on what the lag buffer can recover beyond the hedge.

  1+2+3. FLAG / RECOVERY / DAMAGE on `fpath_hyst` MISS frames (within_r==0 & oracle==1):
     does the excursion (chosen-to-robust distance) exceed the PHYSICAL bound (p99 17.8 /
     max 44.7 -- NEVER below; below p99 is the dead velocity-cap); when it does, does
     REPLACING the chosen position with the robust estimate land within_r of GT far more
     than the chosen position does (recover > damage)? Damage is measured on the correct
     (within_r==1) frames: a flag firing there that replaces a good position out of radius.

GATE (plan.md EXP-L1): on t5/t1 MISS frames, robust-replacement lands within_r
MATERIALLY more often than chosen (recover > damage) AND the flag actually fires. t8 is
reported separately and expected flat. If excursions sit inside the tube (flag never
fires) or the median is contaminated (step-0 runs > 2L, recovery low) -> the lag buffer
has no usable signal beyond the causal hedge -> STOP, do not build the mode.

    python -m ld.detect.lag_smooth_probe --weights data/detect/runs/.../best.pt
"""
from __future__ import annotations

import argparse
import math
import statistics

from ld.config import DATA_DIR
from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _seed
# reuse the read-only scaffold from the hedge gate
from ld.detect.hedge_probe import _read_track, _swept_runs, _grows

# physical motion bounds of the real shape (CLAUDE.md "Physics") -- the reject threshold
# lives at/above these, NEVER below (below p99 == the dead velocity-cap dead-end).
P99 = 17.8
MAX = 44.7
LAGS = (6, 9, 12, 15)        # fixed-lag half-windows L to sweep
MIN_WIN = 5                  # need at least this many samples in a window to fit
STRONG = {"t2", "t6", "t7", "t9", "t10"}
LAGGARD = {"t1", "t5", "t8"}


# ------------------------------------------------------------- robust estimators

def _median_xy(pts):
    """Component-wise temporal median of a list of (x,y)."""
    xs = sorted(p[0] for p in pts)
    ys = sorted(p[1] for p in pts)
    return (statistics.median(xs), statistics.median(ys))


def _theilsen_1d(fs, vs, at):
    """Theil-Sen line (median pairwise slope) of vs over frame-index fs, evaluated at `at`.
    Robust to the outlier swept segment as long as it is the minority of the window."""
    n = len(fs)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            df = fs[j] - fs[i]
            if df != 0:
                slopes.append((vs[j] - vs[i]) / df)
    if not slopes:
        return statistics.median(vs)
    m = statistics.median(slopes)
    b = statistics.median([v - m * f for f, v in zip(fs, vs)])
    return m * at + b


def _theilsen_xy(window, at):
    """window: list of (frame_idx, (x,y)). Returns Theil-Sen estimate at frame `at`."""
    fs = [w[0] for w in window]
    return (_theilsen_1d(fs, [w[1][0] for w in window], at),
            _theilsen_1d(fs, [w[1][1] for w in window], at))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --------------------------------------------------------------------------- run

def run(weights, clips):
    data = []
    for clip in clips:
        name = clip.stem.replace("_cropped_trimmed", "")
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        _sx, _sy, radius, _start = _seed(packs)
        rows = _read_track(clip.stem)
        com_wr = statistics.mean(r["within_r"] for r in rows if r["gt"] is not None)
        data.append(dict(name=name, rows=rows, radius=radius, com_wr=com_wr))
        print(f"  loaded {name}: committed within_r={com_wr:.3f}  radius={radius:.0f}")

    # ============ 0. DURATION: swept-run length vs 2L (the decisive cheap gate) ============
    print("\n=== step-0  swept-lock excursion duration vs window 2L ===")
    print("  (a grown swept run longer than 2L cannot be pulled back by a median/Theil-Sen")
    print("   over [t-L,t+L] -- the whole window is on the fake. Short runs = recoverable.)")
    print(f"  {'clip':>4} | {'#runs':>5} {'#missF':>6} | run-lengths"
          + "".join(f"  <2L({2*L})" for L in LAGS))
    tot_missF = 0
    tot_short = {L: 0 for L in LAGS}
    for d in data:
        runs = [r for r in _swept_runs(d["rows"]) if _grows(r)]
        lens = sorted((len(r) for r in runs), reverse=True)
        missF = sum(lens)
        tot_missF += missF
        short = {}
        for L in LAGS:
            s = sum(n for n in lens if n < 2 * L)
            short[L] = s
            tot_short[L] += s
        tag = "L" if d["name"] in LAGGARD else ("S" if d["name"] in STRONG else " ")
        lenstr = ",".join(str(n) for n in lens[:8]) + ("..." if len(lens) > 8 else "")
        print(f"  {d['name']:>4}{tag}| {len(runs):>5} {missF:>6} | {lenstr:<28}"
              + "".join(f"  {short[L]:>7}" for L in LAGS))
    print(f"  {'ALL':>4} | {'':>5} {tot_missF:>6} |"
          + "".join(f"  {tot_short[L]:>7}" for L in LAGS)
          + f"   (frac in runs<2L: " + " ".join(
              f"L{L}={tot_short[L]/tot_missF:.2f}" for L in LAGS) + ")")
    print("  READ: if most miss-frames live in runs >= 2L for every L, bidirectional adds")
    print("  nothing over the causal freeze -> expect step-1..3 recovery to be low. STOP-signal.")

    # ============ 1+2+3. FLAG / RECOVERY / DAMAGE per lag L ============
    for L in LAGS:
        print(f"\n=== lag L={L}  (window [t-L,t+L]; flag if chosen-to-robust > P99={P99}) ===")
        print(f"  {'clip':>4} | {'miss':>4} {'flag%':>5} {'flag>max%':>9} | "
              f"{'med rec':>7} {'med dmg':>7} | {'TS rec':>7} {'TS dmg':>7}")
        agg = {"miss": 0, "flag": 0, "flagmax": 0,
               "med_rec": 0, "med_dmg_n": 0, "med_dmg": 0,
               "ts_rec": 0, "ts_dmg": 0, "ok": 0}
        per_clip = {}
        for d in data:
            rows = [r for r in d["rows"] if r["gt"] is not None]
            radius = d["radius"]
            by_idx = {r["idx"]: r for r in rows}
            idxs = sorted(by_idx)
            c = dict(miss=0, flag=0, flagmax=0, med_rec=0, med_dmg=0,
                     ts_rec=0, ts_dmg=0, ok=0)
            for r in rows:
                t = r["idx"]
                win = [(j, by_idx[j]["com"]) for j in idxs if abs(j - t) <= L]
                if len(win) < MIN_WIN:
                    continue
                pts = [w[1] for w in win]
                med = _median_xy(pts)
                ts = _theilsen_xy(win, t)
                chosen = r["com"]
                gt = r["gt"]
                exc = _dist(chosen, med)               # excursion vs robust estimate
                flag = exc > P99
                miss = (r["within_r"] == 0 and r["oh"] == 1)
                ok = (r["within_r"] == 1)
                med_in = _dist(med, gt) < radius
                ts_in = _dist(ts, gt) < radius
                if miss:
                    c["miss"] += 1
                    if flag:
                        c["flag"] += 1
                        if exc > MAX:
                            c["flagmax"] += 1
                        # recovery: on a flagged miss frame, would the robust est land in r?
                        c["med_rec"] += med_in
                        c["ts_rec"] += ts_in
                if ok:
                    c["ok"] += 1
                    if flag:                            # damage: flag fired on a CORRECT frame
                        if not med_in:
                            c["med_dmg"] += 1
                        if not ts_in:
                            c["ts_dmg"] += 1
            per_clip[d["name"]] = c
            for k in agg:
                agg[k] += c.get(k, 0) if k in c else agg[k]
            agg["med_dmg_n"] += c["ok"]
            mf = c["miss"] or 1
            okf = c["ok"] or 1
            tag = "L" if d["name"] in LAGGARD else ("S" if d["name"] in STRONG else " ")
            print(f"  {d['name']:>4}{tag}| {c['miss']:>4} "
                  f"{100*c['flag']/mf:>4.0f}% {100*c['flagmax']/mf:>8.0f}% | "
                  f"{c['med_rec']/mf:>7.2f} {c['med_dmg']/okf:>7.3f} | "
                  f"{c['ts_rec']/mf:>7.2f} {c['ts_dmg']/okf:>7.3f}")
        # laggard focus line (the gate is on t5/t1)
        for grp, names in (("t5/t1", ("t5", "t1")), ("t8", ("t8",))):
            miss = sum(per_clip[n]["miss"] for n in names)
            flag = sum(per_clip[n]["flag"] for n in names)
            mrec = sum(per_clip[n]["med_rec"] for n in names)
            trec = sum(per_clip[n]["ts_rec"] for n in names)
            ok = sum(per_clip[n]["ok"] for n in names)
            mdmg = sum(per_clip[n]["med_dmg"] for n in names)
            tdmg = sum(per_clip[n]["ts_dmg"] for n in names)
            mf = miss or 1
            okf = ok or 1
            print(f"   -> {grp:>5} MISS={miss:<4} flag={100*flag/mf:>3.0f}%  "
                  f"med rec={mrec/mf:.2f} dmg={mdmg/okf:.3f}  "
                  f"TS rec={trec/mf:.2f} dmg={tdmg/okf:.3f}")
    print("\n  GATE (t5/t1): recovery >> damage AND flag firing -> proceed to EXP-L2.")
    print("  Else (flag silent or recovery low / damage high) -> lag buffer dead beyond hedge. STOP.")


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
