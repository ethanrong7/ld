"""Step-0 diagnostic (read-only): classify each sustained miss run in the laggard
clips t1/t5/t8 as A (identity creep), B (detection gap), or C (coast runaway).

A = oracle_hit mostly 1 (real box present) but within_r=0  -> identity picked wrong box
B = oracle_hit mostly 0 (no box on GT)                     -> detection gap
C = err_px grows monotonically through the run             -> coast/lock runaway

Reads data/detect/eval/<stem>__fpath_hyst.csv. No new infra; pure CSV forensics.

    python -m ld.detect.step0_failchar
"""
from __future__ import annotations

import csv
from pathlib import Path

from ld.config import DATA_DIR

CLIPS = ["t1", "t5", "t8"]
MODE = "fpath_hyst"
MIN_RUN = 3  # report sustained runs of >= this many consecutive miss frames


def _load(stem: str):
    path = DATA_DIR / "detect" / "eval" / f"{stem}__{MODE}.csv"
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return path, rows


def _runs(rows):
    """Yield (start_idx, end_idx, run_rows) for each maximal within_r==0 run."""
    run = []
    for r in rows:
        if int(float(r["within_r"])) == 0:
            run.append(r)
        else:
            if len(run) >= MIN_RUN:
                yield run
            run = []
    if len(run) >= MIN_RUN:
        yield run


def _classify(run):
    n = len(run)
    oh = [int(float(r["oracle_hit"])) for r in run]
    err = [float(r["err_px"]) for r in run]
    oh_frac = sum(oh) / n
    # monotone-ish growth: last third mean err >> first third mean err
    k = max(1, n // 3)
    first = sum(err[:k]) / k
    last = sum(err[-k:]) / k
    grows = last > first * 1.8 and last > 150
    if oh_frac >= 0.6 and not grows:
        cls = "A identity-creep"
    elif oh_frac < 0.4:
        cls = "B detection-gap"
    elif grows:
        cls = "C coast-runaway"
    else:
        cls = "? mixed"
    return cls, oh_frac, min(err), max(err), first, last


def main():
    for c in CLIPS:
        stem = f"{c}_cropped_trimmed"
        path, rows = _load(stem)
        total = len(rows)
        misses = sum(1 for r in rows if int(float(r["within_r"])) == 0)
        print(f"\n=== {c}  ({total} scored frames, {misses} miss = {misses/total:.1%}) ===")
        print(f"    {path}")
        print(f"    {'frames':>14}  {'n':>3}  {'oracle_hit':>10}  {'err min..max':>16}  {'first->last':>14}  class")
        a = b = cc = q = 0
        for run in _runs(rows):
            f0, f1 = run[0]["idx"], run[-1]["idx"]
            cls, ohf, emin, emax, ef, el = _classify(run)
            tag = cls[0]
            a += tag == "A"; b += tag == "B"; cc += tag == "C"; q += tag == "?"
            print(f"    {f0:>6}-{f1:<6}  {len(run):>3}  {ohf:>10.2f}  {emin:>6.0f}..{emax:<6.0f}  {ef:>6.0f}->{el:<6.0f}  {cls}")
        print(f"    run tally: A(identity)={a}  B(detection)={b}  C(coast)={cc}  ?={q}")


if __name__ == "__main__":
    main()
