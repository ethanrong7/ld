"""Step 9: evaluation harness.

Runs the solver on every `data/t*_cropped_trimmed.mp4`, scores per-frame
distance from the predicted target to the green-cursor ground truth, and prints
a summary table with a pass/fail per clip. Writes `output/eval_summary.csv`.

Pass criterion: median error <= PASS_MEDIAN_PX (target sits inside the shape).
GT cursor is read ONLY here for scoring (on raw frames). The solver preprocesses
frames with ``strip_pointer`` before grayscale (default); use ``--no-strip-pointer``
for ablation only.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from ld.config import DATA_DIR, OUTPUT_DIR
from ld.solver import track_video
from ld.vision.cursor import find_cursor

PASS_MEDIAN_PX = 25.0


def eval_clip(path: str, *, strip_pointer_from_frames: bool = True) -> dict:
    errs: list[float] = []
    coast = {"n": 0}

    def on_frame(idx, frame, st):
        if not st.measured:
            coast["n"] += 1
        gt = find_cursor(frame)
        if gt:
            errs.append(math.hypot(st.x - gt[0], st.y - gt[1]))

    res = track_video(
        path, on_frame=on_frame, strip_pointer_from_frames=strip_pointer_from_frames
    )
    a = np.array(errs) if errs else np.array([float("nan")])
    median = float(np.median(a))
    return {
        "clip": Path(path).name,
        "handoff": res.init.handoff_frame,
        "omega": round(res.init.omega_deg_per_frame, 2),
        "radius": round(res.init.template_radius, 1),
        "scored": len(errs),
        "mean": round(float(np.mean(a)), 1),
        "median": round(median, 1),
        "p90": round(float(np.percentile(a, 90)), 1),
        "max": round(float(np.max(a)), 1),
        "coast": coast["n"],
        "pass": median <= PASS_MEDIAN_PX,
    }


def run(
    glob: str = "t*_cropped_trimmed.mp4",
    csv_path: str | None = None,
    *,
    strip_pointer_from_frames: bool = True,
) -> None:
    csv_path = csv_path or str(OUTPUT_DIR / "eval_summary.csv")
    clips = sorted(DATA_DIR.glob(glob))
    if not clips:
        print(f"No clips matched {glob} in {DATA_DIR}")
        return

    rows = []
    hdr = ("clip", "handoff", "omega", "radius", "scored", "mean",
           "median", "p90", "max", "coast", "pass")
    print(f"{'clip':<26}{'hand':>5}{'omega':>7}{'rad':>6}{'med':>7}{'p90':>7}{'max':>7}{'coast':>6}  pass")
    for c in clips:
        try:
            r = eval_clip(str(c), strip_pointer_from_frames=strip_pointer_from_frames)
        except Exception as e:  # keep going across the batch
            print(f"{c.name:<26}  ERROR: {e}")
            continue
        rows.append(r)
        print(f"{r['clip']:<26}{r['handoff']:>5}{r['omega']:>7}{r['radius']:>6}"
              f"{r['median']:>7}{r['p90']:>7}{r['max']:>7}{r['coast']:>6}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")

    if rows:
        meds = np.array([r["median"] for r in rows])
        npass = sum(r["pass"] for r in rows)
        print(f"\n{npass}/{len(rows)} pass  |  median-of-medians={np.median(meds):.1f}px")
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote -> {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="t*_cropped_trimmed.mp4")
    ap.add_argument("--csv", default=str(OUTPUT_DIR / "eval_summary.csv"))
    ap.add_argument(
        "--no-strip-pointer",
        action="store_true",
        help="disable green/mouse inpaint before tracking (ablation only)",
    )
    args = ap.parse_args()
    run(args.glob, args.csv, strip_pointer_from_frames=not args.no_strip_pointer)


if __name__ == "__main__":
    main()
