"""LD solver CLI.

Subcommands:
  strip-preview  original | stripped side-by-side MP4 (pointer stripping)
  solve          run the rigid-outlier solver, write a per-frame track CSV
  score          run the solver and score it against the green-crosshair GT
  evidence       render the rigid-outlier overlay video
"""
from __future__ import annotations

import argparse


def _strip_preview(args) -> None:
    from ld.debug.strip_preview import run

    run(args.input, args.out_video)


def _solve(args) -> None:
    from ld.config import OUTPUT_DIR
    from ld.solve import solve_clip, write_track_csv

    result = solve_clip(args.input)
    out = args.out_csv or str(OUTPUT_DIR / f"{_stem(args.input)}_track.csv")
    write_track_csv(result, out)
    print(f"start_frame={result.start_frame} seed_radius={result.seed_radius:.1f}px "
          f"frames={len(result.track)}")
    print(f"wrote -> {out}")


def _score(args) -> None:
    from ld.eval import score_clip

    print(score_clip(args.input))


def _evidence(args) -> None:
    from ld.debug.evidence import run

    run(args.input, args.out_video)


def _stem(path: str) -> str:
    from pathlib import Path

    return Path(path).stem


def main() -> None:
    ap = argparse.ArgumentParser(description="LD solver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("strip-preview", help="write original | stripped side-by-side MP4")
    p.add_argument("--input", required=True)
    p.add_argument("--out-video", default=None)
    p.set_defaults(func=_strip_preview)

    p = sub.add_parser("solve", help="run the solver, write per-frame track CSV")
    p.add_argument("--input", required=True)
    p.add_argument("--out-csv", default=None)
    p.set_defaults(func=_solve)

    p = sub.add_parser("score", help="run the solver and score vs green GT")
    p.add_argument("--input", required=True)
    p.set_defaults(func=_score)

    p = sub.add_parser("evidence", help="render rigid-outlier overlay video")
    p.add_argument("--input", required=True)
    p.add_argument("--out-video", default=None)
    p.set_defaults(func=_evidence)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
