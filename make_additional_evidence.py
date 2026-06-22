"""Standalone: turn raw MapleStory captures into Lie-Detector training clips.

The training clips (``data/s1..s12.mp4`` and ``data/t*_cropped_trimmed.mp4``) are
all 744x498 @ 60fps, cropped to *only* the lie-detector board and trimmed to the
minigame itself: they start on the frame the board+countdown appears and end on
the last in-play frame (the success notification fires immediately after).

Every *other* video in ``data/`` is a raw capture: full gameplay window, wrong
resolution, redundant footage before/after the minigame. This script converts
each one into the training-clip format and drops it in ``data/additional_evidence/``.

How it works (per source video):
  1. Detect the board each frame -- it is the one large rectangle of organic tan
     texture with aspect ratio ~1.49 (744/498). Gameplay around it is purple/dark,
     so a simple tan colour mask + largest-aspect-matching contour finds it cleanly.
  2. Take the longest run of consecutive board-present frames (bridging tiny gaps
     from the countdown speech-bubble / flashes). First frame = countdown appears;
     last frame = board removed -> success. This matches the reference clips.
  3. Crop to a single stable board rect (median over the run), resize to 744x498,
     and write an mp4 at 60fps with no audio.

Usage:
    .venv\\Scripts\\python.exe make_additional_evidence.py
    .venv\\Scripts\\python.exe make_additional_evidence.py --dry-run   # detect only
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Board detection / cropping is shared with the training-frame extractor.
from ld.detect.board_crop import OUT_FPS, OUT_W, OUT_H, detect_run, write_clip  # noqa: F401

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = DATA_DIR / "additional_evidence"


def source_videos():
    """All raw captures: every data/*.mp4 that is NOT an existing training clip."""
    out = []
    for p in sorted(DATA_DIR.glob("*.mp4")):
        name = p.stem
        if re.fullmatch(r"s\d+", name):                      # s1..s12
            continue
        if re.fullmatch(r"t\d+_cropped_trimmed", name):      # t1..t10
            continue
        out.append(p)
    return out


def safe_name(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="detect minigame bounds only; do not write clips")
    args = ap.parse_args()

    vids = source_videos()
    print(f"Found {len(vids)} raw videos to process (excluding s*/t* training clips).\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping = []
    for i, path in enumerate(vids, 1):
        print(f"[{i}/{len(vids)}] {path.name}")
        res = detect_run(path)
        if res is None:
            print("    !! no minigame board detected -- skipped\n")
            continue
        start, end, rect, fps, n = res
        dur = (end - start + 1) / OUT_FPS
        print(f"    board rect={rect}  frames {start}-{end} "
              f"({end - start + 1} frames, {dur:.1f}s @60fps; source {n} frames @{fps:.2f})")
        if args.dry_run:
            print()
            continue
        out_path = OUT_DIR / f"a{i:02d}_{safe_name(path.stem)}.mp4"
        written = write_clip(path, out_path, start, end, rect)
        print(f"    -> {out_path.relative_to(DATA_DIR.parent)}  ({written} frames)\n")
        mapping.append((path.name, out_path.name, written))

    if mapping:
        print("Done. Source -> output:")
        for src, dst, nf in mapping:
            print(f"  {src}  ->  additional_evidence/{dst}  ({nf} frames)")


if __name__ == "__main__":
    main()
