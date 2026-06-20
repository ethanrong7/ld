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

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = DATA_DIR / "additional_evidence"

# Target format of the existing training clips.
OUT_W, OUT_H = 744, 498
OUT_FPS = 60.0
TARGET_AR = OUT_W / OUT_H  # ~1.494

# Board-detection / run-selection tunables.
AR_LO, AR_HI = 1.40, 1.60       # board aspect ratio window
MIN_W_FRAC, MIN_H_FRAC = 0.35, 0.35  # board must span this fraction of the frame
GAP_BRIDGE = 15                 # merge board runs separated by <= this many frames
MIN_RUN_FRAMES = 180            # ignore runs shorter than ~3s (false positives)


def board_rect(frame: np.ndarray):
    """Return (x, y, w, h) of the lie-detector board, or None if not present."""
    h, w = frame.shape[:2]
    b, g, r = cv2.split(frame.astype(np.int32))
    # Organic tan texture: red & green high, blue suppressed, warm bias.
    mask = ((r > 110) & (g > 90) & (b < g) & (r >= g) & ((r - b) > 30)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        ar = bw / bh
        if AR_LO < ar < AR_HI and bw > MIN_W_FRAC * w and bh > MIN_H_FRAC * h:
            area = cv2.contourArea(cnt)
            if best is None or area > best[0]:
                best = (area, (x, y, bw, bh))
    return best[1] if best else None


def longest_board_run(rects: list):
    """Given per-frame board rects (None when absent), return (start, end, rect).

    ``end`` is inclusive. ``rect`` is the stable median board box over the run.
    Returns None if no run passes MIN_RUN_FRAMES.
    """
    present = np.array([r is not None for r in rects])
    if not present.any():
        return None

    # Collect [start, end] runs of present frames, then bridge small gaps.
    idx = np.where(present)[0]
    runs = []
    s = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > GAP_BRIDGE:
            runs.append([s, prev])
            s = i
        prev = i
    runs.append([s, prev])

    best = max(runs, key=lambda r: r[1] - r[0])
    start, end = best
    if end - start + 1 < MIN_RUN_FRAMES:
        return None

    boxes = np.array([rects[i] for i in range(start, end + 1) if rects[i] is not None])
    med = np.median(boxes, axis=0).round().astype(int)
    return start, end, tuple(int(v) for v in med)


def detect_run(path: Path):
    """First pass: scan a video, return (start, end, rect, fps, n) or None."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or OUT_FPS
    rects = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rects.append(board_rect(frame))
    cap.release()
    run = longest_board_run(rects)
    if run is None:
        return None
    start, end, rect = run
    return start, end, rect, fps, len(rects)


def crop_resize(frame: np.ndarray, rect):
    x, y, w, h = rect
    H, W = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    crop = frame[y:y + h, x:x + w]
    interp = cv2.INTER_AREA if (w > OUT_W or h > OUT_H) else cv2.INTER_LINEAR
    return cv2.resize(crop, (OUT_W, OUT_H), interpolation=interp)


def write_clip(path: Path, out_path: Path, start: int, end: int, rect):
    """Second pass: re-read source and write the cropped/trimmed clip."""
    cap = cv2.VideoCapture(str(path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, OUT_FPS, (OUT_W, OUT_H))
    fi = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if start <= fi <= end:
            writer.write(crop_resize(frame, rect))
            written += 1
        fi += 1
    cap.release()
    writer.release()
    return written


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
