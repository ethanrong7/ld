"""Detect + crop the Lie-Detector board out of a raw MapleStory capture.

Raw captures are the full gameplay window at the wrong resolution with redundant
footage around the minigame. The board is the one large rectangle of organic tan
texture (aspect ~1.49 = 744/498); the gameplay around it is purple/dark, so a tan
colour mask + largest-aspect-matching contour finds it cleanly.

Shared by:
  - ld.detect.annotate    (extract training frames from new videos in data/)
  - make_additional_evidence.py (build the held-out validation clip set)

`is_board_sized(frame)` lets callers skip cropping for clips already in the
744x498 training format (the s*/t* clips).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Target format of the training clips.
OUT_W, OUT_H = 744, 498
OUT_FPS = 60.0
TARGET_AR = OUT_W / OUT_H  # ~1.494

# Board-detection / run-selection tunables.
AR_LO, AR_HI = 1.40, 1.60            # board aspect ratio window
MIN_W_FRAC, MIN_H_FRAC = 0.35, 0.35  # board must span this fraction of the frame
GAP_BRIDGE = 15                      # merge board runs separated by <= this many frames
MIN_RUN_FRAMES = 180                 # ignore runs shorter than ~3s (false positives)


def is_board_sized(frame: np.ndarray) -> bool:
    """True if the frame is already in the 744x498 training format (no crop needed)."""
    h, w = frame.shape[:2]
    return w == OUT_W and h == OUT_H


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


def crop_resize(frame: np.ndarray, rect) -> np.ndarray:
    x, y, w, h = rect
    H, W = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    crop = frame[y:y + h, x:x + w]
    interp = cv2.INTER_AREA if (w > OUT_W or h > OUT_H) else cv2.INTER_LINEAR
    return cv2.resize(crop, (OUT_W, OUT_H), interpolation=interp)


def write_clip(path: Path, out_path: Path, start: int, end: int, rect) -> int:
    """Second pass: re-read source and write the cropped/trimmed clip. Returns frames written."""
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
