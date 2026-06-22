"""Extract + single-class annotate training frames for the YOLO shape detector.

One command takes you from "I dropped a new video in data/" to labelled frames:

  1. DISCOVER  -- every data/*.mp4 that has not been extracted yet (or --clips).
  2. CROP      -- raw screen captures are cropped to the 744x498 board (board_crop);
                  clips already in that format (s*/t*) are used as-is.
  3. EXTRACT   -- pick TRAIN_FRAMES_PER_CLIP (=5) evenly-spaced frames from the
                  ACTIVE tracking window only: the longest run of frames with no
                  countdown shape, no START overlay, and no end-of-round success
                  popup (all three are large bright blobs the board never shows
                  during play). The green cursor is inpainted out (strip_pointer).
  4. ANNOTATE  -- a drag-to-draw box annotator; one class ("shape"). Every shape
                  on the sheet gets a box (the real one is camouflaged among fakes;
                  box them all -- identity is decided downstream, not by the labels).

Frames -> data/detect/s_frames/, labels -> data/detect/s_labels_single/.
After you quit, the single-class dataset is rebuilt and the train command is
PRINTED (never auto-run -- verify your labels first).

Controls:
  left-drag   draw a box        u  undo last box     c  clear frame
  n / SPACE   save + next       p  save + previous   s  save (stay)
  q / ESC     save + quit

Usage:
  python -m ld.detect.annotate                 # all new videos in data/
  python -m ld.detect.annotate --clips s13 my_capture
  python -m ld.detect.annotate --per-clip 8 --no-build
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ld.config import (DATA_DIR, TRAIN_DATASET_DIR, TRAIN_FRAMES_DIR,
                       TRAIN_FRAMES_PER_CLIP, TRAIN_LABELS_DIR, TRAIN_MANIFEST,
                       TRAIN_RUN_NAME)
from ld.detect.board_crop import OUT_H, OUT_W, crop_resize, detect_run, is_board_sized
from ld.detect.boxes import Box, load_labels, save_labels
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import strip_pointer

WINDOW = "LD annotate  [drag=box  u=undo  c=clear  n=next  p=prev  s=save  q=quit]"
BOX_COLOR = (0, 255, 0)
LIVE_COLOR = (255, 200, 0)
CENTROID_COLOR = (0, 0, 255)


# ---------------------------------------------------------------------------
# Active-window detection (no countdown / START / success frames)
# ---------------------------------------------------------------------------

def _has_overlay(frame: np.ndarray, blob_thresh: int = 50) -> bool:
    """True if the frame has a large bright blob (countdown shape / START / success)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max((cv2.contourArea(c) for c in cnts), default=0) >= blob_thresh


def _in_overlay(frame: np.ndarray) -> bool:
    """A frame is outside the tracking window if the countdown shape OR any big
    bright overlay (START flash at the head, success popup at the tail) is present."""
    return detect_white_shape(frame) is not None or _has_overlay(frame)


def _active_window(overlay: list[bool]) -> tuple[int, int]:
    """Longest run of clean (non-overlay) frames = the in-play tracking window.

    Returns (lo, hi) inclusive indices into `overlay`. This trims the leading
    countdown/START and the trailing success popup in one shot.
    """
    best_lo = best_hi = -1
    best_len = 0
    lo = None
    for i, ov in enumerate(overlay):
        if ov:
            lo = None
            continue
        if lo is None:
            lo = i
        if i - lo + 1 > best_len:
            best_len, best_lo, best_hi = i - lo + 1, lo, i
    if best_lo < 0:                      # no clean run -> fall back to whole span
        return 0, len(overlay) - 1
    return best_lo, best_hi


def _pick_indices(lo: int, hi: int, n: int) -> list[int]:
    """n evenly-spaced indices in [lo, hi] (inclusive)."""
    if hi < lo:
        return []
    span = hi - lo + 1
    if n >= span:
        return list(range(lo, hi + 1))
    return sorted({int(round(v)) for v in np.linspace(lo, hi, n)})


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _safe_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def _read_run(path: Path) -> tuple[list[np.ndarray], list[int]]:
    """Return (board frames in play order, their source frame indices).

    Raw captures are cropped to the board over their longest board-present run;
    clips already in 744x498 format are returned frame-for-frame.
    """
    cap = cv2.VideoCapture(str(path))
    ok, first = cap.read()
    if not ok:
        cap.release()
        return [], []
    cap.release()

    if is_board_sized(first):
        frames, idxs = [], []
        cap = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
            idxs.append(i)
            i += 1
        cap.release()
        return frames, idxs

    run = detect_run(path)
    if run is None:
        return [], []
    start, end, rect, _fps, _n = run
    frames, idxs = [], []
    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if start <= i <= end:
            frames.append(crop_resize(fr, rect))
            idxs.append(i)
        i += 1
    cap.release()
    return frames, idxs


def extract_clip(path: Path, per_clip: int) -> list[dict]:
    """Extract, cursor-strip, and save up to `per_clip` in-play frames. Returns records."""
    stem = _safe_stem(path.stem)
    frames, src_idxs = _read_run(path)
    if not frames:
        print(f"  {stem}: no board run detected -- skipped")
        return []

    overlay = [_in_overlay(f) for f in frames]
    lo, hi = _active_window(overlay)
    picks = _pick_indices(lo, hi, per_clip)
    print(f"  {stem}: {len(frames)} board frames, window [{lo},{hi}] -> "
          f"picking {[src_idxs[p] for p in picks]}")

    TRAIN_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for p in picks:
        frame = strip_pointer(frames[p], strip_green=True)
        h, w = frame.shape[:2]
        name = f"{stem}_{src_idxs[p]:05d}.png"
        cv2.imwrite(str(TRAIN_FRAMES_DIR / name), frame)
        records.append({"image": name, "clip": stem, "frame": src_idxs[p],
                        "width": w, "height": h})
    return records


def _discover(clip_args: list[str] | None) -> list[Path]:
    if clip_args:
        out = []
        for c in clip_args:
            p = Path(c)
            if not p.exists():
                p = DATA_DIR / (c if c.endswith(".mp4") else c + ".mp4")
            if p.exists():
                out.append(p)
            else:
                print(f"  WARNING: {c} not found, skipping")
        return out
    # Auto-discover: every data/*.mp4 with no frames extracted yet.
    done = {f.stem.rsplit("_", 1)[0] for f in TRAIN_FRAMES_DIR.glob("*.png")}
    return [p for p in sorted(DATA_DIR.glob("*.mp4"))
            if _safe_stem(p.stem) not in done]


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------

class _State:
    def __init__(self) -> None:
        self.boxes: list[Box] = []
        self.drawing = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0


def _label_path(image_name: str) -> Path:
    return TRAIN_LABELS_DIR / f"{Path(image_name).stem}.txt"


def _draw(base: np.ndarray, st: _State, rec: dict, i: int, total: int) -> np.ndarray:
    img = base.copy()
    for b in st.boxes:
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), BOX_COLOR, 2)
        cv2.circle(img, (int((b.x1 + b.x2) / 2), int((b.y1 + b.y2) / 2)), 4, CENTROID_COLOR, -1)
    if st.drawing:
        cv2.rectangle(img, (st.x0, st.y0), (st.x1, st.y1), LIVE_COLOR, 2)
    hud = f"{rec['clip']} f{rec['frame']}  boxes={len(st.boxes)}  [{i+1}/{total}]"
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, hud, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def annotate(records: list[dict]) -> None:
    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    st = _State()

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            st.drawing = True
            st.x0 = st.x1 = x
            st.y0 = st.y1 = y
        elif event == cv2.EVENT_MOUSEMOVE and st.drawing:
            st.x1, st.y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP and st.drawing:
            st.drawing = False
            st.x1, st.y1 = x, y
            b = Box(0, st.x0, st.y0, st.x1, st.y1)
            if b.valid():
                st.boxes.append(b)

    cv2.setMouseCallback(WINDOW, on_mouse)

    i = 0
    while 0 <= i < len(records):
        rec = records[i]
        base = cv2.imread(str(TRAIN_FRAMES_DIR / rec["image"]))
        if base is None:
            print(f"  skip (missing image): {rec['image']}")
            i += 1
            continue
        h, w = base.shape[:2]
        lp = _label_path(rec["image"])
        st.boxes = load_labels(lp, w, h) if lp.exists() else []

        while True:
            cv2.imshow(WINDOW, _draw(base, st, rec, i, len(records)))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("n"), ord(" ")):
                save_labels(lp, st.boxes, w, h); i += 1; break
            if key == ord("p"):
                save_labels(lp, st.boxes, w, h); i = max(0, i - 1); break
            if key == ord("s"):
                save_labels(lp, st.boxes, w, h)
                print(f"  saved {rec['image']} ({len(st.boxes)} boxes)")
            elif key == ord("u") and st.boxes:
                st.boxes.pop()
            elif key == ord("c"):
                st.boxes = []
            elif key in (ord("q"), 27):
                save_labels(lp, st.boxes, w, h)
                cv2.destroyAllWindows()
                print(f"  stopped at {i+1}/{len(records)}")
                _write_manifest(records)
                return

    cv2.destroyAllWindows()
    _write_manifest(records)
    print(f"done: labelled {len(records)} frames -> {TRAIN_LABELS_DIR}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class BoxRecord:
    cls: int
    x1: float; y1: float; x2: float; y2: float
    cx: float; cy: float
    w: float; h: float


def _write_manifest(records: list[dict]) -> None:
    out: list[dict] = []
    for rec in records:
        lp = _label_path(rec["image"])
        w, h = rec["width"], rec["height"]
        boxes = load_labels(lp, w, h) if lp.exists() else []
        entry = dict(rec)
        entry["boxes"] = [asdict(BoxRecord(
            cls=b.cls, x1=round(b.x1, 2), y1=round(b.y1, 2),
            x2=round(b.x2, 2), y2=round(b.y2, 2),
            cx=round((b.x1 + b.x2) / 2, 2), cy=round((b.y1 + b.y2) / 2, 2),
            w=round(b.x2 - b.x1, 2), h=round(b.y2 - b.y1, 2))) for b in boxes]
        out.append(entry)
    TRAIN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_MANIFEST.write_text(json.dumps(out, indent=2))
    print(f"manifest -> {TRAIN_MANIFEST}  ({len(out)} frames, "
          f"{sum(len(e['boxes']) for e in out)} boxes)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract + single-class annotate YOLO training frames")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="video stems/paths (default: all not-yet-extracted data/*.mp4)")
    ap.add_argument("--per-clip", type=int, default=TRAIN_FRAMES_PER_CLIP,
                    help=f"frames to sample per video (default: {TRAIN_FRAMES_PER_CLIP})")
    ap.add_argument("--skip-extract", action="store_true",
                    help="re-annotate already-extracted frames (skip extraction)")
    ap.add_argument("--no-build", action="store_true",
                    help="do not rebuild the dataset after annotating")
    args = ap.parse_args()

    if args.skip_extract and TRAIN_MANIFEST.exists():
        records = json.loads(TRAIN_MANIFEST.read_text())
        if args.clips:
            keep = {_safe_stem(c.replace(".mp4", "")) for c in args.clips}
            records = [r for r in records if r["clip"] in keep]
    else:
        clips = _discover(args.clips)
        if not clips:
            raise SystemExit("No videos to extract. Add an .mp4 to data/ or use --skip-extract.")
        print(f"Extracting from {len(clips)} video(s): {[p.name for p in clips]}")
        records = []
        for clip in clips:
            records.extend(extract_clip(clip, args.per_clip))
        if not records:
            raise SystemExit("No frames extracted.")
        TRAIN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        TRAIN_MANIFEST.write_text(json.dumps(records, indent=2))
        print(f"Extracted {len(records)} frames total.\n")

    if not records:
        raise SystemExit("No frames to annotate.")
    print(f"Annotating {len(records)} frames (single class 'shape')...\n")
    annotate(records)

    if not args.no_build:
        from ld.detect.build_dataset import build_dataset
        print()
        build_dataset()


if __name__ == "__main__":
    main()
