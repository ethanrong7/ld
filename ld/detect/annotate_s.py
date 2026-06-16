"""Annotator for s1..s12 training clips (post-countdown frames, cursor stripped).

Picks 5 evenly-spaced frames per clip from the post-countdown window, strips
the green cursor, saves them as PNGs, then opens an interactive drag-to-draw
box annotator. All shapes are class 0 ("shape"). The centroid of each box is
auto-calculated as its geometric centre.

On exit writes:
  data/detect/s_frames/<clip>_<frame>.png          — cursor-stripped frames
  data/detect/s_labels/<stem>.txt                  — YOLO labels (per frame)
  data/detect/s_manifest.json                      — box + centroid metadata

Controls:
  left-drag     draw a new box
  u             undo last box
  c             clear all boxes on this frame
  n / SPACE     save + next frame
  p             save + previous frame
  s             save current frame without advancing
  q / ESC       save + quit

Usage:
  python -m ld.detect.annotate_s
  python -m ld.detect.annotate_s --clips s1 s3 s7   # subset
  python -m ld.detect.annotate_s --per-clip 3
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ld.config import DATA_DIR
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import strip_pointer
from ld.detect.boxes import Box, load_labels, save_labels

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
S_FRAMES_DIR = DATA_DIR / "detect" / "s_frames"
S_LABELS_DIR = DATA_DIR / "detect" / "s_labels"
S_MANIFEST   = DATA_DIR / "detect" / "s_manifest.json"

WINDOW = "LD annotate-s  [drag=box  u=undo  c=clear  n/space=next  p=prev  s=save  q=quit]"
DRAWN_COLOR = (0, 255, 0)    # green
LIVE_COLOR  = (255, 160, 0)  # blue (mid-draw)
CENTROID_COLOR = (0, 0, 255) # red dot


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def _has_overlay(frame: np.ndarray, blob_thresh: int = 50) -> bool:
    """True if the frame still has a large bright blob (countdown/START overlay)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max((cv2.contourArea(c) for c in cnts), default=0)
    return max_area >= blob_thresh


def _post_countdown_start(cap: cv2.VideoCapture, miss_confirm: int = 3) -> int:
    """Return the first frame after the countdown shape AND any overlay are gone."""
    last_seen = -1
    misses = 0
    confirmed = False
    start = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ws = detect_white_shape(frame)
        if ws is not None:
            last_seen = idx
            misses = 0
        elif last_seen >= 0 and not confirmed:
            misses += 1
            if misses >= miss_confirm:
                start = last_seen + 1
                confirmed = True
        idx += 1
    if not confirmed and last_seen >= 0:
        start = last_seen + 1

    # Walk forward from `start` until the overlay (START text, flash) is also gone.
    # Require 3 consecutive clean frames so we don't stop on a single dark frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    clean_streak = 0
    idx = start
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if _has_overlay(frame):
            clean_streak = 0
        else:
            clean_streak += 1
            if clean_streak >= 3:
                # Back up to the first clean frame of the streak
                return idx - 2
        idx += 1
    return start


def _pick_indices(total_frames: int, start: int, n: int) -> list[int]:
    """n evenly-spaced indices in [start, total_frames)."""
    available = total_frames - start
    if available <= 0:
        # Fallback: sample from the second half of the video
        start = total_frames // 2
        available = total_frames - start
    if available <= 0:
        return []
    if n >= available:
        return list(range(start, total_frames))
    picks = np.linspace(start, total_frames - 1, n).round().astype(int)
    return sorted(set(picks.tolist()))


def extract_frames(clip_path: Path, per_clip: int, out_dir: Path) -> list[dict]:
    """Extract, cursor-strip, and save frames. Returns list of record dicts."""
    stem = clip_path.stem  # e.g. "s1"

    # Pass 1: find countdown end + count frames
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"  WARNING: cannot open {clip_path}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = _post_countdown_start(cap, miss_confirm=3)
    cap.release()

    indices = set(_pick_indices(total, start, per_clip))
    print(f"  {stem}: total={total} post_countdown_start={start} "
          f"picking={sorted(indices)}")

    # Pass 2: extract chosen frames
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip_path))
    records: list[dict] = []
    idx = 0
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        if idx in indices:
            frame = strip_pointer(raw, strip_green=True)
            h, w = frame.shape[:2]
            name = f"{stem}_{idx:05d}.png"
            cv2.imwrite(str(out_dir / name), frame)
            records.append({"image": name, "clip": stem, "frame": idx,
                            "width": w, "height": h})
        idx += 1
    cap.release()
    return records


# ---------------------------------------------------------------------------
# Annotator state
# ---------------------------------------------------------------------------

class _State:
    def __init__(self) -> None:
        self.boxes: list[Box] = []
        self.drawing = False
        self.x0 = self.y0 = 0
        self.x1 = self.y1 = 0


def _label_path(image_name: str) -> Path:
    return S_LABELS_DIR / f"{Path(image_name).stem}.txt"


def _load_boxes(image_name: str, w: int, h: int) -> list[Box]:
    lp = _label_path(image_name)
    if lp.exists():
        return load_labels(lp, w, h)
    return []


def _draw_frame(base: np.ndarray, st: _State, rec: dict,
                n_done: int, n_total: int) -> np.ndarray:
    img = base.copy()
    for b in st.boxes:
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
                      DRAWN_COLOR, 2)
        # Centroid dot
        cx = int((b.x1 + b.x2) / 2)
        cy = int((b.y1 + b.y2) / 2)
        cv2.circle(img, (cx, cy), 4, CENTROID_COLOR, -1)
    if st.drawing:
        cv2.rectangle(img, (st.x0, st.y0), (st.x1, st.y1), LIVE_COLOR, 1)
        # Live centroid preview
        lx = (st.x0 + st.x1) // 2
        ly = (st.y0 + st.y1) // 2
        cv2.circle(img, (lx, ly), 4, CENTROID_COLOR, 1)
    hud = (f"{rec['clip']} f{rec['frame']}  boxes={len(st.boxes)}"
           f"  [{n_done+1}/{n_total}]")
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    return img


# ---------------------------------------------------------------------------
# Main annotate loop
# ---------------------------------------------------------------------------

def annotate(records: list[dict]) -> None:
    S_LABELS_DIR.mkdir(parents=True, exist_ok=True)
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
        img_path = S_FRAMES_DIR / rec["image"]
        base = cv2.imread(str(img_path))
        if base is None:
            print(f"  skip (missing image): {img_path}")
            i += 1
            continue
        h, w = base.shape[:2]
        st.boxes = _load_boxes(rec["image"], w, h)

        while True:
            cv2.imshow(WINDOW, _draw_frame(base, st, rec, i, len(records)))
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("n"), ord(" ")):
                save_labels(_label_path(rec["image"]), st.boxes, w, h)
                i += 1
                break
            elif key == ord("p"):
                save_labels(_label_path(rec["image"]), st.boxes, w, h)
                i = max(0, i - 1)
                break
            elif key == ord("s"):
                save_labels(_label_path(rec["image"]), st.boxes, w, h)
                print(f"  saved {rec['image']} ({len(st.boxes)} boxes)")
            elif key == ord("u") and st.boxes:
                st.boxes.pop()
            elif key == ord("c"):
                st.boxes = []
            elif key in (ord("q"), 27):
                save_labels(_label_path(rec["image"]), st.boxes, w, h)
                cv2.destroyAllWindows()
                print(f"  stopped at frame {i+1}/{len(records)}")
                _write_manifest(records)
                return

    cv2.destroyAllWindows()
    _write_manifest(records)
    print(f"done: labelled {len(records)} frames -> {S_LABELS_DIR}")


# ---------------------------------------------------------------------------
# Manifest (JSON with box + centroid info)
# ---------------------------------------------------------------------------

@dataclass
class BoxRecord:
    cls: int
    x1: float; y1: float; x2: float; y2: float
    cx: float; cy: float  # centroid
    w: float; h: float
    cx_norm: float; cy_norm: float  # normalised (YOLO)
    w_norm: float; h_norm: float


def _write_manifest(records: list[dict]) -> None:
    """Rebuild manifest from the saved .txt labels, including centroid fields."""
    out: list[dict] = []
    for rec in records:
        lp = _label_path(rec["image"])
        img_w, img_h = rec["width"], rec["height"]
        boxes_raw = load_labels(lp, img_w, img_h) if lp.exists() else []
        box_list = []
        for b in boxes_raw:
            cx = (b.x1 + b.x2) / 2
            cy = (b.y1 + b.y2) / 2
            bw = b.x2 - b.x1
            bh = b.y2 - b.y1
            box_list.append(asdict(BoxRecord(
                cls=b.cls,
                x1=round(b.x1, 2), y1=round(b.y1, 2),
                x2=round(b.x2, 2), y2=round(b.y2, 2),
                cx=round(cx, 2), cy=round(cy, 2),
                w=round(bw, 2), h=round(bh, 2),
                cx_norm=round(cx / img_w, 6),
                cy_norm=round(cy / img_h, 6),
                w_norm=round(bw / img_w, 6),
                h_norm=round(bh / img_h, 6),
            )))
        entry = dict(rec)
        entry["boxes"] = box_list
        out.append(entry)

    S_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    S_MANIFEST.write_text(json.dumps(out, indent=2))
    total_boxes = sum(len(e["boxes"]) for e in out)
    print(f"manifest -> {S_MANIFEST}  ({len(out)} frames, {total_boxes} boxes)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _find_clips(clip_names: list[str] | None) -> list[Path]:
    if clip_names:
        paths = []
        for name in clip_names:
            stem = name if name.endswith(".mp4") else name + ".mp4"
            p = DATA_DIR / stem
            if not p.exists():
                print(f"  WARNING: {p} not found, skipping")
            else:
                paths.append(p)
        return paths
    found = sorted(DATA_DIR.glob("s[0-9]*.mp4"),
                   key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "0"))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract + annotate post-countdown frames from s1..s12 clips")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip stems to process (default: all s*.mp4 in data/)")
    ap.add_argument("--per-clip", type=int, default=5,
                    help="frames to sample per clip (default: 5)")
    ap.add_argument("--skip-extract", action="store_true",
                    help="skip frame extraction (re-annotate already-extracted frames)")
    args = ap.parse_args()

    clips = _find_clips(args.clips)
    if not clips:
        raise SystemExit("No s*.mp4 clips found in data/. "
                         "Use --clips s1 s2 ... to specify paths.")

    print(f"Found {len(clips)} clip(s): {[p.name for p in clips]}")

    # --- Extract frames -------------------------------------------------------
    all_records: list[dict] = []

    if args.skip_extract and S_MANIFEST.exists():
        print("--skip-extract: loading existing manifest")
        all_records = json.loads(S_MANIFEST.read_text())
        # Filter to requested clips if specified
        if args.clips:
            keep = set(args.clips)
            all_records = [r for r in all_records if r["clip"] in keep]
    else:
        S_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        for clip_path in clips:
            recs = extract_frames(clip_path, args.per_clip, S_FRAMES_DIR)
            all_records.extend(recs)
        if not all_records:
            raise SystemExit("No frames extracted. Check clip paths.")
        # Write a pre-manifest (boxes added after annotation)
        S_MANIFEST.write_text(json.dumps(all_records, indent=2))
        print(f"Extracted {len(all_records)} frames total.")

    if not all_records:
        raise SystemExit("No frames to annotate.")

    print(f"\nStarting annotator ({len(all_records)} frames)...")
    print("Controls: drag=draw box  u=undo  c=clear  n/space=next  p=prev  s=save  q=quit\n")
    annotate(all_records)


if __name__ == "__main__":
    main()
