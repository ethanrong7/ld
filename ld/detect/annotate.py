"""Minimal OpenCV box annotator with GT-prefilled real-shape boxes.

No external labelling tool required. For each sampled frame the real shape's
box is pre-drawn from the green-GT crosshair + countdown radius (recorded in
the manifest); you only draw the *fake* shapes' boxes and, if needed, nudge the
prefilled one. Labels are written in YOLO format alongside the frames.

Controls:
    left-drag     draw a new box
    u             undo last box
    c             clear all boxes on this frame
    r             reset this frame to the GT prefill
    n / SPACE     save + next frame
    p             save + previous frame
    s             save current frame
    q / ESC       save + quit

Saved as ``data/detect/labels/<image>.txt`` (one class: "shape", id 0).

Usage:
    python -m ld.detect.annotate
    python -m ld.detect.annotate --only t3 t7    # filter by clip stem
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from ld.config import (
    DETECT_FRAMES_DIR,
    DETECT_LABELS_DIR,
    DETECT_MANIFEST,
    DETECT_PREFILL_BOX_SCALE,
)
from ld.detect.boxes import Box, load_labels, prefill_box, save_labels

WINDOW = "LD annotate  [drag=box  u=undo  c=clear  r=reset  n/space=next  p=prev  q=quit]"
PREFILL_COLOR = (0, 215, 255)   # amber: came from GT prefill
DRAWN_COLOR = (0, 255, 0)       # green: human-drawn
LIVE_COLOR = (255, 160, 0)      # blue: box being dragged


class _State:
    def __init__(self) -> None:
        self.boxes: list[Box] = []
        self.drawing = False
        self.x0 = self.y0 = 0
        self.x1 = self.y1 = 0


def _label_path(image: str) -> Path:
    return DETECT_LABELS_DIR / f"{Path(image).stem}.txt"


def _initial_boxes(rec: dict, img_w: int, img_h: int) -> list[Box]:
    """Existing labels if present, else a single GT-prefilled real-shape box."""
    lp = _label_path(rec["image"])
    if lp.exists():
        return load_labels(lp, img_w, img_h)
    pf = prefill_box(rec["gt_x"], rec["gt_y"], rec["radius"], DETECT_PREFILL_BOX_SCALE)
    return [pf] if pf is not None else []


def _draw(base, st: _State, rec: dict, n_done: int, n_total: int):
    img = base.copy()
    for b in st.boxes:
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), DRAWN_COLOR, 2)
    if st.drawing:
        cv2.rectangle(img, (st.x0, st.y0), (st.x1, st.y1), LIVE_COLOR, 1)
    hud = f"{rec['clip']} f{rec['frame']}  boxes={len(st.boxes)}  [{n_done+1}/{n_total}]"
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def annotate(records: list[dict]) -> None:
    DETECT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    st = _State()

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            st.drawing = True
            st.x0, st.y0 = st.x1, st.y1 = x, y
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
        img_path = DETECT_FRAMES_DIR / rec["image"]
        base = cv2.imread(str(img_path))
        if base is None:
            print(f"skip (missing image): {img_path}")
            i += 1
            continue
        h, w = base.shape[:2]
        st.boxes = _initial_boxes(rec, w, h)

        while True:
            cv2.imshow(WINDOW, _draw(base, st, rec, i, len(records)))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("n"), ord(" ")):
                save_labels(_label_path(rec["image"]), st.boxes, w, h)
                i += 1
                break
            if key == ord("p"):
                save_labels(_label_path(rec["image"]), st.boxes, w, h)
                i = max(0, i - 1)
                break
            if key == ord("s"):
                save_labels(_label_path(rec["image"]), st.boxes, w, h)
                print(f"saved {rec['image']} ({len(st.boxes)} boxes)")
            elif key == ord("u") and st.boxes:
                st.boxes.pop()
            elif key == ord("c"):
                st.boxes = []
            elif key == ord("r"):
                pf = prefill_box(rec["gt_x"], rec["gt_y"], rec["radius"],
                                 DETECT_PREFILL_BOX_SCALE)
                st.boxes = [pf] if pf is not None else []
            elif key in (ord("q"), 27):
                save_labels(_label_path(rec["image"]), st.boxes, w, h)
                cv2.destroyAllWindows()
                print(f"stopped at {i+1}/{len(records)}")
                return

    cv2.destroyAllWindows()
    print(f"done: labelled {len(records)} frames -> {DETECT_LABELS_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="GT-prefilled box annotator")
    ap.add_argument("--manifest", default=str(DETECT_MANIFEST))
    ap.add_argument("--only", nargs="*", default=None, help="filter by clip stem(s)")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise SystemExit(f"No manifest at {manifest}. Run ld.detect.sample_frames first.")
    records = json.loads(manifest.read_text())
    if args.only:
        keep = set(args.only)
        records = [r for r in records if r["clip"] in keep]
    if not records:
        raise SystemExit("No frames to annotate.")
    annotate(records)


if __name__ == "__main__":
    main()
