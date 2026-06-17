"""Render per-clip evidence videos using the field_coh tracker.

For each clip, runs field_coh (the current best identity mode), then writes
an mp4 overlay showing:
  - YOLO boxes (green rectangles)
  - tracker estimate (red circle = track, orange = coast, grey = acquire)
  - GT crosshair (cyan marker, visible when available)
  - HUD: frame index, tracker state, error px

Output: data/detect/evidence/<clip>_field_coh.mp4

Usage:
    python -m ld.detect.render_evidence --weights data/detect/runs/yolov8n_combined/weights/best.pt
    python -m ld.detect.render_evidence --weights .../best.pt --clips t1 t7
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2

from ld.config import DATA_DIR, DETECT_DIR
from ld.capture.video_source import VideoSource, open_writer
from ld.detect.fusion import detect_fusion_clip, FusionPack
from ld.detect.identity import (
    _dispatch_mode, compute_countdown_lock, _centroid,
)
from ld.detect.constellation import TrackPoint
from ld.detect.eval_modes import _frame_wh, _default_clips

EVIDENCE_DIR = DETECT_DIR / "evidence"

STATE_COLOR = {
    "track":   (0, 0, 255),    # red
    "coast":   (0, 140, 255),  # orange
    "acquire": (180, 180, 180),# grey
}


def render_clip(weights: str, clip: Path, *, conf: float = 0.25,
                imgsz: int = 768, mode: str = "field_coh") -> Path:
    clip = Path(clip)
    name = clip.stem.replace("_cropped_trimmed", "")
    print(f"[{name}] loading detections ...")
    packs = detect_fusion_clip(weights, clip, conf=conf, imgsz=imgsz, use_cache=True)
    lock  = compute_countdown_lock(packs, clip)
    wh    = _frame_wh(clip)

    print(f"[{name}] running {mode} ...")
    track, _hist, start, radius = _dispatch_mode(clip, packs, lock, wh, mode)
    tp_by_idx = {t.idx: t for t in track}
    gt_by_idx = {p.idx: p.gt for p in packs if p.gt is not None}

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_DIR / f"{name}_{mode}.mp4"
    src = VideoSource(clip)
    writer = open_writer(out, src.meta.width, src.meta.height, src.meta.fps or 30.0)

    for idx, frame in src.frames():
        p = packs[idx] if idx < len(packs) else None

        # Draw YOLO boxes  (FusionPack.boxes = (x1,y1,x2,y2,conf) tuples)
        if p is not None and p.boxes:
            for b in p.boxes:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw GT crosshair
        gt = gt_by_idx.get(idx)
        if gt is not None:
            cv2.drawMarker(frame, (int(gt[0]), int(gt[1])), (0, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 20, 2)

        # Draw tracker estimate
        tp = tp_by_idx.get(idx)
        err_str = ""
        if tp is not None and not math.isnan(tp.x):
            col = STATE_COLOR.get(tp.state, (180, 180, 180))
            cx, cy = int(tp.x), int(tp.y)
            cv2.circle(frame, (cx, cy), int(radius), col, 2)
            cv2.circle(frame, (cx, cy), 4, col, -1)
            if gt is not None:
                err = math.hypot(tp.x - gt[0], tp.y - gt[1])
                err_str = f"  err={err:.0f}px"
                # Line from tracker to GT when off
                if err > radius:
                    cv2.line(frame, (cx, cy), (int(gt[0]), int(gt[1])),
                             (0, 80, 255), 1)

        # HUD bar
        state_str = tp.state if tp else "---"
        hud = f"{name} f{idx:04d}  {mode}  [{state_str}]{err_str}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1)

        writer.write(frame)

    writer.release()
    src.release()
    print(f"  -> {out}")
    return out


def main() -> None:
    # Modes excluded from evidence renders — well below competitive threshold
    EXCLUDED_MODES = {"paper", "paper_outlier", "paper_outlier_rank", "chain"}

    ap = argparse.ArgumentParser(description="Render identity tracker evidence videos")
    ap.add_argument("--weights", default="data/detect/runs/yolov8n_single_combined/weights/best.pt")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip stems or paths (default: all t*_cropped_trimmed.mp4)")
    ap.add_argument("--mode", default="fpath",
                    help="identity mode to render (default: fpath)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=768)
    args = ap.parse_args()

    if args.clips:
        clips = []
        for c in args.clips:
            p = Path(c)
            if not p.exists():
                p = DATA_DIR / f"{c}_cropped_trimmed.mp4"
            clips.append(p)
    else:
        clips = _default_clips()

    if args.mode in EXCLUDED_MODES:
        raise SystemExit(f"Mode '{args.mode}' is excluded from evidence renders. "
                         f"Excluded: {sorted(EXCLUDED_MODES)}")

    for clip in clips:
        render_clip(args.weights, clip, conf=args.conf,
                    imgsz=args.imgsz, mode=args.mode)

    print(f"\nDone. Videos in {EVIDENCE_DIR}")


if __name__ == "__main__":
    main()
