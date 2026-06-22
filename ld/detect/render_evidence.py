"""Render per-clip evidence videos for an identity mode.

For each clip, runs the chosen identity mode (default fpath_human, the leader),
then writes an mp4 overlay showing:
  - YOLO boxes (green rectangles)
  - the EMITTED POINT as a single prominent filled red dot at (x,y) -- the (x,y)
    the player's cursor would trace
  - GT crosshair (cyan marker, GT-only diagnostic) + red miss-line when off
  - HUD: frame index, tracker state, error px

Output: data/detect/evidence/<clip>_<mode>.mp4

Usage:
    python -m ld.detect.render_evidence --weights data/detect/runs/yolov8n_single_combined/weights/best.pt
    python -m ld.detect.render_evidence --weights .../best.pt --clips t1 t7 --mode fpath_human
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


def render_clip(weights: str, clip: Path, *, conf: float = 0.25,
                imgsz: int = 768, mode: str = "fpath_human") -> Path:
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
            cx, cy = int(tp.x), int(tp.y)
            # Emitted point: a single prominent filled red dot (the (x,y) the
            # cursor should trace) with a thin dark outline for contrast.
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 0), 1)
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
    from ld.detect.identity import ALL_MODES

    ap = argparse.ArgumentParser(description="Render identity tracker evidence videos")
    ap.add_argument("--weights", default="data/detect/runs/yolov8n_single_combined/weights/best.pt")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip stems or paths (default: all t*_cropped_trimmed.mp4)")
    ap.add_argument("--mode", default="fpath_human", choices=ALL_MODES,
                    help="identity mode to render (default: fpath_human, the leader)")
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

    for clip in clips:
        render_clip(args.weights, clip, conf=args.conf,
                    imgsz=args.imgsz, mode=args.mode)

    print(f"\nDone. Videos in {EVIDENCE_DIR}")


if __name__ == "__main__":
    main()
