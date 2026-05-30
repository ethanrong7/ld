"""Phase-0 diagnostic: per-frame white-shape + green-cursor GT to CSV/video.

Helps verify countdown duration, handoff frame, rotation behaviour, and the
target trajectory (via the GT cursor) before building the tracker.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import find_cursor


def run(input_path: str, out_video: str | None, out_csv: str | None,
        max_frames: int | None = None) -> None:
    src = VideoSource(input_path)
    m = src.meta
    print(f"{Path(input_path).name}: {m.width}x{m.height} {m.fps:.1f}fps "
          f"frames={m.frame_count} dur={m.duration:.2f}s")

    writer = None
    if out_video:
        writer = open_writer(out_video, m.width, m.height, m.fps)
    rows: list[dict] = []

    for idx, frame in src.frames(max_frames):
        ws = detect_white_shape(frame)
        cur = find_cursor(frame)
        rows.append({
            "frame": idx,
            "white": int(ws is not None),
            "white_area": f"{ws.area:.0f}" if ws else "",
            "white_cx": f"{ws.cx:.1f}" if ws else "",
            "white_cy": f"{ws.cy:.1f}" if ws else "",
            "white_angle_deg": f"{math.degrees(ws.angle):.1f}" if ws and not math.isnan(ws.angle) else "",
            "cur_x": f"{cur[0]:.1f}" if cur else "",
            "cur_y": f"{cur[1]:.1f}" if cur else "",
        })

        if writer is not None:
            vis = frame.copy()
            if ws is not None:
                x, y, w, h = ws.bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 1)
                cv2.circle(vis, (int(ws.cx), int(ws.cy)), 3, (0, 200, 255), -1)
            if cur is not None:
                cv2.circle(vis, (int(cur[0]), int(cur[1])), 6, (0, 0, 255), 2)
            cv2.putText(vis, f"f{idx} white={int(ws is not None)}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"f{idx} white={int(ws is not None)}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            writer.write(vis)

    src.release()
    if writer is not None:
        writer.release()
        print(f"wrote video -> {out_video}")

    # handoff = last frame of contiguous initial white-present run
    handoff = None
    for r in rows:
        if r["white"]:
            handoff = r["frame"]
        elif handoff is not None:
            break
    print(f"handoff (last white frame of opening run): {handoff}")
    if handoff is not None:
        print(f"  ~= {handoff / m.fps:.2f}s")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote csv -> {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()
    run(args.input, args.out_video, args.out_csv, args.max_frames)


if __name__ == "__main__":
    main()
