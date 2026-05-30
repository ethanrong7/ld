"""Step 7 diagnostic: full pipeline (countdown init -> seeded tracking).

Analyzes the countdown for the round init, then tracks the target post-handoff
and scores per-frame distance to the green-cursor GT. Writes an annotated video.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.vision.cursor import find_cursor
from ld.vision.motion import to_gray_f32
from ld.vision.template import analyze_round
from ld.vision.tracker import TargetTracker


def run(input_path: str, out_video: str | None) -> None:
    ri = analyze_round(input_path)
    print(f"{Path(input_path).name}: handoff={ri.handoff_frame} "
          f"omega={ri.omega_deg_per_frame:.2f}/f seed={ri.center_seed}")

    src = VideoSource(input_path)
    m = src.meta
    writer = open_writer(out_video, m.width, m.height, m.fps) if out_video else None
    tracker = None

    prev_gray = None
    start = ri.handoff_frame + 1
    errs: list[float] = []
    coast = 0
    for idx, frame in src.frames():
        if idx < start:
            prev_gray = to_gray_f32(frame)
            continue
        cur_gray = to_gray_f32(frame)
        if prev_gray is None:
            prev_gray = cur_gray
            continue
        if tracker is None:
            tracker = TargetTracker(ri, prev_gray)

        st = tracker.step(prev_gray, cur_gray)
        if not st.measured:
            coast += 1
        gt = find_cursor(frame)
        if gt:
            errs.append(math.hypot(st.x - gt[0], st.y - gt[1]))

        if writer is not None:
            vis = frame.copy()
            col = (0, 255, 0) if st.measured else (0, 165, 255)
            cv2.circle(vis, (int(st.x), int(st.y)), int(ri.template_radius), col, 2)
            cv2.drawMarker(vis, (int(st.x), int(st.y)), col, cv2.MARKER_CROSS, 16, 2)
            if gt:
                cv2.circle(vis, (int(gt[0]), int(gt[1])), 6, (0, 0, 255), 2)
            cv2.putText(vis, f"f{idx} {'TRK' if st.measured else 'coast'}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, f"f{idx} {'TRK' if st.measured else 'coast'}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(vis)
        prev_gray = cur_gray

    src.release()
    if writer is not None:
        writer.release()
        print(f"wrote -> {out_video}")
    if errs:
        a = np.array(errs)
        print(f"scored={len(a)} mean={a.mean():.1f} median={np.median(a):.1f} "
              f"p90={np.percentile(a,90):.1f} max={a.max():.1f} coast_frames={coast}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-video", default=None)
    args = ap.parse_args()
    run(args.input, args.out_video)


if __name__ == "__main__":
    main()
