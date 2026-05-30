"""Visualize background-compensated motion residual vs GT cursor.

Post-handoff, estimate the background shift each frame, build the residual map,
take its dominant blob as the candidate target, and compare to the green-cursor
ground truth. Writes a heatmap-overlay video and reports localization error.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.vision.cursor import find_cursor, strip_pointer
from ld.vision.motion import estimate_translation, residual_map, to_gray_f32


def residual_peak(resid: np.ndarray, thresh: int = 60) -> tuple[float, float] | None:
    _, mask = cv2.threshold(resid, thresh, 255, cv2.THRESH_BINARY)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (float(cents[k][0]), float(cents[k][1]))


def run(
    input_path: str,
    out_video: str,
    start: int,
    ema: float = 0.6,
    *,
    strip_pointer_from_frames: bool = True,
) -> None:
    src = VideoSource(input_path)
    m = src.meta
    writer = open_writer(out_video, m.width, m.height, m.fps)
    window = cv2.createHanningWindow((m.width, m.height), cv2.CV_32F)

    prev_gray = None
    acc = None
    errs: list[float] = []
    for idx, frame in src.frames():
        raw = frame
        if strip_pointer_from_frames:
            frame = strip_pointer(frame)
        if idx < start:
            prev_gray = to_gray_f32(frame)
            continue
        cur_gray = to_gray_f32(frame)
        if prev_gray is None:
            prev_gray = cur_gray
            continue

        dx, dy, resp = estimate_translation(prev_gray, cur_gray, window)
        resid = residual_map(prev_gray, cur_gray, dx, dy)
        acc = resid.astype(np.float32) if acc is None else (ema * acc + (1 - ema) * resid)
        acc_u8 = acc.astype(np.uint8)

        peak = residual_peak(acc_u8)
        gt = find_cursor(raw)
        if peak and gt:
            errs.append(math.hypot(peak[0] - gt[0], peak[1] - gt[1]))

        heat = cv2.applyColorMap(acc_u8, cv2.COLORMAP_JET)
        vis = cv2.addWeighted(raw, 0.6, heat, 0.4, 0)
        if peak:
            cv2.drawMarker(vis, (int(peak[0]), int(peak[1])), (255, 255, 255),
                           cv2.MARKER_CROSS, 18, 2)
        if gt:
            cv2.circle(vis, (int(gt[0]), int(gt[1])), 7, (0, 0, 255), 2)
        cv2.putText(vis, f"f{idx} bg=({dx:+.1f},{dy:+.1f})", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"f{idx} bg=({dx:+.1f},{dy:+.1f})", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(vis)
        prev_gray = cur_gray

    src.release()
    writer.release()
    if errs:
        arr = np.array(errs)
        print(f"frames scored={len(arr)} mean_err={arr.mean():.1f}px "
              f"median={np.median(arr):.1f}px p90={np.percentile(arr,90):.1f}px")
    print(f"wrote -> {out_video}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-video", required=True)
    ap.add_argument("--start", type=int, default=117)
    ap.add_argument(
        "--no-strip-pointer",
        action="store_true",
        help="disable green/mouse inpaint before motion residual (ablation only)",
    )
    args = ap.parse_args()
    run(
        args.input,
        args.out_video,
        args.start,
        strip_pointer_from_frames=not args.no_strip_pointer,
    )


if __name__ == "__main__":
    main()
