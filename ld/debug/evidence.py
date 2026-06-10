"""Render an overlay video explaining the rigid-outlier signal.

  green dots  - features moving WITH the rigid sheet (RANSAC inliers: fakes/bg)
  red dots    - features DISAGREEING with the sheet motion (the real shape)
  heatmap     - independent-motion saliency built from the outliers
  cyan ring   - tracker estimate (cursor-free)
  yellow mark - green-crosshair GT, drawn for reference ONLY

Frames are cursor-stripped with the project's strip_pointer.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.config import GATE_RADIUS
from ld.track import OutlierTracker
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import find_cursor, strip_pointer
from ld.vision.motion import estimate_motion, saliency_map


def run(input_path: str, out_video: str | None = None) -> Path:
    from ld.config import OUTPUT_DIR

    video = Path(input_path)
    out = Path(out_video) if out_video else OUTPUT_DIR / "debug" / f"{video.stem}_evidence.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    src = VideoSource(str(video))
    m = src.meta
    writer = open_writer(str(out), m.width, m.height, m.fps)

    prev_gray: np.ndarray | None = None
    tracker: OutlierTracker | None = None
    last_white = None
    radii: list[float] = []
    misses = 0
    started = False

    for idx, raw in src.frames():
        frame = strip_pointer(raw)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        gt = find_cursor(raw)

        if not started:
            ws = detect_white_shape(frame)
            if ws is not None:
                last_white = ws
                radii.append(ws.radius)
                misses = 0
            elif last_white is not None:
                misses += 1
                if misses >= 3:
                    gate = max(GATE_RADIUS, 1.3 * float(np.median(radii)))
                    tracker = OutlierTracker(last_white.cx, last_white.cy, gate_radius=gate)
                    started = True
            if last_white is not None:
                cv2.circle(vis, (int(last_white.cx), int(last_white.cy)),
                           int(max(8, last_white.radius)), (255, 255, 0), 2)
            _annotate(vis, gt, m.width, "acquiring (countdown)")
            writer.write(vis)
            prev_gray = gray
            continue

        if prev_gray is not None:
            field = estimate_motion(prev_gray, gray)
            sal = saliency_map(field, gray.shape)
            tp = tracker.update(idx, sal)

            if sal.max() > 1e-6:
                sn = (sal / sal.max() * 255).astype(np.uint8)
                vis = cv2.addWeighted(vis, 1.0, cv2.applyColorMap(sn, cv2.COLORMAP_JET), 0.45, 0)
            for (x, y) in field.inliers:
                cv2.circle(vis, (int(x), int(y)), 1, (0, 200, 0), -1)
            for (x, y) in field.outliers:
                cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.circle(vis, (int(tp.x), int(tp.y)), 18, (255, 255, 0), 2)
            cv2.putText(vis, "EST", (int(tp.x) + 20, int(tp.y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
            _annotate(vis, gt, m.width, f"tracking [{tp.state}] conf={tp.confidence:.0f}")
            writer.write(vis)
        prev_gray = gray

    src.release()
    writer.release()
    print(f"wrote -> {out}")
    return out


def _annotate(vis, gt, width: int, status: str) -> None:
    if gt is not None:
        cv2.drawMarker(vis, (int(gt[0]), int(gt[1])), (0, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.putText(vis, "GT (ref only)", (int(gt[0]) + 12, int(gt[1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (width, 38), (0, 0, 0), -1)
    cv2.putText(vis, "green=with-paper  red=independent(real)  heat=saliency  cyan=estimate",
                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, status, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
