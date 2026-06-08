"""Write a side-by-side video: original | strip_pointer (no overlays)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource, open_writer
from ld.config import OUTPUT_DIR
from ld.vision.cursor import find_cursor, strip_pointer


def run(input_path: str, out_video: str | None = None) -> Path:
    video = Path(input_path)
    out = Path(out_video) if out_video else OUTPUT_DIR / f"{video.stem}_stripped.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    src = VideoSource(str(video))
    m = src.meta
    writer = open_writer(str(out), m.width * 2, m.height, m.fps)
    found_orig = 0
    found_strip = 0

    for _idx, raw in src.frames():
        stripped = strip_pointer(raw)
        if find_cursor(raw) is not None:
            found_orig += 1
        if find_cursor(stripped) is not None:
            found_strip += 1
        panel = np.hstack([raw, stripped])
        cv2.putText(panel, "original", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "stripped", (m.width + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(panel)

    src.release()
    writer.release()
    print(f"{video.name}: {m.width}x{m.height} {m.fps:.1f}fps frames={m.frame_count}")
    print(f"find_cursor original={found_orig} stripped={found_strip}")
    print(f"wrote -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview pointer stripping on a clip.")
    ap.add_argument("--input", required=True, help="video path, e.g. data/t1_cropped_trimmed.mp4")
    ap.add_argument("--out-video", default=None)
    args = ap.parse_args()
    run(args.input, args.out_video)


if __name__ == "__main__":
    main()
