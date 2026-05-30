"""Phase 1: read video, draw HUD, write annotated debug MP4."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from ld.capture.video_source import VideoSource
from ld.config import DEFAULT_VIDEO, OUTPUT_DIR
from ld.control.mouse import MouseController
from ld.vision.overlay import draw_hud
from ld.vision.roi import apply_roi


def create_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer: {path}")
    return writer


def run(
    input_path: Path,
    output_path: Path,
    *,
    pre_cropped: bool = True,
    max_frames: int | None = None,
) -> dict[str, float]:
    mouse = MouseController(enabled=False, dry_run=True)
    del mouse  # unused until Phase 5; keeps import wired

    t0 = time.perf_counter()
    frames_written = 0

    with VideoSource(input_path) as src:
        meta = src.meta
        writer = create_writer(output_path, meta.width, meta.height, meta.fps)
        try:
            proc_start = time.perf_counter()
            for vf in src.frames():
                if max_frames is not None and vf.frame_idx >= max_frames:
                    break

                panel = apply_roi(vf.bgr, pre_cropped=pre_cropped)
                elapsed = time.perf_counter() - proc_start
                proc_fps = (vf.frame_idx + 1) / elapsed if elapsed > 0 else 0.0

                annotated = draw_hud(
                    panel,
                    frame_idx=vf.frame_idx,
                    timestamp=vf.timestamp,
                    total_frames=meta.frame_count,
                    processing_fps=proc_fps,
                )
                writer.write(annotated)
                frames_written += 1
        finally:
            writer.release()

    wall_s = time.perf_counter() - t0
    avg_fps = frames_written / wall_s if wall_s > 0 else 0.0
    return {
        "frames": float(frames_written),
        "wall_s": wall_s,
        "avg_fps": avg_fps,
        "source_fps": meta.fps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline passthrough with HUD overlay.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Input MP4 (default: data/t1_cropped_trimmed.mp4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=OUTPUT_DIR / "debug_run.mp4",
        help="Output annotated MP4",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (quick smoke test)",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Disable pre-cropped pass-through ROI (for raw gameplay later)",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    try:
        stats = run(
            args.input,
            args.output,
            pre_cropped=not args.full_frame,
            max_frames=args.max_frames,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(
        f"Done: {int(stats['frames'])} frames in {stats['wall_s']:.2f}s "
        f"({stats['avg_fps']:.1f} proc fps, source {stats['source_fps']:.0f} fps)"
    )
    if stats["avg_fps"] < 30:
        print("Warning: processing below 30 fps", file=sys.stderr)
    else:
        print("OK: processing >= 30 fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
