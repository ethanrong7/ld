"""Phase 0: validate test clip inventory and metadata."""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

from ld.config import DATA_DIR, OUTPUT_DIR, TEST_VIDEOS_GLOB


@dataclass(frozen=True)
class VideoInfo:
    name: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def probe_video(path: Path) -> VideoInfo | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        name=path.name,
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
    )


def discover_test_videos(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob(TEST_VIDEOS_GLOB))


def save_sample_frames(videos: list[VideoInfo], out_dir: Path) -> None:
    """Write frame 0 and mid-frame thumbnails for visual sanity check."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for info in videos:
        cap = cv2.VideoCapture(str(info.path))
        if not cap.isOpened():
            continue
        mid = max(0, info.frame_count // 2)
        for idx, tag in ((0, "f000"), (mid, f"f{mid:04d}")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                stem = info.path.stem
                cv2.imwrite(str(out_dir / f"{stem}_{tag}.jpg"), frame)
        cap.release()


def print_report(rows: list[VideoInfo]) -> None:
    if not rows:
        print("No test videos found.", file=sys.stderr)
        return

    resolutions = {r.resolution for r in rows}
    fps_values = {round(r.fps, 3) for r in rows}

    print(f"Found {len(rows)} test clip(s) matching {TEST_VIDEOS_GLOB}\n")
    header = f"{'name':<28} {'resolution':<12} {'fps':>7} {'frames':>7} {'duration':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.name:<28} {r.resolution:<12} {r.fps:7.2f} "
            f"{r.frame_count:7d} {r.duration_s:8.2f}s"
        )

    print()
    if len(resolutions) == 1:
        print(f"Resolution: consistent ({resolutions.pop()})")
    else:
        print(f"Resolution: MIXED — {sorted(resolutions)}")

    if len(fps_values) == 1:
        print(f"FPS: consistent ({fps_values.pop()})")
    else:
        print(f"FPS: MIXED — {sorted(fps_values)}")

    durations = [r.duration_s for r in rows]
    print(f"Duration range: {min(durations):.2f}s – {max(durations):.2f}s")


def write_csv(rows: list[VideoInfo], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "width", "height", "fps", "frame_count", "duration_s", "path"])
        for r in rows:
            w.writerow(
                [r.name, r.width, r.height, r.fps, r.frame_count, f"{r.duration_s:.3f}", r.path]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe t*_cropped_trimmed test videos.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing test clips (default: data/)",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Save frame 0 and mid-frame JPGs to output/phase0_samples/",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=OUTPUT_DIR / "phase0_probe.csv",
        help="Write metadata CSV (default: output/phase0_probe.csv)",
    )
    args = parser.parse_args(argv)

    paths = discover_test_videos(args.data_dir)
    rows: list[VideoInfo] = []
    failed: list[str] = []
    for p in paths:
        info = probe_video(p)
        if info is None:
            failed.append(p.name)
        else:
            rows.append(info)

    print_report(rows)
    if failed:
        print(f"\nFailed to open: {', '.join(failed)}", file=sys.stderr)

    if args.csv and rows:
        write_csv(rows, args.csv)
        print(f"\nCSV: {args.csv}")

    if args.samples and rows:
        sample_dir = OUTPUT_DIR / "phase0_samples"
        save_sample_frames(rows, sample_dir)
        print(f"Samples: {sample_dir}/")

    return 1 if failed else 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
