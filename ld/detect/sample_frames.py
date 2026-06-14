"""Sample diverse, cursor-stripped frames across the t* clips.

For each clip we make two streaming passes:

  Pass 1 (metadata): per frame, record whether the white countdown shape is
    present (and its radius) and where the green GT crosshair is. From this we
    derive the *tracking window* -- the span after the countdown shape blends
    away, where the real shape is camouflaged (the hard case the detector must
    handle) and the GT crosshair is still available to pre-fill its box.

  Pass 2 (extraction): re-read the clip, and for the evenly-spaced chosen
    indices, strip the pointer and save the frame as a PNG.

The result is ``data/detect/frames/<clip>_<frame>.png`` plus a single
``manifest.json`` recording, per saved frame, the source clip, frame index, GT
crosshair (real-shape centre) and the clip's countdown radius. The annotator
uses that to pre-fill the real shape's bounding box.

Usage:
    python -m ld.detect.sample_frames                 # 2/clip, all t* clips
    python -m ld.detect.sample_frames --per-clip 3
    python -m ld.detect.sample_frames --inputs data/t1_cropped_trimmed.mp4 ...
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ld.config import DATA_DIR, DETECT_FRAMES_DIR, DETECT_MANIFEST
from ld.capture.video_source import VideoSource
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import find_cursor, strip_pointer

__all__ = ["SampleRecord", "sample_clip", "sample_clips"]


@dataclass
class SampleRecord:
    """One sampled frame -> one entry in the manifest."""

    image: str          # filename (relative to frames dir)
    clip: str           # source clip stem
    frame: int          # source frame index
    width: int
    height: int
    gt_x: float         # green-GT crosshair x (real-shape centre), or -1
    gt_y: float         # green-GT crosshair y, or -1
    radius: float       # countdown shape radius (px); -1 if countdown missed


@dataclass
class _FrameMeta:
    idx: int
    white_radius: float | None
    gt: tuple[float, float] | None


def _scan(src: VideoSource) -> list[_FrameMeta]:
    """Pass 1: cheap per-frame metadata (no images retained)."""
    metas: list[_FrameMeta] = []
    for idx, raw in src.frames():
        ws = detect_white_shape(raw)
        gt = find_cursor(raw)
        metas.append(_FrameMeta(idx, ws.radius if ws else None, gt))
    return metas


def _tracking_window(metas: list[_FrameMeta], miss_to_confirm: int = 3) -> tuple[int, float]:
    """Return (start_frame, radius): handoff index + median countdown radius.

    ``start_frame`` mirrors ``solve.solve_clip``: the first frame after the
    white countdown shape has been absent for ``miss_to_confirm`` frames.
    """
    radii: list[float] = []
    last_seen = -1
    misses = 0
    start_frame = 0
    confirmed = False
    for m in metas:
        if m.white_radius is not None:
            radii.append(m.white_radius)
            last_seen = m.idx
            misses = 0
        elif last_seen >= 0 and not confirmed:
            misses += 1
            if misses >= miss_to_confirm:
                start_frame = last_seen + 1
                confirmed = True
    if not confirmed and last_seen >= 0:
        start_frame = last_seen + 1
    radius = float(np.median(radii)) if radii else -1.0
    return start_frame, radius


def _choose_indices(metas: list[_FrameMeta], start_frame: int, n: int) -> list[int]:
    """Evenly-spaced frames in the tracking window that have a GT crosshair."""
    candidates = [m.idx for m in metas if m.idx >= start_frame and m.gt is not None]
    if not candidates:  # fall back to any GT frame if the window is degenerate
        candidates = [m.idx for m in metas if m.gt is not None]
    if not candidates:
        return []
    if n >= len(candidates):
        return candidates
    # Even spacing across the candidate span (diversity over time/pose).
    picks = np.linspace(0, len(candidates) - 1, n).round().astype(int)
    return sorted({candidates[i] for i in picks})


def sample_clip(path: str | Path, per_clip: int, out_dir: Path) -> list[SampleRecord]:
    path = Path(path)
    stem = path.stem.replace("_cropped_trimmed", "")
    src = VideoSource(path)
    metas = _scan(src)
    start_frame, radius = _tracking_window(metas)
    chosen = set(_choose_indices(metas, start_frame, per_clip))
    gt_by_idx = {m.idx: m.gt for m in metas}
    src.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[SampleRecord] = []
    src = VideoSource(path)
    for idx, raw in src.frames():
        if idx not in chosen:
            continue
        frame = strip_pointer(raw, strip_green=True)
        h, w = frame.shape[:2]
        name = f"{stem}_{idx:05d}.png"
        cv2.imwrite(str(out_dir / name), frame)
        gt = gt_by_idx.get(idx)
        records.append(SampleRecord(
            image=name, clip=stem, frame=idx, width=w, height=h,
            gt_x=gt[0] if gt else -1.0, gt_y=gt[1] if gt else -1.0,
            radius=radius,
        ))
    src.release()
    print(f"  {stem}: start_frame={start_frame} radius={radius:.1f}px "
          f"-> {len(records)} frames")
    return records


def sample_clips(inputs: list[str | Path], per_clip: int,
                 out_dir: Path = DETECT_FRAMES_DIR,
                 manifest_path: Path = DETECT_MANIFEST) -> list[SampleRecord]:
    out_dir = Path(out_dir)
    all_records: list[SampleRecord] = []
    for path in inputs:
        all_records.extend(sample_clip(path, per_clip, out_dir))

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([asdict(r) for r in all_records], indent=2))
    n_gt = sum(1 for r in all_records if r.gt_x >= 0)
    print(f"\nwrote {len(all_records)} frames ({n_gt} with GT prefill) "
          f"-> {out_dir}\nmanifest -> {manifest_path}")
    return all_records


def _default_inputs() -> list[Path]:
    return sorted(DATA_DIR.glob("t*_cropped_trimmed.mp4"),
                  key=lambda p: int(p.stem.split("_")[0][1:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample cursor-stripped frames for detector labelling")
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="clip paths (default: all data/t*_cropped_trimmed.mp4)")
    ap.add_argument("--per-clip", type=int, default=2, help="frames per clip")
    ap.add_argument("--out-dir", default=str(DETECT_FRAMES_DIR))
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs] if args.inputs else _default_inputs()
    if not inputs:
        raise SystemExit("No input clips found (looked for data/t*_cropped_trimmed.mp4).")
    print(f"sampling {args.per_clip}/clip from {len(inputs)} clips")
    sample_clips(inputs, args.per_clip, Path(args.out_dir))


if __name__ == "__main__":
    main()
