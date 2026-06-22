"""Regression test: LdOnlineTracker (streaming) reproduces batch fpath_human.

Replays each t1-t10 clip frame-by-frame through ``LdOnlineTracker``, feeding the
*cached* per-frame detections (so the box source is byte-identical to the offline
eval -- the online port adds streaming, not a re-detection). The accumulated track
is scored with the same ``score_identity`` and compared against:

  * the published ``fpath_human`` per-clip within_r (the eval CSVs in
    data/detect/eval/<clip>__fpath_human.csv), tolerance +-0.002, and
  * the CSV's per-frame (x, y) emitted points (exact-ish, tolerance 0.5 px),

asserting no per-clip regression. Run::

    python -m ld.detect.test_online --weights data/detect/runs/yolov8n_single_combined/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from ld.capture.video_source import VideoSource
from ld.config import DATA_DIR, DETECT_DIR
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import score_identity
from ld.detect.online import LdOnlineTracker

WR_TOL = 0.002      # within_r tolerance vs published fpath_human
XY_TOL = 0.5        # px tolerance for per-frame emitted point vs CSV


def _clips() -> list[Path]:
    return sorted(DATA_DIR.glob("t*_cropped_trimmed.mp4"),
                  key=lambda p: int(p.stem.split("_")[0][1:]))


def _read_csv(clip_stem: str) -> dict[int, tuple[float, float]]:
    """idx -> (x, y) from the offline fpath_human eval CSV (empty if absent)."""
    path = DETECT_DIR / "eval" / f"{clip_stem}__fpath_human.csv"
    if not path.exists():
        return {}
    out: dict[int, tuple[float, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[int(row["idx"])] = (float(row["x"]), float(row["y"]))
    return out


def _csv_within_r(clip_stem: str) -> float | None:
    path = DETECT_DIR / "eval" / f"{clip_stem}__fpath_human.csv"
    if not path.exists():
        return None
    hits = n = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            hits += int(row["within_r"])
            n += 1
    return hits / n if n else None


def run_clip_online(weights: str, clip: Path) -> tuple:
    """Stream the clip through LdOnlineTracker (cached detections) -> (report, track)."""
    packs = detect_fusion_clip(weights, clip, use_cache=True)
    by_idx = {p.idx: p for p in packs}
    trk = LdOnlineTracker(weights)
    src = VideoSource(clip)
    for idx, frame in src.frames():
        p = by_idx.get(idx)
        det = (p.white, p.boxes) if p is not None else (None, [])
        trk.push_frame(frame, detection=det)
    src.release()
    rep = score_identity(packs, trk.track, trk.start, trk.radius,
                         clip.stem.replace("_cropped_trimmed", ""))
    return rep, trk.track


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip stems e.g. t1 t4 (default: all t1-t10)")
    args = ap.parse_args()

    clips = _clips()
    if args.clips:
        want = set(args.clips)
        clips = [c for c in clips if c.stem.split("_")[0] in want]

    print(f"{'clip':<6}{'online_wr':>11}{'csv_wr':>9}{'d_wr':>8}{'xy_maxdiff':>12}{'status':>10}")
    print("-" * 56)
    wrs: list[float] = []
    ok = True
    for clip in clips:
        rep, track = run_clip_online(args.weights, clip)
        stem = clip.stem
        short = stem.split("_")[0]
        csv_wr = _csv_within_r(stem)
        csv_xy = _read_csv(stem)
        maxdiff = 0.0
        for tp in track:
            if tp.idx in csv_xy and not math.isnan(tp.x):
                cx, cy = csv_xy[tp.idx]
                maxdiff = max(maxdiff, abs(tp.x - cx), abs(tp.y - cy))
        wrs.append(rep.within_r)
        d_wr = (rep.within_r - csv_wr) if csv_wr is not None else float("nan")
        clip_ok = True
        if csv_wr is not None:
            clip_ok = abs(d_wr) <= WR_TOL and maxdiff <= XY_TOL
        ok = ok and clip_ok
        cw = f"{csv_wr:.3f}" if csv_wr is not None else "  n/a"
        dw = f"{d_wr:+.4f}" if csv_wr is not None else "   n/a"
        print(f"{short:<6}{rep.within_r:>11.3f}{cw:>9}{dw:>8}{maxdiff:>12.3f}"
              f"{'PASS' if clip_ok else 'FAIL':>10}")

    mean = sum(wrs) / len(wrs) if wrs else 0.0
    print("-" * 56)
    print(f"{'MEAN':<6}{mean:>11.3f}")
    print(f"\n{'ALL CLIPS PASS' if ok else 'REGRESSION DETECTED'} "
          f"(within_r tol +-{WR_TOL}, per-frame xy tol {XY_TOL}px)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
