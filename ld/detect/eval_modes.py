"""Leaderboard harness: score every identity pick mode across the eval clips.

Runs entirely from the YOLO detection cache (``detect_fusion_clip`` with
``use_cache=True`` -- no inference); each mode is dispatched through the same
``identity._dispatch_mode`` the live tracker uses, so a mode behaves identically
here and in production. Per clip the detections + countdown lock are computed
once and every mode is run against them.

Outputs:
  * a per-(mode, clip) table of within_r / within_1.5r / median-px / oracle /
    conditional (via ``score_identity``);
  * per-frame trace CSVs under ``data/detect/eval/<clip>__<mode>.csv`` for drift
    forensics (gitignored);
  * a ranked, committed ``ld/detect/LEADERBOARD.md`` baseline of record.

Usage:
    python -m ld.detect.eval_modes \
        --weights data/detect/runs/yolov8n_combined/weights/best.pt

Pin ONE weights file: the cache key is an md5 of (path, mtime), so touching the
.pt invalidates the cache and silently changes every number.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from ld.capture.video_source import VideoSource
from ld.config import DATA_DIR, DETECT_DIR
from ld.detect.fusion import FusionPack, detect_fusion_clip
from ld.detect.identity import (ALL_MODES, LockInfo, TrackPoint, _centroid,
                                _dispatch_mode, compute_countdown_lock,
                                score_identity)

# A miss is only a "drift" once the estimate sits outside the shape radius for
# this many consecutive frames (filters single-frame blips from real loss).
DRIFT_RUN = 8
EVAL_DIR = DETECT_DIR / "eval"
DEFAULT_WEIGHTS = "data/detect/runs/yolov8n_combined/weights/best.pt"
LEADERBOARD_MD = Path(__file__).resolve().parent / "LEADERBOARD.md"


@dataclass
class ModeClipRow:
    mode: str
    clip: str
    within_r: float
    within_1p5r: float
    median_px: float
    oracle_within_r: float
    conditional: float
    drift_onset: int   # first frame idx (>= start) drifting for DRIFT_RUN; -1 = never
    drift_frac: float  # drift_onset as a fraction of scored span (1.0 = never drifts)
    n: int


def _default_clips() -> list[Path]:
    return sorted(DATA_DIR.glob("t*_cropped_trimmed.mp4"),
                  key=lambda p: int(p.stem.split("_")[0][1:]))


def _frame_wh(clip: Path) -> tuple[int, int]:
    src = VideoSource(clip)
    wh = (src.meta.width, src.meta.height)
    src.release()
    return wh


def _per_frame_errors(packs: list[FusionPack], track: list[TrackPoint],
                      start: int, radius: float) -> list[dict]:
    """Per-frame error rows mirroring score_identity's internal computation."""
    gt = {p.idx: p.gt for p in packs if p.gt is not None}
    rows: list[dict] = []
    for tp in track:
        if tp.idx < start:
            continue
        g = gt.get(tp.idx)
        if g is None:
            continue
        p = packs[tp.idx]
        oracle_err = float("nan")
        if p.boxes:
            oracle_err = min(math.hypot(_centroid(b)[0] - g[0], _centroid(b)[1] - g[1])
                             for b in p.boxes)
        err = float("nan") if math.isnan(tp.x) else math.hypot(tp.x - g[0], tp.y - g[1])
        rows.append({
            "idx": tp.idx,
            "state": tp.state,
            "x": f"{tp.x:.1f}", "y": f"{tp.y:.1f}",
            "gt_x": f"{g[0]:.1f}", "gt_y": f"{g[1]:.1f}",
            "err_px": "" if math.isnan(err) else f"{err:.1f}",
            "within_r": int(not math.isnan(err) and err < radius),
            "oracle_err": "" if math.isnan(oracle_err) else f"{oracle_err:.1f}",
            "oracle_hit": int(not math.isnan(oracle_err) and oracle_err < radius),
            "n_boxes": len(p.boxes),
        })
    return rows


def _drift_onset(rows: list[dict], radius: float) -> int:
    """First frame idx that begins a run of >= DRIFT_RUN consecutive out-of-radius
    frames; -1 if the track never drifts that long."""
    run = 0
    run_start = -1
    for r in rows:
        out = not r["within_r"]
        if out:
            if run == 0:
                run_start = r["idx"]
            run += 1
            if run >= DRIFT_RUN:
                return run_start
        else:
            run = 0
            run_start = -1
    return -1


def _write_trace_csv(clip_stem: str, mode: str, rows: list[dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"{clip_stem}__{mode}.csv"
    fields = ["idx", "state", "x", "y", "gt_x", "gt_y", "err_px",
              "within_r", "oracle_err", "oracle_hit", "n_boxes"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def eval_clip_all_modes(weights: str, clip: Path, modes: list[str], *,
                        conf: float, imgsz: int) -> list[ModeClipRow]:
    """Score every mode on one clip; detections + lock computed once."""
    packs = detect_fusion_clip(weights, clip, conf=conf, imgsz=imgsz, use_cache=True)
    lock = compute_countdown_lock(packs, clip)
    wh = _frame_wh(clip)
    name = clip.stem.replace("_cropped_trimmed", "")
    out_rows: list[ModeClipRow] = []
    for mode in modes:
        track, _hist, start, radius = _dispatch_mode(clip, packs, lock, wh, mode)
        rep = score_identity(packs, track, start, radius, name)
        frame_rows = _per_frame_errors(packs, track, start, radius)
        _write_trace_csv(clip.stem, mode, frame_rows)
        onset = _drift_onset(frame_rows, radius)
        span_lo = frame_rows[0]["idx"] if frame_rows else start
        span_hi = frame_rows[-1]["idx"] if frame_rows else start
        span = max(1, span_hi - span_lo)
        drift_frac = 1.0 if onset < 0 else (onset - span_lo) / span
        out_rows.append(ModeClipRow(
            mode=mode, clip=name,
            within_r=rep.within_r, within_1p5r=rep.within_1p5r,
            median_px=rep.median_px, oracle_within_r=rep.oracle_within_r,
            conditional=rep.conditional_within_r,
            drift_onset=onset, drift_frac=drift_frac, n=rep.n))
        print(f"  [{name:>3}] {mode:<18} within_r={rep.within_r:.3f} "
              f"median={rep.median_px:6.1f}px drift@{onset if onset >= 0 else '---'}")
    return out_rows


@dataclass
class ModeSummary:
    mode: str
    mean_within_r: float
    mean_within_1p5r: float
    mean_conditional: float
    median_px: float
    mean_drift_frac: float
    n_clips: int


def _summarize(rows: list[ModeClipRow], modes: list[str]) -> list[ModeSummary]:
    summaries: list[ModeSummary] = []
    for mode in modes:
        mr = [r for r in rows if r.mode == mode and r.n > 0]
        if not mr:
            continue
        summaries.append(ModeSummary(
            mode=mode,
            mean_within_r=statistics.mean(r.within_r for r in mr),
            mean_within_1p5r=statistics.mean(r.within_1p5r for r in mr),
            mean_conditional=statistics.mean(r.conditional for r in mr),
            median_px=statistics.median(r.median_px for r in mr),
            mean_drift_frac=statistics.mean(r.drift_frac for r in mr),
            n_clips=len(mr)))
    summaries.sort(key=lambda s: s.mean_within_r, reverse=True)
    return summaries


def _write_leaderboard(summaries: list[ModeSummary], rows: list[ModeClipRow],
                       clips: list[str], weights: str, conf: float, imgsz: int) -> None:
    lines: list[str] = []
    lines.append("# Identity mode leaderboard")
    lines.append("")
    lines.append(f"Weights: `{weights}` · conf={conf} · imgsz={imgsz} · "
                 f"clips: {', '.join(clips)}")
    lines.append("")
    lines.append("`within_r` = fraction of scored frames the estimate lands inside "
                 "the shape radius of GT. `drift_frac` = mean fraction of the scored "
                 "span before a sustained drift begins (1.0 = never drifts). Ranked "
                 "by mean `within_r`.")
    lines.append("")
    lines.append("| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |")
    lines.append("|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|")
    for i, s in enumerate(summaries, 1):
        lines.append(f"| {i} | `{s.mode}` | {s.mean_within_r:.3f} | "
                     f"{s.mean_within_1p5r:.3f} | {s.mean_conditional:.3f} | "
                     f"{s.median_px:.1f} | {s.mean_drift_frac:.2f} | {s.n_clips} |")
    lines.append("")
    # Per-clip within_r for the leader, so per-clip weak spots are visible.
    if summaries:
        best = summaries[0].mode
        lines.append(f"## Per-clip `within_r` — leader (`{best}`)")
        lines.append("")
        br = sorted((r for r in rows if r.mode == best), key=lambda r: r.clip)
        lines.append("| clip | within_r | median_px | oracle | drift_onset |")
        lines.append("|------|---------:|----------:|-------:|------------:|")
        for r in br:
            onset = "never" if r.drift_onset < 0 else str(r.drift_onset)
            lines.append(f"| {r.clip} | {r.within_r:.3f} | {r.median_px:.1f} | "
                         f"{r.oracle_within_r:.3f} | {onset} |")
        lines.append("")
    LEADERBOARD_MD.write_text("\n".join(lines))
    print(f"\nleaderboard -> {LEADERBOARD_MD}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score all identity modes -> leaderboard")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip stems or paths; default = all data/t*_cropped_trimmed.mp4")
    ap.add_argument("--modes", nargs="*", default=list(ALL_MODES),
                    help=f"subset of modes; default = all ({len(ALL_MODES)})")
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
    if not clips:
        raise SystemExit("No input clips found.")

    rows: list[ModeClipRow] = []
    for clip in clips:
        print(f"[{clip.stem}]")
        rows.extend(eval_clip_all_modes(args.weights, clip, args.modes,
                                        conf=args.conf, imgsz=args.imgsz))

    summaries = _summarize(rows, args.modes)
    clip_names = [c.stem.replace("_cropped_trimmed", "") for c in clips]
    print(f"\n{'='*72}\nLEADERBOARD (mean across {len(clip_names)} clips)\n{'='*72}")
    print(f"{'mode':<18} {'within_r':>9} {'within1.5r':>11} {'cond':>7} "
          f"{'med_px':>8} {'drift':>7}")
    for s in summaries:
        print(f"{s.mode:<18} {s.mean_within_r:>9.3f} {s.mean_within_1p5r:>11.3f} "
              f"{s.mean_conditional:>7.3f} {s.median_px:>8.1f} {s.mean_drift_frac:>7.2f}")

    _write_leaderboard(summaries, rows, clip_names, args.weights, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
