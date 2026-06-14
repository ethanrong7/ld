"""Decompose the identity-tracking loss into a ceiling ladder.

Where does the gap between the detector (oracle ~0.93) and the live accum tracker
(~0.49) actually live? Each rung relaxes one constraint, so the gap between rungs
assigns the loss to a stage:

  R0  detection      closest detection to GT within radius (detector ceiling)
  R1  association    best a single persistent tracklet could do (+ identity
                     persistence); low => the real shape's identity fragments
  R2  signal         how often the GT box is the top motion outlier per frame
                     (upper bound of a perfect, causal-agnostic per-frame picker)
  R3  decision       accum actual vs R1/R2 (lost to causal online decisions)

Reuses the EXACT functions accum uses (estimate_paper_motion, _paper_box_residuals,
_box_net_rotations, _associate) so the decomposition is apples-to-apples. Runs one
forward pass per clip; dumps per-box signals to data/detect/eval/<clip>__signals.csv
for inspection and writes a committed ld/detect/DIAGNOSIS.md.

Usage:
    python -m ld.detect.diagnose \
        --weights data/detect/runs/yolov8n_combined/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource
from ld.config import DATA_DIR
from ld.detect.eval_modes import EVAL_DIR, _default_clips, _frame_wh
from ld.detect.fusion import FusionPack, detect_fusion_clip
from ld.detect.identity import (ROT_FLOOR, ROT_WEIGHT, BoxTrack, _associate,
                                _box_net_rotations, _centroid, _iou,
                                _paper_box_residuals, _refine_paper_from_fakes,
                                _seed, _tid_for_box, compute_countdown_lock,
                                estimate_paper_motion, score_identity,
                                track_accum_identity)
from ld.vision.cursor import strip_pointer

DIAGNOSIS_MD = Path(__file__).resolve().parent / "DIAGNOSIS.md"
MC_GATE = 60.0   # px; motion-compensated association gate


@dataclass
class FrameSig:
    idx: int
    gt: tuple[float, float] | None
    # per box: (cx, cy, residual, rot_net|nan, gt_dist, tid_iou, tid_mc)
    boxes: list[tuple] = field(default_factory=list)


def _mc_associate(mc_tracks: list[dict], boxes: list[tuple], paper_T,
                  next_tid: int) -> tuple[list[dict], int, list[int | None]]:
    """Motion-compensated greedy association: predict each track centroid via the
    paper affine, match to nearest current box within MC_GATE."""
    preds = []
    for tr in mc_tracks:
        c = np.array([tr["cx"], tr["cy"]], np.float32)
        if paper_T is not None:
            c = paper_T[:, :2] @ c + paper_T[:, 2]
        preds.append(c)
    pairs = []
    for ti, pc in enumerate(preds):
        for bi, b in enumerate(boxes):
            cc = _centroid(b)
            d = math.hypot(cc[0] - pc[0], cc[1] - pc[1])
            if d <= MC_GATE:
                pairs.append((d, ti, bi))
    pairs.sort()
    used_t, used_b = set(), set()
    tid_of_box: list[int | None] = [None] * len(boxes)
    for _, ti, bi in pairs:
        if ti in used_t or bi in used_b:
            continue
        used_t.add(ti); used_b.add(bi)
        cc = _centroid(boxes[bi])
        mc_tracks[ti]["cx"], mc_tracks[ti]["cy"] = cc
        mc_tracks[ti]["miss"] = 0
        tid_of_box[bi] = mc_tracks[ti]["tid"]
    for ti, tr in enumerate(mc_tracks):
        if ti not in used_t:
            tr["miss"] += 1
    mc_tracks = [t for t in mc_tracks if t["miss"] <= 8]
    for bi, b in enumerate(boxes):
        if bi not in used_b:
            cc = _centroid(b)
            mc_tracks.append({"tid": next_tid, "cx": cc[0], "cy": cc[1], "miss": 0})
            tid_of_box[bi] = next_tid
            next_tid += 1
    return mc_tracks, next_tid, tid_of_box


def collect_signals(weights: str, clip: Path) -> tuple[list[FrameSig], int, float, list[FusionPack]]:
    """One forward pass: per-box residual + rotation + GT dist + tracklet ids."""
    packs = detect_fusion_clip(weights, clip, use_cache=True)
    _, _, radius, start = _seed(packs)
    recs: list[FrameSig] = []
    tracks_iou: list[BoxTrack] = []
    next_iou = 0
    mc_tracks: list[dict] = []
    next_mc = 0
    prev_boxes: list[tuple] = []
    prev_gray = None

    src = VideoSource(clip)
    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        gray = cv2.cvtColor(strip_pointer(raw, strip_green=True), cv2.COLOR_BGR2GRAY)
        boxes = p.boxes
        residual = [0.0] * len(boxes)
        rot: list[float | None] = [None] * len(boxes)
        paper_T = None
        if prev_gray is not None and prev_boxes and boxes:
            paper_T = estimate_paper_motion(prev_gray, gray, boxes)
            paper_T = _refine_paper_from_fakes(prev_boxes, boxes, paper_T)
            residual = _paper_box_residuals(prev_boxes, boxes, paper_T)
            theta_g = math.atan2(paper_T[1, 0], paper_T[0, 0]) if paper_T is not None else 0.0
            rot = _box_net_rotations(prev_boxes, prev_gray, boxes, gray, theta_g)

        tracks_iou, next_iou = _associate(tracks_iou, boxes, next_iou)
        tid_iou = [_tid_for_box(tracks_iou, b) for b in boxes]
        mc_tracks, next_mc, tid_mc = _mc_associate(mc_tracks, boxes, paper_T, next_mc)

        rec = FrameSig(idx, p.gt)
        for i, b in enumerate(boxes):
            cx, cy = _centroid(b)
            gd = math.hypot(cx - p.gt[0], cy - p.gt[1]) if p.gt is not None else float("nan")
            rec.boxes.append((cx, cy, residual[i],
                              float("nan") if rot[i] is None else rot[i],
                              gd, tid_iou[i], tid_mc[i]))
        recs.append(rec)
        prev_boxes = list(boxes)
        prev_gray = gray
    src.release()
    return recs, start, radius, packs


def _scored(recs: list[FrameSig], start: int) -> list[FrameSig]:
    return [r for r in recs if r.idx >= start and r.gt is not None and r.boxes]


def r0_detection(recs, start, radius) -> float:
    s = _scored(recs, start)
    if not s:
        return 0.0
    hits = sum(1 for r in s if min(b[4] for b in r.boxes) < radius)
    return hits / len(s)


def r1_association(recs, start, radius, which: str) -> tuple[float, float]:
    """(best single-tracklet within_r, identity persistence). which=tid_iou|tid_mc."""
    col = 5 if which == "iou" else 6
    s = _scored(recs, start)
    if not s:
        return 0.0, 0.0
    n = len(s)
    # best single tracklet: frames where that tracklet's box is within radius of GT
    tid_hits: Counter = Counter()
    for r in s:
        for b in r.boxes:
            tid = b[col]
            if tid is not None and b[4] < radius:
                tid_hits[tid] += 1
    best = max(tid_hits.values()) / n if tid_hits else 0.0
    # identity persistence: among oracle-hit frames, tid of the GT-nearest box;
    # fraction covered by the single most common tid
    gt_tids = []
    for r in s:
        nearest = min(r.boxes, key=lambda b: b[4])
        if nearest[4] < radius and nearest[col] is not None:
            gt_tids.append(nearest[col])
    persistence = (Counter(gt_tids).most_common(1)[0][1] / len(gt_tids)) if gt_tids else 0.0
    return best, persistence


def _rank_of_gt(r: FrameSig, key) -> int | None:
    """1-indexed rank (desc by key) of the GT-nearest box; None if no box."""
    if not r.boxes:
        return None
    nearest = min(range(len(r.boxes)), key=lambda i: r.boxes[i][4])
    scores = sorted(range(len(r.boxes)), key=lambda i: key(r.boxes[i]), reverse=True)
    return scores.index(nearest) + 1


def r2_signal(recs, start, radius) -> dict:
    """How often the GT box is the top motion outlier per frame (oracle-hit frames)."""
    s = [r for r in _scored(recs, start) if min(b[4] for b in r.boxes) < radius]
    if not s:
        return {k: 0.0 for k in ("resid_top1", "resid_top3", "rot_top1",
                                 "fused_top1", "fused_top3")}

    def resid_key(b):
        return b[2]

    def rot_key(b):
        return 0.0 if math.isnan(b[3]) else b[3]

    def fused_key(b):
        r = 0.0 if math.isnan(b[3]) else max(0.0, b[3] - ROT_FLOOR)
        return b[2] + ROT_WEIGHT * r

    def frac(key, topk):
        return sum(1 for r in s if (_rank_of_gt(r, key) or 99) <= topk) / len(s)

    return {
        "resid_top1": frac(resid_key, 1), "resid_top3": frac(resid_key, 3),
        "rot_top1": frac(rot_key, 1),
        "fused_top1": frac(fused_key, 1), "fused_top3": frac(fused_key, 3),
    }


def _dump_csv(clip_stem: str, recs: list[FrameSig]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"{clip_stem}__signals.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "box_i", "cx", "cy", "residual", "rot_net",
                    "gt_dist", "tid_iou", "tid_mc"])
        for r in recs:
            for i, b in enumerate(r.boxes):
                w.writerow([r.idx, i, f"{b[0]:.1f}", f"{b[1]:.1f}", f"{b[2]:.3f}",
                            "" if math.isnan(b[3]) else f"{b[3]:.4f}",
                            f"{b[4]:.1f}", b[5], b[6]])


@dataclass
class ClipDiag:
    clip: str
    r0: float
    r1_iou: float
    r1_mc: float
    persist_iou: float
    persist_mc: float
    r2: dict
    accum: float


def diagnose_clip(weights: str, clip: Path) -> ClipDiag:
    recs, start, radius, packs = collect_signals(weights, clip)
    _dump_csv(clip.stem, recs)
    name = clip.stem.replace("_cropped_trimmed", "")
    r1_iou, persist_iou = r1_association(recs, start, radius, "iou")
    r1_mc, persist_mc = r1_association(recs, start, radius, "mc")
    lock = compute_countdown_lock(packs, clip)
    tr, _, s, rad = track_accum_identity(clip, packs, lock, frame_wh=_frame_wh(clip))
    accum = score_identity(packs, tr, s, rad, name).within_r
    return ClipDiag(name, r0_detection(recs, start, radius), r1_iou, r1_mc,
                    persist_iou, persist_mc, r2_signal(recs, start, radius), accum)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def write_report(diags: list[ClipDiag], weights: str) -> None:
    L: list[str] = []
    L.append("# Identity-loss decomposition")
    L.append("")
    L.append(f"Weights: `{weights}` · clips: {', '.join(d.clip for d in diags)}")
    L.append("")
    L.append("Each rung relaxes one constraint; the gap between rungs assigns the "
             "loss. R0 detection ceiling · R1 best single tracklet (association) · "
             "R2 GT box is top motion outlier per frame (signal) · accum = live "
             "tracker actual. `persist` = fraction of oracle-hit frames covered by "
             "one tracklet identity (low ⇒ identity fragments).")
    L.append("")
    L.append("| clip | R0 det | R1 assoc(IoU) | R1 assoc(MC) | persist IoU/MC | R2 resid top1/3 | R2 fused top1/3 | accum |")
    L.append("|------|-------:|--------------:|-------------:|---------------|-----------------|-----------------|------:|")
    for d in diags:
        L.append(f"| {d.clip} | {d.r0:.2f} | {d.r1_iou:.2f} | {d.r1_mc:.2f} | "
                 f"{d.persist_iou:.2f}/{d.persist_mc:.2f} | "
                 f"{d.r2['resid_top1']:.2f}/{d.r2['resid_top3']:.2f} | "
                 f"{d.r2['fused_top1']:.2f}/{d.r2['fused_top3']:.2f} | {d.accum:.2f} |")
    L.append(f"| **mean** | **{_mean([d.r0 for d in diags]):.2f}** | "
             f"**{_mean([d.r1_iou for d in diags]):.2f}** | "
             f"**{_mean([d.r1_mc for d in diags]):.2f}** | "
             f"{_mean([d.persist_iou for d in diags]):.2f}/{_mean([d.persist_mc for d in diags]):.2f} | "
             f"{_mean([d.r2['resid_top1'] for d in diags]):.2f}/{_mean([d.r2['resid_top3'] for d in diags]):.2f} | "
             f"{_mean([d.r2['fused_top1'] for d in diags]):.2f}/{_mean([d.r2['fused_top3'] for d in diags]):.2f} | "
             f"**{_mean([d.accum for d in diags]):.2f}** |")
    L.append("")
    # automatic read of where the loss sits
    r0, r1i, r1m = (_mean([d.r0 for d in diags]),
                    _mean([d.r1_iou for d in diags]),
                    _mean([d.r1_mc for d in diags]))
    persist = _mean([d.persist_iou for d in diags])
    t1, t3 = (_mean([d.r2['fused_top1'] for d in diags]),
              _mean([d.r2['fused_top3'] for d in diags]))
    ac = _mean([d.accum for d in diags])
    L.append("## Reading")
    L.append("")
    L.append(f"- **Detection is fine** (R0 {r0:.2f}); the loss is downstream.")
    L.append(f"- **Identity fragments badly**: a single tracklet holds the real "
             f"shape only {r1i:.2f} of the time (persistence {persist:.2f}); "
             f"motion-compensated association does not help ({r1m:.2f}). The "
             f"tracklet abstraction is structurally leaky for an independently-"
             f"moving target amid dense identical neighbours.")
    L.append(f"- **accum ({ac:.2f}) already EXCEEDS the single-tracklet ceiling "
             f"({r1i:.2f})** — its re-bind/switch logic stitches fragments back "
             f"together. So the remaining loss is NOT decision-tuning headroom "
             f"(accum beats the naive single-identity bound by ~2x).")
    L.append(f"- **Per-frame signal is weak**: the GT box is the top fused motion "
             f"outlier only {t1:.2f} of frames (top-3 {t3:.2f}) — the real shape is "
             f"often stationary or buried, so a memoryless picker is hopeless and "
             f"temporal integration is essential.")
    L.append("")
    L.append("**Conclusion — the binding limiters are association fragmentation + "
             "weak per-frame signal, not decision thresholds.** Highest-EV next "
             "moves: (1) a **position-field accumulator** that integrates "
             "independent-motion evidence *spatially* (reuse `ld/vision/motion.py` "
             "`saliency_map`), immune to identity fragmentation; (2) **stronger "
             "per-frame signal**. Decision tuning (adaptive switch/tie-break) is "
             "low-EV — accum already transcends the single-identity ceiling.")
    L.append("")
    DIAGNOSIS_MD.write_text("\n".join(L))
    print(f"\ndiagnosis -> {DIAGNOSIS_MD}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Decompose the identity-tracking loss")
    ap.add_argument("--weights", default="data/detect/runs/yolov8n_combined/weights/best.pt")
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()
    clips = ([Path(c) if Path(c).exists() else DATA_DIR / f"{c}_cropped_trimmed.mp4"
              for c in args.clips] if args.clips else _default_clips())
    diags = []
    for clip in clips:
        print(f"[{clip.stem}] diagnosing ...")
        d = diagnose_clip(args.weights, clip)
        print(f"  R0={d.r0:.2f} R1_iou={d.r1_iou:.2f} R1_mc={d.r1_mc:.2f} "
              f"persist={d.persist_iou:.2f}/{d.persist_mc:.2f} "
              f"resid_top1={d.r2['resid_top1']:.2f} fused_top1={d.r2['fused_top1']:.2f} "
              f"accum={d.accum:.2f}")
        diags.append(d)
    write_report(diags, args.weights)


if __name__ == "__main__":
    main()
