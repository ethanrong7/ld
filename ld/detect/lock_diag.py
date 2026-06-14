"""Diagnose countdown lock quality: mask overlap, lock->GT, frame grabs.

Usage:
    python -m ld.detect.lock_diag --weights data/detect/runs/yolov8n_probe/weights/best.pt
    python -m ld.detect.lock_diag --weights .../best.pt --clips t3 t4 t8
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource
from ld.config import DATA_DIR, DETECT_DIR
from ld.detect.constellation import _seed
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import (
    LockInfo,
    LOCK_MASK_MIN,
    _box_mask_overlap,
    _box_white_circle_overlap,
    _centroid,
    _countdown_white_pack,
    _pick_lock_box,
    _stable_white_anchor,
    _white_mask,
    compute_countdown_lock,
)
from ld.vision.cursor import strip_pointer

OUT_DIR = DETECT_DIR / "evidence" / "lock_diag"


def _oracle_box_index(boxes: list[tuple], gt: tuple[float, float]) -> int:
    return min(range(len(boxes)),
               key=lambda i: math.hypot(_centroid(boxes[i])[0] - gt[0],
                                        _centroid(boxes[i])[1] - gt[1]))


def _lock_scores(boxes: list[tuple], mask: np.ndarray,
                 white: tuple[float, float, float]) -> list[float]:
    scores = []
    for box in boxes:
        s = _box_mask_overlap(box, mask) if mask is not None else 0.0
        if s <= 0:
            s = _box_white_circle_overlap(box, white)
        scores.append(s)
    return scores


def _render(lock_frame: np.ndarray, mask: np.ndarray, boxes: list[tuple],
            lock_i: int, oracle_i: int, gt: tuple[float, float] | None,
            title: str) -> np.ndarray:
    vis = lock_frame.copy()
    overlay = vis.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array((200, 200, 255)) * 0.5).astype(np.uint8)
    vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        if i == lock_i:
            col, thick = (0, 255, 0), 3
            label = "LOCK"
        elif i == oracle_i:
            col, thick = (255, 200, 0), 2
            label = "ORACLE"
        else:
            col, thick = (0, 180, 255), 1
            label = ""
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, thick)
        if label:
            cv2.putText(vis, label, (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    if gt is not None:
        cv2.drawMarker(vis, (int(gt[0]), int(gt[1])), (0, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 18, 2)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(vis, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def diagnose_clip(weights: Path, clip: Path, *, conf: float = 0.25,
                  imgsz: int = 768) -> dict:
    name = clip.stem.replace("_cropped_trimmed", "")
    packs = detect_fusion_clip(weights, clip, conf=conf, imgsz=imgsz)
    lock = compute_countdown_lock(packs, clip)
    _, _, radius, start = _seed(packs)  # type: ignore[arg-type]
    lock_pack = _countdown_white_pack(packs)

    row: dict = {"clip": name, "within_r_clip": None, "ok": False}
    if lock is None or lock_pack is None:
        row["error"] = "no lock"
        return row

    lock_pack = packs[lock.frame]
    gt = lock_pack.gt
    if gt is None:
        gt = packs[start].gt if start < len(packs) else None

    mask_frame: np.ndarray | None = None
    src = VideoSource(clip)
    for idx, frame in src.frames():
        if idx == lock.frame:
            mask_frame = strip_pointer(frame, strip_green=True)
            break
    src.release()
    if mask_frame is None:
        row["error"] = "no frame"
        return row

    anchor_info = _stable_white_anchor(packs)
    anchor = (anchor_info[0], anchor_info[1]) if anchor_info else (lock_pack.white[0], lock_pack.white[1])

    mask = _white_mask(mask_frame)
    boxes = lock_pack.boxes
    scores = _lock_scores(boxes, mask, lock_pack.white)  # type: ignore[arg-type]
    best_mask_s = max(scores) if scores else 0.0
    lock_i = _pick_lock_box(boxes, mask, lock_pack.white, anchor)
    oracle_i = _oracle_box_index(boxes, gt) if gt and boxes else 0

    lock_c = _centroid(boxes[lock_i])
    oracle_c = _centroid(boxes[oracle_i])
    d_lock_gt = math.hypot(lock_c[0] - gt[0], lock_c[1] - gt[1]) if gt else float("nan")
    d_oracle_gt = math.hypot(oracle_c[0] - gt[0], oracle_c[1] - gt[1]) if gt else float("nan")

    # Rank of oracle box by mask overlap score.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    oracle_rank = order.index(oracle_i) + 1 if boxes else -1

    row.update({
        "lock_frame": lock.frame,
        "start_frame": start,
        "n_boxes": len(boxes),
        "radius": radius,
        "lock_score": scores[lock_i],
        "oracle_score": scores[oracle_i],
        "anchor": anchor,
        "anchor_frame": anchor_info[3] if anchor_info else lock.frame,
        "used_anchor_fallback": best_mask_s < LOCK_MASK_MIN,
        "oracle_rank": oracle_rank,
        "lock_to_gt_px": d_lock_gt,
        "oracle_to_gt_px": d_oracle_gt,
        "lock_within_r": d_lock_gt < radius if gt else False,
        "oracle_within_r": d_oracle_gt < radius if gt else False,
        "white": lock_pack.white,
        "top3": [(i, scores[i], _centroid(boxes[i])) for i in order[:3]],
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    title = (f"{name} f{lock.frame} lock->{d_lock_gt:.0f}px "
             f"oracle->{d_oracle_gt:.0f}px r={radius:.0f} "
             f"oracle_rank={oracle_rank}/{len(boxes)}")
    vis = _render(mask_frame, mask, boxes, lock_i, oracle_i, gt, title)
    out_png = OUT_DIR / f"{name}_lock_f{lock.frame}.png"
    cv2.imwrite(str(out_png), vis)
    row["png"] = str(out_png)

    # Nearby countdown frames for context (lock-2, lock-1, lock, start).
    for fidx in sorted({max(0, lock.frame - 2), lock.frame - 1, lock.frame, start}):
        if fidx < 0 or fidx >= len(packs):
            continue
        src = VideoSource(clip)
        frame_img = None
        for idx, frame in src.frames():
            if idx == fidx:
                frame_img = strip_pointer(frame, strip_green=True)
                break
        src.release()
        if frame_img is None:
            continue
        p = packs[fidx]
        ctx = frame_img.copy()
        for i, box in enumerate(p.boxes):
            cv2.rectangle(ctx, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 180, 255), 1)
        if p.gt is not None:
            cv2.drawMarker(ctx, (int(p.gt[0]), int(p.gt[1])), (0, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 14, 2)
        if p.white is not None:
            cv2.circle(ctx, (int(p.white[0]), int(p.white[1])),
                       int(p.white[2]), (255, 255, 255), 2)
        hud = f"{name} f{fidx} white={'Y' if p.white else 'N'} dets={len(p.boxes)}"
        cv2.putText(ctx, hud, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(OUT_DIR / f"{name}_ctx_f{fidx}.png"), ctx)

    row["ok"] = True
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Countdown lock diagnostics")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="default: t3 t4 t8 + controls t1 t7 t9")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=768)
    args = ap.parse_args()

    weights = Path(args.weights)
    if args.clips:
        clips = [DATA_DIR / f"{c}_cropped_trimmed.mp4" for c in args.clips]
    else:
        clips = [DATA_DIR / f"{c}_cropped_trimmed.mp4"
                 for c in ("t3", "t4", "t8", "t1", "t7", "t9")]

    print(f"Lock diagnostics -> {OUT_DIR}/\n")
    print(f"{'clip':<6} {'f_lock':>6} {'lock->GT':>9} {'oracle->GT':>11} "
          f"{'rank':>8} {'lock_sc':>8} {'ora_sc':>8} {'ok?':>5}")
    print("-" * 72)

    for clip in clips:
        if not clip.exists():
            print(f"{clip.stem:<6} MISSING")
            continue
        row = diagnose_clip(weights, clip, conf=args.conf, imgsz=args.imgsz)
        if not row.get("ok"):
            print(f"{row['clip']:<6} ERROR: {row.get('error', '?')}")
            continue
        ok = "Y" if row["lock_within_r"] else "N"
        print(f"{row['clip']:<6} {row['lock_frame']:>6} "
              f"{row['lock_to_gt_px']:>8.0f}px {row['oracle_to_gt_px']:>10.0f}px "
              f"{row['oracle_rank']:>4}/{row['n_boxes']:<3} "
              f"{row['lock_score']:>8.3f} {row['oracle_score']:>8.3f} {ok:>5}")
        if row["used_anchor_fallback"]:
            print(f"       anchor fallback ({row['anchor'][0]:.0f},{row['anchor'][1]:.0f}) "
                  f"f{row['anchor_frame']} mask_best={row['lock_score']:.4f}")
        print(f"       -> {row['png']}")


if __name__ == "__main__":
    main()
