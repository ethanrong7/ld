"""Step-2 detection-knob pre-check (read-only re: identity; runs YOLO at new imgsz/conf,
which detect_fusion_clip caches separately). Question: do the t8/t5 detection-bound
failures (Step-0: t8 f226-248 oracle~0; t8/t5 oversized/undersized lock boxes) improve
with a higher imgsz or lower conf on the ALREADY-TRAINED model -- a near-free win before
any annotation+retrain?

Reports, per (imgsz,conf): overall oracle within_r (fraction of scored frames with a box
centroid within radius of GT) and oracle on the Step-0 problem frame-ranges, plus mean
GT-box size ratio (GT box area vs median box area) on those ranges.

    python -m ld.detect.step2_detknob --weights .../best.pt --clips t8 t5
"""
from __future__ import annotations

import argparse
import math
import statistics

from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _centroid, _seed

# Step-0 problem ranges per clip (sustained miss runs of interest).
RANGES = {
    "t8": [(226, 248), (336, 367), (656, 719)],
    "t5": [(353, 413), (517, 545)],
    "t1": [(133, 164), (306, 321)],
}


def _gt_box(boxes, gt, radius):
    if not boxes or gt is None:
        return None, None
    best_i, best_d = None, 1e18
    for i, b in enumerate(boxes):
        c = _centroid(b)
        d = math.hypot(c[0] - gt[0], c[1] - gt[1])
        if d < best_d:
            best_d, best_i = d, i
    return (best_i, best_d) if best_d < radius else (None, best_d)


def _stats(weights, clip, imgsz, conf):
    name = clip.stem.replace("_cropped_trimmed", "")
    packs = detect_fusion_clip(weights, clip, imgsz=imgsz, conf=conf, use_cache=True)
    _, _, radius, start = _seed(packs)
    n = hit = 0
    rng_hit = {r: [0, 0] for r in RANGES.get(name, [])}
    size_ratios = []
    for p in packs:
        if p.idx < start or p.gt is None or not p.boxes:
            continue
        n += 1
        gi, gd = _gt_box(p.boxes, p.gt, radius)
        if gi is not None:
            hit += 1
        for (a, b) in RANGES.get(name, []):
            if a <= p.idx <= b:
                rng_hit[(a, b)][1] += 1
                if gi is not None:
                    rng_hit[(a, b)][0] += 1
                    areas = [(bx[2] - bx[0]) * (bx[3] - bx[1]) for bx in p.boxes]
                    med = statistics.median(areas)
                    if med > 0:
                        size_ratios.append(areas[gi] / med)
    return name, radius, hit / n if n else 0, n, rng_hit, size_ratios


def run(weights, clips, cfgs):
    for clip in clips:
        name = clip.stem.replace("_cropped_trimmed", "")
        print(f"\n=== {name} ===")
        for imgsz, conf in cfgs:
            nm, radius, orc, n, rng_hit, sr = _stats(weights, clip, imgsz, conf)
            srtxt = (f"  GT-size-ratio med={statistics.median(sr):.2f} "
                     f"min={min(sr):.2f} max={max(sr):.2f}" if sr else "")
            print(f"  imgsz={imgsz} conf={conf:>4}: oracle={orc:.3f} (n={n})")
            for r, (h, tot) in rng_hit.items():
                print(f"      f{r[0]}-{r[1]}: oracle={h/tot if tot else 0:.3f} ({h}/{tot})")
            if srtxt:
                print(f"     {srtxt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=["t8", "t5", "t1"])
    args = ap.parse_args()
    clips = [c for c in _default_clips()
             if c.stem.replace("_cropped_trimmed", "") in args.clips]
    cfgs = [(768, 0.25), (1024, 0.25), (768, 0.10)]
    run(args.weights, clips, cfgs)


if __name__ == "__main__":
    main()
