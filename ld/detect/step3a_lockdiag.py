"""Step-3a diagnostic (read-only): does compute_countdown_lock seed the REAL shape, or
an adjacent fake? Step-0 flagged t1's earliest miss run (f133-164) as identity-creep
starting right at lock onset. If the lock box is far from GT while a closer box exists,
the seed is wrong and poisons the whole early run.

For each clip: lock frame, picked-lock-box centroid, GT there, dist(lock,GT) in radii,
the oracle-nearest box dist, and whether the lock matches the oracle pick.

    python -m ld.detect.step3a_lockdiag --weights .../best.pt
"""
from __future__ import annotations

import argparse
import math

from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _centroid, _seed, compute_countdown_lock


def run(weights, clips):
    print(f"{'clip':>4}  {'lockf':>5}  {'r':>3}  {'dist(lock,GT)':>13}  "
          f"{'dist(oracle,GT)':>15}  seed")
    for clip in clips:
        name = clip.stem.replace("_cropped_trimmed", "")
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        lock = compute_countdown_lock(packs, clip)
        _, _, radius, _ = _seed(packs)
        if lock is None:
            print(f"{name:>4}  (no lock)")
            continue
        p = packs[lock.frame]
        gt = p.gt
        if gt is None:
            # GT may be absent exactly on the lock frame; scan a few frames forward
            for j in range(lock.frame + 1, min(lock.frame + 12, len(packs))):
                if packs[j].gt is not None:
                    gt = packs[j].gt
                    break
        if gt is None or not p.boxes:
            print(f"{name:>4}  {lock.frame:>5}  (no gt/boxes near lock)")
            continue
        dl = math.hypot(lock.cx - gt[0], lock.cy - gt[1])
        od = min(math.hypot(_centroid(b)[0] - gt[0], _centroid(b)[1] - gt[1]) for b in p.boxes)
        # is the lock box the oracle-nearest box?
        seed_ok = abs(dl - od) < 1.0
        flag = "OK" if dl < radius else ("WRONG" if od < radius else "no-box-on-gt")
        print(f"{name:>4}  {lock.frame:>5}  {radius:>3.0f}  "
              f"{dl:>9.0f}px {dl/radius:>4.2f}r  {od:>11.0f}px {od/radius:>4.2f}r  "
              f"{flag}{'' if seed_ok or flag!='WRONG' else '  (closer box existed)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()
    clips = list(_default_clips())
    if args.clips:
        clips = [c for c in clips if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips)


if __name__ == "__main__":
    main()
