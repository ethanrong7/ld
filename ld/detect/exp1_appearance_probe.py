"""EXP-1 GATE: does an APPEARANCE-change channel, orthogonal to motion, rank the
real-shape box #1 on the frames the `fpath_fuse` trellis currently MISSES?

HYPOTHESIS (plan.md EXP-1). The real shape moves/rotates INDEPENDENTLY of the
sheet; the fakes move rigidly with it. So if we sample each box's interior, undo
the GLOBAL sheet rotation (from the RANSAC affine), and compare to the same box one
frame earlier, a FAKE's de-rotated interior should match (rigid => no residual
change) while the REAL shape's interior still changes (independent motion remains).
That residual appearance-change is a per-box signal that is orthogonal to the
motion channels (mass/coherence/curl) the fuse trellis already uses.

WHY THIS IS NEW. NCC failed before because it was translation-only and de-correlated
under the real shape's rotation. The new angle: (a) compare each box to its OWN prior
location (nearest prev box, not a fixed template) and (b) de-rotate by the global
sheet angle first, so a rigid fake cancels to ~0 and only independent motion survives.
We test three operationalizations side by side, plus two rotation-INVARIANT descriptors
(ring/radial-profile, log-polar-FFT magnitude) that need no de-rotation at all:

  derot_ncc : 1 - NCC( derotate(cur box ROI, -theta) , prev box ROI )      [primary]
  raw_ncc   : 1 - NCC( cur box ROI , prev box ROI )            [no de-rotation control]
  ring      : L1 change of radial intensity profile (rotation-invariant)
  logpolar  : L1 change of log-polar-FFT magnitude (rotation+scale-invariant)

For each channel, over scored oracle-hit frames AND over the `fpath_fuse` MISS subset,
we report how often the real-shape box (YOLO box nearest GT, within radius) ranks #1
and within top-3. The motion baselines (mass, windowed coherent-mass, curl) are printed
on the SAME frames for an apples-to-apples comparison.

GATE (from plan.md): if an appearance channel's MISS-top1 on t8 clears ~0.4 (motion's
coherent-mass argmax is ~0.32 there) AND it is not just re-deriving the motion ranking,
it is a genuinely complementary 4th emission term -> build it into
`track_fused_path_identity` and re-sweep weights. If it tracks motion, drop it.

Read-only. Reuses the cohgate outlier-vector cache for the motion baselines; loads
grays once per clip for the appearance channels.

    python -m ld.detect.exp1_appearance_probe --weights .../best.pt --clips t8 t1 t5 t4
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from ld.config import DETECT_DIR
from ld.detect.coh_gate import _compute_ov, _box_coherent_mass
from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.fuse_probe import (
    _box_curl, _windowed, _gt_box_idx, _rank_of, _load_miss_frames,
)
from ld.detect.identity import _centroid, compute_countdown_lock, _seed
from ld.capture.video_source import VideoSource
from ld.vision.cursor import strip_pointer


# ---- appearance descriptors -------------------------------------------------

def _patch(gray, cx, cy, half, deg=0.0):
    """(2*half x 2*half) patch centered at (cx,cy), the image rotated `deg` degrees
    about that center first (deg=0 => plain crop). BORDER_REPLICATE so edge boxes
    don't inject black. Returns float32."""
    size = 2 * half
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(deg), 1.0)
    M[0, 2] += half - cx
    M[1, 2] += half - cy
    p = cv2.warpAffine(gray, M, (size, size), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return p.astype(np.float32)


def _ncc(a, b):
    """Normalized cross-correlation in [-1,1]; robust to brightness/contrast."""
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt((a * a).sum()))
    db = float(np.sqrt((b * b).sum()))
    if da < 1e-6 or db < 1e-6:
        return 0.0
    return float((a * b).sum() / (da * db))


def _ring_profile(patch, n_rings=8):
    """Mean intensity in concentric annuli about the patch center -> rotation-invariant
    descriptor. L2-normalized so it's a shape signature, not a brightness one."""
    h, w = patch.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    rmax = min(cx, cy)
    prof = np.zeros(n_rings, np.float32)
    for k in range(n_rings):
        lo = rmax * k / n_rings
        hi = rmax * (k + 1) / n_rings
        m = (r >= lo) & (r < hi)
        if m.any():
            prof[k] = patch[m].mean()
    nrm = float(np.linalg.norm(prof))
    return prof / nrm if nrm > 1e-6 else prof


def _logpolar_fft(patch):
    """Magnitude spectrum of the log-polar transform -> rotation+scale invariant
    signature (Fourier-Mellin spirit). L2-normalized."""
    h, w = patch.shape
    center = (w / 2.0, h / 2.0)
    maxr = min(center)
    lp = cv2.logPolar(patch, center, maxr / math.log(maxr + 1e-6),
                      cv2.WARP_FILL_OUTLIERS)
    f = np.fft.fft2(lp)
    mag = np.abs(np.fft.fftshift(f)).astype(np.float32).ravel()
    nrm = float(np.linalg.norm(mag))
    return mag / nrm if nrm > 1e-6 else mag


def _nearest_prev_box(prev_boxes, c):
    """Centroid of the prev-frame box nearest to point c (the box's own prior
    location; identical shapes + small frame-to-frame motion make this reliable)."""
    if not prev_boxes:
        return None
    best, bd = None, 1e18
    for b in prev_boxes:
        pc = _centroid(b)
        d = (pc[0] - c[0]) ** 2 + (pc[1] - c[1]) ** 2
        if d < bd:
            bd, best = d, pc
    return best


def _appearance_scores(prev_gray, cur_gray, prev_boxes, cur_boxes, theta_deg, half):
    """Per-box appearance-CHANGE for each descriptor (higher => changed more =>
    more likely the independently-moving real shape)."""
    n = len(cur_boxes)
    derot = [0.0] * n
    raw = [0.0] * n
    ring = [0.0] * n
    logp = [0.0] * n
    rs = 48  # resize for ncc / descriptor stability
    for i, b in enumerate(cur_boxes):
        c = _centroid(b)
        pc = _nearest_prev_box(prev_boxes, c)
        if pc is None:
            continue
        cur_d = cv2.resize(_patch(cur_gray, c[0], c[1], half, -theta_deg), (rs, rs))
        cur_0 = cv2.resize(_patch(cur_gray, c[0], c[1], half, 0.0), (rs, rs))
        prev_0 = cv2.resize(_patch(prev_gray, pc[0], pc[1], half, 0.0), (rs, rs))
        derot[i] = 1.0 - _ncc(cur_d, prev_0)
        raw[i] = 1.0 - _ncc(cur_0, prev_0)
        ring[i] = float(np.abs(_ring_profile(cur_0) - _ring_profile(prev_0)).sum())
        logp[i] = float(np.abs(_logpolar_fft(cur_0) - _logpolar_fft(prev_0)).sum())
    return {"derot_ncc": derot, "raw_ncc": raw, "ring": ring, "logpolar": logp}


def _load_grays(clip, packs):
    """All cursor-stripped gray frames of the clip, indexed; affine theta per pair."""
    src = VideoSource(clip)
    grays = {}
    for idx, frame in src.frames():
        if idx >= len(packs):
            continue
        grays[idx] = cv2.cvtColor(strip_pointer(frame, strip_green=True),
                                  cv2.COLOR_BGR2GRAY)
    src.release()
    return grays


def _theta_deg(prev_gray, cur_gray):
    """Global sheet rotation (degrees, prev->cur) from the RANSAC partial affine."""
    from ld.vision.motion import estimate_motion
    f = estimate_motion(prev_gray, cur_gray)
    if f.affine is None:
        return 0.0
    return math.degrees(math.atan2(f.affine[1, 0], f.affine[0, 0]))


def _affine(prev_gray, cur_gray):
    """2x3 partial affine prev->cur (RANSAC), or None."""
    from ld.vision.motion import estimate_motion
    return estimate_motion(prev_gray, cur_gray).affine


def _to3x3(a):
    M = np.eye(3, dtype=np.float64)
    M[:2, :] = a
    return M


def _baseline_scores(grays, idxs, pos, packs, box, half, base_n, affines):
    """LONGER-BASELINE appearance change: compare box's ROI now vs `base_n` frames ago,
    back-projecting the box center through the COMPOSED rigid affine (a fake lands
    exactly on its prior self => cancels; the real shape's independent motion does not).
    De-rotate by the accumulated global angle. Returns (derotN_change, rawN_change)."""
    j = pos - base_n
    if j < 0:
        return None
    cur_idx, old_idx = idxs[pos], idxs[j]
    if cur_idx not in grays or old_idx not in grays:
        return None
    # compose per-pair affines old->...->cur  (each affines[k] maps idxs[k-1]->idxs[k])
    M = np.eye(3, dtype=np.float64)
    for k in range(j + 1, pos + 1):
        a = affines.get(idxs[k])
        if a is None:
            return None
        M = _to3x3(a) @ M
    c = _centroid(box)
    inv = np.linalg.inv(M)
    old_c = inv @ np.array([c[0], c[1], 1.0])
    old_c = (old_c[0] / old_c[2], old_c[1] / old_c[2])
    theta_acc = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    rs = 48
    cur_d = cv2.resize(_patch(grays[cur_idx], c[0], c[1], half, -theta_acc), (rs, rs))
    cur_0 = cv2.resize(_patch(grays[cur_idx], c[0], c[1], half, 0.0), (rs, rs))
    old_0 = cv2.resize(_patch(grays[old_idx], old_c[0], old_c[1], half, 0.0), (rs, rs))
    return 1.0 - _ncc(cur_d, old_0), 1.0 - _ncc(cur_0, old_0)


# ---- driver -----------------------------------------------------------------

APP_CHANS = ["derot_ncc", "raw_ncc", "ring", "logpolar", "derotN", "rawN"]
MOT_CHANS = ["mass", "coh", "curl"]


def run(weights, clips, win, miss_mode, base_n):
    chans = MOT_CHANS + APP_CHANS
    agg = {c: [0, 0, 0, 0, 0, 0] for c in chans}  # t1_all,t3_all,n_all,t1_miss,t3_miss,n_miss
    print(f"window={win}  baseline_N={base_n}  miss frames from `{miss_mode}`\n")
    for clip in clips:
        packs = detect_fusion_clip(weights, clip, use_cache=True)
        compute_countdown_lock(packs, clip)
        _, _, radius, start = _seed(packs)
        idxs, ov = _compute_ov(clip, packs)
        grays = _load_grays(clip, packs)
        half = int(round(radius))
        name = clip.stem.replace("_cropped_trimmed", "")
        miss = _load_miss_frames(clip.stem, miss_mode)
        pos_of = {ix: pos for pos, ix in enumerate(idxs)}
        # per-pair affine (idxs[k-1]->idxs[k]) + |theta| stats
        affines, thetas = {}, []
        for pos in range(1, len(idxs)):
            pi, ci = idxs[pos - 1], idxs[pos]
            if pi in grays and ci in grays:
                a = _affine(grays[pi], grays[ci])
                affines[ci] = a
                if a is not None:
                    thetas.append(abs(math.degrees(math.atan2(a[1, 0], a[0, 0]))))
        mean_theta = float(np.mean(thetas)) if thetas else 0.0
        loc = {c: [0, 0, 0, 0, 0, 0] for c in chans}
        for idx in idxs:
            if idx < start or idx >= len(packs):
                continue
            p = packs[idx]
            if p.gt is None or not p.boxes:
                continue
            gi = _gt_box_idx(p.boxes, p.gt, radius)
            if gi is None:
                continue
            pos = pos_of[idx]
            if pos == 0:
                continue
            prev_idx = idxs[pos - 1]
            if prev_idx not in grays or idx not in grays:
                continue
            # motion baselines (same as fuse_probe)
            coh = _windowed(idxs, ov, p.boxes, idx, win, _box_coherent_mass)
            curl = _windowed(idxs, ov, p.boxes, idx, win, _box_curl)
            o0 = ov.get(idx)
            mass = []
            for b in p.boxes:
                if o0 is None:
                    mass.append(0.0)
                    continue
                pix, _r, magv = o0
                inb = ((pix[:, 0] >= b[0]) & (pix[:, 0] <= b[2])
                       & (pix[:, 1] >= b[1]) & (pix[:, 1] <= b[3]))
                mass.append(float(magv[inb].sum()))
            # appearance channels
            theta = _theta_deg(grays[prev_idx], grays[idx])
            app = _appearance_scores(grays[prev_idx], grays[idx],
                                     packs[prev_idx].boxes, p.boxes, theta, half)
            # longer-baseline channels (rigid back-projection over base_n frames)
            derotN = [0.0] * len(p.boxes)
            rawN = [0.0] * len(p.boxes)
            for bi, b in enumerate(p.boxes):
                r = _baseline_scores(grays, idxs, pos, packs, b, half, base_n, affines)
                if r is not None:
                    derotN[bi], rawN[bi] = r
            app["derotN"] = derotN
            app["rawN"] = rawN
            scores = {"mass": mass, "coh": coh, "curl": curl, **app}
            in_miss = idx in miss
            for c in chans:
                rk = _rank_of(scores[c], gi)
                loc[c][2] += 1
                if rk == 1: loc[c][0] += 1
                if rk <= 3: loc[c][1] += 1
                if in_miss:
                    loc[c][5] += 1
                    if rk == 1: loc[c][3] += 1
                    if rk <= 3: loc[c][4] += 1
        print(f"[{name}]  radius={radius:.0f}  half={half}  miss_frames={len(miss)}"
              f"  mean|theta/frame|={mean_theta:.3f}deg")
        for c in chans:
            t1, t3, n, mt1, mt3, mn_ = loc[c]
            for k in range(6):
                agg[c][k] += loc[c][k]
            f1 = t1 / n if n else 0
            f3 = t3 / n if n else 0
            mf1 = mt1 / mn_ if mn_ else 0
            mf3 = mt3 / mn_ if mn_ else 0
            tag = " <-app" if c in APP_CHANS else ""
            print(f"   {c:9} all: top1={f1:.3f} top3={f3:.3f} (n={n})   "
                  f"MISS: top1={mf1:.3f} top3={mf3:.3f} (n={mn_}){tag}")
        print()
    print("=" * 74)
    print("OVERALL (all probed clips)")
    for c in chans:
        t1, t3, n, mt1, mt3, mn_ = agg[c]
        tag = " <-app" if c in APP_CHANS else ""
        print(f"   {c:9} all: top1={t1/n if n else 0:.3f} top3={t3/n if n else 0:.3f}   "
              f"MISS: top1={mt1/mn_ if mn_ else 0:.3f} top3={mt3/mn_ if mn_ else 0:.3f} "
              f"(miss_n={mn_}){tag}")
    print("\nGATE: an appearance channel whose MISS-top1 clears ~0.4 on t8 (motion ~0.32)")
    print("AND is not just re-deriving motion's ranking => build it as a 4th additive")
    print("emission term + re-sweep. If app~=motion or app<<motion, the lever is dead.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=["t8", "t1", "t5", "t4"])
    ap.add_argument("--win", type=int, default=12)
    ap.add_argument("--base-n", type=int, default=8)
    ap.add_argument("--miss-mode", default="fpath_fuse")
    args = ap.parse_args()
    clips = [c for c in _default_clips()
             if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.win, args.miss_mode, args.base_n)


if __name__ == "__main__":
    main()
