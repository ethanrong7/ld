"""GATE Stage 1: coherent-mass challenger-override on field's pick sequence.

ROOT CAUSE (re-diagnosed 2026-06-15): on t5/t8 `field` confidently locks onto a
FAKE shape for long blocks (~180px / 3-4 radii off, in `track` state). The real
shape's signal IS present (GT box is top-3 instantaneous-saliency ~78% on those
frames) but field's single CV-gated track never reconsiders it. Instantaneous mass
is a coin-flip (~0.50) between the real shape and the locked fake -> every global-
switch fix (router/fpath/prox) was noise. The untapped separator is the DIRECTIONAL
+ TEMPORAL COHERENCE of the outlier field, which `saliency_map` discards (it keeps
only magnitudes).

This gate tests, end-to-end and LEAVE-ONE-CLIP-OUT, whether a margin-gated
coherent-mass challenger that can OVERRIDE field's locked pick improves real
within_r with no per-clip regression.

  coherent_mass(box, frame) = ||sum resid_vectors_in_box|| * coherence,
      coherence = ||sum resid_vectors|| / sum||resid_vectors||  (0..1)
  accumulated over a causal window W at the box's CURRENT location.
  challenger = argmax accumulated coh-mass; margin = (top-runnerup)/(top+eps)  [causal key]
  OVERRIDE field's emitted (x,y) to the challenger centroid when, for >=C consecutive
  frames, the challenger differs from field's pick, is >FAR radii away, and margin>=TAU.

Read-only on the cache + video; re-runs flow once per clip (vectors not cached by
motion.py). Run, read, decide. Delete after.
"""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

from ld.config import (
    DATA_DIR, FEAT_MAX, FEAT_QUALITY, FEAT_MIN_DIST, LK_WIN, LK_LEVELS,
    RANSAC_THRESH, OUTLIER_RESID_MIN,
)
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import (
    _centroid, _dispatch_mode, compute_countdown_lock, score_identity,
)
from ld.detect.eval_modes import _frame_wh, _default_clips
from ld.vision.cursor import strip_pointer
from ld.capture.video_source import VideoSource


def _outlier_vectors(prev_gray, cur_gray):
    """Re-derive outlier feature positions + residual VECTORS (motion.py keeps only
    magnitudes). Returns (positions_cur, resid_vecs, mags) or None."""
    p0 = cv2.goodFeaturesToTrack(prev_gray, FEAT_MAX, FEAT_QUALITY, FEAT_MIN_DIST)
    if p0 is None:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, p0, None, winSize=(LK_WIN, LK_WIN), maxLevel=LK_LEVELS)
    keep = st.ravel() == 1
    a = p0.reshape(-1, 2)[keep]
    b = p1.reshape(-1, 2)[keep]
    if len(a) < 30:
        return None
    T, _ = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH)
    if T is None:
        return None
    pred = (a @ T[:, :2].T) + T[:, 2]
    resid = b - pred
    mag = np.linalg.norm(resid, axis=1)
    out = mag > OUTLIER_RESID_MIN
    return b[out], resid[out], mag[out]


def _box_coherent_mass(box, ov):
    """Resultant-of-outlier-vectors inside box, weighted by directional coherence."""
    if ov is None:
        return 0.0
    pix, resid, mag = ov
    if pix.shape[0] == 0:
        return 0.0
    inb = ((pix[:, 0] >= box[0]) & (pix[:, 0] <= box[2])
           & (pix[:, 1] >= box[1]) & (pix[:, 1] <= box[3]))
    if inb.sum() < 2:
        return 0.0
    v = resid[inb]
    net = float(np.linalg.norm(v.sum(0)))
    tot = float(mag[inb].sum())
    return net * (net / tot) if tot > 1e-6 else 0.0


def _compute_ov(clip: Path, packs):
    """Per-frame outlier vectors, computed once (the expensive flow pass). Cached to
    disk keyed by clip stem so the (tau,C,W) sweep doesn't repay the flow cost."""
    import pickle
    cache = DATA_DIR / "detect" / "cache" / f"_cohgate_ov_{clip.stem}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    src = VideoSource(clip)
    grays = {}
    for idx, frame in src.frames():
        grays[idx] = cv2.cvtColor(strip_pointer(frame, strip_green=True), cv2.COLOR_BGR2GRAY)
    src.release()
    idxs = sorted(grays)
    ov = {}
    for pos, idx in enumerate(idxs):
        if pos == 0 or idx >= len(packs):
            continue
        ov[idx] = _outlier_vectors(grays[idxs[pos - 1]], grays[idx])
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump((idxs, ov), f)
    return idxs, ov


def _challenger_per_frame(packs, idxs, ov, win):
    """For each frame: (challenger_centroid, margin, challenger_box) by accumulated
    coherent-mass over the last `win` frames at each current box's location."""
    pos_of = {idx: pos for pos, idx in enumerate(idxs)}
    out = {}
    for idx in idxs:
        if idx >= len(packs):
            continue
        p = packs[idx]
        if not p.boxes:
            out[idx] = None
            continue
        pos = pos_of[idx]
        scores = []
        for b in p.boxes:
            acc = 0.0
            for back in range(0, win):
                j = pos - back
                if j < 0:
                    break
                jidx = idxs[j]
                acc += _box_coherent_mass(b, ov.get(jidx))
            scores.append(acc)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = scores[order[0]]
        run = scores[order[1]] if len(order) > 1 else 0.0
        margin = (top - run) / (top + 1e-6) if top > 0 else 0.0
        cx, cy = _centroid(p.boxes[order[0]])
        out[idx] = ((cx, cy), margin, top)
    return out


def _apply_override(track, radius, challenger, tau, c_consec, far_r):
    """Rewrite field's track: when the coh-mass challenger persistently disagrees,
    is FAR away, and is confident (margin>=tau) for >=c_consec frames, emit it.

    Returns a NEW list of (idx, x, y) with overrides applied; mirrors field_lag's
    pick-sequence post-processing (no CV state, no leader change)."""
    from ld.detect.identity import TrackPoint
    far_px = far_r * radius
    streak = 0
    out = []
    for tp in track:
        ch = challenger.get(tp.idx)
        x, y = tp.x, tp.y
        if ch is not None:
            (chx, chy), margin, _top = ch
            disagree = math.hypot(chx - tp.x, chy - tp.y) > far_px
            if disagree and margin >= tau:
                streak += 1
            else:
                streak = 0
            if streak >= c_consec:
                x, y = chx, chy
        else:
            streak = 0
        out.append(TrackPoint(tp.idx, float(x), float(y), tp.state, tp.n_dets))
    return out


def _probe_headroom(data, win, tau, c, far):
    """Headroom probe: of the frames the FINAL (post-override) track gets WRONG while the
    override stays SILENT (emitted == base pick), how often does the UNGATED coherent-mass
    challenger (per-frame argmax centroid) already sit within radius of GT?

    High -> the channel has lots left a continuous emission-term integrator could capture;
    the crude far-jump override only skims it. Low -> override already skims most of it."""
    print("\n=== headroom probe (ungated challenger on silent+wrong frames) ===")
    tot_wrong_silent = tot_recoverable = 0
    for d in data:
        packs, radius = d["packs"], d["radius"]
        gt = {p.idx: p.gt for p in packs if p.gt is not None}
        base = {tp.idx: tp for tp in d["track"]}
        over = {tp.idx: tp for tp in
                _apply_override(d["track"], radius, d["chal"][win], tau, c, far)}
        chal = d["chal"][win]
        wrong_silent = recoverable = 0
        for idx, g in gt.items():
            if idx not in over:
                continue
            o = over[idx]
            fired = (abs(o.x - base[idx].x) > 1e-6 or abs(o.y - base[idx].y) > 1e-6)
            final_wrong = math.hypot(o.x - g[0], o.y - g[1]) > radius
            if final_wrong and not fired:
                wrong_silent += 1
                ch = chal.get(idx)
                if ch is not None and math.hypot(ch[0][0] - g[0], ch[0][1] - g[1]) <= radius:
                    recoverable += 1
        frac = recoverable / wrong_silent if wrong_silent else 0.0
        tot_wrong_silent += wrong_silent
        tot_recoverable += recoverable
        print(f"  {d['name']:>4}: wrong+silent={wrong_silent:>4}  "
              f"challenger-on-GT={recoverable:>4}  recoverable_frac={frac:.3f}")
    overall = tot_recoverable / tot_wrong_silent if tot_wrong_silent else 0.0
    print(f"\n  OVERALL recoverable fraction = {overall:.3f}  "
          f"({tot_recoverable}/{tot_wrong_silent} wrong+silent frames)")
    print("  HIGH (>~0.3) => build a coherence-weighted EMISSION integrator (lots left);"
          "\n  LOW  (<~0.15) => override already skims the channel; ship it as-is.")


def _build_clip(weights, clip: Path, wins, base_mode="field"):
    """Heavy per-clip setup done once: base track, packs, lock, radius, and the
    challenger map for each window size in `wins`. `base_mode` is the tracker the
    coherence override is layered on top of (field, or field_lag to test stacking)."""
    packs = detect_fusion_clip(weights, clip, use_cache=True)
    lock = compute_countdown_lock(packs, clip)
    wh = _frame_wh(clip)
    name = clip.stem.replace("_cropped_trimmed", "")
    track, _h, start, radius = _dispatch_mode(clip, packs, lock, wh, base_mode)
    idxs, ov = _compute_ov(clip, packs)
    chal = {w: _challenger_per_frame(packs, idxs, ov, w) for w in wins}
    base = score_identity(packs, track, start, radius, name).within_r
    return dict(name=name, packs=packs, track=track, start=start,
                radius=radius, chal=chal, base=base)


def _score_cfg(d, win, tau, c, far):
    ov_track = _apply_override(d["track"], d["radius"], d["chal"][win], tau, c, far)
    return score_identity(d["packs"], ov_track, d["start"], d["radius"],
                          d["name"]).within_r


def run(weights, clips, wins, taus, cs, fars, base_mode="field"):
    data = [_build_clip(weights, c, wins, base_mode) for c in clips]
    for d in data:
        print(f"  built {d['name']}: {base_mode} base within_r={d['base']:.3f}")
    names = [d["name"] for d in data]
    base_mean = statistics.mean(d["base"] for d in data)
    print(f"\n{base_mode} baseline mean within_r = {base_mean:.4f}\n")

    cfgs = [(w, t, c, f) for w in wins for t in taus for c in cs for f in fars]

    # In-sample: best config over all clips (optimistic).
    print("=== in-sample grid (mean within_r, delta vs field) ===")
    scored = {}
    for cfg in cfgs:
        wr = [_score_cfg(d, *cfg) for d in data]
        scored[cfg] = wr
        m = statistics.mean(wr)
        worst = min(wr[i] - data[i]["base"] for i in range(len(data)))
        print(f"  W={cfg[0]:>2} tau={cfg[1]:.2f} C={cfg[2]:>2} far={cfg[3]:.1f} | "
              f"mean={m:.4f} ({m-base_mean:+.4f})  worst_clip={worst:+.4f}")

    # Leave-one-clip-out. The project bar is ROBUSTNESS-FIRST: select the config that
    # maximizes the worst train-fold delta (no regression), tie-break by mean, then by
    # the most CONSERVATIVE config (largest C, then largest tau) so a marginal mean edge
    # can't pick an aggressive override that regresses the held-out clip.
    print("\n=== leave-one-clip-out (honest, robustness-first selection) ===")
    loo_wr = []
    for hi, held in enumerate(names):
        best_cfg, best_key = None, None
        for cfg in cfgs:
            wr = scored[cfg]
            others = [wr[i] - data[i]["base"] for i in range(len(data)) if i != hi]
            mean_d = statistics.mean(others)
            worst_d = min(others)
            if worst_d < -0.004:
                continue
            # key: (worst-fold delta, mean delta, conservative C, conservative tau)
            key = (round(worst_d, 4), round(mean_d, 4), cfg[2], cfg[1])
            if best_key is None or key > best_key:
                best_key, best_cfg = key, cfg
        if best_cfg is None:
            chosen = data[hi]["base"]  # no admissible override -> field
            loo_wr.append(chosen)
            print(f"  {held:>4}: no admissible cfg -> field {chosen:.3f}")
        else:
            heldwr = scored[best_cfg][hi]
            loo_wr.append(heldwr)
            print(f"  {held:>4}: {heldwr:.3f} (base {data[hi]['base']:.3f}, "
                  f"{heldwr-data[hi]['base']:+.3f})  cfg=W{best_cfg[0]} tau{best_cfg[1]} "
                  f"C{best_cfg[2]} far{best_cfg[3]}")
    loo_mean = statistics.mean(loo_wr)
    worst = min(loo_wr[i] - data[i]["base"] for i in range(len(data)))
    print(f"\nLOO mean within_r = {loo_mean:.4f}  (field {base_mean:.4f}, "
          f"{loo_mean-base_mean:+.4f})  worst_clip={worst:+.4f}")
    print("VERDICT:", "PASS - clears bar" if (loo_mean > base_mean and worst >= -0.004)
          else "FAIL - does not clear LOO bar with no regression")

    # When a single config is pinned, run the headroom probe on it.
    if len(cfgs) == 1:
        _probe_headroom(data, *cfgs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--wins", nargs="+", type=int, default=[8, 12, 16])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.3, 0.45, 0.6])
    ap.add_argument("--cs", nargs="+", type=int, default=[3, 5, 8])
    ap.add_argument("--fars", nargs="+", type=float, default=[1.5, 2.0])
    ap.add_argument("--base-mode", default="field",
                    help="tracker the coherence override layers on (field|field_lag)")
    args = ap.parse_args()
    clips = _default_clips()
    if args.clips:
        clips = [c for c in clips
                 if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.wins, args.taus, args.cs, args.fars, args.base_mode)


if __name__ == "__main__":
    main()
