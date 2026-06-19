"""STEP-2 sweep: tune the additive (cmass_w, curl_w) emission weights for the
`fpath_fuse` Viterbi integrator, honestly (leave-one-clip-out, no per-clip regression).

Exploits a structural shortcut: with prox_w=0 and reacquire=False (fpath's shipped
config), the per-frame NORMALIZED channels (mass / coherent-mass / curl, each scaled
by its own running peak EMA) and the Viterbi transition penalty are INDEPENDENT of
the emission weights. Only the weighted sum emis = mass_n + cmass_w*cmass_n +
curl_w*curl_n changes. So the expensive optical-flow pass runs ONCE per clip; every
weight config is a fast pure-python re-decode of the same precomputed channels.

The decode here is a faithful copy of track_fused_path_identity's trellis for that
config (trans_w=1.0, trans_cap=None, decode=argmax cumulative). The winning weights
are pinned into identity.FPATH_FUSE_* and re-verified by the real eval_modes harness.

    python -m ld.detect.fuse_sweep --weights .../best.pt
"""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

from ld.detect.eval_modes import _default_clips
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import (_box_coherent_mass, _box_rotational_curl,
                                _box_saliency_mass, _centroid, _seed,
                                compute_countdown_lock, FPATH_TRANS_W,
                                FPATH_MASS_EMA, FPATH_FUSE_WIN)
from ld.vision.cursor import strip_pointer
from ld.vision.motion import estimate_motion, saliency_map
from ld.capture.video_source import VideoSource


def _precompute_channels(weights, clip: Path, wmax: int):
    """One flow pass -> per active frame, the window-INDEPENDENT mass channel plus the
    per-(box, lag) single-frame coherence/curl CONTRIBUTIONS up to lag wmax. Any window
    W <= wmax is then reconstructed by summing the first W lag columns (see _windowed),
    so a window sweep costs one flow pass instead of one per window."""
    packs = detect_fusion_clip(weights, clip, use_cache=True)
    _lock = compute_countdown_lock(packs, clip)
    _sx, _sy, radius, start = _seed(packs)
    lock_frame = _lock.frame if _lock is not None else start

    from collections import deque
    hist: deque = deque(maxlen=wmax)  # newest-first window of past outlier fields
    mass_scale = 0.0
    prev_gray = None
    frames = []  # (idx, cents, mass_n, cmass_contrib[n,L], curl_contrib[n,L], gt, active)
    src = VideoSource(clip)
    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        gray = cv2.cvtColor(strip_pointer(raw, strip_green=True), cv2.COLOR_BGR2GRAY)
        active = idx >= lock_frame
        if prev_gray is not None and active and p.boxes:
            fld = estimate_motion(prev_gray, gray)
            sal = saliency_map(fld, gray.shape)
            cents = [_centroid(b) for b in p.boxes]
            mass = np.array([_box_saliency_mass(b, sal) for b in p.boxes], np.float32)
            fmax = float(mass.max())
            mass_scale = fmax if mass_scale == 0.0 else max(fmax, FPATH_MASS_EMA * mass_scale + (1 - FPATH_MASS_EMA) * fmax)
            mass_n = mass / mass_scale if mass_scale > 0 else mass
            hist.appendleft((fld.outliers, fld.outlier_vectors))
            # contribution from each past frame (lag k) to each current box, kept separate
            # so any window length is a partial sum of these columns.
            L = len(hist)
            cm = np.zeros((len(p.boxes), L), np.float32)
            cu = np.zeros((len(p.boxes), L), np.float32)
            for k, ov in enumerate(hist):
                w1 = [ov]
                for bi, b in enumerate(p.boxes):
                    cm[bi, k] = _box_coherent_mass(b, w1)
                    cu[bi, k] = _box_rotational_curl(b, w1)
            frames.append((idx, cents, mass_n, cm, cu, p.gt, True))
        else:
            frames.append((idx, None, None, None, None, p.gt, False))
        prev_gray = gray
    src.release()
    name = clip.stem.replace("_cropped_trimmed", "")
    return dict(name=name, frames=frames, radius=radius, start=start, lock_frame=lock_frame)


def _windowed(clipdata, win):
    """Build per-frame normalized (mass, cmass, curl) channels for window `win` by summing
    the first `win` lag-columns of the precomputed contributions, then cross-frame
    EMA-normalizing exactly as the production trellis does. Returns a frames list of
    (idx, cents, mass_n, cmass_n, curl_n, gt, active)."""
    cmass_scale = curl_scale = 0.0
    out = []
    for (idx, cents, mass_n, cm, cu, gt, active) in clipdata["frames"]:
        if active and cents is not None:
            cmass_raw = cm[:, :win].sum(axis=1)
            curl_raw = cu[:, :win].sum(axis=1)
            cpk = float(cmass_raw.max())
            cmass_scale = cpk if cmass_scale == 0.0 else max(cpk, FPATH_MASS_EMA * cmass_scale + (1 - FPATH_MASS_EMA) * cpk)
            cmass_n = cmass_raw / cmass_scale if cmass_scale > 0 else cmass_raw
            upk = float(curl_raw.max())
            curl_scale = upk if curl_scale == 0.0 else max(upk, FPATH_MASS_EMA * curl_scale + (1 - FPATH_MASS_EMA) * upk)
            curl_n = curl_raw / curl_scale if curl_scale > 0 else curl_raw
            out.append((idx, cents, mass_n, cmass_n, curl_n, gt, True))
        else:
            out.append((idx, None, None, None, None, gt, False))
    return out


def _decode(frames, radius, start, cmass_w, curl_w, trans_w=FPATH_TRANS_W):
    """Faithful copy of the fpath trellis decode for one weight config (prox_w=0,
    reacquire off). Returns within_r over scored frames (idx >= start)."""
    prev_alpha = None
    prev_cents = None
    last_xy = None
    hits = tot = 0
    for (idx, cents, mass_n, cmass_n, curl_n, gt, active) in frames:
        if active and cents is not None:
            emis = mass_n + cmass_w * cmass_n + curl_w * curl_n
            if prev_alpha is None:
                alpha = emis.copy()
            else:
                alpha = np.empty(len(cents), np.float32)
                for i, (cx, cy) in enumerate(cents):
                    best = -1e18
                    for j, (px, py) in enumerate(prev_cents):
                        dd = math.hypot(cx - px, cy - py) / radius
                        v = prev_alpha[j] - trans_w * dd * dd
                        if v > best:
                            best = v
                    alpha[i] = best + emis[i]
            alpha = alpha - float(alpha.max())
            choice = int(np.argmax(alpha))
            last_xy = cents[choice]
            prev_alpha, prev_cents = alpha, cents
        else:
            prev_alpha = prev_cents = None  # coast: wipe trellis, carry last pos
        if idx >= start and gt is not None and last_xy is not None:
            tot += 1
            if math.hypot(last_xy[0] - gt[0], last_xy[1] - gt[1]) < radius:
                hits += 1
    return hits / tot if tot else 0.0


def run(weights, clips, cmass_ws, curl_ws, wins):
    wmax = max(wins)
    data = [_precompute_channels(weights, c, wmax) for c in clips]
    names = [d["name"] for d in data]
    # per (clip, win) windowed-channel frames, built once and reused across weights.
    wf = {(d["name"], w): _windowed(d, w) for d in data for w in wins}
    rad = {d["name"]: d["radius"] for d in data}
    st = {d["name"]: d["start"] for d in data}

    def dec(name, win, cm, cu):
        return _decode(wf[(name, win)], rad[name], st[name], cm, cu)

    # baseline = channels off == pure mass == fpath (window-independent)
    base = {n: dec(n, wins[0], 0.0, 0.0) for n in names}
    base_mean = statistics.mean(base.values())
    print("pure-mass baseline (cmass=0,curl=0) per clip:")
    for n in names:
        print(f"   {n:>4}: {base[n]:.3f}")
    print(f"   MEAN: {base_mean:.4f}\n")

    cfgs = [(w, cm, cu) for w in wins for cm in cmass_ws for cu in curl_ws]
    scored = {}  # cfg=(win,cmass,curl) -> {name: within_r}
    print(f"{'win':>4} {'cmass':>6} {'curl':>5} | " + " ".join(f"{n:>5}" for n in names) + " |   MEAN  worst_d")
    for cfg in cfgs:
        w, cm, cu = cfg
        wr = {n: dec(n, w, cm, cu) for n in names}
        scored[cfg] = wr
        mean = statistics.mean(wr.values())
        worst_d = min(wr[n] - base[n] for n in names)
        print(f"{w:4d} {cm:6.2f} {cu:5.2f} | "
              + " ".join(f"{wr[n]:5.3f}" for n in names)
              + f" | {mean:6.3f}  {worst_d:+.3f}")

    # Leave-one-clip-out: pick, per held-out clip, the config maximizing the worst
    # TRAIN-fold delta vs base (no-regression bar), tie-break by train mean, then most
    # conservative (smallest window, then smallest weights).
    print("\n=== leave-one-clip-out (no per-clip regression on train folds) ===")
    loo = []
    for hi, held in enumerate(names):
        best_cfg, best_key = None, None
        for cfg in cfgs:
            wr = scored[cfg]
            deltas = [wr[n] - base[n] for k, n in enumerate(names) if k != hi]
            worst_d = min(deltas)
            mean_d = statistics.mean(deltas)
            if worst_d < -0.004:
                continue
            key = (round(worst_d, 4), round(mean_d, 4), -cfg[0], -(cfg[1] + cfg[2]))
            if best_key is None or key > best_key:
                best_key, best_cfg = key, cfg
        if best_cfg is None:
            loo.append(base[held])
            print(f"   {held:>4}: no admissible cfg -> base {base[held]:.3f}")
        else:
            hv = scored[best_cfg][held]
            loo.append(hv)
            print(f"   {held:>4}: {hv:.3f} (base {base[held]:.3f}, {hv-base[held]:+.3f})"
                  f"  cfg=win{best_cfg[0]} cmass{best_cfg[1]} curl{best_cfg[2]}")
    loo_mean = statistics.mean(loo)
    worst = min(loo[i] - base[names[i]] for i in range(len(names)))
    print(f"\nLOO mean within_r = {loo_mean:.4f}  (base {base_mean:.4f}, {loo_mean-base_mean:+.4f})"
          f"  worst_clip={worst:+.3f}")
    # best in-sample cfg with no per-clip regression, for the ship decision
    adm = [c for c in cfgs if min(scored[c][n] - base[n] for n in names) >= -0.004]
    if adm:
        bc = max(adm, key=lambda c: statistics.mean(scored[c].values()))
        print(f"best NO-REGRESSION cfg = win{bc[0]} cmass{bc[1]} curl{bc[2]} "
              f"mean={statistics.mean(scored[bc].values()):.4f}")
    bc2 = max(cfgs, key=lambda c: statistics.mean(scored[c].values()))
    print(f"best in-sample cfg      = win{bc2[0]} cmass{bc2[1]} curl{bc2[2]} "
          f"mean={statistics.mean(scored[bc2].values()):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--cmass", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--curl", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--wins", nargs="+", type=int, default=[8, 12, 16, 20])
    args = ap.parse_args()
    clips = _default_clips()
    if args.clips:
        clips = [c for c in clips
                 if c.stem.replace("_cropped_trimmed", "") in args.clips]
    run(args.weights, clips, args.cmass, args.curl, args.wins)


if __name__ == "__main__":
    main()
