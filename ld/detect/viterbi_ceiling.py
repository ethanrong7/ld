"""Fixed-lag Viterbi ceiling for the position-field signal.

The `field` tracker is a *causal greedy* decision over a per-frame saliency signal
(independent-motion outlier density, snapped to a YOLO box). This asks the strategic
question the diagnosis left open: **given that exact signal, what is the best a
temporal integrator could do with a bounded lookahead of K frames?**

  emission e_t(i)   = saliency mass inside box i at frame t (the field "mass" signal)
  transition c(i,j) = spatial-continuity penalty between chosen centroids
                      (physics: the true shape never moves >~1 radius/frame)

A forward Viterbi trellis (alpha[t] depends only on frames <= t, so it is causal)
maximizes cumulative emission - transition. A fixed-lag decode emits frame (t-K) by
backtracking K steps from the best terminal state at frame t:

  K=0    -> causal filtering (no lookahead)        ~= what `field` competes against
  K=15   -> ~0.5s lag, physically free (see CLAUDE.md "Latency budget")
  K=inf  -> full offline Viterbi (absolute ceiling, not deployable)

Reading the K-curve:
  flat from K=0           -> SIGNAL-limited: lookahead can't help; invest upstream
                             (box-level residual, rotation-as-selector).
  rises then plateaus     -> INTEGRATION-limited: ship a fixed-lag smoother (hold K
                             frames, emit t-K) -- legitimately online.

Reuses the EXACT field signal (estimate_motion -> saliency_map -> _box_saliency_mass)
so the comparison to `field` (LEADERBOARD.md) is apples-to-apples.

Usage:
    python -m ld.detect.viterbi_ceiling \
        --weights data/detect/runs/yolov8n_combined/weights/best.pt
"""
from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ld.capture.video_source import VideoSource
from ld.config import DATA_DIR
from ld.detect.eval_modes import _default_clips, _frame_wh
from ld.detect.fusion import detect_fusion_clip
from ld.detect.identity import _box_saliency_mass, _centroid, _seed
from ld.vision.cursor import strip_pointer
from ld.vision.motion import estimate_motion, saliency_map

CEILING_MD = Path(__file__).resolve().parent / "VITERBI_CEILING.md"

# Transition penalty. Physics: median real-shape step 1.3px, p90 4.5, max 44.7,
# radius ~56. Penalize jumps quadratically in radii; no hard gate (keeps the trellis
# connected across detection gaps). TRANS_W swept in main().
LAGS = (0, 5, 15, 30, 10**9)   # 10**9 stands in for K=inf (full offline)


@dataclass
class FrameObs:
    idx: int
    cents: list[tuple[float, float]] = field(default_factory=list)  # box centroids
    emis: list[float] = field(default_factory=list)                 # saliency mass
    gt_dist: list[float] = field(default_factory=list)              # |centroid - GT|
    has_gt: bool = False


def collect_obs(weights: str, clip: Path) -> tuple[list[FrameObs], int, float]:
    """One forward pass: per-box saliency mass (the field signal) + GT distance.

    Mirrors track_field_identity's signal exactly: estimate_motion -> saliency_map,
    then per-box mass via _box_saliency_mass.
    """
    packs = detect_fusion_clip(weights, clip, use_cache=True)
    _, _, radius, start = _seed(packs)
    obs: list[FrameObs] = []
    prev_gray = None
    src = VideoSource(clip)
    for idx, raw in src.frames():
        if idx >= len(packs):
            break
        p = packs[idx]
        gray = cv2.cvtColor(strip_pointer(raw, strip_green=True), cv2.COLOR_BGR2GRAY)
        o = FrameObs(idx, has_gt=p.gt is not None)
        sal = None
        if prev_gray is not None:
            fld = estimate_motion(prev_gray, gray)
            sal = saliency_map(fld, gray.shape)
        for b in p.boxes:
            cx, cy = _centroid(b)
            o.cents.append((cx, cy))
            o.emis.append(_box_saliency_mass(b, sal) if sal is not None and sal.max() > 0 else 0.0)
            o.gt_dist.append(math.hypot(cx - p.gt[0], cy - p.gt[1]) if p.gt is not None else float("nan"))
        obs.append(o)
        prev_gray = gray
    src.release()
    return obs, start, radius


def _norm_emis(e: list[float]) -> np.ndarray:
    """Normalize a frame's emissions to [0,1] (max-scaled) so TRANS_W is comparable
    across clips/frames regardless of absolute saliency magnitude."""
    a = np.asarray(e, np.float32)
    m = a.max()
    return a / m if m > 0 else a


def fixed_lag_viterbi(obs: list[FrameObs], radius: float, trans_w: float,
                      lag: int) -> dict[int, int]:
    """Forward Viterbi with fixed-lag decode. Returns {frame_idx -> chosen box index}.

    Causal: alpha[t] uses only frames <= t. At each t, once t-lag is reachable, emit
    the box for frame (t-lag) by backtracking `lag` steps from the best state at t.
    State space = box indices present that frame (variable, handles detection gaps).
    """
    decoded: dict[int, int] = {}
    # alpha: best cumulative score ending in each box of the current frame
    prev_cents: list[tuple[float, float]] | None = None
    alpha: np.ndarray | None = None
    back: list[list[int]] = []        # back[t][i] = best predecessor box index at t-1
    frames_with_boxes: list[int] = []  # positions in `back`/trellis -> obs index

    for t, o in enumerate(obs):
        n = len(o.cents)
        if n == 0:
            # no boxes: trellis breaks; reset (rare, but keeps it robust)
            prev_cents, alpha = None, None
            back.append([])
            frames_with_boxes.append(t)
            continue
        emis = _norm_emis(o.emis)
        if alpha is None or prev_cents is None:
            alpha = emis.copy()
            back.append([-1] * n)
        else:
            new_alpha = np.full(n, -1e18, np.float32)
            bp = [-1] * n
            for i, (cx, cy) in enumerate(o.cents):
                best_j, best_v = -1, -1e18
                for j, (px, py) in enumerate(prev_cents):
                    d = math.hypot(cx - px, cy - py)
                    v = alpha[j] - trans_w * (d / radius) ** 2
                    if v > best_v:
                        best_v, best_j = v, j
                new_alpha[i] = best_v + emis[i]
                bp[i] = best_j
            alpha = new_alpha
            back.append(bp)
        prev_cents = o.cents
        frames_with_boxes.append(t)

        # fixed-lag emit: backtrack `lag` steps from current best terminal state
        if alpha is not None and len(alpha) > 0:
            cur = int(np.argmax(alpha))
            steps = min(lag, len(back) - 1)
            ti = len(back) - 1
            node = cur
            for _ in range(steps):
                pj = back[ti][node] if back[ti] else -1
                if pj < 0:
                    break
                node = pj
                ti -= 1
            target_frame = frames_with_boxes[ti]
            if target_frame not in decoded:
                decoded[target_frame] = node

    # flush the tail (frames within `lag` of the end): decode from final best path
    if alpha is not None and len(alpha) > 0:
        node = int(np.argmax(alpha))
        ti = len(back) - 1
        while ti >= 0:
            tf = frames_with_boxes[ti]
            if back[ti]:
                if tf not in decoded:
                    decoded[tf] = node
                pj = back[ti][node] if node < len(back[ti]) else -1
                if pj < 0:
                    break
                node = pj
            ti -= 1
    return decoded


def score_decode(obs: list[FrameObs], decoded: dict[int, int], start: int,
                 radius: float) -> float:
    """within_r over scored frames (idx >= start, has GT)."""
    hit = tot = 0
    for t, o in enumerate(obs):
        if o.idx < start or not o.has_gt or not o.cents:
            continue
        tot += 1
        i = decoded.get(t)
        if i is not None and i < len(o.gt_dist) and o.gt_dist[i] < radius:
            hit += 1
    return hit / tot if tot else 0.0


TRANS_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
FIELD_WITHIN_R = {  # current leader, from LEADERBOARD.md, for side-by-side
    "t1": 0.804, "t2": 0.873, "t3": 0.692, "t4": 0.715, "t5": 0.409,
    "t6": 0.819, "t7": 0.796, "t8": 0.589, "t9": 0.751, "t10": 0.698,
}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed-lag Viterbi ceiling over the field signal")
    ap.add_argument("--weights", default="data/detect/runs/yolov8n_combined/weights/best.pt")
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()
    clips = ([Path(c) if Path(c).exists() else DATA_DIR / f"{c}_cropped_trimmed.mp4"
              for c in args.clips] if args.clips else _default_clips())

    # collect observations once per clip (cache-backed detections; reads frames for saliency)
    data: dict[str, tuple[list[FrameObs], int, float]] = {}
    for clip in clips:
        name = clip.stem.replace("_cropped_trimmed", "")
        print(f"[{name}] collecting field signal ...")
        data[name] = collect_obs(args.weights, clip)

    # For each lag, pick the best TRANS_W by MEAN within_r across clips (the trans
    # weight is a property of the integrator, chosen once -- honest, not per-clip).
    print("\nsweeping trans_w x lag ...")
    # results[lag] = (best_trans_w, {clip: within_r}, mean)
    results: dict[int, tuple[float, dict[str, float], float]] = {}
    for lag in LAGS:
        best = None
        for tw in TRANS_GRID:
            per = {}
            for name, (obs, start, radius) in data.items():
                dec = fixed_lag_viterbi(obs, radius, tw, lag)
                per[name] = score_decode(obs, dec, start, radius)
            m = _mean(list(per.values()))
            if best is None or m > best[2]:
                best = (tw, per, m)
        results[lag] = best
        lag_lbl = "inf" if lag >= 10**8 else str(lag)
        print(f"  K={lag_lbl:>3}: best trans_w={best[0]:<4} mean within_r={best[2]:.3f}")

    write_report(results, data, args.weights)


def write_report(results, data, weights: str) -> None:
    names = list(data.keys())
    field_mean = _mean([FIELD_WITHIN_R.get(n) for n in names])
    L: list[str] = []
    L.append("# Fixed-lag Viterbi ceiling (over the `field` signal)")
    L.append("")
    L.append(f"Weights: `{weights}` · clips: {', '.join(names)}")
    L.append("")
    L.append("Best a temporal integrator could do with a bounded lookahead of K "
             "frames, using the **exact** per-frame signal the `field` tracker "
             "consumes (independent-motion saliency mass per YOLO box). K=0 causal · "
             "K=15 ~0.5s lag (physically free) · K=inf full offline. `trans_w` chosen "
             "once per lag by best mean across clips (no per-clip tuning).")
    L.append("")
    header = "| clip | field (causal) | " + " | ".join(
        f"K={'inf' if k>=10**8 else k}" for k in LAGS) + " |"
    L.append(header)
    L.append("|------|" + "---:|" * (len(LAGS) + 1))
    for n in names:
        row = [f"{FIELD_WITHIN_R.get(n, float('nan')):.3f}"]
        for k in LAGS:
            row.append(f"{results[k][1][n]:.3f}")
        L.append(f"| {n} | " + " | ".join(row) + " |")
    mean_row = [f"**{field_mean:.3f}**"]
    for k in LAGS:
        mean_row.append(f"**{results[k][2]:.3f}**")
    L.append("| **mean** | " + " | ".join(mean_row) + " |")
    L.append("")
    L.append("trans_w per K: " + ", ".join(
        f"K={'inf' if k>=10**8 else k}:{results[k][0]}" for k in LAGS))
    L.append("")

    # automatic reading
    k0 = results[0][2]
    k15 = results[15][2]
    kinf = results[10**9][2]
    L.append("## Reading")
    L.append("")
    L.append(f"- Causal ceiling (K=0) over this signal = **{k0:.3f}**; "
             f"`field` actual = **{field_mean:.3f}**.")
    gain_lag = k15 - k0
    gain_offline = kinf - k0
    L.append(f"- Lookahead gain: K=0→15 = **{gain_lag:+.3f}**, K=0→inf = **{gain_offline:+.3f}**.")
    if gain_lag < 0.02:
        L.append(f"- **SIGNAL-LIMITED.** Bounded lookahead buys ~nothing ({gain_lag:+.3f}); "
                 f"even full offline adds only {gain_offline:+.3f}. The per-frame signal "
                 f"is the ceiling — a fixed-lag smoother is NOT worth building. Invest "
                 f"upstream: box-level rigid residual, rotation-as-selector.")
    else:
        L.append(f"- **INTEGRATION-LIMITED (lookahead helps).** A ~0.5s fixed-lag "
                 f"smoother gains {gain_lag:+.3f} and is legitimately online (hold K "
                 f"frames, emit t-K). Worth building.")
    if field_mean >= k0 - 0.02:
        L.append(f"- Note: `field` ({field_mean:.3f}) ~matches/exceeds the K=0 Viterbi "
                 f"ceiling — the causal greedy is already near-optimal on this signal, "
                 f"reinforcing that the signal (not the decision) is the limiter.")
    L.append("")
    CEILING_MD.write_text("\n".join(L))
    print(f"\nceiling -> {CEILING_MD}")


if __name__ == "__main__":
    main()


