# Plan — Coherent-mass: the untapped signal that separates real from fake on t5/t8

## Root cause of t5/t8 failure (re-diagnosed 2026-06-15, overturns the earlier "identifiability limit" conclusion)

I probed the actual failure structure on the laggards. **The earlier conclusion in CLAUDE.md — that t5/t8 are an "identifiability limit" where no motion signal separates the real shape — was wrong.** The new evidence:

1. **The failure is NOT creep, and NOT a missing signal.** On t5/t8, when `field_lag` is wrong it is wrong by **median 180–200 px (3–4 shape radii)**, in long **confident blocks** (median 19–28 frames, up to 86), in `track` state (not `reacquire`). field **confidently locks onto a fake shape and tracks it for dozens of frames**. (36% of t5 frames, 40% of t8 are "oracle-hit but field-miss" = a box is on the real shape but field is elsewhere.)

2. **The real shape's signal IS present on those exact frames.** On the fake-locked frames, the GT box is the **#1 instantaneous-saliency box 33–44%** of the time and **top-3 ~74–82%**. field's single CV-gated track is simply locked 180 px away and never reconsiders the strong evidence elsewhere.

3. **Why every prior fix failed:** instantaneous saliency mass at the GT box beats the fake field is locked onto only **~50% (a coin-flip)**. So "switch to global-max mass" (router, fpath, prox-fusion) is pure noise on the confusable frames — confirmed by the whole dead-end list. **The scalar field uses (instantaneous outlier-density mass) genuinely cannot separate them.**

**So the binding limiter is: a strong but ambiguous instantaneous signal, integrated by a single locked track that has no mechanism to re-evaluate a competing candidate that sustains coherent evidence.**

## The untapped signal: directional + temporal coherence of the outlier field

`ld/vision/motion.py` `saliency_map` (lines 90–93) stamps only the **magnitude** of each motion-outlier feature, discarding its **direction**. But the real shape's defining physics is *rigid independent translation*: its outlier vectors all point the **same way** (coherent), and that direction **persists across frames**. A fake's spurious outliers (LK noise, occlusion/relief edges) point every which way and don't persist. **Every channel ever tried — outlier density, box-rigid residual, rotation, NCC/log-polar appearance — is a magnitude or a single-frame quantity. None used directional coherence, and none accumulated it over time.**

I defined **accumulated coherent-mass** per box over a causal window W:
- per frame, per box: `coherent_mass = ||Σ resid_vectors|| · (||Σ resid_vectors|| / Σ||resid_vectors||)` — the resultant of the outlier vectors inside the box, weighted by their directional coherence (0..1). Incoherent jitter cancels in the sum; coherent drift survives.
- accumulate that over the last W≈12 frames at the box's current location.

### Measured result (on field's fake-locked miss frames — the frames that matter)

| signal | t5 top1 | t5 top3 | t8 top1 | t8 top3 |
|---|---|---|---|---|
| instantaneous mass (what field uses) | 0.33 | 0.82 | 0.44 | 0.74 |
| outlier-vector coherence (1 frame) | 0.45 | 0.70 | 0.40 | 0.67 |
| windowed-accum mass (magnitude only) | 0.28 | 0.89 | 0.41 | 0.75 |
| **accumulated coherent-mass (W=12)** | **0.73** | **0.86** | **0.41** | **0.70** |

On strong-clip miss frames it also holds (t2 0.86, t7 0.69, t10 0.47) — not a t5 fluke, not a trivial GT bias.

### The causal key (what every prior signal lacked)

The signal's own **margin** (top score vs runner-up, normalized) separates correct from incorrect picks by **+0.19 (t5), +0.21 (t8), +0.30 (t1), +0.36 (t4)** — high margin ⇒ more often correct, the *right* direction. fpath's path-margin was anti-correlated (conviction = lock-in); this is the opposite. **This is the first candidate signal with a usable live confidence**, so it can arbitrate online rather than being an unreachable oracle (the router dead end).

## Implementation strategy

Add accumulated coherent-mass as a **second opinion that can break field's lock**, gated by its causal margin key — NOT a replacement leader (avoids the 0.157 trajectory-led collapse), NOT an emission fusion into field's CV model (avoids the prox-fusion dilution).

### Stage 1 — full pre-build gate (read-only, ~150 lines, no retrain)
Before touching `field`, prove the end-to-end win with a gate that:
1. Computes accumulated coherent-mass per box per frame for all 10 clips (re-runs flow; ~3 min/clip — reuse the `estimate_motion` internals but keep residual *vectors*, which `motion.py` currently discards).
2. Defines a **challenger rule**: when the coherent-mass leader is a box far from field's current pick AND its margin key exceeds a threshold τ for ≥C consecutive frames, override field's pick to the challenger.
3. Sweeps (τ, C, W) and scores real `within_r` through `score_identity`, **leave-one-clip-out**.
4. **Accept only if LOO mean improves with no per-clip regression > 0.004** (the project bar). Guard t2/t7 (strong) explicitly.

If Stage 1 fails the bar, it dies for ~150 lines like every other gate, and the identifiability conclusion is *re-confirmed* from a new angle. If it clears, proceed.

### Stage 2 — productionize into the signal source
1. Extend `ld/vision/motion.py`: have `MotionField`/`estimate_motion` retain per-outlier **residual vectors** (currently only magnitudes survive into `outlier_weights`), and add a `coherent_mass_map` or a per-box helper. Keep `saliency_map` byte-identical so `field` is unchanged and the A/B baseline holds.
2. Add the challenger-override into `track_field_identity` (or a thin wrapper `field_coh` mode, mirroring how `field_lag` wraps `field`) behind config flags, dispatched via `_dispatch_mode`, registered in `ALL_MODES`. Default stays `field_lag` until the new mode wins LOO.
3. Re-run `loo.py` + `eval_modes.py`; update `LEADERBOARD.md` only on a held-out win.

### Composability with field_lag
The challenger-override operates on the pick sequence, so it composes with the shipped fixed-lag confirmation (`field_lag`). Likely final form: `field` → coherent-mass challenger → fixed-lag confirm. Validate the stack end-to-end in Stage 1.

## Key files
- `ld/vision/motion.py` — `estimate_motion` discards residual *vectors* (line 81 keeps only magnitudes); this is the change to expose direction. `saliency_map` lines 85–96.
- `ld/detect/identity.py` — `track_field_identity` (1060), snap/pick at 1133–1148; `_box_saliency_mass` (1051); `_dispatch_mode`/`ALL_MODES`.
- `ld/detect/eval_modes.py`, `ld/detect/loo.py` — scoring harnesses.

## Risks / kill-criteria
- **t8 is weaker** (top1 0.41, not 0.73). The override must help t5 without hurting t8; if the margin gate can't be tuned to do both LOO, ship t5-only behavior only if no t8 regression.
- **Override could destabilize strong clips** (t2/t7) by second-guessing a correct lock. The margin key + consecutive-frame requirement C is the guard; Stage 1 must show strong clips flat.
- If LOO doesn't clear +0 with no regression, **do not ship** — record as the definitive test of the coherence channel and revert to `field_lag`.
