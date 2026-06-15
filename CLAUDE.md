# CLAUDE.md — LD (Lie Detector) real-vs-fake shape tracker

## What this project does

MapleStory's "Lie Detector" minigame shows a sheet of paper ("the sheet") covered
in many **identical** shapes. One shape is **real** (the player must click it); the
rest are **fakes**. The fakes move rigidly with the sheet; the **real shape moves
independently** (it drifts and rotates on its own). This repo takes a screen-capture
clip and, every frame, outputs the pixel position of the real shape.

**This is an ONLINE / LIVE problem.** In production the solver sees frames one at a
time and must emit a position without access to future frames. Any offline,
whole-video technique is valid only as a *diagnostic ceiling*, never as a shippable
tracker. A bounded lookahead (fixed-lag) buffer is permissible — see "Latency
budget" below — but unbounded future access is not.

## Pipeline (two separable stages)

1. **Detection** — a YOLOv8n model (`data/detect/runs/yolov8n_combined/weights/best.pt`)
   finds candidate shape boxes per frame. This stage is **healthy**: oracle within_r
   ≈ **0.93** (a box sits on the real shape ~93% of frames). Detection is NOT the
   bottleneck.

2. **Identity** — decides *which* box is the real shape over time. This is the
   bottleneck and the entire focus of the work. Code in `ld/detect/identity.py`.

## Current best: `field_lag` mode (the default)

`track_field_lag_identity` in `ld/detect/identity.py` — a thin **fixed-lag
confirmation smoother** wrapped around the `field` tracker (below). It defers
committing `field`'s per-frame box pick until a short, physically-free lookahead
window (`FIELD_LAG_K=8` frames) confirms the same box is `field`'s choice in
≥`FIELD_LAG_CONFIRM=0.5` of the window; otherwise it falls back to the raw `field`
pick (do-no-harm). This overrules a transient single-frame **creep** onto an
adjacent fake before it can poison the CV velocity and lock in — the one
empirically-confirmed failure mode. It emits frame `t-K`, so it is legitimately
online (the ~8-frame lag is well inside the latency budget; see below). It is a
confirmation filter on the pick sequence — NOT a velocity cap, NOT a new leader, NOT
an emission fusion (all three of those FAILED; see below).

**Scores:** leave-one-out (honest, held-out) **0.721** (vs `field` 0.693, +0.028),
in-sample **0.721**. The LOO winner (`lag_k=8, confirm=0.5`) is stable across all 10
folds. No clip regresses by more than 0.004 vs `field` (t6 −0.004, t8 −0.003). See
`ld/detect/LEADERBOARD.md`.

## The underlying signal: `field` mode

`track_field_identity` in `ld/detect/identity.py`. Identity-free and
fragmentation-immune; `field_lag` wraps it and `field` remains selectable as a clean
baseline (in-sample 0.714, LOO 0.693):

- Per frame, `ld/vision/motion.py` tracks sparse features prev→cur, fits the global
  rigid sheet motion with RANSAC (`estimate_motion`), and treats motion **outliers**
  (features disagreeing with the sheet) as evidence. `saliency_map` blurs their
  spatial density into a per-frame saliency field.
- `ld/track/tracker.py` `OutlierTracker` follows the saliency peak with a gated
  constant-velocity model; after `REACQUIRE_PATIENCE` low-confidence frames it
  teleports to the global saliency peak ("reacquire").
- The tracked peak is **snapped to a YOLO box** each frame (`snap_mode="mass"` = box
  containing the most saliency; `snap_feedback=True` writes the snapped position back
  into the tracker so it stays box-anchored). **The snap is load-bearing**: field raw
  (no snap) ≈ 0.28, field+snap ≈ 0.59 — saliency says *where* (identity-free), YOLO
  gives the precise box.

## How to evaluate

Everything scores from the cached YOLO detections (no re-inference):

```bash
# Full leaderboard across t1–t10 (regenerates LEADERBOARD.md). ~12 min.
python -m ld.detect.eval_modes --weights data/detect/runs/yolov8n_combined/weights/best.pt

# Subset / single mode (fast). NOTE: a subset run OVERWRITES LEADERBOARD.md with
# only those clips — regenerate the full set afterward.
python -m ld.detect.eval_modes --weights .../best.pt --modes field --clips t5 t8

# Honest held-out generalization (leave-one-clip-out grid search).
python -m ld.detect.loo --weights .../best.pt
```

Per-frame trace CSVs land in `data/detect/eval/<clip>__<mode>.csv` (gitignored):
columns `idx,state,x,y,gt_x,gt_y,err_px,...`. These are the primary debugging tool —
read them to see *how* a track fails (creep vs jump, which state).

`within_r` = fraction of frames the estimate lands inside the shape radius of GT.
Always quote LOO/held-out numbers as the real metric; in-sample is optimistic
(we tune on the same t1–t10 set).

## The measured physics (run on the GT cursor track, all 10 clips)

- Real shape speed: **median 1.3 px/frame, p90 4.5, p99 17.8, max 44.7**.
- Shape radius: **~56 px**.
- So the true shape **never** moves more than ~1 radius in a single frame, and takes
  **~43 frames (1.4s)** at median speed (or ~13 frames at p90) to traverse one radius.

**Latency budget:** a fixed-lag buffer of **~10–15 frames (~0.4–0.5s)** is essentially
free — the shape physically cannot leave its own radius in that window. This bounds
any legitimate online lookahead/smoothing scheme.

## The core diagnosis (do not re-litigate without new evidence)

`ld/detect/diagnose.py` → `ld/detect/DIAGNOSIS.md`. The limiter is **signal +
association fragmentation, NOT online decision logic**:

- A single tracklet holds the real shape only **0.24** of frames (identity fragments
  badly across YOLO boxes because the real shape moves independently → breaks IoU).
- The GT box is the top motion outlier only **0.13** of frames (top-3: 0.37). The
  per-frame signal is weak; the real shape is often nearly stationary or buried.
- `field` (0.71) already far exceeds the single-tracklet ceiling (0.24) — its
  spatial accumulation stitches fragments. **Decision/threshold tuning is low-EV.**

### Failure mode is CREEP, not JUMP (empirically confirmed)

Reading the t5 `field` trace: the track is first lost via **legitimate small steps**
(~12 px, in `track` state) when an adjacent fake's saliency transiently wins — a
*creep* onto the wrong box. The big teleports (300+ px) are all `reacquire` and fire
*after* the track is already lost. **Causal order: creep loses it first → reacquire
teleports later.**

## What has been TRIED and FAILED (don't repeat without a new idea)

Both were output-side motion constraints; both **hurt** because they fight the
symptom (jumps) not the cause (creep), and holding a wrong incumbent locks in error:

1. **Velocity smoother** (`CURSOR_MAX_SPEED` cap on emitted per-frame delta):
   0.714 → 0.703. The cap (18 px) was below p99 real speed, so it fought legitimate
   motion. Reverted.
2. **Confirm-gate** (reject far jumps, hold incumbent until a far region persists K
   frames): 0.672 → 0.627 on the t3/t5/t8/t2/t7 subset. Drift onsets didn't move,
   proving the failure isn't via jumps. Reverted.
3. **Proximity-fused emission in the path integrator** (`fpath` with prox_w>0): a
   CV-proximity term added to the Viterbi emission HURT every clip, including the weak
   ones it was meant to help (t1 0.53→0.24 at prox_w=0.3, t10 0.93→0.04 at 0.6). The
   CV prior reinforces lock-in (prediction pulls the path to stay on a wrong box).
   field's weak-clip robustness lives in its coast/reacquire *dynamics*, not in any
   term addable to an emission. fpath default is now prox_w=0 (pure mass).
4. **field⇄fpath REGIME ROUTER** (all three forms — see "Router dead end" below):
   the ~0.83 per-clip oracle is real but has **no causal key**. Don't rebuild it.

**Lesson: no output-side / decision-side fix and no emission-fusion works. field and
the path integrator are COMPLEMENTARY (see below) but the complementarity is NOT
online-exploitable — proven dead, see "Router dead end". The lever is upstream signal.**

## Router dead end (do not rebuild without a genuinely new causal signal)

Tested 2026-06-15. The field/fpath oracle (~0.83) needs a *causal* signal to pick the
right tracker live. All three candidate signals fail (gate scripts since removed; the
results below are the record):

1. **Per-clip regime threshold** — no causal per-clip statistic
   cleanly separates field-favored (t1,t3,t4,t5) from fpath-favored clips. Peak
   saliency mass is directional (2411 vs 2973, >1 std) but **t1 breaks it** (strong
   peak 2868, field still wins). clean_split=False for every candidate.
2. **Per-frame agreement** — agree 69% of frames @ wr 0.818;
   all loss is in the 31% disagree frames. But on disagree frames *who wins still
   follows the clip regime*, so a regime-label-free tiebreak = baseline exactly
   (field-fallback 0.714, fpath-fallback 0.697). Per-frame oracle 0.824.
3. **fpath path-margin** (best−runnerup α gap, causal) — ANTI-correlated with being
   right: high-margin disagree frames fpath-right 0.30 vs field 0.45. Margin measures
   conviction; conviction is the lock-in that loses on field-favored clips. Optimal
   threshold = "never take fpath" = ship field.

**Conclusion: ship `field`. The router cannot be built online.**

## Box-residual dead end (avenue #2 — gated and killed 2026-06-15)

The "box-level rigid residual is the strongest untapped signal lever" framing was
**not supported by data** (gate script since removed; result is the record). The residual
(`_box_rigid_residuals`, already wired into the `paper` modes; `paper_outlier_rank`
greedy-picks it and scores only 0.347) was tested for the only property that would
matter — **complementarity to flow on flow's blind frames**:

| k | flow top-k | resid top-k | union | resid \| flow-miss | lift |
|---|---|---|---|---|---|
| 1 | 0.555 | 0.129 | 0.625 | 0.156 | +0.069 |
| 3 | 0.886 | 0.347 | 0.923 | **0.321** | +0.036 |

- Residual is weak per-frame (top-1 0.13) — confirms DIAGNOSIS.md (0.14/0.38).
- **Not preferentially complementary**: on flow-MISS frames residual hits 0.321 — at or
  *below* its 0.347 base rate. It does NOT light up when flow is dark; misses ~independent.
- **Attacks the wrong gap**: flow's per-frame top-3 availability is already **0.886**,
  far above field's actual **0.714**. field isn't signal-starved — it fails to *hold*
  available signal (creep = integration-capture loss). Adding signal above your score
  can't move it; best case threading residual through field's integrator ≈ +0.03, and
  weak-channel fusion into field has only ever diluted (prox-fusion, router).

**Pattern across ALL failed avenues** (smoother-cap, confirm-gate, prox-fusion, router×3,
box-residual): field (0.714) sits at the causal-integration ceiling (Viterbi K=0 0.724)
over a signal that's 0.886-available at top-3. The binding limiter is **integration
capture** (hold signal without creeping), which is regime-coupled and resists every
decision-side and signal-fusion fix. The one proven-positive unbuilt lever is the
**fixed-lag smoother** (+0.03, physically free) — see Open avenues #0.

## fpath: causal path integrator (`track_fused_path_identity`, mode `fpath`)

Built from the Viterbi-ceiling insight. Forward Viterbi trellis over YOLO boxes,
fully causal (K=0, ships live): emission = saliency mass normalized by a **running EMA
of peak mass** (cross-frame confidence — NOT per-frame max, which erased the
weak/strong distinction), transition = `trans_w*(jump/radius)^2` continuity penalty.
With prox_w=0 it reproduces the Viterbi-ceiling K=0 numbers exactly (validates the
trellis). Mean **0.698** (vs field 0.714) — but the per-clip split is the headline:

| | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|--|--|--|--|--|--|--|--|--|--|--|
| field | **0.80** | 0.87 | **0.69** | **0.72** | **0.41** | 0.82 | 0.80 | 0.59 | 0.75 | 0.70 |
| fpath | 0.53 | **0.95** | 0.50 | 0.60 | 0.20 | **0.84** | **0.99** | **0.63** | **0.82** | **0.93** |

**field and fpath are strongly COMPLEMENTARY, split by signal regime:**
- **Strong-signal clips** (t2,t6,t7,t8,t9,t10): fpath dominates, +0.02 to **+0.24**
  (t7 0.99, t10 0.93, t2 0.95 — near detector ceiling). The path integrator extracts
  far more than field's greedy peak-follow when saliency is trustworthy.
- **Weak/misleading-signal clips** (t1,t3,t4,t5): field dominates by **0.12–0.27** —
  its coast/CV-model survives stretches where saliency mass lies; fpath chases the lie.

**An oracle picking the better tracker per clip scores ~0.83** — above either alone
AND above the 0.788 offline Viterbi ceiling, because they cover each other's failures.
**This oracle is NOT online-reachable** — see "Router dead end" above; all three
candidate routing signals fail. The ~0.83 is a diagnostic ceiling, not a target.

## Open avenues (highest-EV first), all online-compatible

0. **★ Fixed-lag smoother (K≈15)** — **the current frontier** (promoted after
   box-residual dead-ended, see below). The ONLY proven-positive unbuilt lever:
   Viterbi sweep measured K=0→K=15 = **+0.03** (0.724→0.753), and ~10–15 frames lag is
   physically free (shape can't leave its radius that fast). Legitimately online (hold
   K frames, emit t−K). Modest but real and dilution-free — measured, not hoped. We
   measured the ceiling but never shipped the smoother into `field`/`fpath`.
1. **Fixed-lag (K≈15) integration ceiling** — *diagnostic, DONE.* (the measurement
   behind avenue #0; +0.03 K=0→K=15).
~~Box-level rigid residual (signal)~~ — **DEAD, see "Box-residual dead end" below.**
   The residual is weak per-frame AND not preferentially complementary to flow; it
   attacks per-frame availability (already 0.886, well above field's 0.714) not the
   binding integration-capture gap.
~~Regime router (field⇄fpath)~~ — **DEAD, see "Router dead end" above.** No causal
   signal exploits the ~0.83 oracle. Do not rebuild without a genuinely new signal.
3. **Rotation as a SELECTOR, not a blender** (signal, helps laggards) — rotation
   rescues exactly the clips translation fails (t3, t8) but naive weighted fusion
   dilutes both signals. Use whichever signal is locally confident, don't average.

## Viterbi-ceiling result (`ld/detect/viterbi_ceiling.py` → `VITERBI_CEILING.md`)

Built the fixed-lag Viterbi over the **exact field signal** (per-box saliency mass).
The K-sweep result reframed the strategy:

- **Lookahead alone is weak.** Mean K=0 = 0.724, K=15 = 0.753 (+0.030), K=inf = 0.788
  (+0.064). A free ~0.5s fixed-lag smoother buys only +0.03 — NOT a strong mandate.
- **The per-clip divergence is the real finding.** field and a causal (K=0) Viterbi
  over the same signal are good at *different* clips:
  - **Strong-signal clips** (t2,t6,t7,t9,**t10**): causal Viterbi K=0 *dominates* field
    by up to **+0.24** (t10: field 0.698 → Viterbi-K0 0.937). field is far below its
    own signal's causal ceiling here → a **better causal integrator** would win big.
  - **Weak/misleading-signal clips** (t1,t3,t4,t5): Viterbi (even K=inf) is *worse*
    than field (t1: field 0.804 vs Viterbi-K∞ 0.663). Here saliency mass alone is
    misleading and field wins via its **CV motion model + snap_feedback**, which the
    mass-only emission omits.
- **Synthesis:** not cleanly signal- vs integration-limited — it's *both, separated by
  clip*. The mass-only ceiling is handicapped (it drops field's positional/velocity
  prior). **Highest-EV next move: a better CAUSAL integrator** (Viterbi-style
  cumulative-path tracker) whose emission **fuses saliency mass + CV-proximity** —
  field's two ingredients — capturing the strong-signal wins without regressing the
  weak ones. Lookahead is a minor add-on (+0.03), not the headline.

## Laggard clips

`t5` (0.41, lowest oracle 0.836) and `t8` (0.59) are the weak clips — the real shape
blends most with fakes there. Strong clips for regression-guarding: t2 (0.87), t7 (0.80).

## Conventions / constraints

- **Strictly causal** for anything shipped. Offline only for diagnostics/ceilings.
- All identity modes dispatch through `identity._dispatch_mode` (shared by the live
  `run_clip` and the `eval_modes` harness) so a mode behaves identically in both;
  `ALL_MODES` is the single source of truth for argparse + harness.
- The green crosshair in clips is **GT only** — `ld/vision/cursor.strip_pointer`
  inpaints it out before any motion/tracking so the solver never sees it.
- When validating a change: accept only if it improves the **held-out (LOO)** mean
  with no per-clip regression. In-sample gains alone are not sufficient.
- Eval is cache-backed; pin the weights path (cache key = md5 of path+mtime).

## Key files

| File | Role |
|------|------|
| `ld/detect/identity.py` | All identity modes; `track_field_lag_identity` (the leader/default), `track_field_identity` (its underlying signal), `_dispatch_mode`, `ALL_MODES`, `run_clip`. |
| `ld/vision/motion.py` | `estimate_motion` (rigid RANSAC + outliers), `saliency_map`. The per-frame signal source. |
| `ld/track/tracker.py` | `OutlierTracker` — gated CV peak-follower with reacquire. |
| `ld/vision/cursor.py` | GT green-crosshair detect + `strip_pointer` inpainting. |
| `ld/detect/eval_modes.py` | Leaderboard harness → `LEADERBOARD.md`. |
| `ld/detect/loo.py` | Leave-one-clip-out honest generalization. |
| `ld/detect/diagnose.py` | Loss decomposition → `DIAGNOSIS.md`. |
| `ld/config.py` | Tracker / detection / motion tunables. |
