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

## Current best: `field_coh` mode (the default)

`track_field_coh_identity` in `ld/detect/identity.py` — `field_lag` (below) plus a
**coherence far-jump override**. This is the first signal to break the long-standing
"every motion-derived avenue is exhausted / t5–t8 is an identifiability limit"
conclusion. The new evidence is the **directional + temporal COHERENCE of the outlier
residual VECTORS** — `saliency_map` keeps only residual *magnitudes* and discards
direction, so every prior signal test was magnitude-blind and never used this:

  coherent_mass(box) = ‖Σ resid_vecs_in_box‖ · coherence,
      coherence = ‖Σ resid_vecs‖ / Σ‖resid_vecs‖  ∈ [0,1]
  accumulated over a causal window (`FIELD_COH_OVR_WIN=16`). coherence ≈ 1 when a box's
  independent motion all points one way (the real shape drifting rigidly), ≈ 0 for
  incoherent noise/fakes. challenger = argmax; margin = (top−runnerup)/top [causal key].

The override rewrites `field_lag`'s emitted (x,y) to the challenger centroid when, for
≥`FIELD_COH_OVR_C=8` consecutive frames, the challenger persistently disagrees with the
committed pick, is >`FIELD_COH_OVR_FAR=1.5` radii away, and is confident
(margin≥`FIELD_COH_OVR_TAU=0.30`). It attacks the **escape-from-lock-in** failure: when
`field_lag` has drifted onto a fake, the real shape is *far away* with a coherent signal.
Strictly causal (backward-only window + streak). `motion.py` now exposes
`MotionField.outlier_vectors` and the tracker collects them in a single flow pass.

**Scores:** in-sample **0.742** (vs `field_lag` 0.721, +0.021), gate LOO **0.7442**
(single conservative config, stable across all 10 folds). **No clip regresses** — every
clip improves or ties: t5 **+0.053** (the worst laggard; median err 97→75px), t2 +0.035,
t3 +0.038, t6 +0.028, t7 +0.022, t10 +0.017, t8 +0.009, t1 +0.008, t4 +0.002, t9 +0.000.

**Why it's NOT a dead end like every prior signal:** it is *not* regime-coupled (helps
strong AND weak clips), has a *working causal key* (the coherence margin gate fires
selectively, do-no-harm), and lands its biggest gain on t5. The "identifiability limit"
was really a *magnitude-channel* limit; direction separates the real shape there.

**Headroom (probe, `coh_gate._probe_headroom`):** on the still-wrong+silent frames the
*ungated* coherent challenger already sits on GT **50%** of the time (t5 0.556) — but
**90%+ of those frames are OUTSIDE the snap radius** (t5: 190 out / 4 in). So the signal
is an escape-from-lock-in cue requiring a FAR jump, which is exactly why a local
snap-weight blend FAILED (0 change; see below) and the far-jump override works. Capturing
the rest needs a mechanism that can teleport — see "Open avenues".

## The smoother layer: `field_lag` mode

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
an emission fusion (all three of those FAILED; see below). `field_coh` wraps it and it
remains selectable. Its gains and the override's gains are **additive** (disjoint
failure modes: field_lag kills creep at onset, coh escapes an already-locked fake).

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
box-residual, rotation-selector): field (0.714) sits at the causal-integration ceiling
(Viterbi K=0 0.724) over a signal that's 0.886-available at top-3. The binding limiter is
**integration capture** (hold signal without creeping), which is regime-coupled and
resists every decision-side and signal-fusion fix. The one proven-positive lever was the
**fixed-lag smoother** (+0.03, physically free) — now SHIPPED as `field_lag`.

## Rotation-selector dead end (avenue #3 — gated and killed 2026-06-15)

The "rotation rescues exactly the clips translation fails (t3, t8)" framing in the old
avenue #3 was **not supported by data** (gate `ld/detect/rot_gate.py`, run then removed;
result is the record). Independent-rotation magnitude (`_box_net_rotations`, log-polar
phase correlation — already wired into the `paper`/`accum` modes) was tested against the
same complementarity + causal-key bar that killed box-residual and the router:

| metric | base (all rescuable-detected) | on field's rescuable-miss frames | lift |
|---|---|---|---|
| rot top-1 (GT box is highest-rotation) | 0.09 | 0.09 | **+0.005** |
| rot top-3 | 0.24 | 0.25 | **+0.013** |

- **Not preferentially complementary.** On the frames field gets wrong (but where the GT
  box IS detected, so a picker *could* rescue), rotation points at the GT box at its own
  base rate — lift ≈ 0. It does not light up when field is dark (same failure as residual).
- **The t3/t8 claim is false.** On exactly the clips rotation was supposed to rescue,
  on-miss top1 = base (t3 0.08=0.08, t8 0.09 vs 0.08). Rotation is weakest on the laggards.
- **No causal key.** Splitting miss frames by top-rotation magnitude (the live-observable
  conviction proxy) gives 0.09 high vs 0.09 low — conviction does not mark correctness, so
  there's no online signal to gate a selector on (the router lesson, repeated).

**Conclusion: a rotation selector would dilute exactly like prox-fusion and the router.
Do not build it.** The avenue is removed from the open list. With #0 (fixed-lag) shipped
and #2 (box-residual) + #3 (rotation) both dead, every enumerated downstream/signal-fusion
avenue is exhausted — the remaining lever is a genuinely **stronger or differently-derived
per-frame signal** (new upstream evidence), not any recombination of the existing channels.

## BraveDown reference solver (`data/successful_examples/`, local-only) — what it reveals

A third-party solver clip (988 frames, 1080p) overlays its own internals: persistent
`ID:N` boxes on every shape, a red `REAL_ID:N` box, a green target dot with a confidence
score (~0.96), and a magenta sheet-boundary polygon. The footage also shows its **client**
source (`def main()`: argparse `--server http://localhost:5000`, `--video-path`,
`client = TrackingClient(server_url=...)` + a `display_frame` drawer). Key reads:

- **Client/server split; the tracker is server-side and NEVER shown.** The video does not
  hand us their algorithm — only their behavior.
- **They acquire at the countdown**, before gameplay motion (`REAL_ID:7` already locked on
  the first countdown frame) — same as our `compute_countdown_lock`.
- **Their identity fragments too** (`REAL_ID` migrates 7→27 mid-clip, new box IDs spawn)
  yet the green dot stays glued to the real shape — so they run a **position track that
  outlives box re-ID**, structurally the same as our `field`. No better association scheme.
- **The honest conclusion: the reference does NOT reveal a missing trick.** Its
  architecture mirrors ours; its stable ~0.96 confidence suggests stronger *per-frame
  discrimination* (the signal channel), consistent with every gate's finding.

## Appearance-channel gate (candidate A — NCC template tracker; gated 2026-06-15)

Motivated by the above (an appearance/correlation channel is the one thing we lack — all
our signal is the single LK+RANSAC motion-outlier map). Tested a standalone normalized-
cross-correlation template tracker (`cv2.matchTemplate` CCOEFF_NORMED) seeded at the
countdown lock — pure appearance self-consistency, orthogonal to flow; the peak corr is a
built-in causal key. Gate `ld/detect/corr_gate.py` (run then removed; result is the record):

| policy | NCC within_r | complementarity lift (on-miss − base) | causal key (hi−lo) |
|---|---|---|---|
| static template | 0.06 | +0.004 | +0.038 |
| confidence-adapted | 0.07 | −0.012 | −0.012 |

- **DEAD at the bar**, but informatively. Standalone NCC is near-useless (~0.06) — the
  real shape **rotates independently**, so a translation-only template de-correlates within
  a few frames and locks onto near-identical neighbours on the tan relief. Adapting the
  template made it worse (adapts onto the drift).
- **One real positive — t8.** On t8 (a laggard, field 0.59) static NCC on field's miss
  frames hits 0.23 vs 0.13 base AND its confidence key separates hard (hi 0.42 / lo 0.05).
  So appearance self-consistency IS the right *kind* of orthogonal signal — it has a
  working causal key where the shape is appearance-distinct — but **plain translation-NCC
  is the wrong implementation; rotation kills it.**

**Next lever (unbuilt): a ROTATION-INVARIANT appearance descriptor** — match the patch in
log-polar space (rotation→shift, the machinery already exists in `_logpolar_roi` /
`_box_net_rotations`), or a rotation-normalized template, so self-consistency survives the
real shape's independent spin. This is the one candidate the data still supports; gate it
the same way before building.

### Rotation-invariant appearance gate (candidate A' — log-polar phase corr; gated 2026-06-15)

Built exactly that: a standalone tracker that, among YOLO boxes near the last position
(continuity), picks the one whose **log-polar** view best phase-correlates with the seed
patch (rotation→shift LP absorbs; peak response = causal key). Reused `_logpolar_roi` /
`ROT_ROI_N`. Gate `ld/detect/lpcorr_gate.py` (run then removed; result is the record):

| policy | LP within_r | complementarity lift (on-miss − base) | causal key (hi−lo) |
|---|---|---|---|
| static | **0.24** | −0.030 | +0.086 |
| confidence-adapted | 0.21 | −0.021 | +0.093 |

- **The rotation diagnosis was right** — rotation-invariance lifted the raw tracker **4×**
  (0.24 vs NCC's 0.06). Log-polar phase corr genuinely follows the real shape through its
  spin. The *mechanism* works.
- **But DEAD at the complementarity bar — structurally.** Aggregate lift is **negative**
  (on-miss ≤ base). The per-clip pattern is the tell: LP helps on **t7 (+0.31), t9 (+0.18),
  t3** but is strongly anti-complementary on **t4 (−0.27), t6 (−0.18), t8 (−0.15)** — incl.
  t8, the laggard it was meant to rescue. This is **the fpath regime split reincarnated**:
  the appearance channel rescues the strong-signal clips (where fpath already wins) and
  fails the weak/misleading clips (where field's CV-coast wins), landing on the SAME regime
  axis — with no new causal key (global hi−lo only +0.086) to route on.
- **Collapses into the router dead end.** Real per-clip complementarity, no causal key to
  exploit it live. Same wall as field⇄fpath.

**Conclusion — the appearance channel is exhausted.** Both translation-NCC (rotation kills
it) and rotation-invariant log-polar (works mechanically, but regime-coupled to fpath with
no causal key) fail. Appearance does NOT provide a regime-orthogonal lever; it re-derives
fpath's strong-signal competence by a different route. **The binding limiter remains
integration-capture on the weak/misleading clips (t4, t5, t8), and NO channel we have
tested — flow, residual, rotation, appearance — is preferentially strong THERE with a
causal key.** That specific gap (a signal that fires on field's misses on the weak clips,
with a live confidence) is the only thing worth hunting next; absent a genuinely new such
signal, `field_lag` (LOO 0.721) stands as the shipped ceiling.

## Box-cleanup gate (low-hanging fruit — gated 2026-06-15)

YOLO emits ~18 boxes/frame (vs ~20 real shapes); hypothesis was that edge-partial /
duplicate clutter near the real shape makes the saliency-mass snap pick a wrong neighbour
(a creep source). Tested two FREE, reversible, no-retrain filters on the cached detections
(gate `ld/detect/boxclean_gate.py`, run then removed): a **border filter** (drop boxes whose
centroid is within margin of the frame edge — provably safe, the real shape's centroid never
reaches the edge) and a **conf filter** (raise the 0.25 cache threshold).

- **Border m=30 is a small free win: mean 0.721 → 0.725 (+0.003), oracle FLAT (0.929 —
  never drops the GT box), NO clip regresses** (t9 +0.020, t5 +0.013, rest ±0.000). The
  margin is a cliff: m=20 does nothing, m=45 breaks (t3 −0.049, its oracle 0.889→0.841 —
  clips real shapes riding near the edge). Safe band ~30px. NOT YET SHIPPED (marginal;
  ship only if bundling with another change).
- **Conf filter HURTS** — c=0.40 drops the oracle (0.929→0.850; t5 0.836→0.731): it kills
  real-shape boxes on their low-confidence frames. The oracle is the ceiling, so don't.
- **The snap-ambiguity hypothesis was the valuable miss.** `ambig` (fraction of oracle-hit
  frames with >1 box within snap radius of GT) is HIGH on the laggards — **t5 0.44, t8
  0.65** — and border filtering barely moves it (0.47→0.46). So the snap IS ambiguous on the
  weak clips, but the competing boxes are **genuine neighbouring shapes, not removable
  clutter** — no border/conf filter can touch them. Sharper diagnosis: t5/t8 creep is the
  real shape's TRUE neighbours winning the saliency-mass snap, which loops back to the
  weak-signal / integration-capture wall, not a detection-cleanliness problem.

## Collision-vs-camouflage oracle-gap decomposition (relabel decision — 2026-06-15)

Decided whether the parked data-relabel (drop edge partials, LABEL collision partials) is
worth doing, by decomposing the oracle-MISS frames (gate `ld/detect/collision_diag.py`, run
then removed). Classified each oracle-miss: `covered` (a box contains GT → merged into a
neighbour), `adjacent` (nearest centroid in (r,2r]), `empty` (>2r → true non-detection).

| clip | within_r | oracle | **differentiator gap** | empty% | max oracle gain from relabel |
|---|---|---|---|---|---|
| t5 | 0.409 | 0.836 | **0.427** | 0.22 | ~0.13 (optimistic) |
| t8 | 0.585 | 0.938 | **0.353** | 0.00 | ~0.06 (optimistic) |

**Verdict: relabel STAYS PARKED.** The decision doesn't hinge on the collision/camouflage
split (the `covered` metric is a soft upper bound — with ~18 boxes blanketing the sheet a
GT point lands inside *some* box by area alone, so `covered` can't cleanly separate "merged
into neighbour" from "GT in a fake's box corner"; only `empty` is unambiguous). It hinges on
the gap comparison: on both laggards the **differentiator already leaves 0.35–0.43 on the
table** (boxes that ARE on the real shape, lost to creep), while relabel's entire ceiling
effect is at most 0.06–0.13 — and the recoverable frames are collisions, exactly when creep
is worst, so realized gain ≪ oracle gain (likely <+0.01). Relabel attacks a gap 3–7× smaller
than the one we already fail to close. This confirms the box-cleanup finding from the other
side: **the binding constraint on t5/t8 is the differentiator failing to HOLD boxes it
already has, not missing boxes.** Un-park relabel only if/when the differentiator's creep on
detected boxes is solved and the oracle ceiling becomes the actual limiter.

## Top-K confidence cap gate (2026-06-15)

Tested capping each frame to the K highest-confidence boxes (adaptive version of the failed
fixed-conf filter; idea: only ~20 real shapes exist, so keep ~20 boxes). Gate
`ld/detect/topk_gate.py`, run then removed. The decisive measurement is the **real shape's
confidence RANK** on oracle-hit frames: **median 9–13, max 20–28 per clip** (t8 max 28, t5
max 24). The real shape is a MIDDLING detection, not a top one — being camouflaged (what
makes it the target) also makes YOLO less confident of it.

- **K=20 (the requested cap) REGRESSES**: mean −0.001, drops the oracle on its faint frames
  (t8 0.938→0.919, −0.010; t3 −0.009) — the same failure as the fixed conf filter, rank-based.
- **K=25 is oracle-safe but worthless**: mean +0.001 (within noise, < border filter's +0.003),
  oracle flat 0.929, no regression. Most frames already have <25 boxes so it rarely fires.
- **Lesson / third confirmation:** you CANNOT filter to the real shape by confidence — it is
  precisely the box that LACKS confidence. The "~20 shapes ⇒ keep top 20" premise is false:
  the 20 real shapes are not the 20 most-confident boxes, and the real one in particular
  ranks 15–28 on hard frames. Not shipped.

## Box-envelopment filter (gated and killed 2026-06-15)

Physical observation (correct): identical same-size shapes cannot nest, so a YOLO box that
*fully envelopes* another is an artifact (a loose/merged box). Occurs on **37% of t5 frames**
(323 pairs); the enveloping OUTER box is the oversized/merged one in 210/323. Tested four
drop policies through the REAL `field`/`field_lag` tracker (gate `ld/detect/envelop_gate.py`,
run then removed — NOT the area-biased GT-containment proxy, which misleads: a bigger box
contains the GT *point* more often by area alone).

| policy | mean Δwithin_r (field_lag, t1–t10) | worst clip | oracle |
|---|---|---|---|
| drop_outer | **−0.015** | t5 −0.022 | 0.929→0.923 (drops) |
| drop_inner | −0.007 | **t1 −0.048**, t6 −0.047 | flat |
| drop_lowconf | −0.006 | t4 −0.039 | 0.921 |
| drop_highconf | −0.016 | −0.032 | drops |

- **The intuitive fix (drop_outer) is the WORST** — it lowers the oracle. That loose outer box
  is often the *only* coverage of the real shape mid-drift between two tight detections, so
  removing it strips real-shape boxes.
- **drop_inner is do-no-harm on the laggards** (t5 +0.006, t8 ±0) but **regresses strong clips**
  (t1 0.816→0.781, t6 0.814→0.767) for −0.007 net. Same regime-coupling as every prior gate:
  laggard-local effect that dilutes elsewhere.
- **Same wall as box-cleanup:** the envelopment boxes near the real shape are **genuine competing
  neighbour shapes, not removable clutter.** Geometry alone can't tell the real-shape's enveloping
  box from a fake's. Not shipped.

## Multi-baseline optical-flow dead end (avenue #0-signal — gated and killed 2026-06-15)

The one upstream lever the prior conclusion pointed at: the real shape moves median **1.3 px/frame**
vs the **1.5 px** `OUTLIER_RESID_MIN` threshold, so on a typical frame its own displacement is
BELOW threshold and the 1-frame flow signal vanishes. Hypothesis: a longer baseline (warp t−K→t via
the composed rigid fit; sheet stays rigid, real shape drifts ~K×1.3 px) surfaces it. Probe
(`ld/detect/baseline_probe.py`, run then removed) measured GT-box rank by outlier-saliency mass per
baseline K∈{1,2,4,8}:

| K | t5 top1 | t5 top3 | t8 top1 | t8 top3 |
|---|---|---|---|---|
| **1** | **0.355** | **0.686** | **0.459** | **0.687** |
| 8 | 0.210 | 0.503 | 0.336 | 0.643 |

- **Longer baselines make GT ranking MONOTONICALLY WORSE**, not better. The slow-drift gain is
  swamped by accumulating LK-tracking noise + sheet non-rigidity leaking into the residual floor.
  GT-detection stays flat (~0.83/0.90) but the *ranking* erodes.
- **The 1-frame baseline is already optimal for the flow-outlier channel.** No baseline-stacking of
  flow helps. Closes the "longer/accumulated independent motion" idea from the signal side.

## Temporal-feature learning dead end (experiment 2a — gated and killed 2026-06-15)

Premise of a temporal-stack detector: the real shape is discriminable from its motion HISTORY, not a
single frame. Tested cheaply BEFORE any retrain with a numpy logistic regression over per-box CAUSAL
temporal features (saliency-mass history mean/max/slope over W=8, rigid-residual history, conf-rank,
chain length), trained **leave-one-clip-out**, vs the instantaneous mass baseline (= field's signal).
Gate `ld/detect/temporal_gate.py`, run then removed; **7199 GT-labeled frames** available (the green
cursor labels the real shape every frame — ample training data, NOT the blocker).

| metric | mass-only (field signal) | learned temporal | Δ |
|---|---|---|---|
| mean top-1 | 0.579 | 0.519 | **−0.060** |
| mean top-3 | 0.857 | 0.862 | +0.005 (noise) |
| **t5 top-3** | 0.823 | 0.798 | **−0.026** |
| **t8 top-3** | 0.761 | 0.718 | **−0.043** |

- **No held-out lift; flat-to-negative on the laggards.** The model HAS `m0` (instantaneous mass) as
  an input yet scores *worse* top-1 than mass alone — meaning the temporal features are net noise that
  overfits the training clips and misleads on held-out clips. The history carries no clip-transferable
  discriminative signal.
- **Falsifies the temporal-stack retrain (2b) premise for ~80 lines, no GPU.** A 3-frame-stack YOLO
  would learn the same thing the logreg did: help strong clips, not the laggards, net ~0 held-out.
  Do not retrain on temporal context expecting a laggard win.

## Acquisition/lock probe (orthogonal-bug scan — 2026-06-15)

Scanned lock + early-track + recovery quality (`ld/detect/lock_probe.py`, run then removed) to rule out
a cheap acquisition fix. **Lock is healthy on 8/10 clips** (GT inside lock box, 0.03–0.58r) INCLUDING
both laggards (t5 0.33r, t8 0.58r) — so the laggard loss is confirmed **mid-clip creep, not a bad
handoff** (t5 early[20]=0.40 holds only 8f; t8 early[20]=1.0 holds 72f then recovery only 0.493). Two
findings: (a) **t1 and t4 have a BROKEN lock** — lands on the wrong shape (2.04r / 2.57r, gt_in_box=
False) yet reacquire rescues them to 0.70/0.69 overall. This is a real, isolated, orthogonal bug worth
a cheap fix (leaves points on t1/t4), NOT the laggard lever. (b) the **recovery** column (within_r after
first loss) is lowest on t8 (0.493) and t6 (0.643) — reacquire-to-global-peak often doesn't find its
way back, but that loops to the signal wall (peak saliency ≠ real shape on those frames).

### Synthesis across the 2026-06-15 step-out (4 cheap experiments)

Envelopment, multi-baseline flow, temporal-feature learning, and acquisition all converge on one
conclusion, stronger than "diminishing returns": **on the weak clips (t5, t8) the real shape is NOT
discriminable from any motion-derived signal we can construct** — instantaneous, multi-baseline,
temporal-history, or learned. The GT box sits at top-3 ≈ 0.69–0.76 mass and no transform separates it
from its real neighbours. This is an **identifiability limit, not an engineering gap**: `field_lag`'s
0.72 is at the achievable ceiling of the motion channel. The only thing that could move t5/t8 is
**non-motion-derived evidence** — and appearance (the obvious candidate) was already shown
regime-coupled with no causal key. Remaining concrete TODO with positive EV: the **t1/t4 lock bug**
(orthogonal points), not another signal/decision recombination.

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

**SUPERSEDED 2026-06-15 (coherence win).** The prior conclusion below — "every signal
avenue exhausted, t5–t8 is an identifiability limit" — was FALSE. It rested on the
hidden assumption that the motion channel = saliency *magnitude*. The **directional
coherence of the residual vectors** (a channel `saliency_map` discards) broke it:
`field_coh` is now the default at in-sample 0.742 / gate-LOO 0.7442 (+0.021 over
field_lag), helps every clip, biggest gain on t5. The lesson: "exhausted" meant
exhausted *for the magnitude derivation*; a different derivation of the same flow had
untapped signal. **The current frontier is the coherence headroom + the t1/t4 lock bug.**

0. **Coherence far-jump override** — **SHIPPED** as `field_coh` (the default). Captures the
   escape-from-lock-in slice via a binary far-jump gate. **Still open within it:** the probe
   shows the ungated coherent challenger is on-GT 50% of remaining-wrong frames, almost all
   OUTSIDE the snap radius. The binary override only skims this. **Next highest-EV build: a
   coherence SALIENCY CHANNEL** — fold coherent-mass into the saliency map the tracker +
   reacquire consume, so *reacquire can teleport to the coherent peak* (it jumps far, unlike
   the snap). Target ~0.79–0.82 (the 50% headroom). More invasive (touches coast/reacquire
   dynamics); guard LOO with no regression. The local **snap-weight blend FAILED** (see
   "Snap-weight blend dead end") — must be a far-reaching mechanism.
1. **Fixed-lag smoother (K=8)** — **SHIPPED** as `field_lag`; now wrapped by `field_coh`.
2. **t1/t4 lock bug** — *orthogonal, positive EV, unbuilt.* The countdown lock lands on the
   wrong shape on t1 (2.04r) and t4 (2.57r) — `compute_countdown_lock`/`_pick_lock_box`.
   Reacquire currently rescues both to ~0.70, but a correct lock leaves points on the table.
   Does NOT touch the laggard (t5/t8) wall. See "Acquisition/lock probe".
~~Box-level rigid residual (signal)~~ — **DEAD** ("Box-residual dead end").
~~Multi-baseline / accumulated independent motion (signal)~~ — **DEAD** ("Multi-baseline
   optical-flow dead end"): longer flow baselines rank GT monotonically WORSE; 1-frame is optimal.
~~Temporal-stack / temporal-history detector (signal)~~ — **DEAD** ("Temporal-feature learning
   dead end"): learned LOO model over motion history gives no held-out lift, negative on t5/t8.
~~Regime router (field⇄fpath)~~ — **DEAD** ("Router dead end"). No causal key for the ~0.83 oracle.
~~Rotation as a SELECTOR~~ — **DEAD** ("Rotation-selector dead end").
~~Appearance channel (NCC / log-polar)~~ — **DEAD** ("Appearance-channel gate"): regime-coupled,
   no causal key.

**Revised binding limiter:** NOT an identifiability limit. On t5/t8 the real shape IS
separable from its neighbours by residual-vector *coherence* (a direction-aware motion
signal), which all prior tests — magnitude-only — missed. The remaining gap is a
*mechanism* gap (the override can't reach far-jump frames the snap can't see), not a
signal gap. Build the coherence saliency channel before declaring the channel done.

## Snap-weight blend dead end (option 1 — built and killed 2026-06-15)

The obvious first integration of coherence — reweight the field **snap** to pick
`argmax mass·(1+λ·coherence)` among boxes in the snap radius — gives **EXACTLY ZERO
change** (t5/t2 flip count = 1 frame). Wired as `FIELD_COH_LAMBDA`/`_box_coherent_mass`
in identity.py (left inert, λ=0 default); sweep via `coh_sweep.py`. Root cause **measured,
not theorized**: the snap only considers boxes within ~74px (1.3·radius); >1 box is in
range on just 55/644 t5 frames, so there's rarely a competitor for coherence to prefer.
And the recoverable signal is NOT local — on t5's 321 wrong frames the most-coherent
on-GT box is OUTSIDE the snap radius **190×** vs INSIDE only **4×** (t8: 118/8). The
coherence signal is an escape-from-lock-in cue (the real shape is FAR from the drifted
tracker), which is exactly why the **far-jump override** (`field_coh`) works and a local
snap blend cannot. **Lesson: any mechanism capturing this signal must be able to jump far.**

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
| `ld/detect/identity.py` | All identity modes; `track_field_coh_identity` (the leader/default — field_lag + coherence far-jump override), `track_field_lag_identity` (the smoother layer), `track_field_identity` (the underlying signal), `_box_coherent_mass`, `_dispatch_mode`, `ALL_MODES`, `run_clip`. |
| `ld/vision/motion.py` | `estimate_motion` (rigid RANSAC + outliers), `saliency_map`. The per-frame signal source. `MotionField.outlier_vectors` exposes the residual VECTORS (direction), which `saliency_map` discards — the coherence channel's input. |
| `ld/track/tracker.py` | `OutlierTracker` — gated CV peak-follower with reacquire. |
| `ld/vision/cursor.py` | GT green-crosshair detect + `strip_pointer` inpainting. |
| `ld/detect/eval_modes.py` | Leaderboard harness → `LEADERBOARD.md`. |
| `ld/detect/loo.py` | Leave-one-clip-out honest generalization. |
| `ld/detect/diagnose.py` | Loss decomposition → `DIAGNOSIS.md`. |
| `ld/config.py` | Tracker / detection / motion tunables. |
