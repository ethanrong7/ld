# LD identity — experiment log

Canonical, running log of every identity experiment: hypothesis, what was run, result, and
status. Ground-truth numbers also live in `CLAUDE.md` + `ld/detect/LEADERBOARD.md`; this file is
the narrative index so an agent can see what has been tried (and must not be retried) at a glance.

**Status legend:** `SHIPPED` (in the leader lineage) · `PARKED` (weak, revisitable as an increment)
· `DEAD` (do not retry without a new angle) · `OPEN` (proposed / in progress).

---

## Current state (2026-06-20)

- **Leader: `fpath_freeze` — 0.932 within_r** (oracle 0.958, full-board; LOO 0.9359, no per-clip
  regression). Prior leader `fpath_hedge` 0.899. **+0.033 board, the largest single lift in the
  lineage, and it raised the LAGGARDS** (t8 0.776→0.909, t5 0.809→0.858, t1 0.853→0.895).
- **What it is:** `fpath_hedge` + a residual-gated decode-layer freeze that runs before the churn
  hedge. A 1-box BINARY ("is the box I'm holding a rigid fake?") via the chosen box's cumulative
  N=30 sheet-frame residual: fake ~9–15px (rigid → just detector jitter), real ~45–91px. Residual
  < τ=15 for 1 frame ⇒ freeze the output toward the output-from-lag=6-frames-ago, hold until it
  recovers. The real shape barely moves, so an onset-anchored freeze stays in radius for 20+ frames.
- **Metric:** `within_r` = fraction of scored frames the estimate lands within the shape radius of
  GT. Accept a change **only on LOO mean ↑ with no per-clip regression**. In-sample is optimistic.
- **Laggards now:** t5 0.858, t4 0.880, t1 0.895 (t8 fixed 0.776→0.909). The floor anatomy below is
  for the *pre-freeze* (`fpath_hedge`) state — it diagnoses the long fake-rides the freeze then cured.
- **The two stages:** detection (YOLOv8n, oracle 0.958 — effectively solved) → identity (which box
  is real over time — the bottleneck). `fpath_freeze` does NOT solve identity; it sidesteps it at the
  output layer (detect "on a fake" → freeze; never re-identifies the real box). Code `identity.py`.

### Floor anatomy (per-frame trace analysis, `fpath_hedge`, 2026-06-20)

The floor is **not** scattered jitter — it is a **handful of long lock episodes** where the trellis
rides a fake while the real box is sitting detected nearby:

| clip | within_r | miss % | miss runs (top lengths) | identity-bound¹ | median miss err |
|------|---------:|-------:|-------------------------|----------------:|----------------:|
| t8 | 0.776 | 22.4% | 68, 22, 11, 10, 8 | **81%** | 181 px |
| t5 | 0.809 | 19.1% | 53, 25, 10, 8 | **85%** | 127 px |
| t1 | 0.853 | 14.7% | 32, 20, 16 | 72% | 86 px |
| t4 | 0.856 | 14.4% | 53, 11, 10 | 72% | 296 px |
| t3 | 0.876 | 12.4% | 23, 21, 12, 10 | 52% (**48% detection-bound**) | 76 px |

¹ *identity-bound* = `within_r==0 & oracle_hit==1` — the real box was detected but mis-identified.

**Implications.** (1) The laggard floor is overwhelmingly an **identity** problem on a few sustained
lock runs, not detection. (2) The miss errors are **large** (t8 181px, t4 296px) — the hedge freezes
the output, but during a coherent lock it freezes at the *wrong (fake) location*, so freezing only
*preserves* the error. Raising the floor requires an **identity-layer** fix that re-picks the real
box during a lock — the decode-layer hedge structurally cannot. (3) **t3 is the exception**: ~half
its misses are detection-bound (`oracle_hit==0`) — a targeted-detection problem, not identity.

---

## The lineage that shipped (each accepted on LOO, no per-clip regression)

| ID | Mode | within_r | What it added |
|----|------|---------:|---------------|
| — | `field` | 0.744 | Motion-saliency peak tracker + YOLO snap (snap is load-bearing: raw 0.28 → 0.59) |
| — | `fpath` | 0.797 | Causal Viterbi trellis over YOLO boxes; emission = saliency mass; transition = (jump/r)² |
| — | `fpath_coh` | 0.825 | Coherence-bumped emission `mass*(1+1.8*coh)` |
| EXP-FUSE | `fpath_fuse` | 0.876 | **Additive** emission `mass + 1.5*coherent_mass + 0.5*curl` (each cross-frame-normalised) |
| EXP-3 | `fpath_hyst` | 0.878 | + EMA-coherent-mass **hysteresis override** (distance-agnostic switch, escapes adjacent creep) |
| EXP-HEDGE | `fpath_hedge` | 0.899 | + **decode-layer churn-gated freeze-blend** (output only; identity state untouched) — catches *swept* locks |
| EXP-Q3 | `fpath_freeze` | **0.932** | + **residual-gated decode freeze** (chosen-box N=30 residual < τ ⇒ on a fake ⇒ freeze to lagged anchor) — catches *coherent* locks |

**Reusable insights from the lineage:**
1. Accumulate evidence on moving boxes with a **nearest-centroid-carried EMA**, not a fixed window
   (EXP-3: EMA on-GT 0.51 vs fixed-window 0.39 on t1).
2. Identity wins have **only ever** been **additive, cross-frame-normalised weighted sums** — never
   max-fusion (max amplifies whichever channel spikes on a fake; fuse_probe top-1 0.22).
3. The hedge's separator is trajectory **coherence** `(1−R)`, **not** speed — magnitude of motion is
   *anti-correlated* with real-ness on lock frames (the slow real shape reads low, the box-hop high).
4. **A signal too weak to RANK can be sharp as a per-box BINARY.** The integrated residual fails as a
   15-box override (EXP-Q2b, dead) but is decisive as "is *this* box a fake?" (EXP-Q3, shipped).
5. **Cap a threshold sweep at the physically-motivated value** (here the fake-noise floor). A wider τ
   grid let the LOO pick an unphysical threshold that overfit the low-residual strong clips; capping
   at the floor made all 10 folds independently select the same no-regression τ.
6. The two decode-layer freezes are complementary: churn (`fpath_hedge`) catches *incoherent swept*
   locks, residual (`fpath_freeze`) catches *coherent* locks. Together they cover both miss modes.

---

## Latest finding — EXP-Q1: cumulative sheet-frame residual (long-horizon drift integrator)

**Status: gate run 2026-06-20 — REFINED-POSITIVE (not the clean pass or clean fail the plan
anticipated).** Probe: `ld/detect/sheet_residual_probe.py` (read-only; affines from the cached
`_hedge_aff_*.pkl`, boxes from `detect_fusion_clip` cache — no video decode).

**Hypothesis (plan.md).** Every identity attempt measured the real shape's independent translation
*per-frame* or over 8–10-frame windows — exactly where ~1.3 px/frame sits *under* the affine/detector
noise floor. So the "information limit" is really an **SNR** limit; the standard fix is to
**integrate**: transport each box's centroid into the sheet's reference frame via the cumulative
inverse RANSAC affine and accumulate divergence from rigid prediction over N≈15–90 frames. A fake's
residual is ~0 at all N; the real shape's directional drift grows ~N·v while zero-mean noise grows
~√N·σ. Association is by **affine prediction** (snap to the predicted spot, drop chains > radius) —
which dissolves the EMA mis-association failure because fakes are exactly predictable.

**What the gate measured.** Per clip / per N, on the `fpath_hyst` MISS frames (and a t7-style sanity
on clean clips): where the oracle box ranks by cumulative residual (top1/top3), the separation ratio
`resid(oracle)/p90(resid over fakes)`, and the fake noise floor — and whether these **rise with N**.

**Results.**
- **t7 sanity is healthy** — the readout works: sep 2.42→3.73, top1 0.76→0.88 as N rises. (Unlike
  EXP-R1/EXP-S1, this measurement is not broken on the clean clip.)
- **The residual carries real, NEW signal at moderate N.** Pooled t8/t5/t1 **MISS** frames:

  | N | 15 | 30 | 45 | 60 | 90 |
  |---|---|---|---|---|---|
  | **oracle top1** | 0.51 | **0.65** | 0.62 | 0.52 | 0.49 |
  | sep ratio | 1.85 | 1.76 | 1.70 | 1.67 | 1.61 |

  At N≈30 the oracle box ranks **#1 at 0.65 pooled** (t8 **0.68**, t5 0.67, t1 0.79@N45) — far above
  the mass MISS-top1 bar of **0.32–0.41** that CLAUDE.md's triple-confirmed wall said *nothing* could
  beat. **This is the first channel to materially out-rank mass on t8's MISS frames.**
- **But the literal gate ("SNR must RISE to N=90") FAILS** — it peaks at N≈30 and decays. Two causes,
  both flagged in the plan as risks, both partially real:
  1. **Fakes accumulate ~N, not ~√N** — `fakeP90` grows roughly linearly (t8 MISS 22→31→41→48→58 px):
     the global affine has a small systematic bias / correlated detector jitter, so the fake cloud
     drifts ~N too (a milder form of the dead multi-baseline-flow outcome).
  2. **Real-shape drift saturates** — `orcResid` plateaus (t8 MISS 39→57→64→64→61 px): the real shape
     creeps *locally* within its radius; net displacement is bounded, not unboundedly directional.

**Verdict / takeaway.** The clean negative did **not** happen: integrated translation is **not**
sub-noise — at a *moderate fixed horizon* it lifts t8 MISS top1 from 0.32 to 0.68. The plan's error
was assuming unbounded directional drift; the real signal is a **bounded directional creep best read
at a fixed N≈30**, not a long sweep. This is a live EXP-Q2 candidate — subject to the usual caveat
that strong isolated top-1 ≠ track (the causal-key wall killed EXP-A's `topk` exactly this way), so
it lives or dies on **LOO no-regression**, not the probe.

> **RESOLVED 2026-06-20 → DEAD (see the EXP-Q2b section below).** The caveat won: the residual is a real
> measurement but ranks the real box #1 only ~0.5–0.6 even on *correctly-tracked* strong clips (t9 0.51,
> t10 0.57 — t7's 0.88 was unrepresentative), so a persistence-gated override false-captures and craters
> the strong clips (every config regresses; LOO +0.000). The N≈30 "0.68 on MISS frames" is barely above
> those clips' baseline ~0.6 — moderately-good-everywhere, never sharp enough to re-select boxes online.

### EXP-Q1b — common-mode rejection (does debiasing make the SNR rise?) — MILD DEAD END

**Hypothesis.** EXP-Q1's SNR peaked at N≈30 because the fake cloud grows ~N — *if* that growth is a
**shared** affine bias (all rigid fakes carry the same accumulated drift), subtracting the per-frame
across-box **median residual vector** (common-mode rejection) should cancel it and let the SNR keep
rising. Tested read-only in the same probe (`--debias` path, `_scan_clip(debias=True)`).

**Result: barely any change.** Pooled debiased laggard MISS sepRatio 1.91→1.59 vs raw 1.85→1.61 —
still falls. So the fake cloud's ~N growth is **NOT a shared bias** — it is **per-box detector-centroid
jitter that accumulates independently** (correlated frame-to-frame *within* a box by occlusion/shape,
so it grows ~N, but uncorrelated *across* boxes, so the median is ~0 and CMR removes nothing). This
**tightens** the EXP-Q1 conclusion: the residual is genuinely a **sweet-spot read at N≈30**, not an
integrator that improves with horizon. Do not pursue debiasing further.

### Lock-run forensics (what the floor actually is) — two distinct problems

Inspecting the longest miss runs (the ~50–68-frame episodes that dominate the floor) shows they split
into two causes that want **different** fixes:

| run | len | oracle_hit | real-box size vs median | output err | cause |
|-----|----:|-----------:|------------------------:|-----------:|-------|
| t8 f652–719 | 68 | 62/68 | 1.18× (normal) | 209 px | **pure identity** |
| t8 f347–368 | 22 | 22/22 | 1.05× (normal) | 305 px | **pure identity** |
| t5 f521–545 | 25 | 25/25 | 1.36× (real box 14px from GT!) | 218 px | **pure identity** |
| t5 f361–413 | 53 | 42/53 | **0.55× (undersized)** | 135 px | **box quality** |
| t1 f238–269 | 32 | 13/32 | 0.66× (undersized) | 72 px | box quality + detection |
| t1 f135–154 | 20 | 19/20 | 1.30× (normal) | 85 px | pure identity |

- **Pure-identity locks** (real box detected, *normal size*, output 200–300px away on a fake) → only
  an **identity-layer re-pick** moves them. The N≈30 residual ranks the real box #1 here (0.65–0.68).
- **Undersized-box locks** (t5 f361–413 at 0.55×, t1 f238–269 at 0.66×) → the real box is detected but
  too small, plausibly starving its mass / coherent-mass emission below a neighbouring fake. This is a
  **detection-quality** fix (targeted annotation/retrain), distinct from the dead global imgsz knob.

---

## DEAD ends — do not retry without a genuinely new angle

Full reasoning in `CLAUDE.md`; one-liners here.

**Identity signal (the wall — TRIPLE-CONFIRMED information limit).** On laggard lock frames the real
box is present (oracle ~0.95, top-3 ~84%) but no *per-frame* measurement ranks it #1:
- **Appearance / texture (EXP-1)** — sheet barely rotates (~0.006°/fr), real shape too slow → interior
  change below the resampling-noise floor. Structural; permanently off the table.
- **Rotation as identity, all three levels** — interior NCC (EXP-1), boundary/outline rotation
  (EXP-R1), box-AABB size-pulse (EXP-S1). All fail t7 sanity: sub-pixel spin / detector jitter swamps
  the signal. **No form of rotation-as-identity remains.**
- **Sub-box coherent-mass re-localization (EXP-A)** — real shape's coherent vectors not dominant in
  any sub-window; not a box-dilution artifact. `topk` lifted isolated top-1 but REGRESSED the trellis.
- **Per-frame / short-window translation channels** — mass, curl, coherence, |d_indep|, single-frame
  box rigid residual: all sub-noise on laggard MISS frames.
- **Integrated (cumulative sheet-frame) residual as identity (EXP-Q1/EXP-Q2b)** — the integral is a
  real measurement (t8 MISS top1 0.32→0.68 at N≈30, EXP-Q1) but **not separable enough to re-select
  boxes**: ranks the real box #1 only ~0.5–0.6 even on correctly-tracked strong clips (t9 0.51,
  t10 0.57), so any residual-keyed override false-captures and craters them (EXP-Q2b: every config
  regresses, LOO +0.000). Dead as both an additive emission and a persistence-gated override.

**Detection.**
- **Detection knobs imgsz=1024 / conf=0.10 (step2_detknob)** — oracle↑ does NOT convert; t8 oracle
  0.939→0.983 but identity stays flat 0.769. Global, no causal key.
- **Two-class YOLO, top-K / conf filter** — oracle regresses; real shape ranks 15–28 on hard frames.

**Decode / anti-lock (all fight real motion / hold a wrong incumbent / are regime-coupled).**
- Velocity cap (below p99), confirm-gate (reject far jumps), proximity-fused emission (prox_w>0),
  transition-penalty cap, coherence far-jump reacquire on `fpath_fuse`.
- **Magnitude-trust & sheet-decomposition decode hedges** — magnitude is anti-correlated with
  real-ness; only the **coherence**-trust freeze worked (shipped as `fpath_hedge`).

**Other.**
- Multi-baseline optical flow (K>1) — per-pixel LK noise swamps slow-drift gain.
- Mode routing / ensembling — oracle-router ceiling 0.858 < 0.90; no live regime key.
- Temporal-feature logreg (−0.060), rotation selector (noise), NCC / log-polar trackers.
- t1/t4 countdown-lock "bug" (step3a_lockdiag) — does not exist; lock is oracle-correct on all 10.

---

## PARKED

- **EXP-L1 — fixed-lag bidirectional decode smoother (`lag_smooth_probe.py`).** The catastrophic
  200–400px rides the smoother targets are already gone (the hedge ate them: t1 one 11-frame run, t8
  one 28-frame run, t5 zero). Residual recovery modest (~16–19% of t5/t1 misses, median > Theil–Sen).
  Revisit only as an increment (median, L≈12) if the residual integrator fails. **Its key contribution
  was the reframe** that picked EXP-Q1: the remaining gap is slow adjacent *identity* drift, not rides.

---

## EXP-Q2b — residual-keyed lock-escape OVERRIDE — DEAD (gate run 2026-06-20)

**Status: gated read-only (`ld/detect/resid_override_probe.py`), FAILED — and the failure
diagnostic also kills the lock-gate variant. Do not retry residual-keyed re-selection.**

The recommended build from EXP-Q1: *not* a naive additive emission (EXP-A showed isolated top-1
dilutes into the trellis and regresses) but a mirror of `fpath_hyst`'s distance-agnostic
hysteresis override ([identity.py:1663-1678]) keyed on **integrated-residual dominance** at N≈30 —
switch the emitted box to a challenger whose N-residual **persistently (K frames) and substantially
((1+margin)×)** exceeds the held box's. The probe simulated it **layered on the committed
`fpath_hyst` track** (carry a captured box by the affine-prediction chain; capture on persistent
challenger-dominance; release when the trellis pick rejoins), scored two ways: RAW vs `fpath_hyst`
(0.878) and the ship-path HEDGED (override → the shipped churn-freeze) vs `fpath_hedge` (0.899).

**Result — every config in the (N∈{30,45} × margin∈{.15,.30,.50} × K∈{5,8,12}) grid regresses.**
Worst-clip **−0.22 to −0.40**, concentrated on a *strong* clip (`strongW` −0.27 to −0.40), and the
**laggard mean is itself negative** (−0.06 to −0.13) — the override doesn't even net-help its targets.
LOO finds **zero admissible configs** on every fold → falls back to base, **+0.0000 raw and hedged**.
Worse than EXP-A's `topk` (which at least lifted isolated top-1).

**Root cause — the residual readout is clip-dependent; t7 was unrepresentative.** Re-ran the EXP-Q1
SANITY block (real-box residual rank #1 on *all correctly-tracked* frames, N=30) across **all 10**:

| clip | t7 | t4 | t2 | t6 | t3 | t8 | t5 | t1 | **t10** | **t9** |
|------|---:|---:|---:|---:|---:|---:|---:|---:|--------:|-------:|
| top1 |0.88|0.76|0.72|0.71|0.64|0.62|0.60|0.58|**0.57**|**0.51**|

EXP-Q1 only sanity-checked t7 (0.88) and read it as "during normal tracking the real box already has
the highest residual → the override rarely fires on strong clips." **False.** On the near-perfect
strong clips **t9 (0.51) and t10 (0.57)** a *fake* out-residuals the real box ~half the time even when
the track is correct — so any residual-driven switch craters them (the `strongW −0.34`). The 0.65–0.68
"on laggard MISS frames" that motivated EXP-Q2b is **barely above these clips' baseline all-frame
top1 (~0.6)**: the residual is moderately-good-*everywhere*, never *sharp*, and ~0.6 top-1 among ~15
boxes is far too noisy to drive box re-selection. **This also kills the lock-gate refinement** ("fire
only when the held box's residual is near the fake-floor"): it needs *"real box ⇒ high residual when
tracked"* as its key, and that key is false on t9/t10/t1/t5/t8 — on those clips the real box's residual
sits in/near the fake cloud while correctly tracked, so a floor-gate would fire *on* good tracking and
wreck it too. No gate can rescue a signal that doesn't rank the real box #1 even when it's being held.

**Verdict.** The EXP-A causal-key wall, reconfirmed at the integrated-residual level: a strong isolated
top-1 on a curated MISS subset ≠ a track. EXP-Q1's positive (integrated translation is *not* sub-noise
at N≈30) stands as a *measurement*, but it is not separable enough (sep ratio 1.6–1.9; real box #1 only
~half the time on half the clips) to re-select boxes online. The identity wall holds: on these laggards
the real shape is distinguishable only by independent translation, which even **integrated** is too
noisy to drive a switch without false captures on the strong clips. **Residual-as-identity is dead in
both forms (additive emission and persistence-gated override); do not re-propose.** *But the residual
is not wasted — see EXP-Q3 next, which reuses it as a per-box BINARY rather than a ranking.*

---

## EXP-Q3 — residual-gated decode freeze — SHIPPED as `fpath_freeze` (0.899 → 0.932, the leader)

**Status: gated read-only (`ld/detect/resid_freeze_probe.py`), PASS; built into `identity.py`; live
full-board reproduces the gate.** This is the win that came *out of* EXP-Q2b's failure.

**The reframe.** EXP-Q2b died because *ranking* 15 boxes by residual is unsharp (real box #1 only
~0.5–0.6). But a decode-layer freeze needs only a **1-box BINARY**: "is the box I'm currently HOLDING
a rigid fake?" — and that *is* sharp. Read-only separation (chosen-box N=30 residual, miss vs
correct-track frames): correct-track median ~45–91px vs miss ~9–15px, with corrP25 > missP75 on 9/10
clips (t8 the cleanest: ~55 vs ~9). A fake's residual is just detector jitter (it moves rigidly with
the sheet); the real shape's is its accumulated independent drift.

**Two facts make it land.** (1) A near-oracle **freeze-at-onset ceiling**: freezing the output at its
last in-radius position and holding through a miss run recovers t8 +0.221, t5 +0.161, t1 +0.147 —
because the real shape barely moves (1.3 px/fr), a position frozen at onset stays in-radius for 20+
frames. The shipped churn-hedge missed this because it freezes *late* (after the output already crept
onto the fake). (2) The chosen-box residual is a **causal onset detector** for that freeze.

**The mode.** Maintain the chosen box's cumulative N=30 sheet-frame residual (affine back-walk,
`identity._cumulative_residual`). When it stays below τ=15px for `consec`=1 frame ⇒ on a fake ⇒ freeze
the output toward the output from `lag`=6 frames ago (the pre-creep anchor); hold until the residual
recovers. Runs BEFORE the churn hedge (which then passes the frozen position through: a held position
reads as coherent sheet motion → churn~0 → w~1 → commit). Decode-layer only; trellis state untouched.

**The gate journey (3 iterations — recorded because the dead ends are instructive):**
- **Absolute τ, wide grid {15,20,25,30}:** in-sample τ=15 is the unique no-regression corner (+0.037,
  all 10 flat-or-up), but LOO FAILS — the t10 fold picks τ=30 (best on the other 9) which regresses
  t10 −0.051. An absolute pixel threshold is the right *mechanism* but the grid let LOO pick an
  unphysical value.
- **Relative-drop trigger (freeze when residual < frac × own EMA baseline):** WORSE — regresses every
  clip. Diagnostic: it fires on the real shape's frame-to-frame residual *noise* during good tracking.
  This proved the absolute floor is correct: the fake floor (~9–15px) is **clip-independent** (rigid
  fake = detector jitter), so an absolute threshold near it generalises; a relative one does not.
- **Absolute τ, physically-capped grid {10,12,15,18}:** PASS. Capping at the fake floor (a higher τ
  cuts into the real-shape residual distribution and regresses low-residual strong clips t9/t10) means
  LOO can't pick an unphysical knob. **All 10 folds independently select τ=15/lag=6/consec=1; LOO
  0.9359, worst-clip +0.000.**

**Result (live full-board `eval_modes`, regenerated `LEADERBOARD.md`):** mean **0.932** (+0.033 vs
`fpath_hedge`), every clip flat-or-up. t8 0.776→**0.909** (+0.133), t5 0.809→0.858, t1 0.853→0.895,
t3 0.876→0.913; t2/t10 flat; t7 →1.000. Live 0.932 lands a hair under LOO 0.9359, mostly on t1.

**Code review** (`/code-review` on the two probes): the PASS is sound — the three "spurious gain"
theories (LOO degeneracy, gate fires on slow-correct tracks, absolute-τ inert-on-large-radius) are all
refuted by the run output + separation data. Carried into the port: (a) on a chain-break frame the
freeze conservatively releases (under-fires, never inflates); (b) lag=6 means "6 before onset" only at
consec=1 (the shipped config); (c) freeze→hedge order preserved (hedge passes the freeze through).

---

## OPEN avenues (highest-EV first; grounded in the lock-run forensics above)

1. **EXP-2 — targeted annotation for the *undersized*-box locks (t5 f361–413 @0.55×, t1 f238–269 @0.66×).**
   Now the highest-EV remaining lever: `fpath_freeze` cured t8's long *identity* locks, so the residual
   floor is t3/t5's *undersized-box* runs (a detection-quality issue the freeze can't touch — if the box
   is too small, its mass and residual are both starved). Distinct from the dead global imgsz knob: a
   targeted re-annotate + retrain (medium effort, CPU) could restore the box size on this *specific*
   miss subset. Compose with the freeze.
2. **Squeeze the freeze's chain-break frames.** The freeze currently no-ops when the chosen box's N=30
   residual chain breaks (a gap/coast in the window) — conservative, but a secondary/longer-horizon
   residual or a hold-through-dropout could recover a few more frames. Cheap; gate on no-regression.
3. **t3 detection-bound half.** ~48% of t3 misses are `oracle_hit==0` (real box not detected at all) —
   a detection-targeted problem separate from the identity wall. Lower priority.

**DEAD/closed this session:** **EXP-Q2b residual override** (residual-as-*ranking* — every config
regresses; the residual ranks the real box #1 only ~0.5–0.6 even on correctly-tracked strong clips
t9/t10 → false captures crater them; the lock-gate variant dies for the same missing key);
EXP-Q1b common-mode rejection (fake floor is per-box jitter, not shared bias — debiasing does nothing).
**Residual-as-identity is dead as a ranking — but lives as a per-box BINARY (`fpath_freeze`, shipped).**

**Honest status.** `fpath_freeze` (0.932, LOO 0.9359) is the leader — the largest single lift in the
lineage (+0.033), and it raised the laggards (t8 +0.133) rather than sidestepping them. It came from
the EXP-Q1→Q2b→Q3 arc: EXP-Q1 showed integrated translation is a real *measurement* (t8 MISS top1
0.32→0.68); EXP-Q2b proved it's too unsharp to *rank* boxes (override regresses every clip — DEAD);
EXP-Q3 then used the same residual as a 1-box *binary* ("am I on a fake?") to drive a decode-layer
freeze — sharp, and shipped. The **identity wall still holds**: `fpath_freeze` never re-identifies the
real box, it just detects the box it's on has gone rigid and stops — a *position* win, not an identity
one (the last two lifts, hedge then freeze, are both decode-layer). Every identity-channel candidate
remains dead (appearance, all three rotation levels, sub-box, residual-as-ranking). Remaining floor
work toward oracle 0.958 is **detection-quality** (EXP-2 undersized-box locks t5/t3) and squeezing the
freeze's chain-break frames — not a new identity channel. The *position* ceiling is now ~0.93 and the
realistic target is the 0.958 oracle if EXP-2 lands; the *identity* ceiling is unchanged (~0.90 by
box-id alone), which is exactly why both recent wins live at the output layer.
