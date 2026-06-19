# LD identity — session plan: CUMULATIVE SHEET-FRAME RESIDUAL (long-horizon drift integrator) — written 2026-06-20

Handover doc for the next agent. Ground truth on numbers/dead-ends is CLAUDE.md + `ld/detect/LEADERBOARD.md`.
**Current leader: `fpath_hedge` @ 0.899** (oracle 0.958, LOO 0.8988, no per-clip regression). Laggards:
**t8 (0.776), t5 (0.809), t1 (0.853).** This plan proposes the next thread — a **long-horizon cumulative
independent-displacement discriminator measured in the sheet's reference frame** — with the reasoning that
selected it and a concrete gate-first implementation. It is a HYPOTHESIS with a hard gate; not yet validated.

> Last session ran the **fixed-lag bidirectional decode smoother** gate (EXP-L1, `lag_smooth_probe.py`) and
> it came back WEAK — see "Prior-session status" at the bottom. The most useful thing that gate revealed is
> what *picks this plan*: the catastrophic 200–400px rides are already gone (the hedge ate them), so the
> residual misses are SHORT, SLOW, ~1-radius adjacent drifts. That is not a decode/position problem any more —
> it is an identity problem on slow directional drift, and that is exactly what long-horizon integration is for.

---

## The finding (one sentence)

**Every identity attempt measured the real shape's independent translation PER-FRAME or over 8–10-frame
windows — exactly where ~1.3 px/frame sits *under* the affine/detector noise floor — so the "information
limit" is really an SNR limit, and the standard fix is to INTEGRATE: transport every box's centroid into the
sheet's reference frame via the cumulative inverse RANSAC affine and accumulate its divergence from rigid
prediction over a long causal horizon (N≈30–90 frames), where a fake's residual is identically ~0 at all
horizons while the real shape's directional drift grows ~N·v and zero-mean noise grows only ~√N·σ.**

---

## 1) How I arrived at this (the reasoning chain — read this; it's the whole bet)

**Step A — the lag-smoother gate reframed the remaining gap from DECODE back to IDENTITY.** The gate's step-0
duration analysis (`lag_smooth_probe.py`) found that `fpath_hyst` has almost no catastrophic rides left — only
t1 (one 11-frame run) and t8 (one 28-frame run); **t5 has zero grown swept runs.** The hedge already ate the
big sweeps. So the residual misses are no longer "the output rode a fake 200–400px away" (a decode problem,
now mostly solved) — they are "**identity picked the adjacent fake and the real shape is ~1 radius away,
slowly creeping off it.**" Decode hedging cannot fix that: the correct answer is a few pixels away, not 300.
The remaining gap must be won at IDENTITY, on **slow directional drift**.

**Step B — the "information limit" is an SNR statement, not an information one.** CLAUDE.md's triple-confirmed
wall says the real shape is "distinguishable ONLY by independent translation relative to the rigid sheet, and
on drift-locks it translates slowest → least signal." Read literally that is an SNR claim: signal (independent
translation) is small *per frame* relative to noise (affine + detector jitter). It was triple-confirmed for
**appearance** (EXP-1), **sub-box coherent-mass** (EXP-A), and **detection knobs** (step2_detknob) — none of
which is translation. The translation signal itself has only ever been measured PER-FRAME (mass, curl,
coherence, |d_indep|) or over 8–10-frame windows (coherent-mass, EMA hysteresis, churn). **Nobody has measured
it the way you lift any weak signal out of noise: by integrating it over a long horizon.**

**Step C — integration works *specifically because* a fake's residual is identically zero.** A fake's centroid
is, by construction, a fixed affine transform of the sheet at EVERY frame → its residual from the
rigid-affine prediction is **0 at all horizons** (only noise). The real shape drifts **directionally** (the
failure mode is described everywhere as *coherent slow creep* onto an adjacent fake). Transport each box's
centroid back into a fixed reference frame (frame `t−N`) via the cumulative inverse affine and accumulate the
divergence from rigid prediction:
  - real shape: directional drift integrates as **~N·v** (≈ N·1.3 px if roughly straight; ~80px over N=60),
  - affine/detector noise: zero-mean → integrates as **~√N·σ**.
For directional drift, signal grows FASTER than noise with N. Even on a drift-lock where the real shape is
slowest, a fake's residual is still ~0, so the real shape only has to out-drift the *noise floor*, not the
fakes' motion. **This is the one regime where "slowest on drift-locks" stops being fatal.**

**Step D — the property that makes identity hard per-frame makes the ASSOCIATION easy.** To accumulate a box's
residual over N frames you must correspond it across frames. In the affine-residual frame this is trivial:
fakes land EXACTLY where the inverse affine predicts, so nearest-neighbour-after-affine is unambiguous; the
real shape's per-frame deviation is small enough that its chain never breaks, it just accumulates. The
data-association failure that capped the EMA-hysteresis (nearest-centroid mis-association across sheet
translation, see CLAUDE.md EXP-3b) **dissolves** here, because the affine prediction — not raw proximity — is
the association key. And it is **fully causal** (look back N frames; no lag budget needed at all).

**Why this and not the alternatives:**
- *The fixed-lag decode smoother (last plan).* Gate came back weak — the big rides it targets are already gone
  (Step A). Parked, not pursued (Prior-session status).
- *Extend the hedge to t8's non-swept misses (old avenue 0).* Decode-layer; same ceiling as the hedge — can
  only move position toward `prev_out`, can't resolve a ~1-radius adjacent drift. Increment, not breakthrough.
- *A new appearance/rotation channel.* Dead at all three measurement levels (CLAUDE.md). Do not.
- *Better detection (EXP-2 imgsz / annotation).* step2_detknob: oracle↑ doesn't convert because identity
  can't rank the present box. This plan is the thing that would *make* a present box rankable.

This residual integrator is the unique candidate that (a) targets the ONE signal everyone agrees exists
(independent translation), (b) measures it where it is STRONGEST (integrated, directional, long-horizon)
instead of where every prior attempt measured it (per-frame, sub-noise), and (c) has self-stabilising
association precisely because fakes are rigidly predictable.

---

## 2) Why this is genuinely new (it sits next to 2 dead ideas — it must dodge both)

| Dead/parked thing | What it did | How the cumulative residual differs |
|-------------------|-------------|-------------------------------------|
| Multi-baseline optical flow (K>1), dead | accumulated **per-pixel LK** flow over K baselines → LK noise swamped slow drift | Accumulates the **global RANSAC affine** (robust, one transform/frame) applied to **detector centroids** (thousands-of-pixels aggregates), NOT raw per-pixel flow. Different, far smaller noise source. **But the precedent is the central risk — gate the SNR-vs-N curve (§3).** |
| Box-level rigid residual, dead ("not complementary") | measured the residual at a **single frame** | Single-frame residual IS sub-noise (that is the wall). This integrates the same quantity over N≈30–90 → lifts it above the √N noise floor. The whole bet is that integration changes the SNR. |
| EMA-coherent-mass hysteresis (`fpath_hyst`, shipped) | carried evidence by **nearest-centroid** association over a short EMA | Carries by **affine prediction** (fakes are exactly predictable → clean chain) and integrates **translation** (the real signal), not coherent-mass (a short-window proxy). |

---

## 3) The open question (the make-or-break, NOT yet answered)

**Does the real shape's cumulative residual rise above the FAKE-CLOUD noise floor as N grows — and keep
rising?** Two specific failure modes to watch:
- **Affine/detector noise may accumulate as fast as the drift.** If the global affine has a small systematic
  bias (not zero-mean) or the detector centroid jitter is correlated, the fake cloud's cumulative residual
  grows ~N too, and the SNR never separates — this is the multi-baseline-flow outcome in a new guise. The gate
  MUST plot the separation ratio **as a function of N**: real-residual / p90(fake-residual). If it RISES with
  N → integration is working. If it is flat or falls → noise dominates, stop.
- **Drift may be a random walk, not directional.** If the real shape's independent motion is direction-random
  (not coherent creep), it integrates as √N·v — same order as noise → no separation. The failure descriptions
  say "coherent slow creep," which is directional, so this should be OK — but verify on the laggard MISS
  frames specifically (t8/t5/t1), not just the clean clips.

---

## Experiments (gate FIRST — do not build the mode until §3 is answered on t8/t5/t1)

Per project discipline: read-only probe, accept only on LOO mean improvement with **no per-clip regression**
vs 0.899. Fully causal (look-back window of past frames). New file `ld/detect/sheet_residual_probe.py` —
**clone the read-only scaffold from `ld/detect/hedge_probe.py`**: it already provides `_read_track` (the
`fpath_hyst` chosen-box centroid + gt + within_r + oracle per frame), `_affines` (per-frame global sheet
affine `T_t = estimate_motion(prev,cur).affine`, **already cached to `data/detect/cache/_hedge_aff_*.pkl` for
all 10 clips**), and `_seed` (radius). Boxes per frame come from `detect_fusion_clip(weights, clip,
use_cache=True)` (cached). **No video decode, no pixel pass, no re-detection needed.**

### EXP-Q1 — measure the cumulative-residual SNR vs N (the gate)
For each clip, for each frame `t`, over a causal look-back window of N past frames (default sweep
`N ∈ {15, 30, 45, 60, 90}`):

  1. **Build the per-box backward correspondence chain by affine prediction.** Let `cents[k]` be the box
     centroids at frame `k` (from `detect_fusion_clip` packs). Starting from each box `i` present at frame `t`,
     walk backward: predict its position at `k−1` by `T_k⁻¹` (invert the cached 2×3 affine), snap to the
     nearest actual centroid in `cents[k−1]` (gate the snap at, say, < radius so a broken chain is dropped).
     This yields, for box `i`, a chain `p_t, p_{t−1}, …, p_{t−N}`.
  2. **Compute the cumulative residual in the reference frame `t−N`.** Transport `p_t` back to the `t−N` frame
     via the cumulative inverse affine `T_{t−N+1}⁻¹ ∘ … ∘ T_t⁻¹` → `p_ref`. The residual is
     `resid_i = |p_ref − p_{t−N}^{(i)}|` (how far the box ended up from where rigid motion predicts it should
     be, given where its ancestor was). A fake: `resid ≈ 0` (+ noise). The real shape: `resid ≈` accumulated
     independent drift. (Equivalently accumulate the per-step residual `p_k − T_k(p_{k−1})` in a common frame —
     report whichever is cleaner; both measure the same thing.)
  3. **Report, per clip and overall, on the laggard MISS frames** (`fpath_hyst` miss-mode:
     `within_r==0 & oracle==1`) and as a t7-style SANITY on the easy clips:
     (a) the **separation ratio** `resid(oracle box) / p90(resid over all fake boxes)` and where the oracle box
         RANKS by `resid` (#1 fraction, top-3 fraction) — overall and on MISS frames;
     (b) the **SNR-vs-N curve**: does (a) RISE as N goes 15→90? (the make-or-break — a flat/falling curve =
         noise-dominated = the multi-baseline-flow dead end recurring);
     (c) the **fake noise floor**: mean/p90/p99 of `resid` over fake boxes at each N (this is the √N noise term
         — it must grow SLOWER than the oracle residual).

**Gate question (the bar to clear before any build):** on t8/t5/t1 MISS frames, does the oracle box's
cumulative residual rank it **#1 materially more often than mass (0.32–0.41) / curl / coherence do** (those are
the channels it must beat — see CLAUDE.md failure taxonomy), AND does the separation ratio **RISE with N**? If
the oracle box separates and the SNR climbs with N → proceed to EXP-Q2. If the fake cloud grows as fast as the
oracle residual (flat/falling SNR) or the oracle box still doesn't rank #1 → the translation signal is genuinely
sub-noise even integrated → the information limit is real, **stop and record it** (that is itself a decisive,
publishable result: it would *prove* 0.92–0.93 is the ceiling for this detector stack).

### EXP-Q2 — wire the residual into the trellis (ONLY if Q1 passes)
The residual is an **identity channel**, so it goes into the emission, mirroring `fpath_fuse`'s additive
channels (NOT max-fusion — CLAUDE.md: max amplifies fake spikes; only additive weighted sums have ever won).
In `track_fused_path_identity` ([identity.py:1482]), add a `resid_w` channel computed exactly like `cmass`/
`curl` ([identity.py:1594-1608]): per frame, compute each current box's cumulative residual over the causal
look-back, normalise cross-frame by its own running peak EMA (the load-bearing coast trick), and add
`resid_w * resid[i]` to `emis[i]` ([identity.py:1634-1635]). **Two integration forms to test (LOO picks):**
(a) additive emission channel (above); (b) a distance-agnostic **hysteresis override** like `fpath_hyst`'s
([identity.py:1663-1678]) but keyed on persistent residual dominance instead of EMA-coherent-mass — better
suited to the slow-adjacent-creep failure because it can switch to a box ~1 radius away that the transition
prior would otherwise hold off. Keep the hedge stacked on top (decode layer, untouched). Register `fpath_resid`
(or fold into a new leader) in `ALL_MODES`/`BOARD_MODES` ([identity.py:1846/1854]), dispatch at L1891. LOO-tune
`resid_w` + N via a `*_probe.py` LOO block (clone of `hedge_probe.py`'s, channel precomputed once per clip).
**Accept only on LOO mean ↑ with no per-clip regression vs 0.899.** Then regen the full 8-mode board before
committing.

**Success criteria for the session:** raise the **mean above 0.899**, no per-clip regression, LOO-honest. The
realistic target if it works is a genuine FLOOR lift (t8/t5/t1 toward their ~0.94–0.95 oracle), because this is
an identity fix that attacks the slow-creep misses the hedge can't — unlike the decode wins, which only moved
position. If it works it is the first identity-layer win since `fpath_hyst` and the first to attack the floor
rather than sidestep it.

---

## 4) Realistic ceiling (be honest)

- **Upside is a real floor lift, IF the SNR-vs-N curve climbs.** This is the only proposed method that could
  move t8 (0.776) — its misses are slow coherent creeps, the exact regime integration is built for, and the
  hedge provably can't touch them (it only freezes toward `prev_out`). Oracle caps the laggards at ~0.94–0.95,
  so the headroom is large (t8 +0.16, t5 +0.13, t1 +0.10) IF identity becomes rankable.
- **Base case if EXP-Q1 fails (flat SNR):** this is the LAST untested form of the one real signal. If
  integrating translation over a long horizon STILL can't separate the real shape from the noise floor on the
  laggard MISS frames, then the information limit is not just triple-confirmed for the wrong observables — it is
  confirmed for the RIGHT one, integrated optimally. That conclusively establishes **0.92–0.93 as the ceiling
  for this detector/identity stack**, and the honest next move is to accept it (or change the detector, not the
  identity). Be explicit about which outcome you got — a clean negative here is worth more than another weak
  decode tweak.

---

## Dead-end guardrails (dodge all of these)

- **Plot SNR vs N — that IS the gate.** A single N that looks OK is not enough; the separation ratio must RISE
  with N. A flat curve = noise accumulates as fast as drift = the multi-baseline-flow dead end. Do not proceed
  on one cherry-picked N.
- **Use the global RANSAC affine + detector centroids, NOT per-pixel flow.** Per-pixel LK accumulation is the
  dead multi-baseline-flow path. The whole reason this might dodge it is the far lower noise of one robust
  affine/frame applied to centroid aggregates.
- **Association by affine PREDICTION, not raw proximity.** Nearest-centroid-without-affine is the dead
  EMA-hysteresis association failure. Snap to the affine-predicted spot; drop chains that break (> radius).
- **Additive emission only — never max-fusion.** CLAUDE.md: max-combining amplifies fake spikes (fuse_probe
  top-1 0.22). Every identity win has been an additive cross-frame-normalised weighted sum.
- **Identity-layer change is allowed here (unlike the hedge).** This is a new emission channel / override, so
  it DOES enter the trellis — that is correct and intended. Keep the decode-layer hedge stacked unchanged on
  top; do not fold the residual into the proximity prior (that is the dead proximity-fused emission).
- **No new appearance/rotation/sub-box/detection-knob channel.** All dead (CLAUDE.md). This plan is translation
  only, measured differently.

---

## How to work here

- **Eval:** `python -m ld.detect.eval_modes --weights data/detect/runs/yolov8n_single_combined/weights/best.pt`
  (8-mode board incl. `fpath_hedge`, cached). Subset `--modes <name>` OVERWRITES LEADERBOARD.md — regen full
  board before committing. LOO for fpath-family hyperparams is via the `*_probe.py` LOO blocks, not `loo.py`.
- **Visual forensics:** `_crop_probe.py <clip> <start> <step> <n>` → cursor-stripped GT-centered crops in
  `_crops/`. Use on the laggard creep ranges (`step0_failchar`: t1 f231–241, t8 f340–367) to confirm the drift
  really is directional (the integration's load-bearing assumption) before coding.
- **Per-frame traces:** `data/detect/eval/<clip>__<mode>.csv` (idx,x,y,gt_x,gt_y,within_r,oracle_hit,err_px).
- **Bar:** accept only on LOO mean ↑ with **no per-clip regression** vs 0.899. Gate before building.
- **Key files / scaffolds (with line refs):**
  - `ld/detect/hedge_probe.py` — **clone for `sheet_residual_probe.py`.** Provides `_read_track`, `_affines`
    (per-frame affine, cached to `_hedge_aff_*.pkl` — invert these for the back-transport), `_seed`. `_apply(T,p)`
    at L116 transports a point by an affine; you need its inverse (cv2.invertAffineTransform).
  - `ld/detect/identity.py` — additive channel template (`cmass`/`curl`) at **L1594-1608**, emission assembly
    at **L1630-1635**, hysteresis override template at **L1663-1678**, `track_fused_path_identity` at **L1482**,
    `ALL_MODES`/`BOARD_MODES` at **L1846/1854**, dispatch at L1891. `_box_coherent_mass`/`_box_rotational_curl`
    show the per-box channel pattern. `MotionField.affine` via `ld/vision/motion.py` (`estimate_motion`).
  - `ld/detect/lag_smooth_probe.py` — last session's parked EXP-L1 probe (weak gate; kept as diagnostic).
  - `_crop_probe.py`, `_crops/` — throwaway visual tooling (safe to delete/regenerate).
- **Physics (the integrator's hard prior):** real shape speed median 1.3 px/fr, **p99 17.8, max 44.7**; radius
  ~56px; failure is CREEP (small directional steps onto an adjacent fake), which is what integration exploits.

---

## Prior-session status (don't re-litigate)

- **EXP-L1 (fixed-lag bidirectional decode smoother, `lag_smooth_probe.py`) — WEAK, parked.** Hypothesis: emit
  frame `t−L` after seeing `t+1…t+L`, replace the chosen position with a radius-bounded robust fit
  (median / Theil–Sen over `[t−L,t+L]`) when it exceeds the physical bound (p99 17.8 / max 44.7). Result:
  (1) **step-0 duration** — `fpath_hyst` has almost no catastrophic rides left (t1 one 11-frame run, t8 one
  28-frame run, **t5 zero grown runs**): the hedge already ate the big sweeps, so the bidirectional buffer has
  little to reject. (2) On t5/t1 MISS frames the gate technically passes (median recover 0.14–0.16 vs damage
  0.018, flag firing 34–40% at L=12–15; **median > Theil–Sen**, which injects strong-clip damage), and median
  replacement is broadly low-damage (strong-clip damage ~0). But recovery is **modest** (~16–19% of t5/t1
  misses at best; only ~40–47% of *flagged* misses) and the mechanism is short-excursion smoothing, NOT the
  catastrophic-ride rejection the plan predicted. EXP-L2 was NOT built (the realistic lift is +0.01–0.02 and
  the probe can't establish the real LOO no-regression bar). **The decisive takeaway is step-0's reframe:** the
  big rides are gone → the remaining gap is slow adjacent identity drift → that picks THIS plan. The smoother
  is available to revisit (median, L≈12) as an increment if the residual integrator fails.
- `fpath_hedge` (0.899, LOO 0.8988) remains the leader; it is the CAUSAL special case of the (now-parked) lag
  smoother. Its dead sub-variants are logged in CLAUDE.md.
- **Identity wall recap (read CLAUDE.md dead-ends before proposing anything):** appearance (EXP-1), sub-box
  coherent-mass (EXP-A), all three rotation levels (EXP-1/EXP-R1/EXP-S1), and detection knobs (step2_detknob)
  are all dead. The ONE signal left standing is **independent translation** — and this plan is the first to
  measure it by long-horizon integration rather than per-frame. Do NOT re-propose appearance, rotation (any
  level), sub-box windows, detection knobs, mode routing, or per-frame translation channels.
- Diagnostics kept: `lag_smooth_probe.py`, `box_pulse_probe.py`, `rot_probe.py`, `hedge_probe.py`,
  `step0_failchar.py`, `expA_subbox_probe.py`, `step2_detknob.py`, `step3a_lockdiag.py`,
  `exp1_appearance_probe.py`, `fuse_probe.py`.
