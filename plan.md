# LD identity — session plan (2026-06-19)

Session-scoped working doc. Tells a new agent: what we did, why we're stuck at 0.876,
and the concrete experiments to push **mean within_r ≥ 0.90**. Ground truth lives in
`CLAUDE.md` (leader, taxonomy, dead ends) and `ld/detect/LEADERBOARD.md` (numbers). If
this plan and CLAUDE.md disagree, CLAUDE.md wins — re-verify before trusting either.

---

## Current state (merged to master)

**Leader: `fpath_fuse` @ 0.876 within_r** (conditional 0.913, oracle 0.958). Up from the
prior leader `fpath_coh` 0.825. Verified by `eval_modes`, no per-clip regression.

| clip | within_r | oracle | headroom | residual-loss character |
|------|---------:|-------:|---------:|-------------------------|
| t1 | 0.783 | 0.952 | 0.17 | **lock-in**, creep onto ADJACENT box |
| t2 | 0.978 | 0.978 | ~0 | at ceiling |
| t3 | 0.852 | 0.918 | 0.07 | mixed |
| t4 | 0.849 | 0.944 | 0.09 | detection-gap + emission (fixed a lot this session) |
| t5 | 0.759 | 0.948 | 0.19 | **lock-in**, far teleport-lock |
| t6 | 0.922 | 0.970 | 0.05 | near ceiling |
| t7 | 0.989 | 0.998 | ~0 | at ceiling |
| t8 | 0.767 | 0.939 | 0.17 | **lock-in**, far teleport-lock + signal-limited |
| t9 | 0.924 | 0.973 | 0.05 | near ceiling |
| t10 | 0.937 | 0.964 | 0.03 | near ceiling |
| **mean** | **0.876** | **0.958** | | |

Gap to 0.90 is **+0.024**, concentrated entirely in three lock-in laggards: **t8 (0.767),
t5 (0.759), t1 (0.783)**.

---

## What we did this session

1. **Fixed stale docs.** CLAUDE.md claimed `fpath` 0.855 / oracle 0.974; reality was
   `fpath_coh` 0.825 / oracle 0.958. Regenerated the leaderboard, corrected every number.
2. **Built the gate `fuse_probe.py`** → sharpened the failure taxonomy. On the trellis's
   miss frames, *which channel ranks the real box #1 is regime-split*: coherence wins t4
   (0.60), curl wins t5 (0.50), mass barely wins t1/t8 (~0.40). **Max-fusion is dead**
   (top-1 0.22). Coherence is the best *candidate generator* (top-3 recall on miss frames
   0.842 vs mass 0.803).
3. **Shipped `fpath_fuse` (+0.051).** Replaced `fpath_coh`'s multiplicative `mass*(1+coh)`
   emission with an **additive** sum of cross-frame-normalized channels:
   `norm(mass) + 1.5·norm(coherent_mass) + 0.5·norm(curl)`, fed to the same Viterbi trellis.
   Tuned by LOO (`fuse_sweep.py`). Biggest gains where predicted: t4 +0.106, t5 +0.138.
4. **Window sweep → no gain.** Swept `fuse_win {8,12,16,20}` × weights; nothing beats
   `win12 cmass1.5 curl0.5` without regressing a laggard. Window has no leverage.
5. **Step-3 gate → both lock-in escapes dead.** `step3_gate.py` showed the coherence
   far-jump reacquire has no target (on-GT+far+confident only 0.085/0.020 on t1/t8). The
   transition-penalty cap helps t8 a little but regresses the strong clips (net −0.025..−0.075).

---

## Why we're stuck — the limiting factors

The detector is solved (oracle 0.958); **100% of the residual loss is identity** — picking
the wrong box when the right one is present. Three compounding walls:

1. **Lock-in is desirable on strong clips, fatal on laggards — and there is no causal key
   to tell them apart.** The Viterbi transition penalty makes the path self-reinforcing.
   That's why t2/t6/t7/t9/t10 sit at 0.92–0.99. Any *global* anti-lock mechanism
   (transition cap, softening) trades those clips away for marginal laggard gains. This is
   the central tension and the reason every global escape has failed.

2. **The motion channels (mass / coherence / curl) give top-3 but not reliable top-1 on
   the hard frames.** On miss frames the real box is in the channels' top-3 ~84%, but no
   channel's argmax reliably picks it (t8: mass 0.32, coh 0.24, curl 0.17). The Viterbi
   converts a lot of that top-3 into a track, but on the residual frames the channels
   genuinely cannot separate the real shape from a specific adjacent/confident fake. **t8
   is signal-limited**: coherence's argmax is on the real shape <30% of its miss frames.

3. **The two failure geometries need opposite fixes.** `step3_gate` (2026-06-19):
   - **t1 = adjacent creep** (challenger far only 0.28 of misses) → far-jump escapes can't
     fire; the wrong box is <1.5 radii away.
   - **t5/t8 = far teleport-lock** (challenger far 0.63) → far-jump *could* fire, but the
     coherent challenger is wrong there (on-GT 0.29–0.40), so it would jump to another fake.

**Net:** `fpath_fuse` is at/near the ceiling of what *motion-only* evidence + a single
forward Viterbi pass can extract. Reaching 0.90 needs **either a new orthogonal
discriminator** (so bad locks become distinguishable from good ones) **or better detection
on the confusable frames** — not another trellis/escape tweak. Confirm this framing with a
fresh `fuse_probe`/`step3_gate` run before investing; do not trust these numbers blind.

---

## Future experiments (gated, highest-EV first)

Discipline for every item: **gate first** (cheap read-only probe — does the signal exist?),
then build only if the gate passes, then accept only if **LOO mean improves with no
per-clip regression vs 0.876** (`python -m ld.detect.loo`). Mirror the
`fuse_probe → fuse_sweep` pattern. Probes are "run, read, decide, delete-or-keep".

### EXP-1 — Appearance/texture channel orthogonal to motion (HIGHEST EV)
**Hypothesis:** the real shape's *interior* changes frame-to-frame (it rotates/moves
independently) while a fake's interior is rigid with the sheet. A rotation-aware appearance
descriptor gives a top-1 signal where motion fails — exactly t8's gap.
- **Why new:** NCC failed (translation-only, de-correlates under rotation); log-polar
  phase-corr was regime-coupled. The new angle is a **rotation-INVARIANT magnitude
  descriptor** (ring/annular intensity histogram, Fourier-Mellin magnitude, or Zernike
  moments) of each box ROI, compared frame-to-frame *after de-rotating by the global sheet
  rotation* (from the paper affine). Score = descriptor change; the real shape changes more.
- **Gate (`exp1_appearance_probe.py`):** per box on t8/t1 miss frames, compute the
  descriptor-change; does the real (GT) box rank top-1/top-3? If top-1 > 0.4 on t8 (where
  motion gives 0.32), it's a genuinely complementary channel → build it as a 4th additive
  emission term and re-sweep weights. If top-1 ≈ motion, drop it.
- **Files:** new probe; if it passes, add `appearance_w` term in
  `track_fused_path_identity` emission alongside cmass/curl; extend `fuse_sweep`.

### EXP-2 — Targeted detector annotation on confusable frames (PROVEN LEVER)
**Hypothesis:** tighter/cleaner boxes on the t8/t1 confusable frames change the coherence
computation and reduce identity ambiguity. Detection improvement is the one lever with a
track record (oracle 0.915 → 0.958 historically lifted identity in lock-step).
- **Gate:** inspect the t8 teleport-lock frames (f284–380 region per the old taxonomy;
  re-derive from `data/detect/eval/t8_cropped_trimmed__fpath_fuse.csv`). Is the real box
  missing, oversized, or merged with a neighbour? If detection is sloppy there → annotate.
- **Build:** extract those frames via `annotate_s.py`, add ~5–10 targeted boxes, retrain
  `yolov8n_single_combined` (see CLAUDE.md "How to train"). Re-run full `eval_modes`.
- **Risk:** low (additive examples); the lesson is targeted coverage, not schema changes.

### EXP-3 — Evidence-hysteresis switch WITHOUT the far requirement (CHEAP, MEDIUM EV)
**Hypothesis:** the far-jump reacquire died because t1's creep is *adjacent*; a switch that
fires on *accumulated* coherent-mass dominance regardless of distance could catch adjacent
creep before it locks. (This is `accum`'s mechanism layered on the fuse trellis.)
- **Gate (`exp3_switch_probe.py`):** on t1/t5/t8 miss frames, does the box with the highest
  *long-window* (e.g. 24–40 frame) accumulated coherent-mass sit on GT — *dropping the far
  filter* that step3_gate imposed? step3_gate's on-GT (ignoring far) was t1 0.39 / t5 0.40 /
  t8 0.29 at win12; test whether a LONGER window raises on-GT materially. Only worth building
  if on-GT clears ~0.5 on at least one laggard.
- **Build:** add a hysteresis override to `track_fused_path_identity`: maintain per-box
  leaky-integrated coherent-mass; if a challenger leads the path's box by margin M for K
  consecutive frames, switch + reset path memory. LOO-tune (M, K, window) conservatively to
  protect strong clips. Reuse `accum`'s constants as a starting point.

### EXP-4 — Learned per-box discriminator, physically-motivated features (HIGH EFFORT)
**Hypothesis:** no single hand-set channel separates the real shape, but a small model over
several orthogonal features might. logreg overfit before — the fix is heavy regularization +
strict leave-one-CLIP-out and physically-motivated features only.
- **Features per box:** coherent-mass, curl, paper-residual, mass, size-consistency vs
  neighbours, appearance-change (from EXP-1), temporal persistence.
- **Gate:** leave-one-clip-out AUC for "is real shape" must beat the best single feature by a
  clear margin on held-out clips, or it's overfitting again.
- **Only attempt after EXP-1** (it supplies the orthogonal feature that makes this viable).

### Lower priority / likely dead
- Bidirectional / longer fixed-lag decode: sustained locks (t8 drifts at f239 and *stays*)
  exceed any spec-legal lag (~10–15 frames); only helps transient creep. Low EV.
- Any *global* transition softening / cap: dead (regime-coupled, hurts strong clips).
- Routing/ensembling existing modes: oracle-router ceiling is only 0.858 < 0.90. Dead.

---

## How to work here

- **Eval:** `python -m ld.detect.eval_modes --weights data/detect/runs/yolov8n_single_combined/weights/best.pt`
  (full board, ~18 min, cache-backed). Subset: `--modes fpath_fuse --clips t8 t5` (overwrites
  LEADERBOARD.md — regenerate full board before committing). Honest generalization:
  `python -m ld.detect.loo`.
- **Per-frame forensics:** `data/detect/eval/<clip>__<mode>.csv` (idx, state, x/y, gt, err,
  within_r, oracle_err, oracle_hit). Read these to characterize a failure before coding.
- **Bar:** accept a change only if LOO mean improves with **no per-clip regression** vs 0.876.
- **Discipline:** strictly causal for anything shipped; gate every idea with a read-only probe
  before building; record dead ends in CLAUDE.md.
- **Key files:** `ld/detect/identity.py` (modes, `track_fused_path_identity`, `_dispatch_mode`,
  `ALL_MODES`), `ld/detect/fuse_sweep.py` (LOO weight/window sweep harness, channels
  precomputed once), `ld/detect/fuse_probe.py` / `ld/detect/step3_gate.py` (this session's
  gates — templates for new ones), `ld/vision/motion.py` (`estimate_motion`, `box_coherence`),
  `ld/detect/eval_modes.py`, `ld/detect/loo.py`.
