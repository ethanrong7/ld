# plan.md — Findings summary: the `field_lag` fixed-lag confirmation smoother

This file recorded the implementation plan for the fixed-lag confirmation smoother.
The work is **done, validated, and shipped** (mode `field_lag`, now the default). Below
is the summary of how we got here and what landed, for anyone picking up the thread.

## Outcome (shipped)

`field_lag` is the new leader and default identity tracker.

| metric | `field` (old default) | `field_lag` (new default) |
|---|---|---|
| in-sample within_r (t1–t10) | 0.714 | **0.721** |
| **LOO held-out within_r** | 0.693 | **0.721** (+0.028) |

- Winning config `lag_k=8, confirm=0.5` was selected by **all 10 LOO folds** (optimism
  gap 0.000 — generalizes, does not overfit the t-set).
- **No per-clip regression** beyond noise (worst t6 −0.004, t8 −0.003); gains up to
  +0.017 (t3, t10).
- `field` baseline left byte-identical (verified t2 0.873, t5 0.409, t7 0.796,
  t8 0.589) so the A/B and the 0.714/0.693 baseline remain valid.

## What it does (mechanism)

`field_lag` is a thin post-processor over `field`'s per-frame box-pick sequence. It
**defers committing a pick until an 8-frame lookahead confirms the same box** is field's
choice in ≥50% of the window; otherwise it emits field's pick unchanged (do-no-harm). It
emits frame `t−K`, so it is legitimately online (a fixed lag, never unbounded future
access). The ~8–12 frame lag is physically free: the real shape moves median 1.3 px/frame
(p99 17.8) with radius ~56, so it cannot leave its own radius in that window (CLAUDE.md
"Latency budget").

This directly attacks the diagnosed failure mode — **creep**: a transient single-frame
slide onto an adjacent fake that poisons the constant-velocity estimate and locks in the
error. The confirmation buffer outvotes a 1–2 frame creep blip before it can corrupt the
velocity. The +0.03 magnitude matches the Viterbi-ceiling lookahead bound (K=0 0.724 →
K=15 0.753).

## Code changes (all committed)

- `ld/config.py` — `FIELD_LAG_K=8`, `FIELD_LAG_CONFIRM=0.5` (the LOO-selected winner).
- `ld/detect/identity.py` — `track_field_lag_identity` (the wrapper); an inert
  `chosen_centroids` out-param on `track_field_identity` (keeps `field` byte-identical);
  `field_lag` registered in `ALL_MODES` and `_dispatch_mode`; `run_clip` default →
  `"field_lag"`.
- `ld/detect/loo.py` — added a `--mode field_lag` grid (`lag_k × confirm`); original
  `field` grid kept intact for baseline reproducibility.
- `ld/detect/LEADERBOARD.md` — regenerated on the full t1–t10 set; `field_lag` rank 1.

## How we got here (the reasoning arc)

The bottleneck is identity/discrimination, not detection (oracle ~0.93). The failure is
**creep, not jump** (confirmed from traces). A series of avenues were each killed cheaply
by a pre-build "gate" before committing to an implementation — the discipline that kept
this from burning effort:

1. **Output-side fixes** (velocity cap, confirm-gate) — FAILED; they fight jumps, but the
   failure is creep.
2. **Proximity-fused emission** (`fpath` prox_w>0) — FAILED; CV prior reinforces lock-in.
3. **field⇄fpath router** (3 forms: per-clip regime, per-frame agreement, fpath margin) —
   all FAILED; the ~0.83 oracle has no causal key. (CLAUDE.md "Router dead end".)
4. **Box-rigid residual signal** — FAILED; weak per-frame AND not complementary to flow;
   flow's top-3 availability is already 0.886, so the gap is integration, not signal.
   (CLAUDE.md "Box-residual dead end".)
5. **Successful-example video** (`data/successful_examples/…`, local-only) — a third party
   tracks the real shape through full camouflage. A two-part gate established the key
   mechanism: **trajectory-continuity is a 0.987 per-step cue WHEN anchored to truth**
   (saliency only 0.78), **but a trajectory-LED tracker collapses to 0.157** (error
   accumulation — one wrong pick poisons the velocity, no recovery). `field` already has
   the right architecture (saliency-led + CV gate). The lever is therefore to **protect
   the trajectory anchor from the first wrong step**, not to add signal or change the
   leader — which is exactly the fixed-lag confirmation buffer.

Detailed dead-end records live in `CLAUDE.md` (the canonical project doc) and the
auto-memory notes. The one-off gate scripts (`router_signal`, `router_agreement`,
`resid_complement`, `traj_gate`) were deleted after their findings were recorded.

## Guardrails for future work (do not re-attempt)

- No velocity/speed cap on emitted position (failed smoother).
- No prediction-led pick (the 0.157 collapse).
- No proximity/CV-term fusion into an emission (`fpath` dilution).
- No unbounded lookahead — keep the lag fixed (≤ ~15) so it stays online.
- Always quote LOO, not in-sample, as the real metric; accept changes only on a held-out
  win with no per-clip regression.
