# LD — session plan: EMIT A SINGLE TRACED POINT (x,y), not a box — written 2026-06-20

Handover for the next agent. Ground truth on numbers/dead-ends is CLAUDE.md + `ld/detect/LEADERBOARD.md`
(t1–t10) and `ld/detect/LEADERBOARD_additional_evidence.md` (held-out validation, 2026-06-20).
**Current leader: `fpath_freeze` @ 0.932** (t1–t10) / **0.837** (additional_evidence valid-13).

## The objective shift

So far the whole project answers **"which of the ~12 boxes is the real shape?"** (box-level identity), then
emits that box's centroid as the position. The new goal is to produce **one precise point — the (x, y) the
player's cursor would trace** — i.e. regress the location, not just select a box.

- **We are NOT moving a cursor / controlling the mouse.** The deliverable is purely the predicted **(x, y)
  coordinate per frame** (where the cursor *should* point), emitted online and strictly causally.
- The scoring metric is unchanged and already point-based: `within_r` = is the emitted point inside the shape
  radius of the GT crosshair. So this work is **on-metric** — every gain shows up directly in `within_r`.
- **Evidence overlay change (do this first, it's small):** subsequent evidence videos must show the emitted
  point as a **visibly filled red dot** at (x, y) — not the current thin marker / HUD coordinate. See Phase 0.

## Why there is headroom here (the key insight)

The pipeline already emits a point (`TrackPoint.x/.y`), but it is just the **chosen box's centroid** run
through a CV predictor. The GT crosshair sits at a *specific spot on the shape*, which is **not** the box
centroid when the detector box is **oversized / merged** — exactly the t5 (undersized/odd box) and a03
(1440p-downscaled, oracle 0.942 but identity ~0.64) failure profile. On those frames the **right box can be
chosen and the point still misses** because centroid ≠ true target. Closing that centroid→target offset is a
**localization** problem, fully separate from the (triple-confirmed, walled) identity-ranking problem — so it
is genuinely new headroom toward the 0.958 oracle that does NOT require winning the identity argument.

Note this is **not** the dead EXP-A: that was sub-box coherent-mass used to *re-rank which box* (dead). Here
the box is already chosen; we only refine the *point inside/near it*. Localization ≠ ranking — re-examine, but
keep them conceptually distinct.

## Phase 0 — evidence overlay: filled red dot (small, do first)

In `ld/detect/render_evidence.py` (`render_clip`, the draw block at **L76–96**): today it draws
`cv2.circle(..., int(radius), col, 2)` (radius outline) + `cv2.circle(..., 4, col, -1)` (tiny dot) + a HUD
string. Change the estimate marker to a **single prominent filled red dot** at `(int(tp.x), int(tp.y))`
(e.g. `cv2.circle(frame, (x,y), 8, (0,0,255), -1)` with a thin dark outline for contrast). Keep the cyan GT
crosshair + red miss-line for diagnostics (GT is GT-only, never an input). Drop or shrink the radius ring so
the dot reads clearly. Re-render the leader on the validation set to confirm the look:
`python -m ld.detect.render_evidence --weights .../best.pt --mode fpath_freeze --clips data/additional_evidence/*.mp4`.

## Phase 1 — GATE the point-refinement (read-only, before any build)

Per project discipline: read-only probe first; accept only on LOO mean ↑ with **no per-clip regression**.

Measure, on frames where the **correct box is already chosen** (`fpath_freeze` within_r==1 OR oracle box ==
chosen box), the **centroid→GT offset**: `|centroid(chosen_box) − gt|`. Report per clip (t1–t10 AND
additional_evidence): distribution (median/p90), and crucially the offset **as a function of box size-ratio**
(box area / median box area) — the hypothesis is the offset grows with oversized/merged boxes (t5, a03, t3).
Also report: of the *misses* where the right box was chosen, what fraction would flip to a hit if the point
were the **true shape center** instead of the centroid (an oracle-localization ceiling). If that recoverable
fraction is small (<~5–10% of misses) the lever is weak — say so and stop. If it's material (the size-ratio
trend is real and recovers a chunk of t5/a03), proceed to Phase 2.

Candidate point estimators to compare against the raw centroid (all causal, computed *within* the chosen box):
- **saliency/outlier-peak center** inside the box (`motion.saliency_map` / `MotionField.outlier_vectors`) —
  the moving sub-cluster, which on a merged box should sit on the real shape, not the box middle;
- **detection-weighted center** (if multiple raw boxes overlap, confidence-weighted centroid);
- **template/appearance center** only if it beats the above (appearance is otherwise dead — don't lean on it).

## Phase 2 — wire the chosen estimator into the emitted point (only if Phase 1 passes)

The point is emitted as `pos` → `TrackPoint(p.idx, pos[0], pos[1], ...)` throughout `identity.py` (e.g. the
`fpath` family decode appends at the `track.append(TrackPoint(...))` lines; `_centroid(chosen)` feeds `pos`).
Add a **localization refinement step** that maps `chosen_box → refined_point` right before the CV predictor /
emission, gated so it only moves the point when confident (e.g. only when box size-ratio > threshold, so
well-sized boxes keep their reliable centroid). Keep it a **post-identity, pre-emission** transform — identity
state and the decode-layer freezes (`fpath_hedge`/`fpath_freeze`) are untouched; this only changes *where in
the chosen box* we point. LOO-tune any threshold via a `*_probe.py` block (clone `hedge_probe.py`'s scaffold;
channels precomputed once per clip). **Accept only on LOO mean ↑, no per-clip regression**, on BOTH boards.

## Guardrails

- **Strictly causal, GT is GT-only.** The crosshair is read for scoring then `strip_pointer`-inpainted before
  any tracking — never feed GT into the estimate.
- **Don't reopen identity ranking.** The "which box" wall is triple-confirmed (CLAUDE.md). This plan is about
  the **point inside the chosen box**, a localization lever, not a new identity channel. Do not re-propose
  appearance/rotation/sub-box-*ranking*/detection-knob identity ideas.
- **Validate on both clip sets.** t1–t10 (`LEADERBOARD.md`) is the board of record; additional_evidence
  (valid-13, a03/a07 GT is broken — exclude) is the held-out check. A real point-win should help both,
  especially the oversized-box laggards (t5, a03, t3).
- **`eval_modes` overwrites `LEADERBOARD.md`** — back it up / restore canonical as in the last session
  (`LEADERBOARD_t1_t10.md` is the backup; `LEADERBOARD_additional_evidence.md` is the held-out board).

## Key files / anchors

- `ld/detect/render_evidence.py` L76–96 — overlay draw (Phase 0: filled red dot).
- `ld/detect/identity.py` — `_centroid` (L121); `TrackPoint(...)` emission sites (the `pos` that becomes x,y);
  `track_fused_path_identity` is the `fpath` family decode. Refinement goes box→point just before emission.
- `ld/vision/motion.py` — `saliency_map`, `MotionField.outlier_vectors` (candidate in-box localizers).
- `ld/detect/eval_modes.py` / `LEADERBOARD*.md` — scoring + boards. `hedge_probe.py` — LOO-probe scaffold.
- `make_additional_evidence.py` (repo root) — built the held-out set; rerun if more raw captures are added.
