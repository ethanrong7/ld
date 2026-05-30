I'd like to undertake a computer vision task. Id like you to have a look at some of my data, specifically t1_cropped_trimmed.mp4. you can also have a look at t2, t3, but logic is the same, but the shape maybe slightly different.

Here is the task explained in simple terms.
1 - At the beginning of the video, a countdown appears, it can start from 5 or 2
2 - during the countdown, a white shape in the center will rotate at a constant rate
3 - You may also notice some shapes in the background. The shapes in the background will possess exactly the same outline as the white shape in the center, these shapes however do not rotate but the entire page will rotate. Think of drawing a bunch of shapes on a piece of paper. afterwards you may move the paper around in either x, y or combined direction (but not rotate), this is why the background shapes direction do not change, but their position changes. it is also important to note that their positions relative to each other background shape are fixed, as you would expect using the piece of paper analogy
4 - after the countdown, the white shape will become transparent leaving only its border. the transparent white shape now moves around the screen but not with the underlying background. You can imagine as if it were on its own piece of paper moving at its own x,y position, but its also important to remember that the transparent white shape can rotate, but the other background shapes in (3) do not rotate
5 - you may also notice a green cursor hovering over the transparent white shape throughout the video after the countdown ends, this dictates what the answer is, but it is very important you do not use the green cursor as an indicator for guiding you to the solution, it serves as an answer, not a part of the problem, future samples will not have this green cursor.

Your job is to basically hover over the correct shape (as the green cursor is currently doing)

I dont want you to code anything yet, i want you to watch a couple of videos, and comprehent my points, and make some observations. we will try to tackle this CV project. 

---

# LD Solver — Implementation Plan (LLM-resumable)

This section is written for an AI coding agent. Each step is bite-sized and
self-contained. To resume: read the "Status" line, open the files named in the
current step, and continue. Update the Status line + step checkboxes as you go.

**Status: Steps 0–8 done. Next: Step 9 (eval harness) + Step 10 (CLI).**
Tracker result (median / p90 err vs GT): t1 13.5 / 24.8 px, t2 11.2 / 19.1 px,
t3 10.9 / 19.2 px. Stays locked on the correct shape on all three (heart, circle,
star). Center is biased ~11–13 px (well inside the shape; functionally a correct
hover). FINDING (Step 8): a persistent motion-compensated background MODEL
regressed badly (model drift → median ~160 px). The chosen, better approach is
**pairwise** stabilized differencing (`residual_map`) + gated weighted centroid +
constant-velocity/-ω motion model. See Step 8 for details.

## Context an agent must load first
- Game: MapleStory "Lie Detector". The whole screen is a tan **relief/emboss**
  texture tiled from many copies of ONE shape outline. Per round the shape
  differs (t1=heart/shield, t2=circle, t3=star), so code must be **shape-agnostic**.
- Clips: `data/t*_cropped_trimmed.mp4`, **744x498, 60 fps, ~10–13 s**, one round each.
- **Two phases.** (A) Countdown: target is a SOLID near-white filled shape,
  centered, rotating at constant rate; a blue countdown digit overlaps it.
  (B) Post-countdown: white fill vanishes → target is only relief outline,
  camouflaged among identical background shapes.
- **Core insight:** the background is a rigid sheet that only TRANSLATES
  (no rotation); the target moves on its own x/y AND rotates. So the target is
  found by **motion**, not appearance: estimate global background translation,
  compensate it, and the leftover **motion residual** is the target.
- **Backbone = background stabilization (done PAIRWISE).** Between consecutive
  frames we warp the previous frame by the global background shift, which freezes
  the background; the leftover difference is the independently-moving target. This
  is exactly `motion.residual_map`. So ALL background shapes are held still
  frame-to-frame and the target is the only mover — true for every shape
  regardless of rotation. We do NOT enumerate the n−1 background shapes; we model
  their shared rigid motion and take the outlier. Identity is anchored by the
  countdown seed (we know which blob is the target at handoff), so it's tracking,
  not per-frame re-identification.
  NOTE: a *persistent* warped background model (accumulating shifts over many
  frames) was tried and regressed due to model drift; pairwise stabilization is
  both simpler and far more accurate here. Don't reintroduce the persistent model.
- **Symmetry matters for the rotation cue.** Asymmetric shapes (heart, star) also
  rotate visibly → rotation is a strong *bonus* discriminator and lets us reject
  non-rotating neighbours. A circle (t2) is rotationally symmetric → rotation is
  invisible AND all background circles look identical, so there motion is the
  ONLY cue. Worst case: symmetric target momentarily moving at background velocity
  → no instantaneous cue → coast on the constant-velocity model until relative
  motion resumes.
- **Green cursor** = ground-truth answer marker. Use it ONLY to score accuracy
  on these labelled clips; NEVER as a solver input (real samples lack it).
- Verified numbers (t1): handoff ≈ frame **117** (~1.9 s); rotation ≈
  **−1.7°/frame** (constant); residual localization median error **4.8 px**.

## Conventions
- Env: `.venv` (Windows). Run modules as `.venv\Scripts\python.exe -m ld.<module>`.
- PowerShell mangles inline `python -c "..."`; prefer running module files.
- All generated artifacts go to `output/` (gitignored). Videos in `data/` are local-only.
- Package layout: `ld/{capture,vision,control,debug}`.

## Steps

- [x] **Step 0 — Validate data.** Probe every `data/t*_cropped_trimmed.mp4`
  (resolution, fps, frame count). Extract sample frames to eyeball the game.
  Done: all clips 744x498@60fps.

- [x] **Step 1 — Package scaffold.** Create `ld/config.py` (thresholds/paths),
  `ld/capture/video_source.py` (`VideoSource`, `open_writer`), package
  `__init__.py` files. Acceptance: `from ld.capture.video_source import VideoSource` imports.

- [x] **Step 2 — Green-cursor GT detector.** `ld/vision/cursor.py:find_cursor(frame)`
  → `(x,y)` of the green crosshair or `None`, via HSV mask (config `GREEN_*`).
  Acceptance: returns a stable point across a clip; used only for scoring.

- [x] **Step 3 — White-shape segmentation + handoff.** `ld/vision/countdown.py`:
  `white_mask`, `detect_white_shape(frame)->WhiteShape(cx,cy,area,angle,mask,bbox)`.
  Threshold high-V/low-S to isolate the white fill; exclude the blue digit.
  Acceptance: clean ~10k-px blob during countdown; "handoff" = last frame of the
  opening white-present run.

- [x] **Step 4 — Diagnostic logger.** `ld/debug/analyze.py` writes per-frame CSV
  (white presence/area/centroid/angle + cursor GT) and an annotated video; prints
  handoff frame. Acceptance: `output/t1_diag.csv` shows handoff≈117, monotonic angle.

- [x] **Step 5 — Global background motion + residual.** `ld/vision/motion.py`:
  `estimate_translation` (cv2.phaseCorrelate, Hanning window) → background `(dx,dy)`;
  `residual_map` warps prev by `(dx,dy)`, abs-diffs, blurs, masks borders.
  `ld/debug/residual_diag.py` overlays a residual heatmap + GT, scores peak error.
  Acceptance: residual concentrates on target; median error ≈ 4.8 px on t1.

- [x] **Step 6 — Capture target template + rotation rate.**
  New module `ld/vision/template.py`. During the clean countdown window (skip
  frames where white area spikes from digit glow), collect the white blob masks.
  (a) Pick the most central, stable blob (reject the digit which sits above center).
  (b) Build a canonical, centered, binary **template** of the shape (largest clean
  mask, cropped + centered, with measured radius).
  (c) Fit rotation rate ω: unwrap `WhiteShape.angle` over the clean window, robust
  linear fit (deg/frame). Handle symmetry: heart=mod180, star=mod72, circle=undefined.
  Output a `RoundInit{template, center_seed, omega_deg_per_frame, handoff_frame}`.
  Acceptance: on t1, ω≈−1.7°/frame and a recognizable shield template is produced.

- [x] **Step 7 — Seeded target tracker.** New `ld/vision/tracker.py`. Initialize
  position at the handoff seed (last good white centroid). Each frame: predict with
  a constant-velocity (+ constant-ω) model; build residual map; take the residual
  measurement **near the prediction** (gated search window) instead of global argmax
  — this removes the "arc centroid drift" outliers seen in Step 5. Update a simple
  Kalman/EMA state. Output `(x,y,theta)` per frame. Acceptance: smooth track,
  median error ≤ Step 5, p90 dramatically lower (no 60px arc outliers).

- [x] **Step 8 — Stabilization backbone + center estimator (DONE, with finding).**
  Confirmed the tracker's measurement is built on **pairwise** background
  stabilization (`residual_map`), which already freezes the background each frame.
  Tried and REJECTED a persistent running background model
  (`BackgroundStabilizer`, since deleted): it drifted and median error blew up to
  ~160 px. Also benchmarked center estimators within the gate:
  weighted-centroid = 13.5 px (best), unweighted = 19.5, morphological ring-fill =
  28. Kept the **gated weighted centroid**. Residual ~11–13 px center bias remains
  (the bg-compensated 2-frame diff lights up both the old and new target outline,
  so the centroid sits between them); it is well inside the shape, so the hover is
  correct. Optional future refinement (Step 8b, not required for the task): a
  rotating-template lock for ASYMMETRIC shapes to snap exactly onto centre and
  explicitly reject non-rotating neighbours; skip it for the symmetric circle.

- [ ] **Step 9 — Evaluation harness.** `ld/debug/eval.py` runs the full pipeline
  on all `data/t*_cropped_trimmed.mp4`, scores per-frame distance to green-cursor
  GT, prints mean/median/p90 and a pass/fail (e.g. median ≤ 10 px). Writes a
  summary CSV + per-clip annotated videos. Acceptance: table across t1..t10.

- [ ] **Step 10 — CLI integration.** Fill `ld/main.py` with subcommands:
  `probe`, `analyze`, `residual`, `track`, `eval`. Thin wrappers over the debug
  modules. Acceptance: `python -m ld.main track --input data/t1_...mp4` works.

- [ ] **Step 11 — Robustness / generalization.** Test on clips with the cursor
  removed (`output/t*_nocursor.mp4` if present) and confirm the solver never reads
  the cursor. Stress different shapes/resolutions; tune gates. Add fallbacks for
  failed handoff detection (e.g. countdown trimmed). Acceptance: works without GT.

- [ ] **Step 12 — Live mode (optional).** `ld/capture/screen.py` (screen grab) +
  `ld/control/mouse.py` (move cursor to target). Wire into `main.py live`.
  Acceptance: drives the real game; out of scope until offline accuracy is solid.

## Key risks / notes for the implementer
- **Velocity ambiguity:** when target motion ≈ background motion, residual fades.
  Mitigated via the constant-velocity motion model (coast); the tracker reports
  `measured=False` on those frames (only a handful per clip in practice).
- **Robust background estimate:** the target biases phaseCorrelate slightly; it's
  a minority of pixels so it's usually fine, but consider masking the predicted
  target region before estimating background shift for extra robustness.
- **Digit contamination:** the blue countdown digit's white glow can merge with
  the shape blob (area spikes). Always prefer the central blob; drop spike frames.
- **Shape symmetry** changes how θ is interpreted/unwrapped — don't assume mod180.
