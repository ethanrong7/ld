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

**Status: Steps 0–10 + 11a done. Next: Step 11b (template match in tracker).**
See **“Resume tomorrow — summary & roadmap”** below for findings, commands, and phases A→C.
CLI: `python -m ld.main {probe|template|analyze|residual|track|eval}`.
**Honest eval** (`python -m ld.main eval`, green GT inpainted before tracking):
**0/10 PASS**, median-of-medians **56.5 px** (pass = median ≤ 25 px). Summary
at `output/eval_summary.csv`. The earlier **10/10 PASS @ 13.6 px** was inflated:
the solver never read green as an input, but the moving green crosshair acted as
an implicit motion-residual beacon (ablation: t9 median 14.6 px → 51.8 px when
stripped; t1 13.5 → 255). Default pipeline now calls `ld.vision.cursor.strip_pointer`
on every frame before grayscale; use `--no-strip-pointer` only for ablation.
FINDING (Step 8): persistent background model regressed (~160 px); **pairwise**
`residual_map` + gated weighted centroid is kept but **not sufficient alone** once
the cursor beacon is removed. FINDING (real capture): `ld1440p1` (no GT cursor)
still drifts — needs template/appearance lock + live mouse-disk stripping at
known screen coords (`track_video(..., mouse_at=...)`).

## Resume tomorrow — summary & roadmap

Read this section first when continuing. Steps 0–10 and **11a** are done; **11b**
is the active work item.

### What we learned (ground truth)

| Finding | Implication |
|--------|-------------|
| **Honest eval** (default `strip_pointer`): **0/10 PASS**, median-of-medians **56.5 px** | Motion residual + gated centroid is **not** shippable alone |
| Old **10/10 @ 13.6 px** with `--no-strip-pointer` | Inflated: green cursor was never a code input but acted as a **motion-residual beacon** |
| **Runtime:** all normal runs **do** inpaint green before grayscale | `python -m ld.main track/eval` = honest; GT red dot in MP4 is scored on **raw** frames only |
| **Tracker does not use rotation or template for position** | `omega`/`theta` are updated in state but `_gated_measurement` is **xy residual only**; `RoundInit.template` is unused in `tracker.py` |
| **t1,t3,t4,… ~50–60 px** — visually “ballpark” | Asymmetric shapes **rotate** → stronger residual on the real target vs static embossed decoys (indirect cue, not template match) |
| **t2 circle ~319 px** — wrong | No visible rotation; all decoys identical; late handoff (frame ~379); residual can’t pick the right blob |
| **`ld1440p1`** | Panel crop works; track still drifts (weak page motion + no appearance lock) |
| **Live mouse** | Same risk as green GT: moving **your** cursor paints the heatmap → strip at **known** `(x,y)` via `track_video(..., mouse_at=...)` |

### How to run (tomorrow)

```bash
set PYTHONPATH=.
.venv\Scripts\python.exe -m ld.main eval                    # honest metrics → output/eval_summary.csv
.venv\Scripts\python.exe -m ld.main track --input data/t1_cropped_trimmed.mp4 --out-video output/t1_track.mp4
.venv\Scripts\python.exe -m ld.main eval --no-strip-pointer # ablation only (misleading “pass”)
```

Latest honest **t1–t4** medians (px): t1 **59.2**, t2 **319.1**, t3 **50.8**, t4 **53.1**. Annotated MP4s:
`output/t1_track.mp4` … `output/t4_track.mp4` (green box = prediction, red dot = GT).

Key code: `ld/vision/cursor.py` (`strip_pointer`), `ld/solver.py` (`track_video`),
`ld/vision/tracker.py` (measurement), `ld/vision/template.py` (countdown template + ω, not yet used in track).

### Proposed path forward

**Phase A — Fix measurement (do this next, highest ROI)**  
Goal: honest eval median ≤ 25 px on most t* clips; separate strategy for circle.

1. **Template match inside the gate** (`tracker.py`, Step 11b)  
   - Use countdown binary template from `RoundInit`.  
   - **Asymmetric** (heart, star, spade): search `(x,y)` + small θ window around prediction; seed ω from countdown.  
   - **Symmetric** (circle): **no θ search** — translation-only match (NCC / edge on relief).  
   - Detect symmetry from template shape or unreliable ω / angle variance during countdown.  
2. **Fuse with residual** — template = primary; gated residual centroid = fallback when template score is weak. Circles: favor template + velocity.  
3. **Small fixes** — mask predicted target disk before `phaseCorrelate`; don’t trust ω on symmetric countdown blobs; t2 late-handoff edge case.

**Phase B — Real capture**  
Promote `output/_crop_ld.py` logic into package; run on `data/ld1440p1.mp4` (9.4–16 s); same pointer strip (no green there).

**Phase C — Live** (after offline is good enough)  
Screen grab + `mouse_at` → `strip_pointer` + move mouse to predicted `(x,y)`.

### Do not do next

- Trust eval without pointer stripping.  
- Reintroduce persistent background model (regressed ~160 px).  
- Rotation search on circles.  
- Expect residual-only tuning to go from ~56 px → ≤25 px.

### Immediate next task

Implement **Phase A.1 + A.2** in `ld/vision/tracker.py`, re-run `python -m ld.main eval`, regenerate track MP4s if improved.

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
- **Green cursor** = ground-truth answer marker on t* clips. Use ONLY to score
  accuracy (`find_cursor` in debug callbacks on raw frames). NEVER as a tracking
  input. Before grayscale the pipeline **inpaints** green (and, in live mode, a
  disk at the commanded mouse position) via `strip_pointer` so pointer motion
  cannot create a self-fulfilling residual heatmap.
- **Cursor confound (critical):** even a “cursor-blind” grayscale solver was
  helped ~4× by the green marker’s motion in the residual — same risk when *you*
  move the mouse over the target in live play. Stripping is mandatory on all paths.
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

- [x] **Step 9 — Evaluation harness.** `ld/debug/eval.py` runs `track_video` on all
  `data/t*_cropped_trimmed.mp4`, scores distance to green-cursor GT (raw frame),
  pass/fail median ≤ 25 px, writes `output/eval_summary.csv`.
  **With pointer stripping (default): 0/10 PASS, median-of-medians 56.5 px.**
  **Without stripping (`--no-strip-pointer`): ~10/10 PASS @ ~13.6 px** — not a
  valid production metric (implicit cursor beacon). GT is only read in debug
  callbacks; measurement frames are always preprocessed by `strip_pointer`.

- [x] **Step 10 — CLI integration.** `ld/main.py` exposes subcommands
  `probe`, `template`, `analyze`, `residual`, `track`, `eval` (thin wrappers over
  the modules; each module also exposes a `run(...)` + `main()`). Verified:
  `python -m ld.main track --input data/t4_...mp4` and `probe`/`template` work.

- [x] **Step 11a — Pointer stripping (all pathways).** `ld/vision/cursor.py`:
  `strip_pointer(frame, mouse_xy=..., strip_green=True)` — HSV green inpaint +
  optional live-mouse disk. Called from `track_video`, `analyze_round`,
  `residual_diag` by default. `track_video(..., mouse_at=idx->(x,y))` for live.
  CLI: `--no-strip-pointer` on `template`/`residual`/`track`/`eval` for ablation only.
  Acceptance: default runs never feed pointer pixels into grayscale/residual.

- [ ] **Step 11b — Template / appearance lock.** Motion residual alone fails once
  the cursor beacon is removed (see eval above) and on real captures (`ld1440p1`).
  Add template correlation in the tracker gate as the **primary** measurement
  (rotation search for asymmetric shapes only; **circles are symmetric** — match
  on position/outline without θ, motion-only residual as fallback). Acceptance:
  honest eval median ≤ 25 px on t* clips; stable track on nocursor / ld1440p1.

- [ ] **Step 12 — Live mode (optional).** `ld/capture/screen.py` + `ld/control/mouse.py`
  + `main.py live`. Pass last commanded mouse `(x,y)` into `strip_pointer` each frame
  so your own cursor does not steer the heatmap. Acceptance: drives the real game.

## Key risks / notes for the implementer
- **Pointer / cursor coupling:** any visible cursor (green GT or your mouse) creates
  high-contrast motion in the residual at the pointer location. Always strip before
  `to_gray_f32`. In live mode strip at **known** mouse coords, not only HSV green.
- **Do not trust pre-stripping eval numbers** for product readiness; the ~13.6 px
  headline was with the beacon present. Current honest baseline is ~56 px median.
- **Velocity ambiguity:** when target motion ≈ background motion, residual fades.
  Mitigated via coasting (`measured=False`); worse on real captures with long
  stretches of near-zero page motion (e.g. `ld1440p1`).
- **Robust background estimate:** the target biases phaseCorrelate slightly; it's
  a minority of pixels so it's usually fine, but consider masking the predicted
  target region before estimating background shift for extra robustness.
- **Digit contamination:** the blue countdown digit's white glow can merge with
  the shape blob (area spikes). Always prefer the central blob; drop spike frames.
- **Shape symmetry** changes how θ is interpreted/unwrapped — don't assume mod180.
