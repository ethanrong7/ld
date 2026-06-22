# CLAUDE.md — LD (Lie Detector) real-vs-fake shape tracker

## What this project does

MapleStory's "Lie Detector" minigame shows a sheet covered in many **identical** shapes. One shape is **real** (the player must click it); the rest are **fakes**. Fakes move rigidly with the sheet; the **real shape moves independently**. This repo takes a screen-capture clip and outputs the pixel position of the real shape every frame.

**Online / live constraint.** The solver sees frames one at a time and must emit a position without future frames. A bounded fixed-lag buffer (~10–15 frames, ~0.5s) is permissible — the shape physically cannot leave its radius in that window.

## Setup

### 1. Activate virtual environment

**Mac/Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```
> If you get an execution policy error: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 2. Install dependencies

**Mac/Linux:**
```bash
.venv/bin/python -m pip install -r requirements.txt
```

**Windows:**
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Train the YOLO model

The best model is `yolov8n_single_combined` (single-class, oracle 0.958). Weights are not committed to the repo — you must train locally. Takes ~12 min on M3 Pro, longer on Windows/CPU.

The single-class dataset is built from `data/detect/s_frames` + `data/detect/s_labels_single`
(canonical paths in `ld/config.py` as `TRAIN_*`).

**Mac/Linux:**
```bash
.venv/bin/python -m ld.detect.build_dataset      # -> data/detect/dataset_single_combined
.venv/bin/python -m ld.detect.train --name yolov8n_single_combined
```

**Windows:**
```powershell
.venv\Scripts\python.exe -m ld.detect.build_dataset
.venv\Scripts\python.exe -m ld.detect.train --name yolov8n_single_combined
```

Weights will be saved to `data/detect/runs/yolov8n_single_combined/weights/best.pt`.
To add new training data, just drop a video in `data/` and run `python -m ld.detect.annotate`
(it discovers, crops, extracts 5 in-play frames, labels, rebuilds, and prints the train command).

### 4. Generate the evidence video (leader)

```bash
# Mac/Linux
.venv/bin/python -m ld.detect.render_evidence \
    --weights data/detect/runs/yolov8n_single_combined/weights/best.pt --mode fpath_human
```
```powershell
# Windows
.venv\Scripts\python.exe -m ld.detect.render_evidence --weights data/detect/runs/yolov8n_single_combined/weights/best.pt --mode fpath_human
```

Output: `data/detect/evidence/<clip>_<mode>.mp4` (any mode in `identity.ALL_MODES` is renderable
via `--mode`). The 2026-06-22 cleanup removed the field/paper/accum/chain families and the old
OutlierTracker solver; the `fpath` lineage is all that remains.

---

## Pipeline (two separable stages)

1. **Detection** — YOLOv8n (`data/detect/runs/yolov8n_single_combined/weights/best.pt`) finds candidate shape boxes per frame. Oracle within_r ≈ **0.958** (mean over t1–t10). Detection is effectively solved. The real shape is camouflaged and ranks ~10–13 in confidence; you cannot filter by confidence without losing it.

2. **Identity** — decides *which* box is the real shape over time. This is the bottleneck and the entire focus of remaining work. Code in `ld/detect/identity.py`.

## Current best: `fpath_human` (0.940 within_r) — box engine `fpath_freeze` (0.932)

**The shipped leader is `fpath_human`** (0.940; `fpath_freeze`'s box decisions + a human-cursor
1-Euro/deadband output filter — see the mode stack below). This section explains `fpath_freeze`,
the box-decision engine underneath it (byte-identical picks). `fpath_freeze` was the prior leader
(0.932, verified full-board `eval_modes` + honest LOO, 2026-06-20, against `yolov8n_single_combined`). It is `fpath_hedge` **plus a residual-gated decode-layer freeze** that runs BEFORE the churn hedge. The whole win comes from answering a **1-box BINARY** question — *"is the box I am currently holding a rigid fake?"* — which is sharp where the 15-box *ranking* (`fpath_resid` override, EXP-Q2b) was hopeless. The detector: the chosen box's **cumulative N=30 sheet-frame residual** (its integrated drift relative to the rigid sheet, via the affine back-walk). A fake reads ~9–15px (it moves with the sheet → residual is just detector jitter, a clip-/radius-independent floor); the real shape reads ~45–91px (its accumulated independent drift). When the held box's residual falls below **τ=15** for **1** frame, we have locked onto a fake → FREEZE the output toward a **lagged pre-creep anchor** (the output **6** frames ago, before the creep started) and hold until the residual recovers. It works because the real shape barely moves (median 1.3 px/fr), so an onset-anchored freeze stays within radius for 20+ frame runs. Identity/trellis state is byte-identical to `fpath_hyst`/`fpath_hedge` (decode-layer only). Oracle mean is **0.958** — and `fpath_freeze` (0.932) now sits within ~0.026 of it.

| clip | within_r | oracle | Δ vs fpath_hedge |
|------|----------|--------|------------------|
| t1 | 0.895 | 0.952 | +0.042 |
| t2 | 0.978 | 0.978 | — |
| t3 | 0.913 | 0.918 | +0.037 |
| t4 | 0.880 | 0.944 | +0.024 |
| t5 | 0.858 | 0.948 | **+0.049** |
| t6 | 0.959 | 0.970 | +0.022 |
| t7 | 1.000 | 0.998 | +0.007 |
| t8 | 0.909 | 0.939 | **+0.133** |
| t9 | 0.958 | 0.973 | +0.022 |
| t10 | 0.974 | 0.964 | — |
| **mean** | **0.932** | **0.958** | **+0.033** |

**This is the largest single lift in the lineage (+0.033 full-board), and it raised the LAGGARDS rather than sidestepping them** — t8 +0.133, t1 +0.042, t5 +0.049 — while every clip is flat-or-up (no per-clip regression). t7 even reaches 1.000 (>oracle 0.998: the freeze holds a position within radius on a frame where the nearest detected box centroid was just outside — a decode-layer output can beat the per-frame oracle box). Honest LOO over a *physically-capped* grid (`τ∈{10,12,15,18} × lag∈{6} × consec∈{1,2}`, `resid_freeze_probe.py`): **0.9359, worst-clip +0.000, all 10 folds independently select τ=15/lag=6/consec=1** (the unique no-regression corner). The grid is capped at the fake-noise floor on PHYSICAL grounds — a higher τ cuts into the real-shape residual distribution and regresses the low-residual strong clips (t9/t10), so the LOO cannot pick an unphysical threshold (the optimism trap of the wider τ grid). Live full-board (0.932) lands a hair under the LOO (0.9359), mostly on t1.

**Why this succeeded where everything else hit the wall.** Prior work fought the *identity* question ("which of the ~12 boxes is real?") where the signal is genuinely ambiguous (the triple-confirmed information limit). `fpath_freeze` sidesteps it: it never re-identifies the real box, it just detects that the box it is ON has gone rigid (a fake) and **stops chasing** — exploiting the physics (real shape slow, fakes rigid) at the OUTPUT layer, with a signal that only becomes visible by *integrating* over 30 frames. The over-the-shoulder lineage: EXP-Q1 found the integrated residual is real signal (t8 MISS top1 0.32→0.68); EXP-Q2b proved it's too unsharp to RANK boxes (override regresses every clip — dead); EXP-Q3 then used it as a 1-box *binary* and shipped.

Remaining laggards: **t5 (0.858), t4 (0.880), t1 (0.895)** — the freeze cleared most of t8's long fake-rides (0.776→0.909). What's left is shorter locks + t3/t5's *undersized-box* runs (a detection-quality issue, EXP-2) and t4's residual misses. The prior leader `fpath_hedge` (0.899) was `fpath_hyst` + a churn-gated freeze-blend (caught the *swept* incoherent hops); `fpath_hyst` (0.878) was `fpath_fuse` + EMA-coherent-mass hysteresis.

**Failure taxonomy (fuse_probe, 2026-06-19).** On the frames the trellis misses, which channel ranks the real box #1 splits by clip — no single channel wins everywhere (the persistent causal-key wall). This is what motivated the additive emission and what Viterbi continuity resolves:
- **t4 (coh #1 0.60), t5 (curl #1 0.50)** — *emission-channel* failures. **Fixed by `fpath_fuse`.**
- **t1 (mass #1 0.41, coh demotes it), t8 (mass #1 0.32)** — *path lock-in* failures: need a lock-in escape, not a new channel. **t1 partially addressed by `fpath_hyst`** (EMA-coherent-mass hysteresis, +0.015). t8 barely moves (+0.004): its coherence channel is signal-limited (on-GT <0.30), so no override can reliably find the real box there.
- **Max-fusion of channels is a dead end** (fuse_probe top-1 0.22): max-combining amplifies whichever channel spikes on a fake. The win came from ADDITIVE weighted sums, never max.
- Coherence (windowed `_box_coherent_mass`, not the instantaneous `box_coherence` `fpath_coh` used) is the dominant channel; curl is a small complement that removes the t2 regression coherent-mass alone would cause.

## Held-out validation: `additional_evidence` (15 clips, 2026-06-20)

A second, never-trained-on clip set was built from the raw captures in `data/` (the 12 `MapleStory - ….mp4`
files + `ld1080p1/p2` + `ld1440p1`) by `make_additional_evidence.py` (repo root, standalone). It crops each
raw capture to just the lie-detector board (largest tan rectangle, aspect ~1.49; auto-handles
720p/768p/1080p/1440p → resize to **744×498**) and trims to the minigame (countdown board appears → board
removed at success), producing `data/additional_evidence/a01…a15_<source>.mp4` in the **exact s/t format**
(744×498, 60fps, no audio, GT crosshair retained). GT is the in-game green crosshair (auto-derived, noisier
than hand labels), stripped by `strip_pointer` before detection like the s/t clips.

**Result — `fpath_freeze` leads here too, and the whole ranking reproduces t1–t10.** Scored over the 13
valid clips (a03 + a07 excluded: incomplete/static GT). Saved to `ld/detect/LEADERBOARD_additional_evidence.md`
(canonical t1–t10 board is `LEADERBOARD.md`).

| mode | add-evidence (valid-13) | t1–t10 |
|------|------------------------:|-------:|
| `fpath_human` | **0.859** | 0.940 |
| `fpath_freeze` | 0.837 | 0.932 |
| `fpath_fuse` | 0.814 | 0.876 |
| `fpath_hedge` | 0.811 | 0.899 |
| `fpath_hyst` / `fpath_coh` | 0.800 | 0.878 / 0.825 |
| `fpath` | 0.780 | 0.797 |
| `field_coh` / `field` / `field_lag` | 0.741 / 0.740 / 0.730 | 0.773 / 0.744 / 0.749 |

Absolute numbers are lower (noisier auto-GT, harder/odd clips like the 1440p-downscaled a03 whose oracle is
0.942 but identity ~0.64), but the **fpath-family > field-family ordering and `fpath_freeze` on top hold
exactly** — independent cross-dataset confirmation of the lineage. **No method falls below 0.70 on valid GT**
(weakest `field_lag` 0.730), so the field family is dominated-but-not-prunable. Evidence renders:
`data/detect/evidence/aNN_*_fpath_freeze.mp4`.

## Identity mode stack

The leaderboard default (`identity.BOARD_MODES`) is now **`fpath` + `fpath_human` + an oracle ceiling row**. `ALL_MODES` keeps the full `fpath` lineage (`fpath`, `fpath_coh`, `fpath_fuse`, `fpath_hyst`, `fpath_hedge`, `fpath_freeze`, `fpath_human`) runnable by name (`eval_modes --modes <name>`) so the tuning probes can regenerate their per-mode CSVs. The field/paper/accum/chain/`fpath_reacq` families were removed in the 2026-06-22 cleanup (the table below documents them as history; the rows other than `fpath`/`fpath_human` are no longer dispatchable).

| Mode | within_r | Description |
|------|----------|-------------|
| `fpath_human` | **0.940** | `fpath_freeze` + human-cursor output dynamics (1-Euro + 2px deadband on the emitted point) — current leader; box decisions IDENTICAL to `fpath_freeze` |
| `fpath_freeze` | 0.932 | `fpath_hedge` + residual-gated decode freeze (chosen-box N=30 residual < τ ⇒ on a fake ⇒ freeze to lagged anchor) — the identity/metric reference (byte-identical box picks) |
| `fpath_hedge` | 0.899 | `fpath_hyst` + decode-layer churn-gated freeze-blend |
| `fpath_hyst` | 0.878 | `fpath_fuse` + EMA-coherent-mass hysteresis override |
| `fpath_fuse` | 0.876 | `fpath` + additive `mass + 1.5*coherent_mass + 0.5*curl` emission |
| `fpath_coh` | 0.825 | `fpath` + coherence-bumped emission (`mass*(1+1.8*coh)`) |
| `fpath` | 0.797 | Viterbi path integrator, pure saliency-mass emission (ablation base) |
| `field_coh` | 0.773 | `field_lag` + coherence far-jump override |
| `field_lag` | 0.749 | Fixed-lag confirmation smoother (K=8) over `field` |
| `field` | 0.744 | Motion-saliency peak tracker + YOLO snap |

**fpath / fpath_coh / fpath_fuse / fpath_hyst / fpath_hedge / fpath_freeze** per frame: a causal forward trellis over YOLO boxes; emission = a weighted sum of normalized motion-evidence channels, transition penalty = `(jump_in_radii)^2`. Decode = argmax cumulative score → that box's centroid feeds a CV predictor. `fpath`=mass only; `fpath_coh`=mass×coherence-bump; `fpath_fuse`=additive mass + windowed coherent-mass + curl (each cross-frame-normalized so weak frames stay low-emission and the transition prior carries — the load-bearing coast trick). `fpath_hyst`=`fpath_fuse` + a distance-agnostic EMA-coherent-mass hysteresis switch that escapes adjacent creep. `fpath_hedge`=`fpath_hyst` + a **decode-layer churn-gated freeze-blend** on the OUTPUT (identity state untouched): when the chosen box's recent sheet-removed motion is large but directionally incoherent (a box-hop), freeze the output toward its last value; when coherent (real-shape burst) or small (stable), commit — catches the *swept* locks. `fpath_freeze`=`fpath_hedge` + a **residual-gated freeze** running BEFORE the hedge: maintain the chosen box's cumulative N=30 sheet-frame residual (the affine back-walk in `identity._cumulative_residual`); when it collapses below τ=15px (a 1-box "am I on a rigid fake?" test) freeze the output toward the output-from-`lag`-frames-ago and hold until it recovers — catches the *coherent* identity-locks the churn gate is blind to. Together the two freezes cover both miss modes. `fpath_human`=`fpath_freeze` + a **human-cursor output-dynamics filter** applied as the LAST transform on the emitted point (`ld/track/humanize.py`, `HumanCursor`): a strictly-causal **1-Euro filter** (speed-adaptive low-pass — heavy smoothing at rest kills the lock-wobble, light smoothing on bursts avoids lag) + a **2px deadband** (don't chase sub-pixel detector noise). Box decisions are BYTE-IDENTICAL to `fpath_freeze` — it only reshapes *how the point moves between them* (jitter ↓, freeze-snaps eased into glides). It is the first decode-layer transform that improved BOTH smoothness AND `within_r` (the smoothing kills jitter-induced single-frame pop-outs): **+0.008 t1–t10 (0.932→0.940), +0.022 add-board (0.837→0.859), flat-or-up on every clip, RMS jerk −61%/−65%, velocity-reversals −97%/−98%**. Tune fuse weights via `ld/detect/fuse_sweep.py`, the hysteresis via `ld/detect/exp3_sweep.py`, the hedge's `churn_hi` via `ld/detect/hedge_probe.py`, the freeze's `τ/lag/consec` via `ld/detect/resid_freeze_probe.py`, the human-cursor `min_cutoff/beta/deadband/v_max/lag` via `ld/detect/cursor_physics_probe.py` (all LOO, channels/affines/residuals precomputed once per clip). **Reusable insights:** (1) accumulate evidence on moving boxes with a *nearest-centroid-carried EMA*, not a fixed window (EXP-3: EMA on-GT 0.51 vs fixed-window 0.39 on t1); (2) when you can't name the box, hedge the *position* — gate the hedge on trajectory **coherence** (`(1−R)`), not magnitude; (3) a signal too weak to RANK boxes can still be sharp as a 1-box BINARY — the integrated residual fails as a 15-box override (EXP-Q2b) but works as "is *this* box a fake?" (EXP-Q3); (4) cap a threshold sweep at the physically-motivated value (the fake-noise floor) so the LOO can't pick an unphysical knob that overfits; (5) `within_r` rewards *position*, so an output-dynamics smoother (1-Euro) that kills per-frame jitter is not a tax but a small LIFT — the jitter caused single-frame radius pop-outs that smoothing removes (`fpath_human`, +0.008/+0.022 both boards while making the dot human-like).

**field family (REMOVED 2026-06-22, documented for history).** The non-trellis `field`/`field_lag`/`field_coh`
modes and their `OutlierTracker` backbone (`ld/track/tracker.py`) were deleted in the cleanup — all were
dominated by the `fpath` lineage (≤0.773 vs 0.940). For the record: **field** fit global rigid sheet motion
(RANSAC) in `motion.py`, treated outlier features as a saliency field, followed the peak with a gated CV
tracker, and snapped to the highest-saliency YOLO box (snap was load-bearing: raw ≈ 0.28 → +snap ≈ 0.59);
**field_coh** added a coherence far-jump override. They beat `fpath_coh` on t1 (no path memory to lock in),
which is why they were kept on the board for a while before the decode-layer freezes closed that gap.

## Detection model history

| Model | Oracle | Notes |
|-------|--------|-------|
| original `yolov8n_combined` | 0.915 | First s* model; no edge examples; t1 oracle 0.806 (shape drifts to top edge, missed) |
| `yolov8n_combined` (edge re-annotated) | 0.901 | Added edge boxes single class — hurt t4/t5, net regression |
| `yolov8n_combined-5` (2-class) | — | shape_full/shape_partial split regressed overall; too few samples per class |
| **`yolov8n_single_combined`** | **0.958** | **Current best.** s1–s12 + 5 targeted t1 edge frames (top-left, f225–300), single class, collapsed partial→full labels. |

**Key lesson:** the two-class approach failed at ~55 frames — splitting halved effective instances per class. The right fix was targeted visual coverage of the specific failure position, not a label schema change.

## Physics

- Real shape speed: **median 1.3 px/frame, p90 4.5, p99 17.8, max 44.7**
- Shape radius: **~56 px**
- Failure mode: **CREEP not JUMP** — track is first lost via small steps (~12 px) onto an adjacent fake. Big teleports are `reacquire` firing *after* the track is already lost.

## How to train YOLO

Training data: `data/detect/s_frames/` (PNGs) + `data/detect/s_labels_single/` (YOLO .txt, single class). ~12 boxes/frame. Canonical paths are the `TRAIN_*` constants in `ld/config.py`.

```bash
# Add new data: drop a video in data/, then one interactive session does it all
# (discover -> board-crop -> extract 5 in-play frames -> single-class label -> rebuild)
python -m ld.detect.annotate
python -m ld.detect.annotate --skip-extract    # re-annotate already-extracted frames

# Build the dataset on its own + train (~12 min on M3 Pro)
python -m ld.detect.build_dataset
python -m ld.detect.train --name yolov8n_single_combined
# Weights: data/detect/runs/yolov8n_single_combined/weights/best.pt
```

`annotate` extracts only **in-play** frames (no countdown / START / success overlay — the
longest run clean of large bright blobs) and crops raw screen captures to the 744×498 board
via `ld/detect/board_crop.py` (shared with `make_additional_evidence.py`).

Annotator controls: drag=draw box, `u`=undo, `c`=clear, `n`/space=next, `p`=prev, `s`=save, `q`=quit. Single class ("shape") — box every shape on the sheet.

Key training notes:
- Always use `yolov8n.pt` (pretrained COCO), not random init.
- The real shape ranks ~10–13 in confidence; **never filter by confidence**.
- Run with `.venv/bin/python` (Python 3.12 venv at project root).

## How to evaluate

```bash
# Full leaderboard across t1–t10 (~12 min)
python -m ld.detect.eval_modes --weights data/detect/runs/yolov8n_single_combined/weights/best.pt

# Single mode / subset (fast) — NOTE: overwrites LEADERBOARD.md
python -m ld.detect.eval_modes --weights .../best.pt --modes fpath fpath_human --clips t4 t8
```

The default board is `fpath_human` (leader) + `fpath` (ablation base) + an `oracle` ceiling
row. The intermediate `fpath_*` stages are still runnable by name (`--modes fpath_hyst …`),
e.g. to regenerate the per-mode eval CSVs the tuning probes read.

Per-frame trace CSVs: `data/detect/eval/<clip>__<mode>.csv`. Primary debugging tool — read to see how a track fails (creep vs jump, which state).

Always quote LOO numbers (from the tuning probes) as the real metric. In-sample is optimistic.

## How to generate evidence videos

```bash
# Leader (default)
python -m ld.detect.render_evidence \
    --weights data/detect/runs/yolov8n_single_combined/weights/best.pt --mode fpath_human

# Output: data/detect/evidence/<clip>_<mode>.mp4
```

Overlay legend: green rectangles = YOLO detections, red filled dot = the emitted cursor
guess, cyan crosshair = GT, red line = miss (outside radius).

## Dead ends (do not revisit without a new angle)

| Avenue | Result | Why it failed |
|--------|--------|---------------|
| Velocity cap (output-side) | 0.714→0.703 | Cap below p99 real speed; fought legitimate motion |
| Confirm-gate (reject far jumps) | 0.672→0.627 | Failure is creep not jumps; holding wrong incumbent locks in error |
| Proximity-fused emission (`fpath` prox_w>0) | Hurts every clip | CV prior reinforces lock-in on wrong box |
| field⇄fpath regime router (3 forms) | No causal key | ~0.83 oracle (old model) but per-clip regime has no live-observable signal |
| Box-level rigid residual | Not complementary | Hits flow-miss frames at base rate; attacks wrong gap |
| Multi-baseline optical flow (K>1) | GT rank monotonically worse | LK noise accumulation swamps slow-drift gain |
| Temporal-feature learning (logreg LOO) | −0.060 top-1 | History features overfit; net noise on held-out clips |
| Rotation selector | Lift +0.005 (noise) | Not preferentially complementary; no causal key |
| NCC template tracker | within_r 0.06 | Real shape rotates; translation-only template de-correlates |
| Log-polar phase corr tracker | Regime-coupled | Rescues strong-signal clips only; no causal key |
| Coherence snap-weight blend | Zero change | Signal is outside snap radius; must be far-reaching |
| Box cleanup (border/conf/envelop) | ≤+0.003, some regress | Competing boxes are genuine neighbours, not removable clutter |
| Top-K / conf filter | Oracle drops | Real shape ranks 15–28 on hard frames; filtering by confidence loses it |
| Two-class YOLO (shape_full/shape_partial) | Oracle regressed | Insufficient data per class at ~55 frames; single-class + targeted examples wins |
| Coherence far-jump reacquire on `fpath_fuse` | Gate failed (step3_gate) | On t1/t8 residual misses a far coherent challenger is on-GT+far only 0.085/0.020. t1 creep is onto ADJACENT boxes (far-jump never fires); t8 coherence doesn't point at the real shape (on-GT 0.29). |
| Transition-penalty cap on `fpath_fuse` | Net −0.025..−0.075 | Caps lock-in depth → small t8 gain (+0.027) but regresses strong clips that depend on lock-in stability (t2 −0.048, t10 −0.023). Regime-coupled, same as `fpath_reacq`. |
| Appearance/texture channel (EXP-1, `exp1_appearance_probe`) | Gate failed; worse than motion | Hypothesis: real shape's interior changes more than a rigid fake's after de-rotating by the global sheet angle. Dead two ways: (1) **the sheet barely rotates** — mean \|θ/frame\| ≈ 0.006° → de-rotation is a no-op (`derot_ncc`==`raw_ncc` to 3dp), no rotational appearance signal exists. (2) On miss frames the real box ranks WORSE by appearance than by motion (t8 MISS-top1 0.16 vs mass 0.32); a longer N=8 baseline COLLAPSES to 0.02 — at median 1.3 px/frame interior change is in the resampling-noise floor, and on sustained drift-locks the real shape moves slowest, so appearance-change is anti-correlated with real-ness. Ring-histogram + log-polar-FFT (rotation-invariant) variants also < motion. |
| Sub-box coherent-mass re-localization (EXP-A, `expA_subbox_probe`) | Gate failed on target; trellis regressed | Hypothesis: oversized/merged GT boxes dilute coherent-mass with peripheral incoherent noise outliers; measuring over the tight in-box coherent sub-cluster recovers the rank. **DEAD: the signal-limit is not box-dilution.** Gate (whole/wcentroid/topk/grid sub-clusterers): t8/t5 MISS-top1 stayed flat (~0.22/0.29) for every variant — the real shape's coherent vectors are not dominant in ANY sub-window. `topk` lifted t1's *isolated* MISS-top1 0.29→0.42 (robust to keep_frac), but feeding it into the trellis (`fpath_sub`) REGRESSED every clip (mean 0.878→0.841, t1 0.798→**0.736**, t5→0.641). Reconfirms the causal-key wall: isolated-channel top-1 ≠ track. |
| Detection knobs imgsz=1024 / conf=0.10 (EXP-2 free variant, `step2_detknob`) | Oracle↑ does NOT convert to identity↑; fails no-regression | imgsz=1024 lifts oracle to ~0.98 on all laggards (t8 **0.939→0.983**, gap f226–248 0.13→0.74) and tightens t8's oversized boxes (GT size-ratio med 1.25→1.00, max 3.50→2.49) — yet full-board `fpath_hyst`@1024 is mean 0.878→0.881 (+0.003) **with t4 −0.046, t2/t9/t10 dips**, and **t8 identity stays flat 0.771→0.769 despite oracle 0.983**. Confirms t8/t5's wall is a genuine IDENTITY signal-limit, not detection: the real box being present doesn't help because motion channels can't rank it, and higher-res adds confusable boxes elsewhere. The 0.915→0.958 "oracle lifts identity in lockstep" lesson was about box QUALITY on already-detected shapes, not adding boxes on hard frames. (conf=0.10 only adds more clutter.) Detection settings are global — no causal key to use 1024 only where it helps. |
| t1/t4 countdown-lock "bug" (`step3a_lockdiag`) | Bug does not exist | Long-suspected: lock seeds the wrong shape on t1/t4. **DEAD: the lock is correct on all 10 clips** — `compute_countdown_lock` picks the oracle-nearest box within radius every time (dist(lock,GT)==dist(oracle,GT)). t1's f133 drift is genuine post-lock identity creep, not a seed error. Remove from open avenues. |
| Decode-layer hedge — *magnitude*-trust + sheet-decomposition variants (`hedge_probe`) | Both dead; the *coherence*-trust variant SHIPPED as `fpath_hedge` (see leader) | The two intuitive forms of the decode-layer hedge fail, but a third does not. (1) **Sheet-decomposition coast** `p_t=p_{t-1}+(c_t−T(c_{t-1}))` scores 0.00 on swept frames: during a lock the output **hops between boxes** (|d_indep| 43.6/11.5 px/fr), so integrating the chosen box's per-frame motion drifts — only a hold-position **freeze** recovers (it recovers 100%). (2) **Magnitude-gated trust is ANTI-correlated**: chosen-box |d_indep| is *higher* on laggards (9.8–10.3 px/fr) than on strong clips (3.3–7.4) — the slow correctly-tracked real shape reads LOW-confidence, the churning fake-hop reads HIGH, so "low motion → freeze" fires backwards (safe but inert, laggard gain +0.001). **The fix that worked:** trigger on DIRECTIONAL COHERENCE, not magnitude — `churn = mean|d_indep|*(1−R)`, R=|Σd_indep|/Σ|d_indep|; a coherent real-shape burst has R≈1→churn≈0→commit, an incoherent box-hop has R≈0→churn high→freeze. Shipped as `fpath_hedge` (+0.021, LOO 0.8988, worst-clip +0.000). Lesson: at the decode layer the separator is trajectory *shape*, not *speed*. |
| Direct outline-rotation discriminator (EXP-R1, `rot_probe`) | Gate failed; t7-sanity broke the estimator | Hypothesis (plan.md): the real tile's **border** spins independently (~1.25°/frame on t7) while fakes stay axis-aligned with the rigid sheet (0.006°/frame) — a human's actual tell. Prior rotation work measured the wrong thing (EXP-1 de-rotated by the *sheet* angle then read *interior* NCC; NCC/log-polar ran full-patch as trackers; `curl` is an indirect LK-flow proxy). This probe measured boundary orientation DIRECTLY: Sobel edge map per box, **interior masked to a boundary annulus** (the fill is orientation-less sloshing specular), then inter-frame rotation via a ±8° rotation-bank-NCC argmax (`rot_bank`) and log-polar+phase-corr (`rot_lp`), accumulated over a 10-frame window. **DEAD, two ways.** (1) **t7 SANITY FAILS:** on the clean clip where the spin is *visually obvious*, rotation ranks the real box #1 only ~0.37–0.40 of all-frames vs mass 0.76 / coh 0.90 — the one provably-rotating tile is not a clear #1, so fakes accumulate comparable rotation from **noise**. A 1.25°/frame spin at r≈25px is a **~0.5px sub-pixel boundary step**, i.e. the same resampling-noise floor that killed EXP-1; mask tuning ([0.35,1.0]→[0.55,1.0]) didn't lift it. (2) **t8 (priority laggard) never separates:** MISS-top1 only ties/marginally beats mass (0.33→0.38) and top-3 is *worse* (0.61–0.67 vs mass 0.80) — bad for a Viterbi emission that leans on top-3 continuity. No single estimator beats both mass AND curl on all of t1/t5/t8 (rot_bank wins t1 0.467, rot_lp wins t5 0.495, both weak on t8). Genuinely orthogonal to the trellis failure (MISS-top1 ≈ all-top1, doesn't collapse on hard frames — unlike mass/coh) but orthogonal-and-weak is exactly what logreg already overfit. Reconfirms the triple-confirmed t8/t5 identity information-limit: the real shape is separable only by independent translation, slowest on drift-locks. Do not re-propose boundary/edge rotation. |
| Integrated sheet-frame residual as identity (EXP-Q1/EXP-Q2b, `sheet_residual_probe` / `resid_override_probe`) | Measurement real; re-selection DEAD | Hypothesis (plan.md EXP-Q1): the identity wall is an SNR limit, not an information one — per-frame independent translation (~1.3 px/fr) is sub-noise, but **integrating** each box's divergence from rigid sheet prediction over N≈30 frames (affine-chained association, drop chains > radius) should make the real shape's directional drift grow ~N·v above the ~√N·σ noise. **EXP-Q1 was a real positive:** at N≈30 the residual ranks the real box #1 on pooled laggard MISS frames at **0.65–0.68** (t8 0.68) — the *first* channel to beat the mass MISS-top1 wall of 0.32–0.41. **But EXP-Q2b (the persistence-gated hysteresis override, mirroring `fpath_hyst` but keyed on residual-dominance) is DEAD: every config in (N∈{30,45}×margin∈{.15,.30,.50}×K∈{5,8,12}) regresses, worst-clip −0.22..−0.40 on a STRONG clip, laggard-mean itself negative, LOO finds 0 admissible configs → +0.000 raw and hedged.** Root cause: the EXP-Q1 SANITY (real-box residual rank #1 on *correctly-tracked* frames) is clip-dependent and **t7 (0.88) was unrepresentative** — on near-perfect strong clips **t9 0.51, t10 0.57** a fake out-residuals the real box ~half the time, so any residual-keyed switch false-captures and craters them. The 0.68 "on MISS frames" is barely above those clips' baseline all-frame top1 (~0.6): moderately-good-everywhere, never sharp. This **also kills the lock-gate variant** (fire only when held-box residual near the fake-floor) — it needs "real box ⇒ high residual when tracked" as its key, false on t9/t10/t1/t5/t8. Residual-as-identity dead in BOTH forms (additive emission AND override). The EXP-A causal-key wall, reconfirmed: strong isolated top-1 on a curated MISS subset ≠ a track. **BUT the residual is NOT wasted — EXP-Q3 (`fpath_freeze`, shipped, the leader) reuses the very same residual as a 1-box BINARY ("is the box I'm ON a fake?", sharp: ~9–15px fake vs ~45–91px real) rather than a 15-box ranking, and freezes the output — +0.033 board, no per-clip regression. The lesson: a signal too weak to RANK can be sharp as a per-box yes/no.** |
| Box-dimension rotation pulse discriminator (EXP-S1, `box_pulse_probe`) | Gate failed; t7-sanity broke the readout | Hypothesis (plan.md): a rotating square's tight AABB PULSES in size (side `= s·(\|cosθ\|+\|sinθ\|)`, area up to 2× at 45°) while a rigid fake's box is dead-flat — so the per-box AABB size-oscillation read STRAIGHT OFF THE DETECTOR over a long window is a rotation readout ABOVE the pixel noise floor that killed EXP-1/EXP-R1 (the box is a detector aggregate of thousands of pixels). The deliberate successor to EXP-R1: same observable (rotation), different measurement level (detector geometry) to dodge the sub-pixel floor; and rotation doesn't stop when translation stalls, so it *could* survive the drift-locks where translation (the info-limit's only signal) goes to zero. Probe read ONLY `packs[idx].boxes` dims (no pixel pass): per box, walk backward by nearest-prev-centroid over window W and collect side `s=√(w·h)`; four pulse stats — `area_cv` (magnitude), `side_ac1` (lag-1 autocorr, the STRUCTURE primary per the hedge lesson), `side_smooth`, `side_trend`. **DEAD, both ways (W=15 and W=25).** (1) **t7 SANITY FAILS:** on the clip where the spin is visually obvious, the best pulse stat ranks the real box #1 only ~0.09–0.11 of all-frames (`side_ac1`) vs mass 0.76 / coh 0.91 — the detector boxes do NOT track the AABB pulse; YOLO box size-jitter (frame-to-frame) swamps the `s·(\|cos\|+\|sin\|)` signal. (2) **Laggards never separate:** every pulse stat's MISS-top1 (best `area_cv` ~0.14 overall) is far below BOTH mass (0.37) AND curl (0.27) on t8/t5/t1 — and the STRUCTURE stats (the "real bet") are *weaker* than the magnitude `area_cv`, near-zero on MISS (`side_ac1` overall MISS-top1 0.007). Longer window (W=25) is WORSE (accumulates more jitter). Confirms the §3 base-case: detector box noise exceeds the pulse, so box-geometry rotation joins interior-appearance (EXP-1) and edge-rotation (EXP-R1) as signal-limited. **There is now NO untested orthogonal identity observable left** — interior appearance, edge/outline rotation, AND box-geometry rotation are all empirically dead. Do not re-propose any form of rotation-as-identity. |
| In-box point refinement / localization (EXP-LOC, `localize_probe`, 2026-06-21) | Structural hypothesis confirmed; gate failed — no causal key | Hypothesis (plan.md "EMIT A SINGLE TRACED POINT"): the pipeline emits the chosen box's CENTROID, but the GT crosshair sits at a *specific spot on the shape* ≠ centroid when the detector box is oversized/merged (the t5/a03/t3 profile) — so the right box can be chosen and the point still misses. Refining the point *inside the already-chosen box* is a LOCALIZATION lever, separate from the (walled) identity-ranking problem. **The structural claim is REAL:** on correct-box-chosen frames the centroid→GT offset grows monotonically with box size-ratio (centroid-miss% by SR bucket `<1.2/1.2–1.6/1.6–2.5/≥2.5` = 5.0/8.2/12.5/21.0% on t1–t10), and the oracle-localization ceiling is large — GT lies *inside* the chosen box on **0.899** (t) / **0.986** (add) of right-box misses (they'd flip to hits with a perfect localizer). **DEAD anyway, no causal key:** the only available in-box estimators — **saliency-peak** (`saliency_map` argmax in box) and **outlier-weighted-centroid** (`owc`) — are both NOISIER than the raw centroid and net-REGRESS on every clip even ungated (owc mean Δ −0.063 t / −0.090 add, worst −0.130/−0.183; saliency far worse). The plan's **size-ratio-GATED** swap (refine only oversized boxes, keep the centroid elsewhere) nets **negative at every threshold** on both boards (best `sr>3.0`: mean −0.001, worst-clip −0.010 t / −0.003 add) — tightening the gate only converges to zero by swapping nothing. Also the recoverable headroom is tiny on t1–t10 (right-box misses are 89/4965 ≈ 1.8% of correct-box frames → ceiling ≈ +0.016; larger ~7.5% on add). The centroid is already the best CAUSAL point inside the box; the moving sub-cluster sits off the GT more often than the box middle. Same shape as the identity wall — a real signal with no online key. Do not re-propose saliency/owc point refinement; revisit only with a NEW in-box localizer that beats the centroid on *well-sized* boxes (the safety scope), not just the rare oversized ones. (Phase 0 — the filled-red-dot evidence overlay — DID ship; it's metric-neutral.) |

## Open avenues (highest-EV first)

Target: ≥0.90 mean within_r — **cleared: 0.932** (`fpath_freeze`, full-board; LOO 0.9359), from the 0.899→0.878→0.876→0.825 lineage. **Both** of the last two lifts came from the *decode/output* layer (churn-hedge, then residual-freeze), NOT from cracking identity — `within_r` rewards position, not box-id. Gate every idea with a read-only probe before building; accept only on LOO improvement with no per-clip regression. Short version of what's left:

**The big laggard t8 is fixed (0.776→0.909).** `fpath_freeze`'s residual-freeze caught its long coherent fake-rides that the churn hedge was blind to. The new laggards are **t5 (0.858), t4 (0.880), t1 (0.895)** — what remains is shorter locks plus t3/t5's *undersized-box* runs (detection-quality, EXP-2) and t4's residual misses. Next-target ideas:

0. **DONE — `fpath_freeze` (avenue 0, shipped).** The old "extend the hedge to t8's non-swept locks" avenue is solved: instead of a second *churn* trigger, the residual-as-binary fake-detector freeze recovered them (+0.133 on t8). Possible increments: a longer/secondary freeze for the residual frames where the chain breaks (the freeze currently no-ops there — conservative), or composing the freeze with EXP-2's box-quality fix on t5.

**The t8/t5/t1 *identity* floor is a TRIPLE-CONFIRMED information-limit (read before proposing an identity fix). Note `fpath_freeze` did NOT break it** — it sidesteps it at the output layer (detect "on a fake" → freeze; never re-identifies the real box). The limit still holds for any *identity*-channel idea: On the laggard lock frames the real box is present (oracle ~0.95, top-3 ~84%) but no measurement ranks it #1: (1) **appearance/rotation** is structurally dead (EXP-1: sheet barely rotates ≈0.006°/frame, real shape too slow → interior change below noise floor); (2) **sub-box re-localization** of coherent-mass is dead (EXP-A: t8/t5 MISS-top1 flat ~0.22/0.29 for every sub-clusterer; the real shape's coherent vectors aren't dominant in any sub-window — *not* a box-dilution artifact); (3) **better detection** doesn't convert (`step2_detknob`: imgsz=1024 lifts t8 oracle 0.939→0.983 and shrinks its oversized boxes, but t8 identity stays flat 0.769 — the real box being *present* doesn't help when its independent motion is below what mass/coherent-mass/curl can separate). The real shape is distinguishable **only by independent translation relative to the rigid sheet**, and on these sustained drift-locks it translates slowest → least signal. **(4) Even *integrated* translation doesn't convert (EXP-Q1/EXP-Q2b, 2026-06-20):** the cumulative sheet-frame residual at N≈30 IS a real measurement (lifts t8 MISS top1 0.32→0.68) — so the per-frame "sub-noise" framing was incomplete — but it is not *sharp* enough to re-select boxes online (ranks the real box #1 only ~0.5–0.6 even on correctly-tracked strong clips t9/t10), so a persistence-gated override false-captures and regresses every clip (LOO +0.000). This is a genuine information limit, not a measurement or detection artifact. Do **not** re-propose appearance channels, sub-box windows, "just improve detection," **or residual-keyed re-selection (emission or override).**

Remaining ideas, all LOW-EV given the above — gate hard, expect little:

1. **EXP-2 — targeted detector annotation, ONLY for t5's *undersized* lock (f369–413, GT box 0.26–0.41×).** Distinct from the (now-dead) imgsz knob: undersized boxes may genuinely clip the real shape's outlier vectors, losing mass — a *quality* fix retraining could address where the global imgsz knob can't (it regressed t4). t8's case is shown identity-bound regardless of box quality, so do **not** annotate t8 expecting a lift. Medium effort (interactive annotate + CPU retrain); uncertain EV.
2. **EXP-3b — affine-carried hysteresis association (untested; cheap-ish).** The `fpath_hyst` EMA carries evidence by nearest-centroid association, which can mis-associate across sheet translation. Carrying by the global affine instead (`MotionField.affine`) could let the hysteresis escape the C-type coast-runaways (Step-0: t1 f231–241, t8 f340–367). **But** this improves *association*, not *signal* — and signal is the wall (t8 EMA-on-GT <0.30 won't dominate even with perfect carry), so the realistic upside is a few t1 frames. Gate first (extend the OV cache with affines; compare affine-carry vs nearest-centroid EMA-on-GT, target >0.51 on t1).
3. **EXP-4 — learned per-box discriminator (only if a NEW orthogonal geometric feature appears).** Appearance + sub-box geometry are both now dead, so there is currently no untapped feature to learn over. Do not attempt without one. logreg already overfit at this data size (−0.060).

   **Permanently off the table:** interior appearance/texture channels (structural); **all three forms of rotation-as-identity** — interior NCC (EXP-1), direct outline/boundary-rotation (EXP-R1: sub-pixel spin below the pixel floor, fails t7 sanity), AND box-dimension rotation pulse (EXP-S1: detector box size-jitter swamps the AABB pulse, also fails t7 sanity); sub-box coherent-mass windows (EXP-A dead); global detection knobs / transition softening/cap (regime-coupled); mode routing/ensembling (oracle-router ceiling 0.858 < 0.90); "improve detection to lift t8" (oracle 0.98 doesn't convert). **No untested orthogonal identity observable remains** — rotation has now been mined at the interior-pixel, boundary-pixel, AND detector-geometry levels and is dead at all three.

**Honest status:** `fpath_freeze` (0.932, LOO 0.9359) is the leader and the largest single lift in the lineage (+0.033 full-board), reached by **two** decode-layer ideas stacked: hedge the *position* when the pick churns incoherently (`fpath_hedge`, catches swept locks), then *freeze* the position when the held box reads as a rigid fake (`fpath_freeze`, catches coherent locks). The win is that `within_r` rewards position, not box-id, so you never have to win the (unwinnable) identity argument — you only have to notice you're on a fake and stop. The identity information-limit on t5/t1 (and the now-fixed t8) is **unchanged and still real**: `fpath_freeze` sidesteps it (detect "on a fake" via the integrated residual *binary*, freeze; never re-identifies the real box). Every *identity*-channel candidate remains dead — interior appearance (EXP-1), all three rotation levels (EXP-R1 outline, EXP-S1 box-pulse, both fail t7 sanity), sub-box coherent-mass (EXP-A), and residual-as-*ranking* (EXP-Q2b, every config regresses). Remaining headroom toward oracle 0.958: the **detection-quality** undersized-box fix (EXP-2, for t3/t5) and squeezing the residual-freeze's chain-break frames — NOT a new identity channel. The identity ceiling for this stack is unchanged; the *position* ceiling is now ~0.93 and climbing toward the 0.958 oracle.

## Conventions

- **Strictly causal** for anything shipped. Offline only for diagnostics/ceilings.
- All identity modes dispatch through `identity._dispatch_mode`. `ALL_MODES` is the full `fpath` lineage; `BOARD_MODES` (`fpath`, `fpath_human`) is the default `eval_modes` board (oracle ceiling added as a row). Intermediate `fpath_*` stages still run via `eval_modes --modes <name>`.
- The green crosshair in clips is **GT only** — `cursor.strip_pointer` inpaints it before any tracking.
- Accept a change only if it improves **LOO mean with no per-clip regression**.
- Eval is cache-backed; cache key is md5(path+mtime) of the weights file.

## Key files

| File | Role |
|------|------|
| `ld/detect/identity.py` | Identity modes (`fpath`…`fpath_human`); `_dispatch_mode`, `ALL_MODES`, `BOARD_MODES`, `run_clip` |
| `ld/detect/fusion.py` | `detect_fusion_clip` — cached YOLO detection per clip (`FusionPack`) |
| `ld/vision/motion.py` | `estimate_motion` (RANSAC), `saliency_map`, `MotionField.outlier_vectors` |
| `ld/vision/cursor.py` | GT crosshair detect + `strip_pointer` inpainting |
| `ld/detect/eval_modes.py` | Leaderboard harness → `LEADERBOARD.md` |
| `ld/detect/render_evidence.py` | Per-clip overlay video (YOLO boxes + red-dot guess) |
| `ld/track/humanize.py` | `HumanCursor` (1-Euro + deadband) + `humanize_track` — output dynamics (`fpath_human`) |
| **Training** | |
| `ld/detect/annotate.py` | Discover + board-crop + extract 5 in-play frames + single-class annotate |
| `ld/detect/board_crop.py` | Board detection + crop (shared with `make_additional_evidence.py`) |
| `ld/detect/build_dataset.py` | Builds the single-class YOLO dataset; prints the train command |
| `ld/detect/train.py` | Fine-tunes YOLOv8n from COCO-pretrained weights |
| `ld/detect/infer.py` | Run weights on held-out frames for a by-eye box check |
| `make_additional_evidence.py` | (repo root) crop+trim raw `data/*.mp4` → `data/additional_evidence/` (held-out set) |
| **Tuning probes (LOO, kept for retuning the leader)** | |
| `ld/detect/fuse_sweep.py` / `exp3_sweep.py` | fuse-weight / hysteresis LOO sweeps |
| `ld/detect/hedge_probe.py` | `fpath_hedge` churn_hi LOO sweep |
| `ld/detect/resid_freeze_probe.py` | `fpath_freeze` τ/lag/consec LOO sweep |
| `ld/detect/cursor_physics_probe.py` | `fpath_human` 1-Euro/deadband LOO sweep |
| `ld/detect/probe_common.py` | Shared helpers for the above probes |
| `ld/config.py` | Tracker / detection / motion / training (`TRAIN_*`) / human-cursor tunables |

## Session experiment log (don't retry — quick index)

Compact list so an agent knows what has already been run and what NOT to retry. Full reasoning for the
detailed ones is in the Dead ends table above; numbered EXP-* hypotheses live (or lived) in `plan.md`.

- **DONE / SHIPPED — human-cursor output dynamics (`fpath_human`, `ld/track/humanize.py`, `cursor_physics_probe.py`, 2026-06-21).**
  The "make the emitted (x,y) move like a hand" plan. A strictly-causal **1-Euro filter + 2px deadband** applied as
  the LAST transform on the emitted point (decode-layer only; box decisions byte-identical to `fpath_freeze`).
  Gated OFFLINE on the eval CSVs (no detector re-run): swept `(min_cutoff, beta, deadband, v_max, a_max, lag)`;
  the simplest config (1-Euro `min_cutoff=1.0, beta=0.007` + `deadband=2`) was the single config admissible on
  BOTH full boards. **Result: within_r +0.008 t1–t10 (0.932→0.940) / +0.022 add (0.837→0.859), flat-or-up on
  every clip (worst −0.003, a single-frame flip within tolerance), RMS jerk −61%/−65%, velocity-reversals
  −97%/−98%, p99 jump 47→31 / 44→27 px.** First decode-layer transform to improve smoothness AND within_r (the
  smoothing removes jitter-induced single-frame radius pop-outs). **PD steering (bounded-velocity) and fixed-lag
  (L=8/12) both REGRESSED within_r** (PD worst −0.014..−0.022; lag worst −0.023..−0.071 — over-smoothed/lagged
  off fast bursts) and were dropped — 1-Euro alone (the cheapest layer) was the win, matching the project's
  fewer-params-generalize lesson. Tune via `cursor_physics_probe.py`; params in `ld/config.py` (`HUMAN_*`).
- **EXP-L1 — fixed-lag bidirectional decode smoother (`lag_smooth_probe.py`) — WEAK, PARKED.** Emit frame
  `t−L` after seeing `t±L`, replace the chosen position with a radius-bounded robust fit (median/Theil–Sen)
  when its excursion exceeds the physical bound (p99 17.8 / max 44.7). Gate result: the catastrophic 200–400px
  rides are ALREADY GONE (`fpath_hyst` step-0: only t1 has one 11-frame run, t8 one 28-frame run, **t5 zero**);
  the hedge ate them. Residual recovery is modest (~16–19% of t5/t1 misses; median > Theil–Sen, which damages
  strong clips) and is short-excursion smoothing, not ride-rejection. EXP-L2 not built (realistic lift only
  +0.01–0.02). Revisit only as an increment (median, L≈12) — but note the residual integrator below is now DEAD,
  so EXP-L1 is the *last* identity-adjacent lead and even it is a weak decode-layer smoother, not a new signal.
- **DONE / DEAD — cumulative sheet-frame residual integrator (EXP-Q1) + residual override (EXP-Q2b).**
  (`sheet_residual_probe.py`, `resid_override_probe.py`, 2026-06-20.) Transported each box's centroid into the
  sheet frame via the cumulative inverse RANSAC affine and accumulated divergence from rigid prediction over
  N≈15–90. EXP-Q1 was a **real positive** (t8 MISS top1 0.32→**0.68** at N≈30 — first channel to beat the mass
  wall), so the SNR limit was *partly* real. **But the follow-through EXP-Q2b (persistence-gated residual-dominance
  override, the only build form that respects the EXP-A causal-key lesson) is DEAD: every config regresses, LOO
  +0.000.** The residual is moderately-good-everywhere (ranks the real box #1 only ~0.5–0.6 even on
  *correctly-tracked* strong clips t9 0.51 / t10 0.57 — t7's 0.88 was unrepresentative), never sharp enough to
  re-select boxes without false-capturing and cratering the strong clips. Confirms ~0.90 is this stack's identity
  ceiling; the remaining floor work is **detection-quality** (EXP-2 undersized boxes), not a new identity channel.
- **DONE / DEAD — in-box point refinement / localization (EXP-LOC, `localize_probe.py`, 2026-06-21).** The
  "emit a traced point not a box" pivot (plan.md). Phase 0 (filled-red-dot evidence overlay) SHIPPED. Phase 1
  gate confirmed the *structural* claim — centroid→GT offset grows with box size-ratio, and GT lies inside the
  chosen box on 0.899 (t) / 0.986 (add) of right-box misses (a real oracle-localization ceiling) — **but failed
  on causal key:** the available in-box estimators (saliency-peak, outlier-weighted-centroid) are noisier than
  the raw centroid and net-regress on every clip; the size-ratio-gated swap nets negative at every threshold
  (best `sr>3.0` mean −0.001, worst-clip still <0). Phase 2 NOT built. The centroid is already the best causal
  in-box point; recoverable headroom on t1–t10 is only ~1.8% of correct-box frames. Don't re-propose saliency/owc.

### Already dead — do NOT retry (one-liners; see Dead ends table for why)

- Per-frame / short-window translation channels: mass, curl, coherence, |d_indep|, box-level rigid residual (single-frame). All sub-noise on laggard MISS frames.
- Integrated (cumulative sheet-frame) residual as identity (EXP-Q1/EXP-Q2b) — real measurement (t8 MISS top1 0.32→0.68 @N≈30) but not sharp enough to re-select boxes (ranks real box #1 only ~0.5–0.6 on correctly-tracked strong clips); override regresses every config, LOO +0.000. Dead as emission AND override.
- Appearance/texture as identity (EXP-1) — sheet barely rotates (~0.006°/fr), interior change below resampling-noise floor.
- Rotation as identity, ALL THREE levels: interior NCC (EXP-1), boundary/outline rotation (EXP-R1), box-AABB size-pulse (EXP-S1). All fail t7 sanity (sub-pixel / detector jitter swamps the spin).
- Sub-box coherent-mass re-localization (EXP-A) — real shape's coherent vectors not dominant in any sub-window; not a box-dilution artifact.
- Detection knobs imgsz=1024 / conf=0.10 (step2_detknob) — oracle↑ does NOT convert; identity still can't rank the present box; global, no causal key.
- Multi-baseline optical flow (K>1) — per-pixel LK noise accumulation swamps slow-drift gain.
- Velocity cap (below p99), confirm-gate (rejects far jumps), proximity-fused emission (prox_w>0), transition-penalty cap — all anti-lock dead ends (fight real motion / hold wrong incumbent / regime-coupled).
- Magnitude-trust & sheet-decomposition decode hedges (`hedge_probe`) — magnitude is anti-correlated with real-ness; only the COHERENCE-trust freeze worked (shipped as `fpath_hedge`).
- Mode routing / ensembling (oracle-router ceiling 0.858), two-class YOLO, top-K/conf filter, NCC/log-polar trackers, temporal-feature logreg, rotation selector. All dead.
- t1/t4 countdown-lock "bug" (step3a_lockdiag) — does not exist; lock is oracle-correct on all 10 clips.
