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

The best model is `yolov8n_single_combined` (single-class, oracle 0.974). Weights are not committed to the repo — you must train locally. Takes ~12 min on M3 Pro, longer on Windows/CPU.

**Mac/Linux:**
```bash
.venv/bin/python -m ld.detect.build_s_dataset \
    --labels-dir data/detect/s_labels_single \
    --out-dir data/detect/dataset_single_combined

.venv/bin/python -m ld.detect.train \
    --data data/detect/dataset_single_combined/dataset.yaml \
    --name yolov8n_single_combined
```

**Windows:**
```powershell
.venv\Scripts\python.exe -m ld.detect.build_s_dataset --labels-dir data/detect/s_labels_single --out-dir data/detect/dataset_single_combined

.venv\Scripts\python.exe -m ld.detect.train --data data/detect/dataset_single_combined/dataset.yaml --name yolov8n_single_combined
```

Weights will be saved to `data/detect/runs/yolov8n_single_combined/weights/best.pt`.

### 4. Generate evidence videos for all identity methods

Run evidence renders for all competitive identity modes across t1–t10:

**Mac/Linux:**
```bash
for mode in fpath field_coh field_lag field; do
    .venv/bin/python -m ld.detect.render_evidence \
        --weights data/detect/runs/yolov8n_single_combined/weights/best.pt \
        --mode $mode
done
```

**Windows:**
```powershell
foreach ($mode in @("fpath", "field_coh", "field_lag", "field")) {
    .venv\Scripts\python.exe -m ld.detect.render_evidence --weights data/detect/runs/yolov8n_single_combined/weights/best.pt --mode $mode
}
```

Output: `data/detect/evidence/<clip>_<mode>.mp4`

---

## Pipeline (two separable stages)

1. **Detection** — YOLOv8n (`data/detect/runs/yolov8n_single_combined/weights/best.pt`) finds candidate shape boxes per frame. Oracle within_r ≈ **0.974**. Detection is effectively solved. The real shape is camouflaged and ranks ~10–13 in confidence; you cannot filter by confidence without losing it.

2. **Identity** — decides *which* box is the real shape over time. This is the bottleneck and the entire focus of remaining work. Code in `ld/detect/identity.py`.

## Current best: `fpath` (0.855 within_r)

`fpath` is now the best mode with the current detector. The regime split that previously made it unreliable (collapsing on weak-signal clips) has largely resolved — with near-perfect oracle (0.974), fpath's Viterbi path integration dominates across the board.

| clip | within_r | oracle |
|------|----------|--------|
| t1 | 0.806 | 0.983 |
| t2 | 0.963 | 0.975 |
| t3 | 0.902 | 0.943 |
| t4 | 0.729 | 0.953 |
| t5 | 0.718 | 0.974 |
| t6 | 0.875 | 0.989 |
| t7 | 0.993 | 1.000 |
| t8 | 0.664 | 0.971 |
| t9 | 0.945 | 0.966 |
| t10 | 0.957 | 0.990 |
| **mean** | **0.855** | **0.974** |

Remaining laggards: t4 (0.729) and t8 (0.664) — identity still creeps on these despite near-perfect detection.

## Identity mode stack

| Mode | within_r | Description |
|------|----------|-------------|
| `fpath` | **0.855** | Viterbi path integrator — current leader |
| `field_coh` | 0.773 | `field_lag` + coherence far-jump override |
| `field_lag` | 0.746 | Fixed-lag confirmation smoother (K=8) over `field` |
| `field` | 0.741 | Motion-saliency peak tracker + YOLO snap |

**field** per frame: `motion.py` fits global rigid sheet motion (RANSAC), treats outlier features as evidence, blurs into a saliency field → `OutlierTracker` follows the peak with a gated CV model → snapped to the highest-saliency YOLO box. Snap is load-bearing (field raw ≈ 0.28, field+snap ≈ 0.59).

**field_coh** adds directional coherence of residual vectors as an escape-from-lock-in override — fires when a far challenger is persistently coherent for ≥8 frames. Relevant for variance reduction (lower worst-clip floor than fpath).

**Why fpath now leads:** the old regime split (fpath collapsed on t1/t5 with old detector) was a detection problem, not an integration problem. With oracle at 0.974, fpath's path memory becomes an asset — when the real shape is almost always in a box, Viterbi integration locks on and stays locked.

## Detection model history

| Model | Oracle | Notes |
|-------|--------|-------|
| original `yolov8n_combined` | 0.915 | First s* model; no edge examples; t1 oracle 0.806 (shape drifts to top edge, missed) |
| `yolov8n_combined` (edge re-annotated) | 0.901 | Added edge boxes single class — hurt t4/t5, net regression |
| `yolov8n_combined-5` (2-class) | — | shape_full/shape_partial split regressed overall; too few samples per class |
| **`yolov8n_single_combined`** | **0.974** | **Current best.** s1–s12 + 5 targeted t1 edge frames (top-left, f225–300), single class, collapsed partial→full labels. |

**Key lesson:** the two-class approach failed at ~55 frames — splitting halved effective instances per class. The right fix was targeted visual coverage of the specific failure position, not a label schema change.

## Physics

- Real shape speed: **median 1.3 px/frame, p90 4.5, p99 17.8, max 44.7**
- Shape radius: **~56 px**
- Failure mode: **CREEP not JUMP** — track is first lost via small steps (~12 px) onto an adjacent fake. Big teleports are `reacquire` firing *after* the track is already lost.

## How to train YOLO

Training data: `data/detect/s_frames/` (PNGs) + `data/detect/s_labels_single/` (YOLO .txt, single class). 60 frames across s1–s12 + 5 t1 edge frames, ~12 boxes/frame.

```bash
# 1. Build the dataset
python -m ld.detect.build_s_dataset \
    --labels-dir data/detect/s_labels_single \
    --out-dir data/detect/dataset_single_combined

# 2. Train (~12 min on M3 Pro)
python -m ld.detect.train \
    --data data/detect/dataset_single_combined/dataset.yaml \
    --name yolov8n_single_combined

# Weights: data/detect/runs/yolov8n_single_combined/weights/best.pt
```

Re-annotate or add frames:
```bash
python -m ld.detect.annotate_s --skip-extract   # re-annotate existing frames
python -m ld.detect.annotate_s                  # extract new frames + annotate
```

Annotator controls: `f`=full mode, `x`=partial mode, click existing box=toggle class, `t`=toggle last, `u`=undo, `n`=next, `p`=prev, `q`=quit.

Key training notes:
- Always use `yolov8n.pt` (pretrained COCO), not random init.
- The real shape ranks ~10–13 in confidence; **never filter by confidence**.
- Run with `.venv/bin/python` (Python 3.12 venv at project root).

## How to evaluate

```bash
# Full leaderboard across t1–t10 (~12 min)
python -m ld.detect.eval_modes --weights data/detect/runs/yolov8n_single_combined/weights/best.pt

# Single mode / subset (fast) — NOTE: overwrites LEADERBOARD.md
python -m ld.detect.eval_modes --weights .../best.pt --modes fpath --clips t4 t8

# Honest held-out LOO
python -m ld.detect.loo --weights .../best.pt
```

Per-frame trace CSVs: `data/detect/eval/<clip>__<mode>.csv`. Primary debugging tool — read to see how a track fails (creep vs jump, which state).

Always quote **LOO** numbers as the real metric. In-sample is optimistic.

## How to generate evidence videos

```bash
# All t1–t10 with fpath (current best)
python -m ld.detect.render_evidence \
    --weights data/detect/runs/yolov8n_single_combined/weights/best.pt \
    --mode fpath

# Other competitive modes
python -m ld.detect.render_evidence --weights .../best.pt --mode field_coh

# Output: data/detect/evidence/<clip>_<mode>.mp4
```

Overlay legend: green rectangles = YOLO detections, red circle = `track`, orange = `coast`, grey = `acquire`, cyan crosshair = GT, red line = miss (outside radius).

Excluded from evidence renders: `paper`, `paper_outlier`, `paper_outlier_rank`, `chain` — all well below competitive modes.

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

## Open avenues (highest-EV first)

0. **t4/t8 identity creep** — oracle is near-perfect (0.953/0.971) but within_r is 0.729/0.664. Pure identity failure despite good detection. Coherence saliency channel is the best unbuilt lever: fold coherent-mass into the saliency map so reacquire can teleport to the coherent peak.
1. **t1/t4 lock bug** — countdown lock lands on wrong shape. Fix in `compute_countdown_lock` / `_pick_lock_box`. Orthogonal to laggard wall.

## Conventions

- **Strictly causal** for anything shipped. Offline only for diagnostics/ceilings.
- All identity modes dispatch through `identity._dispatch_mode`. `ALL_MODES` is the single source of truth.
- The green crosshair in clips is **GT only** — `cursor.strip_pointer` inpaints it before any tracking.
- Accept a change only if it improves **LOO mean with no per-clip regression**.
- Eval is cache-backed; cache key is md5(path+mtime) of the weights file.

## Key files

| File | Role |
|------|------|
| `ld/detect/identity.py` | All identity modes; `_dispatch_mode`, `ALL_MODES`, `run_clip` |
| `ld/vision/motion.py` | `estimate_motion` (RANSAC), `saliency_map`, `MotionField.outlier_vectors` |
| `ld/track/tracker.py` | `OutlierTracker` — gated CV peak-follower with reacquire |
| `ld/vision/cursor.py` | GT crosshair detect + `strip_pointer` inpainting |
| `ld/detect/eval_modes.py` | Leaderboard harness → `LEADERBOARD.md` |
| `ld/detect/loo.py` | Leave-one-clip-out honest generalization |
| `ld/detect/render_evidence.py` | Per-clip overlay video renderer |
| `ld/detect/annotate_s.py` | Frame extraction + drag-to-draw box annotator |
| `ld/detect/build_s_dataset.py` | Builds YOLO dataset from s_frames/s_labels |
| `ld/detect/train.py` | Fine-tunes YOLOv8n from COCO-pretrained weights |
| `ld/config.py` | Tracker / detection / motion tunables |
