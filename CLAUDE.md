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

1. **Detection** — YOLOv8n (`data/detect/runs/yolov8n_single_combined/weights/best.pt`) finds candidate shape boxes per frame. Oracle within_r ≈ **0.958** (mean over t1–t10). Detection is effectively solved. The real shape is camouflaged and ranks ~10–13 in confidence; you cannot filter by confidence without losing it.

2. **Identity** — decides *which* box is the real shape over time. This is the bottleneck and the entire focus of remaining work. Code in `ld/detect/identity.py`.

## Current best: `fpath_coh` (0.825 within_r)

`fpath_coh` is the leader (verified `eval_modes` regen, 2026-06-19, against `yolov8n_single_combined`). It is the `fpath` Viterbi path integrator with a coherence-bumped emission (`mass*(1+1.8*coh)`). Oracle mean is **0.958** — detection is solved; every remaining loss is identity creep onto an adjacent fake on frames where the real box *is* present.

| clip | within_r | oracle |
|------|----------|--------|
| t1 | 0.702 | 0.952 |
| t2 | 0.955 | 0.978 |
| t3 | 0.810 | 0.918 |
| t4 | 0.745 | 0.944 |
| t5 | 0.617 | 0.948 |
| t6 | 0.877 | 0.970 |
| t7 | 0.991 | 0.998 |
| t8 | 0.732 | 0.939 |
| t9 | 0.909 | 0.973 |
| t10 | 0.915 | 0.964 |
| **mean** | **0.825** | **0.958** |

Laggards: **t5 (0.617), t1 (0.702), t8 (0.732), t4 (0.745)** — pure identity creep (oracle ~0.95 on all). An oracle router over *all existing modes* caps at **0.858**, so reaching ≥0.90 needs a better identity signal, not ensembling.

**Failure taxonomy (fuse_probe, 2026-06-19).** On the frames `fpath_coh` misses, which channel ranks the real box #1 splits by clip — no single channel wins everywhere (the persistent causal-key wall):
- **t4 (coh #1 0.60), t5 (curl #1 0.50)** — *emission-channel* failures: a richer per-frame emission (directional coherence / rotational curl) ranks the real box higher than saliency-mass does. Addressable by a better emission.
- **t1 (mass #1 0.41, coh demotes it), t8 (mass #1 0.32)** — *path lock-in* failures: mass already ranks the real box in top-3 ~79%, but the Viterbi path won't switch off the fake it locked onto earlier. Needs a lock-in escape, not a new channel. (This is why `fpath_coh` t1 ≈ `fpath` t1 — coherence doesn't help t1.)
- **Max-fusion of channels is a dead end** (fuse_probe top-1 0.22): max-combining amplifies whichever channel spikes on a fake. Use additive/weighted sums, never max.
- Coherence is the best *candidate-generator*: top-3 recall on miss frames 0.842 vs mass 0.803.

## Identity mode stack

| Mode | within_r | Description |
|------|----------|-------------|
| `fpath_coh` | **0.825** | `fpath` + coherence-bumped emission (`mass*(1+1.8*coh)`) — current leader |
| `fpath` | 0.797 | Viterbi path integrator, pure saliency-mass emission |
| `fpath_reacq` | 0.783 | `fpath` + global-mass reacquire + transition cap (net regression) |
| `field_coh` | 0.773 | `field_lag` + coherence far-jump override |
| `field_lag` | 0.749 | Fixed-lag confirmation smoother (K=8) over `field` |
| `field` | 0.744 | Motion-saliency peak tracker + YOLO snap |

**fpath / fpath_coh** per frame: a causal forward trellis over YOLO boxes; emission = normalized saliency-mass (× coherence bump for `fpath_coh`), transition penalty = `(jump_in_radii)^2`. Decode = argmax cumulative score → that box's centroid feeds a CV predictor. Locks on and stays locked — an asset on strong clips (t2/t6/t7/t9/t10), a liability on lock-in laggards (t1/t8).

**field** per frame: `motion.py` fits global rigid sheet motion (RANSAC), treats outlier features as evidence, blurs into a saliency field → `OutlierTracker` follows the peak with a gated CV model → snapped to the highest-saliency YOLO box. Snap is load-bearing (field raw ≈ 0.28, field+snap ≈ 0.59). The `field` family beats `fpath_coh` on t1 (0.84 vs 0.70) — no path memory to lock in.

**field_coh** adds directional coherence of residual vectors as an escape-from-lock-in override — fires when a far challenger is persistently coherent for ≥8 frames.

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

Target: ≥0.90 mean within_r (from 0.825). Diagnosis below from `fuse_probe` (2026-06-19). `ld/detect/fuse_probe.py` is the gate that produced the taxonomy.

0. **Richer additive emission for the `fpath` trellis (t4/t5 lever).** t4/t5 are emission-channel failures: coherence ranks the real box #1 on 60% of t4 misses, curl on 50% of t5 misses, vs mass ~43%. Replace `fpath_coh`'s multiplicative `mass*(1+λcoh)` with a **weighted additive** emission `norm(mass) + w_c·norm(coh) + w_r·norm(curl)` (NOT max — max-fusion is a dead end, fuse_probe top-1 0.22), swept by LOO. Coherence top-3 recall on miss frames is 0.842 (mass 0.803) — the candidate set is there for Viterbi to resolve.
1. **Lock-in escape for the trellis (t1/t8 lever).** t1/t8 are NOT emission failures — mass already ranks the real box top-3 ~79%, but the path won't switch off the fake it locked onto. Needs a coherence/curl-driven reacquire (teleport + reset path memory when off the coherent peak ≥K frames) or adaptive transition softening on low-confidence frames. The `field` family already beats `fpath_coh` on t1 (0.84 vs 0.70) precisely because it has no path memory — confirms lock-in is the cause.
2. **t1/t4 lock bug** — countdown lock lands on wrong shape. Fix in `compute_countdown_lock` / `_pick_lock_box`. Orthogonal to laggard wall.

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
