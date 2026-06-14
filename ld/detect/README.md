# Detector path (`ld/detect`)

YOLO-based detection + identity tracking for Lie Detector clips. YOLO boxes every
shape; `identity` picks which box is the real one using paper-motion residuals.

## Clone and get working

Requires **Python ≥ 3.10** (3.12 recommended). macOS/Linux and Windows both work.

```bash
git clone <repo-url> ld && cd ld

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-detect.txt   # ultralytics + torch
export PYTHONPATH=.                                  # add to your shell profile
```

On Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-detect.txt
$env:PYTHONPATH = "."
```

### What is (and isn't) in git

| In git | Local only (gitignored) |
|--------|---------------------------|
| `data/combined*` — hand-labelled JSON + frame PNGs | `data/t*_cropped_trimmed.mp4` — eval clips |
| `data/detect/dataset_combined/` — YOLO train/val tree | `data/detect/runs/` — trained weights |
| `data/detect/labels/` + `manifest.json` — older probe labels | `data/detect/cache/`, `evidence/` — derived outputs |
| All `ld/` source code | `yolov8n.pt` — auto-downloaded on first train |

**You must supply your own mp4 clips.** Drop them in `data/` with the naming
convention `t1_cropped_trimmed.mp4` … `t10_cropped_trimmed.mp4` (same layout as
the eval set used during development).

### Train the model (first time)

The committed dataset is ready to train; no labelling step required for a fresh
clone:

```bash
.venv/bin/python -m ld.detect.train \
  --data data/detect/dataset_combined/dataset.yaml \
  --name yolov8n_combined \
  --device mps          # Apple Silicon; use cuda or cpu elsewhere
```

This downloads `yolov8n.pt` on first run (~6 MB) and writes weights to
`data/detect/runs/yolov8n_combined/weights/best.pt`. Training takes a few
minutes on MPS.

To rebuild the dataset from the committed `data/combined*` annotations:

```bash
.venv/bin/python -m ld.detect.build_combined_dataset
```

## Generate a labelled video with bounding boxes

Two commands produce overlay mp4s on a `t*` clip. Both strip the green GT
crosshair before inference (solver never sees it).

### 1. Raw YOLO detections (every shape boxed)

```bash
.venv/bin/python -m ld.detect.infer \
  --weights data/detect/runs/yolov8n_combined/weights/best.pt \
  --clip data/t1_cropped_trimmed.mp4
```

Output: `data/detect/runs/t1_cropped_trimmed_yolov8n_clip.mp4`

- Green rectangles on every detected shape
- Confidence score above each box
- HUD: frame index + detection count

### 2. Identity tracking (real shape highlighted)

```bash
.venv/bin/python -m ld.detect.identity \
  --weights data/detect/runs/yolov8n_combined/weights/best.pt \
  --inputs data/t1_cropped_trimmed.mp4 \
  --evidence
```

Output: `data/detect/evidence/t1_cropped_trimmed_identity.mp4`

- **Green thick box** — shape track chosen as the real one
- **Orange thin boxes** — other detected shapes (fakes)
- **Yellow cross** — ground-truth marker (for scoring only, not used by tracker)
- **Red circle** — identity track position
- HUD: frame index, detection count, locked track id

Run all eval clips at once (omit `--inputs`):

```bash
.venv/bin/python -m ld.detect.identity \
  --weights data/detect/runs/yolov8n_combined/weights/best.pt \
  --evidence
```

Prints per-clip accuracy (`within_r`, median px error) and writes one evidence
mp4 per clip under `data/detect/evidence/`.

## Full workflow (from scratch)

If you want to label new frames and retrain instead of using the committed
dataset:

```bash
# 1. Sample cursor-stripped frames from your mp4s.
.venv/bin/python -m ld.detect.sample_frames --per-clip 2

# 2. Label fake-shape boxes (real shape pre-filled from green crosshair).
.venv/bin/python -m ld.detect.annotate

# 3. Build YOLO dataset (clip-wise train/val split).
.venv/bin/python -m ld.detect.build_dataset

# 4. Fine-tune.
.venv/bin/python -m ld.detect.train --device mps

# 5. Validate on held-out frames.
.venv/bin/python -m ld.detect.infer \
  --weights data/detect/runs/yolov8n_probe/weights/best.pt
```

## Reading the verdict

- **`train`** — val mAP@0.5, curves, `val_batch*_pred.jpg` under the run dir.
- **`infer`** — boxes-per-frame stats; ~N consistent boxes per frame is healthy.
- **`identity`** — `within_r` (fraction of frames inside shape radius), median px
  error vs GT crosshair.

## Known risk

The sheet is a dense near-tiling of near-identical embossed shapes. YOLO + NMS
struggles with many overlapping instances; if recall is poor, the blocker is
likely data volume, not just tuning.

## Artifacts layout

```
data/detect/
  frames/           sampled PNGs                         (gitignored)
  labels/           YOLO .txt labels                     (kept)
  manifest.json     per-frame clip/frame/GT/radius       (kept)
  dataset/          probe YOLO tree                      (gitignored)
  dataset_combined/ combined-annotation YOLO tree        (kept)
  runs/             weights + training curves            (gitignored)
  cache/            fusion/detection JSON caches         (gitignored)
  evidence/         identity overlay mp4s                (gitignored)
```
