# Detector-path viability test (`ld/detect`)

A cheap, throwaway-able probe of the pivot: from motion-saliency guessing to
**multi-object detection + rigid-constellation tracking**. One question only:

> Can a YOLOv8-nano, fine-tuned on ~15–25 hand-labelled frames, cleanly box
> *every* shape instance on held-out clips?

If yes, the detector path is worth building out. If no — and augmentation
doesn't rescue it — the blocker is data volume (only ~10 clips), and the next
move is synthetic frame generation before revisiting.

This is a **validation harness, not the production pipeline.** It does not yet
do constellation tracking or real-shape inference; it answers the prerequisite
"does detection even work" question.

## Environment

The repo code uses `X | None` runtime unions (needs Python ≥3.10). System
`python3` is 3.9, so use the dedicated venv:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt          # opencv + numpy (sampler/annotate)
.venv/bin/pip install -r requirements-detect.txt   # ultralytics (train/infer; heavy)
export PYTHONPATH=.
```

`ultralytics` is imported lazily, so steps 1–3 work with just opencv+numpy.

## Workflow

```bash
# 1. Sample diverse, cursor-stripped frames across t1..t10 (+ a GT manifest).
.venv/bin/python -m ld.detect.sample_frames --per-clip 2

# 2. Label them. The REAL shape's box is pre-filled from the green GT crosshair;
#    you only draw the FAKE shapes' boxes (drag), then n/space to advance.
.venv/bin/python -m ld.detect.annotate

# 3. Assemble a YOLO dataset (clip-wise train/val split -> tests unseen clips).
.venv/bin/python -m ld.detect.build_dataset            # auto-holds out ~20% of clips

# 4. Fine-tune yolov8n (heavy augmentation; downloads yolov8n.pt on first run).
.venv/bin/python -m ld.detect.train --device mps       # mps on Apple Silicon

# 5. Run on the held-out clips and eyeball the boxes.
.venv/bin/python -m ld.detect.infer \
    --weights data/detect/runs/yolov8n_probe/weights/best.pt
```

## Reading the verdict

- **`train`** prints val mAP@0.5 and writes curves + `val_batch*_pred.jpg`.
- **`infer`** prints boxes-per-frame stats and saves boxed overlays. A healthy
  constellation of N shapes should yield ~N boxes per frame, consistently,
  without flooding the sheet with false positives.

## Known risk (seen in the first sample)

The sheet is a **dense, overlapping near-tiling of near-identical embossed
shapes**, not sparse discrete objects. Two consequences to confirm during
labelling:

1. **Labelling cost/ambiguity** — "box every shape" may mean dozens of
   overlapping, ambiguous boxes per frame. Decide a consistent rule (e.g. only
   box clearly-separable shape *centres*) before starting, or the labels will be
   noisy and the mAP meaningless.
2. **NMS / dense-object difficulty** — YOLO + NMS struggles with many
   overlapping near-identical instances. If recall is poor, that's a signal
   about the approach, not just the data.

## Artifacts

Everything lives under `data/detect/` (gitignored except the hand-made
`labels/` and `manifest.json`, since frames are reproducible from the mp4s):

```
data/detect/
  frames/        sampled cursor-stripped PNGs        (gitignored)
  labels/        YOLO .txt labels (your work)        (kept)
  manifest.json  per-frame clip/frame/GT/radius      (kept)
  dataset/       built YOLO tree + dataset.yaml      (gitignored)
  runs/          ultralytics outputs + weights       (gitignored)
```
