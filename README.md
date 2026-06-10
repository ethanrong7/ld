# LD Solver

Computer-vision solver for MapleStory Lie Detector clips.

## Approach: rigid-paper outlier tracking

The real and fake shapes are visually identical (same embossed outline), so
**appearance cannot tell them apart — only motion can**. The background "sheet"
moves as one near-rigid body, so every fake shares a single global transform;
the real shape moves independently (its own translation, plus rotation that the
fakes never have).

The solver exploits this at the **feature-point level**:

1. **Acquire** — follow the white countdown shape to seed a position and scale
   (no crosshair involved).
2. **Estimate sheet motion** — track features between frames and fit the global
   rigid transform with RANSAC. Inliers move *with* the paper (fakes/background).
3. **Outlier saliency** — features that *disagree* with that transform are the
   independently-moving real shape; their spatial density is the per-frame
   signal.
4. **Track** — a recursive constant-velocity tracker gates the saliency around
   its prediction, coasts when the shape slows (and stops shedding outliers),
   and re-acquires from the global saliency peak when it falls off-track.

This is point-based, so it is immune to the edge/registration noise that defeats
pixel-level background subtraction. The green crosshair is **only** ever read to
score accuracy, never as a solver input.

See `output/debug/*_evidence.mp4` (green = moves-with-paper, red = independent /
real shape, heat = saliency, cyan = estimate, yellow = GT for reference only).

### Status

Causal and cursor-free. On the `t*` eval clips the estimate lands within
~1.5× the shape radius for the large majority of frames (median ~50–75px,
shape radius ~50–60px). Closing the gap to "inside the shape every frame"
is the next phase (outlier-tracklet linking, a particle filter, and fusing
the independent-rotation cue for non-circular shapes).

## Pointer stripping

Kept from the earlier work and used on every frame before processing —
inpainting the green GT crosshair (and optionally a live mouse disk) so the
solver never sees a moving cursor beacon.

## Pointer stripping

`ld/vision/cursor.py`:

- `strip_pointer(frame)` — inpaint green crosshair pixels (t* eval clips)
- `find_cursor(frame)` — locate green GT centroid (scoring only, never tracking input)
- `mouse_xy` kwarg — inpaint a disk at the live cursor position

Tunables in `ld/config.py`: `GREEN_*`, `POINTER_INPAINT_RADIUS`, `POINTER_RADIUS`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
set PYTHONPATH=.
```

Clips live in `data/` (gitignored). Test clips: `data/t*_cropped_trimmed.mp4`.

## Modules

- `ld/vision/motion.py` — rigid sheet-motion + independent-motion saliency
- `ld/vision/countdown.py` — white-shape acquisition (seed position + radius)
- `ld/track/tracker.py` — recursive gated tracker with re-acquisition
- `ld/solve.py` — acquisition + tracking pipeline (per-frame track)
- `ld/eval/score.py` — accuracy vs the green GT (evaluation only)
- `ld/debug/evidence.py` — overlay video explaining the signal

Tunables live in `ld/config.py` (`FEAT_*`, `OUTLIER_*`, `SALIENCY_*`, gate /
velocity / re-acquire settings, `WHITE_*` acquisition, `GREEN_*` GT).

## CLI

```bash
# original | stripped side-by-side (pointer stripping sanity check)
python -m ld.main strip-preview --input data/t1_cropped_trimmed.mp4

# run the solver -> per-frame track CSV (frame,x,y,confidence,state)
python -m ld.main solve --input data/t1_cropped_trimmed.mp4

# run the solver and score it against the green GT
python -m ld.main score --input data/t1_cropped_trimmed.mp4

# render the rigid-outlier overlay video
python -m ld.main evidence --input data/t1_cropped_trimmed.mp4
```
