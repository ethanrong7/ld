# LD — MapleStory "Lie Detector" shape tracker

MapleStory's Lie Detector minigame shows a sheet covered in **identical** shapes.
One is **real** (you must click it); the rest are **fakes**. Fakes move rigidly with
the sheet; the real shape moves independently. This repo takes a screen-capture clip
and outputs the pixel position of the real shape every frame.

Two stages:

1. **Detection** — a single-class YOLOv8n boxes every shape candidate per frame
   (`yolov8n_single_combined`, oracle ≈ 0.958 within-radius).
2. **Identity** — decides *which* box is the real shape over time. The shipped
   method is **`fpath_human`** (≈ 0.940 within-radius): a causal Viterbi trellis over
   the boxes with a decode-layer freeze + human-cursor output filter. See
   [CLAUDE.md](CLAUDE.md) for the full method, lineage, and dead-ends.

Everything is **strictly causal** (no future frames) so it ports to live play. The
green crosshair is only ever read as evaluation GT — it is inpainted out
(`strip_pointer`) before any detection and is never a tracker input.

## Setup

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
```bash
# Mac/Linux
source .venv/bin/activate
.venv/bin/python -m pip install -r requirements.txt
```

Trained weights are **not** committed — train locally (see below). They land at
`data/detect/runs/yolov8n_single_combined/weights/best.pt`.

---

## Job 1 — Train on new data

Drop one or more raw capture videos into `data/`, then run the three steps. The first
does discovery + cropping + frame extraction + labelling in one interactive session.

```powershell
# 1. Extract + label. Auto-discovers new data/*.mp4, crops raw captures to the
#    744x498 board, pulls 5 in-play frames each (no countdown/START/success frames),
#    and opens the single-class box annotator. On quit it rebuilds the dataset and
#    PRINTS the train command (it never trains automatically -- verify your labels first).
.venv\Scripts\python.exe -m ld.detect.annotate

# 2. (re)build the YOLO dataset on its own if needed
.venv\Scripts\python.exe -m ld.detect.build_dataset

# 3. Train (only after you've eyeballed the labels)
.venv\Scripts\python.exe -m ld.detect.train --name yolov8n_single_combined
```

Annotator controls: **drag** = draw a box · **u** = undo · **c** = clear ·
**n/space** = next · **p** = prev · **s** = save · **q** = quit. Box *every* shape
on the sheet (the real one is camouflaged among the fakes; identity is decided
downstream, not by the labels).

Artifacts: frames → `data/detect/s_frames/`, labels → `data/detect/s_labels_single/`,
dataset → `data/detect/dataset_single_combined/`.

Quick by-eye check of a trained model:
```powershell
.venv\Scripts\python.exe -m ld.detect.infer --weights data/detect/runs/yolov8n_single_combined/weights/best.pt
```

---

## Job 2 — Leaderboard + evidence video

```powershell
# Leaderboard across the t1..t10 eval clips -> ld/detect/LEADERBOARD.md
# Scores fpath_human (leader), fpath (ablation base), and the oracle ceiling.
.venv\Scripts\python.exe -m ld.detect.eval_modes --weights data/detect/runs/yolov8n_single_combined/weights/best.pt

# Evidence video: YOLO boxes (green) + the red-dot cursor guess + GT crosshair (cyan)
.venv\Scripts\python.exe -m ld.detect.render_evidence --weights data/detect/runs/yolov8n_single_combined/weights/best.pt --mode fpath_human
# -> data/detect/evidence/<clip>_fpath_human.mp4
```

Run a single mode/clip subset: `eval_modes --modes fpath fpath_human --clips t4 t8`.
The full `fpath` lineage (`fpath`, `fpath_coh`, `fpath_fuse`, `fpath_hyst`,
`fpath_hedge`, `fpath_freeze`, `fpath_human`) stays runnable by name for retuning; the
default leaderboard reports the leader + ablation base + oracle.

A second, never-trained-on validation set lives in `data/additional_evidence/` (built
by `make_additional_evidence.py`); its board is `ld/detect/LEADERBOARD_additional_evidence.md`.

---

## Layout

| Path | Role |
|------|------|
| `ld/detect/annotate.py` | discover + crop + extract + single-class annotate |
| `ld/detect/build_dataset.py` | build single-class YOLO dataset |
| `ld/detect/train.py` | fine-tune YOLOv8n |
| `ld/detect/board_crop.py` | board detection + crop (shared with `make_additional_evidence.py`) |
| `ld/detect/identity.py` | identity tracker (`fpath`…`fpath_human`), `_dispatch_mode` |
| `ld/detect/eval_modes.py` | leaderboard harness → `LEADERBOARD.md` |
| `ld/detect/render_evidence.py` | overlay video (boxes + red-dot guess) |
| `ld/detect/fusion.py` | cached YOLO detection (`detect_fusion_clip`) |
| `ld/vision/`, `ld/track/` | motion/saliency, cursor strip, human-cursor output filter |

Tunables live in `ld/config.py`. See **[CLAUDE.md](CLAUDE.md)** for the method, the
experiment history, and the dead-ends.
