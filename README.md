# Lie Detector Solver

Classical CV pipeline for the MapleStory Lie Detector mini-game: detect ring blobs, lock the initial white shape as **REAL**, track by persistent ID, and target its centroid.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Test clips live under `data/` as `t*_cropped_trimmed.mp4` (pre-cropped LD panel, one round per file). Large videos are gitignored; keep them locally.

## Phase 0 — Validate test data

Probe all trimmed test clips (resolution, fps, frame count, duration):

```bash
python -m ld.debug.probe_videos
```

Optional: save first/mid-frame thumbnails and CSV:

```bash
python -m ld.debug.probe_videos --samples --csv output/phase0_probe.csv
```

Or via main:

```bash
python -m ld.main probe
```

## Project layout

```
ld/                  # Python package
  config.py          # thresholds and paths
  capture/           # video / screen sources (Phase 1+)
  vision/            # detect, track, overlay (Phase 2+)
  control/           # mouse (Phase 7)
  debug/             # offline tools
data/                # local test videos (not committed)
output/              # generated debug runs (gitignored)
```

## Phase 1 — Offline passthrough

Read a test clip, draw frame index / timestamp HUD, write annotated MP4:

```bash
python -m ld.debug.run_offline --input data/t1_cropped_trimmed.mp4 --output output/t1_debug.mp4
```

Or via main:

```bash
python -m ld.main offline --input data/t1_cropped_trimmed.mp4 --output output/t1_debug.mp4
```

Quick smoke test (first 60 frames):

```bash
python -m ld.debug.run_offline --max-frames 60 --output output/smoke.mp4
```

Expect processing ≥30 fps on trimmed clips. Output goes to `output/` (gitignored).

## Phase 2 — Shape detection (all entities)

Find **every ring** as a bbox: **1 REAL + ~10–12 decoys**, all **same fixed size**, decoys at **stable slot positions**, REAL boxed via white-blob then moving localization. See [docs/phase2.md](docs/phase2.md) for full spec.

```bash
python -m ld.debug.run_offline --input data/t1_cropped_trimmed.mp4 --output output/t1_phase2.mp4 --snapshots
```

**Pass if:** ~10–14 cyan boxes on typical frames; white shape boxed at start; decoy boxes **don’t drift or resize**; one box **moves** with REAL. Snapshots in `output/phase2_snapshots/`.

## Next: Phase 3

Initialize **REAL** as the single white blob on the first bright frame; green box + persistent ID.
