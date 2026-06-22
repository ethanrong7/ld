# Repo cleanup — completed 2026-06-22

The repo was pared down to do exactly two jobs, plus the shipped core they depend on.
Everything experimental/superseded was removed.

## What the repo does now

**Job 1 — train on new data** (single-class YOLO):
`ld.detect.annotate` → `ld.detect.build_dataset` → `ld.detect.train`
(see README "Job 1"). `annotate` auto-discovers new `data/*.mp4`, crops raw captures
to the 744×498 board (`ld.detect.board_crop`), extracts 5 in-play frames per video
(no countdown / START / success frames — the longest run clean of bright overlays),
opens the single-class annotator, then rebuilds the dataset and **prints** (never runs)
the train command.

**Job 2 — leaderboard + evidence**:
`ld.detect.eval_modes` → `ld/detect/LEADERBOARD.md` (fpath_human, fpath, oracle row);
`ld.detect.render_evidence --mode fpath_human` → overlay video with YOLO boxes + the
red-dot cursor guess.

Shipped detector: `yolov8n_single_combined`. Shipped identity: `fpath_human`.

## What changed in the cleanup

- **identity.py**: `BOARD_MODES = (fpath, fpath_human)`; `ALL_MODES` = the `fpath`
  lineage only. Removed the field/paper/accum/chain/`fpath_reacq` families, the
  duplicate `_render_evidence`, and ~1300 lines of dead helpers/constants. The
  intermediate `fpath_*` stages stay runnable (`--modes <name>`) so the tuning probes
  can regenerate their per-mode eval CSVs.
- **Old OutlierTracker solver removed**: `ld/main.py`, `ld/solve.py`, `ld/debug/`,
  `ld/eval/`, `ld/track/tracker.py`, `ld/detect/loo.py`, `ld/detect/outlier_track.py`,
  `ld/detect/viterbi_ceiling.py`, `ld/detect/diagnose.py`. `fusion.py` was stripped to
  `detect_fusion_clip` + `FusionPack` (the only parts the pipeline uses).
- **Probes**: 16 dead probes deleted. The 5 tuning probes for the kept method
  (`cursor_physics_probe`, `resid_freeze_probe`, `hedge_probe`, `fuse_sweep`,
  `exp3_sweep`) were kept; shared helpers they pulled from now-deleted probes were
  moved into `ld/detect/probe_common.py`.
- **Training pipeline unified to single-class**: replaced `annotate.py`/`annotate_s.py`
  /`sample_frames.py` with one `annotate.py`; replaced `build_s_dataset.py`/
  `build_dataset.py`/`build_combined_dataset.py` with one single-class `build_dataset.py`;
  added `board_crop.py` (shared with `make_additional_evidence.py`). Canonical training
  paths now live in `ld/config.py` (`TRAIN_*`).
- **Docs**: consolidated to `CLAUDE.md` + `README.md` + the two `LEADERBOARD*.md`.
  Deleted `experiment.md`, `DIAGNOSIS.md`, `VITERBI_CEILING.md`, `LEADERBOARD_t1_t10.md`,
  `ld/detect/README.md`.

## Pending / optional

- **data/ housekeeping** (not done — needs confirmation): stale dataset/label dirs that
  the shipped model doesn't use can be removed once verified — `data/detect/dataset_s`,
  `data/detect/dataset`, `data/detect/dataset_combined`, `data/combined*`,
  `data/detect/frames` + `labels`, `data/detect/s_labels` (2-class). Keep
  `dataset_single_combined`, `s_frames`, `s_labels_single`, all raw `data/*.mp4`, and
  `data/additional_evidence/`.
- Re-run `eval_modes` against the trained weights to regenerate `LEADERBOARD.md` with the
  new 2-mode + oracle layout (cached detections make this fast).
