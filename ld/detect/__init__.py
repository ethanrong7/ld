"""YOLO detection + identity tracking for the LD solver.

Two jobs live here:

TRAINING (drop a video in data/ -> retrained detector):
  1. ``annotate``      -- discover new data/*.mp4, crop to the board, extract 5
     in-play frames (no countdown/START/success), and single-class box-annotate.
  2. ``build_dataset`` -- arrange frames/labels into a YOLO tree (clip-wise
     train/val split) and emit ``dataset.yaml``; prints the train command.
  3. ``train``         -- fine-tune yolov8n with heavy augmentation (scarce data).
  4. ``infer``         -- run trained weights on held-out frames for a by-eye check.

EVALUATION / EVIDENCE (compare identity methods, render the result):
  5. ``eval_modes``     -- score identity modes across t1..t10 -> LEADERBOARD.md.
  6. ``render_evidence``-- overlay video: YOLO boxes + the red-dot cursor guess.

The shipped detector is single-class; the shipped identity method is ``fpath_human``
(see ``identity.ALL_MODES``). The green crosshair is only a labelling/eval GT,
never a detector input.
"""
