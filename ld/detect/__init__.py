"""Detector-path viability tooling for the LD solver.

A lightweight test of the pivot from motion-saliency guessing to multi-object
detection + rigid-constellation tracking. The question this package answers is
narrow and cheap: *can a YOLOv8-nano, fine-tuned on a handful of hand-labelled
frames, cleanly box every shape instance on held-out frames?*

Workflow (each step is a runnable module; see ``ld/detect/README.md``):

  1. ``sample_frames``  -- pull diverse, cursor-stripped frames across t1..t10,
     writing a manifest with the green-GT location + countdown radius per frame.
  2. ``annotate``       -- a minimal OpenCV box editor. The real shape's box is
     pre-filled from the GT crosshair; the human only adds the fake boxes.
  3. ``build_dataset``  -- arrange images/labels into a YOLO tree (clip-wise
     train/val split to avoid leakage) and emit ``dataset.yaml``.
  4. ``train``          -- fine-tune yolov8n with heavy augmentation (data is
     scarce: ~10 clips only).
  5. ``infer``          -- run the trained weights on held-out frames and render
     boxed overlays for a by-eye verdict.

Nothing here touches the green crosshair except as a labelling convenience and
(downstream) as evaluation GT -- never as a detector input.
"""
