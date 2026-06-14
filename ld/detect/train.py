"""Fine-tune YOLOv8-nano on the hand-labelled frames.

This is a *viability probe*, not a production train. Defaults are tuned for the
scarce-data regime (only ~10 clips): start from COCO-pretrained yolov8n, train
at full clip resolution, and lean on heavy augmentation (mosaic, scale, flips,
HSV jitter) to manufacture variety the 10 clips don't have on their own.

Read the verdict off the val mAP printed at the end and the curves under the
run dir. If val mAP@0.5 is high and ``infer`` boxes look clean, the detector
path is viable. If not -- and augmentation didn't rescue it -- the blocker is
data volume, and the next move is synthetic generation (composite shapes onto
sampled sheet backgrounds) before revisiting.

Requires ``ultralytics`` (see requirements-detect.txt). Imported lazily so the
rest of ld.detect stays importable without the heavy dep.

Usage:
    python -m ld.detect.train
    python -m ld.detect.train --epochs 150 --imgsz 768 --device mps
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ld.config import DETECT_DATASET_DIR, DETECT_RUNS_DIR

__all__ = ["train"]

# Augmentation profile for a handful of near-identical clips. Geometric +
# photometric jitter is the cheapest proxy for "more clips" we have.
AUG = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=180.0,        # shapes appear at arbitrary rotation
    translate=0.1, scale=0.5, shear=2.0,
    flipud=0.5, fliplr=0.5,
    mosaic=1.0, mixup=0.1,
)


def train(data_yaml: Path = DETECT_DATASET_DIR / "dataset.yaml",
          model: str = "yolov8n.pt",
          epochs: int = 100, imgsz: int = 768, batch: int = 8,
          device: str | None = None, name: str = "yolov8n_probe") -> Path:
    from ultralytics import YOLO  # lazy: heavy (torch) dependency

    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise SystemExit(f"No dataset at {data_yaml}. Run ld.detect.build_dataset first.")

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs, imgsz=imgsz, batch=batch,
        device=device,                       # None -> ultralytics auto (cuda/mps/cpu)
        project=str(DETECT_RUNS_DIR), name=name,
        patience=30, seed=0, deterministic=True,
        **AUG,
    )
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else DETECT_RUNS_DIR / name
    best = save_dir / "weights" / "best.pt"
    print(f"\nbest weights -> {best}")
    print("Inspect: results.png, val_batch*_pred.jpg under the run dir.")
    print("Then: python -m ld.detect.infer --weights", best)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8n on labelled LD frames")
    ap.add_argument("--data", default=str(DETECT_DATASET_DIR / "dataset.yaml"))
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--name", default="yolov8n_probe")
    args = ap.parse_args()
    train(Path(args.data), args.model, args.epochs, args.imgsz, args.batch,
          args.device, args.name)


if __name__ == "__main__":
    main()
