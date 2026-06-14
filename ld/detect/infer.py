"""Run trained YOLOv8n weights on held-out frames and render the verdict.

Two things to eyeball:
  1. The overlay PNGs (boxes drawn) -- does it cleanly box *every* shape, not
     just the most salient one, without flooding the sheet with false boxes?
  2. The per-image box counts -- a constellation of N shapes should yield ~N
     boxes per frame, consistently across frames.

By default it runs on the val split of the built dataset (the held-out clips),
which is the honest test. Point ``--source`` at any folder of PNGs to try ad-hoc
frames.

Requires ``ultralytics``. Usage:
    python -m ld.detect.infer --weights data/detect/runs/yolov8n_probe/weights/best.pt
    python -m ld.detect.infer --weights .../best.pt --source data/detect/frames --conf 0.25
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ld.config import DETECT_DATASET_DIR, DETECT_RUNS_DIR

__all__ = ["infer"]


def infer(weights: Path, source: Path, conf: float = 0.25,
          imgsz: int = 768, name: str = "yolov8n_infer") -> Path:
    from ultralytics import YOLO  # lazy: heavy dependency

    weights = Path(weights)
    if not weights.exists():
        raise SystemExit(f"No weights at {weights}. Run ld.detect.train first.")

    yolo = YOLO(str(weights))
    results = yolo.predict(
        source=str(source), conf=conf, imgsz=imgsz,
        save=True, project=str(DETECT_RUNS_DIR), name=name, exist_ok=True,
    )
    counts = [len(r.boxes) for r in results]
    out_dir = DETECT_RUNS_DIR / name
    if counts:
        import statistics
        print(f"frames={len(counts)} boxes/frame: "
              f"min={min(counts)} median={int(statistics.median(counts))} max={max(counts)} "
              f"mean={statistics.mean(counts):.1f}")
    print(f"overlays -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run YOLOv8n on held-out frames")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default=str(DETECT_DATASET_DIR / "images" / "val"),
                    help="image folder (default: dataset val split = held-out clips)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--name", default="yolov8n_infer")
    args = ap.parse_args()
    infer(Path(args.weights), Path(args.source), args.conf, args.imgsz, args.name)


if __name__ == "__main__":
    main()
