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

from ld.config import DETECT_RUNS_DIR, TRAIN_DATASET_DIR
from ld.capture.video_source import VideoSource, open_writer
from ld.vision.cursor import strip_pointer

__all__ = ["infer", "infer_clip"]


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


def infer_clip(weights: Path, clip: Path, *, conf: float = 0.25,
               imgsz: int = 768, name: str = "yolov8n_clip",
               strip_green: bool = True) -> Path:
    """Run YOLO on every frame of a clip and write a box overlay mp4."""
    import statistics

    import cv2
    from ultralytics import YOLO

    weights = Path(weights)
    clip = Path(clip)
    if not weights.exists():
        raise SystemExit(f"No weights at {weights}")
    if not clip.exists():
        raise SystemExit(f"No clip at {clip}")

    yolo = YOLO(str(weights))
    out = DETECT_RUNS_DIR / f"{clip.stem}_{name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    src = VideoSource(clip)
    writer = open_writer(out, src.meta.width, src.meta.height, src.meta.fps or 30.0)
    counts: list[int] = []
    for _idx, raw in src.frames():
        frame = strip_pointer(raw, strip_green=strip_green)
        res = yolo.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        n = 0
        if res.boxes is not None and len(res.boxes):
            n = len(res.boxes)
            xyxy = res.boxes.xyxy.cpu().numpy()
            cfs = res.boxes.conf.cpu().numpy()
            for box, cf in zip(xyxy, cfs):
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cf:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        counts.append(n)
        hud = f"f{_idx} dets={n}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()
    src.release()
    if counts:
        print(f"frames={len(counts)} boxes/frame: "
              f"min={min(counts)} median={int(statistics.median(counts))} "
              f"max={max(counts)} mean={statistics.mean(counts):.1f}")
    print(f"overlay -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run YOLOv8n on held-out frames")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default=str(TRAIN_DATASET_DIR / "images" / "val"),
                    help="image folder (default: dataset val split = held-out clips)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--name", default="yolov8n_infer")
    ap.add_argument("--clip", default=None, help="mp4 clip for frame-by-frame overlay video")
    args = ap.parse_args()
    if args.clip:
        infer_clip(Path(args.weights), Path(args.clip), conf=args.conf,
                   imgsz=args.imgsz, name=args.name)
    else:
        infer(Path(args.weights), Path(args.source), args.conf, args.imgsz, args.name)


if __name__ == "__main__":
    main()
