"""Cached YOLO detection per clip -> list[FusionPack].

`detect_fusion_clip` runs the trained YOLO detector over every frame of a clip
(green crosshair stripped first), records the boxes + GT + countdown-white shape,
and caches the result keyed by md5(weights path + mtime). It is the single
detection entry point for the identity tracker, the leaderboard, and the evidence
renderer -- everything downstream consumes its `FusionPack` list.

Usage:
    from ld.detect.fusion import detect_fusion_clip, FusionPack
    packs = detect_fusion_clip(weights, clip)   # cached after first run
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ld.capture.video_source import VideoSource
from ld.config import DETECT_DIR
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import find_cursor, strip_pointer

__all__ = ["FusionPack", "detect_fusion_clip"]

FUSION_CACHE_DIR = DETECT_DIR / "cache" / "fusion"


@dataclass
class FusionPack:
    idx: int
    gt: tuple[float, float] | None
    white: tuple[float, float, float] | None
    boxes: list[tuple[float, float, float, float, float]]  # x1,y1,x2,y2,conf

    @property
    def dets(self) -> list[tuple[float, float, float]]:
        return [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[4]) for b in self.boxes]


def _weights_tag(weights: Path) -> str:
    st = weights.stat()
    return hashlib.md5(f"{weights.resolve()}:{st.st_mtime_ns}".encode()).hexdigest()[:10]


def _fusion_cache(clip: Path, weights: Path, conf: float, imgsz: int) -> Path:
    return FUSION_CACHE_DIR / f"{clip.stem}_{_weights_tag(weights)}_c{conf}_s{imgsz}.json"


def detect_fusion_clip(weights: str | Path, clip: str | Path, *, conf: float = 0.25,
                       imgsz: int = 768, use_cache: bool = True) -> list[FusionPack]:
    weights, clip = Path(weights), Path(clip)
    cache = _fusion_cache(clip, weights, conf, imgsz)
    if use_cache and cache.exists():
        raw = json.loads(cache.read_text())
        return [FusionPack(p["idx"],
                           tuple(p["gt"]) if p["gt"] else None,
                           tuple(p["white"]) if p["white"] else None,
                           [tuple(b) for b in p["boxes"]]) for p in raw]

    from ultralytics import YOLO  # lazy: heavy (torch) dependency

    model = YOLO(str(weights))
    packs: list[FusionPack] = []
    src = VideoSource(clip)
    for idx, frame in src.frames():
        gt = find_cursor(frame)
        stripped = strip_pointer(frame, strip_green=True)
        ws = detect_white_shape(stripped)
        res = model.predict(stripped, conf=conf, imgsz=imgsz, verbose=False)[0]
        boxes: list[tuple[float, float, float, float, float]] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cfs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cf in zip(xyxy, cfs):
                boxes.append((float(x1), float(y1), float(x2), float(y2), float(cf)))
        packs.append(FusionPack(idx, gt,
                                (ws.cx, ws.cy, ws.radius) if ws else None, boxes))
    src.release()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([
        {"idx": p.idx, "gt": list(p.gt) if p.gt else None,
         "white": list(p.white) if p.white else None,
         "boxes": [list(b) for b in p.boxes]} for p in packs]))
    return packs
