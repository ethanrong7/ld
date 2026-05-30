"""Video reading/writing helpers for offline clips."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


class VideoSource:
    """Thin wrapper over cv2.VideoCapture with metadata + iteration."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        self.meta = VideoMeta(
            path=self.path,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self.cap.get(cv2.CAP_PROP_FPS) or 0.0,
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def open_writer(path: str | Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {path}")
    return writer
