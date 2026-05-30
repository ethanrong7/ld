"""Read MP4 frames for offline processing."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoFrame:
    frame_idx: int
    bgr: object  # numpy.ndarray
    timestamp: float


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoSource:
    """Iterate frames from a video file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cap: cv2.VideoCapture | None = None
        self._meta: VideoMeta | None = None

    def open(self) -> VideoMeta:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 60.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cap = cap
        self._meta = VideoMeta(self.path, width, height, fps, frame_count)
        return self._meta

    @property
    def meta(self) -> VideoMeta:
        if self._meta is None:
            return self.open()
        return self._meta

    def frames(self) -> Iterator[VideoFrame]:
        if self._cap is None:
            self.open()
        assert self._cap is not None
        meta = self.meta
        idx = 0
        while True:
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                break
            ts = idx / meta.fps if meta.fps > 0 else 0.0
            yield VideoFrame(frame_idx=idx, bgr=bgr, timestamp=ts)
            idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
