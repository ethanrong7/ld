"""YOLO bounding-box helpers (normalized cx,cy,w,h <-> pixel xyxy)."""
from __future__ import annotations

from pathlib import Path

__all__ = ["Box", "to_yolo_line", "parse_yolo_line", "load_labels", "save_labels",
           "prefill_box"]


class Box:
    """Pixel-space axis-aligned box with a class id."""

    __slots__ = ("cls", "x1", "y1", "x2", "y2")

    def __init__(self, cls: int, x1: float, y1: float, x2: float, y2: float):
        self.cls = cls
        self.x1, self.y1 = min(x1, x2), min(y1, y2)
        self.x2, self.y2 = max(x1, x2), max(y1, y2)

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def valid(self, min_side: float = 3.0) -> bool:
        return self.w >= min_side and self.h >= min_side


def to_yolo_line(b: Box, img_w: int, img_h: int) -> str:
    cx = (b.x1 + b.x2) / 2.0 / img_w
    cy = (b.y1 + b.y2) / 2.0 / img_h
    w = b.w / img_w
    h = b.h / img_h
    return f"{b.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def parse_yolo_line(line: str, img_w: int, img_h: int) -> Box | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    cls, cx, cy, w, h = (float(p) for p in parts)
    cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
    return Box(int(cls), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def load_labels(path: str | Path, img_w: int, img_h: int) -> list[Box]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[Box] = []
    for line in path.read_text().splitlines():
        if line.strip():
            b = parse_yolo_line(line, img_w, img_h)
            if b is not None:
                out.append(b)
    return out


def save_labels(path: str | Path, boxes: list[Box], img_w: int, img_h: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [to_yolo_line(b, img_w, img_h) for b in boxes if b.valid()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def prefill_box(gt_x: float, gt_y: float, radius: float, scale: float,
                cls: int = 0) -> Box | None:
    """Box centred on the GT crosshair, side = 2*radius*scale (the real shape)."""
    if gt_x < 0 or gt_y < 0 or radius <= 0:
        return None
    half = radius * scale
    return Box(cls, gt_x - half, gt_y - half, gt_x + half, gt_y + half)
