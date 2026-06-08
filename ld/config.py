"""Shared paths and pointer-stripping tunables."""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# --- Green cursor (ground-truth answer marker, HSV) ------------------------
# Bright pure-green crosshair (~0,255,0 in BGR). Used ONLY as GT for
# evaluation on the t*-clips; never as a tracking input.
GREEN_LOWER = (40, 120, 120)
GREEN_UPPER = (85, 255, 255)
GREEN_MIN_AREA = 20

# --- Pointer stripping -----------------------------------------------------
# Inpaint radius for TELEA fill; disk radius for live-mouse exclusion.
POINTER_INPAINT_RADIUS = 3
POINTER_RADIUS = 28  # px; covers crosshair + ring at 744p; scale for live
