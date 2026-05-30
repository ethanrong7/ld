"""Shared thresholds, paths, and tunables for the LD solver."""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# --- Video -----------------------------------------------------------------
EXPECTED_FPS = 60.0

# --- White countdown shape segmentation (HSV) ------------------------------
# The pre-countdown target is rendered as a solid, near-pure-white fill:
# high value, low saturation. The countdown digit is blue/cyan and is
# excluded by its saturation/hue.
WHITE_S_MAX = 60
WHITE_V_MIN = 205
WHITE_MIN_AREA = 400  # px; ignore specular noise

# --- Green cursor (ground-truth answer marker, HSV) ------------------------
# Bright pure-green crosshair (~0,255,0 in BGR). Used ONLY as GT for
# evaluation on the t*-clips; never as a tracking input.
GREEN_LOWER = (40, 120, 120)
GREEN_UPPER = (85, 255, 255)
GREEN_MIN_AREA = 20

# --- Overlay ---------------------------------------------------------------
OVERLAY_FONT_SCALE = 0.5
OVERLAY_FONT_THICKNESS = 1
