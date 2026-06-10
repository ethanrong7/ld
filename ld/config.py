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

# --- Countdown / acquisition -----------------------------------------------
# The white countdown shape is bright and desaturated; the cyan numbers are
# bright but saturated, so an upper-saturation bound rejects them.
WHITE_V_MIN = 200
WHITE_S_MAX = 70
WHITE_MIN_AREA = 800  # px^2 of the largest white blob to count as the shape

# --- Rigid paper-motion + independent-motion saliency ----------------------
# Track features, fit the global rigid sheet motion (RANSAC), then treat
# features that disagree with it as belonging to the independently moving
# real shape.
FEAT_MAX = 1200
FEAT_QUALITY = 0.01
FEAT_MIN_DIST = 7
LK_WIN = 21
LK_LEVELS = 3
RANSAC_THRESH = 2.0          # px reprojection tolerance for "moves with paper"
OUTLIER_RESID_MIN = 1.5      # px; residual above this => independent motion
OUTLIER_RESID_CAP = 6.0      # clamp per-feature vote weight
OUTLIER_DOT_RADIUS = 4       # px stamp per outlier into the vote map
SALIENCY_SIGMA = 18.0        # gaussian blur of the vote map
BORDER_MARGIN = 25           # ignore warp/flow artifacts near the frame edge

# --- Tracker ---------------------------------------------------------------
GATE_RADIUS = 70.0           # px search radius around the predicted position
UPDATE_ALPHA = 0.4           # measurement blend toward the gated peak
VEL_DAMP = 0.7               # constant-velocity damping
VEL_MAX = 6.0                # px/frame; the real shape drifts slowly (~1-2 px)
SALIENCY_EMA = 0.5           # temporal smoothing of the saliency map
SALIENCY_FLOOR = 0.2         # min saliency to consider any signal present
REACQUIRE_FRAC = 0.5         # gated peak must be >= this fraction of the global
                             # peak to be trusted; else the dominant cluster is
                             # outside the gate and we are likely off-track
REACQUIRE_PATIENCE = 6       # frames off-track before jumping to the global peak
