"""Central thresholds and paths for the Lie Detector solver."""
from __future__ import annotations

from pathlib import Path

# Repo layout (package lives at repo_root/ld/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

# Test clips: pre-cropped LD panel, trimmed to one LD round
DEFAULT_VIDEO = DATA_DIR / "t1_cropped_trimmed.mp4"
TEST_VIDEOS_GLOB = "t*_cropped_trimmed.mp4"

# Detection
MIN_BLOB_AREA = 200
MAX_BLOB_AREA = 15_000
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0
DEDUPE_IOU_THRESH = 0.5

# Green game cursor (exclude from contours)
CURSOR_HSV_LOW = (35, 80, 80)
CURSOR_HSV_HIGH = (85, 255, 255)

# REAL init (white shape)
WHITE_V_MIN = 200
WHITE_S_MAX = 80

# Tracking
IOU_MATCH_THRESH = 0.3
DECOY_MAX_DRIFT_PX = 15
REAL_MIN_MOVE_PX = 8
REAL_LOST_FRAMES_BEFORE_REACQUIRE = 3

# Mouse / evaluation
TARGET_SMOOTHING = True
MAX_CURSOR_ERROR_PX = 25

# Offline debug overlay
OVERLAY_FONT_SCALE = 0.55
OVERLAY_FONT_THICKNESS = 1
