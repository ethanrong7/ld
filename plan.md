# Plan: LD Solver Integration (ld → rotk)

## Context

The rotk bot already detects the presence of a Lie Detector minigame (white-frame template match at
`CALIBRATED_MANUAL_REGION = (980, 480, 1040, 520)` in `coordinate_service.py`). When detected it plays
an alarm and hibernates for 60 seconds — it does nothing else. The goal is to extend this into a full
automated solver: capture the board, run the `fpath_human` tracker on live frames, move the cursor to
the solver's output, and exit cleanly on success or failure.

**Key measurements established by exploration:**
- rotk game frame: 1366×768 (config-driven, currently hardcoded as defaults)
- ld board canonical size: 744×498 px (all training data at this resolution, solver outputs in this space)
- Solver output `(x, y)` from `fpath_human` are **board-crop coordinates** (0–744, 0–498), not screen coords
- Board detection: tan-color mask + aspect ratio ~1.494 (from `board_crop.py`)
- `fpath_human` is causal (no future frames) — suitable for online use; the batch `run_clip` API is the only
  current entry point, so a streaming wrapper must be created

---

## Part 1 — Board Detection & ROI Crop in rotk

**What:** Locate the LD dialog in the game frame and produce a canonical 744×498 crop each frame.

**Files to create/modify:**
- `rotk/src/vision/__init__.py` — create `src/vision/` package
- `rotk/src/vision/ld_board.py` — board detection + crop

**ld_board.py implementation:**

```python
def find_ld_board_rect(frame_bgr: np.ndarray) -> tuple[int,int,int,int] | None:
    """
    Adapted from ld/detect/board_crop.py board_rect().
    Returns (x, y, w, h) in frame-pixel coords, or None if not found.
    Finds the largest tan-colored rectangle with AR in [1.40, 1.60] spanning
    >= 35% of frame width and height.
    """
    # Tan mask: r>110, g>90, b<g, r>=g, (r-b)>30
    # morphologyEx CLOSE with 9x9 kernel
    # findContours -> largest with AR + size constraints

def crop_to_board(frame_bgr: np.ndarray, rect: tuple) -> np.ndarray:
    """Extract and resize to 744×498 (INTER_AREA for downscale, INTER_LINEAR for upscale)."""

def board_rect_to_screen(
    rect: tuple[int,int,int,int],
    game_frame_x_offset: int,
    game_frame_y_offset: int
) -> tuple[int,int,int,int]:
    """Convert in-game-frame board rect to absolute screen coordinates."""
```

**One-shot calibration approach:** Since the bot runs at constant resolution, call `find_ld_board_rect`
on the first frame after LD is detected, cache the rect for the duration of the LD session (the dialog
doesn't move). Re-run detection only if the cached rect produces an empty crop.

**Verification:** Write a standalone script `rotk/scripts/test_ld_board.py` that loads a saved screenshot
of the LD dialog at 1366×768, calls `find_ld_board_rect`, draws the rect, and saves the output. Confirm
the crop looks identical to a 744×498 frame from the `ld` training data.

---

## Part 2 — Streaming Solver API (ld repo)

**What:** Create a frame-by-frame `LdOnlineTracker` class that wraps `fpath_human` logic (currently only
accessible via batch `run_clip`). This goes in the ld repo since the identity/trellis logic lives there.

**File to create:** `ld/detect/online.py`

**Class interface:**

```python
class LdOnlineTracker:
    def __init__(self, weights_path: str):
        # load YOLOv8n model once
        # initialize HumanCursor (min_cutoff=1.0, beta=0.007, deadband=2)
        # zero out all stateful fields below

    # Per-frame state (mirrors what track_fused_path_identity accumulates across its loop):
    # - prev_frame_gray: np.ndarray | None  (for LK optical flow)
    # - prev_kps: list of keypoints
    # - trellis_scores: dict[box_id -> float]  (running Viterbi log-scores)
    # - chosen_box: box centroid carrying through
    # - resid_chain: list of (frame, centroid) for cumulative residual (N=30 window)
    # - resid_accum: float  (current cumulative residual)
    # - churn_buf: deque of recent independent-motion vectors (for hedge)
    # - freeze_anchor: (x, y) | None  (held position during freeze)
    # - hcursor: HumanCursor instance
    # - frame_idx: int

    def push_frame(self, frame_744x498: np.ndarray) -> tuple[float, float] | None:
        """
        Process one 744×498 BGR frame. Returns (x, y) in board-crop coords
        once tracking is locked (returns None during countdown/acquire phase).
        Strictly causal — no future frames used.
        """

    def reset(self):
        """Reset all state for a new LD session."""
```

**Implementation notes:**
- Extract the per-frame logic from `track_fused_path_identity()` in `identity.py` (lines ~374–701)
- The countdown lock (`compute_countdown_lock`) runs on the first valid frame where the board
  countdown overlay is absent — reuse the existing function
- YOLO inference: call `self.model(frame, verbose=False)[0].boxes` directly (same as `detect_fusion_clip`)
- Optical flow: LK on the previous gray frame → `estimate_motion(prev, curr)` from `ld/vision/motion.py`
- Reuse `_box_coherent_mass`, `_box_curl`, `saliency_map` from existing modules — no duplication
- The `HumanCursor` from `ld/track/humanize.py` is already stateful; just call `.update(x, y, dt)` each frame

**Verification:** `ld/detect/test_online.py` — run `LdOnlineTracker` frame-by-frame over clips t1–t10
(reading the pre-cropped video), compare the emitted `(x, y)` sequence against the eval CSV
(`data/detect/eval/<clip>__fpath_human.csv`). Assert `within_r` matches the published 0.940 number
(tolerance ±0.002 for floating-point order-of-ops differences).

---

## Part 3 — Cross-Platform Mouse Control (rotk)

**What:** Create a mouse-movement backend that works on both Windows and macOS, following the same
plugin pattern as the existing keyboard backends in `src/plugins/input/`.

**File to create:** `rotk/src/plugins/input/mouse_backend.py`

```python
class MouseBackend(ABC):
    @abstractmethod
    def move_to(self, x: int, y: int) -> None: ...
    @abstractmethod
    def click(self, x: int, y: int) -> None: ...

class PynputMouseBackend(MouseBackend):
    """Cross-platform via pynput.mouse — works on both Windows and macOS."""
    def __init__(self):
        from pynput.mouse import Controller
        self._mouse = Controller()

    def move_to(self, x: int, y: int) -> None:
        self._mouse.position = (x, y)

    def click(self, x: int, y: int) -> None:
        from pynput.mouse import Button
        self._mouse.position = (x, y)
        self._mouse.click(Button.left)
```

**Coordinate transform helper** (add to `ld_board.py`):

```python
def board_to_screen(
    bx: float, by: float,           # solver output in 744×498 space
    board_rect_screen: tuple,       # (left, top, w, h) in screen coords
) -> tuple[int, int]:
    left, top, w, h = board_rect_screen
    sx = int(left + bx * (w / 744))
    sy = int(top  + by * (h / 498))
    return sx, sy
```

**pynput is already a dependency** (used in `keyboard_backend.py`), so no new package needed.

**Verification:** `rotk/scripts/test_mouse.py` — move cursor to screen center (683, 384), wait 1s,
move to a known corner; print actual position via `CoordinateService.getMouseScreenPosition()` and
assert within ±2px. Run on both Windows and macOS.

---

## Part 4 — Success & Fail Detection (rotk)

**What:** Detect when the LD session ends — either the board disappears (success) or the cage/fail
screen appears.

**Files to create/modify:**
- `rotk/assets/ld_cage.png` — crop the cage image from the second screenshot provided; save at the exact
  resolution it appears in a 1366×768 game frame
- `rotk/src/vision/ld_board.py` — add two detection functions

```python
def is_ld_gone(frame_bgr: np.ndarray) -> bool:
    """
    Returns True when the LD board is no longer visible.
    Re-run find_ld_board_rect; if it returns None, LD is gone.
    Also check the existing ld-content template has disappeared
    from CALIBRATED_MANUAL_REGION — belt-and-suspenders.
    """

def is_ld_fail(frame_bgr: np.ndarray, cage_template: np.ndarray) -> bool:
    """
    Returns True when the cage/failed UI is visible.
    Template match cage_template against the center ROI of the frame
    at TM_CCOEFF_NORMED >= 0.85.
    """
    # ROI: center band, e.g. (300, 150, 766, 550) at 1366×768 — tune after asset capture
```

**Verification:** Capture a screenshot of the success state (board gone) and fail state (cage visible).
Run each detection function against the screenshot and assert correct True/False.

---

## Part 5 — LD Solver Service (rotk)

**What:** A background service (analogous to `RuneDetector`) that orchestrates the full LD solve loop.
Lives in `rotk/src/core/ld_solver_service.py`.

**Events to define** (add alongside existing event definitions in `src/core/`):

```python
@dataclass
class LdStartedEvent: pass
@dataclass
class LdSuccessEvent: pass
@dataclass
class LdFailedEvent: pass
```

**Class skeleton:**

```python
class LdSolverService:
    """
    Triggered when LD content is detected. Grabs frames, runs the solver,
    moves the cursor. Terminates on success (board gone) or fail (cage visible).
    """
    def __init__(
        self,
        coordinate_service: CoordinateService,
        mouse_backend: MouseBackend,
        weights_path: str,
        game_frame_offsets: tuple[int, int],   # (x_offset, y_offset) from config
        event_subscribers: list[Callable],
    ): ...

    def start(self):
        """Spawn the solver thread. Idempotent if already running."""

    def stop(self):
        """Signal the thread to stop cleanly."""

    def _run(self):
        tracker = LdOnlineTracker(self.weights_path)
        board_rect_screen = None
        cage_template = cv2.imread(ASSETS / "ld_cage.png")
        success_streak = 0
        fail_streak = 0

        while not self._stop_event.is_set():
            frame = self.coordinate_service.getLastFrame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Detect/cache board rect (in screen coords) on first frame
            if board_rect_screen is None:
                rect = find_ld_board_rect(frame)
                if rect is None:
                    time.sleep(0.033)
                    continue
                board_rect_screen = board_rect_to_screen(rect, *self.game_frame_offsets)
                self._publish(LdStartedEvent())

            # Crop and push to solver
            crop = crop_to_board(frame, rect)
            xy = tracker.push_frame(crop)
            if xy is not None:
                sx, sy = board_to_screen(*xy, board_rect_screen)
                self.mouse_backend.move_to(sx, sy)

            # Termination checks (require N consecutive frames to avoid spurious triggers)
            if is_ld_gone(frame):
                success_streak += 1
                if success_streak >= 5:
                    self._publish(LdSuccessEvent())
                    break
            else:
                success_streak = 0

            if is_ld_fail(frame, cage_template):
                fail_streak += 1
                if fail_streak >= 3:
                    self._publish(LdFailedEvent())
                    break
            else:
                fail_streak = 0

            time.sleep(1.0 / 60)  # 60 Hz target
```

**Thread name:** `LdSolverService` (follow rotk convention).

**Verification:** Unit test with mocked `coordinate_service` that replays frames from a saved t4 clip.
Assert `LdSuccessEvent` fires within the clip duration, and cursor positions are within ±radius
of GT for ≥90% of frames.

---

## Part 6 — Bootstrap & FSM Integration (rotk)

**What:** Wire `LdSolverService` into the existing app bootstrap so it starts automatically on LD
detection and the FSM pauses the bot while the solver runs.

**Files to modify:**
- `rotk/src/app/bootstrap.py` — instantiate `LdSolverService`, subscribe to LD events
- `rotk/src/core/coordinate_service.py` — extend the existing `_sample_ld_content_once` to invoke a
  callback instead of only playing a sound

**Changes to coordinate_service.py:**

The existing `_run_ld_content_detector` plays a sound and hibernates. Replace `_play_alarm_sound()`
with a configurable callback so the service stays decoupled:

```python
# in __init__:
self._on_ld_detected: Callable | None = None

# in _sample_ld_content_once, where alarm currently fires:
self._play_alarm_sound()          # keep existing alarm
if self._on_ld_detected:
    self._on_ld_detected()        # new hook
```

**Changes to bootstrap.py:**

```python
ld_solver = LdSolverService(
    coordinate_service=coord_service,
    mouse_backend=PynputMouseBackend(),
    weights_path=config["ld_solver"]["weights_path"],
    game_frame_offsets=(config["app"]["game_frame_x_offset"], config["app"]["game_frame_y_offset"]),
    event_subscribers=[handle_ld_event],
)

def handle_ld_event(event):
    if isinstance(event, LdStartedEvent):
        app_context.ld_active = True
    elif isinstance(event, (LdSuccessEvent, LdFailedEvent)):
        app_context.ld_active = False
        log.info("LD session ended: %s", type(event).__name__)

coord_service._on_ld_detected = ld_solver.start
```

**Bot pause:** Add `ld_active: bool = False` to `AppContext`. In `BotRunner._tick_skills()` and
`module_runtime.handle_move_to_anchor_request()`, guard with:

```python
if app_context.ld_active:
    return  # don't send keyboard actions while LD solver is controlling the cursor
```

**Config additions to `config/base.json`:**

```json
"ld_solver": {
    "weights_path": "data/detect/runs/yolov8n_single_combined/weights/best.pt",
    "enabled": true
}
```

**Verification:** End-to-end against a live game session:
1. Trigger a Lie Detector minigame
2. Assert cursor moves to follow the real shape (visible in game)
3. Assert `LdSuccessEvent` fires when the board disappears and bot resumes movement
4. Assert `LdFailedEvent` fires on cage screen (test by letting the timer expire)
5. Assert bot keyboard actions stop during LD and resume after

---

## Dependency & Asset Checklist

| Item | Where | Action |
|------|--------|--------|
| `ultralytics` | rotk `requirements.txt` | Add (already in ld repo's requirements) |
| `pynput.mouse` | rotk | Already present via pynput; confirm `Controller` is importable |
| ld repo importable | rotk env | `pip install -e ../ld` in the rotk venv so `from ld.detect.online import LdOnlineTracker` works |
| YOLO weights | shared path | Set `ld_solver.weights_path` in config to point at ld repo's best.pt |
| `ld_cage.png` | `rotk/assets/` | Crop cage region from fail screenshot at 1366×768 and save |
| `ld-content.png` | `rotk/assets/` | Already present — used by existing LD trigger |

---

## Implementation Order

Parts are designed for independent build + verify before integration:

1. **Part 2** first — the streaming solver is the hardest piece and lives entirely in ld repo.
   Gate with the regression test against eval CSVs before touching rotk.
2. **Part 1 + Part 4** — board detection and termination conditions are pure CV; test offline
   against saved screenshots.
3. **Part 3** — mouse backend is a ~20-line wrapper; verify with a standalone movement script.
4. **Part 5** — assemble Parts 1–4 into the service; verify with a replayed clip.
5. **Part 6** — wire into bootstrap last, after Parts 1–5 pass independently.

---

## Session Breakdown & Progress Checklist

Do NOT start a session until all items in the previous session are checked off.
The `ld_cage.png` asset (between sessions 3 and 4) requires a manual human step — pause and ask the user to provide it before starting session 4.

---

### Session 1 — Streaming Solver (ld repo only)  ✅ COMPLETE (2026-06-22)

**Scope:** Create the online frame-by-frame API in the ld repo. Do not touch rotk.

**Result:** `LdOnlineTracker` reproduces batch `fpath_human` byte-for-byte — all 10 clips
Δwithin_r = **0.0000**, mean **0.940**, per-frame xy max-drift 0.05 px (float32 rounding).
Live-YOLO path also verified end-to-end (t7 = 1.000).

- [x] Create `ld/detect/online.py` with `LdOnlineTracker` class
- [x] `__init__` loads YOLOv8n weights and initialises all stateful fields (trellis scores, resid chain, churn buffer, freeze anchor, HumanCursor, prev frame)
- [x] `push_frame(frame_744x498)` processes one frame and returns `(x, y) | None` — strictly causal
- [x] `reset()` clears all state for a new session
- [x] All per-frame logic ported from `track_fused_path_identity()` in `identity.py`: YOLO inference → optical flow → saliency → emission scores → Viterbi update → residual accumulation → freeze gate → churn hedge → HumanCursor
- [x] Create `ld/detect/test_online.py` that replays clips t1–t10 frame-by-frame through `LdOnlineTracker`
- [x] Regression test passes: per-clip `within_r` matches published `fpath_human` values (0.940 mean, tolerance ±0.002) — **all 10 clips Δ=0.0000, mean 0.940, per-frame xy max-drift 0.05px (float32 rounding)**
- [x] No per-clip regression vs the eval CSVs in `data/detect/eval/<clip>__fpath_human.csv`

---

### Session 2 — Board Detection, Crop & Termination Detection (rotk, offline only)  🟡 PARTIAL (2026-06-22)

**Scope:** Vision utilities in rotk. No service wiring, no live game, no mouse movement.

**Done so far:** the board-detection / crop / coordinate-transform code (Part 1) is built and
verified end-to-end against *raw* captures — `find_ld_board_rect` isolates the minigame board on
both a 1920×1080 clip (`ld1080p1`) and a 1366×768 clip (`a04` source), `crop_to_board` produces the
canonical 744×498, and the full crop→`LdOnlineTracker`→overlay pipeline scored **0.910 within_r** vs
the in-game crosshair on `ld1080p1` (cf. offline reference `a01` 0.890). Verified via
`rotk/scripts/verify_ld_sample.py` (crop-space overlay) and `rotk/scripts/overlay_fullframe.py`
(overlay mapped back onto the original full frame). **Remaining:** `is_ld_fail` + the `ld_cage.png`
asset (blocked on the human step below).

- [x] Create `rotk/src/vision/__init__.py`
- [x] Create `rotk/src/vision/ld_board.py`
- [x] `find_ld_board_rect(frame_bgr)` implemented — tan-color mask + morphological close + largest contour with AR ∈ [1.40, 1.60] + ≥35% frame size; returns `(x, y, w, h) | None`
- [x] `crop_to_board(frame_bgr, rect)` resizes to exactly 744×498 (INTER_AREA downscale, INTER_LINEAR upscale)
- [x] `board_rect_to_screen(rect, x_offset, y_offset)` converts to absolute screen coordinates
- [x] `board_to_screen(bx, by, board_rect_screen)` transforms solver output to screen pixel
- [x] `is_ld_gone(frame_bgr)` returns True when `find_ld_board_rect` returns None
- [ ] `is_ld_fail(frame_bgr, cage_template)` returns True when cage template matches at TM_CCOEFF_NORMED ≥ 0.85 — **blocked on `ld_cage.png` asset**
- [x] ~~Create `rotk/scripts/test_ld_board.py`~~ — superseded by `rotk/scripts/verify_ld_sample.py` + `overlay_fullframe.py`, which exercise `find_ld_board_rect`/`crop_to_board` on real raw clips end-to-end (stronger than a single-screenshot check)
- [x] `is_ld_gone` returns True on a screenshot where LD board is absent, False when board is present — confirmed via the longest-board-run scan (board present 660–1475 on `a04`, absent elsewhere)
- [ ] `is_ld_fail` returns True on a screenshot of the cage screen — **blocked on `ld_cage.png` asset**

---

### Session 3 — Mouse Backend (rotk, offline only)

**Scope:** Cross-platform mouse control. Standalone verification only.

- [ ] Create `rotk/src/plugins/input/mouse_backend.py`
- [ ] `MouseBackend` abstract base class with `move_to(x, y)` and `click(x, y)`
- [ ] `PynputMouseBackend` implemented using `pynput.mouse.Controller`
- [ ] Works on Windows (primary target) — verified by running `rotk/scripts/test_mouse.py`
- [ ] Works on macOS — verified (or noted as untested if no Mac available)
- [ ] `test_mouse.py` moves cursor to screen center, waits 1s, moves to a corner, reads back position via `CoordinateService.getMouseScreenPosition()` and asserts within ±2px

> **HUMAN STEP REQUIRED BEFORE SESSION 4:** Crop `ld_cage.png` from the cage/fail screenshot at 1366×768 resolution and save to `rotk/assets/ld_cage.png`. The cage is the dark barred-cylinder UI that appears on a failed LD attempt.

---

### Session 4 — LD Solver Service (rotk, offline integration)

**Scope:** Assemble Parts 1–3 into `LdSolverService`. Verify with a replayed clip, not a live game.
Requires Session 1–3 complete and `ld_cage.png` in `rotk/assets/`.

- [ ] `pip install -e ../ld` confirmed working in rotk venv (`from ld.detect.online import LdOnlineTracker` imports cleanly)
- [ ] `ultralytics` added to `rotk/requirements.txt`
- [ ] Define `LdStartedEvent`, `LdSuccessEvent`, `LdFailedEvent` dataclasses (in `rotk/src/core/` alongside existing events)
- [ ] Create `rotk/src/core/ld_solver_service.py` with `LdSolverService` class
- [ ] Constructor accepts: `coordinate_service`, `mouse_backend`, `weights_path`, `game_frame_offsets`, `event_subscribers`
- [ ] `start()` is idempotent — spawns thread only if not already running
- [ ] `stop()` signals thread cleanly via event
- [ ] Main loop: get frame → find/cache board rect → crop → `push_frame` → transform → `mouse_backend.move_to`
- [ ] Success condition: 5 consecutive frames where `is_ld_gone` is True → publish `LdSuccessEvent` + stop
- [ ] Fail condition: 3 consecutive frames where `is_ld_fail` is True → publish `LdFailedEvent` + stop
- [ ] Thread named `LdSolverService`
- [ ] Unit test with mocked `coordinate_service` replaying frames from a saved clip (e.g. t4): asserts `LdSuccessEvent` fires, cursor positions within radius of GT on ≥90% of frames

---

### Session 5 — Bootstrap & FSM Integration (rotk, live wiring)

**Scope:** Wire `LdSolverService` into the running app. Final end-to-end.
Requires all previous sessions complete.

- [ ] `rotk/src/core/coordinate_service.py` — add `_on_ld_detected: Callable | None = None` field
- [ ] `_sample_ld_content_once` calls `self._on_ld_detected()` after the existing alarm (alarm kept)
- [ ] `rotk/src/app/bootstrap.py` — instantiate `LdSolverService` and `PynputMouseBackend`
- [ ] `handle_ld_event` sets/clears `app_context.ld_active`
- [ ] `coord_service._on_ld_detected = ld_solver.start` wired up
- [ ] `ld_active: bool = False` added to `AppContext`
- [ ] `BotRunner._tick_skills()` guards on `app_context.ld_active` — skips keyboard actions while True
- [ ] `module_runtime.handle_move_to_anchor_request()` guards on `app_context.ld_active`
- [ ] `"ld_solver"` section added to `config/base.json` with `weights_path` and `enabled` keys
- [ ] **E2E verified live:** cursor tracks the real shape during a live LD minigame
- [ ] **E2E verified live:** `LdSuccessEvent` fires when board disappears; bot resumes keyboard movement
- [ ] **E2E verified live:** `LdFailedEvent` fires when cage screen appears; bot resumes keyboard movement
- [ ] Bot sends no keyboard actions during the LD solve window

---

## Current status & next steps (updated 2026-06-23)

**Where we are.** Session 1 is ✅ complete (streaming solver, byte-faithful to `fpath_human`, mean
0.940). Session 2 is 🟡 partial — the board detection/crop/transform code (Part 1) is built and
verified end-to-end on raw 1080p + 1366×768 captures; only `is_ld_fail` remains (blocked on the
`ld_cage.png` asset). Sessions 3–6 are untouched.

**Code locations (current):**
- `ld/detect/online.py` — `LdOnlineTracker` (streaming `fpath_human`)
- `ld/detect/test_online.py` — regression test (t1–t10 vs eval CSVs)
- `rotk/src/vision/ld_board.py` — `find_ld_board_rect` / `crop_to_board` / `board_rect_to_screen` / `board_to_screen` / `is_ld_gone`
- `rotk/scripts/verify_ld_sample.py`, `rotk/scripts/overlay_fullframe.py` — offline verification + evidence renderers

**Next steps, in order:**

1. **Finish Session 2 — `is_ld_fail` + asset (BLOCKED on human step).**
   Human must crop `ld_cage.png` from a failed-attempt screenshot at 1366×768 and drop it in
   `rotk/assets/`. Then implement `is_ld_fail(frame, cage_template)` (template match TM_CCOEFF_NORMED
   ≥ 0.85 over a center ROI) and confirm True/False on a fail vs non-fail screenshot.

2. **Session 3 — Mouse backend (rotk, no blockers).**
   `rotk/src/plugins/input/mouse_backend.py`: `MouseBackend` ABC + `PynputMouseBackend`
   (`pynput.mouse.Controller`). NOTE: `pynput` is NOT currently in rotk's deps (only `keyboard` is) —
   confirm/add it. Verify with `rotk/scripts/test_mouse.py` (move → read back within ±2px).

3. **Session 4 — `LdSolverService` (rotk, offline integration).** Requires Sessions 1–3 + `ld_cage.png`.
   Add `ultralytics` to `rotk/requirements.txt` and make `ld` importable in the rotk venv
   (`pip install -e ../ld`, or the sys.path bootstrap the verify scripts already use — note rotk's venv
   currently lacks `ultralytics`, which is why the verify scripts run from ld's venv). Define the three
   events, build the threaded service, unit-test with a mocked `coordinate_service` replaying a clip.

4. **Session 5/6 — Bootstrap & FSM wiring + live E2E.** Wire `_on_ld_detected → ld_solver.start`,
   `ld_active` bot-pause guards, config, and verify live.

**Carry-over notes / decisions:**
- The live rotk service does NOT need the two-pass longest-board-run scan the verify scripts use — it
  only crops *after* the existing LD-content detector fires, so a one-shot `find_ld_board_rect` + cache
  is enough (per the plan's one-shot-calibration approach).
- `board_to_screen` / `board_rect_to_screen` already exist for Session 4's cursor mapping.
- Cross-venv reality: ld's venv has `ultralytics`/`torch`; rotk's has `torch` but not `ultralytics`.
  Session 4 must resolve this (add `ultralytics` to rotk, or run the solver via the installed `ld` pkg).
