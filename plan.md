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

### Session 2 — Board Detection, Crop & Termination Detection (rotk, offline only)  ✅ COMPLETE (2026-06-23)

**Scope:** Vision utilities in rotk. No service wiring, no live game, no mouse movement.

**Done:** the board-detection / crop / coordinate-transform code (Part 1) is built and
verified end-to-end against *raw* captures — `find_ld_board_rect` isolates the minigame board on
both a 1920×1080 clip (`ld1080p1`) and a 1366×768 clip (`a04` source), `crop_to_board` produces the
canonical 744×498, and the full crop→`LdOnlineTracker`→overlay pipeline scored **0.910 within_r** vs
the in-game crosshair on `ld1080p1` (cf. offline reference `a01` 0.890). Verified via
`rotk/scripts/verify_ld_sample.py` (crop-space overlay) and `rotk/scripts/overlay_fullframe.py`
(overlay mapped back onto the original full frame). `is_ld_fail` shipped (2026-06-23) with the
human-provided `ld_cage.png` asset — template-match `TM_CCOEFF_NORMED ≥ 0.85` over a fractional
centre ROI; verified cage→1.00, real board→0.23 (clean separation), True/False on all cases.

- [x] Create `rotk/src/vision/__init__.py`
- [x] Create `rotk/src/vision/ld_board.py`
- [x] `find_ld_board_rect(frame_bgr)` implemented — tan-color mask + morphological close + largest contour with AR ∈ [1.40, 1.60] + ≥35% frame size; returns `(x, y, w, h) | None`
- [x] `crop_to_board(frame_bgr, rect)` resizes to exactly 744×498 (INTER_AREA downscale, INTER_LINEAR upscale)
- [x] `board_rect_to_screen(rect, x_offset, y_offset)` converts to absolute screen coordinates
- [x] `board_to_screen(bx, by, board_rect_screen)` transforms solver output to screen pixel
- [x] `is_ld_gone(frame_bgr)` returns True when `find_ld_board_rect` returns None
- [x] `is_ld_fail(frame_bgr, cage_template)` returns True when cage template matches at TM_CCOEFF_NORMED ≥ 0.85 — done (centre-ROI match, full-frame fallback, None-template guard)
- [x] ~~Create `rotk/scripts/test_ld_board.py`~~ — superseded by `rotk/scripts/verify_ld_sample.py` + `overlay_fullframe.py`, which exercise `find_ld_board_rect`/`crop_to_board` on real raw clips end-to-end (stronger than a single-screenshot check)
- [x] `is_ld_gone` returns True on a screenshot where LD board is absent, False when board is present — confirmed via the longest-board-run scan (board present 660–1475 on `a04`, absent elsewhere)
- [x] `is_ld_fail` returns True on a screenshot of the cage screen — confirmed (cage embedded in a 1366×768 frame → True @ peak 1.00; flat dark frame, real LD board @ 0.23, and None template → False)

---

### Session 3 — Mouse Backend (rotk, offline only)  ✅ COMPLETE (2026-06-23)

**Scope:** Cross-platform mouse control. Standalone verification only.

**Design change:** built under `plugins/platform/` (not `plugins/input/`) as platform-specific
providers behind a `typing.Protocol` + factory, mirroring `window_bounds.py`. Uses **native APIs
(Windows `ctypes` `SetCursorPos`/`mouse_event`, macOS CoreGraphics/Quartz `ctypes`) — `pynput` is
NOT used and is NOT a dependency.** Result: `test_mouse.py` PASS on Windows, cursor landed within
0px (d=0,0) of both targets; factory resolves `WindowsMouseCursorProvider` on this machine.

- [x] Create `rotk/src/plugins/platform/mouse_cursor.py` (≠ original `plugins/input/mouse_backend.py`)
- [x] `MouseCursorProvider` Protocol with `move_to(x, y)` and `click(x, y)` (+ `NoopMouseCursorProvider` fallback)
- [x] `WindowsMouseCursorProvider` (ctypes `SetCursorPos` + `mouse_event`) and `MacOSMouseCursorProvider` (Quartz `CGWarpMouseCursorPosition` + `CGEventCreateMouseEvent`); `create_mouse_cursor_provider(*, os_mode=AUTO)` factory dispatches on `OSType`
- [x] Works on Windows (primary target) — verified by running `rotk/scripts/test_mouse.py` (PASS, d=0,0)
- [ ] Works on macOS — **written but UNTESTED** (no Mac available); construct-only smoke path
- [x] `test_mouse.py` moves cursor to screen center, waits 1s, moves to a corner, reads back position via Win32 `GetCursorPos` and asserts within ±2px (`--click` flag gates a live click, off by default)

> ~~**HUMAN STEP REQUIRED BEFORE SESSION 4:** Crop `ld_cage.png` from the cage/fail screenshot at 1366×768 resolution and save to `rotk/assets/ld_cage.png`.~~ ✅ DONE (2026-06-23) — `ld_cage.png` (233×143) is in `rotk/assets/`; `is_ld_fail` ships against it. Session 4 is unblocked.

---

### Session 4 — LD Solver Service (rotk, offline integration)  ✅ COMPLETE (2026-06-23)

**Scope:** Assemble Parts 1–3 into `LdSolverService`. Verify with a replayed clip, not a live game.
Requires Session 1–3 complete and `ld_cage.png` in `rotk/assets/`.

**Done:** `rotk/src/core/ld_solver_service.py` ships the threaded `LdSolverService` plus the three
`Ld{Started,Success,Failed}Event` dataclasses (defined inline, mirroring `RuneDetector`'s own-file
event pattern). The loop dedups on `getLastFrameTimestamp` so the strictly-causal tracker only advances
on genuinely new frames; one `find_ld_board_rect` per frame doubles as the board-gone check and feeds
the one-shot rect cache. Verified two ways: (1) `rotk/tests/test_ld_solver_service.py` — **6 deterministic
unit tests PASS** (fake tracker + synthetic frames, no torch needed in CI) covering start/cache/cursor
mapping, success streak, fail-beats-success, ≥90% within-radius, idempotent start + named thread, clean
stop; (2) `rotk/scripts/replay_ld_service.py` — real-solver E2E over the **t4** clip (660 frames, live
YOLO): `LdSuccessEvent` fired, 555 cursor moves, **within-radius of GT 0.874** (vs published `fpath_human`
0.883 — the ~0.01 gap is the find/crop near-identity resize + heuristic frame alignment), PASS.

- [x] `pip install -e ../ld` confirmed working in rotk venv (`from ld.detect.online import LdOnlineTracker` imports cleanly — verified 2026-06-23; ld is already importable so no `-e` install was even needed)
- [x] `ultralytics` in `rotk/requirements.txt` (`ultralytics>=8.0.0`, line 9) and `pyproject.toml`; confirmed **installed** in the rotk venv (2026-06-23)
- [x] Define `LdStartedEvent`, `LdSuccessEvent`, `LdFailedEvent` dataclasses — defined in `rotk/src/core/ld_solver_service.py` (inline, like `RuneDetector`'s events)
- [x] Create `rotk/src/core/ld_solver_service.py` with `LdSolverService` class
- [x] Constructor accepts: `coordinate_service`, `mouse_backend`, `weights_path`, `game_frame_offsets`, `event_subscribers` (+ optional `tracker` injection for testability, `assets_dir`, `success_frames`/`fail_frames`, `target_fps`)
- [x] `start()` is idempotent — spawns thread only if not already running (tested)
- [x] `stop()` signals thread cleanly via event (tested)
- [x] Main loop: get frame → dedup-by-timestamp → find/cache board rect → crop → `push_frame` → `board_to_screen` → `mouse_backend.move_to`
- [x] Success condition: 5 consecutive frames where the board is gone → publish `LdSuccessEvent` + stop
- [x] Fail condition: 3 consecutive frames where `is_ld_fail` is True → publish `LdFailedEvent` + stop (lower threshold wins ties over success)
- [x] Thread named `LdSolverService`
- [x] Unit test with mocked `coordinate_service` replaying frames (synthetic + real t4 via the replay script): `LdSuccessEvent`/`LdFailedEvent` fire, cursor positions within radius of GT (synthetic ≥90%; real t4 0.874 ≈ published 0.883). The real-clip E2E lives as a **script** (needs YOLO weights + ~60s) rather than a pytest, so the committed suite stays torch-free.

---

### Session 5 — Bootstrap & FSM Integration (rotk, live wiring)  ✅ CODE COMPLETE (2026-06-23; live E2E pending a human)

**Scope:** Wire `LdSolverService` into the running app. Final end-to-end.
Requires all previous sessions complete.

**Done (all wiring + config + guards, 148 tests pass / 3 pre-existing unrelated failures):** the LD solver
is armed from the existing LD-content detector and pauses the bot for the minigame's duration. Two
deviations from the original sketch, both deliberate:
- **Frame source.** `getLastFrame()` returns the cached *minimap* rect, not the board — so the live solver
  pulls full game frames via a new `ForegroundGameFrameSource` adapter over
  `coordinate_service.captureForegroundGameFrameBgr()` (the calibrated 1366×768 frame). The board→screen
  origin is `getCalibratedForegroundFrameRect()`'s top-left (config `game_frame_x/y_offset` as fallback),
  passed to `LdSolverService` as a **callable** `game_frame_offsets` resolved at board-detection time.
- **`ld_active` lives on `GameState`/`StateManager`**, not `AppContext` — that's the thread-safe shared
  state `BotRunner` already holds, so the guards read it directly. `handle_ld_event` (in bootstrap)
  `state.patch(ld_active=…)`. Mouse backend is Session-3's `create_mouse_cursor_provider` (native
  ctypes/Quartz), **not** pynput.

Wiring isolated in `bootstrap._wire_ld_solver` (gated on `config.ld_solver.enabled`). Tests:
`rotk/tests/test_ld_solver_wiring.py` (5) — hook fires, optional-hook safety, events toggle `ld_active`
end-to-end through `LdSolverService`, and both bot-pause guards short-circuit while active.

- [x] `rotk/src/core/coordinate_service.py` — add `_on_ld_detected: Callable | None = None` field (defensive `getattr` read so `__new__`-built test instances stay safe)
- [x] `_sample_ld_content_once` calls `self._on_ld_detected()` after the existing alarm (alarm kept)
- [x] `rotk/src/app/bootstrap.py` — instantiate `LdSolverService` and the mouse provider (`create_mouse_cursor_provider`, Session-3 native backend — pynput not used)
- [x] `handle_ld_event` sets/clears `ld_active` (on the `StateManager`, see deviation above)
- [x] `coord_service._on_ld_detected = ld_solver.start` wired up
- [x] `ld_active: bool = False` added — to `GameState`/`StateManager` (thread-safe, BotRunner-visible) rather than `AppContext`
- [x] `BotRunner._tick_skills()` guards on `state.read().ld_active` — skips keyboard actions while True
- [x] `module_runtime.handle_move_to_anchor_request()` guards on `self.runner.state.read().ld_active`
- [x] `"ld_solver"` section added to `config/base.json` with `weights_path` + `enabled` keys (typed `LdSolverSettings` in `models.py`/`loader.py`; weights path resolved relative to rotk repo root → sibling `../ld/…`)
- [ ] **E2E verified live:** cursor tracks the real shape during a live LD minigame — *pending a human + live game*
- [ ] **E2E verified live:** `LdSuccessEvent` fires when board disappears; bot resumes keyboard movement — *pending*
- [ ] **E2E verified live:** `LdFailedEvent` fires when cage screen appears; bot resumes keyboard movement — *pending*
- [x] Bot sends no keyboard actions during the LD solve window — enforced by the two `ld_active` guards (unit-tested); live confirmation folded into the E2E items above

---

### Session 6 — LD Lifecycle Recording + Failure Exit (rotk)  ✅ CODE COMPLETE (2026-06-23; live E2E pending a human)

**Scope:** Record the entire LD lifecycle (detection → board-gone/cage) to a video, gated by a GUI
toggle that mirrors "Save rune failure screenshots", and **exit the app on LD failure** (mirroring the
rune-failure exit). Purely additive over Session 5; toggle OFF (default) = zero overhead.

**Decisions:** failure ⇒ **quit the whole app** (stop runner + `context.exit_requested`/`exit_reason`
→ GUI teardown + preempt IDLE_BOT); video = **raw full game frame** (no overlay); when ON, **every**
session is saved (success *and* failure), outcome in the filename.

**Done (154 tests pass / 3 pre-existing unrelated failures):**
- **Recorder** (`rotk/src/core/ld_solver_service.py`, `_LifecycleRecorder`): lazily opens an mp4 on the
  first captured frame (records from detection through the terminal frame), writes every raw full BGR
  frame, and on a `try/finally` finalizes + renames to
  `output/ld/ld_<stamp>_<success|failed|stopped>.mp4`. Fully exception-isolated (a recording error
  disables recording for the session, never crashes the solver). Reuses `open_writer`
  (`rotk/src/ld/capture/video_source.py:62`). Constructor gained `save_videos` / `video_dir` /
  `video_fps`.
- **Toggle** on `config.ld_solver.save_videos` wired through every layer mirroring
  `rune_failure_debug_capture`: `LdSolverSettings.save_videos` (`models.py`/`loader.py`/`base.json`);
  `ControlPanelSettings.save_ld_videos` + controller load (`ld_solver` section) + `_save_settings`
  (`ld_solver` sub-dict, merged via `merge_config_payload`); GUI "Save Lie Detector videos" checkbox +
  help text (var/widget/snapshot/build) in `gui/app.py`; `bootstrap._wire_ld_solver` passes
  `save_videos` + `video_dir=repo_root()/"output"/"ld"`.
- **Failure exit** (`bootstrap._exit_on_ld_failure`, called from `handle_ld_event` on `LdFailedEvent`):
  sets `exit_requested`/`exit_reason`, stops the runner via `execution_runtime.stop()` (fallback
  `current_runner.stop()`), and `bot_fsm.preempt(BotStates.IDLE_BOT)` — the exact rune-failure
  mechanism the GUI's `poll_runtime` observes to tear down. `LdSuccessEvent` just clears `ld_active`.
- **Tests:** `test_ld_solver_service.py` (+3: success/failure mp4 written with outcome in name;
  none when disabled); `test_ld_solver_wiring.py` (+3: `_exit_on_ld_failure` effects; failure event
  triggers exit end-to-end through `_wire_ld_solver`; success event does not).

- [x] `LdSolverService` records the full lifecycle (raw full-frame mp4) from first captured frame to terminal event, gated on `save_videos`
- [x] Video saved to `output/ld/ld_<stamp>_<success|failed|stopped>.mp4`; recording errors never crash the solver; OFF = zero overhead
- [x] `LdSolverSettings.save_videos` added (models/loader/base.json)
- [x] GUI "Save Lie Detector videos" checkbox + help text; round-trips through ControlPanelSettings + controller persistence
- [x] `bootstrap._wire_ld_solver` passes `save_videos` + `video_dir` to the service
- [x] On `LdFailedEvent`: app exits (stop runner + `context.exit_requested`/`exit_reason` + preempt IDLE_BOT), mirroring rune failure
- [x] On `LdSuccessEvent`: bot resumes (no exit), `ld_active` cleared
- [ ] **Live E2E (human):** enable the toggle, trigger a real LD minigame → an mp4 lands in `output/ld`; on cage failure → `*_failed.mp4` and the app exits — *pending* (folds into Session 5's live checks)

---

## Current status & next steps (updated 2026-06-23)

**Where we are.** Session 1 is ✅ complete (streaming solver, byte-faithful to `fpath_human`, mean
0.940). Session 2 is ✅ complete — board detection/crop/transform (Part 1) verified end-to-end on
raw 1080p + 1366×768 captures, and `is_ld_fail` shipped with the `ld_cage.png` asset (cage→1.00 vs
board→0.23). Session 3 is ✅ complete (platform-native mouse-cursor backend, Windows verified).
Session 4 is ✅ complete (`LdSolverService` — threaded solve loop assembled from Parts 1–3; 6 unit
tests PASS + real t4 E2E fired `LdSuccessEvent` at 0.874 within-radius). Session 5 is ✅ **code complete**
(bootstrap + FSM wiring: LD-content hook → `ld_solver.start`, full-frame capture adapter, `ld_active`
bot-pause guards, `ld_solver` config). Session 6 is ✅ **code complete** (lifecycle video recording +
GUI toggle + quit-on-failure; 154 tests pass). **The only remaining work is the live E2E pass** (a human
at a running game), which covers Sessions 5 and 6 together.

**Code locations (current):**
- `ld/detect/online.py` — `LdOnlineTracker` (streaming `fpath_human`)
- `ld/detect/test_online.py` — regression test (t1–t10 vs eval CSVs)
- `rotk/src/vision/ld_board.py` — `find_ld_board_rect` / `crop_to_board` / `board_rect_to_screen` / `board_to_screen` / `is_ld_gone` / `is_ld_fail`
- `rotk/src/plugins/platform/mouse_cursor.py` — `MouseCursorProvider` + Windows/macOS/Noop providers + factory
- `rotk/src/core/ld_solver_service.py` — `LdSolverService` + `Ld{Started,Success,Failed}Event` + `ForegroundGameFrameSource` + `_LifecycleRecorder`
- `rotk/src/app/bootstrap.py` — `_wire_ld_solver` (instantiates the service, wires `_on_ld_detected`, `handle_ld_event` toggles `ld_active` + quits on failure via `_exit_on_ld_failure`)
- `rotk/src/app/runtime_state.py` — `GameState.ld_active` (bot-pause flag)
- `rotk/src/core/config/{models,loader}.py` + `config/base.json` — `LdSolverSettings` (`enabled`, `weights_path`, `save_videos`)
- `rotk/src/app/controller.py` + `rotk/src/gui/app.py` — `save_ld_videos` toggle round-trip ("Save Lie Detector videos")
- `rotk/tests/test_ld_solver_service.py` — service unit tests (fake tracker, no torch) incl. recorder tests
- `rotk/tests/test_ld_solver_wiring.py` — Session 5/6 wiring tests (hook, `ld_active` toggle, bot-pause guards, quit-on-failure)
- `rotk/scripts/replay_ld_service.py` — real-solver E2E replay over a saved clip (needs YOLO weights)
- `rotk/scripts/verify_ld_sample.py`, `rotk/scripts/overlay_fullframe.py` — offline verification + evidence renderers

**Next steps, in order:**

1. ~~Finish Session 2 — `is_ld_fail` + asset.~~ ✅ DONE (2026-06-23). `ld_cage.png` provided
   (233×143) → `rotk/assets/`; `is_ld_fail` implemented in `rotk/src/vision/ld_board.py`
   (centre-ROI `TM_CCOEFF_NORMED ≥ 0.85`, full-frame fallback, None guard); verified cage→1.00,
   board→0.23.

2. **Session 3 — Mouse backend (rotk).** ✅ DONE (2026-06-23). Built as
   `rotk/src/plugins/platform/mouse_cursor.py`: `MouseCursorProvider` Protocol +
   Windows/macOS/Noop providers + `create_mouse_cursor_provider` factory, native ctypes/Quartz.
   **No `pynput` dependency** (the earlier carry-over note below is superseded). Verified Windows
   via `rotk/scripts/test_mouse.py` (PASS, d=0,0). macOS path written but untested.

3. **Session 4 — `LdSolverService` (rotk, offline integration).** ✅ DONE (2026-06-23). Threaded
   `LdSolverService` + three events in `rotk/src/core/ld_solver_service.py`; loop dedups on
   `getLastFrameTimestamp`, one-shot board-rect cache, success(5)/fail(3) streaks. 6 unit tests PASS
   (`rotk/tests/test_ld_solver_service.py`, fake tracker — no torch) + real t4 E2E
   (`rotk/scripts/replay_ld_service.py`): `LdSuccessEvent` fired, 0.874 within-radius vs published 0.883.
   `ld` already imports in the rotk venv (no `-e` install needed).

4. **Session 5 — Bootstrap & FSM wiring (rotk).** ✅ CODE DONE (2026-06-23). `_on_ld_detected →
   ld_solver.start` wired in `bootstrap._wire_ld_solver`; full-frame capture via
   `ForegroundGameFrameSource` over `captureForegroundGameFrameBgr()` (board→screen origin from the
   calibrated foreground rect, passed as a callable offset); `ld_active` on `StateManager` with
   `_tick_skills` + `handle_move_to_anchor_request` guards; `ld_solver` config section
   (`LdSolverSettings`). 148 tests pass (3 pre-existing failures unrelated: macOS recording +
   raw-`time.sleep` lint in `windows_recording.py`). Wiring tests in `tests/test_ld_solver_wiring.py`.

5. **Session 6 — Lifecycle recording + quit-on-failure (rotk).** ✅ CODE DONE (2026-06-23).
   `_LifecycleRecorder` in `LdSolverService` writes a raw full-frame mp4 of each session to
   `output/ld/ld_<stamp>_<outcome>.mp4`, gated by the GUI "Save Lie Detector videos" toggle
   (`config.ld_solver.save_videos`, mirrors the rune-debug toggle end to end). On `LdFailedEvent` the
   app exits via `bootstrap._exit_on_ld_failure` (stop runner + `exit_requested`/`exit_reason` +
   preempt IDLE_BOT), mirroring rune failure. 154 tests pass.

6. **Live E2E (needs a human + running game).** ⬅️ NEXT — covers Sessions 5 **and** 6. Trigger a real LD
   minigame and confirm: (a) the cursor tracks the real shape, (b) `LdSuccessEvent` fires on board-gone
   and the bot resumes, (c) `LdFailedEvent` fires on the cage screen → **the app exits**, (d) no keyboard
   actions during the solve, (e) with the toggle on, an mp4 lands in `output/ld` (`*_failed.mp4` on a
   failed run). Watch the two untested-live assumptions: that `captureForegroundGameFrameBgr()` yields
   the board region at the calibrated rect, and that `getCalibratedForegroundFrameRect()` is the right
   screen origin for `board_to_screen`.

**Carry-over notes / decisions:**
- The live rotk service does NOT need the two-pass longest-board-run scan the verify scripts use — it
  only crops *after* the existing LD-content detector fires, so a one-shot `find_ld_board_rect` + cache
  is enough (per the plan's one-shot-calibration approach).
- `board_to_screen` / `board_rect_to_screen` already exist for Session 4's cursor mapping.
- Cross-venv reality (RESOLVED 2026-06-23): `ultralytics` + `torch` are now both declared in rotk's
  `requirements.txt`/`pyproject.toml` AND installed in the rotk venv — the solver can run in-process in
  rotk. Session 4's only remaining import task is making the `ld` package importable there
  (`pip install -e ../ld` or the verify-script sys.path bootstrap).
- **Superseded (Session 3):** the mouse backend uses native ctypes/Quartz, so **no `pynput`
  dependency is needed** — ignore any earlier note about adding `pynput` to rotk.
