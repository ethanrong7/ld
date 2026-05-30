# Phase 2 — Sheet registration & shape boxing

## The game (corrected model)

The LD panel is **not** a sparse field of ~11 discrete rings on a background. After
the countdown it is a **dense, edge-to-edge camouflage carpet** built from many copies
of **one silhouette** (the same shape shown solid-white at the start).

Two motions are layered on the panel:

1. **The fake sheet** — every decoy shape. The fakes form a **rigid lattice** (fixed
   relative geometry) that **translates in x/y as one piece** (no rotation). Think of a
   pattern drawn on paper, then the paper slid around. Their positions relative to each
   other never change.
2. **The REAL shape** — the instance that was solid white during the countdown, then
   fades translucent. It moves on its **own independent path**, *different* from the
   sheet motion.

Implication: you cannot find the real by appearance in a single frame (after the fade it
is visually identical to the fakes) and you cannot find it by raw motion (everything
moves). You find it by **cancelling the sheet motion** — once the fakes are frozen, the
real is the only thing left moving.

## Evidence (measured on `t1_cropped_trimmed.mp4`, 600f @ 60fps)

- **Rigid sheet confirmed.** A single global shift raises 5-frame correlation from
  ~0.3–0.5 to ~0.82–0.88 (phase-correlation response 0.88–0.97 at consecutive frames).
  The field is a translating textured sheet, not a morphing one.
- **Real moves independently.** Over frames 120→300 the player crosshair moved
  `(+10, -163)` while the accumulated sheet shift was `(-177, +135)` — different
  magnitude and opposite direction.
- **Real = residual mover.** Stabilising a frame onto a reference (warp by the negative
  accumulated shift) freezes the fakes; a frame-difference then shows a single bright blob
  exactly on the crosshair. When registration is correct, the blob lands **2–6px** from
  the crosshair (target is `MAX_CURSOR_ERROR_PX = 25`).

This replaces the previous "fixed decoy slots + bright-blob REAL" model, which assumed a
sparse scene and failed (see `output/t3_phase2_review/verdict.txt`).

## Ground truth (free)

In every `t*` clip the player's **green crosshair** sits on the real shape. Extract its
centre per frame (HSV green) as a **pseudo-ground-truth REAL trajectory** to score any
detector objectively. Not a bot input — validation only.

## Goal

Per frame, output:

- **1 box** on the REAL shape (residual mover after sheet cancellation).
- **n boxes** on the fakes (the frozen lattice), so we explicitly know **what not to
  track**.

All boxes share **one canonical size** from the white-phase silhouette. Phase 2 does not
yet label/persist REAL vs fake semantically beyond the single live box (Phase 3+).

## Pipeline

```
LD panel frame
  → exclude green cursor (HSV) + countdown/START UI band
  → [white phase] lock canonical silhouette template + box size + REAL seed centre
  → [every frame] estimate global sheet shift (dx,dy) via phase correlation (gap = 1)
  → stabilise: warp current frame by -accumulated_shift so the fake lattice freezes
  → REAL = brightest residual blob in stabilised diff, accepted via tracked motion gate
  → FAKES = canonical-template lattice, slid by accumulated_shift (boxed, not re-detected)
  → draw cyan boxes (fakes) + distinct box (REAL) + index + HUD
```

## Subtasks (build order)

Phase 2 is split into independently verifiable subtasks, sequenced to de-risk early.
Each produces a checkable artifact; the registration step (2c) is the linchpin.

| # | Subtask | Output / verify |
|---|---------|-----------------|
| **2a** | **Crosshair GT extractor** | Per-frame green-crosshair centroid → REAL trajectory. The scoring harness for everything else. *Build first.* |
| **2b** | **White-phase init** | Solid-white REAL → seed centroid + canonical silhouette template + fixed box size. Verify: box on the white shape during countdown. |
| **2c** | **Sheet registration** | Per-frame phase-corr `(dx,dy)` at gap = 1. Verify: response ~0.85+, smooth shift, accumulated path matches the texture slide. **Linchpin.** |
| **2d** | **Stabilise + residual REAL (single-shot)** | Warp by `-shift`, diff, brightest non-border blob. Verify: blob near GT when registration is good (~2–6px observed). |
| **2e** | **Tracked motion gate** | Prediction + gate radius + coast-on-miss over 2d. Verify: median error **< 25px** vs GT across the clip. **Acceptance bar.** |
| **2f** | **Fake lattice boxing** | Template-match silhouette in stabilised frame → lattice; slide by accumulated shift. Verify: boxes ride the texture, no drift. |
| **2g** | **Overlay + snapshots + HUD** | Wire into `run_offline`: REAL box, fake boxes, GT overlay, error readout, snapshot PNGs. |

**Scope notes:**

- **2a–2e are the core solve** (acquire + track REAL). **2f (boxing fakes) is a
  robustness / disambiguation aid, not a prerequisite** — the residual method finds REAL
  without it. Build 2f only if 2e's tracker needs help rejecting confusers. (Boxing the
  fakes was the key *diagnostic* that proved the rigid-sheet model, but in the final
  pipeline it is optional rather than the foundation.)
- **2c is the linchpin.** Solid registration makes 2d/2e easy; period-slips wobble
  everything downstream. Concentrate verification effort there.

Suggested first move: build **2a + 2c** together (cheap, and they immediately confirm the
approach holds across the whole clip).

### Step 1 — Preprocessing & exclusions

- Input: pre-cropped panel (`roi.py` pass-through on `t*_cropped_trimmed`).
- Mask out before registration/diff:
  - Green cursor: HSV `(40–85, 120–255, 120–255)`, dilated.
  - Countdown / START band (center text) during the intro frames.
- Grayscale + Hanning window for phase correlation.

### Step 2 — White-phase init (canonical template + REAL seed)

On the frames where REAL is solid white (high V, low S, center):

1. Threshold the bright blob → centroid = **REAL seed position**.
2. Extract the **silhouette / outline template** and record the **canonical box size**
   `(w, h)` (fixed for all entities, all frames).

Cache template + box size + seed for the rest of the clip.

### Step 3 — Sheet registration (per frame)

- `phaseCorrelate(prev, cur)` on windowed grayscale → per-frame global shift `(dx, dy)`.
- **Use gap = 1** (response 0.88–0.97); accumulate frame-to-frame. Avoid large gaps —
  on a repeating texture, big-gap correlation can lock onto the **wrong lattice period**
  (off-by-one-tile), which is the main failure mode.
- Maintain `accumulated_shift` = running sum from the reference frame.

### Step 4 — REAL detection (residual mover)

- Warp current frame by `-accumulated_shift` → stabilised frame (fakes frozen).
- Diff stabilised frame against the reference (or short rolling stabilised buffer), blur,
  ignore a border margin (warp edges are artifacts).
- Candidate REAL = brightest residual peak.
- **Tracked gate:** keep a predicted REAL position (last position + velocity). Accept the
  peak only if within a gate radius of the prediction; otherwise hold prediction / coast.
  This rejects the period-slip outliers that cause 100px+ jumps.

### Step 5 — FAKE lattice (boxing what not to track)

- In a clean stabilised frame, template-match the canonical silhouette → set of fake
  centres forming the lattice (constant within a clip).
- Each frame, draw fixed-size boxes at `lattice_centre + accumulated_shift`. No per-frame
  re-detection (no drift, no breathing).
- Use the lattice as a **rejection mask**: the REAL is the instance carrying motion that
  is *not* explained by the lattice/sheet.

### Step 6 — Overlay

- Fake boxes: cyan + index. REAL box: distinct color (still Phase 2, no semantic label
  beyond the single live box).
- HUD: frame, time, sheet shift `(dx,dy)`, REAL est, (debug) error vs crosshair GT.

## Acceptance criteria

On `data/t1_cropped_trimmed.mp4` (sanity-check another `t*`):

| Check | Pass condition |
|-------|----------------|
| Registration | Per-frame phase-corr response high (~0.85+) at gap = 1; sheet shift smooth |
| Stabilisation | Stabilised diff shows a single dominant non-border blob (the real) |
| REAL accuracy | Tracked REAL within **≤ 25px** of crosshair GT on the majority of post-fade frames |
| REAL motion | Exactly one box moves on a path **different** from the sheet shift |
| Fake stability | Fake boxes ride the sheet shift; no drift/breathing relative to the texture |
| White start | White-phase frame has a box on the solid REAL |
| Artifacts | No boxes locked to countdown digits, title chrome, or warp borders |
| Review | 5–10 snapshot PNGs with boxes + GT crosshair overlay |

## What Phase 2 is / is not

- **Is:** register the sheet, freeze the lattice, box fakes, isolate the residual REAL,
  validated against crosshair GT.
- **Phase 3:** semantic REAL tag + persistent `real_track_id`.
- **Phase 4:** full IoU tracking / re-acquisition robustness.
- **Phase 5:** mouse target / cursor error metrics (uses the same GT pipeline).

## Pitfalls

- **Large-gap registration** → off-by-one-lattice-period shift → fakes don't cancel →
  false REAL blob. Register at gap = 1 and accumulate.
- **No motion gate** → single bad frame throws REAL 100px+. Always gate against a tracked
  prediction.
- **Warp border band** → strong false diff at edges. Mask a margin before peak-picking.
- **Re-detecting fakes per frame** → boxes slide/breathe. Detect lattice once; slide it.
- **Trusting raw optical flow** → repeating texture defeats it (aperture problem). Use
  phase correlation for global motion.
- **Calibrate per video** (`t1`…`t10`); lattice period/phase is stable within a round but
  not assumed global across clips.

## Key parameters (`config.py`, to tune)

```python
# Canonical box (from white-phase silhouette)
CANONICAL_BOX_W = 120
CANONICAL_BOX_H = 120

# Exclusions
CURSOR_HSV_LOW = (40, 120, 120)
CURSOR_HSV_HIGH = (85, 255, 255)
COUNTDOWN_CENTER_X = (0.38, 0.62)
COUNTDOWN_CENTER_Y = (0.38, 0.58)

# White-phase init
WHITE_V_MIN = 200
WHITE_S_MAX = 80

# Sheet registration
REGISTER_GAP = 1
PHASECORR_MIN_RESPONSE = 0.6        # below this, coast on prediction
STAB_DIFF_BORDER_MARGIN = 60

# REAL tracking gate
REAL_GATE_RADIUS_PX = 40            # max accepted jump vs prediction
MAX_CURSOR_ERROR_PX = 25            # acceptance target vs crosshair GT
```

## CLI (target)

```bash
python -m ld.debug.run_offline \
  --input data/t1_cropped_trimmed.mp4 \
  --output output/t1_phase2.mp4 \
  --snapshots
```
