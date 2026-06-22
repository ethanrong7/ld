"""Streaming (frame-by-frame) wrapper around the ``fpath_human`` solver.

The shipped leader ``fpath_human`` (see ``identity.track_fused_path_identity``) is
only reachable via the *batch* ``run_clip`` API: it opens a whole clip, runs YOLO
over every frame (cached), computes the countdown lock from the full pack list, and
only then iterates. ``LdOnlineTracker`` exposes the *same* logic one frame at a time
so it can drive a live cursor (rotk integration, Part 2 of plan.md).

Faithfulness contract
---------------------
The per-frame state machine here is a line-for-line port of the track-phase body of
``track_fused_path_identity`` run with the ``fpath_human`` parameters (additive
fuse emission + EMA-coherent-mass hysteresis + churn hedge + residual freeze +
human-cursor output dynamics). The only carried state from the acquire phase into
the track phase is ``prev_gray = gray(lock_frame)`` and ``pos = lock centroid``
(every accumulator -- ``mass_scale``, ``ovecs_hist``, ``ema_cm`` ... -- starts fresh
at the first track frame), so we do not need to replay the acquire frames: we start
the machine at the lock frame. ``test_online.py`` replays t1-t10 through this class
feeding the *cached* detections (so the box source matches the offline eval exactly)
and asserts the accumulated track reproduces the published ``fpath_human`` within_r.

Online discipline
-----------------
Strictly causal. The countdown lock needs ``miss_to_confirm=3`` white-less frames to
confirm the countdown ended (mirrors ``_countdown_white_pack``), so there is a bounded
handoff lag: ``push_frame`` returns ``None`` throughout the countdown/acquire phase
and begins returning ``(x, y)`` once the lock is established. This is the bounded
fixed-lag buffer the project permits (~a few frames at handoff, then live).
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ld.detect.constellation import TrackPoint, _seed
from ld.detect.fusion import FusionPack
from ld.detect.identity import (
    FPATH_FREEZE_CONSEC, FPATH_FREEZE_LAG, FPATH_FREEZE_N, FPATH_FREEZE_TAU,
    FPATH_FUSE_CMASS_W, FPATH_FUSE_CURL_W, FPATH_FUSE_WIN,
    FPATH_HEDGE_CHURN_HI, FPATH_HEDGE_WIN,
    FPATH_HYST_ALPHA, FPATH_HYST_CONSEC, FPATH_HYST_MARGIN,
    FPATH_MASS_EMA, FPATH_TRANS_W, FPATH_VEL_DAMP,
    LockInfo, _box_coherent_mass, _box_rotational_curl, _box_saliency_mass,
    _centroid, _countdown_white_pack, _cumulative_residual, _pick_lock_box,
    _stable_white_anchor, _white_mask,
)
from ld.track.humanize import HumanCursor
from ld.vision.countdown import detect_white_shape
from ld.vision.cursor import strip_pointer
from ld.vision.motion import estimate_motion, saliency_map

__all__ = ["LdOnlineTracker"]

_MISS_TO_CONFIRM = 3   # white-less frames that confirm the countdown ended (matches _seed)


class LdOnlineTracker:
    """Frame-by-frame ``fpath_human`` tracker for a single LD session.

    Usage::

        trk = LdOnlineTracker("data/detect/runs/.../best.pt")
        for frame in stream:                 # 744x498 BGR frames
            xy = trk.push_frame(frame)       # (x, y) in board-crop coords, or None
            if xy is not None:
                move_cursor(*xy)
        trk.reset()                          # for the next session
    """

    def __init__(self, weights_path: str | Path | None = None, *,
                 conf: float = 0.25, imgsz: int = 768, frame_wh=(744, 498),
                 trans_w: float = FPATH_TRANS_W, vel_damp: float = FPATH_VEL_DAMP,
                 mass_ema: float = FPATH_MASS_EMA,
                 cmass_w: float = FPATH_FUSE_CMASS_W, curl_w: float = FPATH_FUSE_CURL_W,
                 fuse_win: int = FPATH_FUSE_WIN,
                 hyst_alpha: float = FPATH_HYST_ALPHA, hyst_margin: float = FPATH_HYST_MARGIN,
                 hyst_consec: int = FPATH_HYST_CONSEC,
                 hedge_churn_hi: float = FPATH_HEDGE_CHURN_HI, hedge_win: int = FPATH_HEDGE_WIN,
                 freeze_tau: float = FPATH_FREEZE_TAU, freeze_lag: int = FPATH_FREEZE_LAG,
                 freeze_consec: int = FPATH_FREEZE_CONSEC, freeze_n: int = FPATH_FREEZE_N):
        self.weights_path = str(weights_path) if weights_path is not None else None
        self.conf = conf
        self.imgsz = imgsz
        self.fw, self.fh = frame_wh
        # fpath_human tunables (defaults == the shipped leader; see identity._dispatch_mode)
        self.trans_w = trans_w
        self.vel_damp = vel_damp
        self.mass_ema = mass_ema
        self.cmass_w = cmass_w
        self.curl_w = curl_w
        self.fuse_win = fuse_win
        self.hyst_alpha = hyst_alpha
        self.hyst_margin = hyst_margin
        self.hyst_consec = hyst_consec
        self.hedge_churn_hi = hedge_churn_hi
        self.hedge_win = hedge_win
        self.freeze_tau = freeze_tau
        self.freeze_lag = freeze_lag
        self.freeze_consec = freeze_consec
        self.freeze_n = freeze_n
        self._model = None   # lazy YOLO load
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear all per-session state for a fresh LD session."""
        self.frame_idx = 0
        self.track: list[TrackPoint] = []        # complete emitted record (mirrors batch `track`)
        self._established = False                # lock found, track machine running
        self.start = 0
        self.radius = 55.0
        self.lock: LockInfo | None = None
        self.lock_frame = 0
        # countdown buffering: light packs for the whole countdown + rolling stripped frames
        self._cd_packs: list[FusionPack] = []
        self._roll: deque = deque(maxlen=_MISS_TO_CONFIRM + 1)   # (idx, stripped_bgr) of last frames
        self._last_white_idx = -1
        self._white_misses = 0
        self._seen_white = False
        # track-phase machine state (initialised at handoff in _begin_track)
        self._pos = None
        self._vel = None
        self._prev_cents = None
        self._prev_alpha = None
        self._prev_gray = None
        self._active = False
        self._lost = 0
        self._mass_scale = 0.0
        self._cmass_scale = 0.0
        self._curl_scale = 0.0
        self._ovecs_hist: deque | None = None
        self._ema_cm = None
        self._ema_cents = None
        self._hyst_streak = 0
        self._hpos = None
        self._hwin: deque | None = None
        self._hprev_cent = None
        self._rhist: deque | None = None
        self._freeze_buf: deque | None = None
        self._fpos = None
        self._frozen = False
        self._low_streak = 0
        self._hcursor: HumanCursor | None = None

    # -------------------------------------------------------------- detection
    def _ensure_model(self):
        if self._model is None:
            if self.weights_path is None:
                raise RuntimeError("LdOnlineTracker needs weights_path to run YOLO "
                                   "(or pass detection=(white, boxes) to push_frame)")
            from ultralytics import YOLO  # lazy: heavy (torch) dependency
            self._model = YOLO(self.weights_path)
        return self._model

    def _detect(self, stripped: np.ndarray):
        """Run YOLO + white-shape on a stripped frame -> (white, boxes), as fusion does."""
        ws = detect_white_shape(stripped)
        res = self._ensure_model().predict(stripped, conf=self.conf, imgsz=self.imgsz,
                                           verbose=False)[0]
        boxes: list[tuple[float, float, float, float, float]] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cfs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cf in zip(xyxy, cfs):
                boxes.append((float(x1), float(y1), float(x2), float(y2), float(cf)))
        white = (ws.cx, ws.cy, ws.radius) if ws else None
        return white, boxes

    # ------------------------------------------------------------------- API
    def push_frame(self, frame_bgr: np.ndarray, *, detection=None):
        """Process one 744x498 BGR frame; return ``(x, y)`` in board-crop coords or ``None``.

        ``None`` is returned during the countdown/acquire phase (until the lock is
        confirmed) and on coast/acquire frames that have no emitted position. Pass
        ``detection=(white, boxes)`` (cached detections) to bypass YOLO -- used by the
        regression test so the box source matches the offline eval byte-for-byte.
        """
        idx = self.frame_idx
        self.frame_idx += 1
        stripped = strip_pointer(frame_bgr, strip_green=True)
        if detection is not None:
            white, boxes = detection
        else:
            white, boxes = self._detect(stripped)
        pack = FusionPack(idx, None, white, boxes)

        if not self._established:
            return self._buffer_countdown(idx, pack, stripped)
        gray = cv2.cvtColor(stripped, cv2.COLOR_BGR2GRAY)
        return self._step(pack, gray)

    # ------------------------------------------------------- countdown handoff
    def _buffer_countdown(self, idx: int, pack: FusionPack, stripped: np.ndarray):
        """Accumulate countdown frames; on the 3-miss confirm, compute the lock and
        replay the lock + the (already-buffered) first track frames. Returns the
        current frame's emitted point once the machine is running, else None."""
        self._cd_packs.append(pack)
        self._roll.append((idx, stripped))
        if pack.white is not None:
            self._seen_white = True
            self._last_white_idx = idx
            self._white_misses = 0
        elif self._seen_white:
            self._white_misses += 1

        if not (self._seen_white and self._white_misses >= _MISS_TO_CONFIRM):
            return None   # still inside the countdown -> nothing to emit yet

        # Countdown confirmed ended at this frame. Compute seed + lock exactly as the
        # batch path (compute_countdown_lock), using the buffered packs + the stripped
        # frame at the lock index (held in the rolling buffer).
        self._compute_lock()
        return self._replay_handoff()

    def _compute_lock(self) -> None:
        packs = self._cd_packs
        _sx, _sy, self.radius, self.start = _seed(packs)
        lock_pack = _countdown_white_pack(packs)
        self.lock = None
        self.lock_frame = self.start
        if lock_pack is None or not lock_pack.boxes or lock_pack.white is None:
            return
        anchor_info = _stable_white_anchor(packs)
        anchor = ((anchor_info[0], anchor_info[1]) if anchor_info is not None
                  else (lock_pack.white[0], lock_pack.white[1]))
        stripped_at_lock = next((s for i, s in self._roll if i == lock_pack.idx), None)
        mask = _white_mask(stripped_at_lock) if stripped_at_lock is not None else None
        best_i = _pick_lock_box(lock_pack.boxes, mask, lock_pack.white, anchor)
        box = lock_pack.boxes[best_i]
        cx, cy = _centroid(box)
        self.lock = LockInfo(box, cx, cy, lock_pack.idx)
        self.lock_frame = lock_pack.idx

    def _begin_track(self) -> None:
        """Initialise the track-phase machine at the lock frame (mirrors the batch
        loop's lock branch: pos = lock centroid, accumulators fresh)."""
        if self.lock is not None:
            cx, cy = self.lock.cx, self.lock.cy
        else:
            sx, sy, _, _ = _seed(self._cd_packs)
            cx, cy = sx, sy
        self._pos = np.array([cx, cy], np.float32)
        self._vel = np.zeros(2, np.float32)
        self._active = True
        self._prev_cents = None
        self._prev_alpha = None
        self._lost = 0
        self._mass_scale = self._cmass_scale = self._curl_scale = 0.0
        self._ovecs_hist = deque(maxlen=self.fuse_win)
        self._ema_cm = None
        self._ema_cents = None
        self._hyst_streak = 0
        self._hpos = None
        self._hwin = deque(maxlen=self.hedge_win)
        self._hprev_cent = None
        self._rhist = deque(maxlen=self.freeze_n + 1)
        self._freeze_buf = deque(maxlen=max(self.freeze_lag, 1))
        self._fpos = None
        self._frozen = False
        self._low_streak = 0
        self._hcursor = None

    def _replay_handoff(self):
        """Run the lock frame + the buffered track frames (lock_frame+1 .. current)
        through the machine. Returns the current (last) frame's emitted point."""
        self._begin_track()
        # The lock frame itself emits an (unscored) acquire point and seeds prev_gray.
        roll = list(self._roll)
        lock_entry = next(((i, s) for i, s in roll if i == self.lock_frame), None)
        if lock_entry is None:
            # No countdown white at all (degenerate); nothing locked -> keep buffering.
            self._established = False
            return None
        lf_idx, lf_strip = lock_entry
        self.track.append(TrackPoint(lf_idx, float(self._pos[0]), float(self._pos[1]),
                                     "acquire", len(self._packs_at(lf_idx).boxes)))
        self._prev_gray = cv2.cvtColor(lf_strip, cv2.COLOR_BGR2GRAY)
        self._established = True
        out = None
        for i, strip in roll:
            if i <= self.lock_frame:
                continue
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            out = self._step(self._packs_at(i), gray)
        # Countdown buffers no longer needed.
        self._cd_packs = []
        self._roll = deque(maxlen=_MISS_TO_CONFIRM + 1)
        return out

    def _packs_at(self, idx: int) -> FusionPack:
        return self._cd_packs[idx - self._cd_packs[0].idx]

    # ------------------------------------------------------- per-frame machine
    def _step(self, p: FusionPack, gray: np.ndarray):
        """One iteration of the fpath_human track-phase body (port of
        identity.track_fused_path_identity). Appends a TrackPoint and returns the
        emitted (x, y), or None for an acquire/coast frame with no scored output."""
        radius = self.radius
        x, y, state = float(self._pos[0]), float(self._pos[1]), "coast"
        cur_affine = None
        frame_cents = None
        if self._prev_gray is not None and self._active and p.boxes:
            fld = estimate_motion(self._prev_gray, gray)
            cur_affine = fld.affine
            sal = saliency_map(fld, gray.shape)
            cents = [_centroid(b) for b in p.boxes]
            mass = np.array([_box_saliency_mass(b, sal) for b in p.boxes], np.float32)
            fmax = float(mass.max())
            self._mass_scale = (fmax if self._mass_scale == 0.0
                                else max(fmax, self.mass_ema * self._mass_scale
                                         + (1 - self.mass_ema) * fmax))
            mass = mass / self._mass_scale if self._mass_scale > 0 else mass
            # additive windowed coherent-mass + curl channels (fpath_fuse)
            cmass = np.zeros(len(p.boxes), np.float32)
            curl = np.zeros(len(p.boxes), np.float32)
            self._ovecs_hist.append((fld.outliers, fld.outlier_vectors))
            win = list(self._ovecs_hist)
            if self.cmass_w > 0:
                cm = np.array([_box_coherent_mass(b, win) for b in p.boxes], np.float32)
                cpk = float(cm.max())
                self._cmass_scale = (cpk if self._cmass_scale == 0.0
                                     else max(cpk, self.mass_ema * self._cmass_scale
                                              + (1 - self.mass_ema) * cpk))
                cmass = cm / self._cmass_scale if self._cmass_scale > 0 else cm
            if self.curl_w > 0:
                cu = np.array([_box_rotational_curl(b, win) for b in p.boxes], np.float32)
                upk = float(cu.max())
                self._curl_scale = (upk if self._curl_scale == 0.0
                                    else max(upk, self.mass_ema * self._curl_scale
                                             + (1 - self.mass_ema) * upk))
                curl = cu / self._curl_scale if self._curl_scale > 0 else cu
            # leaky EMA of instantaneous coherent-mass, nearest-centroid carried (fpath_hyst)
            use_hyst = self.hyst_alpha > 0 and self.hyst_consec > 0
            if use_hyst:
                inst_cm = np.array(
                    [_box_coherent_mass(b, [(fld.outliers, fld.outlier_vectors)])
                     for b in p.boxes], np.float32)
                if self._ema_cm is not None and self._ema_cents is not None:
                    new_ema = inst_cm.copy()
                    for i, (cx, cy) in enumerate(cents):
                        bd, bj = 1e18, -1
                        for pj, (px, py) in enumerate(self._ema_cents):
                            dd = (px - cx) ** 2 + (py - cy) ** 2
                            if dd < bd:
                                bd, bj = dd, pj
                        if bj >= 0:
                            new_ema[i] = (self.hyst_alpha * self._ema_cm[bj]
                                          + (1 - self.hyst_alpha) * inst_cm[i])
                    self._ema_cm = new_ema
                else:
                    self._ema_cm = inst_cm
                self._ema_cents = cents
            # emission = additive fuse (mass + cmass + curl); coh/prox weights are 0 in
            # fpath_human so those terms drop out (see identity.track_fused_path_identity).
            emis = (mass + self.cmass_w * cmass + self.curl_w * curl).astype(np.float32)
            if self._prev_alpha is None or self._prev_cents is None:
                alpha = emis.copy()
            else:
                alpha = np.empty(len(cents), np.float32)
                for i, (cx, cy) in enumerate(cents):
                    best = -1e18
                    for j, (px, py) in enumerate(self._prev_cents):
                        dd = math.hypot(cx - px, cy - py) / radius
                        cost = self.trans_w * dd * dd
                        v = self._prev_alpha[j] - cost
                        if v > best:
                            best = v
                    alpha[i] = best + emis[i]
            alpha = alpha - float(alpha.max())
            choice = int(np.argmax(alpha))
            # EMA-coherent-mass hysteresis override (fpath_hyst)
            hyst_fired = False
            if use_hyst and self._ema_cm is not None:
                chal = int(np.argmax(self._ema_cm))
                if (chal != choice and self._ema_cm[chal]
                        > (1.0 + self.hyst_margin) * max(float(self._ema_cm[choice]), 1e-9)):
                    self._hyst_streak += 1
                else:
                    self._hyst_streak = 0
                if self._hyst_streak >= self.hyst_consec:
                    choice = chal
                    hyst_fired = True
                    self._hyst_streak = 0
                    self._vel[:] = 0.0
            nx, ny = cents[choice]
            self._vel = (self.vel_damp * self._vel
                         + (1.0 - self.vel_damp) * (np.array([nx, ny], np.float32) - self._pos))
            x, y, state = nx, ny, "track"
            frame_cents = cents
            self._prev_alpha, self._prev_cents = (None, None) if hyst_fired else (alpha, cents)
            self._lost = 0
        else:
            self._pos = self._pos + self._vel
            self._vel = self._vel * self.vel_damp
            self._prev_alpha, self._prev_cents = None, None
            self._lost += 1

        if self.fw is not None:
            x = float(np.clip(x, 0, self.fw - 1))
            y = float(np.clip(y, 0, self.fh - 1))
        self._pos = np.array([x, y], np.float32)

        # residual-gated freeze (fpath_freeze), BEFORE the hedge
        invaff = (cv2.invertAffineTransform(np.asarray(cur_affine, np.float64))
                  if cur_affine is not None else None)
        self._rhist.append((frame_cents, invaff))
        committed = np.array([x, y], np.float64)
        rv = (_cumulative_residual(self._rhist, committed, self.freeze_n, radius)
              if state == "track" else None)
        low = rv is not None and rv < self.freeze_tau
        self._low_streak = self._low_streak + 1 if low else 0
        if self._low_streak >= self.freeze_consec:
            if not self._frozen:
                self._fpos = (np.asarray(self._freeze_buf[0], np.float64)
                              if (self.freeze_lag > 0 and len(self._freeze_buf) > 0)
                              else committed.copy())
                self._frozen = True
        else:
            self._frozen = False
            self._fpos = committed.copy()
        self._freeze_buf.append(committed.copy())
        x, y = float(self._fpos[0]), float(self._fpos[1])

        out_x, out_y = x, y
        # decode-layer churn-gated freeze hedge (fpath_hedge)
        c_t = np.array([x, y], np.float64)
        if state == "track" and cur_affine is not None and self._hprev_cent is not None:
            T = cur_affine
            self._hwin.append(c_t - (T[:, :2] @ self._hprev_cent + T[:, 2]))
        if state == "track":
            self._hprev_cent = c_t.copy()
        if self._hwin:
            mags = [float(math.hypot(v[0], v[1])) for v in self._hwin]
            s = np.sum(self._hwin, axis=0)
            tot_m = sum(mags)
            R = float(math.hypot(s[0], s[1]) / tot_m) if tot_m > 1e-9 else 0.0
            churn = (tot_m / len(mags)) * (1.0 - R)
            w = min(max(1.0 - churn / self.hedge_churn_hi, 0.0), 1.0)
        else:
            w = 1.0
        self._hpos = c_t.copy() if self._hpos is None else w * c_t + (1.0 - w) * self._hpos
        out_x, out_y = float(self._hpos[0]), float(self._hpos[1])

        # human-cursor output dynamics (fpath_human): LAST transform on the emitted point
        if p.idx >= self.start:
            if self._hcursor is None:
                self._hcursor = HumanCursor(fps=60.0)
            out_x, out_y = self._hcursor.update((out_x, out_y))

        self.track.append(TrackPoint(p.idx, out_x, out_y, state, len(p.boxes)))
        self._prev_gray = gray
        return (out_x, out_y) if p.idx >= self.start else None
