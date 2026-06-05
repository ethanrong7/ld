"""Template matching inside the tracker gate (Step 11b).

Post-handoff the target is a relief outline. We match an edge template derived
from the countdown mask against gradient magnitude in a local ROI. Asymmetric
shapes search a small θ window; symmetric (circle-like) templates search
translation only.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ld.config import (
    TEMPLATE_CIRCULARITY,
    TEMPLATE_MATCH_MIN,
    TEMPLATE_THETA_RANGE,
    TEMPLATE_THETA_STEP,
)


def is_symmetric_template(
    template: np.ndarray,
    omega_deg_per_frame: float,
    angle_span_deg: float,
) -> bool:
    """True when rotation is not a usable discriminator (e.g. circle)."""
    cnts, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri > 0:
            circ = 4.0 * math.pi * area / (peri * peri)
            if circ >= TEMPLATE_CIRCULARITY:
                return True
    if abs(omega_deg_per_frame) < 0.6 and angle_span_deg < 20.0:
        return True
    return False


def _gradient_mag(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _edge_template(template: np.ndarray) -> np.ndarray:
    """Ring/outline template — post-handoff target is embossed border only."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(template, k, iterations=2)
    ero = cv2.erode(template, k, iterations=2)
    ring = cv2.subtract(dil, ero)
    if cv2.countNonZero(ring) < 50:
        ring = cv2.Canny(template, 40, 120)
        ring = cv2.dilate(ring, np.ones((3, 3), np.uint8), iterations=1)
    return ring


def _crop_centered(template: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return tight template patch and (center_offset_x, center_offset_y) in patch coords."""
    ys, xs = np.nonzero(template)
    if len(xs) == 0:
        return template, template.shape[1] / 2, template.shape[0] / 2
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    patch = template[y0:y1, x0:x1]
    tcx = (template.shape[1] - 1) / 2.0 - x0
    tcy = (template.shape[0] - 1) / 2.0 - y0
    return patch, tcx, tcy


def _rotate_patch(patch: np.ndarray, degrees: float) -> np.ndarray:
    h, w = patch.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(
        patch, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def _extract_patch(gray: np.ndarray, cx: float, cy: float, half: int) -> np.ndarray | None:
    h, w = gray.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    x0, x1 = max(0, ix - half), min(w, ix + half)
    y0, y1 = max(0, iy - half), min(h, iy + half)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    ph, pw = 2 * half, 2 * half
    canvas = np.zeros((ph, pw), np.float32)
    canvas[0 : patch.shape[0], 0 : patch.shape[1]] = patch
    return cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _match_patch_in_roi(
    roi: np.ndarray,
    patch: np.ndarray,
    x0: int,
    y0: int,
    px: float,
    py: float,
    search_radius: float,
    tcx: float,
    tcy: float,
) -> tuple[float, float, float] | None:
    ph, pw = patch.shape[:2]
    if ph >= roi.shape[0] or pw >= roi.shape[1]:
        return None
    res = cv2.matchTemplate(roi, patch, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    cx = x0 + max_loc[0] + tcx
    cy = y0 + max_loc[1] + tcy
    if (cx - px) ** 2 + (cy - py) ** 2 > search_radius ** 2:
        return None
    if max_val < TEMPLATE_MATCH_MIN:
        return None
    return cx, cy, float(max_val)


def match_in_gate(
    gray: np.ndarray,
    template: np.ndarray,
    px: float,
    py: float,
    ptheta: float,
    search_radius: float,
    *,
    symmetric: bool,
    appearance: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Best (x, y, score) for template center in the gate, or None."""
    edge_t, tcx, tcy = _crop_centered(_edge_template(template))
    th, tw = edge_t.shape[:2]
    if th < 8 or tw < 8:
        return None

    pad = int(math.ceil(search_radius)) + max(th, tw)
    h, w = gray.shape[:2]
    ix, iy = int(round(px)), int(round(py))
    x0 = max(0, ix - pad)
    y0 = max(0, iy - pad)
    x1 = min(w, ix + pad + 1)
    y1 = min(h, iy + pad + 1)
    if x1 - x0 < tw + 4 or y1 - y0 < th + 4:
        return None

    roi = _gradient_mag(gray[y0:y1, x0:x1])

    if symmetric:
        thetas = [0.0]
    else:
        thetas = [
            ptheta + float(d)
            for d in np.arange(
                -TEMPLATE_THETA_RANGE,
                TEMPLATE_THETA_RANGE + 0.01,
                TEMPLATE_THETA_STEP,
            )
        ]

    best_score = -1.0
    best_xy: tuple[float, float] | None = None

    if symmetric and appearance is not None:
        ap = _match_patch_in_roi(
            roi,
            appearance,
            x0,
            y0,
            px,
            py,
            search_radius,
            appearance.shape[1] / 2,
            appearance.shape[0] / 2,
        )
        if ap is not None:
            return ap
        return None

    for deg in thetas:
        t_rot = _rotate_patch(edge_t, float(deg))
        rh, rw = t_rot.shape[:2]
        if rh >= roi.shape[0] or rw >= roi.shape[1]:
            continue
        res = cv2.matchTemplate(roi, t_rot, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val <= best_score:
            continue
        # max_loc is top-left of match; convert to frame center using template centroid offset
        cx = x0 + max_loc[0] + tcx
        cy = y0 + max_loc[1] + tcy
        if (cx - px) ** 2 + (cy - py) ** 2 > search_radius ** 2:
            continue
        best_score = float(max_val)
        best_xy = (cx, cy)

    if best_xy is None or best_score < TEMPLATE_MATCH_MIN:
        return None
    return best_xy[0], best_xy[1], best_score
