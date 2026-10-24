"""Step 2: find the plot area (the rectangle bounded by the axis lines).

Axis spines are long, dark, straight lines. Real journal figures often also
have an outer frame around the whole figure, so we look for L-shaped corners
(a horizontal line whose left end meets a vertical line's bottom end) and pick
the tightest such box that still contains most of the colored data.
(Later: back this up with the YOLO 'plot_area' detector.)
"""
from __future__ import annotations

import cv2
import numpy as np


def _dark_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    local = cv2.GaussianBlur(gray, (0, 0), 15)
    dark = ((gray.astype(int) < local.astype(int) - 40) | (gray < 110)) & (hsv[:, :, 1] < 90)
    return dark.astype(np.uint8) * 255


def _segments(mask, horizontal: bool, min_len: int):
    """Connected straight segments as (x0, y0, x1, y1)."""
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    segs = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if (w if horizontal else h) >= min_len:
            segs.append((int(x), int(y), int(x + w - 1), int(y + h - 1)))
    return segs


def _colored_fraction(img, box, colored):
    x0, y0, x1, y1 = box
    tot = colored.sum()
    return colored[y0:y1 + 1, x0:x1 + 1].sum() / tot if tot else 0.0


def find_plot_box(img: np.ndarray):
    """Return (x0, y0, x1, y1) in pixels, or None."""
    h, w = img.shape[:2]
    dark = _dark_mask(img)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 6), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 6)))
    hl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk)
    vl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
    # small vertical closing so tick marks don't split a line
    hl = cv2.morphologyEx(hl, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    vl = cv2.morphologyEx(vl, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
    hs = _segments(hl, True, int(0.2 * w))
    vs = _segments(vl, False, int(0.2 * h))
    if not hs or not vs:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    colored = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50)).astype(np.uint8)

    tol = max(8, int(0.015 * max(w, h)))
    cands = []
    for hx0, hy0, hx1, hy1 in hs:
        y_axis = (hy0 + hy1) // 2
        for vx0, vy0, vx1, vy1 in vs:
            x_axis = (vx0 + vx1) // 2
            # L corner: vertical line's bottom meets horizontal line's left end
            if abs(vy1 - y_axis) <= tol and abs(hx0 - x_axis) <= tol:
                top = vy0
                # right edge: a vertical line rising from the right end, else the x-line's end
                right = hx1
                box = (x_axis, top, right, y_axis)
                if right - x_axis < 0.2 * w or y_axis - top < 0.2 * h:
                    continue
                cands.append(box)
    if cands:
        scored = []
        for b in cands:
            frac = _colored_fraction(img, b, colored)
            area = (b[2] - b[0]) * (b[3] - b[1])
            scored.append((frac >= 0.6, -area, b))
        # prefer boxes holding the colored data, then the smallest one
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        best = scored[0][2]
        return tuple(int(v) for v in best)

    # fallback: bottom-most long horizontal + left-most long vertical
    bottom = max(s[3] for s in hs)
    left = min(s[0] for s in vs)
    right = max(s[2] for s in hs if s[3] == bottom)
    top = min(s[1] for s in vs if s[0] == left)
    if right - left < 0.2 * w or bottom - top < 0.2 * h:
        return None
    return int(left), int(top), int(right), int(bottom)
