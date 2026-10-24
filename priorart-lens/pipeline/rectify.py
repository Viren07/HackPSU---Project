"""Step 1: undo camera perspective.

Find the sheet of paper / screen (the biggest bright quadrilateral) and warp it
flat. Returns the flattened image and the homography photo -> flat, so results
can be projected back onto the original photo for the AR-style overlay.
"""
from __future__ import annotations

import cv2
import numpy as np


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def find_page(img: np.ndarray, min_area_frac: float = 0.2):
    """Return 4 ordered corners of the page, or None if no clear page is found."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    s = scale if scale < 1 else 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # bright paper on darker background
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < min_area_frac * small.shape[0] * small.shape[1]:
            break
        if area > 0.97 * small.shape[0] * small.shape[1]:
            continue  # whole frame: image is already flat
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            best = approx
            break
    if best is None:
        return None
    return _order_corners(best / s)


def rectify(img: np.ndarray):
    """Flatten the page. Returns (flat_img, H_photo_to_flat, found: bool)."""
    corners = find_page(img)
    if corners is None:
        return img, np.eye(3), False
    tl, tr, br, bl = corners
    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    M = cv2.getPerspectiveTransform(corners, dst)
    flat = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC)
    # flatten lighting a little so color thresholds behave
    return flat, M, True
