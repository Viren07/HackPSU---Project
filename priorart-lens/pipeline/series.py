"""Step 4: separate the data series by color and trace them.

Colored pixels inside the plot box are clustered in Lab color space. Each
cluster becomes a series; it is traced column-by-column (line) or by blob
centroids (scatter). Legend swatches are masked out using the legend text OCR
already found inside the plot area.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class Series:
    color_rgb: tuple
    kind: str  # "line" | "scatter"
    px: np.ndarray  # pixel x (full-image coords)
    py: np.ndarray  # pixel y
    mask: np.ndarray  # pixel mask (plot-box crop) used for verification
    thickness: float


def legend_mask(others, box, shape):
    """Mask covering the legend: non-numeric text inside the plot + swatch area to its left."""
    x0, y0, x1, y1 = box
    m = np.zeros(shape[:2], np.uint8)
    inside = [t for t in others if x0 < t.cx < x1 and y0 < t.cy < y1 and any(c.isalpha() for c in t.text)]
    for t in inside:
        hgt = max(8, t.height)
        lx = int(max(x0, t.box[0] - 3.5 * hgt))
        cv2.rectangle(m, (lx, int(t.box[1] - 0.4 * hgt)), (int(t.box[2] + 4), int(t.box[3] + 0.4 * hgt)), 255, -1)
    return m


def _cluster_colors(lab_px: np.ndarray, k_max=6, merge_dist=28.0, min_frac=0.03):
    n = len(lab_px)
    if n < 30:
        return []
    sample = lab_px[np.random.default_rng(0).choice(n, min(n, 6000), replace=False)]
    k = min(k_max, len(np.unique(sample, axis=0)))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(sample)
    centers = list(km.cluster_centers_)
    counts = list(np.bincount(km.labels_, minlength=k).astype(float))
    merged = True
    while merged and len(centers) > 1:
        merged = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if np.linalg.norm(centers[i] - centers[j]) < merge_dist:
                    tot = counts[i] + counts[j]
                    centers[i] = (centers[i] * counts[i] + centers[j] * counts[j]) / tot
                    counts[i] = tot
                    del centers[j], counts[j]
                    merged = True
                    break
            if merged:
                break
    total = sum(counts)
    return [c for c, n_ in zip(centers, counts) if n_ / total >= min_frac]


def extract_series(img: np.ndarray, box, legend: np.ndarray | None = None, step: int = 2):
    x0, y0, x1, y1 = box
    pad = 4
    crop = img[y0 + pad:y1 - pad, x0 + pad:x1 - pad]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    colored = (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50)
    if legend is not None:
        colored &= legend[y0 + pad:y1 - pad, x0 + pad:x1 - pad] == 0

    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    ys, xs = np.nonzero(colored)
    centers = _cluster_colors(lab[ys, xs])
    if not centers:
        return []

    # assign each colored pixel to nearest center
    C = np.array(centers)
    d = np.linalg.norm(lab[ys, xs][:, None, :] - C[None], axis=2)
    lbl = d.argmin(1)
    near = d.min(1) < 40

    out = []
    for ci in range(len(C)):
        m = np.zeros(colored.shape, np.uint8)
        sel = (lbl == ci) & near
        m[ys[sel], xs[sel]] = 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n_cc, cc, stats, cent = cv2.connectedComponentsWithStats(m)
        if n_cc <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        cw = crop.shape[1]
        # scatter: many small blobs of similar size; line: dominated by wide components
        is_scatter = (n_cc - 1) >= 6 and widths.max() < 0.08 * cw
        rgb = cv2.cvtColor(np.uint8([[C[ci]]]), cv2.COLOR_LAB2RGB)[0, 0]
        if is_scatter:
            keep = areas > 0.3 * np.median(areas)
            pts = cent[1:][keep]
            order = np.argsort(pts[:, 0])
            px, py = pts[order, 0], pts[order, 1]
            thick = float(np.sqrt(np.median(areas[keep]) / np.pi))
            kind = "scatter"
        else:
            # drop small specks (anti-aliasing leftovers, other-series crossings)
            keep_ids = np.where(areas >= max(15, 0.02 * areas.max()))[0] + 1
            m = np.isin(cc, keep_ids).astype(np.uint8) * 255
            cols = np.arange(0, m.shape[1], step)
            px, py = [], []
            for c in cols:
                rr = np.nonzero(m[:, c])[0]
                if len(rr):
                    px.append(c)
                    py.append(np.median(rr))
            if len(px) < 5:
                continue
            px, py = np.array(px, float), np.array(py, float)
            length = np.hypot(np.diff(px), np.diff(py)).sum()
            thick = float(max(1.0, (m > 0).sum() / max(length, 1)))
            kind = "line"
        out.append(Series(tuple(int(v) for v in rgb), kind, px + x0 + pad, py + y0 + pad, m, thick))
    return out
