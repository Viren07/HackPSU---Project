"""Step 5: render-and-compare verification.

Redraw every extracted series from its extracted numbers, then compare the
redrawn pixels with the pixels actually in the image. If the tool misread a
series, the two will not line up and the score drops.

  precision = how much of what we drew is really there
  recall    = how much of what is really there we drew
  match     = F1 of the two (with a small pixel tolerance)
"""
from __future__ import annotations

import io

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def render_series_mask(s, box, shape):
    x0, y0, _, _ = box
    pad = 4
    m = np.zeros(shape, np.uint8)
    pts = np.stack([s.px - x0 - pad, s.py - y0 - pad], 1)
    t = max(1, int(round(s.thickness)))
    if s.kind == "line":
        # break the polyline where the trace has gaps (e.g. line hidden by another)
        gaps = np.where(np.diff(pts[:, 0]) > 12)[0] + 1
        for seg in np.split(pts, gaps):
            if len(seg) > 1:
                cv2.polylines(m, [seg.astype(np.int32)], False, 255, t)
    else:
        for x, y in pts:
            cv2.circle(m, (int(x), int(y)), t, 255, -1)
    return m


def compare(observed: np.ndarray, rendered: np.ndarray, tol: int = 2):
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    obs, ren = observed > 0, rendered > 0
    if ren.sum() == 0 or obs.sum() == 0:
        return 0.0, 0.0, 0.0
    precision = (ren & (cv2.dilate(observed, k) > 0)).sum() / ren.sum()
    recall = (obs & (cv2.dilate(rendered, k) > 0)).sum() / obs.sum()
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(precision), float(recall), float(f1)


def verify(series_list, box):
    results = []
    for s in series_list:
        ren = render_series_mask(s, box, s.mask.shape)
        p, r, f = compare(s.mask, ren)
        results.append({"precision": round(p, 3), "recall": round(r, 3), "match": round(f, 3)})
    return results


def redraw_chart(series_data, xfit, yfit, box, size) -> bytes:
    """Re-plot the extracted numbers as a clean matplotlib chart (PNG bytes)."""
    x0, y0, x1, y1 = box
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    for s in series_data:
        col = np.array(s["color_rgb"]) / 255
        if s["kind"] == "line":
            ax.plot(s["x"], s["y"], color=col, lw=2)
        else:
            ax.scatter(s["x"], s["y"], color=col, s=20)
    if xfit and yfit:
        ax.set_xlim(*sorted(xfit.to_value([x0, x1])))
        ax.set_ylim(*sorted(yfit.to_value([y1, y0])))
        if xfit.log:
            ax.set_xscale("log")
        if yfit.log:
            ax.set_yscale("log")
    ax.set_title("Redrawn from extracted data", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
