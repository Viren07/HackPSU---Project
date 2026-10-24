"""The full PriorArt Lens pipeline: photo in, verified data out."""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from .axes import find_plot_box
from .rectify import rectify
from .series import extract_series, legend_mask
from .ticks import calibrate, ocr_axis_strips, run_ocr
from .verify import redraw_chart, verify


def _png_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def _overlay(img, series, box, color=(255, 0, 255)):
    out = img.copy()
    if box:
        cv2.rectangle(out, box[:2], box[2:], (0, 200, 255), 2)
    for s in series:
        for x, y in zip(s.px[::2], s.py[::2]):
            cv2.circle(out, (int(x), int(y)), 3, (0, 0, 0), -1)
            cv2.circle(out, (int(x), int(y)), 2, color, -1)
    return out


def _overlay_on_photo(photo, series, H):
    if np.allclose(H, np.eye(3)):
        return None
    Hinv = np.linalg.inv(H)
    out = photo.copy()
    for s in series:
        pts = np.stack([s.px[::2], s.py[::2]], 1).astype(np.float32)[None]
        back = cv2.perspectiveTransform(pts, Hinv)[0]
        for x, y in back:
            cv2.circle(out, (int(x), int(y)), 4, (0, 0, 0), -1)
            cv2.circle(out, (int(x), int(y)), 3, (255, 0, 255), -1)
    return out


def run(img_rgb: np.ndarray, do_rectify: bool = True, want_images: bool = True) -> dict:
    t0 = time.time()
    timings = {}
    photo = img_rgb
    if do_rectify:
        flat, H, found = rectify(photo)
    else:
        flat, H, found = photo, np.eye(3), False
    timings["rectify"] = time.time() - t0

    box = find_plot_box(flat)
    timings["axes"] = time.time() - t0
    if box is None:
        return {"ok": False, "error": "Could not find the plot axes. Try a straighter, closer photo.",
                "rectified": found, "rectified_png": _png_b64(flat) if want_images else None}

    texts = run_ocr(flat)
    xfit, yfit, others = calibrate(texts, box, flat.shape, ocr_axis_strips(flat, box))
    timings["ocr"] = time.time() - t0

    series = extract_series(flat, box, legend_mask(others, box, flat.shape))
    checks = verify(series, box)
    timings["series+verify"] = time.time() - t0

    series_out = []
    for s, chk in zip(series, checks):
        d = {"color_rgb": list(s.color_rgb), "kind": s.kind, "n_points": int(len(s.px)), "verify": chk,
             "px": s.px.round(1).tolist(), "py": s.py.round(1).tolist()}
        if xfit and yfit:
            d["x"] = np.round(xfit.to_value(s.px), 6).tolist()
            d["y"] = np.round(yfit.to_value(s.py), 6).tolist()
        series_out.append(d)

    def axis_info(f):
        if f is None:
            return None
        return {"log": f.log, "ticks_found": f.n_ticks, "ticks_consistent": f.n_inliers,
                "residual_px": round(f.residual_px, 2), "ticks": [[v, round(p, 1)] for v, p in f.ticks]}

    calibrated = bool(xfit and yfit)
    match = float(np.mean([c["match"] for c in checks])) if checks else 0.0
    result = {
        "ok": True,
        "calibrated": calibrated,
        "rectified": found,
        "plot_box": list(box),
        "x_axis": axis_info(xfit),
        "y_axis": axis_info(yfit),
        "series": series_out,
        "overall_match": round(match, 3),
        "warnings": [],
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
    }
    if not calibrated:
        result["warnings"].append("Could not read enough tick labels on one axis; values are in pixels.")
    if checks and match < 0.8:
        result["warnings"].append("Low render-and-compare match: check the overlay before trusting these numbers.")
    if not series:
        result["warnings"].append("No colored data series found (black/gray series are not supported yet).")

    if want_images:
        result["rectified_png"] = _png_b64(_overlay(flat, series, box))
        ph = _overlay_on_photo(photo, series, H)
        result["photo_overlay_png"] = _png_b64(ph) if ph is not None else None
        if calibrated and series_out:
            png = redraw_chart(series_out, xfit, yfit, box, (box[2] - box[0] + 160, box[3] - box[1] + 120))
            result["redrawn_png"] = base64.b64encode(png).decode()
    return result
