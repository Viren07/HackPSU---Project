"""Benchmark the pipeline against synthetic ground truth.

Error metric: for each true point, |predicted y - true y| as a % of the y-axis
range (so "1.5%" means off by 1.5% of the axis height). Series are matched to
ground truth by color.

Usage:
  python -m synth.generate --n 50 --out data/synth
  python -m scripts.evaluate --data data/synth
  python -m synth.generate --n 50 --out data/synth_cam --camera
  python -m scripts.evaluate --data data/synth_cam --rectify
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline import run


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def score_sample(res, truth):
    if not res.get("ok") or not res.get("calibrated"):
        return None, "failed" if not res.get("ok") else "uncalibrated"
    yr = truth["ylim"][1] - truth["ylim"][0]
    errs = []
    preds = [s for s in res["series"] if "x" in s]
    for ts in truth["series"]:
        if not preds:
            return None, "no_series"
        tc = hex_rgb(ts["color"])
        best = min(preds, key=lambda s: np.linalg.norm(np.array(s["color_rgb"]) - tc))
        px, py = np.array(best["x"]), np.array(best["y"])
        order = np.argsort(px)
        px, py = px[order], py[order]
        tx, ty = np.array(ts["x"]), np.array(ts["y"])
        inside = (tx >= px.min()) & (tx <= px.max())
        if truth["kind"] == "line":
            yi = np.interp(tx[inside], px, py)
            errs += list(np.abs(yi - ty[inside]) / yr * 100)
        else:
            for x, y in zip(tx, ty):
                d = np.hypot((px - x) / (truth["xlim"][1] - truth["xlim"][0]), (py - y) / yr)
                errs.append(d.min() * 100)
    return float(np.mean(errs)) if errs else None, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--rectify", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.data)
    labels = sorted((d / "labels").glob("*.json"))
    if args.limit:
        labels = labels[:args.limit]
    errs, status, matches = [], {}, []
    for lp in labels:
        truth = json.loads(lp.read_text())
        img = cv2.cvtColor(cv2.imread(str(d / "images" / f"{lp.stem}.png")), cv2.COLOR_BGR2RGB)
        res = run(img, do_rectify=args.rectify, want_images=False)
        e, st = score_sample(res, truth)
        status[st] = status.get(st, 0) + 1
        if e is not None:
            errs.append(e)
            matches.append(res["overall_match"])
        print(f"{lp.stem}: {st:13s} err={'%.2f%%' % e if e is not None else '-':>8}  "
              f"match={res.get('overall_match', '-')}")
    print("\n=== summary ===")
    print("status:", status)
    if errs:
        errs = np.array(errs)
        print(f"mean error: {errs.mean():.2f}% of axis range   median: {np.median(errs):.2f}%")
        print(f"samples under 2% error: {(errs < 2).mean() * 100:.0f}%")
        m = np.array(matches)
        if len(m) > 3:
            print(f"corr(match score, error): {np.corrcoef(m, errs)[0, 1]:.2f}  (negative = verification works)")


if __name__ == "__main__":
    main()
