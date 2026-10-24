"""Synthetic chart generator.

Makes matplotlib line/scatter plots with exact ground truth, so we can
(1) benchmark the extraction pipeline and (2) later train a YOLO detector.

Each sample writes:
  <out>/images/<id>.png         the chart (optionally "photographed")
  <out>/labels/<id>.json        ground truth: series data, axis ranges, plot box
  <out>/yolo/<id>.txt           YOLO boxes: 0=plot_area 1=x_tick_label 2=y_tick_label 3=legend

Usage:
  python -m synth.generate --n 200 --out data/synth
  python -m synth.generate --n 200 --out data/synth_cam --camera
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
YOLO_CLASSES = ["plot_area", "x_tick_label", "y_tick_label", "legend"]


def _random_series(rng: np.random.Generator, n_series: int, kind: str):
    x0 = float(rng.choice([0, 0, 1, 10, 100]))
    span = float(rng.choice([1, 5, 10, 50, 100, 1000]))
    n = int(rng.integers(15, 60)) if kind == "line" else int(rng.integers(10, 40))
    xs = np.sort(rng.uniform(x0, x0 + span, n)) if kind == "scatter" else np.linspace(x0, x0 + span, n)
    scale = float(rng.choice([1, 10, 100, 0.1]))
    out = []
    for _ in range(n_series):
        a, b, c = rng.normal(0, 1, 3)
        t = (xs - x0) / span
        ys = scale * (a * t + b * np.sin(3 * t + c) + 0.15 * rng.normal(0, 1, n) + 3)
        out.append((xs, ys))
    return out


def make_chart(seed: int, kind: str | None = None):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    kind = kind or ("line" if rng.random() < 0.65 else "scatter")
    n_series = int(rng.integers(1, 4))
    series = _random_series(rng, n_series, kind)
    colors = random.sample(PALETTE, n_series)

    w, h = float(rng.uniform(5, 8)), float(rng.uniform(3.5, 5.5))
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    for i, ((xs, ys), col) in enumerate(zip(series, colors)):
        label = f"Method {chr(65 + i)}"
        if kind == "line":
            ax.plot(xs, ys, color=col, lw=float(rng.uniform(1.5, 2.8)), label=label)
        else:
            ax.scatter(xs, ys, color=col, s=float(rng.uniform(15, 40)), label=label)
    if rng.random() < 0.4:
        ax.grid(True, alpha=0.3)
    ax.set_xlabel(random.choice(["Epoch", "Time (s)", "Iterations", "Dataset size"]))
    ax.set_ylabel(random.choice(["Accuracy", "Loss", "mAP", "Error (%)"]))
    has_legend = n_series > 1 and rng.random() < 0.8
    if has_legend:
        ax.legend(loc=random.choice(["upper left", "upper right", "lower right", "best"]))
    fig.tight_layout()
    fig.canvas.draw()

    W, H = fig.canvas.get_width_height()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    renderer = fig.canvas.get_renderer()

    def to_px(bbox):
        # matplotlib display coords have origin bottom-left
        return [float(bbox.x0), float(H - bbox.y1), float(bbox.x1), float(H - bbox.y0)]

    plot_box = to_px(ax.get_window_extent(renderer))
    boxes = [(0, plot_box)]
    for lbl in ax.get_xticklabels():
        if lbl.get_visible() and lbl.get_text():
            boxes.append((1, to_px(lbl.get_window_extent(renderer))))
    for lbl in ax.get_yticklabels():
        if lbl.get_visible() and lbl.get_text():
            boxes.append((2, to_px(lbl.get_window_extent(renderer))))
    if has_legend and ax.get_legend() is not None:
        boxes.append((3, to_px(ax.get_legend().get_window_extent(renderer))))

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    plt.close(fig)

    truth = {
        "kind": kind,
        "image_size": [W, H],
        "plot_box": plot_box,
        "xlim": list(xlim),
        "ylim": list(ylim),
        "series": [
            {"color": c, "x": xs.tolist(), "y": ys.tolist()} for (xs, ys), c in zip(series, colors)
        ],
    }
    return img, truth, boxes


def photograph(img: np.ndarray, rng: np.random.Generator):
    """Fake a phone photo: paper margin, perspective tilt, lighting, blur, noise.

    Returns the distorted image and the 3x3 homography applied (chart -> photo).
    """
    h, w = img.shape[:2]
    pad = int(0.12 * max(w, h))
    paper = np.full((h + 2 * pad, w + 2 * pad, 3), 245, np.uint8)
    paper[pad:pad + h, pad:pad + w] = img
    ph, pw = paper.shape[:2]

    canvas_w, canvas_h = int(pw * 1.3), int(ph * 1.3)
    j = 0.08
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    off = np.float32([(canvas_w - pw) / 2, (canvas_h - ph) / 2])
    dst = src + off + rng.uniform(-j, j, (4, 2)).astype(np.float32) * np.float32([pw, ph])
    M = cv2.getPerspectiveTransform(src, dst)
    bg = rng.integers(40, 110, 3).tolist()
    out = cv2.warpPerspective(paper, M, (canvas_w, canvas_h), borderValue=bg)

    # uneven lighting
    gx = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.9, 1.1), canvas_w)
    gy = np.linspace(rng.uniform(0.85, 1.05), rng.uniform(0.9, 1.1), canvas_h)
    out = np.clip(out * np.outer(gy, gx)[..., None], 0, 255).astype(np.uint8)
    k = int(rng.choice([1, 3, 3, 5]))
    if k > 1:
        out = cv2.GaussianBlur(out, (k, k), 0)
    out = np.clip(out + rng.normal(0, 4, out.shape), 0, 255).astype(np.uint8)

    T = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], np.float64)
    return out, M @ T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--camera", action="store_true", help="apply fake phone-photo distortion")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("images", "labels", "yolo"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(YOLO_CLASSES))

    for i in range(args.n):
        seed = args.seed + i
        img, truth, boxes = make_chart(seed)
        H_img, W_img = img.shape[:2]
        if args.camera:
            img, Hm = photograph(img, np.random.default_rng(seed + 10_000))
            truth["homography"] = Hm.tolist()
            # warp boxes (axis-aligned bound of warped corners)
            new = []
            for cls, (x0, y0, x1, y1) in boxes:
                pts = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])[None]
                wp = cv2.perspectiveTransform(pts, Hm)[0]
                new.append((cls, [wp[:, 0].min(), wp[:, 1].min(), wp[:, 0].max(), wp[:, 1].max()]))
            boxes = new
            H_img, W_img = img.shape[:2]
        sid = f"{i:05d}"
        cv2.imwrite(str(out / "images" / f"{sid}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        (out / "labels" / f"{sid}.json").write_text(json.dumps(truth))
        lines = []
        for cls, (x0, y0, x1, y1) in boxes:
            cx, cy = (x0 + x1) / 2 / W_img, (y0 + y1) / 2 / H_img
            bw, bh = (x1 - x0) / W_img, (y1 - y0) / H_img
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "yolo" / f"{sid}.txt").write_text("\n".join(lines))
    print(f"wrote {args.n} samples to {out}")


if __name__ == "__main__":
    main()
