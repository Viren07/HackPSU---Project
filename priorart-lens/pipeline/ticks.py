"""Step 3: read the tick labels and calibrate pixel -> data value.

OCR every text box, keep the numeric ones just below the x-axis / left of the
y-axis, then robustly fit value = a * pixel + b (or log10 for log axes).
The fit residual doubles as a calibration sanity check.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

_ocr = None


_ocr_small = None


def get_ocr():
    """Reader for whole figures."""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr = RapidOCR()
    return _ocr


def get_ocr_small():
    """Reader for small crops (single numbers, table cells). The default detector upscales any
    image smaller than 736 px, which makes each tiny crop ~10x slower for no accuracy gain."""
    global _ocr_small
    if _ocr_small is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_small = RapidOCR()
        # stop the detector from upscaling small crops to 736 px, and skip the rotation classifier
        # (internals differ between rapidocr versions; skip the speed-up if they are not there)
        det = getattr(_ocr_small, "text_detector", None) or getattr(_ocr_small, "text_det", None)
        for op in getattr(det, "preprocess_op", []) or []:
            if hasattr(op, "limit_type"):
                op.limit_type, op.limit_side_len = "max", 960
        _ocr_small.use_angle_cls = False
    return _ocr_small


@dataclass
class TextBox:
    text: str
    box: tuple  # x0, y0, x1, y1
    score: float

    @property
    def cx(self):
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self):
        return (self.box[1] + self.box[3]) / 2

    @property
    def height(self):
        return self.box[3] - self.box[1]


def run_ocr(img: np.ndarray, up: float | None = None, offset=(0, 0)) -> list[TextBox]:
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return []
    if up is None:
        up = 2.0 if max(h, w) < 1400 else 1.0
    big = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up != 1 else img
    # white border helps the text detector with labels touching the crop edge
    big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    result, _ = get_ocr()(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    out = []
    for quad, text, score in result or []:
        q = (np.array(quad, dtype=float) - 20) / up
        q[:, 0] += offset[0]
        q[:, 1] += offset[1]
        out.append(TextBox(text.strip(), (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()), float(score)))
    return out


def _label_blobs(strip: np.ndarray, horizontal: bool):
    """Find tick-label bounding boxes in an axis strip without the text detector.

    OCR text detectors often miss short isolated labels like '0' or '10', so we
    segment them ourselves: dark connected components, minus tick marks, grouped
    into labels by spacing.
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
    dark = (gray < min(140, int(np.median(gray)) - 50)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(dark)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 3:
            continue
        if horizontal and y <= 3 and h <= 9 and w <= 3:
            continue  # x tick mark hanging off the axis
        if not horizontal and x + w >= strip.shape[1] - 3 and h <= 3:
            continue  # y tick mark
        comps.append([x, y, x + w, y + h])
    if not comps:
        return []
    comps = np.array(comps)
    hs = comps[:, 3] - comps[:, 1]
    med_h = np.median(hs[hs > 3]) if (hs > 3).any() else np.median(hs)

    if horizontal:
        # keep the first text row (tick labels), not the axis title below it
        top = comps[:, 1].min()
        row = comps[comps[:, 1] < top + 0.6 * med_h]
        row = row[np.argsort(row[:, 0])]
        groups, cur = [], [row[0]]
        for c in row[1:]:
            if c[0] - cur[-1][2] > 0.6 * med_h:
                groups.append(cur)
                cur = [c]
            else:
                cur.append(c)
        groups.append(cur)
    else:
        comps = comps[hs < 2.5 * med_h]  # drop the rotated axis title
        if len(comps) == 0:
            return []
        # tick labels hug the axis: keep components whose right edge is near the right
        right = comps[:, 2].max()
        comps = comps[comps[:, 2] > right - 6 * med_h]
        comps = comps[np.argsort(comps[:, 1])]
        groups, cur = [], [comps[0]]
        for c in comps[1:]:
            cy0, cy1 = min(g[1] for g in cur), max(g[3] for g in cur)
            if c[1] < cy1 - 0.3 * med_h and c[3] > cy0:
                cur.append(c)
            else:
                groups.append(cur)
                cur = [c]
        groups.append(cur)
    boxes = []
    for g in groups:
        g = np.array(g)
        boxes.append((g[:, 0].min(), g[:, 1].min(), g[:, 2].max(), g[:, 3].max()))
    return boxes


def _read_blobs(strip: np.ndarray, boxes, offset):
    """Paste label crops side by side into one text line, OCR it once, map back."""
    if not boxes:
        return []
    target_h = 40
    pieces, spans, x = [], [], 30
    for (a, b, c, d) in boxes:
        crop = strip[max(0, b - 2):d + 2, max(0, a - 2):c + 2]
        s = target_h / max(1, crop.shape[0])
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        pieces.append((x, crop))
        spans.append((x, x + crop.shape[1]))
        x += crop.shape[1] + 45
    canvas = np.full((target_h + 40, x + 30, 3), 255, np.uint8)
    for px_, crop in pieces:
        canvas[20:20 + crop.shape[0], px_:px_ + crop.shape[1]] = crop
    # single-glyph labels get low recognizer confidence (~0.5); accept them here
    result, _ = get_ocr_small()(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), text_score=0.3)
    texts = {i: [] for i in range(len(boxes))}
    for quad, text, _score in result or []:
        q = np.array(quad, float)
        tokens = text.split()
        if not tokens:
            continue
        # spread tokens across the detected box, then assign each to the nearest span
        qx0, qx1 = q[:, 0].min(), q[:, 0].max()
        for k, tok in enumerate(tokens):
            cx = qx0 + (qx1 - qx0) * (k + 0.5) / len(tokens)
            i = int(np.argmin([abs(cx - (s0 + s1) / 2) for s0, s1 in spans]))
            texts[i].append(tok)
    out = []
    for i, (a, b, c, d) in enumerate(boxes):
        if texts[i]:
            out.append(TextBox("".join(texts[i]), (a + offset[0], b + offset[1], c + offset[0], d + offset[1]), 1.0))
    return out


def ocr_axis_strips(img: np.ndarray, box) -> list[TextBox]:
    """Read tick labels from the strips below the x-axis and left of the y-axis."""
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    band = int(0.12 * h)
    out = []
    a, b, c, d = max(0, x0 - 30), min(h - 1, y1 + 2), min(w, x1 + 30), min(h, y1 + band)
    if d - b > 4:
        strip = img[b:d, a:c]
        out += _read_blobs(strip, _label_blobs(strip, True), (a, b))
    a, b, c, d = 0, max(0, y0 - 15), max(1, x0 - 2), min(h, y1 + 15)
    if c - a > 4:
        strip = img[b:d, a:c]
        out += _read_blobs(strip, _label_blobs(strip, False), (a, b))
    return out


_NUM = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?%?$")


def parse_number(s: str):
    s = s.replace("−", "-").replace("–", "-").replace(",", "").replace(" ", "")
    s = s.replace("O", "0").replace("o", "0") if re.search(r"\d", s) else s
    # a tick mark next to the label often gets read as a trailing dash: '100-'
    s = re.sub(r"[-_.|]+$", "", s) if re.search(r"\d[-_.|]+$", s) else s
    if not _NUM.match(s):
        return None
    return float(s.rstrip("%"))


def split_numeric(tb: TextBox, horizontal: bool):
    """A text box may contain several tick labels glued together ('0 20 40')."""
    parts = tb.text.replace("−", "-").split()
    vals = [parse_number(p) for p in parts]
    if not parts or any(v is None for v in vals):
        return []
    if len(parts) == 1:
        return [(vals[0], tb.cx if horizontal else tb.cy)]
    x0, x1 = tb.box[0], tb.box[2]
    total = sum(len(p) for p in parts) + (len(parts) - 1)
    out, pos = [], 0
    for p, v in zip(parts, vals):
        mid = pos + len(p) / 2
        out.append((v, x0 + (x1 - x0) * mid / total))
        pos += len(p) + 1
    return out if horizontal else []


@dataclass
class AxisFit:
    a: float
    b: float
    log: bool
    n_ticks: int
    n_inliers: int
    residual_px: float
    ticks: list = field(default_factory=list)  # (value, pixel) used

    def to_value(self, px):
        v = self.a * np.asarray(px, float) + self.b
        return np.power(10.0, v) if self.log else v

    def to_pixel(self, val):
        v = np.log10(np.asarray(val, float)) if self.log else np.asarray(val, float)
        return (v - self.b) / self.a


def _fit(ticks, log: bool, direction: int = 0):
    vals = np.array([t[0] for t in ticks], float)
    px = np.array([t[1] for t in ticks], float)
    if log:
        if np.any(vals <= 0):
            return None
        vals = np.log10(vals)
    best = None
    for i, j in itertools.combinations(range(len(px)), 2):
        if abs(px[i] - px[j]) < 5 or vals[i] == vals[j]:
            continue
        a = (vals[i] - vals[j]) / (px[i] - px[j])
        b = vals[i] - a * px[i]
        if not np.isfinite(a) or a == 0 or not np.isfinite(b):
            continue
        if direction and np.sign(a) != direction:
            continue
        with np.errstate(all="ignore"):
            err = np.abs((vals - b) / a - px)
        inl = np.isfinite(err) & (err < 4.0)
        if inl.sum() < 2:
            continue
        score = (inl.sum(), -err[inl].mean())
        if best is None or score > best[0]:
            best = (score, inl)
    if best is None or best[0][0] < 2:
        return None
    inl = best[1]
    A = np.vstack([px[inl], np.ones(inl.sum())]).T
    a, b = np.linalg.lstsq(A, vals[inl], rcond=None)[0]
    resid = float(np.abs((vals[inl] - b) / a - px[inl]).mean())
    used = [ticks[k] for k in np.where(inl)[0]]
    return AxisFit(float(a), float(b), log, len(ticks), int(inl.sum()), resid, used)


def fit_axis(ticks, direction: int = 0):
    """direction=+1: values must increase with pixel (x axis); -1: decrease (y axis).
    This stops e.g. a decreasing row of at-risk counts being taken for x tick labels."""
    if len(ticks) < 2:
        return None
    lin = _fit(ticks, log=False, direction=direction)
    lg = _fit(ticks, log=True, direction=direction) if len(ticks) >= 3 else None
    if lg and (lin is None or lg.n_inliers > lin.n_inliers):
        return lg
    return lin


def _ticks_from(texts, box, img_shape):
    x0, y0, x1, y1 = box
    h = img_shape[0]
    band = 0.12 * h
    xt, yt, used = [], [], set()
    for k, t in enumerate(texts):
        if y1 - 2 < t.cy < y1 + band and x0 - 25 < t.cx < x1 + 25:
            got = split_numeric(t, horizontal=True)
            if got:
                xt += got
                used.add(k)
        elif t.box[2] < x0 + 6 and y0 - 15 < t.cy < y1 + 15:
            v = parse_number(t.text)
            if v is not None:
                yt.append((v, t.cy))
                used.add(k)
    return xt, yt, used


def _dedupe(ticks, tol=4.0):
    out = []
    for v, p in sorted(ticks, key=lambda t: t[1]):
        if out and abs(out[-1][1] - p) < tol:
            continue
        out.append((v, p))
    return out


def calibrate(texts: list[TextBox], box, img_shape, strip_texts: list[TextBox] | None = None):
    """Return (x_fit, y_fit, other_texts) where other_texts are non-tick boxes.

    `texts` is full-image OCR (also used for legend text); `strip_texts` is the
    high-zoom OCR of the axis strips, which catches small tick labels.
    """
    xt, yt, used = _ticks_from(texts, box, img_shape)
    if strip_texts:
        sx, sy, _ = _ticks_from(strip_texts, box, img_shape)
        xt, yt = _dedupe(xt + sx), _dedupe(yt + sy)
    others = [t for k, t in enumerate(texts) if k not in used]
    return fit_axis(xt, +1), fit_axis(yt, -1), others
