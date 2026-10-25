"""Read a Kaplan-Meier figure: survival curves + number-at-risk table.

Pipeline
  rectify -> plot box -> axis calibration (reused from the chart reader)
  -> colored curves traced column by column, forced non-increasing
  -> number-at-risk rows read from the band under the plot
  -> each row matched to its curve by color (or legend name, or order)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.axes import find_plot_box
from pipeline.rectify import rectify
from pipeline.series import extract_series, legend_mask
from pipeline.ticks import _read_blobs, calibrate, ocr_axis_strips, parse_number, run_ocr

CONTROL_WORDS = ("placebo", "control", "chemo", "standard", "docetaxel", "best supportive", "observation",
                 "comparator", "soc", "sorafenib", "usual")


@dataclass
class Arm:
    name: str
    color: tuple
    t: np.ndarray  # digitized times
    s: np.ndarray  # digitized survival 0..1
    px: np.ndarray
    py: np.ndarray
    mask: np.ndarray
    thickness: float
    t_risk: list
    n_risk: list


def _lab(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)


def _row_bands(region: np.ndarray):
    hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    # sensitive threshold: thin strokes like '1' in light colored text are faint
    ink = (gray < 205) | ((hsv[:, :, 1] > 60) & (hsv[:, :, 2] < 245))
    # remove long straight lines (figure frames, rules) so they don't glue rows together
    u8 = ink.astype(np.uint8)
    vlines = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((45, 1), np.uint8))
    hlines = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((1, 120), np.uint8))
    ink = ink & (vlines == 0) & (hlines == 0)
    rows = ink.sum(1) > 1
    bands, start = [], None
    for y, on in enumerate(rows):
        if on and start is None:
            start = y
        if not on and start is not None:
            if y - start >= 5:
                bands.append((start, y))
            start = None
    if start is not None and len(rows) - start >= 5:
        bands.append((start, len(rows)))
    # tightly packed tables: one blank-free band can hold several text rows.
    # split by clustering glyphs on their vertical centers.
    out = []
    for b0, b1 in bands:
        n, _, st, _ = cv2.connectedComponentsWithStats(ink[b0:b1].astype(np.uint8))
        comps = [st[i] for i in range(1, n) if st[i][4] >= 3 and st[i][3] >= 4]
        if not comps:
            continue
        mh = float(np.median([c[3] for c in comps]))
        if (b1 - b0) <= 1.6 * mh:
            out.append((b0, b1))
            continue
        cys = sorted((c[1] + c[3] / 2, c[1], c[1] + c[3]) for c in comps)
        grp = [cys[0]]
        for c in cys[1:]:
            if c[0] - grp[-1][0] > 0.55 * mh:
                out.append((b0 + min(g[1] for g in grp), b0 + max(g[2] for g in grp)))
                grp = [c]
            else:
                grp.append(c)
        out.append((b0 + min(g[1] for g in grp), b0 + max(g[2] for g in grp)))
    return [b for b in out if b[1] - b[0] >= 5], ink


def tick_grid(xfit, box):
    """All tick times/pixels on the x axis (OCR may have missed some labels)."""
    vals = sorted(v for v, _ in xfit.ticks)
    # OCR can miss labels (12, 15, 18, 24): the true step is the smallest gap, not the median
    diffs = [d for d in np.diff(vals) if d > 1e-9]
    step = float(min(diffs)) if diffs else None
    if not step or step <= 0:
        return np.array(vals), xfit.to_pixel(vals)
    t_lo, t_hi = sorted(xfit.to_value([box[0], box[2]]))
    k0, k1 = int(np.ceil(t_lo / step - 0.15)), int(np.floor(t_hi / step + 0.15))
    times = np.arange(k0, k1 + 1) * step
    return times, xfit.to_pixel(times)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def _digits(s: str):
    """Pull an integer out of an OCR string ('b126' -> 126, 'I-TKI165' -> 165, 'O' -> 0)."""
    s = _norm(s)
    s = re.sub(r"\(.*?\)|\(.*$", "", s)  # '84 (4)' -> '84': drop censor counts in parentheses
    s = s.replace("O", "0").replace("o", "0").replace("D", "0")
    m = re.findall(r"\d+", s)
    return int(m[-1]) if m else None


def _ocr_crop(gray: np.ndarray, b, height: int):
    from pipeline.ticks import get_ocr_small as get_ocr

    a, y0, c, y1 = b
    cr = gray[max(0, y0 - 3):y1 + 3, max(0, a - 3):c + 3]
    if cr.size == 0:
        return None
    k = height / cr.shape[0]
    cr = cv2.resize(cr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    cr = cv2.copyMakeBorder(cr, 30, 30, 60, 60, cv2.BORDER_CONSTANT, value=255)
    r, _ = get_ocr()(cv2.cvtColor(cr, cv2.COLOR_GRAY2BGR), text_score=0.3)
    return _digits("".join(t[1] for t in (r or [])))


def _ocr_tiled(gray: np.ndarray, b, height: int = 48, copies: int = 3):
    """Text detectors often ignore a single isolated character ('7').
    Repeat the crop side by side ('7  7  7') so it looks like a text line,
    then keep the value the copies agree on."""
    from collections import Counter

    from pipeline.ticks import get_ocr_small as get_ocr

    a, y0, c, y1 = b
    cr = gray[max(0, y0 - 3):y1 + 3, max(0, a - 3):c + 3]
    if cr.size == 0:
        return None
    k = height / cr.shape[0]
    cr = cv2.resize(cr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    gap = np.full((cr.shape[0], int(0.7 * height)), 255, np.uint8)
    parts = []
    for i in range(copies):
        parts += [cr] + ([gap] if i < copies - 1 else [])
    line = cv2.copyMakeBorder(np.hstack(parts), 30, 30, 60, 60, cv2.BORDER_CONSTANT, value=255)
    r, _ = get_ocr()(cv2.cvtColor(line, cv2.COLOR_GRAY2BGR), text_score=0.3)
    text = " ".join(t[1] for t in (r or []))
    text = _norm(text).replace("O", "0").replace("o", "0")
    toks = re.findall(r"\d+", text)
    if len(toks) >= 2:
        val, cnt = Counter(toks).most_common(1)[0]
        return int(val) if cnt >= 2 else None
    if len(toks) == 1:  # copies merged into one token, e.g. '777' or '151515'
        t = toks[0]
        if len(t) % copies == 0 and t == t[: len(t) // copies] * copies:
            return int(t[: len(t) // copies])
    return None


def _best_sequence(cands: list[list[int]]):
    """Choose one candidate (or none) per column so the counts never increase,
    maximizing how many columns are filled; earlier candidates are preferred."""
    n = len(cands)
    # state: (filled, -penalty) best ending at column i with value v
    best = {}
    for i in range(n):
        for rank, v in enumerate(cands[i]):
            score, back = (1, -rank), None
            for (j, u), (sc, _) in best.items():
                if j < i and u >= v:
                    cand = (sc[0] + 1, sc[1] - rank)
                    if cand > score:
                        score, back = cand, (j, u)
            if (i, v) not in best or score > best[(i, v)][0]:
                best[(i, v)] = (score, back)
    if not best:
        return {}
    key = max(best, key=lambda k: best[k][0])
    out = {}
    while key is not None:
        out[key[0]] = key[1]
        key = best[key][1]
    return out


def _number_centers(ink_band: np.ndarray, bh: int):
    """x-centers of glyph groups (numbers) in one table row."""
    n, _, st, _ = cv2.connectedComponentsWithStats(ink_band.astype(np.uint8))
    comps = sorted([st[i][:4] for i in range(1, n) if st[i][4] >= 3], key=lambda c: c[0])
    groups = []
    for x, y, w, h in comps:
        if groups and x - groups[-1][1] <= 0.5 * bh:
            groups[-1][1] = max(groups[-1][1], x + w)
        else:
            groups.append([x, x + w])
    return [(a + b) / 2 for a, b in groups]


def _refine_grid(times, gpx, bands, ink):
    """Some journals print at-risk counts more often than axis labels (every 1.5 months
    with labels every 3). Find the finest subdivision of the tick grid the numbers sit on."""
    spacing = float(np.median(np.diff(gpx)))
    centers = []
    for b0, b1 in bands:
        cs = [c for c in _number_centers(ink[b0:b1], b1 - b0) if gpx[0] - 0.3 * spacing <= c <= gpx[-1] + 0.3 * spacing]
        if len(cs) >= 3:
            centers += cs
    if len(centers) < 4:
        return times, gpx
    centers = np.array(centers)
    for k in (1, 2, 3, 4):
        sub = spacing / k
        off = (centers - gpx[0]) / sub
        frac = np.mean(np.abs(off - np.round(off)) * sub < 0.22 * sub)
        if frac >= 0.85:
            break
    if k == 1:
        return times, gpx
    n = (len(gpx) - 1) * k + 1
    new_px = gpx[0] + np.arange(n) * spacing / k
    new_t = times[0] + np.arange(n) * (times[1] - times[0]) / k
    return new_t, new_px


def _consistency_assign(rows, curves, xfit, yfit, y_scale):
    """Pick the row-to-curve assignment that is physically possible:
    the share of patients still AT RISK can never exceed the share still ALIVE
    (censoring only removes people), so n(t)/n(0) <= S(t) must hold."""
    from itertools import permutations

    def s_at(c, t):
        tt = xfit.to_value(c.px)
        ss = np.minimum.accumulate(np.clip(yfit.to_value(c.py) / y_scale, 0, 1))
        o = np.argsort(tt)
        return float(np.interp(t, tt[o], ss[o]))

    best, best_score = None, None
    k = min(len(rows), len(curves))
    for perm in permutations(range(len(curves)), k):
        score = 0.0
        for ri, ci in zip(range(k), perm):
            r = rows[ri]
            n0 = r["n"][0] if r["t"] and r["t"][0] < 1e-6 else max(r["n"])
            for t, n in zip(r["t"], r["n"]):
                score += max(0.0, n / n0 - s_at(curves[ci], t) - 0.01)
        if best_score is None or score < best_score:
            best, best_score = perm, score
    return {ri: ci for ri, ci in zip(range(k), best)}, best_score


def _longest_nonincreasing(seq):
    """Indices of the longest non-increasing subsequence (drops OCR misreads)."""
    n = len(seq)
    best, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if seq[j] >= seq[i] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = int(np.argmax(best)) if n else -1
    out = []
    while i != -1:
        out.append(i)
        i = prev[i]
    return out[::-1]


def _read_pair_row(band_gray: np.ndarray, ink_band: np.ndarray, dy: int):
    """Rows like 'Sintilimab 145 (0) 84 (4) 54 (9) ...' (at risk, with censored in brackets).
    Each 'N (c)' group is cut out and read on its own. Returns (label, counts) or None."""
    from pipeline.ticks import get_ocr_small as get_ocr

    bh = ink_band.shape[0]
    n, _, st, _ = cv2.connectedComponentsWithStats(ink_band.astype(np.uint8))
    comps = sorted([tuple(st[i][:4]) for i in range(1, n) if st[i][4] >= 3], key=lambda c: c[0])
    if len(comps) < 6:
        return None
    groups = []
    for x, y, w, h in comps:
        if groups and x - groups[-1][2] <= 0.75 * bh:
            g = groups[-1]
            groups[-1] = [g[0], min(g[1], y), max(g[2], x + w), max(g[3], y + h)]
        else:
            groups.append([x, y, x + w, y + h])

    def read(gb):
        x0, y0, x1, y1 = gb
        cr = band_gray[max(0, y0 + dy - 3):y1 + dy + 3, max(0, x0 - 4):x1 + 4]
        if cr.size == 0:
            return ""
        k = 48 / cr.shape[0]
        cr = cv2.resize(cr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        cr = cv2.copyMakeBorder(cr, 30, 30, 60, 60, cv2.BORDER_CONSTANT, value=255)
        r, _ = get_ocr()(cv2.cvtColor(cr, cv2.COLOR_GRAY2BGR), text_score=0.3)
        return _norm(" ".join(t[1] for t in (r or [])))

    # cheap test first: read a few groups in the middle of the row; pair tables show '('
    probe = groups[len(groups) // 2: len(groups) // 2 + 2]
    if not any("(" in read(gb) for gb in probe):
        return None
    texts = [read(gb) for gb in groups]
    is_pair = [bool(re.search(r"\(", t)) for t in texts]
    if sum(is_pair) < 3:
        return None
    label_parts, counts = [], []
    for gb, t in zip(groups, texts):
        if not counts and re.search(r"[A-Za-z]{3,}", t) and "(" not in t:
            label_parts.append(re.sub(r"[^A-Za-z \-]", "", t))
            continue
        m = re.match(r"\s*([0-9OoDlI|]+)\s*\(", t)
        if m:
            v = _digits(m.group(1).replace("l", "1").replace("I", "1").replace("|", "1"))
        else:
            # the bare count may have lost its bracket: try the tiled reader on the left part
            v = None
        if v is None:
            x0, y0, x1, y1 = gb
            left = band_gray[:, max(0, x0 - 2): x0 + int(0.45 * (x1 - x0))]
            v = _ocr_tiled(left, (0, max(0, y0 + dy), left.shape[1], y1 + dy))
        counts.append(v)
    if not counts or counts[0] is None:
        return None
    return " ".join(label_parts).strip(), counts


def read_at_risk(img: np.ndarray, box, xfit, start_y: int):
    """Return rows {"label", "color", "t", "n"} from the number-at-risk table.

    At-risk counts sit under the x-axis ticks, so glyphs are clustered around
    each tick column; glyphs left of the first column form the arm name. Each
    count is read several ways and the combination that forms a valid
    (never increasing) sequence wins.
    """
    h = img.shape[0]
    region = img[start_y:h]
    bands, ink = _row_bands(region)
    times, gpx = tick_grid(xfit, box)
    if len(gpx) < 2:
        return []
    times, gpx = _refine_grid(times, gpx, bands, ink)
    spacing = float(np.median(np.diff(gpx)))
    zone = 0.45 * spacing
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(float)
    rows = []
    for b0, b1 in bands:
        bh = b1 - b0
        pa = max(0, b0 - 5)
        g = gray[pa:b1 + 5]
        g = (np.clip((g - g.min()) / (255 - g.min() + 1e-9), 0, 1) * 255).astype(np.uint8)
        dy = b0 - pa
        pr = _read_pair_row(g, ink[b0:b1], dy)
        if pr is not None:
            name, counts = pr
            # censor-pair tables: one pair per time point, starting at 0
            grid_t = [float(t) for t in times if t >= -1e-9]
            if counts and counts[0] and counts[0] > 0 and len(counts) <= len(grid_t):
                idx = [i for i, c in enumerate(counts) if c is not None]
                sub = [counts[i] for i in idx]
                keep = [idx[j] for j in _longest_nonincreasing(sub)]
                kk = [grid_t[i] for i in keep]
                vals = [counts[i] for i in keep]
                if len(vals) >= 3 and keep[0] == 0:
                    rows.append({"label": name or f"Arm {len(rows) + 1}", "color": None, "t": kk, "n": vals})
            continue
        n, _, st, _ = cv2.connectedComponentsWithStats(ink[b0:b1].astype(np.uint8))
        comps = sorted([tuple(st[i][:4]) for i in range(1, n) if st[i][4] >= 3], key=lambda c: c[0])
        cols, used = {}, set()
        for k, gx in enumerate(gpx):
            near = [i for i, (x, y, w, hh) in enumerate(comps) if abs(x + w / 2 - gx) <= zone and i not in used]
            if not near:
                continue
            # grow a tight cluster outward from the glyph closest to the tick
            seed = min(near, key=lambda i: abs(comps[i][0] + comps[i][2] / 2 - gx))
            members = {seed}
            grown = True
            while grown:
                grown = False
                lo = min(comps[i][0] for i in members)
                hi = max(comps[i][0] + comps[i][2] for i in members)
                for i in near:
                    x, _, w, _ = comps[i]
                    if i not in members and (lo - (x + w) <= 0.5 * bh and x - hi <= 0.5 * bh):
                        members.add(i)
                        grown = True
            used |= members
            c = np.array([comps[i] for i in members])
            # pad sideways so faint thin glyphs next to the cluster are included
            padx = int(0.35 * bh)
            lo_lim = int(gx - zone) if k == 0 else int(gx - zone)
            cols[k] = (max(lo_lim, int(c[:, 0].min()) - padx), int(c[:, 1].min()) + dy,
                       min(int(gx + zone), int((c[:, 0] + c[:, 2]).max()) + padx), int((c[:, 1] + c[:, 3]).max()) + dy)
        if len(cols) < 3:
            continue
        ks = sorted(cols)
        boxes = [cols[k] for k in ks]
        comp_read = {}
        for t in ([] if len(boxes) > 12 else _read_blobs(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), boxes, (0, 0))):
            for k, bx in zip(ks, boxes):
                if abs(t.box[0] - bx[0]) < 1:
                    comp_read[k] = _digits(t.text)
        cands = []
        for k in ks:
            cs = []
            # individual crops are most reliable. Read once plainly and once tiled; only when
            # those disagree (or fail) spend more reads at other zoom levels.
            first = [_ocr_crop(g, cols[k], 48), _ocr_tiled(g, cols[k])]
            reads = first if (first[0] is not None and first[0] == first[1]) else \
                first + [_ocr_crop(g, cols[k], 64), _ocr_crop(g, cols[k], 32), comp_read.get(k)]
            for v in reads:
                if v is not None and v not in cs:
                    cs.append(v)
            cands.append(cs)
        pick = _best_sequence(cands)
        if len(pick) < 3:
            continue
        kk = [ks[i] for i in sorted(pick)]
        vals = [pick[i] for i in sorted(pick)]
        if vals[0] <= 0 or kk[0] != ks[0]:
            if vals[0] <= 0:
                continue  # x tick-label row (starts at 0, increases)
        # arm name: glyphs left of the first count column
        lab = [c for i, c in enumerate(comps) if i not in used and c[0] + c[2] < gpx[0] - 0.2 * spacing]
        name = ""
        if lab:
            lb = np.array(lab)
            lbox = [(lb[:, 0].min(), lb[:, 1].min() + dy, (lb[:, 0] + lb[:, 2]).max(), (lb[:, 1] + lb[:, 3]).max() + dy)]
            lt = _read_blobs(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), lbox, (0, 0))
            name = re.sub(r"\s*\d+$", "", _norm(" ".join(t.text for t in lt)))
        band = region[b0:b1]
        pix = band[ink[b0:b1]]
        hsv = cv2.cvtColor(pix[None], cv2.COLOR_RGB2HSV)[0]
        sat = pix[hsv[:, 1] > 80]
        color = tuple(int(c) for c in np.median(sat, 0)) if len(sat) > 0.3 * len(pix) else None
        rows.append({"label": name or f"Arm {len(rows) + 1}", "color": color,
                     "t": [float(times[k]) for k in kk], "n": vals})
    return rows


def _key(s: str) -> str:
    return re.sub(r"[^a-z]", "", _norm(s).lower())


def _match_by_curve_labels(rows, curves, texts, box):
    """Journals often print the arm name right next to its curve. Find each row's
    name inside the plot and assign the curve that runs closest to it."""
    x0, y0, x1, y1 = box
    picks = []
    for ri, r in enumerate(rows):
        k = _key(r["label"])
        if len(k) < 4:
            continue
        best = None
        for t in texts:
            if not (x0 - 20 < t.cx < x1 + 40 and y0 - 40 < t.cy < y1 + 5):
                continue
            tk = _key(t.text)
            if not tk or not (tk.startswith(k[:6]) or k.startswith(tk[:6]) or k[:6] in tk):
                continue
            d = []
            for c in curves:
                dx = np.clip(c.px, t.box[0], t.box[2]) - c.px
                dy = np.clip(c.py, t.box[1], t.box[3]) - c.py
                d.append(float(np.min(np.hypot(dx, dy))))
            o = np.argsort(d)
            ratio = d[o[1]] / max(d[o[0]], 1.0) if len(d) > 1 else 1.0
            if best is None or ratio > best[0]:
                best = (ratio, int(o[0]))
        if best and best[0] >= 1.5:
            picks.append((best[0], ri, best[1]))
    assign, used = {}, set()
    for ratio, ri, ci in sorted(picks, reverse=True):
        if ri not in assign and ci not in used:
            assign[ri] = ci
            used.add(ci)
    return assign


def _match_rows_to_curves(rows, curves, legend_items, texts=None, box=None):
    """Assign each at-risk row to a curve index."""
    if not rows or not curves:
        return {}
    assign = {}
    colored = all(r["color"] is not None for r in rows)
    if texts and box and not colored:  # black tables: use arm names printed by the curves
        assign = _match_by_curve_labels(rows, curves, texts, box)
        if len(assign) == min(len(rows), len(curves)):
            return assign
        if len(assign) == min(len(rows), len(curves)) - 1:
            free_r = [ri for ri in range(len(rows)) if ri not in assign]
            free_c = [ci for ci in range(len(curves)) if ci not in assign.values()]
            if free_r and free_c:
                assign[free_r[0]] = free_c[0]
            return assign
        assign = {}
    if all(r["color"] is not None for r in rows):
        used = set()
        for ri, r in enumerate(rows):
            d = [np.linalg.norm(_lab(r["color"]) - _lab(c.color_rgb)) if ci not in used else 1e9
                 for ci, c in enumerate(curves)]
            ci = int(np.argmin(d))
            assign[ri] = ci
            used.add(ci)
        return assign
    if legend_items:
        used = set()
        for ri, r in enumerate(rows):
            best, bs = None, -1
            for name, col in legend_items:
                score = len(set(name.lower().split()) & set(r["label"].lower().split()))
                if score > bs:
                    best, bs = col, score
            if best is not None and bs > 0:
                d = [np.linalg.norm(_lab(best) - _lab(c.color_rgb)) if ci not in used else 1e9
                     for ci, c in enumerate(curves)]
                ci = int(np.argmin(d))
                assign[ri] = ci
                used.add(ci)
    for ri in range(len(rows)):  # fallback: order
        if ri not in assign and ri < len(curves):
            free = [ci for ci in range(len(curves)) if ci not in assign.values()]
            if free:
                assign[ri] = free[0]
    return assign


def _legend_items(img, others, box):
    """(text, swatch color) for legend entries inside the plot area."""
    x0, y0, x1, y1 = box
    items = []
    for t in others:
        if not (x0 < t.cx < x1 and y0 < t.cy < y1 and any(c.isalpha() for c in t.text)):
            continue
        hgt = max(8, t.height)
        a, b = int(max(0, t.box[0] - 3.5 * hgt)), int(t.box[0])
        sw = img[int(t.box[1]):int(t.box[3]), a:b]
        if sw.size == 0:
            continue
        hsv = cv2.cvtColor(sw, cv2.COLOR_RGB2HSV)
        px = sw[(hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 50)]
        if len(px):
            items.append((t.text, tuple(int(v) for v in np.median(px, 0))))
    return items


def read_km_figure(img_rgb: np.ndarray, do_rectify: bool = True):
    photo = img_rgb
    flat, H, found = rectify(photo) if do_rectify else (photo, np.eye(3), False)
    box = find_plot_box(flat)
    if box is None:
        raise ValueError("Could not find the plot axes.")
    texts = run_ocr(flat)
    xfit, yfit, others = calibrate(texts, box, flat.shape, ocr_axis_strips(flat, box))
    if xfit is None or yfit is None:
        raise ValueError("Could not read the axis numbers (tick labels).")
    # Survival plots start at time 0 on the y-axis. If the x labels were read shifted by a
    # whole tick (a consistent OCR misalignment passes every linear-fit check), undo it.
    x_shift_note = None
    vals = sorted(v for v, _ in xfit.ticks)
    diffs = [d for d in np.diff(vals) if d > 1e-9]
    if diffs and not xfit.log:
        step = min(diffs)
        t0 = float(xfit.to_value(box[0]))
        k = t0 / step
        if abs(t0) > 0.3 * step and abs(k - round(k)) < 0.2:
            xfit.b -= t0  # value = a*px + b  ->  value(x0) = 0
            xfit.ticks = [(v - t0, px) for v, px in xfit.ticks]
            x_shift_note = (f"The x-axis labels looked shifted by {abs(round(k)) * step:g}; corrected so "
                            f"time 0 sits at the y-axis. Check the time grid below.")
    ymax = max(v for v, _ in yfit.ticks)
    y_scale = 100.0 if ymax > 1.5 else 1.0

    # survival curves are always step lines; shades of one hue belong to one curve
    curves = extract_series(flat, box, legend_mask(others, box, flat.shape), step=1,
                            force_line=True, merge_hue_deg=25)
    curves = [c for c in curves if c.kind == "line"]
    if not curves:
        raise ValueError("No colored survival curves found.")

    # at-risk table lives below the x tick labels and axis title
    x_title_bottom = max([t.box[3] for t in others if t.cy > box[3]] + [box[3] + 0.06 * flat.shape[0]])
    start_y = int(min(flat.shape[0] - 10, box[3] + 0.04 * flat.shape[0]))
    rows = read_at_risk(flat, box, xfit, start_y)
    if len(rows) > len(curves):
        # more candidate rows than curves (e.g. a stray axis-label row): keep the best ones
        def quality(r):
            named = not r["label"].startswith("Arm ")
            starts0 = bool(r["t"]) and abs(r["t"][0]) < 1e-6
            return (2 if named else 0) + (2 if starts0 else 0) + 0.1 * len(r["n"])
        best = sorted(range(len(rows)), key=lambda i: -quality(rows[i]))[:len(curves)]
        rows = [rows[i] for i in sorted(best)]
    legend_items = _legend_items(flat, others, box)
    assign = _match_rows_to_curves(rows, curves, legend_items, texts, meta_box := box)
    colored = rows and all(r["color"] is not None for r in rows)
    if len(rows) >= 2 and len(curves) >= 2 and not colored:
        # black tables: verify (or override) the assignment with the at-risk vs survival logic
        cons, _ = _consistency_assign(rows, curves, xfit, yfit, 100.0 if max(v for v, _ in yfit.ticks) > 1.5 else 1.0)
        assign = cons

    arms = []
    for ri, ci in assign.items():
        c = curves[ci]
        t = xfit.to_value(c.px)
        s = yfit.to_value(c.py) / y_scale
        arms.append(Arm(rows[ri]["label"], c.color_rgb, t, s, c.px, c.py, c.mask, c.thickness,
                        rows[ri]["t"], rows[ri]["n"]))
    endpoint, endpoint_evidence = detect_endpoint(texts, box)
    printed_hr = detect_printed_hr(texts)
    meta = {"x_shift_note": x_shift_note, "endpoint": endpoint, "endpoint_evidence": endpoint_evidence, "printed_hr": printed_hr,
            "rectified": found, "box": box, "xfit": xfit, "yfit": yfit, "y_scale": y_scale,
            "n_curves": len(curves), "n_rows": len(rows), "flat": flat, "photo": photo, "H": H,
            "x_title_bottom": x_title_bottom}
    return arms, meta


def guess_treatment_index(arms) -> int:
    """Which arm is the experimental treatment? Control words, else first listed."""
    for i, a in enumerate(arms):
        if any(w in a.name.lower() for w in CONTROL_WORDS):
            return 1 - i if len(arms) == 2 else 0
    return 0


# ---------------------------------------------------------------- outcome + printed HR

ENDPOINTS = {
    "os": "Overall survival",
    "pfs": "Progression-free survival",
    "dfs": "Disease-free survival",
    "efs": "Event-free survival",
    "rfs": "Relapse-free survival",
}


def _classify_endpoint(text: str):
    t = _norm(text).lower().replace(" ", "")
    if "progression" in t or "pfs" in t:
        return "pfs"
    if "disease-free" in t or "diseasefree" in t or "dfs" in t:
        return "dfs"
    if "event-free" in t or "eventfree" in t or "efs" in t:
        return "efs"
    if "relapse" in t or "recurrence-free" in t or "rfs" in t:
        return "rfs"
    if "overallsurvival" in t or "hazardratiofordeath" in t:
        return "os"
    return None


def detect_endpoint(texts, box):
    """What the curve measures. The y-axis title is the strongest evidence; other text on the
    figure is the fallback. Returns (code or None, evidence string)."""
    x0, y0, x1, y1 = box
    axis_titles = [t for t in texts if t.box[2] < x0 + 5 and (t.box[3] - t.box[1]) > 1.5 * (t.box[2] - t.box[0])]
    for t in axis_titles:
        c = _classify_endpoint(t.text)
        if c:
            return c, f"y-axis title: '{_norm(t.text)}'"
    found = {}
    for t in texts:
        c = _classify_endpoint(t.text)
        if c:
            found.setdefault(c, _norm(t.text))
    if len(found) == 1:
        c, txt = next(iter(found.items()))
        return c, f"text on the figure: '{txt}'"
    return None, "could not tell from the figure" if not found else "figure mentions several outcomes"


def _dec(s: str):
    """OCR decimals: '0·73', '0.73', '0-98', '0,98' -> 0.73 / 0.98."""
    m = re.fullmatch(r"([0-9])[.·,\-]([0-9]{1,3})", s)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def detect_printed_hr(texts):
    """Read a hazard ratio printed on the figure, e.g. 'HR 0.73 (95% CI 0.54-0.98)'."""
    for t in texts:
        s = _norm(t.text).replace("–", "-").replace("—", "-")
        m = re.search(r"(?:HR|hazard\s*ratio[a-z\s]*)[^0-9]{0,6}([0-9][.·,][0-9]{1,3})", s, re.I)
        if not m:
            continue
        hr = _dec(m.group(1).replace(",", "."))
        rest = s[m.end():]
        nums = [_dec(x) for x in re.findall(r"[0-9][.·,\-][0-9]{1,3}", rest)]
        nums = [n for n in nums if n is not None]
        # first two numbers after the HR that bracket it are the CI (skip '95' from '95% CI')
        lo = hi = None
        for i in range(len(nums) - 1):
            if nums[i] <= hr <= nums[i + 1]:
                lo, hi = nums[i], nums[i + 1]
                break
        if hr and 0.05 < hr < 5:
            return {"hr": hr, "lo": lo, "hi": hi, "text": s}
    return None
