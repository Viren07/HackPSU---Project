"""End-to-end trial analysis: figure -> digitized arms -> rebuilt patients -> HR.

Two stages so the UI can let a person correct the at-risk table:
  read_trial(img)            image -> digitized curves + at-risk table (JSON-able)
  analyze_trial(read, ...)   digitized data -> reconstructed IPD, HR, checks, figures
"""
from __future__ import annotations

import base64
import io

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# kraft-paper chart theme to match the web page (applied only to our own charts)
KRAFT = {
    "figure.facecolor": "#f2e4d0", "axes.facecolor": "#f2e4d0", "savefig.facecolor": "#f2e4d0",
    "text.color": "#36261a", "axes.labelcolor": "#36261a", "axes.edgecolor": "#6b4c34",
    "xtick.color": "#36261a", "ytick.color": "#36261a", "legend.frameon": False,
}


def kraft(fn):
    def wrapped(*a, **k):
        with plt.rc_context(KRAFT):
            return fn(*a, **k)
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped
import numpy as np  # noqa: E402

from pipeline.verify import compare  # noqa: E402

from .extract import guess_treatment_index, read_km_figure  # noqa: E402
from .guyot import clean_curve, reconstruct  # noqa: E402
from .stats import hazard_ratio, km_curve, median_survival, to_frame  # noqa: E402


def _b64png(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def read_trial(img_rgb: np.ndarray, do_rectify: bool = True) -> dict:
    arms, meta = read_km_figure(img_rgb, do_rectify=do_rectify)
    flat = meta["flat"]
    overlay = flat.copy()
    x0, y0, x1, y1 = meta["box"]
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255), 2)
    for a in arms:
        for x, y in zip(a.px[::3], a.py[::3]):
            cv2.circle(overlay, (int(x), int(y)), 2, (255, 0, 255), -1)
    warnings = []
    if meta["n_curves"] != 2:
        warnings.append(f"Found {meta['n_curves']} curves; expected 2 arms.")
    if len(arms) < 2:
        warnings.append("Could not match two at-risk rows to two curves. Check the table below.")
    for a in arms:
        if not a.t_risk or a.t_risk[0] > 1e-6:
            warnings.append(f"'{a.name}': the count at time 0 is missing. Please fill it in.")
    if meta.get("x_shift_note"):
        warnings.append(meta["x_shift_note"])
    if meta.get("endpoint") is None:
        warnings.append("Could not tell what this curve measures. Please choose the outcome below.")
    xf = meta["xfit"]
    return {
        "ok": True,
        "endpoint": meta.get("endpoint"),
        "endpoint_evidence": meta.get("endpoint_evidence"),
        "printed_hr": meta.get("printed_hr"),
        "rectified": meta["rectified"],
        "treatment_index": guess_treatment_index(arms) if arms else 0,
        "arms": [{
            "name": a.name, "color": list(a.color),
            "t": np.round(a.t, 4).tolist(), "s": np.round(np.clip(a.s, 0, 1), 5).tolist(),
            "t_risk": a.t_risk, "n_risk": a.n_risk,
            "px": np.round(a.px, 1).tolist(), "py": np.round(a.py, 1).tolist(),
            "thickness": a.thickness,
        } for a in arms],
        "axis": {"x_ticks": [v for v, _ in xf.ticks], "y_percent": meta["y_scale"] > 1,
                 "x_max": float(xf.to_value(x1))},
        "warnings": warnings,
        "overlay_png": _b64png(overlay),
        # kept server-side-free: the client sends 'arms' back for analysis
        "_fit": {"xa": xf.a, "xb": xf.b, "ya": meta["yfit"].a, "yb": meta["yfit"].b,
                 "y_scale": meta["y_scale"], "box": list(meta["box"]), "shape": list(flat.shape[:2])},
        "_mask_png": None,
    }


def _pixel_check(arm, t_grid, s_rec, fit):
    """Render the rebuilt curve into pixels and compare with the curve in the image."""
    h, w = fit["shape"]
    x0, y0, x1, y1 = fit["box"]
    obs = np.zeros((h, w), np.uint8)
    for x, y in zip(arm["px"], arm["py"]):
        cv2.circle(obs, (int(round(x)), int(round(y))), max(1, int(arm["thickness"] / 2)), 255, -1)
    ren = np.zeros_like(obs)
    xs = (np.asarray(t_grid) - fit["xb"]) / fit["xa"]
    ys = (np.asarray(s_rec) * fit["y_scale"] - fit["yb"]) / fit["ya"]
    pts = np.stack([xs, ys], 1)
    pts = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1)]
    if len(pts) > 1:
        cv2.polylines(ren, [pts.astype(np.int32)], False, 255, max(1, int(arm["thickness"])))
    roi = (slice(y0, y1), slice(x0, x1))
    return compare(obs[roi], ren[roi], tol=3)


def analyze_trial(read: dict, treatment_index: int | None = None, published_hr: float | None = None) -> dict:
    arms = read["arms"]
    if len(arms) != 2:
        raise ValueError("Need exactly two arms to compare.")
    ti = read.get("treatment_index", 0) if treatment_index is None else treatment_index
    order = {"treatment": arms[ti], "control": arms[1 - ti]}
    ipd, checks, curves = {}, {}, {}
    t_max = max(max(a["t"]) for a in arms)
    grid = np.linspace(0, t_max, 300)
    for role, a in order.items():
        if not a["t_risk"] or len(a["t_risk"]) < 2:
            raise ValueError(f"'{a['name']}' needs at least two at-risk counts.")
        t_s, s = clean_curve(a["t"], a["s"])
        times, events = reconstruct(t_s, s, a["t_risk"], a["n_risk"])
        ipd[role] = (times, events)
        s_rec = km_curve(times, events, grid)
        s_dig = np.interp(grid, t_s, s)
        rec_risk = [int((times >= tr).sum()) for tr in a["t_risk"]]
        chk = {
            "curve_max_gap": float(np.max(np.abs(s_rec - s_dig))),
            "curve_mean_gap": float(np.mean(np.abs(s_rec - s_dig))),
            "at_risk_published": a["n_risk"], "at_risk_rebuilt": rec_risk,
            "n_patients": int(len(times)), "n_events": int(events.sum()),
            "median": median_survival(times, events),
        }
        if "_fit" in read and a.get("px"):
            p, r, f = _pixel_check(a, grid, s_rec, read["_fit"])
            chk["pixel_match"] = round(f, 3)
        checks[role] = chk
        curves[role] = {"name": a["name"], "color": a["color"], "grid": grid.round(3).tolist(),
                        "s_rebuilt": s_rec.round(4).tolist(), "s_digitized": s_dig.round(4).tolist()}

    hr = hazard_ratio(to_frame(ipd))
    res = {
        "ok": True,
        "treatment": order["treatment"]["name"], "control": order["control"]["name"],
        "hr": hr, "checks": checks, "curves": curves,
    }
    if published_hr:
        res["published_hr"] = published_hr
        res["hr_error_pct"] = round(abs(hr["hr"] - published_hr) / published_hr * 100, 1)
        # agreement check: a big gap usually means a misread at-risk number or curve
        res["agrees_with_paper"] = res["hr_error_pct"] <= 10
    res["rebuilt_png"] = _rebuilt_figure(curves, ipd, order, hr)
    return res


@kraft
def _rebuilt_figure(curves, ipd, order, hr) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=110)
    for role in ("treatment", "control"):
        c = curves[role]
        col = np.array(c["color"]) / 255
        ax.plot(c["grid"], c["s_digitized"], color=col, lw=6, alpha=0.25)
        ax.step(c["grid"], c["s_rebuilt"], where="post", color=col, lw=1.6,
                label=f"{c['name']} (rebuilt, n={len(ipd[role][0])})")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Rebuilt from reconstructed patients   HR {hr['hr']:.2f} "
                 f"({hr['lo']:.2f}-{hr['hi']:.2f})", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.02, "thick = read from figure, thin = rebuilt patients", transform=ax.transAxes,
            ha="right", fontsize=8, color="#666")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
