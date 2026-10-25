"""Survival statistics on reconstructed patient data, plus meta-analysis.

  hazard_ratio(ipd)      Cox model HR (treatment vs control) with 95% CI
  median_survival(...)   KM median per arm
  pool(trials)           DerSimonian-Laird random-effects pooling of log HRs
  forest_plot(...)       the standard meta-analysis figure (PNG bytes)
"""
from __future__ import annotations

import io

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
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from scipy import stats as st  # noqa: E402


def to_frame(arms: dict) -> pd.DataFrame:
    """arms = {"treatment": (times, events), "control": (times, events)}"""
    rows = []
    for name, treat in (("treatment", 1), ("control", 0)):
        t, e = arms[name]
        rows.append(pd.DataFrame({"time": t, "event": e, "treat": treat}))
    df = pd.concat(rows, ignore_index=True)
    df["time"] = df["time"].clip(lower=1e-6)
    return df


def hazard_ratio(df: pd.DataFrame) -> dict:
    cph = CoxPHFitter().fit(df, "time", "event")
    s = cph.summary.loc["treat"]
    return {"hr": float(np.exp(s["coef"])), "lo": float(np.exp(s["coef lower 95%"])),
            "hi": float(np.exp(s["coef upper 95%"])), "log_hr": float(s["coef"]),
            "se": float(s["se(coef)"]), "p": float(s["p"])}


def median_survival(t, e):
    m = KaplanMeierFitter().fit(t, e).median_survival_time_
    return None if not np.isfinite(m) else float(m)


def km_curve(t, e, grid):
    kmf = KaplanMeierFitter().fit(t, e)
    return kmf.survival_function_at_times(grid).values


def pool(trials: list[dict]) -> dict:
    """Random-effects meta-analysis (DerSimonian-Laird) of log hazard ratios.

    trials: [{"name", "log_hr", "se"}, ...]
    """
    y = np.array([t["log_hr"] for t in trials], float)
    se = np.array([t["se"] for t in trials], float)
    w = 1 / se ** 2
    fixed = np.sum(w * y) / np.sum(w)
    q = float(np.sum(w * (y - fixed) ** 2))
    k = len(y)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    mu = float(np.sum(wr * y) / np.sum(wr))
    se_mu = float(np.sqrt(1 / np.sum(wr)))
    i2 = float(max(0.0, (q - (k - 1)) / q) * 100) if q > 0 and k > 1 else 0.0
    z = mu / se_mu
    return {
        "hr": float(np.exp(mu)), "lo": float(np.exp(mu - 1.96 * se_mu)), "hi": float(np.exp(mu + 1.96 * se_mu)),
        "p": float(2 * st.norm.sf(abs(z))), "tau2": tau2, "i2": i2, "q": q, "k": k,
        "weights": (wr / wr.sum() * 100).round(1).tolist(),
    }


@kraft
def forest_plot(trials: list[dict], pooled: dict, title="Overall survival: treatment vs control") -> bytes:
    k = len(trials)
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.5 * (k + 2)), dpi=110)
    ys = np.arange(k, 0, -1) + 1
    for yv, t, wt in zip(ys, trials, pooled["weights"]):
        ax.plot([t["lo"], t["hi"]], [yv, yv], color="#333", lw=1.4)
        ax.plot(t["hr"], yv, "s", color="#1f4e9c", ms=4 + wt / 6)
        ax.text(1.02, yv, f"{t['hr']:.2f} ({t['lo']:.2f}-{t['hi']:.2f})", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9)
    # pooled diamond
    yp = 1
    ax.fill([pooled["lo"], pooled["hr"], pooled["hi"], pooled["hr"]], [yp, yp + 0.25, yp, yp - 0.25],
            color="#d1495b")
    ax.text(1.02, yp, f"{pooled['hr']:.2f} ({pooled['lo']:.2f}-{pooled['hi']:.2f})",
            transform=ax.get_yaxis_transform(), va="center", fontsize=9, fontweight="bold")
    ax.axvline(1, color="#888", lw=1, ls="--")
    ax.set_xscale("log")
    lo = min([t["lo"] for t in trials] + [pooled["lo"]]) * 0.85
    hi = max([t["hi"] for t in trials] + [pooled["hi"]]) * 1.15
    ax.set_xlim(lo, max(hi, 1.2))
    ax.set_xticks([v for v in (0.25, 0.5, 0.75, 1, 1.5, 2) if lo <= v <= max(hi, 1.2)])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(list(ys) + [yp])
    ax.set_yticklabels([t["name"] for t in trials] + [f"Pooled (random effects), I²={pooled['i2']:.0f}%"])
    ax.set_ylim(0.3, k + 1.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Hazard ratio (log scale)   ← favors treatment | favors control →")
    ax.set_title(title, fontsize=11, loc="left")
    fig.subplots_adjust(left=0.32, right=0.74)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def indirect(a: dict, b: dict) -> dict:
    """Bucher indirect comparison of A vs B through a shared comparator C.

    a = A vs C, b = B vs C, each {"log_hr", "se"}.  log HR(A vs B) = log HR(A vs C) - log HR(B vs C);
    the variances add, so the indirect estimate is always less certain than either trial.
    """
    d = a["log_hr"] - b["log_hr"]
    se = float(np.sqrt(a["se"] ** 2 + b["se"] ** 2))
    z = d / se
    return {"hr": float(np.exp(d)), "lo": float(np.exp(d - 1.96 * se)), "hi": float(np.exp(d + 1.96 * se)),
            "log_hr": float(d), "se": se, "p": float(2 * st.norm.sf(abs(z)))}


@kraft
def indirect_plot(a: dict, b: dict, res: dict, name_a: str, name_b: str, common: str,
                  event: str = "death") -> bytes:
    """Three rows: A vs common, B vs common (from the trials), and A vs B (indirect)."""
    rows = [(f"{name_a} vs {common}", a, "#1f4e9c", "s"), (f"{name_b} vs {common}", b, "#2a9d8f", "s")]
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=110)
    for yv, (label, t, col, mk) in zip((3, 2), rows):
        ax.plot([t["lo"], t["hi"]], [yv, yv], color="#333", lw=1.4)
        ax.plot(t["hr"], yv, mk, color=col, ms=9)
        ax.text(1.02, yv, f"{t['hr']:.2f} ({t['lo']:.2f}-{t['hi']:.2f})", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9)
    yp = 1
    ax.fill([res["lo"], res["hr"], res["hi"], res["hr"]], [yp, yp + 0.25, yp, yp - 0.25], color="#d1495b")
    ax.text(1.02, yp, f"{res['hr']:.2f} ({res['lo']:.2f}-{res['hi']:.2f})", transform=ax.get_yaxis_transform(),
            va="center", fontsize=9, fontweight="bold")
    ax.axvline(1, color="#888", lw=1, ls="--")
    ax.axhline(1.55, color="#ddd", lw=1)
    ax.set_xscale("log")
    lo = min(a["lo"], b["lo"], res["lo"]) * 0.85
    hi = max(a["hi"], b["hi"], res["hi"], 1.2) * 1.15
    ax.set_xlim(lo, hi)
    ax.set_xticks([v for v in (0.25, 0.5, 0.75, 1, 1.5, 2, 3) if lo <= v <= hi])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([3, 2, 1])
    ax.set_yticklabels([rows[0][0], rows[1][0], f"{name_a} vs {name_b} (through {common})"])
    ax.set_ylim(0.4, 3.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"Hazard ratio for {event} (log scale).  Below 1: the first-named drug in that row is better")
    ax.set_title(f"Comparing {name_a} and {name_b} through their shared {common} arm", fontsize=11, loc="left")
    fig.subplots_adjust(left=0.34, right=0.78)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
