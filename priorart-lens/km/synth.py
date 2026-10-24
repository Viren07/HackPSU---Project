"""Synthetic randomized trials drawn as journal-style Kaplan-Meier figures.

Each trial has real (simulated) patient-level data, so we know the TRUE hazard
ratio and can check how well we reconstruct it from the picture alone.

Writes for each trial:
  <out>/images/<id>.png    KM figure with a number-at-risk table underneath
  <out>/truth/<id>.json    patient data, true Cox HR, at-risk table, arm names

Usage:
  python -m km.synth --n 20 --out data/km
  python -m km.synth --n 20 --out data/km_cam --camera
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402

ARM_PAIRS = [
    ("Pembrolizumab", "Chemotherapy"), ("Nivolumab", "Docetaxel"), ("Osimertinib", "Standard EGFR-TKI"),
    ("Drug A", "Placebo"), ("Experimental", "Control"), ("Atezolizumab", "Chemotherapy"),
]
COLOR_PAIRS = [("#1f4e9c", "#d1495b"), ("#2a9d8f", "#e76f51"), ("#5e3c99", "#e66101"), ("#0072b2", "#d55e00")]


def simulate_trial(rng: np.random.Generator):
    n = int(rng.integers(120, 420))
    n_ctrl = n // 2
    n_trt = n - n_ctrl
    hr = float(rng.uniform(0.45, 1.0))
    median_ctrl = float(rng.uniform(8, 20))  # months
    shape = float(rng.uniform(0.8, 1.5))  # Weibull shape
    follow = float(rng.choice([24, 30, 36, 48]))
    accrual = follow * 0.5

    def arm(n_arm, hr_arm):
        # Weibull with proportional hazards: S(t) = exp(-(t/scale)^k * hr)
        scale = median_ctrl / (np.log(2) ** (1 / shape))
        u = rng.uniform(size=n_arm)
        t = scale * (-np.log(u) / hr_arm) ** (1 / shape)
        dropout = rng.exponential(follow * 3, n_arm)
        admin = follow - rng.uniform(0, accrual, n_arm)
        obs = np.minimum.reduce([t, dropout, admin])
        event = (t <= np.minimum(dropout, admin)).astype(int)
        return obs, event

    t_ctrl, e_ctrl = arm(n_ctrl, 1.0)
    t_trt, e_trt = arm(n_trt, hr)
    df = pd.DataFrame({
        "time": np.concatenate([t_trt, t_ctrl]),
        "event": np.concatenate([e_trt, e_ctrl]),
        "treat": np.concatenate([np.ones(n_trt), np.zeros(n_ctrl)]),
    })
    cph = CoxPHFitter().fit(df, "time", "event")
    s = cph.summary.loc["treat"]
    true_hr = {"hr": float(np.exp(s["coef"])), "lo": float(np.exp(s["coef lower 95%"])),
               "hi": float(np.exp(s["coef upper 95%"])), "p": float(s["p"])}
    return df, true_hr, follow


def draw_km(df, follow, arms, colors, rng, dpi=110):
    step = 6 if follow <= 36 else 12
    ticks = np.arange(0, follow + 0.01, step)
    pct = rng.random() < 0.5  # y axis as 0-100% or 0-1
    fig = plt.figure(figsize=(float(rng.uniform(6.5, 8)), float(rng.uniform(5.2, 6.2))), dpi=dpi)
    ax = fig.add_axes([0.22, 0.36, 0.72, 0.58])
    at_risk = {}
    for k, (name, col) in enumerate(zip(arms, colors)):
        d = df[df.treat == (1 - k)]
        kmf = KaplanMeierFitter().fit(d.time, d.event)
        sf = kmf.survival_function_
        y = sf.iloc[:, 0].values * (100 if pct else 1)
        ax.step(sf.index.values, y, where="post", color=col, lw=2.0, label=name)
        cens = d.time[d.event == 0].values
        cy = kmf.survival_function_at_times(cens).values * (100 if pct else 1)
        ax.plot(cens, cy, "|", color=col, ms=6, mew=1.2)
        at_risk[name] = [int((d.time >= t).sum()) for t in ticks]
    ax.set_xlim(0, follow)
    ax.set_ylim(0, 105 if pct else 1.05)
    ax.set_xticks(ticks)
    ax.set_yticks(np.arange(0, 101, 20) if pct else np.round(np.arange(0, 1.01, 0.2), 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Months since randomization")
    ax.set_ylabel("Overall survival (%)" if pct else "Probability of survival")
    ax.legend(loc="upper right", frameon=False)

    # number-at-risk table, aligned with the x ticks
    trans = ax.get_xaxis_transform()
    fig.text(0.02, 0.21, "No. at risk", fontsize=9, fontweight="bold")
    for r, (name, col) in enumerate(zip(arms, colors)):
        yfig = 0.16 - r * 0.055
        fig.text(0.02, yfig, name, fontsize=9, color=col)
        for t, nrisk in zip(ticks, at_risk[name]):
            xfig = fig.transFigure.inverted().transform(trans.transform((t, 0)))[0]
            fig.text(xfig, yfig, str(nrisk), fontsize=9, ha="center", color=col)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img, {"ticks": ticks.tolist(), "at_risk": at_risk, "percent": pct}


def make_trial(seed: int):
    rng = np.random.default_rng(seed)
    df, true_hr, follow = simulate_trial(rng)
    arms = list(ARM_PAIRS[seed % len(ARM_PAIRS)])
    colors = list(COLOR_PAIRS[seed % len(COLOR_PAIRS)])
    img, table = draw_km(df, follow, arms, colors, rng)
    truth = {
        "arms": arms, "colors": colors, "true_hr": true_hr, "follow_up": follow, **table,
        "n": {arms[0]: int((df.treat == 1).sum()), arms[1]: int((df.treat == 0).sum())},
        "events": {arms[0]: int(df[df.treat == 1].event.sum()), arms[1]: int(df[df.treat == 0].event.sum())},
    }
    return img, truth, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/km")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        img, truth, _ = make_trial(args.seed + i)
        if args.camera:
            from synth.generate import photograph

            img, _ = photograph(img, np.random.default_rng(args.seed + i + 5000))
        sid = f"trial_{i:03d}"
        cv2.imwrite(str(out / "images" / f"{sid}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        (out / "truth" / f"{sid}.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {args.n} trials to {out}")


if __name__ == "__main__":
    main()
