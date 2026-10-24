"""Smoke tests: run with `pytest -q`."""
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from pipeline import run
from synth.generate import make_chart, photograph


def _err(res, truth):
    s = res["series"][0]
    ts = truth["series"][0]
    px, py = np.array(s["x"]), np.array(s["y"])
    o = np.argsort(px)
    tx, ty = np.array(ts["x"]), np.array(ts["y"])
    inside = (tx >= px.min()) & (tx <= px.max())
    yr = truth["ylim"][1] - truth["ylim"][0]
    return np.mean(np.abs(np.interp(tx[inside], px[o], py[o]) - ty[inside])) / yr


def _single_line_chart(seed=1):
    for s in range(seed, seed + 50):
        img, truth, _ = make_chart(s, kind="line")
        if len(truth["series"]) == 1:
            return img, truth
    raise RuntimeError("no single-series chart found")


def test_clean_chart():
    img, truth = _single_line_chart()
    res = run(img, do_rectify=False, want_images=False)
    assert res["ok"] and res["calibrated"]
    assert _err(res, truth) < 0.03
    assert res["overall_match"] > 0.9


def test_photographed_chart():
    img, truth = _single_line_chart(5)
    photo, _ = photograph(img, np.random.default_rng(3))
    res = run(photo, do_rectify=True, want_images=False)
    assert res["ok"] and res["rectified"]


def test_api_health():
    assert TestClient(app).get("/api/health").json() == {"ok": True}


def test_guyot_recovers_hr_from_exact_curves():
    from lifelines import KaplanMeierFitter

    from km.guyot import reconstruct
    from km.stats import hazard_ratio, to_frame
    from km.synth import make_trial

    _, truth, df = make_trial(101)
    grid = np.linspace(0, truth["follow_up"], 600)
    arms = {}
    for role, treat, name in (("treatment", 1, truth["arms"][0]), ("control", 0, truth["arms"][1])):
        d = df[df.treat == treat]
        s = KaplanMeierFitter().fit(d.time, d.event).survival_function_at_times(grid).values
        arms[role] = reconstruct(grid, s, truth["ticks"], truth["at_risk"][name])
        assert abs(len(arms[role][0]) - len(d)) <= 2
    hr = hazard_ratio(to_frame(arms))["hr"]
    assert abs(hr - truth["true_hr"]["hr"]) / truth["true_hr"]["hr"] < 0.05


def test_km_figure_end_to_end():
    from km.pipeline import analyze_trial, read_trial
    from km.synth import make_trial

    img, truth, _ = make_trial(103)
    read = read_trial(img, do_rectify=False)
    assert len(read["arms"]) == 2
    res = analyze_trial(read, treatment_index=0)
    assert abs(res["hr"]["hr"] - truth["true_hr"]["hr"]) / truth["true_hr"]["hr"] < 0.1


def test_pooling():
    from km.stats import pool

    p = pool([{"log_hr": np.log(0.7), "se": 0.1}, {"log_hr": np.log(0.8), "se": 0.12}])
    assert 0.65 < p["hr"] < 0.85 and p["lo"] < p["hr"] < p["hi"]
