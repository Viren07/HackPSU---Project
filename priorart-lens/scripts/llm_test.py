"""Ask a multimodal LLM (Gemini) to estimate the hazard ratio from a survival figure with the answers hidden.

The figure is CheckMate 017 with the printed hazard ratio, medians and death counts covered.
Paper's answer: HR 0.59 (95% CI 0.44 to 0.79), median OS 9.2 vs 6.0 months.

Run from the project folder (needs GEMINI_API_KEY in .env):
    .venv/Scripts/python.exe -m scripts.llm_test                   famous figure, 5 runs
    .venv/Scripts/python.exe -m scripts.llm_test --synthetic       8 never-published figures

The famous figure can be answered from memory (the model may have read the paper). The synthetic
figures were generated from simulated patients for this test, so the only way to answer is to measure.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from km.assistant import API, FALLBACK_MODELS, load_env

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "data" / "llm_test" / "checkmate017_masked.png"
TRUTH = {"hr": 0.59, "lo": 0.44, "hi": 0.79, "median_nivolumab": 9.2, "median_docetaxel": 6.0}
OURS = {"hr": 0.591, "lo": 0.45, "hi": 0.78, "median_nivolumab": 9.2, "median_docetaxel": 6.0}

PROMPT = """This is a Kaplan-Meier figure from a randomized trial ({new} vs {old}), with a
number-at-risk table. Estimate, as accurately as you can:
1. the hazard ratio for death, {new} vs {old}, with its 95% confidence interval
2. the median overall survival in months for each arm
Reply with ONLY a JSON object, no other text:
{{"hr": number, "lo": number, "hi": number, "median_{a}": number, "median_{b}": number}}"""
SYNTH = ROOT / "data" / "llm_test" / "synthetic"


def ask_once(model: str, key: str, img_b64: str, new="nivolumab", old="docetaxel") -> dict:
    a, b = ("nivolumab", "docetaxel") if new == "nivolumab" else ("new", "old")
    prompt = PROMPT.format(new=new, old=old, a=a, b=b)
    body = {"contents": [{"role": "user", "parts": [
        {"inline_data": {"mime_type": "image/png", "data": img_b64}}, {"text": prompt}]}],
        "generationConfig": {"temperature": 1.0}}
    req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in answer: {text[:200]}")
    return {k: float(v) for k, v in json.loads(m.group(0)).items() if isinstance(v, (int, float))}


def pick_model(args, key, img_b64, **kw):
    models = [args.model or os.environ.get("GEMINI_MODEL")] if (args.model or os.environ.get("GEMINI_MODEL")) else FALLBACK_MODELS
    for m in models:  # first model this key can use right now
        try:
            return m, ask_once(m, key, img_b64, **kw)
        except urllib.error.HTTPError as e:
            print(f"{m}: HTTP {e.code}, trying next model")
    raise SystemExit("No Gemini model answered.")


def synthetic(args, key):
    cases = json.loads((SYNTH / "truth.json").read_text())
    model, rows = None, []
    print("Asking the model about figures it cannot have seen before...\n")
    for c in cases:
        img_b64 = base64.b64encode((SYNTH / c["file"]).read_bytes()).decode()
        guesses = []
        for k in range(args.runs):
            try:
                if model is None:
                    model, g = pick_model(args, key, img_b64, new=c["new"], old=c["old"])
                else:
                    g = ask_once(model, key, img_b64, new=c["new"], old=c["old"])
                guesses.append(g.get("hr"))
            except Exception as e:
                print(f"{c['file']} run {k + 1} failed: {e}")
            time.sleep(1)
        rows.append({**c, "llm_hr": guesses})
    print(f"Model: {model}   ({args.runs} run(s) per figure)\n")
    print(f"{'figure':<14}{'label':<34}{'true HR':>8}{'LLM HR':>16}{'LLM off':>9}{'ours':>7}{'ours off':>9}")
    llm_err, our_err = [], []
    for r in rows:
        g = [x for x in r["llm_hr"] if x]
        t = r["truth"]["hr"]
        e = [abs(x - t) / t * 100 for x in g]
        llm_err += e
        our_err.append(r["ours"]["error_pct"])
        gtxt = ", ".join(f"{x:.2f}" for x in g) or "n/a"
        print(f"{r['file'][:-4]:<14}{(r['new'] + ' vs ' + r['old'])[:33]:<34}{t:>8.2f}{gtxt:>16}"
              f"{(statistics.mean(e) if e else float('nan')):>8.0f}%{r['ours']['hr']:>7.2f}{r['ours']['error_pct']:>8.1f}%")
    if llm_err:
        print(f"\nLLM:  mean error {statistics.mean(llm_err):.1f}%, worst {max(llm_err):.1f}%")
    print(f"Ours: mean error {statistics.mean(our_err):.1f}%, worst {max(our_err):.1f}%")
    out = ROOT / "data" / "llm_test" / "results_synthetic.json"
    out.write_text(json.dumps({"model": model, "rows": rows}, indent=2))
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=None, help="runs (default 5 famous, 1 per synthetic figure)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--synthetic", action="store_true", help="use the 8 never-published figures")
    args = ap.parse_args()
    load_env(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Add GEMINI_API_KEY to .env first.")
    if args.synthetic:
        args.runs = args.runs or 1
        return synthetic(args, key)
    args.runs = args.runs or 5
    img_b64 = base64.b64encode(IMAGE.read_bytes()).decode()

    model, first = pick_model(args, key, img_b64)

    runs = [first]
    for i in range(args.runs - 1):
        time.sleep(1)
        try:
            runs.append(ask_once(model, key, img_b64))
        except Exception as e:
            print(f"run {i + 2} failed: {e}")

    print(f"\nModel: {model}   runs: {len(runs)}   (figure with HR, medians and deaths hidden)\n")
    print(f"{'run':<6}{'HR':>7}{'95% CI':>16}{'median nivo':>14}{'median doce':>14}{'HR off by':>12}")
    for i, r in enumerate(runs, 1):
        off = abs(r.get("hr", 0) - TRUTH["hr"]) / TRUTH["hr"] * 100
        print(f"{i:<6}{r.get('hr', float('nan')):>7.2f}   {r.get('lo', float('nan')):.2f} to {r.get('hi', float('nan')):.2f}"
              f"{r.get('median_nivolumab', float('nan')):>14.1f}{r.get('median_docetaxel', float('nan')):>14.1f}{off:>11.1f}%")
    hrs = [r["hr"] for r in runs if "hr" in r]
    errs = [abs(h - TRUTH["hr"]) / TRUTH["hr"] * 100 for h in hrs]
    print(f"\nLLM:   HR range {min(hrs):.2f} to {max(hrs):.2f}, mean error {statistics.mean(errs):.1f}%, worst {max(errs):.1f}%")
    print(f"Ours:  HR {OURS['hr']:.3f} ({OURS['lo']} to {OURS['hi']}), error 0.2%, same answer every run")
    print(f"Paper: HR {TRUTH['hr']} ({TRUTH['lo']} to {TRUTH['hi']}), medians {TRUTH['median_nivolumab']} vs {TRUTH['median_docetaxel']}")
    out = ROOT / "data" / "llm_test" / "results.json"
    out.write_text(json.dumps({"model": model, "runs": runs, "truth": TRUTH, "ours": OURS}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
