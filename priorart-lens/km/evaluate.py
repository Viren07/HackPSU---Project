"""Benchmark: picture of a trial -> reconstructed hazard ratio vs the TRUE one.

Usage:
  python -m km.synth --n 20 --out data/km
  python -m km.evaluate --data data/km
  python -m km.synth --n 20 --out data/km_cam --camera
  python -m km.evaluate --data data/km_cam --rectify
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .pipeline import analyze_trial, read_trial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/km")
    ap.add_argument("--rectify", action="store_true")
    args = ap.parse_args()
    d = Path(args.data)
    errs, table_ok, status = [], [], {}
    for tp in sorted((d / "truth").glob("*.json")):
        truth = json.loads(tp.read_text())
        img = cv2.cvtColor(cv2.imread(str(d / "images" / f"{tp.stem}.png")), cv2.COLOR_BGR2RGB)
        try:
            read = read_trial(img, do_rectify=args.rectify)
            names = [a["name"] for a in read["arms"]]
            if len(read["arms"]) != 2:
                raise ValueError("Need exactly two arms to compare.")
            # match read arms to true arms by curve color (truth arm 0 = treatment)
            tc = [np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) for c in truth["colors"]]
            d0 = [np.linalg.norm(np.array(a["color"]) - tc[0]) for a in read["arms"]]
            ti = int(np.argmin(d0))
            ok_rows = [read["arms"][ti]["n_risk"] == truth["at_risk"][truth["arms"][0]],
                       read["arms"][1 - ti]["n_risk"] == truth["at_risk"][truth["arms"][1]]]
            table_ok += ok_rows
            res = analyze_trial(read, treatment_index=ti)
        except Exception as e:
            status["failed"] = status.get("failed", 0) + 1
            print(f"{tp.stem}: FAILED ({e})")
            continue
        status["ok"] = status.get("ok", 0) + 1
        t_hr, r_hr = truth["true_hr"]["hr"], res["hr"]["hr"]
        err = abs(r_hr - t_hr) / t_hr * 100
        errs.append(err)
        ci_ok = res["hr"]["lo"] <= t_hr <= res["hr"]["hi"]
        print(f"{tp.stem}: true HR {t_hr:.3f}  rebuilt {r_hr:.3f}  ({err:4.1f}% off)  "
              f"tables {'OK' if all(ok_rows) else 'needs fix'}  true HR in CI: {ci_ok}  names {names}")
    print("\n=== summary ===", status)
    if errs:
        e = np.array(errs)
        print(f"HR error: mean {e.mean():.1f}%  median {np.median(e):.1f}%  within 5%: {(e < 5).mean() * 100:.0f}%")
        print(f"at-risk rows read perfectly: {np.mean(table_ok) * 100:.0f}%")


if __name__ == "__main__":
    main()
