# PriorArt Lens

**Cancer trials publish survival curves, not patient data. Photograph the curves, rebuild the
patients, check the result against the paper, and pool trials into a meta-analysis.**

HackPSU Fall 2026, Research Track: "Build a tool that makes scientific research faster."

## The problem
To compare cancer treatments across trials, researchers need each trial's patient-level survival
data. Trials almost never share it. They publish a Kaplan-Meier figure plus a "number at risk"
table instead. Today researchers hand-digitize those curves point by point, then run a
reconstruction algorithm. It's slow and error-prone, and it sits at the start of nearly every
survival meta-analysis and health-technology assessment.

## What PriorArt Lens does
1. **Read the figure** from a phone photo, screenshot or PDF crop: straighten it, find the axes,
   read the tick labels, and trace each treatment arm's curve by color.
2. **Read the number-at-risk table** under the figure and match each row to its curve.
3. **Let a person check the numbers** in an editable table (OCR is never 100%).
4. **Rebuild the patients** with the Guyot et al. (2012) algorithm.
5. **Verify**: redraw the curves from the rebuilt patients and compare them with the figure
   (curve gap + pixel match), and compare our hazard ratio with the one the paper reports.
6. **Pool trials**: random-effects meta-analysis (DerSimonian-Laird), with a forest plot and a
   plain-English verdict.

## Setup (Windows, VS Code)
In a **PowerShell** terminal inside the project folder:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open http://localhost:8000. Test figures are in `data\km\images` (clean) and
`data\km_cam\images` (fake phone photos; keep "Straighten photo" on for those).
The original general-purpose chart reader is at http://localhost:8000/chart.

**From Git Bash** use forward slashes: `.venv/Scripts/python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

**Phone:** same Wi-Fi, open `http://<laptop-ip>:8000` (`ipconfig` shows the IP). Allow Python
through the firewall. Venue Wi-Fi often blocks this, so use a phone hotspot as the backup.

## Code map
```
km/synth.py      simulated trials (real patient data) drawn as journal-style figures
km/extract.py    read curves + number-at-risk table from an image
km/guyot.py      Guyot 2012 reconstruction of individual patient data
km/stats.py      Cox hazard ratio, medians, random-effects pooling, forest plot
km/pipeline.py   read_trial() and analyze_trial(), used by the API
km/evaluate.py   benchmark: figure -> rebuilt HR vs the TRUE HR
pipeline/        general chart reader (rectify, axes, ticks, series, verify)
app/             FastAPI server + web UI (survival.html is the main page)
```

## Benchmarks (simulated trials with known true hazard ratios)
```powershell
.\.venv\Scripts\python.exe -m km.evaluate --data data/km
.\.venv\Scripts\python.exe -m km.evaluate --data data/km_cam --rectify
```

| Test | Result |
|---|---|
| Reconstruction from exact curves (15 trials) | HR within ~1.5% on average; exact patient counts |
| Clean figures, fully automatic (20 trials) | mean HR error 1.8%, 19/20 within 5%; true HR inside our 95% CI in 20/20; 92% of at-risk rows read perfectly |
| Fake phone photos, fully automatic (20 trials) | 14/20 complete automatically; median HR error 1.4%; at-risk rows 50% perfect. The editable table catches the rest |

## Next steps (in priority order)
- [ ] **Real trials**: 3 to 5 open-access trials comparing the same two treatments, each with a
      KM figure + at-risk table + reported HR. Compare our HR with the paper's for the evidence slide.
- [ ] Photo robustness: blur-aware OCR and better curve tracing on photos.
- [ ] Black/gray curves (solid vs dashed), common in journals.
- [ ] Uncertainty from extraction (bootstrap over digitization noise), shown as wider CIs.
- [ ] Devpost write-up, 5-minute video, printed figures for the demo table.

## Honest limits
- Reconstructed data approximates the original patient data; it is not the real data.
- Colored curves only for now; same-colored or black/gray curves need work.
- Two-arm trials only.
