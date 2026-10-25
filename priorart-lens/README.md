# ☕ Bean

**Studies publish survival curves, not their data. Bean reads the figure, rebuilds the
individual-level data, checks itself against the paper, and compares treatments that were never
tested head to head.**

HackPSU Fall 2026, Research Track: "Build a tool that makes scientific research faster."
(The project started under the name PriorArt Lens.)

**Live demo:** https://huggingface.co/spaces/YOUR-USERNAME/bean

## The problem
Researchers comparing treatments across studies need each study's individual-level survival data.
Studies almost never share it. They publish a Kaplan-Meier figure and a "number at risk" table
instead. Today researchers hand-trace those curves point by point, type in the table, and run a
reconstruction script. It takes hours per figure, it is error-prone, and nobody checks the result.

Doctors also often need to choose between two new drugs that were **never tested against each
other**, only each against the same older treatment in separate trials. A valid comparison exists,
but it needs good data from both trials first.

## What Bean does
1. **Bring in a figure.** Upload a screenshot or phone photo, **or paste a paper link** (DOI,
   PubMed, PMC or a journal URL) and type the figure, like "Figure 3C". For open-access papers Bean
   finds the paper, downloads the figure, reads the caption and fills in the trial name. Drag a box
   around the panel you want.
2. **Read it automatically** with computer vision and OCR: axes, tick labels, each arm's curve,
   the number-at-risk table, the outcome (overall vs progression-free survival) and any printed
   hazard ratio. No LLM reads the figure.
3. **Check the numbers** in an editable table, with warnings when something looks off.
4. **Rebuild the patients** with the Guyot et al. (2012) algorithm: hazard ratio, 95% CI, p-value,
   medians.
5. **Verify**: redraw the curves from the rebuilt patients and measure the match (curve gap and
   pixel match), and compare with the paper's hazard ratio (flagged if more than 10% off).
6. **Compare two new treatments** tested against the same old one (Bucher anchored indirect
   comparison). Unsafe comparisons are blocked: mixed outcomes, different control treatments, or a
   drug compared with itself.
7. **Pool** several studies of the same drug (DerSimonian-Laird random effects, forest plot, I²).
8. **Ask Mr. Bean**, a coffee-bean help assistant powered by Google Gemini. He answers questions
   about the site, the computer vision and the statistics, and explains the numbers on your screen.

## Accuracy
Against the hazard ratios reported in the papers:

| Figure | Bean | Paper | Off by |
|---|---|---|---|
| CheckMate 017 (nivolumab vs docetaxel, OS) | 0.591 | 0.59 | 0.2% |
| OAK squamous (atezolizumab vs docetaxel, OS) | 0.743 | 0.73 | 1.8% |
| OAK squamous, fetched from the paper link | 0.73 | 0.73 | 0.6% |
| ORIENT-3 (PFS) | 0.484 | 0.52 | 6.9% |
| S-1 vs docetaxel | 1.044 | 0.945 | 10.5%, flagged by Bean |

Simulated trials with known true answers (`km/evaluate.py`): mean HR error 1.8% on clean figures,
true HR inside our 95% CI in 20 of 20. On 8 fresh figures made for the LLM test, mean error 1.3%.

Example: nivolumab vs atezolizumab, never tested head to head, compared through their shared
docetaxel arm: HR 0.79 (95% CI 0.53 to 1.20). It leans toward nivolumab, and Bean correctly says
the data cannot confirm a difference.

### Why not just ask an LLM?
`scripts/llm_test.py` asks Gemini to estimate the hazard ratio from figures with the answers hidden.
On the famous CheckMate 017 figure it returned the paper's exact numbers, including a confidence
interval that cannot be read from the picture: it had memorized the paper. `--synthetic` runs the
same test on 8 never-published figures, where there is nothing to remember. Bean measures; it does
not recall.

## Setup (Windows, VS Code)
In a **PowerShell** terminal inside the `priorart-lens` folder:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open http://localhost:8000. **From Git Bash** use forward slashes:
`.venv/Scripts/python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

**Mr. Bean:** get a free key at https://aistudio.google.com/apikey, copy `.env.example` to `.env`
and set `GEMINI_API_KEY=...`, then restart. The key stays on the server and `.env` is git-ignored.
Optional: `GEMINI_MODEL=gemini-3.6-flash`.

Tests: `.\.venv\Scripts\python.exe -m pytest -q` (35 tests).

## Deploy (Hugging Face Spaces)
The `Dockerfile` runs the web app and API in one container on port 7860.
1. Create a Space with SDK **Docker** (template Blank, free CPU).
2. In Settings → Variables and secrets, add the secret `GEMINI_API_KEY`.
3. Upload `app/`, `km/`, `pipeline/`, `Dockerfile` and `requirements.txt` (keep the Space's own
   README; `deploy/SPACE_README.md` shows the settings it needs).

## Code map
```
app/main.py          FastAPI server: read, analyze, compare, pool, paper lookup, chat
app/static/          web UI (survival.html is the main page)
km/extract.py        read curves + number-at-risk table + outcome + printed HR from an image
km/guyot.py          Guyot 2012 reconstruction of individual-level data
km/stats.py          Cox hazard ratio, medians, indirect comparison, pooling, charts
km/pipeline.py       read_trial() and analyze_trial(), used by the API
km/paper.py          paper link + "Figure 2A" -> figure image and caption (Europe PMC, PMC)
km/assistant.py      Mr. Bean (Gemini) with the knowledge base
km/synth.py          simulated trials drawn as journal-style figures
km/evaluate.py       benchmark: figure -> rebuilt HR vs the TRUE HR
pipeline/            general chart reader: rectify, axes, ticks (RapidOCR), series, verify
scripts/llm_test.py  LLM comparison test
data/real/           real trial figures with citations (SOURCES.md)
```

## Methods
- Reconstruction: Guyot P, Ades AE, Ouwens MJ, Welton NJ. *BMC Med Res Methodol* 2012;12:9.
- Indirect comparison: Bucher HC et al. *J Clin Epidemiol* 1997;50:683-691.
- Pooling: DerSimonian R, Laird N. *Control Clin Trials* 1986;7:177-188.

## Honest limits
- Rebuilt data approximates the original data, typically within 1 to 2% on the hazard ratio; it is not the real patient data.
- Validated on oncology trials; works on any Kaplan-Meier figure with a number-at-risk table.
- Two arms per figure. Black-and-white figures with overlapping curves are harder.
- Indirect comparisons assume the two trials enrolled similar patients. Bean checks the outcome
  and the shared control arm; the researcher must check the populations.
- Paper links work for open-access papers only; supplementary figures need a screenshot.
- Saved studies are stored in each visitor's browser.
