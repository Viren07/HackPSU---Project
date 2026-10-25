"""Mr. Bean: the in-page help assistant, answered by Gemini.

The API key stays on the server. Set GEMINI_API_KEY in the environment or in a `.env` file at the
project root (never committed). GEMINI_MODEL optionally picks the model.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]


def load_env(path: Path):
    """Tiny .env reader: KEY=VALUE lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


KNOWLEDGE = r"""
# PriorArt Lens: what it is
A web tool (HackPSU Fall 2026, research track: "build a tool that makes scientific research faster").
Cancer trials publish Kaplan-Meier (KM) survival curves and a "number at risk" table, not their
patient-level data. PriorArt Lens reads the figure automatically, rebuilds approximate patient-level
data (one row per patient: time, died or censored), checks the rebuild against the paper, and compares
two NEW drugs that were each tested against the SAME OLD drug in separate trials.
Main demo: CheckMate 017 (nivolumab vs docetaxel, squamous NSCLC) and OAK panel C (atezolizumab vs
docetaxel, squamous). Result: nivolumab vs atezolizumab HR about 0.79 (95% CI about 0.53 to 1.20):
leans toward nivolumab but not conclusive.
Why it matters: researchers doing meta-analyses and health technology assessments must hand-digitize
curves point by point (tools like WebPlotDigitizer), then run the reconstruction in R. That takes
hours per figure and is error-prone. PriorArt Lens does it in seconds and verifies itself.

# How to use the site (step by step)
The bar at the top shows 4 steps: 1 Read figure, 2 Check numbers, 3 Rebuild patients, 4 Compare.
Step 1, Add a trial (two ways):
 a) Upload: type a trial name (optional), optionally the hazard ratio the paper reports, then press
    "Photo or upload of survival figure". Phone photos: keep "Straighten photo" ON. Screenshots or PDF
    crops: turn it OFF. The crop must include the curves AND the number-at-risk table under them.
 b) From the paper: paste a paper link, DOI or PubMed id, type the figure like "Figure 2A" or
    "Figure 3 C - Squamous", press "Find figure". The tool looks the paper up on Europe PMC / PubMed
    Central, downloads that figure, shows title, authors, journal and caption, and fills the trial
    name. Drag a box around the wanted panel (curves plus its at-risk table) and press "Read this box",
    or "Use the whole figure" if there is only one panel. Only open-access papers work (those in PMC).
    Supplementary figures are not supported; upload a screenshot instead. Small panels are enlarged
    automatically before reading.
Step 2, Check what we read: the left image shows the traced curves (magenta dots) and the plot box.
 - "What does this curve measure?": the outcome (overall survival OS, progression-free survival PFS,
   disease-free, event-free, relapse-free). Detected from the y-axis title or text on the figure, or
   from the paper's caption when the figure came from a link. If not detected you must choose it.
 - If the hazard ratio is printed on the figure, it is read and filled into the "reported" box.
 - Each arm has an editable name, a "new treatment" radio button, and editable at-risk counts at each
   time point. Fix any misread number, then press "Rebuild patients & compute hazard ratio".
 - Warnings (e.g. a curve or row could not be matched, or the x-axis seemed shifted) appear at the top.
Step 3, Rebuilt trial: shows the hazard ratio (big number), its 95% CI and p-value, the comparison with
 the paper's HR (% off; over 10% is flagged as a likely misread), patients rebuilt and deaths/events,
 median survival per arm, curve fit (average gap between rebuilt and read curve, in percentage points)
 and pixel match (share of the original curve's pixels covered by the redrawn curve). The right image
 overlays thick (read from figure) and thin (rebuilt) curves. Press "Save this study" to keep it.
Your studies: table of saved studies (stored in this browser only, localStorage). You can also add a
 study by its reported HR and 95% CI when there is no usable graph (expandable section). "Remove" and
 "Clear all studies" delete them.
Step 4, Compare two new treatments: pick Study A and Study B and press Compare. It is blocked, with an
 explanation, when: both test the same new drug (use pooling instead), the outcomes differ (e.g. OS vs
 PFS; never mix them), an outcome is missing, or the old (shared) treatments differ. The verdict says
 which drug had lower risk, by how much, and whether the 95% range excludes "no difference". A chart
 shows both trials and the comparison.
Pooling (collapsed "Combine several studies of the same drug"): only for studies of the same new drug
 vs the same old drug with the same outcome. Random-effects meta-analysis with a forest plot and I².
Other page: /chart is the older general chart reader.
Running it: `.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`, then
 open http://localhost:8000. Setup: `powershell -ExecutionPolicy Bypass -File .\setup.ps1`.
 Mr. Bean needs GEMINI_API_KEY in a .env file at the project root.

# Computer vision pipeline (no AI model, no LLM reads the figure; classic CV + OCR)
1. Straighten (optional): find the page's 4 corners and apply a homography (perspective transform) so
   a tilted phone photo becomes flat.
2. Find the plot box: detect long straight dark lines; the x and y axes form an "L" corner; choose the
   smallest box that contains at least 60% of the colored (curve) pixels, with fallbacks.
3. Read axis numbers: RapidOCR (an open-source ONNX OCR engine, runs locally) reads tick labels in
   strips beside the axes. A straight line maps pixels to values, fit robustly (RANSAC-style) with the
   rule that x values increase to the right and y values decrease going up the image. y may be 0-100%
   or 0-1. Time 0 must sit at the y-axis; if the reading looks shifted by one tick it is corrected and
   a warning is shown.
4. Separate the curves by color: pixels inside the plot are converted to Lab color space and clustered
   with k-means; shades of the same hue are merged; the legend area is masked out. Each color is one arm.
5. Trace each curve column by column (forced to be a continuous step line), convert pixels to
   (time, survival) with the axis fits, and clean it so survival starts at 1 and never goes up.
6. Read the number-at-risk table: find text rows under the plot (removing table lines), OCR each row,
   with small tiles so single digits like "7" are not missed, handle "84 (4)" censor-pair formats, and
   keep the longest non-increasing sequence (counts can only go down). Numbers are aligned to the x-axis
   tick times.
7. Match rows to curves: by color of the row label when the table is colored; for black tables by label
   names near the curves plus a consistency rule: n(t)/n(0) cannot be larger than S(t).
8. Detect the outcome (y-axis title first, other figure text second) and a printed hazard ratio.
Typical read takes about 4 to 8 seconds on a laptop.

# Math
Kaplan-Meier curve: the share of patients still event-free over time; it drops at each death/event;
 "+" tick marks are censored patients (left the study or still alive at data cutoff, not dead).
Number at risk: patients still being followed (alive and not censored) at each time.
Reconstruction: Guyot et al. 2012 (BMC Medical Research Methodology 12:9). Inside each interval between
 at-risk times, it estimates how many patients were censored (spread evenly) and how many died so the
 rebuilt KM curve reproduces the read curve and the at-risk counts. Output: one row per patient (time,
 event). We also force the total patients to equal the at-risk count at time 0.
Hazard ratio (HR): from a Cox proportional hazards model (Python library lifelines) on the rebuilt
 patients. HR < 1: the new drug has a lower risk of the event at any given moment; HR 0.74 means about
 26% lower. HR = 1 means no difference. It is about the instantaneous risk ("hazard"), not the total
 number of deaths: everyone eventually dies; the drug buys time.
95% confidence interval (CI): the range the true HR probably lies in. If the whole range is below 1 the
 benefit is statistically clear; if it includes 1, the data cannot confirm a difference.
p-value: chance of seeing a difference at least this big if the drugs were truly equal; < 0.05 is the
 usual cutoff.
Median survival: time when half the patients have had the event.
Why rebuild instead of just reading the paper's HR: rebuilt data lets you compute anything (medians,
 survival at any time, other models, subgroups consistent with the figure), check the paper, and handle
 papers that do not print an HR; it also gives a standard error needed for comparisons.
Comparing two new drugs (indirect treatment comparison, Bucher 1997): trial 1 gives HR(A vs C), trial 2
 gives HR(B vs C), same old drug C. log HR(A vs B) = log HR(A vs C) - log HR(B vs C); its standard error
 is sqrt(SE_A^2 + SE_B^2), so the comparison is less precise than either trial. It keeps each trial's
 randomization ("anchored" on C). Assumption: the two trials enrolled similar patients (same disease
 stage, histology, prior treatment, same outcome definition). Used in health technology assessment
 (e.g. NICE). It is not a substitute for a head-to-head trial.
Pooling: DerSimonian-Laird random-effects meta-analysis of log HRs. I² = share of variation between
 trials beyond chance: below 25% consistent, 25 to 50% moderate, above 50% very different.

# Accuracy (validated)
Real figures: CheckMate 017 HR 0.591 vs paper 0.59 (0.2% off); OAK squamous 0.743 vs 0.73 (1.8%);
 OAK from the paper link (panel C) 0.73 to 0.74 vs 0.73; ORIENT-3 (PFS) 0.484 vs 0.52 (6.9%); a S-1 trial
 1.044 vs 0.945 (10.5%, flagged). Synthetic benchmark: all at-risk tables read correctly, mean HR error
 2.1%. Rebuilt data approximates, not replaces, the true patient data, typically within 1 to 2% on HR.

# Limits
Needs a KM figure with a number-at-risk table. Two arms per figure. Black-and-white figures with
 overlapping curves are harder. Indirect comparisons assume similar trial populations; the tool checks
 outcome type and the shared old drug, but the researcher must check the populations (e.g. squamous vs
 non-squamous). Saved studies live in the browser only.
Why not just ask an LLM: LLMs estimate numbers from images and can be confidently wrong; this tool
 measures pixels, reads the table, shows every number for checking, and verifies against the paper.
Privacy: figures are processed on the local server; only Mr. Bean's questions go to Gemini.
"""

PERSONA = """You are Mr. Bean, a friendly coffee bean who is the help assistant inside PriorArt Lens.
Answer questions about how to use the site, how it works (computer vision, OCR, statistics, the
reconstruction method, comparisons), how to read its results, and the science behind it, using the
knowledge below. Rules:
- Be accurate. If the knowledge does not cover something about this site, say you are not sure rather
  than inventing features. General statistics or oncology concepts may be explained from general knowledge.
- Keep answers short and plain (usually 2 to 6 sentences, or a short list for steps). Explain jargon
  simply. Use **bold** sparingly. Never use em dashes.
- If the user's current page state is provided, use it to explain THEIR numbers.
- Do not give medical advice for a specific patient; this is a research tool.
- Off-topic questions: answer briefly if harmless, then steer back to the tool.
- A light coffee pun now and then is fine; do not overdo it.
"""


class AssistantError(Exception):
    pass


def ask(messages: list[dict], page_state: dict | None = None, timeout: int = 40) -> str:
    """messages: [{"role": "user"|"assistant", "text": ...}], oldest first."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise AssistantError("Mr. Bean is not set up yet: add GEMINI_API_KEY=your-key to a .env file in the "
                             "project folder, then restart the server.")
    msgs = [m for m in messages if m.get("text")][-12:]
    if not msgs or msgs[-1].get("role") != "user":
        raise AssistantError("Ask me a question first.")
    system = PERSONA + "\n" + KNOWLEDGE
    if page_state:
        system += "\n# The user's current page state (JSON)\n" + json.dumps(page_state)[:6000]
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                      "parts": [{"text": str(m["text"])[:4000]}]} for m in msgs],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1200},
    }
    models = [os.environ["GEMINI_MODEL"]] if os.environ.get("GEMINI_MODEL") else FALLBACK_MODELS
    last = None
    for model in models:
        req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code in (400, 401, 403) and "API key" in detail:
                raise AssistantError("Gemini rejected the API key. Check GEMINI_API_KEY in your .env file.")
            if e.code == 429:
                raise AssistantError("Mr. Bean is out of free Gemini requests for the moment. Try again in a minute.")
            last = f"{model}: HTTP {e.code} {detail}"
            continue  # model not available for this key: try the next one
        except Exception as e:
            raise AssistantError(f"Could not reach Gemini ({e}). Check the internet connection.")
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if text:
            return text.replace("—", ", ")
        last = f"{model}: empty answer"
    raise AssistantError(f"Gemini did not answer ({last}).")
