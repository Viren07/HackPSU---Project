"""PriorArt Lens web server.

Run:   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Open:  http://localhost:8000        survival-curve meta-analysis (main demo)
       http://localhost:8000/chart  general chart reader
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from km.pipeline import analyze_trial, read_trial
from km.assistant import AssistantError, ask, load_env
from km.paper import PaperError, find_figure
from km.stats import forest_plot, indirect, indirect_plot, pool
from pipeline import run

ROOT = Path(__file__).parent
load_env(ROOT.parent / ".env")  # GEMINI_API_KEY for Mr. Bean (the file is git-ignored)
app = FastAPI(title="PriorArt Lens")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


async def _load_image(image: UploadFile):
    data = np.frombuffer(await image.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if max(h, w) > 2000:  # phone photos are huge; speed things up
        s = 2000 / max(h, w)
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    elif max(h, w) < 1000:  # small crops (e.g. one panel of a journal figure): enlarge so small digits can be read
        s = min(2.5, 1300 / max(h, w))
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "survival.html")


@app.get("/chart")
def chart_page():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/extract")
async def extract(image: UploadFile = File(...), rectify: bool = Form(True)):
    img = await _load_image(image)
    if img is None:
        return JSONResponse({"ok": False, "error": "Could not read that image."}, status_code=400)
    try:
        return run(img, do_rectify=rectify)
    except Exception as e:  # keep the demo alive no matter what
        return JSONResponse({"ok": False, "error": f"Pipeline error: {e}"}, status_code=500)


@app.post("/api/km/read")
async def km_read(image: UploadFile = File(...), rectify: bool = Form(True)):
    img = await _load_image(image)
    if img is None:
        return JSONResponse({"ok": False, "error": "Could not read that image."}, status_code=400)
    try:
        return read_trial(img, do_rectify=rectify)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)


@app.post("/api/km/analyze")
def km_analyze(payload: dict = Body(...)):
    try:
        return analyze_trial(payload["read"], payload.get("treatment_index"), payload.get("published_hr"))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)


@app.post("/api/km/pool")
def km_pool(payload: dict = Body(...)):
    trials = payload.get("trials", [])
    if len(trials) < 2:
        return JSONResponse({"ok": False, "error": "Add at least two trials to pool."}, status_code=422)
    try:
        pooled = pool(trials)
        png = forest_plot(trials, pooled, payload.get("title") or "Overall survival: treatment vs control")
        return {"ok": True, "pooled": pooled, "forest_png": base64.b64encode(png).decode()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)


@app.post("/api/paper/figure")
def paper_figure(payload: dict = Body(...)):
    """Paper link + 'Figure 2A' -> paper details, the figure image and its caption."""
    try:
        res = find_figure(payload.get("link", ""), payload.get("figure", ""))
    except PaperError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Could not fetch the figure: {e}"}, status_code=500)
    bgr = cv2.imdecode(np.frombuffer(res.pop("image"), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"ok": False, "error": "The figure image could not be opened."}, status_code=422)
    ok, buf = cv2.imencode(".png", bgr)
    res["figure"]["image_png"] = base64.b64encode(buf.tobytes()).decode()
    res["figure"]["size"] = [int(bgr.shape[1]), int(bgr.shape[0])]
    return {"ok": True, **res}


@app.post("/api/km/indirect")
def km_indirect(payload: dict = Body(...)):
    """Compare two new treatments that were each tested against the same old one."""
    try:
        a, b = payload["a"], payload["b"]
        res = indirect(a, b)
        png = indirect_plot(a, b, res, a["treatment"], b["treatment"], payload.get("common") or a.get("control", "control"),
                            payload.get("event") or "death")
        return {"ok": True, "result": res, "plot_png": base64.b64encode(png).decode()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)


@app.post("/api/chat")
def chat(payload: dict = Body(...)):
    """Mr. Bean, the help assistant (Gemini). The API key never leaves the server."""
    try:
        return {"ok": True, "answer": ask(payload.get("messages") or [], payload.get("page_state"))}
    except AssistantError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)
