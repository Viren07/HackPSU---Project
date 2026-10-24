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
from km.stats import forest_plot, indirect, indirect_plot, pool
from pipeline import run

ROOT = Path(__file__).parent
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


@app.post("/api/km/indirect")
def km_indirect(payload: dict = Body(...)):
    """Compare two new treatments that were each tested against the same old one."""
    try:
        a, b = payload["a"], payload["b"]
        res = indirect(a, b)
        png = indirect_plot(a, b, res, a["treatment"], b["treatment"], payload.get("common") or a.get("control", "control"))
        return {"ok": True, "result": res, "plot_png": base64.b64encode(png).decode()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)
