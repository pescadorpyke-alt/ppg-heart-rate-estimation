"""FastAPI inference service for the edge heart-rate model.

POST /predict  -> heart rate (bpm) from c seconds of context + one 8 s window of
                  raw wrist PPG (64 Hz) and accelerometer (32 Hz, in g)
GET  /health   -> liveness + model metadata (file, SHA-256, context limits)

Serves models/ppg_hr.onnx with ONNX Runtime (src/infer.py). The model expects
causally filtered input, so requests carry c >= 2 s of signal before the window
for the filter to settle (DECISIONS.md D23).

Run locally:
    uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import math
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# Allow `uvicorn api.main:app` from the project root to import the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src.infer import load_model, predict_hr

MIN_PPG = C.FS_PPG * (C.WINDOW_S + C.API_MIN_CONTEXT_S)
MAX_PPG = C.FS_PPG * (C.WINDOW_S + C.API_MAX_CONTEXT_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup (fail fast if the model file is missing or bad).
    load_model()
    yield


app = FastAPI(
    title="PPG Heart-Rate Estimation API",
    description="Wearable PPG + accelerometer -> heart rate (bpm), edge model served with ONNX Runtime.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    # FastAPI's default 422 body echoes the offending input back. A NaN/Infinity
    # in the request cannot be written as JSON, so the default handler crashed;
    # return where and why validation failed, without the raw values.
    errors = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


class PredictRequest(BaseModel):
    """Raw wrist signal: c seconds of context followed by the 8 s window to score."""
    ppg: list[float] = Field(
        ..., description=(f"Raw BVP/PPG @ {C.FS_PPG} Hz: {C.FS_PPG} x (8 + c) samples, "
                          f"c = {C.API_MIN_CONTEXT_S}..{C.API_MAX_CONTEXT_S} whole seconds of "
                          "context before the window."),
    )
    acc: list[list[float]] = Field(
        ..., description=(f"Raw 3-axis ACC @ {C.FS_ACC} Hz in g, rows [x, y, z]: "
                          f"{C.FS_ACC} x (8 + c) rows, covering the same seconds as ppg."),
    )

    @field_validator("ppg")
    @classmethod
    def _check_ppg(cls, v):
        if not MIN_PPG <= len(v) <= MAX_PPG or len(v) % C.FS_PPG:
            raise ValueError(
                f"ppg must be {C.FS_PPG} x (8 + c) samples with c a whole number of seconds in "
                f"[{C.API_MIN_CONTEXT_S}, {C.API_MAX_CONTEXT_S}] ({MIN_PPG}..{MAX_PPG}), got {len(v)}")
        # JSON parsers accept NaN/Infinity; a single one would poison the filter.
        if not all(math.isfinite(x) for x in v):
            raise ValueError("ppg must contain only finite numbers (no NaN or Infinity)")
        return v

    @field_validator("acc")
    @classmethod
    def _check_acc(cls, v):
        if any(len(row) != 3 for row in v):
            raise ValueError("each acc row must have exactly 3 values [x, y, z]")
        if not all(math.isfinite(x) for row in v for x in row):
            raise ValueError("acc must contain only finite numbers (no NaN or Infinity)")
        return v

    @model_validator(mode="after")
    def _same_duration(self):
        if len(self.acc) * C.FS_PPG != len(self.ppg) * C.FS_ACC:
            raise ValueError(
                f"acc must cover the same duration as ppg: {len(self.ppg)} ppg samples need "
                f"{len(self.ppg) * C.FS_ACC // C.FS_PPG} acc rows, got {len(self.acc)}")
        return self

    @property
    def context_s(self) -> int:
        return len(self.ppg) // C.FS_PPG - C.WINDOW_S


class PredictResponse(BaseModel):
    heart_rate_bpm: float
    context_s: int
    model: str
    model_loso_mae_bpm: float


@app.get("/", response_class=HTMLResponse)
def root():
    return f"""
    <html><head><title>PPG Heart-Rate Estimation API</title>
    <style>body{{font-family:system-ui,sans-serif;max-width:680px;margin:40px auto;
    padding:0 16px;line-height:1.5;color:#222}}code,pre{{background:#f4f4f5;
    border-radius:6px}}pre{{padding:12px;overflow:auto}}a{{color:#c026d3}}</style></head>
    <body>
    <h1>PPG Heart-Rate Estimation API</h1>
    <p>Estimates heart rate (bpm) from a wrist <b>PPG</b> signal + 3-axis
    <b>accelerometer</b> with a 47 K-parameter edge model (70 KB as INT8 TFLite),
    trained on the public
    <a href="https://archive.ics.uci.edu/dataset/495/ppg+dalia">PPG-DaLiA</a> dataset.
    Leave-one-subject-out MAE: about {C.API_MODEL_LOSO_MAE_BPM} bpm. Not a medical device.</p>
    <ul>
      <li><b>POST <code>/predict</code></b>: {C.API_MIN_CONTEXT_S}-{C.API_MAX_CONTEXT_S} s of
      context + one 8 s window (PPG @ 64 Hz, ACC @ 32 Hz in g) &rarr; heart rate in bpm</li>
      <li><b>GET <code>/health</code></b>: status + model metadata</li>
      <li><b>GET <a href="/docs"><code>/docs</code></a></b>: interactive API explorer</li>
    </ul>
    <pre>curl -X POST .../predict -H "Content-Type: application/json" -d @example_payload.json</pre>
    </body></html>
    """


@app.get("/health")
def health():
    try:
        _, meta = load_model()
        return {"status": "ok", "model": meta}
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"model not loaded: {e}") from e


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        _, meta = load_model()
        hr = predict_hr(np.asarray(req.ppg), np.asarray(req.acc))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e
    return PredictResponse(heart_rate_bpm=round(hr, 1), context_s=req.context_s,
                           model=meta["name"], model_loso_mae_bpm=meta["loso_mae_bpm"])
