"""Phase 3 — FastAPI inference service.

POST /predict  -> HR (bpm) from one raw 8 s window (PPG @64Hz + ACC @32Hz).
GET  /health   -> liveness + loaded-model metadata.

Run locally:
    uvicorn api.main:app --reload --port 8000
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
from pydantic import BaseModel, Field, field_validator

# Allow `uvicorn api.main:app` from the project root to import the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.infer import load_model, predict_hr  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup (fail fast if the checkpoint is missing or bad).
    load_model()
    yield


app = FastAPI(
    title="PPG Heart-Rate Estimation API",
    description="Wearable PPG + accelerometer -> heart rate (bpm).",
    version="1.0.0",
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
    """One raw 8 s window straight from the wrist sensor (unfiltered)."""
    ppg: list[float] = Field(
        ..., description=f"Raw BVP/PPG @ {C.FS_PPG} Hz, exactly {C.PPG_WINDOW_LEN} samples."
    )
    acc: list[list[float]] = Field(
        ...,
        description=f"Raw 3-axis ACC @ {C.FS_ACC} Hz, shape ({C.ACC_WINDOW_LEN}, 3) "
                    "as a list of [x, y, z] rows.",
    )

    @field_validator("ppg")
    @classmethod
    def _check_ppg(cls, v):
        if len(v) != C.PPG_WINDOW_LEN:
            raise ValueError(f"ppg must have exactly {C.PPG_WINDOW_LEN} samples, got {len(v)}")
        # JSON parsers accept NaN/Infinity; a single one would poison the filter.
        if not all(math.isfinite(x) for x in v):
            raise ValueError("ppg must contain only finite numbers (no NaN or Infinity)")
        return v

    @field_validator("acc")
    @classmethod
    def _check_acc(cls, v):
        if len(v) != C.ACC_WINDOW_LEN:
            raise ValueError(f"acc must have exactly {C.ACC_WINDOW_LEN} rows, got {len(v)}")
        if any(len(row) != 3 for row in v):
            raise ValueError("each acc row must have exactly 3 values [x, y, z]")
        if not all(math.isfinite(x) for row in v for x in row):
            raise ValueError("acc must contain only finite numbers (no NaN or Infinity)")
        return v


class PredictResponse(BaseModel):
    heart_rate_bpm: float
    model_val_mae_bpm: float | None = None


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><title>PPG Heart-Rate Estimation API</title>
    <style>body{font-family:system-ui,sans-serif;max-width:680px;margin:40px auto;
    padding:0 16px;line-height:1.5;color:#222}code,pre{background:#f4f4f5;
    border-radius:6px}pre{padding:12px;overflow:auto}a{color:#c026d3}</style></head>
    <body>
    <h1>🫀 PPG Heart-Rate Estimation API</h1>
    <p>Estimates heart rate (bpm) from a wrist <b>PPG</b> signal + 3-axis
    <b>accelerometer</b>, using a small 1D CNN trained on the public
    <a href="https://archive.ics.uci.edu/dataset/495/ppg+dalia">PPG-DaLiA</a> dataset.
    Wearable sensor data → actionable health metric.</p>
    <ul>
      <li><b>POST <code>/predict</code></b> — one 8 s window (PPG 512@64Hz + ACC 256×3@32Hz) → HR in bpm</li>
      <li><b>GET <code>/health</code></b> — status + model metadata</li>
      <li><b>GET <a href="/docs"><code>/docs</code></a></b> — interactive API explorer</li>
    </ul>
    <pre>curl -X POST .../predict -H "Content-Type: application/json" -d @example_payload.json</pre>
    <p>Validation MAE ≈ 12 bpm (held-out subjects); naive baseline ≈ 24 bpm.</p>
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
    return PredictResponse(heart_rate_bpm=round(hr, 1), model_val_mae_bpm=meta.get("val_mae"))
