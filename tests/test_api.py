"""API: serves the committed models/ppg_hr.onnx with context + window input (D23)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import config as C

ROOT = C.PROJECT_ROOT


@pytest.fixture(scope="module")
def client():
    from src import infer
    infer.load_model.cache_clear()
    from api.main import app
    with TestClient(app) as c:      # runs the lifespan startup (loads the model)
        yield c


def payload(rng, context_s=C.API_MIN_CONTEXT_S, extra_ppg=0, acc_rows=None, acc_cols=3):
    seconds = C.WINDOW_S + context_s
    n_acc = acc_rows if acc_rows is not None else seconds * C.FS_ACC
    t = np.arange(seconds * C.FS_PPG + extra_ppg) / C.FS_PPG
    ppg = 50 * np.sin(2 * np.pi * 1.25 * t) + rng.normal(0, 2, t.size)   # 75 bpm pulse
    acc = rng.normal(0, 0.02, (n_acc, acc_cols)) + ([0, 0, 1.0] if acc_cols == 3 else 0)
    return {"ppg": ppg.tolist(), "acc": acc.tolist()}


def test_health_reports_the_committed_model_file(client):
    m = client.get("/health").json()["model"]
    spec = json.loads((ROOT / "models" / "ppg_hr.json").read_text())
    assert m["name"] == "ds_32hz_fixed_causal"
    assert m["sha256"] == spec["files"]["ppg_hr.onnx"]["sha256"]
    assert (m["min_context_s"], m["max_context_s"]) == (C.API_MIN_CONTEXT_S, C.API_MAX_CONTEXT_S)


@pytest.mark.parametrize("context_s", [C.API_MIN_CONTEXT_S, 8, C.API_MAX_CONTEXT_S])
def test_predict_accepts_any_allowed_context(client, rng, context_s):
    r = client.post("/predict", json=payload(rng, context_s))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["context_s"] == context_s
    assert 30 < body["heart_rate_bpm"] < 240
    assert body["model"] == "ds_32hz_fixed_causal"


def test_example_payload_is_valid(client):
    body = json.loads((ROOT / "api" / "example_payload.json").read_text())
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["context_s"] == C.API_MIN_CONTEXT_S


@pytest.mark.parametrize("kwargs", [
    {"context_s": 0},                    # no context: rejected, not silently worse (D23)
    {"context_s": 1},                    # below the minimum
    {"context_s": C.API_MAX_CONTEXT_S + 1},
    {"extra_ppg": 1},                    # not a whole number of seconds
    {"acc_rows": 300},                   # acc does not cover the same duration as ppg
    {"acc_cols": 2},
])
def test_bad_shapes_are_rejected_with_422(client, rng, kwargs):
    assert client.post("/predict", json=payload(rng, **kwargs)).status_code == 422


@pytest.mark.parametrize("field,bad", [("ppg", float("nan")), ("ppg", float("inf")),
                                       ("acc", float("nan")), ("acc", float("-inf"))])
def test_non_finite_values_are_rejected_with_422(client, rng, field, bad):
    doc = payload(rng)
    if field == "ppg":
        doc["ppg"][100] = bad
    else:
        doc["acc"][10][2] = bad
    # json.dumps writes NaN / Infinity tokens, which Python's JSON parser accepts,
    # so the request reaches validation (regression test for a crash on NaN).
    r = client.post("/predict", content=json.dumps(doc), headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "finite" in r.text


def test_missing_field_is_rejected(client, rng):
    assert client.post("/predict", json={"ppg": payload(rng)["ppg"]}).status_code == 422


def test_api_prediction_equals_running_the_onnx_file_directly(client, rng):
    import onnxruntime as ort

    from src.preprocessing import preprocess_with_context
    doc = payload(rng, context_s=8)
    api_hr = client.post("/predict", json=doc).json()["heart_rate_bpm"]
    x = preprocess_with_context(np.array(doc["ppg"]), np.array(doc["acc"]))
    sess = ort.InferenceSession(str(ROOT / "models" / "ppg_hr.onnx"),
                                providers=["CPUExecutionProvider"])
    direct = float(sess.run(None, {"x": x[None, :, None, :].astype(np.float32)})[0][0])
    assert api_hr == pytest.approx(direct, abs=0.05)


def test_api_constants_match_the_measured_results():
    """The API's minimum context and advertised MAE come from result files (D23)."""
    ctx = json.loads((ROOT / "results" / "context_eval_ds_32hz_fixed_causal_summary.json").read_text())
    assert C.API_MIN_CONTEXT_S == ctx["chosen_min_context_s"]
    stream = json.loads((ROOT / "results" / "stream_eval_ds_32hz_fixed_causal_summary.json").read_text())
    assert C.API_MODEL_LOSO_MAE_BPM == round(stream["streaming_fp32"]["mae_mean_over_subjects"], 2)
