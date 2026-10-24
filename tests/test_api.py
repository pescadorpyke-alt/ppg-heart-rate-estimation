"""API validation, served with a randomly initialised checkpoint (no trained model needed)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.model import PPGHeartRateCNN


@pytest.fixture
def client(tmp_path, monkeypatch):
    torch.manual_seed(0)
    ckpt = tmp_path / "best.pt"
    torch.save({"model_state": PPGHeartRateCNN().state_dict(), "val_mae": 12.09,
                "epoch": 1, "in_channels": 4, "window_len": 512}, ckpt)
    monkeypatch.setenv("MODEL_LOCAL_PATH", str(ckpt))
    from src import infer
    infer.load_model.cache_clear()
    from api.main import app
    with TestClient(app) as c:      # runs the lifespan startup (loads the model)
        yield c
    infer.load_model.cache_clear()


def payload(rng, n_ppg=512, n_acc=256, acc_cols=3):
    return {"ppg": rng.normal(size=n_ppg).tolist(),
            "acc": rng.normal(size=(n_acc, acc_cols)).tolist()}


def test_health_reports_model_metadata(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model"]["window_len"] == 512


def test_predict_returns_a_finite_heart_rate(client, rng):
    r = client.post("/predict", json=payload(rng))
    assert r.status_code == 200
    assert np.isfinite(r.json()["heart_rate_bpm"])


@pytest.mark.parametrize("kwargs", [{"n_ppg": 511}, {"n_ppg": 513}, {"n_acc": 255},
                                    {"acc_cols": 2}])
def test_wrong_shapes_are_rejected_with_422(client, rng, kwargs):
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
    # so the request reaches validation (a regression test for a 500 on NaN).
    r = client.post("/predict", content=json.dumps(doc), headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "finite" in r.text


def test_missing_field_is_rejected(client, rng):
    assert client.post("/predict", json={"ppg": payload(rng)["ppg"]}).status_code == 422
