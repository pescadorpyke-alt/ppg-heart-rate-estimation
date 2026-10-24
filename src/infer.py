"""Inference for the API: the committed edge model, served with ONNX Runtime.

The API serves `models/ppg_hr.onnx`, the deployable `ds_32hz_fixed_causal` model
(FP32; the INT8 TFLite for devices has the same weights). It runs exactly the file
in git, needs no PyTorch, and the file's SHA-256 is reported by /health.

Input: c seconds of context + one 8 s window of raw signal (D23). The whole
segment is filtered causally and the last 8 s are the model input, matching what
the model was trained on (D21).

Environment variables:
    MODEL_LOCAL_PATH   path of the ONNX model (default: models/ppg_hr.onnx)
    MODEL_S3_BUCKET + MODEL_S3_KEY   if set and the local file is missing,
                       download the ONNX model from S3 (AWS Lambda deploy)
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache

import numpy as np

from . import config as C
from .preprocessing import preprocess_with_context

DEFAULT_MODEL = C.PROJECT_ROOT / "models" / "ppg_hr.onnx"
SPEC_PATH = C.PROJECT_ROOT / "models" / "ppg_hr.json"


def _resolve_model() -> str:
    local = os.environ.get("MODEL_LOCAL_PATH", str(DEFAULT_MODEL))
    bucket, key = os.environ.get("MODEL_S3_BUCKET"), os.environ.get("MODEL_S3_KEY")
    if bucket and key and not os.path.exists(local):
        import boto3  # imported lazily so local use needs no AWS credentials
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        print(f"Downloading model from s3://{bucket}/{key} -> {local}")
        boto3.client("s3").download_file(bucket, key, local)
    if not os.path.exists(local):
        raise FileNotFoundError(f"Model not found at {local}. Expected models/ppg_hr.onnx "
                                "(committed) or MODEL_S3_BUCKET/MODEL_S3_KEY.")
    return local


@lru_cache(maxsize=1)
def load_model():
    """Load the ONNX model once (cached). Returns (session, metadata)."""
    import onnxruntime as ort

    path = _resolve_model()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    spec = json.loads(SPEC_PATH.read_text()) if SPEC_PATH.exists() else {}
    meta = {
        "name": spec.get("model", "ds_32hz_fixed_causal"),
        "file": os.path.basename(path),
        "sha256": sha256,
        "input_shape": sess.get_inputs()[0].shape,
        "loso_mae_bpm": C.API_MODEL_LOSO_MAE_BPM,
        "min_context_s": C.API_MIN_CONTEXT_S,
        "max_context_s": C.API_MAX_CONTEXT_S,
    }
    return sess, meta


def predict_hr(ppg_raw: np.ndarray, acc_raw: np.ndarray) -> float:
    """Heart rate (bpm) for one raw segment: ppg 64 x (8 + c), acc 32 x (8 + c) x 3."""
    sess, _ = load_model()
    x = preprocess_with_context(ppg_raw, acc_raw, fs_out=32, acc_norm="fixed")   # (4, 256)
    x = x[None, :, None, :].astype(np.float32)                                  # (1, 4, 1, 256)
    return float(sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)[0])
