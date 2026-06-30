"""Phase 3 — inference: load checkpoint, preprocess one raw window, predict HR.

The checkpoint can come from a local file (default) or be downloaded from S3 at
startup (Phase 4 deploy) — controlled by env vars:
    MODEL_LOCAL_PATH  (default: config.CHECKPOINT_DIR/best.pt)
    MODEL_S3_BUCKET + MODEL_S3_KEY  (if set, downloaded to MODEL_LOCAL_PATH)
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

from . import config as C
from .model import PPGHeartRateCNN
from .preprocessing import preprocess_window


def _resolve_checkpoint() -> str:
    local = os.environ.get("MODEL_LOCAL_PATH", str(C.CHECKPOINT_DIR / "best.pt"))
    bucket = os.environ.get("MODEL_S3_BUCKET")
    key = os.environ.get("MODEL_S3_KEY")
    if bucket and key and not os.path.exists(local):
        import boto3  # imported lazily so local use needs no AWS creds
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        print(f"Downloading model from s3://{bucket}/{key} -> {local}")
        boto3.client("s3").download_file(bucket, key, local)
    if not os.path.exists(local):
        raise FileNotFoundError(
            f"Checkpoint not found at {local}. Train first (src.train) or set "
            "MODEL_S3_BUCKET/MODEL_S3_KEY."
        )
    return local


@lru_cache(maxsize=1)
def load_model() -> tuple[PPGHeartRateCNN, dict]:
    """Load model once (cached)."""
    path = _resolve_checkpoint()
    # Checkpoint holds only tensors + simple scalars => weights_only=True is safe
    # and blocks arbitrary-code execution during unpickling.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = PPGHeartRateCNN(in_channels=ckpt.get("in_channels", 4))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {
        "val_mae": ckpt.get("val_mae"),
        "epoch": ckpt.get("epoch"),
        "window_len": ckpt.get("window_len", C.PPG_WINDOW_LEN),
    }
    return model, meta


@torch.no_grad()
def predict_hr(ppg_raw: np.ndarray, acc_raw: np.ndarray) -> float:
    """Predict HR (bpm) from one raw window (PPG 512@64Hz, ACC 256x3@32Hz)."""
    model, _ = load_model()
    x = preprocess_window(ppg_raw, acc_raw)            # (4, 512)
    xt = torch.from_numpy(x).unsqueeze(0)              # (1, 4, 512)
    return float(model(xt).item())
