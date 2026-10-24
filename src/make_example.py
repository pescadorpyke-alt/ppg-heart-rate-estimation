"""Dump one raw request to api/example_payload.json for trying the API.

The payload is c seconds of context + one 8 s window of raw wrist signal, exactly
what POST /predict expects (D23), plus the ECG ground truth printed for
comparison. Note: the served model was trained on all 15 subjects (D20), so a
prediction on this window is a demo, not an accuracy measurement.
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from .preprocessing import load_wrist

SUBJECT = 4
WINDOW_K = 1500          # window index (starts at 2 * k seconds)
CONTEXT_S = C.API_MIN_CONTEXT_S


def main():
    d = load_wrist(SUBJECT)
    bvp, acc, labels = d["bvp"], d["acc"], d["label"]
    p0 = WINDOW_K * C.WINDOW_STRIDE_S * C.FS_PPG - CONTEXT_S * C.FS_PPG
    a0 = WINDOW_K * C.WINDOW_STRIDE_S * C.FS_ACC - CONTEXT_S * C.FS_ACC
    n_p = (C.WINDOW_S + CONTEXT_S) * C.FS_PPG
    n_a = (C.WINDOW_S + CONTEXT_S) * C.FS_ACC
    payload = {"ppg": np.round(bvp[p0 : p0 + n_p], 4).tolist(),
               "acc": np.round(acc[a0 : a0 + n_a], 6).tolist()}
    out = C.PROJECT_ROOT / "api" / "example_payload.json"
    out.write_text(json.dumps(payload))
    print(f"Subject S{SUBJECT}, window {WINDOW_K}, {CONTEXT_S} s context")
    print(f"  ppg samples: {len(payload['ppg'])}  acc rows: {len(payload['acc'])}")
    print(f"  GROUND-TRUTH HR (ECG): {float(labels[WINDOW_K]):.1f} bpm")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
