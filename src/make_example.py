"""Dump one RAW validation window to api/example_payload.json for API testing.

Picks a window from a held-out validation subject and writes the unfiltered
PPG + ACC exactly as the API expects, plus prints the ground-truth HR so we can
compare the API's prediction against truth.
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from .preprocessing import load_subject

SUBJECT = 4          # held-out validation subject
WINDOW_K = 1500      # arbitrary window index


def main():
    d = load_subject(SUBJECT)
    bvp = np.asarray(d["signal"]["wrist"]["BVP"], dtype=np.float64).reshape(-1)
    acc = np.asarray(d["signal"]["wrist"]["ACC"], dtype=np.float64)
    labels = np.asarray(d["label"], dtype=np.float64).reshape(-1)

    p0 = WINDOW_K * C.WINDOW_STRIDE_S * C.FS_PPG
    a0 = WINDOW_K * C.WINDOW_STRIDE_S * C.FS_ACC
    ppg_win = bvp[p0 : p0 + C.PPG_WINDOW_LEN]
    acc_win = acc[a0 : a0 + C.ACC_WINDOW_LEN]
    true_hr = float(labels[WINDOW_K])

    payload = {"ppg": ppg_win.tolist(), "acc": acc_win.tolist()}
    out = C.PROJECT_ROOT / "api" / "example_payload.json"
    out.write_text(json.dumps(payload))

    print(f"Subject S{SUBJECT}, window {WINDOW_K}")
    print(f"  ppg samples: {len(payload['ppg'])}  acc rows: {len(payload['acc'])}")
    print(f"  GROUND-TRUTH HR: {true_hr:.1f} bpm")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
