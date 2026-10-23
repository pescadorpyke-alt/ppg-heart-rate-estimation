"""Phase 4: accuracy of the exported fold models under three preprocessing paths (D19).

For each subject s, the fold-s TFLite models (never trained on s) are run on s
with inputs built three ways:
    offline     training preprocessing: zero-phase filter over the whole
                recording, then windows (= the Phase 3 numbers)
    per_window  what api/main.py does: zero-phase filter of each 8 s window alone
    streaming   causal filter over the continuous stream, 2 s chunks, ring
                buffers (src/stream.py), as a device would run it

Export environment:
    D:\\ppg-dalia-hr\\.venv-export\\Scripts\\python.exe -m src.stream_eval
Outputs results/stream_eval_<model>.csv and _summary.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from . import config as C
from .export import run_tflite, to_channels_last
from .loso import INPUTS
from .preprocessing import load_wrist, preprocess_window, x_path
from .stream import stream_subject

MODEL, VARIANT = "ds_32hz_fixed", "hr_stratified_2d"
PATHS = ["offline", "per_window", "streaming"]


def main():
    inp = INPUTS[MODEL]
    X = np.load(x_path(inp["fs"], inp["acc_norm"]), mmap_mode="r")
    meta = np.load(C.PROCESSED_DIR / "all_meta.npz")
    y, subj, act = meta["y"], meta["subject"], meta["activity"]
    pred = {(p, f): np.full(len(y), np.nan) for p in PATHS for f in ("fp32", "int8")}

    rows = []
    for s in C.ALL_SUBJECTS:
        t0 = time.perf_counter()
        te = np.flatnonzero(subj == s)
        d = load_wrist(s)
        n = len(te)
        assert n == len(d["label"]) and np.allclose(d["label"][:n], y[te])
        xs = {"offline": np.asarray(X[te])}
        xs["per_window"] = np.stack([
            preprocess_window(d["bvp"][128 * k : 128 * k + 512], d["acc"][64 * k : 64 * k + 256],
                              inp["fs"], inp["acc_norm"]) for k in range(n)])
        xs["streaming"] = stream_subject(d["bvp"], d["acc"], n, inp["fs"], inp["acc_norm"])
        folder = C.HEAVY_ROOT / "export" / MODEL / VARIANT / f"S{s}"
        blobs = {f: (folder / f"model_{f}.tflite").read_bytes() for f in ("fp32", "int8")}
        line = []
        for p in PATHS:
            xin = to_channels_last(xs[p], "2d")
            for f, blob in blobs.items():
                pr = run_tflite(blob, xin)
                pred[(p, f)][te] = pr
                mae = float(np.abs(pr - y[te]).mean())
                rows.append({"test_subject": s, "path": p, "format": f, "test_mae": mae})
                line.append(f"{p}/{f} {mae:.2f}")
        print(f"S{s:<2} " + " | ".join(line) + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)

    summary = {}
    for p in PATHS:
        for f in ("fp32", "int8"):
            m = np.array([r["test_mae"] for r in rows if r["path"] == p and r["format"] == f])
            e = np.abs(pred[(p, f)] - y)
            summary[f"{p}_{f}"] = {
                "mae_mean_over_subjects": float(m.mean()), "mae_std_over_subjects": float(m.std()),
                "per_activity_mae": {name: float(e[act == a].mean())
                                     for a, name in C.ACTIVITY_NAMES.items()},
            }
    base = np.array([r["test_mae"] for r in rows if r["path"] == "offline" and r["format"] == "int8"])
    for p in ("per_window", "streaming"):
        m = np.array([r["test_mae"] for r in rows if r["path"] == p and r["format"] == "int8"])
        summary[f"{p}_int8"]["vs_offline_int8"] = {
            "mean_diff": float((m - base).mean()), "worse_on_subjects": int((m > base).sum())}

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_DIR / f"stream_eval_{MODEL}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    (C.RESULTS_DIR / f"stream_eval_{MODEL}_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    for k, v in summary.items():
        extra = ""
        if "vs_offline_int8" in v:
            extra = (f"  | vs offline INT8: {v['vs_offline_int8']['mean_diff']:+.3f}, "
                     f"worse on {v['vs_offline_int8']['worse_on_subjects']}/15")
        print(f"{k:<17} MAE {v['mae_mean_over_subjects']:.3f} ± {v['mae_std_over_subjects']:.3f}{extra}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
