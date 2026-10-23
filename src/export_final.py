"""Phase 4: export the deployable model to models/ (committed, small files).

Same path as the per-fold export (D13, D16, D17): 2-D copy -> ONNX -> onnx2tf
SavedModel -> TFLite FP32 and INT8; INT8 calibrated on 500 windows spread over
heart rate, drawn from all 15 subjects (all of them are training data for this
model). Writes models/ppg_hr_{fp32,int8,int8_full}.tflite, models/ppg_hr.onnx
and models/ppg_hr.json (input spec + quantization parameters).

The INT8-vs-FP32 agreement printed here is a sanity check on training data, not
an accuracy result (D20).

Export environment:
    D:\\ppg-dalia-hr\\.venv-export\\Scripts\\python.exe -m src.export_final
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch

from . import config as C
from .export import (N_CALIB, calib_indices, make_interpreter, run_tflite, to_channels_last,
                     to_onnx, to_saved_model, to_tflite)
from .loso import INPUTS, MODELS
from .model import to_2d
from .preprocessing import x_path

MODEL = "ds_32hz_fixed_causal"


def main():
    ckpt = torch.load(C.CHECKPOINT_DIR / f"final_{MODEL}.pt", map_location="cpu", weights_only=True)
    model = MODELS[MODEL]()
    model.load_state_dict(ckpt["model_state"])
    model2d = to_2d(model.eval())
    inp = INPUTS[MODEL]
    X = np.load(x_path(inp["fs"], inp["acc_norm"], inp["filt"]), mmap_mode="r")
    y = np.load(C.PROCESSED_DIR / "all_meta.npz")["y"]

    out = C.PROJECT_ROOT / "models"
    out.mkdir(exist_ok=True)
    work = C.HEAVY_ROOT / "export" / "final"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    c, T = inp["shape"]
    to_onnx(model2d, (c, 1, T), out / "ppg_hr.onnx")
    saved = to_saved_model(out / "ppg_hr.onnx", work / "saved_model")
    rng = np.random.default_rng(C.SEED)
    calib = np.asarray(X[calib_indices(np.arange(len(y)), y, "hr_stratified", rng)])
    blobs = {}
    for fmt in ("fp32", "int8", "int8_full"):
        blobs[fmt] = to_tflite(saved, fmt, to_channels_last(calib, "2d"))
        (out / f"ppg_hr_{fmt}.tflite").write_bytes(blobs[fmt])

    # Sanity check (training data!): exported formats agree with PyTorch.
    idx = np.sort(np.random.default_rng(1).choice(len(y), 3000, replace=False))
    xs = np.asarray(X[idx])
    with torch.no_grad():
        p_torch = model(torch.from_numpy(xs)).numpy()
    agree = {}
    for fmt, blob in blobs.items():
        p = run_tflite(blob, to_channels_last(xs, "2d"))
        agree[fmt] = {"max_abs_diff_vs_torch_bpm": float(np.abs(p - p_torch).max()),
                      "median_abs_diff_vs_torch_bpm": float(np.median(np.abs(p - p_torch)))}

    it = make_interpreter(blobs["int8_full"])
    it.allocate_tensors()
    i8, o8 = it.get_input_details()[0], it.get_output_details()[0]
    spec = {
        "model": MODEL,
        "trained_on": f"PPG-DaLiA S1-S15, all windows, {ckpt['epochs']} epochs (D20)",
        "accuracy_estimate": "LOSO of the same recipe: see results/stream_eval_ds_32hz_fixed_causal_summary.json",
        "input": {
            "tflite_shape_nhwc": [1, 1, T, c],
            "onnx_shape_nchw": [1, c, 1, T],
            "window": "8 s, new window every 2 s",
            "channels": ["PPG (BVP), z-scored per window",
                         "ACC x / 0.1 g", "ACC y / 0.1 g", "ACC z / 0.1 g"],
            "sample_rate_hz": 32,
            "preprocessing": ("causal 4th-order Butterworth band-pass 0.5-4.0 Hz per sensor, "
                              "state kept across windows (src/stream.py); PPG filtered at 64 Hz "
                              "then every 2nd sample kept; ACC filtered at its native 32 Hz; "
                              "ACC in g"),
        },
        "output": "heart rate in bpm, one value per window",
        "int8_full_quantization": {
            "input_scale": float(i8["quantization"][0]), "input_zero_point": int(i8["quantization"][1]),
            "output_scale": float(o8["quantization"][0]), "output_zero_point": int(o8["quantization"][1]),
            "output_max_bpm": float((127 - o8["quantization"][1]) * o8["quantization"][0]),
        },
        "calibration": f"{N_CALIB} windows spread over heart rate (D16), all subjects",
        "files": {f.name: {"bytes": f.stat().st_size,
                           "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                  for f in sorted(out.glob("ppg_hr*")) if f.suffix != ".json"},
        "sanity_check_on_training_data": agree,
    }
    (out / "ppg_hr.json").write_text(json.dumps(spec, indent=2))
    for name, info in spec["files"].items():
        print(f"{name:<24} {info['bytes'] / 1024:7.1f} KB")
    for fmt, a in agree.items():
        print(f"{fmt:<10} vs PyTorch on 3000 training windows: median |diff| "
              f"{a['median_abs_diff_vs_torch_bpm']:.3f}, max {a['max_abs_diff_vs_torch_bpm']:.2f} bpm")
    print(f"INT8 output max: {spec['int8_full_quantization']['output_max_bpm']:.1f} bpm")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
