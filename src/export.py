"""Phase 3: export LOSO fold models PyTorch -> ONNX -> TFLite (FP32, FP16, INT8).

Runs in the export environment (TensorFlow + onnx2tf), see DECISIONS.md D13:
    D:\\ppg-dalia-hr\\.venv-export\\Scripts\\python.exe -m src.export --model ds_32hz_fixed

For each fold (test subject s):
  1. load the fold's PyTorch checkpoint (trained without s, Phase 2)
  2. export ONNX, convert with onnx2tf to a TF SavedModel
     (onnx2tf turns the channels-first input (1, 4, 256) into channels-last
      (1, 256, 4), the layout TFLite kernels expect)
  3. convert the SavedModel to four TFLite files:
       fp32        no quantization
       fp16        weights stored as float16 (dequantized at load on CPU)
       int8        weights + activations int8, float32 input/output
       int8_full   everything int8, including input and output (MCU style)
     INT8 calibration: 500 windows from the fold's TRAINING subjects only,
     spread evenly over heart rate by default (--calib, see D16).
  4. run every format on the held-out subject and on a fixed check batch,
     recording test MAE and the max difference from PyTorch.

Outputs:
    <HEAVY_ROOT>/export/<model>/<calib>_<layout>/S{s}/{model.onnx, *.tflite}
    results/quant_<model>_<calib>_<layout>.csv          per fold and format: test MAE, size, max |diff|
    results/quant_<model>_<calib>_<layout>_summary.json mean +/- std over folds per format
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch

from . import config as C
from .loso import INPUTS, MODELS, val_subject_for
from .model import to_2d
from .preprocessing import x_path

N_CALIB = 500


def to_channels_last(x: np.ndarray, layout: str) -> np.ndarray:
    """(n, C, T) -> TFLite input layout: (n, T, C) for 1d, (n, 1, T, C) for 2d."""
    return x.transpose(0, 2, 1) if layout == "1d" else x[:, :, None, :].transpose(0, 2, 3, 1)


def calib_indices(train_idx, y, mode: str, rng) -> np.ndarray:
    """Pick N_CALIB calibration windows from the fold's TRAINING windows.

    random:        uniform random sample (first version).
    hr_stratified: equal share per 10 bpm heart-rate bin across the training
                   range, so rare high-HR windows are always included. INT8
                   sets its output range from the calibration predictions, so
                   a random sample left a ceiling of 145-174 bpm (D16).
    """
    if mode == "random":
        return np.sort(rng.choice(train_idx, N_CALIB, replace=False))
    hr = y[train_idx]
    edges = np.arange(np.floor(hr.min() / 10) * 10, hr.max() + 10, 10)
    bins = [train_idx[(hr >= lo) & (hr < lo + 10)] for lo in edges[:-1]]
    bins = [b for b in bins if len(b)]
    quota = N_CALIB // len(bins)
    picked = np.concatenate([rng.choice(b, min(quota, len(b)), replace=False) for b in bins])
    rest = np.setdiff1d(train_idx, picked)
    picked = np.concatenate([picked, rng.choice(rest, N_CALIB - len(picked), replace=False)])
    return np.sort(picked)
FORMATS = ["fp32", "fp16", "int8", "int8_full"]


def load_fold_model(name: str, s: int) -> torch.nn.Module:
    ckpt = torch.load(C.CHECKPOINT_DIR / f"loso_{name}_S{s}.pt", map_location="cpu",
                      weights_only=True)
    assert ckpt["model"] == name and ckpt["test_subject"] == s
    model = MODELS[name]()
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


def to_onnx(model: torch.nn.Module, shape: tuple[int, ...], path: Path) -> None:
    dummy = torch.zeros(1, *shape)
    torch.onnx.export(model, (dummy,), str(path), input_names=["x"],
                      output_names=["hr"], opset_version=17, dynamo=False)


def to_saved_model(onnx_path: Path, out_dir: Path) -> Path:
    import onnx2tf

    # tf_converter backend: onnx2tf builds a TF graph and writes a SavedModel, and
    # TensorFlow's own TFLite converter does the quantization below (the newer
    # default backend, flatbuffer_direct, writes .tflite directly and skips TF).
    onnx2tf.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(out_dir),
                    non_verbose=True, tflite_backend="tf_converter",
                    output_signaturedefs=True)
    if not (out_dir / "saved_model.pb").exists():
        raise RuntimeError(f"onnx2tf wrote no SavedModel in {out_dir}")
    return out_dir


def to_tflite(saved_model: Path, fmt: str, calib_nwc: np.ndarray) -> bytes:
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    if fmt == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif fmt in ("int8", "int8_full"):
        def rep():
            for x in calib_nwc:
                yield [x[None].astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if fmt == "int8_full":
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
    return conv.convert()


def make_interpreter(model_bytes: bytes, threads: int = 1):
    """TFLite interpreter. Uses the one bundled with TensorFlow 2.21, where the
    XNNPACK delegate loads; the ai_edge_litert 2.1.2 Windows wheel never applied
    XNNPACK and ran INT8 ~160x slower in our measurements (D18)."""
    import warnings

    import tensorflow as tf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # tf.lite.Interpreter deprecation notice
        return tf.lite.Interpreter(model_content=model_bytes, num_threads=threads)


def run_tflite(model_bytes: bytes, x_nwc: np.ndarray) -> np.ndarray:
    """Predict HR for (n, T, 4) float inputs, one window at a time (batch 1, as on
    a device). For int8 input/output, quantizes the input and dequantizes the
    output with the scales stored in the model."""
    it = make_interpreter(model_bytes)
    it.allocate_tensors()
    inp_d, out_d = it.get_input_details()[0], it.get_output_details()[0]
    preds = np.empty(len(x_nwc), dtype=np.float32)
    for i in range(len(x_nwc)):
        xb = x_nwc[i : i + 1]
        if inp_d["dtype"] == np.int8:
            scale, zero = inp_d["quantization"]
            xb = np.clip(np.round(xb / scale + zero), -128, 127)
        it.set_tensor(inp_d["index"], xb.astype(inp_d["dtype"]))
        it.invoke()
        yb = it.get_tensor(out_d["index"]).reshape(-1)[0]
        if out_d["dtype"] == np.int8:
            scale, zero = out_d["quantization"]
            yb = (float(yb) - zero) * scale
        preds[i] = yb
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ds_32hz_fixed", choices=sorted(MODELS))
    ap.add_argument("--subjects", default="", help="comma list (default: all 15)")
    ap.add_argument("--calib", default="hr_stratified", choices=["random", "hr_stratified"])
    ap.add_argument("--layout", default="2d", choices=["1d", "2d"],
                    help="2d: export the numerically identical Conv2d copy (to_2d), "
                         "so depthwise convs become DEPTHWISE_CONV_2D (D17)")
    args = ap.parse_args()
    subjects = ([int(s) for s in args.subjects.split(",")] if args.subjects
                else C.ALL_SUBJECTS)
    torch.set_num_threads(1)

    inp = INPUTS[args.model]
    X = np.load(x_path(inp["fs"], inp["acc_norm"]), mmap_mode="r")
    meta = np.load(C.PROCESSED_DIR / "all_meta.npz")
    y, subj = meta["y"], meta["subject"]
    rng = np.random.default_rng(C.SEED)

    rows = []
    for s in subjects:
        t0 = time.perf_counter()
        out_dir = C.HEAVY_ROOT / "export" / args.model / f"{args.calib}_{args.layout}" / f"S{s}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        model = load_fold_model(args.model, s)
        if args.layout == "2d":
            model = to_2d(model)
        v = val_subject_for(s)
        train_idx = np.flatnonzero((subj != s) & (subj != v))
        calib = np.asarray(X[calib_indices(train_idx, y, args.calib, rng)])
        test_idx = np.flatnonzero(subj == s)
        x_test = np.asarray(X[test_idx])

        with torch.no_grad():
            pred_torch = model(torch.from_numpy(x_test)).numpy()
        mae_torch = float(np.abs(pred_torch - y[test_idx]).mean())

        onnx_path = out_dir / "model.onnx"
        c, T = inp["shape"]
        onnx_shape = (c, 1, T) if args.layout == "2d" else (c, T)
        to_onnx(model, onnx_shape, onnx_path)
        x_test_onnx = x_test[:, :, None, :] if args.layout == "2d" else x_test
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        pred_onnx = np.array([sess.run(None, {"x": x_test_onnx[i:i + 1]})[0].reshape(-1)[0]
                              for i in range(len(x_test))], dtype=np.float32)
        rows.append({"test_subject": s, "format": "pytorch_fp32", "test_mae": mae_torch,
                     "bytes": None, "max_abs_diff_vs_torch": 0.0})
        rows.append({"test_subject": s, "format": "onnx_fp32",
                     "test_mae": float(np.abs(pred_onnx - y[test_idx]).mean()),
                     "bytes": onnx_path.stat().st_size,
                     "max_abs_diff_vs_torch": float(np.abs(pred_onnx - pred_torch).max())})

        saved = to_saved_model(onnx_path, out_dir / "saved_model")
        # TFLite is channels-last: (n, T, 4) for 1d, (n, 1, T, 4) for 2d.
        x_test_nwc = to_channels_last(x_test, args.layout)
        calib_nwc = to_channels_last(calib, args.layout)
        for fmt in FORMATS:
            blob = to_tflite(saved, fmt, calib_nwc)
            (out_dir / f"model_{fmt}.tflite").write_bytes(blob)
            pred = run_tflite(blob, x_test_nwc)
            rows.append({"test_subject": s, "format": f"tflite_{fmt}",
                         "test_mae": float(np.abs(pred - y[test_idx]).mean()),
                         "bytes": len(blob),
                         "max_abs_diff_vs_torch": float(np.abs(pred - pred_torch).max())})
        shutil.rmtree(saved, ignore_errors=True)
        fold = [r for r in rows if r["test_subject"] == s]
        print(f"S{s:<2} " + " | ".join(f"{r['format']} {r['test_mae']:.3f}" for r in fold)
              + f"  ({time.perf_counter() - t0:.0f} s)", flush=True)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{args.calib}_{args.layout}"
    if len(subjects) != len(C.ALL_SUBJECTS):
        tag += "_partial"
    with open(C.RESULTS_DIR / f"quant_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    summary = {}
    for fmt in sorted({r["format"] for r in rows}, key=[r["format"] for r in rows].index):
        rs = [r for r in rows if r["format"] == fmt]
        m = np.array([r["test_mae"] for r in rs])
        summary[fmt] = {
            "mae_mean_over_subjects": float(m.mean()), "mae_std_over_subjects": float(m.std()),
            "bytes": rs[0]["bytes"],
            "max_abs_diff_vs_torch": float(max(r["max_abs_diff_vs_torch"] for r in rs)),
        }
    (C.RESULTS_DIR / f"quant_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    for fmt, d in summary.items():
        size = f"{d['bytes'] / 1024:7.1f} KB" if d["bytes"] else "      -   "
        print(f"{fmt:<15} MAE {d['mae_mean_over_subjects']:.3f} ± {d['mae_std_over_subjects']:.3f}"
              f" | {size} | max |diff| vs torch {d['max_abs_diff_vs_torch']:.4f} bpm")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
