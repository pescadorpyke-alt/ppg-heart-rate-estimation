"""Phase 3: single-window latency of every exported format, constrained CPU setup.

"Constrained" here means (DECISIONS.md D1, D14): the process is pinned to ONE
CPU core, every runtime uses ONE thread, batch size is 1, and each format gets
100 warm-up runs before 2,000 timed runs. This is still a desktop CPU and is
much faster than a microcontroller, so these numbers are NOT device latency.

TFLite runtimes ("tier"), all via TensorFlow 2.21's bundled interpreter:
    xnnpack      default delegate (XNNPACK): SIMD-optimised kernels
    builtin      TFLite's own optimised kernels, no delegate
    reference    TFLite's plain reference kernels (BUILTIN_REF): no
                 optimisation at all, the closest to a simple embedded build
plus `litert_default`: the ai_edge_litert 2.1.2 package's interpreter, kept
because on this Windows machine it never applied XNNPACK and ran INT8 very
slowly (D18).

Both export layouts are timed: 2d (shipped, depthwise convs become
DEPTHWISE_CONV_2D) and 1d (grouped CONV_2D), see D17.

Runs in the export environment:
    D:\\ppg-dalia-hr\\.venv-export\\Scripts\\python.exe -m src.benchmark --model ds_32hz_fixed
Outputs results/benchmark_<model>.csv and _meta.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch

from . import config as C
from .export import load_fold_model, to_channels_last
from .loso import INPUTS
from .preprocessing import x_path

WARMUP, RUNS = 100, 2000


def timed(fn, x_list) -> np.ndarray:
    """Run fn on each input; return per-call latency in microseconds."""
    for i in range(WARMUP):
        fn(x_list[i % len(x_list)])
    out = np.empty(RUNS)
    for i in range(RUNS):
        x = x_list[i % len(x_list)]
        t = time.perf_counter_ns()
        fn(x)
        out[i] = (time.perf_counter_ns() - t) / 1000
    return out


def tflite_fn(path, tier: str):
    import warnings

    import tensorflow as tf

    if tier == "litert_default":
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=str(path), num_threads=1)
    else:
        R = tf.lite.experimental.OpResolverType
        resolver = {"xnnpack": R.AUTO, "builtin": R.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                    "reference": R.BUILTIN_REF}[tier]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            it = tf.lite.Interpreter(model_path=str(path), num_threads=1,
                                     experimental_op_resolver_type=resolver)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    if inp["dtype"] == np.int8:
        scale, zero = inp["quantization"]
        prep = lambda x: np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)  # noqa: E731
    else:
        prep = lambda x: x  # noqa: E731

    def fn(x_nwc):
        it.set_tensor(inp["index"], prep(x_nwc))
        it.invoke()
        return it.get_tensor(out["index"])
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ds_32hz_fixed")
    ap.add_argument("--fold", type=int, default=1,
                    help="which fold's exported files to time (all folds share "
                         "the same architecture, so size and latency do not "
                         "depend on the fold)")
    ap.add_argument("--calib", default="hr_stratified")
    ap.add_argument("--layouts", default="2d,1d")
    args = ap.parse_args()

    try:
        import psutil
        psutil.Process().cpu_affinity([0])
        pinned = "core 0"
    except Exception as e:   # pragma: no cover
        pinned = f"not pinned ({type(e).__name__})"
    torch.set_num_threads(1)

    inp = INPUTS[args.model]
    X = np.load(x_path(inp["fs"], inp["acc_norm"]), mmap_mode="r")
    idx = np.random.default_rng(C.SEED).choice(len(X), 200, replace=False)
    xs_ncw = [np.asarray(X[i : i + 1], dtype=np.float32) for i in np.sort(idx)]

    model = load_fold_model(args.model, args.fold)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1

    # (format, runtime, layout, file, fn, inputs)
    cases = [("pytorch_fp32", "eager", "1d", None, lambda x: model(torch.from_numpy(x)), xs_ncw)]
    for layout in args.layouts.split(","):
        d = C.HEAVY_ROOT / "export" / args.model / f"{args.calib}_{layout}" / f"S{args.fold}"
        xs_onnx = [x[:, :, None, :].copy() for x in xs_ncw] if layout == "2d" else xs_ncw
        xs_tfl = [to_channels_last(x, layout).copy() for x in xs_ncw]
        sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        cases.append(("onnx_fp32", "onnxruntime", layout, d / "model.onnx",
                      lambda x, s=sess: s.run(None, {"x": x}), xs_onnx))
        for fmt in ["fp32", "fp16", "int8", "int8_full"]:
            p = d / f"model_{fmt}.tflite"
            for tier in ["xnnpack", "builtin", "reference", "litert_default"]:
                cases.append((f"tflite_{fmt}", tier, layout, p, tflite_fn(p, tier), xs_tfl))

    rows = []
    with torch.no_grad():
        for fmt, runtime, layout, path, fn, xs in cases:
            lat = timed(fn, xs)
            r = {"format": fmt, "runtime": runtime, "layout": layout,
                 "bytes": path.stat().st_size if path else None,
                 "median_us": float(np.median(lat)), "p95_us": float(np.percentile(lat, 95)),
                 "p99_us": float(np.percentile(lat, 99)), "mean_us": float(lat.mean())}
            rows.append(r)
            size = f"{r['bytes'] / 1024:7.1f} KB" if r["bytes"] else "      -   "
            print(f"{fmt:<17}{runtime:<15}{layout:<4}{size} | median {r['median_us']:8.1f} us | "
                  f"p95 {r['p95_us']:8.1f} us | p99 {r['p99_us']:8.1f} us", flush=True)

    import onnxruntime
    import tensorflow as tf
    meta = {
        "model": args.model, "fold_files": f"S{args.fold}", "input_shape": inp["shape"],
        "warmup": WARMUP, "runs": RUNS, "threads": 1, "cpu_affinity": pinned,
        "batch": 1, "cpu": platform.processor(), "machine": platform.machine(),
        "os": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
        "tensorflow": tf.__version__,
    }
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_DIR / f"benchmark_{args.model}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    (C.RESULTS_DIR / f"benchmark_{args.model}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{meta['cpu']} | {pinned} | 1 thread | batch 1")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
