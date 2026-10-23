"""Phase 3: single-window latency of every exported format, constrained CPU setup.

"Constrained" here means (DECISIONS.md D1, D14): the process is pinned to ONE
CPU core, every runtime uses ONE thread, batch size is 1, and each format gets
100 warm-up runs before 2,000 timed runs. This is still a desktop CPU and is
much faster than a microcontroller, so these numbers are NOT device latency.

TFLite is measured twice:
    default      the interpreter's default delegate (XNNPACK: SIMD-optimised)
    reference    built-in kernels only, no default delegate (closer to a
                 plain CPU without desktop SIMD instructions)

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
from .export import load_fold_model
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


def tflite_fn(path, delegate: str):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    kw = {}
    if delegate == "reference":
        kw["experimental_op_resolver_type"] = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    it = Interpreter(model_path=str(path), num_threads=1, **kw)
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
    xs_nwc = [x.transpose(0, 2, 1).copy() for x in xs_ncw]

    d = C.HEAVY_ROOT / "export" / args.model / f"S{args.fold}"
    model = load_fold_model(args.model, args.fold)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])

    cases = [("pytorch_fp32", "eager", None, lambda x: model(torch.from_numpy(x)), xs_ncw),
             ("onnx_fp32", "onnxruntime", d / "model.onnx",
              lambda x: sess.run(None, {"x": x}), xs_ncw)]
    for fmt in ["fp32", "fp16", "int8", "int8_full"]:
        p = d / f"model_{fmt}.tflite"
        for delegate in ["default", "reference"]:
            cases.append((f"tflite_{fmt}", delegate, p, tflite_fn(p, delegate), xs_nwc))

    rows = []
    with torch.no_grad():
        for fmt, runtime, path, fn, xs in cases:
            lat = timed(fn, xs)
            r = {"format": fmt, "runtime": runtime,
                 "bytes": path.stat().st_size if path else None,
                 "median_us": float(np.median(lat)), "p95_us": float(np.percentile(lat, 95)),
                 "p99_us": float(np.percentile(lat, 99)), "mean_us": float(lat.mean())}
            rows.append(r)
            size = f"{r['bytes'] / 1024:7.1f} KB" if r["bytes"] else "      -   "
            print(f"{fmt:<17}{runtime:<12}{size} | median {r['median_us']:8.1f} us | "
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
