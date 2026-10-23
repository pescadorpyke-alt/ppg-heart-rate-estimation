"""Phase 4: simulated on-device loop on one held-out subject (no hardware, D1).

Replays a recorded subject the way firmware would receive it, every 2 s:
    1. new sensor samples arrive (128 PPG @ 64 Hz, 64 x 3 ACC @ 32 Hz)
    2. streaming preprocessing: causal filter, ring buffers  (src/stream.py)
    3. quantize the window to int8, run the fully-INT8 TFLite model, dequantize
    4. log heart rate + timings
Process pinned to one CPU core, interpreter with one thread (XNNPACK, D18).
The model is the fold model for that subject, so it never saw this person.

Memory reported:
    flash        TFLite file size
    state        preprocessor buffers + filter state, as float32
    activations  lower bound: largest (input + output) of one layer;
                 upper bound: sum of every activation tensor with no reuse.
                 The real tensor arena lies in between (memory planning reuses
                 buffers); measuring it exactly needs TFLite Micro on a device.

Export environment:
    D:\\ppg-dalia-hr\\.venv-export\\Scripts\\python.exe -m src.device_demo --subject 7
Outputs results/device_demo_<model>_S<s>.csv / _summary.json and
docs/device_demo_<model>_S<s>.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from . import config as C
from .export import make_interpreter
from .loso import INPUTS
from .preprocessing import load_wrist
from .stream import ACC_STEP, PPG_STEP, StreamingPreprocessor


def activation_bounds(it) -> tuple[int, int]:
    """(lower, upper) bytes for activations, from the interpreter's own graph."""
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    size = lambda i: int(np.prod(tensors[i]["shape"])) * np.dtype(tensors[i]["dtype"]).itemsize  # noqa: E731
    ops = it._get_ops_details()
    produced = {o for op in ops for o in op["outputs"]}
    graph_in = [d["index"] for d in it.get_input_details()]
    acts = set(produced) | set(graph_in)
    lower = 0
    for op in ops:
        ins = [i for i in op["inputs"] if i in acts]
        lower = max(lower, sum(size(i) for i in ins) + sum(size(o) for o in op["outputs"]))
    upper = sum(size(i) for i in acts)
    return lower, upper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ds_32hz_fixed_causal")
    ap.add_argument("--variant", default="hr_stratified_2d")
    ap.add_argument("--subject", type=int, default=7)
    ap.add_argument("--format", default="int8_full")
    args = ap.parse_args()
    try:
        import psutil
        psutil.Process().cpu_affinity([0])
        pinned = "core 0"
    except Exception as e:   # pragma: no cover
        pinned = f"not pinned ({type(e).__name__})"

    inp = INPUTS[args.model]
    path = C.HEAVY_ROOT / "export" / args.model / args.variant / f"S{args.subject}" / f"model_{args.format}.tflite"
    blob = path.read_bytes()
    it = make_interpreter(blob, threads=1)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zero = ind["quantization"]
    out_scale, out_zero = outd["quantization"]
    act_lo, act_hi = activation_bounds(it)

    d = load_wrist(args.subject)
    bvp, acc, labels = d["bvp"], d["acc"], d["label"]
    meta = np.load(C.PROCESSED_DIR / "all_meta.npz")
    act = meta["activity"][meta["subject"] == args.subject]
    sp = StreamingPreprocessor(inp["fs"], inp["acc_norm"])

    first_p, first_a = C.PPG_WINDOW_LEN - PPG_STEP, C.ACC_WINDOW_LEN - ACC_STEP
    sp.push(bvp[:first_p], acc[:first_a])
    rows = []
    print(f"[device] model {path.name} ({len(blob) / 1024:.1f} KB) | subject S{args.subject} "
          f"(held out) | {pinned}, 1 thread")
    for k in range(len(labels)):
        p0, a0 = first_p + k * PPG_STEP, first_a + k * ACC_STEP
        t0 = time.perf_counter_ns()
        (kk, win), = sp.push(bvp[p0 : p0 + PPG_STEP], acc[a0 : a0 + ACC_STEP])
        t1 = time.perf_counter_ns()
        x = win[None, :, None, :].transpose(0, 2, 3, 1)          # (1, 1, 256, 4)
        if ind["dtype"] == np.int8:
            x = np.clip(np.round(x / in_scale + in_zero), -128, 127).astype(np.int8)
        it.set_tensor(ind["index"], x.astype(ind["dtype"]))
        it.invoke()
        yq = it.get_tensor(outd["index"]).reshape(-1)[0]
        hr = float((float(yq) - out_zero) * out_scale) if outd["dtype"] == np.int8 else float(yq)
        t2 = time.perf_counter_ns()
        rows.append({"k": kk, "t_s": 8 + 2 * kk, "true_bpm": float(labels[k]), "pred_bpm": hr,
                     "activity": C.ACTIVITY_NAMES[int(act[k])],
                     "preprocess_us": (t1 - t0) / 1000, "inference_us": (t2 - t1) / 1000})
        if k % 600 == 0:
            r = rows[-1]
            print(f"[device] t={r['t_s']:5d}s  HR {r['pred_bpm']:6.1f} bpm  (ECG {r['true_bpm']:6.1f})  "
                  f"{r['activity']:<12} pre {r['preprocess_us']:6.1f} us  infer {r['inference_us']:6.1f} us")

    pre = np.array([r["preprocess_us"] for r in rows])
    inf = np.array([r["inference_us"] for r in rows])
    err = np.abs(np.array([r["pred_bpm"] - r["true_bpm"] for r in rows]))
    summary = {
        "model": args.model, "file": str(path.name), "subject": args.subject, "windows": len(rows),
        "mae_bpm": float(err.mean()),
        "flash_bytes": len(blob), "preprocessor_state_bytes": sp.state_bytes(),
        "activation_bytes_lower": act_lo, "activation_bytes_upper": act_hi,
        "preprocess_us_median": float(np.median(pre)), "preprocess_us_p99": float(np.percentile(pre, 99)),
        "inference_us_median": float(np.median(inf)), "inference_us_p99": float(np.percentile(inf, 99)),
        "duty_cycle_pct_median": float(np.median(pre + inf) / 2e6 * 100),
        "cpu_affinity": pinned, "threads": 1,
    }
    tag = f"{args.model}_S{args.subject}"
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_DIR / f"device_demo_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    (C.RESULTS_DIR / f"device_demo_{tag}_summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["t_s"] for r in rows]) / 60
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=120)
    ax.plot(t, [r["true_bpm"] for r in rows], color="#444444", lw=1.2, label="ECG reference")
    ax.plot(t, [r["pred_bpm"] for r in rows], color="#d1495b", lw=0.9, alpha=0.85,
            label=f"INT8 model, streaming (MAE {summary['mae_bpm']:.1f} bpm)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("heart rate (bpm)")
    ax.set_title(f"Simulated device loop, held-out subject S{args.subject}: "
                 f"{len(blob) / 1024:.0f} KB model, {summary['inference_us_median']:.0f} µs/window on 1 core")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    (C.PROJECT_ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(C.PROJECT_ROOT / "docs" / f"device_demo_{tag}.png")

    print(f"\n[device] {len(rows)} windows, MAE {summary['mae_bpm']:.2f} bpm")
    print(f"[device] flash {len(blob) / 1024:.1f} KB | preprocessor state {sp.state_bytes()} B | "
          f"activations {act_lo / 1024:.1f}-{act_hi / 1024:.1f} KB (lower-upper bound)")
    print(f"[device] per 2 s window: preprocess {summary['preprocess_us_median']:.0f} us, "
          f"inference {summary['inference_us_median']:.0f} us (median) -> "
          f"CPU busy {summary['duty_cycle_pct_median']:.3f}% of the time")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
