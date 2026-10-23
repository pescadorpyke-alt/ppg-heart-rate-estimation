"""How are INT8-vs-FP32 differences distributed per window, and where are the big ones?

Re-runs the saved per-fold TFLite files (FP32 and INT8) on each held-out subject
and reports percentiles of |INT8 - FP32| per window, plus the inputs of the
windows with the largest differences. Export environment:
    D:\\ppg-dalia-hr\\.venv-export\\Scripts\\python.exe -m src.quant_errors
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from . import config as C
from .export import run_tflite
from .loso import INPUTS
from .preprocessing import x_path


def main(model: str = "ds_32hz_fixed", variant: str = "hr_stratified_2d"):
    from .export import make_interpreter, to_channels_last
    inp = INPUTS[model]
    X = np.load(x_path(inp["fs"], inp["acc_norm"]), mmap_mode="r")
    meta = np.load(C.PROCESSED_DIR / "all_meta.npz")
    y, subj, act = meta["y"], meta["subject"], meta["activity"]

    diffs, rows, ceilings = [], [], []
    for s in C.ALL_SUBJECTS:
        d = C.HEAVY_ROOT / "export" / model / variant / f"S{s}"
        it = make_interpreter((d / "model_int8_full.tflite").read_bytes())
        sc, zp = it.get_output_details()[0]["quantization"]
        ceiling = (127 - zp) * sc   # largest HR the INT8 model can output
        te = np.flatnonzero(subj == s)
        x = to_channels_last(np.asarray(X[te]), variant.rsplit("_", 1)[1])
        p32 = run_tflite((d / "model_fp32.tflite").read_bytes(), x)
        p8 = run_tflite((d / "model_int8.tflite").read_bytes(), x)
        diff = np.abs(p8 - p32)
        diffs.append(diff)
        acc_peak = np.abs(x[..., 1:]).reshape(len(x), -1).max(axis=1)
        for i in np.argsort(diff)[-3:]:
            rows.append({"subject": s, "window": int(i), "abs_diff": float(diff[i]),
                         "fp32": float(p32[i]), "int8": float(p8[i]), "true": float(y[te][i]),
                         "activity": C.ACTIVITY_NAMES[int(act[te][i])],
                         "acc_peak_0p1g": float(acc_peak[i]),
                         "acc_peak_pct_of_subject": float((acc_peak < acc_peak[i]).mean() * 100)})
        yt = y[te]
        ceilings.append({"subject": s, "int8_output_max_bpm": float(ceiling),
                         "pct_test_true_hr_above": float((yt > ceiling).mean() * 100),
                         "pct_test_fp32_pred_above": float((p32 > ceiling).mean() * 100)})
        print(f"S{s:<2} |INT8-FP32| median {np.median(diff):.3f}  p99 {np.percentile(diff, 99):.2f}"
              f"  max {diff.max():5.2f} bpm | INT8 output max {ceiling:6.1f} bpm, "
              f"true HR above it {ceilings[-1]['pct_test_true_hr_above']:.2f}%", flush=True)

    diff = np.concatenate(diffs)
    pct = {f"p{q}": float(np.percentile(diff, q)) for q in (50, 90, 99, 99.9)}
    share = {f"over_{t}_bpm": float((diff > t).mean() * 100) for t in (1, 2, 5, 10)}
    print(f"\nAll {len(diff)} windows: " + ", ".join(f"{k} {v:.3f}" for k, v in pct.items()))
    print("Share of windows with |INT8 - FP32| above: "
          + ", ".join(f"{k.split('_')[1]} bpm {v:.2f}%" for k, v in share.items()))
    print("\nLargest differences:")
    for r in sorted(rows, key=lambda r: -r["abs_diff"])[:8]:
        print(f"  S{r['subject']:<2} win {r['window']:<5} |diff| {r['abs_diff']:5.2f}  fp32 {r['fp32']:6.1f}"
              f"  int8 {r['int8']:6.1f}  true {r['true']:6.1f}  {r['activity']:<12}"
              f" ACC peak {r['acc_peak_0p1g']:5.2f} x0.1g (above {r['acc_peak_pct_of_subject']:.1f}% of subject)")
    cmin = min(c["int8_output_max_bpm"] for c in ceilings)
    cmax = max(c["int8_output_max_bpm"] for c in ceilings)
    print(f"INT8 output ceiling across folds: {cmin:.1f} .. {cmax:.1f} bpm "
          f"(training HR goes up to {y.max():.1f})")
    out = {"variant": variant, "percentiles_abs_diff_bpm": pct, "pct_windows_over": share,
           "int8_output_ceiling_per_fold": ceilings,
           "largest": sorted(rows, key=lambda r: -r["abs_diff"])[:15]}
    (C.RESULTS_DIR / f"quant_errors_{model}_{variant}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(variant=sys.argv[1] if len(sys.argv) > 1 else "hr_stratified_2d")
