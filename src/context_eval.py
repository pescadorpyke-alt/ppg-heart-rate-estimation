"""How much signal before the window does a stateless API need? (D23)

For each subject s, the LOSO fold model of ds_32hz_fixed_causal (never trained on
s) is run on s with inputs built from c seconds of context + the 8 s window,
filtered causally from a cold start (preprocess_with_context), for several c.
Reference: full streaming (filter running since the start of the recording),
i.e. what the model was trained and evaluated on. Only windows with >= 30 s of
history are used, so every c is scored on the same windows.

    python -m src.context_eval
Writes results/context_eval_ds_32hz_fixed_causal.csv and _summary.json.
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
import torch

from . import config as C
from .loso import INPUTS, MODELS
from .preprocessing import load_wrist, preprocess_with_context, x_path

MODEL = "ds_32hz_fixed_causal"
CONTEXTS_S = [0, 2, 4, 8, 16, 30]
MIN_HISTORY_S = 30


def main():
    inp = INPUTS[MODEL]
    X = np.load(x_path(inp["fs"], inp["acc_norm"], inp["filt"]), mmap_mode="r")
    meta = np.load(C.PROCESSED_DIR / "all_meta.npz")
    y, subj = meta["y"], meta["subject"]
    torch.set_num_threads(2)
    k0 = MIN_HISTORY_S // C.WINDOW_STRIDE_S          # first window with >= 30 s history
    rows = []
    for s in C.ALL_SUBJECTS:
        ckpt = torch.load(C.CHECKPOINT_DIR / f"loso_{MODEL}_S{s}.pt", map_location="cpu",
                          weights_only=True)
        model = MODELS[MODEL]()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        te = np.flatnonzero(subj == s)[k0:]
        ks = np.arange(k0, k0 + len(te))
        d = load_wrist(s)
        inputs = {"streaming": np.asarray(X[te])}
        for c in CONTEXTS_S:
            inputs[f"c{c}"] = np.stack([
                preprocess_with_context(
                    d["bvp"][128 * k - 64 * c : 128 * k + 512],
                    d["acc"][64 * k - 32 * c : 64 * k + 256], inp["fs"], inp["acc_norm"])
                for k in ks])
        line = []
        with torch.no_grad():
            for name, x in inputs.items():
                pred = model(torch.from_numpy(x)).numpy()
                mae = float(np.abs(pred - y[te]).mean())
                diff = float(np.abs(x - inputs["streaming"]).max())
                rows.append({"test_subject": s, "input": name, "test_mae": mae,
                             "max_abs_input_diff_vs_streaming": diff})
                line.append(f"{name} {mae:.2f}")
        print(f"S{s:<2} " + " | ".join(line), flush=True)

    summary = {}
    ref = np.array([r["test_mae"] for r in rows if r["input"] == "streaming"])
    for name in ["streaming"] + [f"c{c}" for c in CONTEXTS_S]:
        m = np.array([r["test_mae"] for r in rows if r["input"] == name])
        summary[name] = {"mae_mean_over_subjects": float(m.mean()),
                         "mae_std_over_subjects": float(m.std()),
                         "mean_diff_vs_streaming": float((m - ref).mean()),
                         "max_input_diff_vs_streaming": float(max(
                             r["max_abs_input_diff_vs_streaming"] for r in rows if r["input"] == name))}
    ok = [c for c in CONTEXTS_S if summary[f"c{c}"]["mean_diff_vs_streaming"] <= 0.1]
    summary["chosen_min_context_s"] = min(ok) if ok else None
    with open(C.RESULTS_DIR / f"context_eval_{MODEL}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    (C.RESULTS_DIR / f"context_eval_{MODEL}_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    for name, v in summary.items():
        if isinstance(v, dict):
            print(f"{name:<10} MAE {v['mae_mean_over_subjects']:.3f} ± {v['mae_std_over_subjects']:.3f} "
                  f"| vs streaming {v['mean_diff_vs_streaming']:+.3f} | max input diff "
                  f"{v['max_input_diff_vs_streaming']:.3g}")
    print(f"chosen minimum context (D23 rule): {summary['chosen_min_context_s']} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
