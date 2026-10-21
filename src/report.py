"""Build the README result tables from the files in results/.

Prints markdown so the README numbers come straight from script output:
    python -m src.report --model cnn_v0
"""
from __future__ import annotations

import argparse
import csv
import json

from . import config as C

# Deep PPG (Reiss et al. 2019), Table 11: CNN ensemble MAE per activity on
# PPG-DaLiA, for context. https://pmc.ncbi.nlm.nih.gov/articles/PMC6679242/
DEEP_PPG_ACTIVITY = {
    "transition": 8.77, "sitting": 4.93, "stairs": 16.98, "table_soccer": 12.16,
    "cycling": 12.48, "driving": 4.96, "lunch": 5.22, "walking": 9.21,
    "working": 3.84,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_v0")
    args = ap.parse_args()
    R = C.RESULTS_DIR

    with open(R / "baselines_per_subject.csv") as f:
        base = {r["method"]: r for r in csv.DictReader(f)}
    with open(R / f"loso_{args.model}.csv") as f:
        cnn = {int(r["test_subject"]): r for r in csv.DictReader(f)}
    summ = json.loads((R / f"loso_{args.model}_summary.json").read_text())
    with open(R / "fft_diagnostics_activity.csv") as f:
        fft_act = {r["activity"]: r for r in csv.DictReader(f)}

    subs = C.ALL_SUBJECTS
    print("| Subject | Constant | Naive FFT | DSP, soft tracking | CNN | CNN best epoch |")
    print("|---|---|---|---|---|---|")
    for s in subs:
        r = cnn[s]
        print(f"| S{s} | {float(base['constant'][f'S{s}']):.2f} "
              f"| {float(base['naive'][f'S{s}']):.2f} "
              f"| {float(base['+track_soft'][f'S{s}']):.2f} "
              f"| {float(r['test_mae']):.2f} | {r['best_epoch']} / {r['epochs_run']} |")
    print(f"| **Mean ± std** | {float(base['constant']['mean']):.2f} ± {float(base['constant']['std']):.2f} "
          f"| {float(base['naive']['mean']):.2f} ± {float(base['naive']['std']):.2f} "
          f"| {float(base['+track_soft']['mean']):.2f} ± {float(base['+track_soft']['std']):.2f} "
          f"| **{summ['mae_mean_over_subjects']:.2f} ± {summ['mae_std_over_subjects']:.2f}** | |")
    worst = max(subs, key=lambda s: float(cnn[s]["test_mae"]))
    print(f"\nCNN worst subject: S{worst}; window-weighted CNN MAE "
          f"{summ['mae_window_weighted']:.2f}")

    print("\n| Activity | Windows | Naive FFT | CNN (this repo) | Deep PPG ensemble (paper) |")
    print("|---|---|---|---|---|")
    for name, mae in summ["per_activity_mae"].items():
        print(f"| {name.replace('_', ' ')} | {fft_act[name]['windows']} "
              f"| {float(fft_act[name]['mae']):.2f} | {mae:.2f} "
              f"| {DEEP_PPG_ACTIVITY[name]:.2f} |")

    hit_cap = [f"S{s}" for s in subs if int(cnn[s]["epochs_run"]) == summ["epochs_max"]
               and int(cnn[s]["best_epoch"]) == summ["epochs_max"]]
    print(f"\nFolds whose best epoch was the last allowed epoch: {hit_cap or 'none'}")
    total = sum(float(cnn[s]["train_seconds"]) for s in subs)
    print(f"Total training time: {total / 3600:.2f} h")


if __name__ == "__main__":
    main()
