"""Why does the naive FFT baseline fail? Classify every window's error.

For each window, the naive estimate (largest PPG spectral peak, 30-240 bpm) is
put into one of these groups, checked in this order:
    ok           |error| <= 10 bpm
    motion       pick is within 6 bpm of the dominant ACC frequency
    harmonic     pick is within 10 % of 2 x true HR
    subharmonic  pick is within 10 % of true HR / 2
    low_edge     pick is below 45 bpm (filter-edge / baseline energy)
    other        anything else
Then reports, per group, the share of windows and the share of total absolute
error, plus MAE per activity.

Run after `python -m src.preprocessing --all`:
    python -m src.diagnose_fft
"""
from __future__ import annotations

import csv

import numpy as np

from . import config as C
from .baseline import fft_hr_estimate


def dominant_acc_bpm(acc_bp: np.ndarray, fs: int = C.FS_PPG) -> float:
    """Dominant frequency (bpm) of the summed ACC power spectrum, 30-240 bpm.

    acc_bp: (3, T) band-passed ACC (z-normalised channels rescaled by their
    pre-normalisation std, so axes with more motion weigh more).
    """
    n = acc_bp.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = (np.abs(np.fft.rfft(acc_bp * np.hanning(n), axis=1)) ** 2).sum(axis=0)
    band = (freqs >= C.BP_LOWCUT_HZ) & (freqs <= C.BP_HIGHCUT_HZ)
    return float(freqs[band][np.argmax(power[band])] * 60.0)


def classify(pred: float, true: float, acc_bpm: float) -> str:
    if abs(pred - true) <= 10:
        return "ok"
    if abs(pred - acc_bpm) <= 6:
        return "motion"
    if abs(pred - 2 * true) <= 0.1 * 2 * true:
        return "harmonic"
    if abs(pred - true / 2) <= 0.1 * true / 2:
        return "subharmonic"
    if pred < 45:
        return "low_edge"
    return "other"


def main():
    X = np.load(C.PROCESSED_DIR / "all_X.npy", mmap_mode="r")
    meta = np.load(C.PROCESSED_DIR / "all_meta.npz")
    y, subj, act, scale = meta["y"], meta["subject"], meta["activity"], meta["scale"]

    n = len(y)
    pred = np.empty(n)
    group = np.empty(n, dtype=object)
    for i in range(n):
        x = np.asarray(X[i])
        pred[i] = fft_hr_estimate(x[0])
        acc_bp = x[1:] * scale[i, 1:, None]
        group[i] = classify(pred[i], y[i], dominant_acc_bpm(acc_bp))
    err = np.abs(pred - y)

    groups = ["ok", "motion", "harmonic", "subharmonic", "low_edge", "other"]
    print(f"Naive FFT, all {n} windows, 15 subjects")
    print(f"  window-weighted MAE: {err.mean():.2f} bpm")
    per_subj = np.array([err[subj == s].mean() for s in C.ALL_SUBJECTS])
    print(f"  per-subject MAE: mean {per_subj.mean():.2f} +/- {per_subj.std():.2f} bpm\n")

    print(f"{'group':<12} {'% windows':>10} {'% of abs error':>15} {'MAE in group':>13}")
    rows = []
    for g in groups:
        m = group == g
        row = {
            "group": g,
            "pct_windows": 100 * m.mean(),
            "pct_abs_error": 100 * err[m].sum() / err.sum(),
            "mae_in_group": err[m].mean() if m.any() else float("nan"),
        }
        rows.append(row)
        print(f"{g:<12} {row['pct_windows']:10.1f} {row['pct_abs_error']:15.1f} "
              f"{row['mae_in_group']:13.1f}")

    print(f"\n{'activity':<13} {'windows':>8} {'MAE':>7} {'% motion picks':>15}")
    act_rows = []
    for a, name in C.ACTIVITY_NAMES.items():
        m = act == a
        r = {"activity": name, "windows": int(m.sum()), "mae": err[m].mean(),
             "pct_motion": 100 * (group[m] == "motion").mean()}
        act_rows.append(r)
        print(f"{name:<13} {r['windows']:8d} {r['mae']:7.2f} {r['pct_motion']:15.1f}")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path, data in [("fft_diagnostics_groups.csv", rows),
                       ("fft_diagnostics_activity.csv", act_rows)]:
        with open(C.RESULTS_DIR / path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            for r in data:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in r.items()})
    print(f"\nWrote {C.RESULTS_DIR}/fft_diagnostics_*.csv")


if __name__ == "__main__":
    main()
