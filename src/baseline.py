"""Phase 5 — naive baselines on the validation set, for honest comparison.

1. Constant baseline: always predict the training-mean HR.
2. Classic signal-processing baseline (no ML, no accelerometer): estimate HR from
   the dominant frequency of the PPG window via FFT, picking the spectral peak in
   the plausible HR band (0.5-4 Hz). This is the standard non-deep PPG-HR baseline;
   it has no way to reject motion artefacts, which is exactly what the CNN + ACC
   input is meant to fix.

Operates on the already-processed windows (no pickle load needed).
"""
from __future__ import annotations

import numpy as np

from . import config as C


def fft_hr_estimate(ppg_window: np.ndarray, fs: int = C.FS_PPG) -> float:
    """Dominant-frequency HR (bpm) for one PPG window (length PPG_WINDOW_LEN)."""
    n = len(ppg_window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(ppg_window * np.hanning(n)))
    band = (freqs >= C.BP_LOWCUT_HZ) & (freqs <= C.BP_HIGHCUT_HZ)
    peak_freq = freqs[band][np.argmax(mag[band])]
    return float(peak_freq * 60.0)


def main():
    d = np.load(C.PROCESSED_DIR / "windows.npz")
    y_train, X_val, y_val = d["y_train"], d["X_val"], d["y_val"]

    # 1. Constant (train-mean) baseline.
    const_pred = np.full_like(y_val, y_train.mean())
    const_mae = np.mean(np.abs(y_val - const_pred))

    # 2. FFT peak baseline on the PPG channel (channel 0).
    fft_pred = np.array([fft_hr_estimate(x[0]) for x in X_val])
    fft_mae = np.mean(np.abs(y_val - fft_pred))

    print(f"Validation windows: {len(y_val)}")
    print(f"  Constant (train-mean={y_train.mean():.1f}) baseline MAE: {const_mae:.2f} bpm")
    print(f"  FFT peak-frequency baseline MAE:            {fft_mae:.2f} bpm")
    print(f"  (CNN model val MAE for reference:           12.09 bpm)")


if __name__ == "__main__":
    main()
