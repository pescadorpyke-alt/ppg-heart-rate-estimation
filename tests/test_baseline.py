"""DSP baseline on synthetic signals with a known heart rate."""
from __future__ import annotations

import numpy as np

from src.baseline import VARIANTS, DSPConfig, dsp_estimate, fft_hr_estimate
from src.export import calib_indices, to_channels_last

from .conftest import sine


def test_naive_fft_finds_the_heart_rate_within_one_bin():
    ppg = sine(1.2, 8, 64)                         # 72 bpm, 8 s window
    assert abs(fft_hr_estimate(ppg) - 72) <= 3.75  # 512-point FFT: 7.5 bpm bins


def test_zero_padding_gives_finer_estimates():
    ppg = sine(1.23, 8, 64)                        # 73.8 bpm, between coarse bins
    fine = dsp_estimate(ppg, np.zeros((3, 512)), VARIANTS["+zeropad"])
    assert abs(fine - 73.8) < 1.0


def test_motion_rejection_skips_the_accelerometer_peak():
    hr_hz, motion_hz = 1.3, 2.0                    # 78 bpm pulse, 120 steps/min arm swing
    ppg = sine(hr_hz, 8, 64) + 2.0 * sine(motion_hz, 8, 64)   # motion dominates the PPG
    acc = np.stack([sine(motion_hz, 8, 64, amp=0.5)] * 3)
    naive = dsp_estimate(ppg, acc, DSPConfig(nfft=4096, fmin_bpm=40, fmax_bpm=200))
    rejected = dsp_estimate(ppg, acc, VARIANTS["+acc"])
    assert abs(naive - 120) < 2                    # without ACC it picks the motion
    assert abs(rejected - 78) < 2                  # with ACC it finds the pulse


def test_soft_tracking_prefers_the_peak_closest_to_the_last_estimate():
    ppg = sine(1.0, 8, 64) + 0.9 * sine(1.6, 8, 64)   # peaks at 60 and 96 bpm
    cfg = DSPConfig(nfft=4096, fmin_bpm=40, fmax_bpm=200, track="soft")
    assert abs(dsp_estimate(ppg, np.zeros((3, 512)), cfg, prev=None) - 60) < 2
    assert abs(dsp_estimate(ppg, np.zeros((3, 512)), cfg, prev=95) - 96) < 2


def test_channels_last_layouts():
    x = np.zeros((2, 4, 256))
    assert to_channels_last(x, "1d").shape == (2, 256, 4)
    assert to_channels_last(x, "2d").shape == (2, 1, 256, 4)


def test_calibration_sampling_covers_high_heart_rates():
    """D16: stratified calibration must include the rare high-HR windows."""
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.uniform(50, 100, 9_900), rng.uniform(150, 190, 100)])
    idx = np.arange(len(y))
    strat = calib_indices(idx, y, "hr_stratified", np.random.default_rng(1))
    rand = calib_indices(idx, y, "random", np.random.default_rng(1))
    assert len(strat) == len(np.unique(strat)) == 500
    assert (y[strat] > 150).sum() > 5 * max(1, (y[rand] > 150).sum())
