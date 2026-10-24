"""Shared synthetic signals. No test needs the dataset, TensorFlow or a trained model."""
from __future__ import annotations

import numpy as np
import pytest

FS_PPG, FS_ACC = 64, 32


def sine(freq_hz: float, seconds: float, fs: int, amp: float = 1.0, phase: float = 0.0):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t + phase)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def recording(rng):
    """60 s of synthetic wrist data: 75 bpm pulse + noise, ACC in g with gravity."""
    seconds = 60
    bvp = sine(1.25, seconds, FS_PPG, amp=50) + rng.normal(0, 5, seconds * FS_PPG)
    acc = np.stack([sine(0.9, seconds, FS_ACC, amp=0.2, phase=p) for p in (0, 1, 2)], axis=1)
    acc = acc + np.array([0.0, 0.0, 1.0]) + rng.normal(0, 0.01, acc.shape)
    return bvp, acc
