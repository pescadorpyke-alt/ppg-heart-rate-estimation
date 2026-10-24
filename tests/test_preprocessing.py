"""Preprocessing: filters, window assembly, and streaming == causal-training (D19, D21)."""
from __future__ import annotations

import numpy as np
import pytest

from src import config as C
from src.preprocessing import (
    _znorm,
    assemble_window,
    bandpass,
    bandpass_causal,
    preprocess_window,
    window_activity,
)
from src.stream import ACC_STEP, PPG_STEP, StreamingPreprocessor, stream_subject

from .conftest import FS_PPG, sine


@pytest.mark.parametrize("filt", [bandpass, bandpass_causal])
def test_bandpass_keeps_heart_rate_band_and_removes_the_rest(filt):
    def kept(freq_hz):
        x = sine(freq_hz, 60, FS_PPG)
        y = filt(x, FS_PPG)[FS_PPG * 20 : FS_PPG * 40]   # middle, away from start-up
        return np.abs(y).max()

    assert kept(1.5) > 0.9      # 90 bpm passes
    assert kept(0.05) < 0.1     # baseline wander removed
    assert kept(12.0) < 0.1     # high-frequency noise removed


def test_bandpass_handles_multichannel_acc():
    acc = np.stack([sine(1.5, 30, 32) for _ in range(3)], axis=1)
    for filt in (bandpass, bandpass_causal):
        assert filt(acc, 32).shape == acc.shape


@pytest.mark.parametrize("fs_out,acc_norm,length", [(64, "zscore", 512), (32, "zscore", 256),
                                                     (32, "fixed", 256)])
def test_assemble_window_shape_and_dtype(rng, fs_out, acc_norm, length):
    x = assemble_window(rng.normal(size=512), rng.normal(size=(256, 3)), fs_out, acc_norm)
    assert x.shape == (4, length)
    assert x.dtype == np.float32


def test_ppg_is_zscored_and_fixed_acc_keeps_g_units(rng):
    ppg, acc = rng.normal(5, 3, 512), rng.normal(0, 0.2, (256, 3))
    x = assemble_window(ppg, acc, 32, "fixed")
    assert abs(x[0].mean()) < 1e-5 and abs(x[0].std() - 1) < 1e-3
    np.testing.assert_allclose(x[1:], (acc / C.ACC_SCALE).T, rtol=1e-5)


def test_fixed_scaling_tells_still_from_moving(rng):
    still = assemble_window(rng.normal(size=512), rng.normal(0, 0.001, (256, 3)), 32, "fixed")
    moving = assemble_window(rng.normal(size=512), rng.normal(0, 0.2, (256, 3)), 32, "fixed")
    assert moving[1:].std() > 50 * still[1:].std()
    # Per-window z-scoring makes both look the same (the reason for D11).
    z_still = assemble_window(rng.normal(size=512), rng.normal(0, 0.001, (256, 3)), 32, "zscore")
    assert abs(z_still[1:].std() - 1) < 1e-3


def test_assemble_window_rejects_unknown_options(rng):
    with pytest.raises(ValueError):
        assemble_window(rng.normal(size=512), rng.normal(size=(256, 3)), fs_out=16)
    with pytest.raises(ValueError):
        assemble_window(rng.normal(size=512), rng.normal(size=(256, 3)), acc_norm="minmax")


def test_znorm_constant_channel_does_not_divide_by_zero():
    out = _znorm(np.ones((4, 100)))
    assert np.isfinite(out).all()


def test_preprocess_window_validates_lengths(rng):
    with pytest.raises(ValueError, match="PPG window"):
        preprocess_window(rng.normal(size=500), rng.normal(size=(256, 3)))
    with pytest.raises(ValueError, match="ACC window"):
        preprocess_window(rng.normal(size=512), rng.normal(size=(256, 2)))
    assert preprocess_window(rng.normal(size=512), rng.normal(size=(256, 3))).shape == (4, 512)


def test_streaming_equals_causal_filtering_of_whole_recording(recording):
    """The property the deployed model relies on: windows built by the streaming
    device code are the same as the causal windows it was trained on."""
    bvp, acc = recording
    n = (len(bvp) // FS_PPG - C.WINDOW_S) // C.WINDOW_STRIDE_S + 1
    streamed = stream_subject(bvp, acc, n, fs_out=32, acc_norm="fixed")
    bvp_f, acc_f = bandpass_causal(bvp, C.FS_PPG), bandpass_causal(acc, C.FS_ACC)
    for k in (0, n // 2, n - 1):
        offline = assemble_window(bvp_f[PPG_STEP * k : PPG_STEP * k + C.PPG_WINDOW_LEN],
                                  acc_f[ACC_STEP * k : ACC_STEP * k + C.ACC_WINDOW_LEN],
                                  32, "fixed")
        np.testing.assert_allclose(streamed[k], offline, atol=1e-5)


def test_streaming_differs_from_zero_phase_training(recording):
    """Why D21 exists: causal and zero-phase windows are not the same signal."""
    bvp, acc = recording
    streamed = stream_subject(bvp, acc, 5)
    zp = assemble_window(bandpass(bvp, 64)[:512], bandpass(acc, 32)[:256], 32, "fixed")
    assert np.abs(streamed[0] - zp).mean() > 0.1


def test_stream_emits_one_window_per_hop_and_reports_state_size(recording):
    bvp, acc = recording
    sp = StreamingPreprocessor()
    first_p, first_a = C.PPG_WINDOW_LEN - PPG_STEP, C.ACC_WINDOW_LEN - ACC_STEP
    assert sp.push(bvp[:first_p], acc[:first_a]) == []
    for k in range(3):
        out = sp.push(bvp[first_p + k * PPG_STEP : first_p + (k + 1) * PPG_STEP],
                      acc[first_a + k * ACC_STEP : first_a + (k + 1) * ACC_STEP])
        assert [w for w, _ in out] == [k]
        assert out[0][1].shape == (4, 256)
    assert sp.state_bytes() == 5248


def test_stream_rejects_chunks_longer_than_one_hop(recording):
    bvp, acc = recording
    with pytest.raises(ValueError):
        StreamingPreprocessor().push(bvp[:2000], acc[:1000])


def test_window_activity_is_majority_vote():
    act = np.array([1] * 8 + [2] * 24 + [3] * 40, dtype=float)   # 4 Hz labels
    assert window_activity(act, 0) == 2      # window 0 = samples 0..31: 8 x 1, 24 x 2
    assert window_activity(act, 100) == -1   # past the end
