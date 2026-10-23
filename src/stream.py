"""Streaming (causal) preprocessing, the way a wearable would run it (D19).

Training filters each whole recording with a zero-phase filter (sosfiltfilt),
which needs future samples. A device instead receives samples in small chunks
and must filter each one as it arrives. StreamingPreprocessor does exactly that:

    - causal band-pass (sosfilt) for PPG (64 Hz) and ACC (32 Hz), with the
      filter state carried from one chunk to the next;
    - ring buffers holding only the last 8 s of filtered samples
      (512 PPG + 256 x 3 ACC floats);
    - every 2 s (128 PPG / 64 ACC new samples) a window is emitted, built with
      the same assemble_window() used in training, so the filter is the only
      difference from the training pipeline.

Window k covers PPG samples [128k, 128k + 512) and ACC samples [64k, 64k + 256),
the same alignment as the dataset labels.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from . import config as C
from .preprocessing import _bandpass_sos, assemble_window

PPG_STEP = C.WINDOW_STRIDE_S * C.FS_PPG   # 128 new PPG samples per window
ACC_STEP = C.WINDOW_STRIDE_S * C.FS_ACC   # 64 new ACC samples per window


class StreamingPreprocessor:
    def __init__(self, fs_out: int = 32, acc_norm: str = "fixed"):
        self.fs_out, self.acc_norm = fs_out, acc_norm
        self.sos_ppg = _bandpass_sos(C.FS_PPG)
        self.sos_acc = _bandpass_sos(C.FS_ACC)
        self.zi_ppg = None            # filter state, set from the first sample
        self.zi_acc = None
        self.ppg_buf = np.zeros(C.PPG_WINDOW_LEN)          # last 8 s, filtered
        self.acc_buf = np.zeros((C.ACC_WINDOW_LEN, 3))
        self.n_ppg = 0                # samples received so far
        self.n_acc = 0
        self.k = 0                    # index of the next window to emit

    def state_bytes(self) -> int:
        """Memory this preprocessor keeps between calls, as float32 on a device."""
        floats = (self.ppg_buf.size + self.acc_buf.size
                  + self.sos_ppg.shape[0] * 2 + self.sos_acc.shape[0] * 2 * 3)
        return 4 * floats

    def _filter_ppg(self, x: np.ndarray) -> np.ndarray:
        if self.zi_ppg is None:   # start in steady state at the first value (causal)
            self.zi_ppg = sosfilt_zi(self.sos_ppg) * x[0]
        y, self.zi_ppg = sosfilt(self.sos_ppg, x, zi=self.zi_ppg)
        return y

    def _filter_acc(self, x: np.ndarray) -> np.ndarray:
        if self.zi_acc is None:
            self.zi_acc = sosfilt_zi(self.sos_acc)[:, :, None] * x[0][None, None, :]
        y, self.zi_acc = sosfilt(self.sos_acc, x, axis=0, zi=self.zi_acc)
        return y

    def push(self, ppg: np.ndarray, acc: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Feed raw samples (any chunk size); return [(k, window (4, T)), ...]
        for every window whose last sample arrived in this call."""
        ppg = np.asarray(ppg, dtype=np.float64).reshape(-1)
        acc = np.asarray(acc, dtype=np.float64).reshape(-1, 3)
        if len(ppg):
            f = self._filter_ppg(ppg)
            self.ppg_buf = np.concatenate([self.ppg_buf, f])[-C.PPG_WINDOW_LEN:]
            self.n_ppg += len(ppg)
        if len(acc):
            f = self._filter_acc(acc)
            self.acc_buf = np.concatenate([self.acc_buf, f])[-C.ACC_WINDOW_LEN:]
            self.n_acc += len(acc)
        out = []
        # Emit window k once both streams have reached its end. Chunks are
        # expected to be at most one hop (2 s) long, so at most one window is due.
        while (self.n_ppg >= self.k * PPG_STEP + C.PPG_WINDOW_LEN
               and self.n_acc >= self.k * ACC_STEP + C.ACC_WINDOW_LEN):
            if (self.n_ppg != self.k * PPG_STEP + C.PPG_WINDOW_LEN
                    or self.n_acc != self.k * ACC_STEP + C.ACC_WINDOW_LEN):
                raise ValueError("push at most one 2 s hop per call, aligned to the hop")
            out.append((self.k, assemble_window(self.ppg_buf, self.acc_buf,
                                                self.fs_out, self.acc_norm)))
            self.k += 1
        return out


def stream_subject(bvp: np.ndarray, acc: np.ndarray, n_windows: int,
                   fs_out: int = 32, acc_norm: str = "fixed") -> np.ndarray:
    """Replay a recording through StreamingPreprocessor in 2 s chunks, as a
    device would receive it. Returns (n_windows, 4, T) float32."""
    sp = StreamingPreprocessor(fs_out, acc_norm)
    # First chunk: the first 6 s, so that window 0 completes on the next hop.
    first_p = C.PPG_WINDOW_LEN - PPG_STEP
    first_a = C.ACC_WINDOW_LEN - ACC_STEP
    assert sp.push(bvp[:first_p], acc[:first_a]) == []
    windows = []
    for k in range(n_windows):
        p0, a0 = first_p + k * PPG_STEP, first_a + k * ACC_STEP
        got = sp.push(bvp[p0 : p0 + PPG_STEP], acc[a0 : a0 + ACC_STEP])
        assert len(got) == 1 and got[0][0] == k
        windows.append(got[0][1])
    return np.stack(windows)
