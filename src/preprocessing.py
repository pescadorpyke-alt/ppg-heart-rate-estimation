"""Phase 1 — data pipeline for PPG-DaLiA.

Loads per-subject pickles, band-pass filters the wrist PPG + accelerometer,
segments into 8 s windows (2 s hop), z-normalizes per window, and splits
train/val *by subject* (no window leakage between sets).

PPG-DaLiA structure (each S{i}.pkl, pickled with latin1 encoding):
    data['signal']['wrist']['BVP']  -> (N_ppg, 1)  @ 64 Hz   (our PPG)
    data['signal']['wrist']['ACC']  -> (N_acc, 3)  @ 32 Hz   (motion reference)
    data['label']                   -> (N_win,)              HR in bpm
The label is computed by the dataset authors over a sliding 8 s window shifted
by 2 s, so label index k aligns exactly with:
    PPG window = BVP[k*2*64 : k*2*64 + 8*64]   (512 samples)
    ACC window = ACC[k*2*32 : k*2*32 + 8*32]   (256 samples)

Model input per window: (4, 512) float32 — channel 0 = PPG, channels 1-3 = ACC
(x, y, z), with ACC upsampled 32->64 Hz so all channels share a time axis.
Each channel is z-normalized within its own window.
"""
from __future__ import annotations

import pickle

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config as C


# --- Loading -----------------------------------------------------------------
def load_subject(subject_id: int) -> dict:
    """Load one PPG-DaLiA subject pickle. Needs latin1 (pickled under Py2).

    Security note: pickle.load executes arbitrary code on untrusted input.
    Here the input is the official UCI PPG-DaLiA dataset (CC BY 4.0), which is
    distributed *only* as pickles — there is no JSON/CSV alternative. Files are
    downloaded once from archive.ics.uci.edu over HTTPS and not user-supplied,
    so this is a trusted, required use.
    """
    path = C.RAW_DIR / f"S{subject_id}.pkl"
    if not path.exists():
        # Some extractions nest pickles under PPG_FieldStudy/S{i}/S{i}.pkl
        alt = C.RAW_DIR / "PPG_FieldStudy" / f"S{subject_id}" / f"S{subject_id}.pkl"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"Subject pickle not found: {path} (also tried {alt}). "
                "Extract the dataset into config.RAW_DIR."
            )
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


# --- Filtering ---------------------------------------------------------------
def _bandpass_sos(fs: int):
    nyq = 0.5 * fs
    return butter(
        C.BP_ORDER,
        [C.BP_LOWCUT_HZ / nyq, C.BP_HIGHCUT_HZ / nyq],
        btype="band",
        output="sos",
    )


def bandpass(signal: np.ndarray, fs: int) -> np.ndarray:
    """Zero-phase band-pass along axis 0. signal: (T,) or (T, channels)."""
    sos = _bandpass_sos(fs)
    return sosfiltfilt(sos, signal, axis=0)


def _upsample_acc(acc_win: np.ndarray, out_len: int) -> np.ndarray:
    """Linearly upsample ACC window (256, 3) -> (out_len, 3) to match PPG rate."""
    src_len = acc_win.shape[0]
    src_t = np.linspace(0.0, 1.0, src_len)
    dst_t = np.linspace(0.0, 1.0, out_len)
    return np.stack(
        [np.interp(dst_t, src_t, acc_win[:, c]) for c in range(acc_win.shape[1])],
        axis=1,
    )


def _znorm(x: np.ndarray) -> np.ndarray:
    """Per-channel z-norm of a (channels, T) window."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


# --- Single-window preprocessing (shared by training and inference) ----------
def preprocess_window(ppg_raw: np.ndarray, acc_raw: np.ndarray) -> np.ndarray:
    """Turn one RAW window into the model input (4, 512) float32.

    ppg_raw: (PPG_WINDOW_LEN,) raw BVP @ 64 Hz   (512 samples = 8 s)
    acc_raw: (ACC_WINDOW_LEN, 3) raw ACC @ 32 Hz (256 samples = 8 s)

    Applies the same band-pass + ACC upsample + per-channel z-norm used to build
    the training set, so the API sees inputs identical in distribution to training.

    Note: training band-passes each full recording then slices windows; here we
    band-pass the provided window. Edge transients of the zero-phase filter make
    this a minor, documented train/serve difference (small vs. the model's MAE).
    """
    ppg = np.asarray(ppg_raw, dtype=np.float64).reshape(-1)
    acc = np.asarray(acc_raw, dtype=np.float64)
    if ppg.shape[0] != C.PPG_WINDOW_LEN:
        raise ValueError(f"PPG window must be {C.PPG_WINDOW_LEN} samples, got {ppg.shape[0]}")
    if acc.shape != (C.ACC_WINDOW_LEN, 3):
        raise ValueError(f"ACC window must be ({C.ACC_WINDOW_LEN}, 3), got {acc.shape}")

    ppg_f = bandpass(ppg, C.FS_PPG)
    acc_f = bandpass(acc, C.FS_ACC)
    acc_up = _upsample_acc(acc_f, C.PPG_WINDOW_LEN)            # (512, 3)
    stacked = np.concatenate([ppg_f.reshape(-1, 1), acc_up], axis=1).T   # (4, 512)
    return _znorm(stacked).astype(np.float32)


# --- Windowing ---------------------------------------------------------------
def windows_for_subject(subject_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for one subject. X: (n, 4, 512) float32, y: (n,) float32."""
    data = load_subject(subject_id)
    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=np.float64).reshape(-1)
    acc = np.asarray(data["signal"]["wrist"]["ACC"], dtype=np.float64)  # (N, 3)
    labels = np.asarray(data["label"], dtype=np.float64).reshape(-1)

    # Filter full signals once (zero-phase), then slice windows.
    bvp_f = bandpass(bvp, C.FS_PPG)
    acc_f = bandpass(acc, C.FS_ACC)

    ppg_step = C.WINDOW_STRIDE_S * C.FS_PPG   # 128
    acc_step = C.WINDOW_STRIDE_S * C.FS_ACC   # 64

    X, y = [], []
    for k, hr in enumerate(labels):
        p0 = k * ppg_step
        a0 = k * acc_step
        ppg_win = bvp_f[p0 : p0 + C.PPG_WINDOW_LEN]
        acc_win = acc_f[a0 : a0 + C.ACC_WINDOW_LEN]
        # Drop trailing windows that run past the signal end.
        if len(ppg_win) < C.PPG_WINDOW_LEN or len(acc_win) < C.ACC_WINDOW_LEN:
            break
        if not np.isfinite(hr) or hr <= 0:
            continue
        acc_up = _upsample_acc(acc_win, C.PPG_WINDOW_LEN)        # (512, 3)
        stacked = np.concatenate(
            [ppg_win.reshape(-1, 1), acc_up], axis=1
        ).T                                                       # (4, 512)
        X.append(_znorm(stacked).astype(np.float32))
        y.append(np.float32(hr))

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_split(
    train_subjects=None, val_subjects=None
) -> dict[str, np.ndarray]:
    """Build train/val arrays, split by subject."""
    train_subjects = train_subjects or C.TRAIN_SUBJECTS
    val_subjects = val_subjects or C.VAL_SUBJECTS

    def collect(subjects):
        xs, ys = [], []
        for s in subjects:
            Xs, ys_ = windows_for_subject(s)
            xs.append(Xs)
            ys.append(ys_)
            print(f"  subject S{s}: {len(ys_)} windows, "
                  f"HR {ys_.min():.0f}-{ys_.max():.0f} bpm")
        return np.concatenate(xs), np.concatenate(ys)

    print(f"Building TRAIN from subjects {train_subjects}")
    X_train, y_train = collect(train_subjects)
    print(f"Building VAL from subjects {val_subjects}")
    X_val, y_val = collect(val_subjects)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
    }


def main():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split = build_split()
    out = C.PROCESSED_DIR / "windows.npz"
    np.savez_compressed(out, **split)
    print(f"\nSaved {out}")
    print(f"  X_train {split['X_train'].shape}  y_train {split['y_train'].shape}")
    print(f"  X_val   {split['X_val'].shape}  y_val   {split['y_val'].shape}")


if __name__ == "__main__":
    main()
