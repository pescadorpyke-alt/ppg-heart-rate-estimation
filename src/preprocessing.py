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
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

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


def load_wrist(subject_id: int) -> dict[str, np.ndarray]:
    """Wrist signals + labels for one subject, cached as a small .npz.

    The pickles are 0.8-1.7 GB each, mostly 700 Hz chest signals we never use.
    The first call loads the pickle and writes CACHE_DIR/S{i}_wrist.npz
    (a few MB); later calls read only that file.

    Returns: bvp (N,) @64 Hz, acc (M, 3) @32 Hz, label (K,) bpm,
             activity (A,) @4 Hz (IDs, see config.ACTIVITY_NAMES).
    """
    cache = C.CACHE_DIR / f"S{subject_id}_wrist.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    data = load_subject(subject_id)
    out = {
        "bvp": np.asarray(data["signal"]["wrist"]["BVP"], dtype=np.float64).reshape(-1),
        "acc": np.asarray(data["signal"]["wrist"]["ACC"], dtype=np.float64),
        "label": np.asarray(data["label"], dtype=np.float64).reshape(-1),
        "activity": np.asarray(data["activity"], dtype=np.float64).reshape(-1),
    }
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


# --- Filtering ---------------------------------------------------------------
def _bandpass_sos(fs: int):
    nyq = 0.5 * fs
    return butter(
        C.BP_ORDER,
        [C.BP_LOWCUT_HZ / nyq, C.BP_HIGHCUT_HZ / nyq],
        btype="band",
        output="sos",
    )


def bandpass_causal(signal: np.ndarray, fs: int) -> np.ndarray:
    """Causal band-pass along axis 0 (each output depends only on past samples),
    started in steady state at the first sample. Identical to what
    stream.StreamingPreprocessor computes chunk by chunk (checked), so training
    on it matches a device that filters as samples arrive (D19, D21)."""
    sos = _bandpass_sos(fs)
    x = np.asarray(signal, dtype=np.float64)
    zi = sosfilt_zi(sos)
    zi = zi * x[0] if x.ndim == 1 else zi[:, :, None] * x[0][None, None, :]
    return sosfilt(sos, x, axis=0, zi=zi)[0]


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


def assemble_window(ppg_bp: np.ndarray, acc_bp: np.ndarray, fs_out: int = 64,
                    acc_norm: str = "zscore") -> np.ndarray:
    """Band-passed PPG (512,) @64 Hz + ACC (256, 3) @32 Hz -> model input.

    fs_out=64:  ACC linearly upsampled to 64 Hz -> (4, 512)  (original input)
    fs_out=32:  PPG decimated to 32 Hz (every 2nd sample; safe because it is
                already band-limited to 4 Hz, far below the new 16 Hz Nyquist)
                -> (4, 256), half the samples for every conv layer.
    acc_norm="zscore": every channel z-scored per window (original).
    acc_norm="fixed":  PPG z-scored; ACC (in g) divided by ACC_SCALE (0.1 g),
                       so motion strength is kept.
    """
    if fs_out == 64:
        acc_rs = _upsample_acc(acc_bp, C.PPG_WINDOW_LEN)          # (512, 3)
        ppg_rs = ppg_bp
    elif fs_out == 32:
        acc_rs = acc_bp                                            # (256, 3)
        ppg_rs = ppg_bp[::2]                                       # (256,)
    else:
        raise ValueError(f"fs_out must be 64 or 32, got {fs_out}")
    stacked = np.concatenate([ppg_rs.reshape(-1, 1), acc_rs], axis=1).T
    if acc_norm == "zscore":
        return _znorm(stacked).astype(np.float32)
    if acc_norm == "fixed":
        out = stacked / C.ACC_SCALE
        out[:1] = _znorm(stacked[:1])
        return out.astype(np.float32)
    raise ValueError(f"acc_norm must be 'zscore' or 'fixed', got {acc_norm}")


# --- Single-window preprocessing (shared by training and inference) ----------
def preprocess_window(ppg_raw: np.ndarray, acc_raw: np.ndarray, fs_out: int = 64,
                      acc_norm: str = "zscore") -> np.ndarray:
    """Turn one RAW window into the model input, float32.

    ppg_raw: (PPG_WINDOW_LEN,) raw BVP @ 64 Hz   (512 samples = 8 s)
    acc_raw: (ACC_WINDOW_LEN, 3) raw ACC @ 32 Hz (256 samples = 8 s)

    Applies the same band-pass + resampling + normalisation used to build the
    training set (see assemble_window), so the API sees inputs identical in
    distribution to training. Defaults give the original (4, 512) input.

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
    return assemble_window(ppg_f, acc_f, fs_out, acc_norm)


# --- Windowing ---------------------------------------------------------------
def window_activity(activity: np.ndarray, k: int) -> int:
    """Activity ID for window k: the most common 4 Hz label inside the 8 s window."""
    a0 = k * C.WINDOW_STRIDE_S * C.FS_ACTIVITY
    seg = activity[a0 : a0 + C.WINDOW_S * C.FS_ACTIVITY].astype(int)
    if seg.size == 0:
        return -1
    return int(np.bincount(seg).argmax())


def windows_for_subject(subject_id: int, return_meta: bool = False,
                        fs_out: int = 64, acc_norm: str = "zscore",
                        filt: str = "zerophase"):
    """Return (X, y) for one subject. X: (n, 4, T) float32, y: (n,) float32.

    T = 512 for fs_out=64 (default), 256 for fs_out=32; see assemble_window.

    With return_meta=True, also returns a dict with per-window
    `activity` (n,) int and `scale` (n, 4) float32: the std of each band-passed
    channel *before* z-normalisation (how strong the pulse / motion was).
    """
    data = load_wrist(subject_id)
    bvp, acc, labels = data["bvp"], data["acc"], data["label"]

    # Filter full signals once (zero-phase), then slice windows.
    # zerophase: training default (needs future samples, offline only);
    # causal: what a streaming device computes (D21).
    bp = bandpass_causal if filt == "causal" else bandpass
    bvp_f = bp(bvp, C.FS_PPG)
    acc_f = bp(acc, C.FS_ACC)

    ppg_step = C.WINDOW_STRIDE_S * C.FS_PPG   # 128
    acc_step = C.WINDOW_STRIDE_S * C.FS_ACC   # 64

    X, y, act, scale = [], [], [], []
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
        X.append(assemble_window(ppg_win, acc_win, fs_out, acc_norm))
        y.append(np.float32(hr))
        if return_meta:
            act.append(window_activity(data["activity"], k))
            scale.append(np.concatenate([[ppg_win.std()], acc_win.std(axis=0)]))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if not return_meta:
        return X, y
    meta = {
        "activity": np.asarray(act, dtype=np.int8),
        "scale": np.asarray(scale, dtype=np.float32),
    }
    return X, y, meta


def x_path(fs_out: int = 64, acc_norm: str = "zscore", filt: str = "zerophase"):
    """Where build_all saves X for an input option (original input: all_X.npy)."""
    if (fs_out, acc_norm, filt) == (64, "zscore", "zerophase"):
        return C.PROCESSED_DIR / "all_X.npy"
    suffix = "" if filt == "zerophase" else f"_{filt}"
    return C.PROCESSED_DIR / f"all_X_{fs_out}hz_{acc_norm}{suffix}.npy"


def build_all(subjects=None, fs_out: int = 64, acc_norm: str = "zscore",
              filt: str = "zerophase") -> dict[str, np.ndarray]:
    """Windows for every subject, with subject ID and metadata per window.

    Used for leave-one-subject-out evaluation, so no train/val split here.
    """
    subjects = subjects or C.ALL_SUBJECTS
    parts = {"X": [], "y": [], "subject": [], "activity": [], "scale": []}
    for s in subjects:
        Xs, ys, meta = windows_for_subject(s, return_meta=True, fs_out=fs_out,
                                           acc_norm=acc_norm, filt=filt)
        parts["X"].append(Xs)
        parts["y"].append(ys)
        parts["subject"].append(np.full(len(ys), s, dtype=np.int8))
        parts["activity"].append(meta["activity"])
        parts["scale"].append(meta["scale"])
        print(f"  subject S{s}: {len(ys)} windows, HR {ys.min():.0f}-{ys.max():.0f} bpm")
    return {k: np.concatenate(v) for k, v in parts.items()}


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


def main_all(fs_out: int = 64, acc_norm: str = "zscore", filt: str = "zerophase"):
    """All 15 subjects -> X array (see x_path) + all_meta.npz.

    The metadata (labels, subject, activity, scale) does not depend on the input
    option, so one all_meta.npz serves every X file.
    """
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    d = build_all(fs_out=fs_out, acc_norm=acc_norm, filt=filt)
    out = x_path(fs_out, acc_norm, filt)
    np.save(out, d.pop("X"))
    np.savez(C.PROCESSED_DIR / "all_meta.npz", **d)
    print(f"\nSaved {len(d['y'])} windows to {out}")


def main():
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split = build_split()
    out = C.PROCESSED_DIR / "windows.npz"
    np.savez_compressed(out, **split)
    print(f"\nSaved {out}")
    print(f"  X_train {split['X_train'].shape}  y_train {split['y_train'].shape}")
    print(f"  X_val   {split['X_val'].shape}  y_val   {split['y_val'].shape}")


if __name__ == "__main__":
    import sys

    # python -m src.preprocessing --all [--fs 32] [--acc-norm fixed] [--filter causal]
    args = sys.argv[1:]
    if "--all" in args:
        fs = int(args[args.index("--fs") + 1]) if "--fs" in args else 64
        norm = args[args.index("--acc-norm") + 1] if "--acc-norm" in args else "zscore"
        filt = args[args.index("--filter") + 1] if "--filter" in args else "zerophase"
        main_all(fs, norm, filt)
    else:
        main()
