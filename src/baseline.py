"""Non-learned baselines, evaluated per subject on all 15 PPG-DaLiA subjects.

1. Constant: predict the mean HR of the *other* 14 subjects (leave-one-subject-out,
   so the held-out subject's labels are never used).
2. DSP (spectral peak picking), built up step by step so each step's effect is
   measured separately:
     naive     largest PPG spectral peak, 512-point FFT, 30-240 bpm
     +zeropad  4096-point FFT (0.94 bpm bins instead of 7.5 bpm)
     +band     search only 40-200 bpm
     +acc      skip PPG peaks that sit on a strong, periodic accelerometer peak
     +track    search only near the previous estimate (causal, streaming-friendly)
   All parameters are fixed up front (see DSPConfig); nothing is fit to the data,
   so evaluating on every subject involves no leakage.

Run after `python -m src.preprocessing --all`:
    python -m src.baseline
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace

import numpy as np

from . import config as C


def fft_hr_estimate(ppg_window: np.ndarray, fs: int = C.FS_PPG) -> float:
    """Naive dominant-frequency HR (bpm): 512-point FFT, peak in 30-240 bpm.

    Kept unchanged from the original repo as the 'naive' reference.
    """
    n = len(ppg_window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(ppg_window * np.hanning(n)))
    band = (freqs >= C.BP_LOWCUT_HZ) & (freqs <= C.BP_HIGHCUT_HZ)
    peak_freq = freqs[band][np.argmax(mag[band])]
    return float(peak_freq * 60.0)


@dataclass(frozen=True)
class DSPConfig:
    nfft: int = 512               # FFT length (zero-padded if > window length)
    fmin_bpm: float = 30.0        # search band
    fmax_bpm: float = 240.0
    use_acc: bool = False         # reject peaks that coincide with periodic motion
    acc_tol_bpm: float = 6.0      #   "coincide" = within this distance
    acc_concentration: float = 0.3  # ACC counts as periodic if its top peak
                                  #   (+/- acc_tol) holds this share of ACC power
    n_candidates: int = 3         # PPG peaks considered, strongest first
    argmax_only: bool = False     # True = original code: global max in band,
                                  #   even on the band edge (no peak finding)
    track: str = "none"           # "none" | "window" | "soft"
                                  #   window: search only prev +/- track_bpm
                                  #   soft:   among the candidates, take the one
                                  #           closest to prev if within
                                  #           track_bpm, else the strongest
    track_bpm: float = 15.0


def _power_spectrum(x: np.ndarray, nfft: int, fs: int = C.FS_PPG):
    """x: (..., T). Hann window, zero-padded to nfft. Returns freqs_bpm, power."""
    n = x.shape[-1]
    spec = np.fft.rfft(x * np.hanning(n), n=nfft, axis=-1)
    return np.fft.rfftfreq(nfft, d=1.0 / fs) * 60.0, np.abs(spec) ** 2


def _local_peaks(p: np.ndarray) -> np.ndarray:
    """Indices of local maxima, strongest first."""
    idx = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    return idx[np.argsort(p[idx])[::-1]]


def dsp_estimate(ppg: np.ndarray, acc_bp: np.ndarray, cfg: DSPConfig,
                 prev: float | None = None) -> float:
    """HR (bpm) for one window. ppg: (T,), acc_bp: (3, T) band-passed ACC."""
    f, p = _power_spectrum(ppg, cfg.nfft)
    lo, hi = cfg.fmin_bpm, cfg.fmax_bpm
    if cfg.track == "window" and prev is not None:
        lo, hi = max(lo, prev - cfg.track_bpm), min(hi, prev + cfg.track_bpm)
    band = (f >= lo) & (f <= hi)
    fb, pb = f[band], p[band]

    if cfg.argmax_only:
        return float(fb[np.argmax(pb)])
    peaks = _local_peaks(pb)
    if peaks.size == 0:                      # monotonic in band: take the max
        peaks = np.array([int(np.argmax(pb))])
    cands = fb[peaks[: cfg.n_candidates]]

    if cfg.use_acc:
        fa, pa = _power_spectrum(acc_bp, cfg.nfft)
        pa = pa.sum(axis=0)
        aband = (fa >= cfg.fmin_bpm) & (fa <= cfg.fmax_bpm)
        fa, pa = fa[aband], pa[aband]
        top = fa[np.argmax(pa)]
        near = np.abs(fa - top) <= cfg.acc_tol_bpm
        if pa[near].sum() / (pa.sum() + 1e-12) >= cfg.acc_concentration:
            # Periodic motion: drop candidates on the motion peak or its 2x / 0.5x.
            motion = np.array([top, 2 * top, top / 2])
            keep = [c for c in cands
                    if np.min(np.abs(motion - c)) > cfg.acc_tol_bpm]
            if keep:
                cands = np.array(keep)
    if cfg.track == "soft" and prev is not None:
        d = np.abs(cands - prev)
        if d.min() <= cfg.track_bpm:
            return float(cands[np.argmin(d)])
    return float(cands[0])


def dsp_sequence(X: np.ndarray, scale: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    """Run the DSP estimator over one subject's windows, in time order."""
    preds = np.empty(len(X))
    prev = None
    for i in range(len(X)):
        x = np.asarray(X[i], dtype=np.float64)
        acc_bp = x[1:] * scale[i, 1:, None]   # undo z-norm: real motion amplitude
        prev = preds[i] = dsp_estimate(x[0], acc_bp, cfg, prev)
    return preds


_ACC = DSPConfig(nfft=4096, fmin_bpm=40, fmax_bpm=200, use_acc=True)
VARIANTS = {
    "naive": DSPConfig(argmax_only=True),          # identical to fft_hr_estimate
    "+zeropad": DSPConfig(nfft=4096),
    "+band": DSPConfig(nfft=4096, fmin_bpm=40, fmax_bpm=200),
    "+acc": _ACC,
    # First tracking attempt (kept as a reported negative result): a hard
    # +/-15 bpm search window locks onto a wrong frequency and never recovers.
    "+track_window": replace(_ACC, track="window"),
    # Revised after seeing the result above (same data, disclosed as such).
    "+track_soft": replace(_ACC, track="soft"),
}
# Soft tracking without the ACC step, to see what each adds on its own.
VARIANTS["track_soft_no_acc"] = replace(VARIANTS["+track_soft"], use_acc=False)


def main():
    X = np.load(C.PROCESSED_DIR / "all_X.npy", mmap_mode="r")
    meta = np.load(C.PROCESSED_DIR / "all_meta.npz")
    y, subj, scale = meta["y"], meta["subject"], meta["scale"]

    results = {}   # method -> per-subject MAE array
    # Constant baseline, leave-one-subject-out.
    results["constant"] = np.array(
        [np.abs(y[subj == s] - y[subj != s].mean()).mean() for s in C.ALL_SUBJECTS])
    for name, cfg in VARIANTS.items():
        maes = []
        for s in C.ALL_SUBJECTS:
            m = subj == s
            pred = dsp_sequence(X[m], scale[m], cfg)
            maes.append(np.abs(pred - y[m]).mean())
        results[name] = np.array(maes)
        print(f"  {name:<13} done", flush=True)

    print(f"\nPer-subject MAE (bpm), all 15 subjects:")
    print(f"{'method':<14}" + "".join(f"{'S' + str(s):>6}" for s in C.ALL_SUBJECTS)
          + f"{'mean':>8}{'std':>6}")
    for name, maes in results.items():
        print(f"{name:<14}" + "".join(f"{v:6.1f}" for v in maes)
              + f"{maes.mean():8.2f}{maes.std():6.2f}")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / "baselines_per_subject.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"S{s}" for s in C.ALL_SUBJECTS] + ["mean", "std"])
        for name, maes in results.items():
            w.writerow([name] + [round(v, 4) for v in maes]
                       + [round(maes.mean(), 4), round(maes.std(), 4)])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
