"""Central configuration: paths, signal constants, training defaults.

Heavy artifacts (raw dataset, processed windows, model checkpoints) live OUTSIDE
the OneDrive-synced project folder, on D:\\ppg-dalia-hr, to avoid syncing
gigabytes. Override any path with an environment variable if needed.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Heavy storage root (raw data / processed / checkpoints). Not in OneDrive.
HEAVY_ROOT = Path(os.environ.get("PPG_HEAVY_ROOT", r"D:\ppg-dalia-hr"))
RAW_DIR = HEAVY_ROOT / "raw"            # extracted PPG-DaLiA pickles (S1.pkl ...)
PROCESSED_DIR = HEAVY_ROOT / "processed"
CHECKPOINT_DIR = HEAVY_ROOT / "checkpoints"
CACHE_DIR = HEAVY_ROOT / "cache"        # wrist-only copy of each pickle (fast reload)
RESULTS_DIR = PROJECT_ROOT / "results"  # small CSV/JSON results, committed to git

# --- PPG-DaLiA signal constants ---------------------------------------------
# Wrist device = Empatica E4 (our model inputs).
FS_PPG = 64          # BVP/PPG sampling rate (Hz)
FS_ACC = 32          # wrist accelerometer sampling rate (Hz)
FS_ACTIVITY = 4      # activity-label sampling rate in the pickles (Hz)
ALL_SUBJECTS = list(range(1, 16))   # S1 ... S15

# Activity IDs from the dataset readme (0 = transition between activities).
ACTIVITY_NAMES = {
    0: "transition", 1: "sitting", 2: "stairs", 3: "table_soccer", 4: "cycling",
    5: "driving", 6: "lunch", 7: "walking", 8: "working",
}
# Ground-truth HR labels are provided at one value per 2 s (sliding 8 s / 2 s).
LABEL_PERIOD_S = 2

# --- Windowing (PPG-HR literature standard) ---------------------------------
WINDOW_S = 8                       # analysis window length (seconds)
WINDOW_STRIDE_S = 2               # hop between consecutive windows (overlap)
PPG_WINDOW_LEN = WINDOW_S * FS_PPG     # 512 samples
ACC_WINDOW_LEN = WINDOW_S * FS_ACC     # 256 samples

# --- Band-pass filter (remove baseline wander + high-freq / motion noise) ---
# Human HR ~ 30-220 bpm => 0.5-3.67 Hz. We keep a slightly wider band.
BP_LOWCUT_HZ = 0.5
BP_HIGHCUT_HZ = 4.0
BP_ORDER = 4

# --- Model input options (Phase 2) -------------------------------------------
# Used by acc_norm="fixed": ACC is divided by this constant instead of being
# z-scored per window, so motion strength is kept (per-window z-scoring makes a
# still wrist look as "active" as a moving one).
# Units: the pickles store wrist ACC in g (|ACC| ~= 1.01 at rest, checked); the
# 1/64 g in the dataset readme applies to the raw E4 CSVs, not the pickles.
# 0.1 g puts band-passed arm motion (~0.18 g std while walking) near unit scale,
# like the z-scored PPG channel, while a still wrist stays near 0.
ACC_SCALE = 0.1

# --- Subject split (split by SUBJECT to avoid leakage) ----------------------
# Small subset for fast, defensible training (portfolio, not SOTA).
# 5 train + 2 val subjects (split by SUBJECT to avoid leakage).
TRAIN_SUBJECTS = [1, 2, 3, 6, 7]
VAL_SUBJECTS = [4, 5]

# --- Training defaults -------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

# --- Inference API (D23) ------------------------------------------------------
# Seconds of raw signal the client sends before the 8 s window, so the causal
# filter can settle. Minimum chosen by the rule in DECISIONS.md D23 from
# results/context_eval_ds_32hz_fixed_causal_summary.json (2 s: within 0.1 bpm
# of full streaming; 0 s was 0.13 bpm worse). The maximum only bounds payload size.
API_MIN_CONTEXT_S = 2
API_MAX_CONTEXT_S = 60
# LOSO MAE of the served recipe (FP32, streaming input, mean over 15 subjects),
# from results/stream_eval_ds_32hz_fixed_causal_summary.json ("streaming_fp32").
API_MODEL_LOSO_MAE_BPM = 9.22
