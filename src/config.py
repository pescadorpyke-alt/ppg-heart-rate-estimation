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

# --- PPG-DaLiA signal constants ---------------------------------------------
# Wrist device = Empatica E4 (our model inputs).
FS_PPG = 64          # BVP/PPG sampling rate (Hz)
FS_ACC = 32          # wrist accelerometer sampling rate (Hz)
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
