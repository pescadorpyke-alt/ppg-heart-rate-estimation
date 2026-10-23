# PPG Heart-Rate Estimation — wearable sensor data → actionable health metric

Estimating **heart rate (bpm)** from a wrist **PPG** (photoplethysmography) signal
plus **3-axis accelerometer**, using the public **PPG-DaLiA** dataset. The core
challenge in wrist PPG is *motion artefact*: arm movement corrupts the optical
pulse signal, so a naive HR estimate drifts badly during activity. The
accelerometer gives the model a motion reference to compensate.

This is a **portfolio project**, scoped deliberately: the goal is a defensible,
working end-to-end pipeline (signal processing → PyTorch model → live deployed
API), **not** state-of-the-art accuracy.

> **Status:** being turned into an edge-AI project (branch `edge-ai`).
> Evaluated with **leave-one-subject-out over all 15 subjects**. The shipped
> edge model is a **70 KB INT8 TFLite** network (46,753 params, 1.61 M MACs per
> window) that reaches **9.24 ± 4.83 bpm** MAE **on the streaming pipeline a
> device can actually run** (causal filtering, a reading every 2 s). That is the
> same accuracy as the original CNN (9.47 ± 4.94, offline only) at 7× less
> compute. The best DSP baseline gets 15.10 ± 6.70 and predicting the mean
> 18.57 ± 7.68 (see [Results](#results)).
> The **public API** at
> [pescas-ppg-heart-rate.hf.space](https://pescas-ppg-heart-rate.hf.space/docs)
> still serves the original model, trained on the old 5-subject split.
> Design choices and their trade-offs: [`DECISIONS.md`](DECISIONS.md).

---

## Pipeline architecture

```
Empatica E4 (wrist)                          training                  deploy
┌─────────────────┐   8 s windows,    ┌──────────────┐   best.pt   ┌─────────────────┐
│ PPG  64 Hz      │   band-pass,      │ 1D CNN        │ ─────────▶  │ Docker image    │
│ ACC  32 Hz x3   │ ─ normalize, ───▶ │ (PPG + ACC)   │             │ FastAPI /predict│
└─────────────────┘   subject split   │ → HR (bpm)    │             └────────┬────────┘
                                       └──────────────┘                       ▼
RespiBAN ECG 700 Hz ─▶ HR ground truth (label)            live: HF Spaces  ·  also: AWS Lambda+S3
```

## Project layout

```
ppg-heart-rate-estimation/
├── data/                 # pointer only — raw/processed live on D:\ppg-dalia-hr
├── src/
│   ├── config.py         # paths, signal constants, training defaults
│   ├── preprocessing.py  # load + cache, band-pass, window, normalize, splits
│   ├── model.py          # original CNN + depthwise separable edge CNN
│   ├── profile_model.py  # params, MACs, INT8 KB vs the edge budget
│   ├── report.py         # README tables from results/
│   ├── train.py          # reusable training loop (fit), original fixed split
│   ├── loso.py           # leave-one-subject-out evaluation, 15 folds
│   ├── baseline.py       # constant + step-by-step DSP baselines
│   ├── diagnose_fft.py   # why the naive FFT fails, per error type / activity
│   ├── export.py         # PyTorch -> ONNX -> TFLite FP32/FP16/INT8, per fold
│   ├── quant_errors.py   # per-window INT8 vs FP32 error, INT8 output ceiling
│   ├── benchmark.py      # latency per format/runtime, 1 core / 1 thread
│   ├── stream.py         # streaming (causal) preprocessing, as on a device
│   ├── stream_eval.py    # MAE: offline vs per-window vs streaming input
│   ├── device_demo.py    # simulated device loop: timing, memory, plot
│   ├── train_final.py    # deployable model, all 15 subjects
│   ├── export_final.py   # deployable model -> models/
│   ├── infer.py          # load checkpoint + predict one window
│   └── make_example.py   # dump a real window to api/example_payload.json
├── results/              # small CSV/JSON results produced by the scripts above
├── models/               # deployable model: INT8/FP32 TFLite, ONNX, input spec
├── docs/                 # DEVICE.md (real-device next steps), demo plot
├── api/
│   └── main.py           # FastAPI inference service (/, /predict, /health, /docs)
├── deploy/               # AWS Lambda deploy (Dockerfile, scripts, IAM, trade-off)
├── hf/                   # Hugging Face Space (Dockerfile, README, guide)
├── DECISIONS.md          # every non-trivial choice, alternatives, trade-off
├── requirements.txt
└── README.md
```

## Dataset

[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) — UCI ML
Repository #495, **CC BY 4.0** (free, no registration). 15 subjects performing
everyday activities, recorded with a wrist Empatica E4 and a chest RespiBAN.

- **Inputs (wrist):** BVP/PPG @ 64 Hz, 3-axis ACC @ 32 Hz
- **Label (chest):** HR derived from ECG @ 700 Hz

All **15 subjects** are used (64,697 windows). S6 has only 1.46 h of data: the
dataset readme states its recording is valid only for the first 1.5 h, and
`S6.pkl` already contains just that part.

### Download

1. Download `data.zip` (2.7 GB) from the dataset page above.
2. Extract the per-subject pickles (`S1.pkl` … `S15.pkl`) into `D:\ppg-dalia-hr\raw\`
   (or set `PPG_HEAVY_ROOT` to your chosen location).

## Setup

```powershell
# from project root
py -m venv D:\ppg-dalia-hr\.venv          # venv kept off OneDrive
D:\ppg-dalia-hr\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PyTorch CPU wheels for Python 3.14 (Windows) are available — no GPU required for
this small model.

---

## Preprocessing decisions

All implemented in [`src/preprocessing.py`](src/preprocessing.py); constants in
[`src/config.py`](src/config.py).

- **Inputs:** wrist PPG (BVP @ 64 Hz) + wrist 3-axis accelerometer (@ 32 Hz). The
  accelerometer is included so the model can learn to discount motion artefacts —
  the central difficulty of wrist PPG.
- **Label alignment:** PPG-DaLiA ships HR labels (derived from chest ECG) at one
  value per 2 s, each computed over a sliding **8 s** window. This exactly matches
  the literature-standard windowing, so we adopt it directly: window index *k*
  maps to `BVP[k·128 : k·128+512]` and `ACC[k·64 : k·64+256]`. Verified on real
  data: 9212 s of signal → 4603 labels = `(9212−8)/2 + 1`, an exact match (no
  fabricated or interpolated labels).
- **Windowing:** 8 s windows, 2 s hop ⇒ 75 % overlap. Standard in the PPG-HR
  literature; gives dense supervision and HR updates every 2 s.
- **Band-pass filter:** 4th-order Butterworth, **0.5–4.0 Hz**, zero-phase
  (`sosfiltfilt`). Removes baseline wander (<0.5 Hz) and high-frequency noise while
  keeping the plausible HR band (30–240 bpm). Applied to **both** PPG and ACC, so
  the motion reference lives in the same band as the cardiac signal the model must
  separate it from.
- **Channel alignment:** ACC is linearly upsampled 32→64 Hz so all four channels
  share a 512-sample time axis ⇒ model input shape `(4, 512)`.
- **Normalization:** per-window, per-channel z-score (mean 0, std 1). Makes the
  model robust to per-recording amplitude/DC differences. Verified on output:
  mean ≈ 1e-11, std ≈ 0.998.
- **Split by SUBJECT, never by window.** Overlapping windows from the same person
  are near-duplicates, so a random window split would leak and overstate accuracy.

## Evaluation protocol

**Leave-one-subject-out (LOSO) over all 15 subjects** ([`src/loso.py`](src/loso.py)):
in each of 15 folds, one subject is the test set, the next subject in order is the
validation set (used only for early stopping and choosing the best epoch), and the
other 13 train the model. The test subject is never seen during training or model
selection. The headline number is the **mean ± std of the 15 per-subject MAEs**,
which is also how the dataset paper reports results. Why this protocol, and what
it costs: [`DECISIONS.md`](DECISIONS.md) D3, D4 and D8.

## Results

### Main result: LOSO, all 15 subjects

| Method | Input | Params | MAE, bpm (mean ± std over subjects) |
|---|---|---|---|
| Constant (mean HR of the other 14 subjects) | — | 0 | 18.57 ± 7.68 |
| Naive FFT peak (original code) | PPG | 0 | 18.38 ± 8.09 |
| DSP: + zero-pad, 40–200 bpm band, motion-peak rejection | PPG + ACC | 0 | 15.78 ± 6.28 |
| DSP: + soft tracking | PPG + ACC | 0 | 15.74 ± 5.10 |
| DSP: soft tracking, no motion rejection | PPG | 0 | 15.10 ± 6.70 |
| **1D CNN (4 conv blocks)** | PPG + ACC | 118,945 | **9.47 ± 4.94** |

The DSP baseline was built one step at a time so each step's effect is visible
(full table and the step that made things *worse* in [`DECISIONS.md`](DECISIONS.md)
D5). Per-subject numbers: [`results/`](results/).

### Per subject

MAE in bpm, from [`src/report.py`](src/report.py). "Best epoch" is the epoch
picked on the validation subject, out of the epochs run before early stopping.

| Subject | Constant | Naive FFT | DSP, soft tracking | CNN | CNN best epoch |
|---|---|---|---|---|---|
| S1 | 19.54 | 12.40 | 12.26 | 6.47 | 8 / 13 |
| S2 | 12.70 | 14.76 | 13.87 | 6.57 | 7 / 12 |
| S3 | 11.99 | 17.77 | 11.61 | 5.26 | 5 / 10 |
| S4 | 9.89 | 20.27 | 18.24 | 8.21 | 9 / 14 |
| S5 | 39.25 | 42.33 | 28.29 | 23.88 | 6 / 11 |
| S6 | 32.16 | 25.20 | 21.58 | 13.26 | 11 / 15 |
| S7 | 14.38 | 9.35 | 10.71 | 5.26 | 4 / 9 |
| S8 | 18.19 | 16.16 | 18.10 | 11.91 | 2 / 7 |
| S9 | 16.48 | 24.75 | 20.17 | 10.07 | 5 / 10 |
| S10 | 11.48 | 14.77 | 13.53 | 6.81 | 9 / 14 |
| S11 | 22.25 | 25.34 | 21.02 | 15.46 | 1 / 6 |
| S12 | 21.45 | 13.70 | 9.67 | 10.80 | 5 / 10 |
| S13 | 17.59 | 10.44 | 12.12 | 5.17 | 8 / 13 |
| S14 | 13.89 | 14.41 | 14.08 | 5.43 | 6 / 11 |
| S15 | 17.25 | 13.97 | 10.79 | 7.46 | 5 / 10 |
| **Mean ± std** | 18.57 ± 7.68 | 18.38 ± 8.09 | 15.74 ± 5.10 | **9.47 ± 4.94** | |

- The CNN beats the soft-tracking DSP on 14 of 15 subjects. The exception is
  **S12** (10.80 vs 9.67).
- **S5 dominates the spread.** Without it the CNN's per-subject MAEs run 5.2–15.5.
- No fold's best epoch was the last one allowed (15), so the epoch cap (D8) did
  not cut training short. Validation MAE is noisy from epoch to epoch (often by
  2–3 bpm), and best epochs range from 1 to 11. Total training time: 4.26 h.

### Per activity

Window-weighted MAE in bpm (all 15 test subjects pooled). The Deep PPG column is
copied from Table 11 of that paper, for context.

| Activity | Windows | Naive FFT | CNN (this repo) | Deep PPG ensemble (paper) |
|---|---|---|---|---|
| transition | 17515 | 22.20 | 10.92 | 8.77 |
| sitting | 4569 | 3.69 | 4.87 | 4.93 |
| stairs | 3239 | 37.55 | 21.78 | 16.98 |
| table soccer | 2310 | 29.44 | 12.22 | 12.16 |
| cycling | 3473 | 29.45 | 15.53 | 12.48 |
| driving | 6843 | 12.95 | 6.45 | 4.96 |
| lunch | 13554 | 13.75 | 6.49 | 5.22 |
| walking | 4697 | 27.66 | 12.91 | 9.21 |
| working | 8497 | 9.19 | 5.42 | 3.84 |

- The CNN roughly halves the FFT error during motion (stairs, table soccer,
  cycling, walking), which is the problem this project targets.
- **While sitting, the naive FFT is better than the CNN** (3.69 vs 4.87). With no
  motion, the PPG peak is clean and simple peak-picking is hard to beat.
- The CNN is close to the 60 M-parameter Deep PPG ensemble on sitting and table
  soccer, and furthest behind on stairs, walking and cycling.

### Edge-aware model variants (Phase 2)

Budget per 8 s window ([`DECISIONS.md`](DECISIONS.md) D2): ≤ 5 M MACs, ≤ 256 KB
INT8 weights, ≤ 64 KB INT8 peak activations. Each variant changes one thing from
the one above it. All ran the same 15-fold LOSO, training budget and seed. Sizes
are measured by [`src/profile_model.py`](src/profile_model.py) (BatchNorm folded
into the convs); the table is printed by `python -m src.report --variants`.

| Model | Change | Params | MACs | INT8 weights | Peak act. | Fits | **Val MAE** | Test MAE |
|---|---|---|---|---|---|---|---|---|
| `cnn_v0` | original, 64 Hz | 118,945 | 11.49 M | 115.5 KB | 24.0 KB | no | 8.12 ± 3.76 | 9.47 ± 4.94 |
| `cnn_v0_32hz` | 32 Hz input | 118,945 | 5.75 M | 115.5 KB | 12.0 KB | no | 8.41 ± 4.41 | 8.81 ± 4.18 |
| `ds_32hz` | depthwise separable convs | 46,753 | 1.61 M | 44.5 KB | 12.0 KB | yes | 8.49 ± 4.39 | 9.05 ± 4.33 |
| **`ds_32hz_fixed`** | ACC scaled by 0.1 g, not z-scored | 46,753 | **1.61 M** | 44.5 KB | 12.0 KB | yes | **8.08 ± 4.05** | **9.00 ± 4.55** |
| `ds_wide_32hz_fixed` | wider channels | 110,641 | 3.80 M | 106.2 KB | 18.0 KB | yes | 8.28 ± 4.31 | 9.18 ± 4.79 |

MAE in bpm, mean ± std over the 15 subjects. **The variant is chosen on validation
MAE, never test MAE** (D12), so the test subjects play no part in picking it.

- **Chosen: `ds_32hz_fixed`.** It has the lowest validation MAE, but every gap in
  this table is under 0.5 bpm, which I treat as a tie (one seed per variant). The
  claim is therefore about cost: **7× fewer MACs and 2.5× fewer parameters than
  the original CNN, at the same accuracy.**
- **32 Hz input** halves the work with no measurable loss: the heart-rate band
  ends at 4 Hz, far below the 16 Hz Nyquist limit (D9).
- **Depthwise separable convs** cut MACs another 3.6× with no measurable loss
  (D10). But MACs are not speed: in a training benchmark on this CPU the depthwise
  model was only ~2.5× faster than the original, not 7×.
- **Wider is not better:** spending the budget back on channels (3.80 M MACs) did
  not help.
- **Fixed ACC scaling** lets the model see that the wrist is still. Error at rest
  drops (sitting 5.16 → 3.78, working 5.26 → 4.20 bpm), but rises in transitions
  and table soccer, so overall it is about neutral (D11).

Per-activity MAE (window-weighted, bpm) for the five variants is in
`results/loso_<model>_summary.json`.

### Export and quantization (Phase 3)

`ds_32hz_fixed` exported PyTorch → ONNX → TensorFlow → TFLite
([`src/export.py`](src/export.py), D13, D17). **Every fold's model is converted
and scored on its own held-out subject**, so these are LOSO numbers like the
rest. INT8 calibration uses 500 windows from that fold's training subjects only,
spread evenly over heart rate (D16).

| Format | MAE, bpm (15 subjects) | File size | Median latency, 1 core / 1 thread |
|---|---|---|---|
| PyTorch FP32 (eager) | 9.003 ± 4.549 | — | 1,323 µs |
| ONNX FP32 (ONNX Runtime) | 9.003 ± 4.549 | 183.7 KB | 83 µs |
| TFLite FP32 | 9.003 ± 4.549 | 184.4 KB | 57 µs |
| TFLite FP16 | 9.004 ± 4.549 | 98.2 KB | 57 µs |
| **TFLite INT8** (float in/out) | **9.071 ± 4.565** | **70.6 KB** | 83 µs |
| TFLite INT8, int8 in/out (MCU style) | 9.071 ± 4.565 | 70.4 KB | 99 µs |

Latency: TFLite with XNNPACK, process pinned to one core, one thread, batch 1,
100 warm-up + 2,000 timed runs ([`src/benchmark.py`](src/benchmark.py), D14). This
is a 3.6 GHz desktop core (Intel i3-10100), **not** a wearable: the numbers
compare formats, they are not device latency. All variants are far inside the
200 ms budget, and the INT8 file is well inside 256 KB.

- **INT8 costs 0.07 bpm** of MAE, far under the 0.5 bpm threshold that was
  committed *before* the results (D15), so quantization-aware training was not
  needed.
- **INT8 is 2.6× smaller than FP32 but not faster on this CPU** (83 vs 57 µs with
  XNNPACK). On a desktop with fast float SIMD and a tiny model, INT8 buys size,
  not speed.
- **A hidden INT8 ceiling, found and fixed:** with random calibration, each fold's
  INT8 model could not output more than 145–174 bpm, because random calibration
  windows rarely have a high heart rate. Averages hid it (+0.04 bpm); single
  cycling windows were up to 25 bpm off. Calibrating evenly across heart rate
  raised the ceiling to 167–195 bpm and cut the worst window to 10.3 bpm (D16).
- **Exported FP32 matches PyTorch** to within 0.0001 bpm on every window.

Every runtime and kernel set, both export layouts: `results/benchmark_ds_32hz_fixed.csv`.
Two anomalies on this Windows machine are kept in that file rather than hidden
(D18): the `ai_edge_litert` 2.1.2 interpreter never applied XNNPACK and ran INT8
in ~12 ms, and TFLite's built-in INT8 kernels (3.9 ms) were slower than its plain
reference kernels (1.8 ms).

### Running it like a device (Phase 4, no hardware)

A wearable cannot filter with zero phase, the way training did: that needs future
samples. It filters each sample as it arrives (causal) and produces a heart rate
every 2 s. [`src/stream.py`](src/stream.py) implements that: a causal band-pass
with state carried between 2 s chunks and 8 s ring buffers, 5,248 bytes of
state. Measured with each fold's INT8 model on its held-out subject
([`src/stream_eval.py`](src/stream_eval.py)):

| Preprocessing path | Model trained on zero-phase input | **Model trained on causal input** |
|---|---|---|
| Offline (as in training) | 9.07 ± 4.57 | 9.24 ± 4.83 |
| Per window, zero-phase (the current API) | 9.18 ± 4.56 | 11.75 ± 5.94 |
| **Streaming, causal (a device)** | 11.47 ± 4.88 | **9.24 ± 4.83** |

INT8 MAE in bpm, 15 subjects. **The model that looked best offline lost 2.4 bpm
on the pipeline a device can actually run**, worse on all 15 subjects. Under a
rule committed before the measurement (D19), it was retrained on causally
filtered input, which is bit-identical to what the streaming code produces. On the
device pipeline that brings it back to 9.24 bpm (D21). Lesson: evaluate the
pipeline you will deploy, not just the model.

**Simulated device loop** ([`src/device_demo.py`](src/device_demo.py)): a held-out
subject replayed in 2 s sensor chunks → streaming preprocessing → fully-INT8
inference, pinned to one core:

![Simulated device loop on held-out subject S7](docs/device_demo_ds_32hz_fixed_causal_S7.png)

S7 is an easier-than-average subject (5.0 bpm here; the mean over all 15 is 9.24).
Per 2 s window: preprocessing 177 µs + inference 114 µs (median) on a desktop
core, so the CPU is busy 0.015 % of the time. Memory: 70.4 KB model, activations
between 12.0 and 55.3 KB (lower and upper bound), 5.1 KB of preprocessing state.
These are desktop numbers. What a microcontroller or Android build would need,
and how to measure it there, is in [`docs/DEVICE.md`](docs/DEVICE.md).

**Deployable model:** [`models/`](models/) holds the final model (the same recipe
trained on all 15 subjects, D20) as INT8 / FP32 TFLite and ONNX, plus
`ppg_hr.json` with the exact input spec and quantization parameters. Its accuracy
estimate is the LOSO result above; no data was left over to measure it directly.

### Comparison with published results on PPG-DaLiA

All numbers below are copied from the cited papers, not re-run. MAE in bpm,
mean ± std over subjects where given.

| Method | MAE (bpm) | Model size | Source |
|---|---|---|---|
| SpaMa (classical) | 15.56 ± 7.5 | — | Deep PPG, Table 4 |
| SpaMaPlus (classical) | 11.06 ± 4.8 | — | Deep PPG, Table 4 |
| Schaeck2017 (classical) | 20.45 ± 7.1 | — | Deep PPG, Table 4 |
| Deep PPG CNN, single model, averaged | 8.82 ± 3.8 | 8.5 M params | Deep PPG, Table 10 |
| Deep PPG CNN ensemble (7 models) | 7.65 ± 4.2 | ≈ 60 M params | Deep PPG, Tables 10, 12 |
| Deep PPG resource-constrained CNN | 9.99 ± 5.9 | 26 K params | Deep PPG, Sec. 5 |
| Q-PPG-L, INT8 on STM32WB55 | 4.41 | 412 kB | Q-PPG, Table IV |
| Q-PPG-S, INT4 on STM32WB55 | 7.73 | 1.9 kB | Q-PPG, Table IV |
| This repo: original 1D CNN, FP32 | 9.47 ± 4.94 | 118,945 params | this README |
| This repo: `ds_32hz_fixed`, FP32 | 9.00 ± 4.55 | 46,753 params, 1.61 M MACs | this README |
| This repo: `ds_32hz_fixed`, INT8 TFLite, offline input | 9.07 ± 4.57 | 70.6 KB file | this README |
| **This repo: `ds_32hz_fixed_causal`, INT8, streaming input** | **9.24 ± 4.83** | 70.4 KB file | this README |

Sources:
- **Deep PPG:** A. Reiss, I. Indlekofer, P. Schmidt, K. Van Laerhoven,
  *Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks*,
  Sensors 19(14):3079, 2019 — the dataset paper.
  [Open-access full text (PMC6679242)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679242/)
- **Q-PPG:** A. Burrello et al., *Q-PPG: Energy-Efficient PPG-based Heart Rate
  Monitoring on Wearable Devices*, IEEE TBioCAS 15(6), 2021.
  [arXiv:2203.14907](https://arxiv.org/abs/2203.14907)

**Where this repo sits:** the edge model (47 K parameters, 9.00 ± 4.55 bpm) and
the original CNN (119 K, 9.47 ± 4.94) both fall between Deep PPG's 26 K-parameter
resource-constrained CNN (9.99 ± 5.9) and its 8.5 M single CNN (8.82 ± 3.8). Both
are clearly behind the best published results; Q-PPG shows that much smaller
models can do better, with architecture search and post-processing across
windows that this repo does not use.

How comparable these are:
- **Deep PPG** uses leave-one-session-out with one session per subject, i.e. the
  same per-subject testing as here; it does not state which subject was used for
  validation on PPG-DaLiA.
- **Q-PPG** groups the 15 subjects into 4 random folds and tests each subject
  once, with validation taken from the held-out fold. Close to this protocol but
  not identical. Its best-listed result (3.61 bpm) fine-tunes on data from the
  test subject, so it is left out here.
- Q-PPG's models also use post-processing across windows; this repo's CNN
  predicts each window independently.

### Why the naive FFT fails

[`src/diagnose_fft.py`](src/diagnose_fft.py) classifies every naive-FFT error
(all 64,697 windows):

| Where the FFT pick landed | % of windows | % of total error |
|---|---|---|
| Within 10 bpm of the true HR | 60.0 | 9.7 |
| On the dominant accelerometer frequency | 8.0 | 19.6 |
| At half the true HR | 6.0 | 16.3 |
| Below 45 bpm (filter edge) | 2.4 | 8.2 |
| At twice the true HR | 0.3 | 1.2 |
| None of the above | 23.4 | 45.0 |

It is near-perfect while sitting (3.7 bpm MAE) and worst on stairs (37.6 bpm),
walking (27.7) and cycling (29.5). Motion corrupts the spectrum in more ways than
one clean motion peak, which is why simply removing the accelerometer peak helps
only a little.

### The original fixed split, and why it misled

The first version of this repo trained on S1, S2, S3, S6, S7 and validated on S4,
S5, with no separate test set. Re-running that original code unchanged reproduces
its numbers exactly:

| Method (old split) | Val MAE (bpm) |
|---|---|
| Naive FFT peak | 31.39 |
| Constant (train mean) | 23.94 |
| 1D CNN, 118,945 params (best of 33 epochs, epoch 21) | 12.09 |

Two things were wrong with drawing conclusions from it:
1. **S5 is the hardest subject** for every method run here, the CNN included
   (23.88 bpm), and for
   SpaMa, SpaMaPlus and the CNN in the Deep PPG paper (Table 10). Its mean HR is
   125.8 bpm; 13 of the other subjects average 71–103 (S6: 119.5).
   With S5 as half the validation set, the naive FFT looked much worse than
   guessing the mean (31.4 vs 23.9). Over all 15 subjects they are level
   (18.38 vs 18.57).
2. **The validation set also picked the best epoch**, so 12.09 bpm was optimistic.

An earlier, smaller CNN (44.7k params) was reported at 17.6 bpm on this split. Its
code is not in the repo, so that number could not be re-run.

### Reproduce

```bash
python -m src.preprocessing --all     # 64,697 windows from the 15 pickles
python -m src.baseline                # constant + DSP steps, per subject
python -m src.diagnose_fft            # FFT error breakdown
python -m src.loso --model cnn_v0     # 15-fold LOSO CNN (~5 h on a 2-core CPU)

# Phase 2: 32 Hz inputs, then the edge variants (add --resume after a crash)
python -m src.preprocessing --all --fs 32
python -m src.preprocessing --all --fs 32 --acc-norm fixed
python -m src.profile_model           # params / MACs / KB vs the budget
python -m src.loso --model ds_32hz_fixed
python -m src.report --variants       # Phase 2 comparison table

# Phase 3, in the export environment (TensorFlow 2.21, onnx2tf, ONNX Runtime)
python -m src.export --model ds_32hz_fixed            # 15 folds x FP32/FP16/INT8
python -m src.quant_errors hr_stratified_2d           # per-window INT8 error, ceilings
python -m src.benchmark --model ds_32hz_fixed         # latency, 1 core / 1 thread

# Phase 4: causal input, streaming evaluation, device loop, final model
python -m src.preprocessing --all --fs 32 --acc-norm fixed --filter causal
python -m src.loso --model ds_32hz_fixed_causal                 # training env
python -m src.export --model ds_32hz_fixed_causal               # export env
python -m src.stream_eval ds_32hz_fixed_causal                  # export env
python -m src.device_demo --model ds_32hz_fixed_causal --subject 7
python -m src.train_final && python -m src.export_final        # -> models/
```

## API usage

FastAPI service in [`api/main.py`](api/main.py). The API takes a **raw**
(unfiltered) 8 s window straight from the wrist sensor and runs the exact same
preprocessing used in training (band-pass → ACC upsample → per-window z-norm),
so there is no train/serve skew.

Run locally:

```powershell
D:\ppg-dalia-hr\.venv\Scripts\Activate.ps1
uvicorn api.main:app --port 8000          # from project root
```

Generate a real example window (and its ground-truth HR) with
`python -m src.make_example`, then:

```bash
# Health + loaded-model metadata
curl http://127.0.0.1:8000/health
# -> {"status":"ok","model":{"val_mae":12.09,"epoch":21,"window_len":512}}

# Predict from a raw window (api/example_payload.json; ground truth 98.5 bpm)
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d @api/example_payload.json
# -> {"heart_rate_bpm":107.1,"model_val_mae_bpm":12.09}
```

**Request schema** (`POST /predict`):

| field | type | shape | meaning |
|-------|------|-------|---------|
| `ppg` | `float[]` | `512` | raw BVP/PPG @ 64 Hz (8 s) |
| `acc` | `float[][]` | `256 × 3` | raw 3-axis ACC @ 32 Hz (8 s), rows `[x,y,z]` |

Input is validated (exact lengths / shape); wrong sizes return **HTTP 422** with a
descriptive message. Interactive docs at `/docs`.

## Deployment

### Live demo 🟢

The API is deployed and public on **Hugging Face Spaces** (free tier, no card,
nothing to maintain):

- **App:** https://huggingface.co/spaces/Pescas/ppg-heart-rate
- **API host:** `https://pescas-ppg-heart-rate.hf.space`
- **Interactive docs:** https://pescas-ppg-heart-rate.hf.space/docs

Verified live (ground-truth HR for this window = 98.5 bpm):

```bash
$ curl https://pescas-ppg-heart-rate.hf.space/health
{"status":"ok","model":{"val_mae":12.09,"epoch":21,"window_len":512}}

$ curl -X POST https://pescas-ppg-heart-rate.hf.space/predict \
    -H "Content-Type: application/json" -d @api/example_payload.json
{"heart_rate_bpm":107.1,"model_val_mae_bpm":12.09}
```

> Free Spaces sleep after ~48 h idle and wake on the next request (brief cold start).

### AWS (production architecture)

The repo also ships a complete **AWS Lambda** (container image) + **Function URL**
deployment — chosen over EC2 because it scales to zero (no idle cost, no
maintenance; full **EC2-vs-Lambda trade-off** in [`deploy/README.md`](deploy/README.md)).
The same FastAPI app runs unchanged on Lambda via [Mangum](https://mangum.io/)
([`deploy/lambda_handler.py`](deploy/lambda_handler.py)), with the model loaded
from **S3** on cold start. Idempotent provision + teardown:

```powershell
powershell -ExecutionPolicy Bypass -File deploy\deploy.ps1     # provision + deploy
powershell -ExecutionPolicy Bypass -File deploy\teardown.ps1   # remove everything
```

Not currently provisioned (AWS requires a payment method); the live demo above
runs the identical image on Hugging Face instead. Reproduce the HF deploy with
[`hf/HUGGINGFACE.md`](hf/HUGGINGFACE.md).

## Limitations

Stated honestly — this is a scoped portfolio project, not a research result.

- **15 subjects is still a small population.** LOSO tests every subject, but the
  per-subject spread is large, so a new person could land far from the mean.
- **Capped training.** Each LOSO fold trains for at most 15 epochs (D8 in
  `DECISIONS.md`) to keep 15 folds affordable on a CPU; learning curves are saved
  in `results/` to check that the cap does not cost accuracy.
- **Not state-of-the-art.** See the comparison table above: the best published
  PPG-DaLiA results are clearly better. The goal here is a small model that fits
  an edge budget, not the lowest MAE.
- **DSP baseline adjusted once on the evaluation data.** The tracking step was
  revised after its first version made results worse (disclosed in D5).
- **Minimal hyperparameter tuning** of the CNN: no learning-rate or architecture
  sweep yet (architecture work is Phase 2).
- **The deployed API still serves the old-split model**, not a LOSO-evaluated one.
  The shipped causal model needs the API's preprocessing switched to the
  causal streaming path first (D21).
- **No real device yet.** Device-side latency, memory and energy are not
  measured; see `docs/DEVICE.md` for what to measure and how.
- **Train/serve filtering nuance.** Training band-passes each full recording then
  slices windows; the API band-passes the single submitted window. Zero-phase
  filter edge effects make these slightly different — small relative to the model's
  MAE, but real (documented in `src/preprocessing.preprocess_window`).
- **Per-window prediction only.** No temporal smoothing across consecutive windows,
  which real wearables use to stabilise the HR trace.
- **CPU-trained**, ~1 min — fine for this size, but caps how far the model was pushed.
