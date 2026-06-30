# PPG Heart-Rate Estimation — wearable sensor data → actionable health metric

Estimating **heart rate (bpm)** from a wrist **PPG** (photoplethysmography) signal
plus **3-axis accelerometer**, using the public **PPG-DaLiA** dataset. The core
challenge in wrist PPG is *motion artefact*: arm movement corrupts the optical
pulse signal, so a naive HR estimate drifts badly during activity. The
accelerometer gives the model a motion reference to compensate.

This is a **portfolio project**, scoped deliberately: the goal is a defensible,
working end-to-end pipeline (signal processing → PyTorch model → live deployed
API), **not** state-of-the-art accuracy.

> **Status:** complete and live. Pipeline → model (val MAE **12.1 bpm**, vs 23.9
> naive baseline) → **public API** at
> [pescas-ppg-heart-rate.hf.space](https://pescas-ppg-heart-rate.hf.space/docs).

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
ppg-dalia-hr/
├── data/            # pointer only — raw/processed live on D:\ppg-dalia-hr
├── src/
│   ├── config.py        # paths, signal constants, training defaults
│   ├── preprocessing.py  # load, band-pass, window, normalize, subject split
│   ├── model.py          # 1D CNN regressor
│   ├── train.py          # training loop, MAE, checkpointing
│   └── infer.py          # load checkpoint + predict one window
│   ├── baseline.py       # naive baselines (constant + FFT peak)
│   └── make_example.py   # dump a real window to api/example_payload.json
├── api/
│   └── main.py           # FastAPI inference service (/, /predict, /health, /docs)
├── deploy/               # AWS Lambda deploy (Dockerfile, scripts, IAM, trade-off)
├── hf/                   # Hugging Face Space (Dockerfile, README, guide)
├── notebooks/            # exploration
├── requirements.txt
└── README.md
```

## Dataset

[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) — UCI ML
Repository #495, **CC BY 4.0** (free, no registration). 15 subjects performing
everyday activities, recorded with a wrist Empatica E4 and a chest RespiBAN.

- **Inputs (wrist):** BVP/PPG @ 64 Hz, 3-axis ACC @ 32 Hz
- **Label (chest):** HR derived from ECG @ 700 Hz

We use a **subset of subjects** (not all 15) to keep training fast.

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
- **Train/val split — by SUBJECT, not by window.** Train = S1, S2, S3, S6, S7
  (20,359 windows); Val = S4, S5 (9,221 windows). A random window split would leak
  near-identical overlapping windows across sets and massively overstate accuracy.
  Subject split also exposes real distribution shift (e.g. val subject S5 reaches
  187 bpm, near the top of the training range) — the honest, harder setting.

**Naive baseline:** predicting the constant train-mean HR for every val window
gives **23.9 bpm MAE** (with the 5-subject train set) — the number the model
must beat.

## Results

Validation = subjects S4, S5 (held out entirely). Train = S1, S2, S3, S6, S7.

| Model | Val MAE (bpm) |
|-------|---------------|
| FFT peak-frequency baseline (classic DSP, PPG only) | 31.4 |
| Constant baseline (predict train-mean HR) | 23.9 |
| 1D CNN — early version (44.7k params, 3 conv blocks) | 17.6 |
| **1D CNN (118.9k params, 4 conv blocks, PPG + ACC)** | **12.1** |

**The key result is the FFT baseline being *worse* than guessing the mean.** The
classic signal-processing approach — take the dominant frequency of the PPG
spectrum — fails because, during everyday motion, the strongest spectral peak is
often a *motion harmonic*, not the heartbeat. This is precisely the problem the
project targets: feeding the **accelerometer** alongside the PPG lets the CNN
learn to discount motion, nearly halving the constant-baseline error.

The model is **not** state-of-the-art (published PPG-DaLiA results reach ~4–8 bpm
MAE); see *Limitations*. Scaling the first CNN up also cut **train** MAE from ~10
to ~7.3 bpm — the small model was underfitting, not over-regularized.

Reproduce the baselines: `python -m src.baseline`.

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

- **Small subject set (7 of 15):** train = 5 subjects, val = 2. Not representative
  of population variance; results would shift with different subjects.
- **Validation used for model selection.** With only 2 held-out subjects there is
  no separate test set, so the 12.1 bpm figure is mildly optimistic (the best
  epoch was chosen on this same val set). A cleaner setup would use k-fold
  leave-subjects-out cross-validation.
- **Not state-of-the-art.** Published PPG-DaLiA results reach ~4–8 bpm MAE with
  larger models, more data, and motion-compensation algorithms. 12 bpm is a solid
  *small-model* result, no more.
- **Minimal hyperparameter tuning.** One deliberate, cheap tuning pass (model size
  + subjects); no learning-rate / architecture sweep.
- **Train/serve filtering nuance.** Training band-passes each full recording then
  slices windows; the API band-passes the single submitted window. Zero-phase
  filter edge effects make these slightly different — small relative to the model's
  MAE, but real (documented in `src/preprocessing.preprocess_window`).
- **Per-window prediction only.** No temporal smoothing across consecutive windows,
  which real wearables use to stabilise the HR trace.
- **CPU-trained**, ~1 min — fine for this size, but caps how far the model was pushed.
