---
title: PPG Heart Rate Estimation API
emoji: 🫀
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PPG Heart-Rate Estimation API

Estimates heart rate (bpm) from a wrist **PPG** signal + 3-axis **accelerometer**,
using a small 1D CNN trained on the public
[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) dataset —
*wearable sensor data → actionable health metric*.

- **`POST /predict`** — one 8 s window (PPG 512 @64 Hz + ACC 256×3 @32 Hz) → HR in bpm
- **`GET /health`** — status + model metadata
- **`GET /docs`** — interactive API explorer

Validation MAE ≈ 12 bpm on held-out subjects (naive baseline ≈ 24 bpm). Trained
deliberately small (portfolio, not SOTA). Full project, training pipeline, and the
AWS Lambda production-deploy scripts: see the source repository.
