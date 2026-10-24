# Model card: wrist PPG heart-rate estimator (`models/ppg_hr_int8_full.tflite`)

**Not a medical device.** This model is a portfolio / research artefact. It has
not been clinically validated and must not be used for diagnosis, treatment or
any safety-relevant decision.

## Model details

| | |
|---|---|
| Task | Regress heart rate (bpm) from one 8 s window of wrist PPG + 3-axis accelerometer, one estimate every 2 s |
| Architecture | 1-D CNN, first block standard conv, blocks 2–4 depthwise separable; global average pool; 2-layer head (`src/model.py: DSHeartRateCNN`) |
| Size | 46,753 parameters; 1.61 M MACs per window; 70.4 KB as fully-INT8 TFLite |
| Input | `(1, 1, 256, 4)` NHWC, int8 (scale / zero point in `models/ppg_hr.json`): PPG z-scored per window, ACC x/y/z in units of 0.1 g, 32 Hz |
| Preprocessing | Causal 4th-order Butterworth band-pass 0.5–4 Hz per sensor, state kept across windows (`src/stream.py`); this exact pipeline is what training used (DECISIONS.md D21) |
| Output | Heart rate in bpm (int8, dequantize with the output scale). Maximum representable output: 177.4 bpm |
| Training | PPG-DaLiA, all 15 subjects, 6 epochs, Adam, L1 loss (D20). Recipe chosen and evaluated with leave-one-subject-out (D3, D12) |
| Other formats | `models/ppg_hr_int8.tflite` (float in/out), `ppg_hr_fp32.tflite`, `ppg_hr.onnx` |

## Intended use

- Demonstrating an edge-AI workflow end to end: evaluation, model design under
  a compute budget, quantization, and a streaming on-device pipeline.
- Research and education on PPG heart-rate estimation under motion.

**Out of scope:** medical or clinical use; arrhythmia or any rhythm analysis
(the model outputs an average rate only); use on sensors other than a wrist
Empatica E4-like PPG + accelerometer without re-validation; use on people or
conditions not represented in the data below.

## Data

[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) (CC BY 4.0;
Reiss et al., Sensors 2019, [PMC6679242](https://pmc.ncbi.nlm.nih.gov/articles/PMC6679242/)):
15 subjects wearing an Empatica E4 on the non-dominant wrist during about 2.5 h
of daily activities (sitting, stairs, table soccer, cycling, driving, lunch,
walking, office work, and transitions). Reference heart rate comes from chest ECG
with manually corrected R-peaks, one label per 8 s window every 2 s: 64,697
windows in total.

Who is in it (from each subject's questionnaire): 7 male / 8 female, aged 21–55,
**Fitzpatrick skin types 2–4 only** (type 2: 1 subject, type 3: 11, type 4: 3).

## Evaluation

All numbers: leave-one-subject-out over the 15 subjects (each subject is tested
by a model that never saw them), INT8 TFLite, **streaming input as a device
would compute it**. Mean ± std over subjects. Source:
`results/stream_eval_ds_32hz_fixed_causal*`.

**Overall MAE: 9.24 ± 4.83 bpm** (median subject 6.56 bpm).

Per subject:

| Subject | Sex | Age | Skin type | MAE (bpm) |
|---|---|---|---|---|
| S1 | m | 34 | 3 | 6.57 |
| S2 | m | 28 | 3 | 6.54 |
| S3 | m | 25 | 3 | 5.15 |
| S4 | m | 25 | 4 | 6.56 |
| S5 | f | 21 | 3 | **22.30** |
| S6 | f | 37 | 3 | 13.17 |
| S7 | f | 21 | 3 | 5.02 |
| S8 | m | 43 | 3 | 14.43 |
| S9 | f | 28 | 4 | 8.59 |
| S10 | f | 55 | 4 | 6.19 |
| S11 | f | 24 | 3 | 13.72 |
| S12 | m | 43 | 3 | 13.01 |
| S13 | f | 21 | 3 | 5.50 |
| S14 | f | 26 | 3 | 5.71 |
| S15 | m | 28 | 2 | 6.08 |

Group averages (too few subjects per group to support any conclusion; listed so
they are not hidden): male 8.33 (n = 7), female 10.02 (n = 8); skin type 2: 6.08
(n = 1), type 3: 10.10 (n = 11), type 4: 7.11 (n = 3).

Per activity (window-weighted MAE, bpm): sitting 3.84, working 4.40, lunch 6.21,
driving 6.53, transition 11.47, walking 12.67, cycling 12.78, table soccer 13.42,
stairs 20.90.

For comparison, from the dataset paper (same per-subject protocol): classical
SpaMaPlus 11.06 ± 4.8, a 26 K-parameter CNN 9.99 ± 5.9, an 8.5 M-parameter CNN
8.82 ± 3.8 and a 7-model ensemble 7.65 ± 4.2 bpm. See the README for the full
table and caveats.

## Limitations and risks

- **Large differences between people.** One subject (S5, mean heart rate 126
  bpm) has 22 bpm error; four more are above 13 bpm. The average hides this.
- **Motion still dominates the error:** stairs, table soccer, cycling and walking
  are 3.3–5.4× worse than sitting.
- **Skin tone coverage is narrow** (Fitzpatrick 2–4). Optical PPG depends on
  skin pigmentation; the model has not been tested on very light or darker skin
  (types 1, 5, 6).
- **Small population:** 15 adults, one device type, one data-collection site.
- **High heart rates:** the INT8 model cannot output above 177.4 bpm, and the
  model under-predicts very high heart rates even in FP32 (D16).
- **No temporal smoothing** across windows; consecutive readings can jump.
- **Preprocessing must match exactly.** Feeding zero-phase-filtered windows (as
  the current public API does) to this model raises MAE to 11.75 bpm (D21).
- **Evaluation used the same 15 subjects** that the final model was trained on;
  the LOSO number is an estimate for this recipe, not a measurement of this file.
- **No real device yet:** latency and memory were measured on a desktop CPU only
  (docs/DEVICE.md).
