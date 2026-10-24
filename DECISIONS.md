# Design decisions

One entry per non-trivial choice: what was chosen, what else was considered, and
the trade-off. Newest entries at the bottom.

---

## D1. Target hardware: none yet, benchmark on a constrained CPU setup

**Chosen:** No physical device for now. The model is exported to TFLite/LiteRT and
benchmarked on a desktop CPU in a deliberately constrained setup (one thread,
interpreter only, warm-up runs, median and p95 latency over many windows). Real
on-device deployment is a documented next step.

**Alternatives:**
- *Android phone:* realistic for a wearable companion app, but needs a device and
  a Kotlin app before any result exists.
- *Microcontroller (ESP32-S3, Arduino Nano 33 BLE Sense):* the most "embedded"
  option, but needs the board, a toolchain and firmware work.

**Trade-off:** I lose a real-device latency number, which is the most convincing
evidence for an edge project. In return, every number I do report can be
reproduced by anyone with a laptop, and the model is designed against a fixed
budget (D2) so it is ready to move to a device later. A desktop CPU is much
faster than a microcontroller, so its latency is *not* presented as device
latency.

---

## D2. Edge budget per 8 s window

**Chosen** (approved 2026-10-21):

| Resource | Budget | Why |
|---|---|---|
| Flash: INT8 weights | ≤ 256 KB | Leaves room for application code on a small device |
| RAM: peak activations, INT8 | ≤ 64 KB | Fits a microcontroller-class tensor arena |
| Compute | ≤ 5 M MACs | Hardware-independent, so it carries over to any target |
| Latency, 1 CPU thread | ≤ 200 ms | Must finish well inside the 2 s gap between windows |

**Starting point** (measured by running `src/model.py` with forward hooks on a
`(1, 4, 512)` input; INT8 sizes assume 1 byte per value):

| Resource | Current CNN | Within budget? |
|---|---|---|
| Parameters | 118,945 | — |
| INT8 weights | 116.2 KB | yes |
| Peak activations, INT8 | 24.0 KB | yes |
| MACs per window | 11,485,312 | **no** (about 2.3× over) |

**Alternatives:**
- *Latency only:* simple, but a fast desktop makes almost any model look fine.
- *No budget, just "make it smaller":* gives no way to say when the job is done.

**Trade-off:** MACs are a proxy. They ignore memory traffic and how well each
operator is optimised, so two models with the same MACs can run at different
speeds. Pairing a MAC cap with a measured latency covers both sides. Flash and RAM
already fit; compute is the constraint that drives the Phase 2 model work.

---

## D3. Evaluation: leave-one-subject-out over all 15 subjects

**Chosen:** 15 folds. In each fold one subject is the test set, the *next*
subject in order (S15 wraps to S1) is the validation set, used only for early
stopping and picking the best epoch, and the other 13 subjects are for training.
I report the MAE of each test subject, then the mean ± standard deviation over
the 15 subjects. Windows are the dataset's own: 8 s long, every 2 s.

**Alternatives:**
- *The old fixed split* (train 5 subjects, validate on 2, no test set): fast, but
  the reported number was also the number used to pick the model, and it depended
  heavily on which 2 subjects were chosen.
- *The Q-PPG protocol:* subjects grouped into 4 random folds, with the held-out
  fold split into a test subject and validation subjects. Close to mine, but the
  random grouping is harder to reproduce exactly.
- *No validation subject, fixed number of epochs:* uses one more subject for
  training, but the epoch count would have to come from somewhere, and choosing
  it by looking at test results is leakage.

**Trade-off:** 15 trainings cost about 15× the compute of a single split. In
return, every subject is tested exactly once by a model that never saw them, and
the spread across subjects becomes visible. That spread is large: S5 is the hardest
subject for every baseline I have run, and for three of the four methods in the
Deep PPG paper's per-subject table. The old validation
set contained S5, which is why the old naive FFT looked worse than predicting the
mean (31.4 vs 23.9 bpm). Over all 15 subjects the two are level (18.38 vs 18.57).

---

## D4. Headline metric: mean of per-subject MAE

**Chosen:** compute the MAE for each subject, then report the mean ± std of those
15 numbers. The window-weighted MAE (every window counts once) is saved too.

**Alternatives:** window-weighted MAE only. Subjects with longer recordings count
more, and S6 (1.46 h of valid data) would count about half as much as the others.

**Trade-off:** the per-subject mean answers "how well does this work for a new
person?", which is the question that matters for a wearable. It is also how the
Deep PPG paper reports results: its per-subject values average to its headline
7.65 bpm. That makes the comparison fair.

---

## D5. DSP baseline: built step by step, parameters fixed up front

**Chosen:** spectral peak picking, improved one step at a time so each step's
effect is measured on its own (`src/baseline.py`): zero-padding, a 40-200 bpm
search band, rejecting PPG peaks that match periodic accelerometer motion, then
tracking the previous estimate. Every parameter was set before running. Nothing
is fit to labels, so running it on all 15 subjects involves no training leakage.

**What happened** (mean ± std of per-subject MAE, bpm):

| Step | MAE |
|---|---|
| Constant (mean HR of the other 14 subjects) | 18.57 ± 7.68 |
| Naive FFT (original code) | 18.38 ± 8.09 |
| + zero-padding to 4096 points | 17.80 ± 8.12 |
| + 40-200 bpm band | 16.36 ± 7.36 |
| + reject periodic-motion peaks | 15.78 ± 6.28 |
| + hard tracking window (±15 bpm) | 22.31 ± 7.38 |
| + soft tracking instead (revised) | 15.74 ± 5.10 |
| soft tracking, no motion rejection | 15.10 ± 6.70 |

**Disclosure:** the hard tracking window was my first attempt and it made results
worse. Once it locks onto a wrong frequency it can never leave. Soft tracking (pick
the strong peak closest to the last estimate, or jump to the strongest if none is
close) was designed *after* seeing that result on the same data. That is one
round of adjustment, and both versions stay in the table.

**Trade-off:** a hand-designed DSP pipeline has a handful of parameters. Each
extra rule tuned against the evaluation data makes the baseline look better than
it would on new data, so I stopped after one revision. The best DSP result is in
the same range as SpaMa (15.56 ± 7.5) in the Deep PPG paper, and behind its
SpaMaPlus (11.06 ± 4.8). A stronger classical baseline is possible, but it is not
the point of this project.

---

## D6. Wrist-only cache of the dataset

**Chosen:** on first load, save each subject's wrist signals, labels and activity
to `cache/S{i}_wrist.npz` (162.5 MB for all 15), and read only that afterwards.

**Alternatives:** load the original pickles every time (0.8-1.7 GB each, mostly
700 Hz chest signals that are never used).

**Trade-off:** one extra derived copy on disk to keep in sync. In return, building
all 64,697 windows takes 30 s instead of several minutes, which matters when
running 15-fold experiments repeatedly.

---

## D7. Activity label per window: majority vote

**Chosen:** each 8 s window gets the most common of its 32 activity samples (4 Hz).

**Check:** the resulting counts per activity are within 20 windows of the counts
in Table 11 of the Deep PPG paper (e.g. walking 4,697 in both), and the totals
agree exactly (64,697).

**Trade-off:** windows that straddle two activities get one label. That only
matters for per-activity error tables, not for training.

---

## D8. Training budget per fold: at most 15 epochs, stop after 5 without improvement

**Chosen:** in leave-one-subject-out runs, each fold trains for at most 15 epochs
and stops early when validation MAE has not improved for 5 epochs.

**Why:** measured on this machine, one epoch on 13 subjects (about 56,000 windows)
takes about 94 s. The original setting (up to 80 epochs, patience 12) ran 33
epochs on the old split, so 15 folds would need about 13 hours. With the cap, the
worst case is about 6 hours.

**Alternatives:**
- *Keep 80 / 12:* most faithful to the original training, but too slow to repeat
  for every Phase 2 variant.
- *Use fewer folds:* faster, but gives up the per-subject picture that D3 is for.

**Trade-off:** a capped model may be slightly undertrained. Each fold now sees
about 2.8× more training windows per epoch than the old split did, and every
fold's learning curve is saved in `results/loso_<model>_summary.json`, so
whether the cap cost accuracy can be checked rather than guessed. All Phase 2
variants use the same budget, so their comparison stays fair.

**Outcome (cnn_v0, 15 folds):** no fold's best epoch was the last allowed one.
Best epochs ranged from 1 to 11, and all but one fold stopped early. The whole
run took 4.26 h of training. So for this model the cap did not cut training short.

**Phase 2 variants:** in one fold out of 75 the best epoch was the 15th, the
last allowed: `cnn_v0_32hz` on S5, the hardest subject. That fold may be
slightly undertrained; every other fold peaked earlier.

---

## D9. Input at 32 Hz instead of 64 Hz

**Chosen:** feed the model 256 samples per channel (32 Hz) instead of 512. The
PPG is decimated by 2 after band-passing; the accelerometer is used at its native
32 Hz instead of being upsampled.

**Why it is safe:** every signal is band-passed to 0.5–4 Hz first, and 4 Hz is far
below the 16 Hz Nyquist limit at 32 Hz. So no heart-rate information is lost, and
the original pipeline's upsampling of ACC to 64 Hz only created interpolated
samples.

**Alternatives:** keep 64 Hz (the original), or go lower (e.g. 16 Hz, Nyquist
8 Hz). Lower rates still fit the 4 Hz band, but leave less room for the shape of
each pulse.

**Trade-off:** every conv layer does half the work and holds half the
activations, for free in theory. Whether the model loses accuracy is measured
directly (`cnn_v0` vs `cnn_v0_32hz`: same network, only the rate changes).

**Outcome:** MACs 11.49 M → 5.75 M. Test MAE went 9.47 → 8.81 bpm while
validation MAE went 8.12 → 8.41. They move in opposite directions and both
changes are small, so the reading is "no measurable loss", not "better".

---

## D10. Depthwise separable convolutions

**Chosen:** blocks 2–4 split each conv into a depthwise conv (one filter per
channel) and a 1×1 pointwise conv (mixes channels). The first block stays a normal
conv because it has only 4 input channels.

**Alternatives:**
- *Just make the original CNN narrower:* simple, but cuts capacity everywhere.
- *Fewer blocks:* cheaper, but each output sees a shorter stretch of signal.
- *Neural architecture search, as in Q-PPG:* the strongest results, but far more
  compute than this project has.

**Trade-off:** at the same widths, MACs drop from 5.75 M to 1.61 M at 32 Hz. But
MACs are not speed: in a CPU training benchmark the depthwise model was only
about 2.5× faster than the original, not 7×, because depthwise convs do little
arithmetic per memory access and libraries optimise them less. Fewer parameters
can also mean lower accuracy, so a wider version that spends the budget back
(`ds_wide_32hz_fixed`, 3.80 M MACs) is tested too.

**Outcome:** `cnn_v0_32hz` → `ds_32hz`: params 118,945 → 46,753, MACs 5.75 M →
1.61 M, validation MAE 8.41 → 8.49, test MAE 8.81 → 9.05. Both changes are under
the 0.5 bpm noise threshold (D12): the model got 3.6× cheaper for no measurable
loss. Spending the budget back did not help: `ds_wide_32hz_fixed` (3.80 M MACs)
scored 8.28 validation / 9.18 test, against 8.08 / 9.00 for the narrow version
with the same input. More capacity is not what limits this model.

---

## D11. Accelerometer scaled by a fixed 0.1 g instead of z-scored per window

**Chosen:** the PPG channel is still z-scored per window (its amplitude depends on
skin contact and is not meaningful), but the three ACC channels are divided by a
fixed 0.1 g.

**Why:** z-scoring each window rescales a still wrist's sensor noise to the same
size as real arm motion, so the model cannot tell "no motion" from "lots of
motion". With a fixed scale, sitting gives ACC values around 0.03 and walking or
stairs around 1.7–1.8 (medians of per-window std, measured).

**How the constant was chosen (disclosure):** I first used 1/64, from the dataset
readme's "units of 1/64 g". That made the ACC channels almost zero, because the
*pickles* already store ACC in g (|ACC| = 1.01 at rest, measured); the 1/64 g
applies to the raw E4 CSV files. I then chose 0.1 g after seeing the motion range
above. It is one physical constant, not fit to heart-rate labels.

**Alternatives:** keep z-scoring (original); divide by a per-subject statistic
(needs calibration data from each new user); add the motion strength as a 5th
channel next to the z-scored ACC (more input, more MACs).

**Trade-off:** the model now sees how strong the motion is, but depends on the
sensor's units. A different wearable must deliver ACC in g, or be rescaled first.
The effect is measured directly (`ds_32hz` vs `ds_32hz_fixed`).

**Outcome:** overall it is close to neutral: validation MAE 8.49 → 8.08, test MAE
9.05 → 9.00. Per activity it does what it was meant to do: error at rest drops
(sitting 5.16 → 3.78 bpm, working 5.26 → 4.20), because the model can now see
that the wrist is still. But error rises in transitions (10.43 → 11.24) and table
soccer (11.62 → 12.89). Keeping it is a judgement call: a wearable spends most of
its time at rest, so better resting accuracy is worth more than these numbers
weight it.

---

## D12. Choosing a Phase 2 variant on validation MAE, not test MAE

**Chosen:** each variant changes one thing from the previous one, and all run the
same 15-fold LOSO with the same training budget (D8) and seed. The variant to
carry into Phase 3 is the one with the lowest **mean validation MAE** across the
15 folds. The test MAE of every variant is reported, but not used to choose.

**Why:** if I picked the variant with the best test MAE, the test subjects would
be helping to choose the model, and the reported number would be optimistic. That
is the same mistake the old split made.

**Alternatives:** pick by test MAE (simpler, optimistic); a nested
cross-validation (cleaner, about 15× more compute).

**Trade-off:** validation subjects are single people, so mean validation MAE is
noisy too. Each variant is also trained with one seed, so differences under about
half a bpm should be treated as ties rather than wins. Seed-to-seed variation has
not been measured yet.

**Outcome:** `ds_32hz_fixed` has the lowest mean validation MAE (8.08 ± 4.05) and
is carried into Phase 3. Its margins over the other variants are 0.04 (`cnn_v0`),
0.20 (`ds_wide_32hz_fixed`), 0.33 (`cnn_v0_32hz`) and 0.41 (`ds_32hz`) bpm, all
under 0.5. So by my own rule this is a tie on accuracy, and the honest claim is:
**7× fewer MACs and 2.5× fewer parameters than the original, at the same
accuracy**, not "a more accurate model". Its test MAE is 9.00 ± 4.55 bpm.

---

## D13. Export path: PyTorch → ONNX → TensorFlow → TFLite, in its own environment

**Chosen:** `torch.onnx.export` (the TorchScript-based exporter, opset 17) →
`onnx2tf` with its `tf_converter` backend, which writes a TensorFlow SavedModel →
TensorFlow's `TFLiteConverter` for FP32, FP16 and INT8. The model is exported with
batch size 1, which is how a device runs it. All of this lives in a separate
virtual environment (`.venv-export`: TensorFlow 2.21, onnx2tf 2.6.9, ONNX Runtime
1.26, LiteRT interpreter), so TensorFlow's dependencies cannot break training.

**Check:** on fold S1, the ONNX and TFLite FP32 models give the same test MAE as
PyTorch (5.839 bpm) and differ from it by less than 0.0001 bpm on every window.

**Alternatives:**
- *onnx2tf's newer default backend (`flatbuffer_direct`):* writes `.tflite`
  directly, but supports fewer operators and quantization modes and skips
  TensorFlow's standard quantizer.
- *Google's direct PyTorch → TFLite converter (ai-edge-torch / LiteRT Torch):*
  one step instead of three. Not used because the ONNX route is the most common
  in industry, works on Windows, and gives an ONNX file usable by other runtimes
  too.
- *The new `torch.export`-based ONNX exporter:* the default since PyTorch 2.9;
  the older exporter was kept because its output is what onnx2tf is most tested on.

**Trade-off:** three tools instead of one, and the input layout changes on the way
(channels-first `(1, 4, 256)` in PyTorch/ONNX becomes channels-last `(1, 256, 4)`
in TFLite). In return every step can be checked against PyTorch, and each format
uses its runtime's standard tools.

---

## D14. Benchmark setup: one pinned core, one thread, batch 1

**Chosen** (`src/benchmark.py`): the process is pinned to one CPU core, every
runtime uses one thread, batch size 1, 100 warm-up runs, then 2,000 timed runs;
median, p95 and p99 latency are reported. TFLite is timed with its default
XNNPACK kernels and again with plain built-in kernels.

**Alternatives:** time with all cores (higher throughput, but a wearable gets a
fraction of one small core); run on an emulator such as Renode/QEMU for a
Cortex-M (closer to a device, but emulators do not give trustworthy timing).

**Trade-off:** this measures relative cost between formats on a real CPU and
proves the model runs end-to-end through each runtime. It is still a 3.6 GHz
desktop core, so the absolute numbers are **not** device latency, and they are
never presented as such.

---

## D15. When to try quantization-aware training (rule set before the results)

**Chosen:** post-training INT8 quantization first, calibrated per fold on 500
windows from that fold's training subjects only. Quantization-aware training (QAT)
is tried **only if** the INT8 model's LOSO MAE is more than **0.5 bpm** worse than
FP32 (mean over the 15 subjects). 0.5 bpm is the same noise threshold as D12.

This rule is committed before the 15-fold INT8 results exist (only fold S1 had
been seen: 5.839 FP32 vs 5.976 INT8).

**Alternatives:** always do QAT (more work and another 15-fold training run, for
a gain that may be within noise); never do QAT (cheaper, but would hide a real
accuracy loss if there is one).

**Trade-off:** a threshold set in advance means the decision cannot be bent to
fit the result.

**Outcome (15 folds, shipped 2-D export with HR-stratified calibration):** INT8
MAE 9.071 ± 4.565 bpm vs FP32 9.003 ± 4.549, a difference of +0.068 bpm. That is
well under 0.5 bpm, so **QAT was not tried.**

---

## D16. INT8 calibration windows spread evenly over heart rate

**Chosen:** the 500 INT8 calibration windows are drawn from the fold's training
subjects with an equal share per 10 bpm heart-rate bin (`--calib hr_stratified`
in `src/export.py`), instead of uniformly at random.

**Why (found after the first 15-fold run, disclosed):** INT8 stores every
tensor's value range, learned from the calibration windows. With 500 random
windows only about 10 are above 150 bpm, so the model's output range came out too
narrow: the largest heart rate each fold's INT8 model could output was 144.8–173.9
bpm, though training data goes up to 187 bpm. The averages hid this (INT8 was
only +0.04 bpm worse than FP32), but on single windows INT8 sat flat at its
ceiling and was up to 25.1 bpm away from FP32, all of them during cycling. Up to
3.4 % of a test subject's windows had a true heart rate above their fold's
ceiling.

**What changed** (all 15 folds, measured on the 1-D export; D17 later moved to a
2-D export, where stratified INT8 scores 9.071):

| | Random | HR-stratified |
|---|---|---|
| Calibration windows above 150 bpm (of 500) | ~10 | 132 |
| INT8 output ceiling across folds | 144.8–173.9 bpm | 167.5–194.5 bpm |
| Worst window, \|INT8 − FP32\| | 25.09 bpm | 10.32 bpm |
| Windows with \|INT8 − FP32\| > 10 bpm | 0.02 % | 0.00 % |
| INT8 test MAE (FP32: 9.003) | 9.041 | 9.065 |

With stratified calibration, the ceiling changes the answer on at most 0.23 % of
any subject's windows (S6). Where a true heart rate is still above the ceiling,
the FP32 model does not predict that high either: the model itself
under-predicts very high heart rates, which is a model limitation, not a
quantization one.

**Alternatives:**
- *More random calibration windows:* would include more high-HR windows, but only
  by chance, and makes conversion slower.
- *Keep the last layer in float:* removes the output ceiling entirely, but adds a
  float op to an otherwise all-integer model, which a microcontroller may not
  support well.
- *Predict a normalised target* (e.g. (HR − 100) / 50) and rescale outside the
  model: would not change the clipping, only move where it happens.

**Trade-off:** the mean INT8 MAE is 0.024 bpm worse than with random calibration
(9.065 vs 9.041), well within noise. In exchange the model no longer has a hidden
ceiling around 145 bpm, which matters most exactly when the user is exercising.
I would rather report a slightly worse average than ship a model that silently
cannot follow a high heart rate.

---

## D17. Export a 2-D copy of the network (Conv1d → Conv2d with 1×k kernels)

**Chosen:** before export, `src/model.py:to_2d()` rebuilds the trained 1-D model
from 2-D layers with the same weights (input `(1, 4, 1, 256)`). In PyTorch the
two give identical outputs (max difference 0.0 bpm on 256 windows).

**Why (found while checking the benchmark):** the ONNX → TFLite converter turned
every 1-D depthwise conv into a *grouped* `CONV_2D`. TFLite has dedicated kernels
for `DEPTHWISE_CONV_2D`, which the 2-D copy produces, but not for grouped
`CONV_2D`. Measured effect, one core, one thread:

| Runtime | 1-D export | 2-D export |
|---|---|---|
| ONNX Runtime, FP32 | 149 µs | 83 µs |
| TFLite built-in kernels, FP32 | 384 µs | 292 µs |
| TFLite XNNPACK, FP32 | 56.7 µs | 56.8 µs |
| INT8 file size | 71.8 KB | 70.6 KB |

XNNPACK already handled the grouped conv well, so it gains nothing; every other
runtime gets faster.

**Alternatives:** keep the 1-D model and accept slower kernels on runtimes other
than XNNPACK; or write the model with `Conv2d` from the start (one code path, but
every training-side shape becomes 4-D for no training benefit).

**Trade-off:** one extra conversion step, guarded by an exact-equality check. The
accuracy of the 2-D export was re-measured on all 15 folds rather than assumed:
INT8 9.071 (2-D) vs 9.065 (1-D), FP32 identical.

---

## D18. Which TFLite interpreter the numbers come from

**Chosen:** accuracy and latency use the interpreter bundled with TensorFlow 2.21
(`tf.lite.Interpreter`), in three tiers: XNNPACK (default), TFLite built-in
kernels, and plain reference kernels. The `ai_edge_litert` 2.1.2 interpreter is
timed too, but only as a recorded anomaly.

**Why (measured):** on this Windows machine, `ai_edge_litert` 2.1.2 never applied
the XNNPACK delegate (its default and no-delegate timings were identical) and ran
INT8 in about 12 ms per window, against 83 µs for the same file in TensorFlow's
interpreter with XNNPACK. Benchmarking only with it would have reported INT8 as
roughly 40× slower than FP32, which is a property of that build, not of the model.

**Also measured, not explained:** with TFLite's built-in kernels (no XNNPACK),
INT8 takes 3.9 ms, slower than the plain reference kernels (1.8 ms). I report this
as measured on this Windows build and do not claim a cause.

**Trade-off:** `tf.lite.Interpreter` is deprecated in favour of LiteRT, so this
choice may need revisiting. Until then, one table shows all four runtimes, so the
anomalies are visible instead of hidden behind one number.

---

## D19. Streaming (causal) preprocessing on the device, and when to retrain for it (rule set before the results)

**The problem:** training band-passes each *whole* recording with a zero-phase
filter (`sosfiltfilt`), which runs the filter forwards and then backwards, so
every output sample depends on future samples. A wearable cannot do that: it has
to filter each sample as it arrives (`sosfilt`, causal, keeping the filter state
between windows). The existing API does a third thing: it zero-phase filters each
8 s window on its own, which adds edge effects at both ends of the window.

**Chosen:** measure all three on held-out subjects with the INT8 fold models
(fold *s* model on subject *s* only):
1. *offline*: training-time preprocessing (the Phase 3 result, 9.071 bpm);
2. *per-window*: what `api/main.py` does today;
3. *streaming*: a causal filter over the continuous stream, as a device would run.

**Rule, committed before running it:** if streaming is more than **0.5 bpm**
worse than offline (mean over the 15 subjects), retrain with causal filtering in
training so training and device see the same signal. Otherwise keep the model and
document the gap.

**Trade-off:** retraining costs another 15-fold run (about 5 h here). Not
measuring it would mean the reported 9.07 bpm is for a pipeline no device can run.

---

## D20. One deployable model: all 15 subjects, fixed 5 epochs

**Chosen:** the shipped model is `ds_32hz_fixed` trained on all 15 subjects for a
fixed 5 epochs: the median best epoch across its 15 LOSO folds (values 1–11).

**Alternatives:** ship one of the fold models (trained on 13 subjects, so less
data); keep one subject aside for early stopping (one fewer subject, and the stop
point depends on which subject).

**Trade-off:** with every subject in training there is no test data left for this
exact model, so **its accuracy is not measured directly**. The honest estimate is
the LOSO result of the same recipe (Phase 3: INT8 9.07 ± 4.57 bpm). Anything
measured on this final model is a sanity check (for example, that its INT8 and
FP32 versions agree), not an accuracy claim.

---

## D21. Retrain on causally filtered input (D19 rule triggered)

**What D19 measured** (INT8 fold models, each on its held-out subject):

| Preprocessing path | Zero-phase-trained model | **Causal-trained model** |
|---|---|---|
| Offline (each model's own training-style input) | 9.071 ± 4.565 | 9.235 ± 4.827 |
| Per-window zero-phase (today's API) | 9.178 ± 4.563 | 11.748 ± 5.941 |
| **Streaming, causal (device)** | 11.474 ± 4.876 | **9.235 ± 4.827** |

The zero-phase model lost 2.40 bpm under streaming and was worse on all 15
subjects, far over the 0.5 bpm threshold set in advance, so it was retrained.

**Chosen:** `ds_32hz_fixed_causal`: the same network, input options, training
budget and seed as `ds_32hz_fixed`, trained on windows filtered causally over the
whole recording. Those windows are bit-identical to what the streaming
preprocessor computes on a device (checked on S1, S6 and S15), so for this model
"offline" and "streaming" are the same input and give the same result.

**Result:** on the device pipeline, MAE drops from 11.47 to **9.24 bpm** (INT8).
On held-out S7 in the simulated device loop: 7.94 → 5.02 bpm. The causal model is
0.16 bpm worse than the zero-phase model was on *offline* input (9.24 vs 9.07),
which is the honest cost of being deployable: a causal filter has less
information than a zero-phase one. Its validation MAE was slightly lower (7.92 vs
8.08).

**New trade-off this creates:** the causal model now does badly on the API's
per-window zero-phase preprocessing (11.75 bpm). Training and serving must use the
same filter, so the API can only serve this model after its preprocessing is
switched to the causal streaming path. Until then the deployed API keeps the old
model.

**Alternatives:**
- *Keep the zero-phase model, filter with zero phase on the device* using a delay
  (wait for future samples before outputting): it adds latency to every reading
  and still differs from whole-recording filtering at window edges.
- *Train on a mix of both filters:* a model that tolerates both, but it is
  evaluated on neither cleanly.

---

## D22. Engineering tooling: what is tested, pinned and automated

**Chosen:**
- **pytest, with no dataset needed** (45 tests). The tests check properties
  that a regression would silently break, rather than re-running experiments:
  - streaming windows equal causal training windows (the basis of D21);
  - filter pass and stop bands;
  - parameter and MAC counts against the edge budget;
  - the 2-D export copy giving identical outputs;
  - the DSP baseline on synthetic signals with a known heart rate;
  - INT8 calibration covering high heart rates (D16);
  - API input validation.
- **ruff** for lint (pycodestyle, pyflakes, import order, bugbear, pyupgrade).
- **Exact version pins** in `requirements.txt` / `requirements-dev.txt` /
  `requirements-export.txt`: the versions every result was produced with.
  Packages the code never imports were removed.
- **A Python CLI (`src/cli.py`) as the entry point, with a thin Makefile on
  top**, because the project was developed on Windows, where `make` is usually not
  installed.
- **GitHub Actions** runs ruff and pytest on every push, with CPU-only PyTorch.
  The same steps were run in a fresh clone and a fresh virtual environment before
  committing.

**Alternatives:**
- *Tests that run the real pipeline on the dataset:* closer to the real thing,
  but CI cannot download 2.7 GB on every push, and a full LOSO run takes hours.
- *Version ranges instead of pins:* fewer upgrade chores, but "same code,
  different numbers" becomes possible.
- *Make only:* standard on Linux, but not usable out of the box on Windows.

**Trade-off:** the tests prove that the pieces behave as designed, not that the
reported accuracy reproduces. Reproducing the numbers still needs the dataset and
the CLI steps (hours on a CPU). The pins mean dependency upgrades are deliberate
and need a re-run to confirm results.

**Found by the tests:** the API accepted NaN/Infinity. Once they were rejected,
FastAPI's default 422 response tried to echo the NaN back as JSON and crashed.
Both are fixed (`api/main.py`).
