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
