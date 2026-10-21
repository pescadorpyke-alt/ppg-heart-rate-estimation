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
the spread across subjects becomes visible. That spread is large: S5 is hard for
every method I have run, and for the published methods too. The old validation
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
