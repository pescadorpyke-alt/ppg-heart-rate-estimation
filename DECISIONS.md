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
