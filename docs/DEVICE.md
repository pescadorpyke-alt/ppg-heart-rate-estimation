# Running on a real device: what is done, what is next

There is no hardware in this project yet (DECISIONS.md D1). Everything below
"Measured" was run on a desktop CPU in a constrained setup (one pinned core, one
thread). Everything under "Next steps" has **not** been done, and no device
number in this repo is estimated or extrapolated.

## Measured (simulated device loop)

`src/device_demo.py` replays a held-out subject the way firmware would receive
it: every 2 s, 128 new PPG samples (64 Hz) and 64 new 3-axis ACC samples (32 Hz)
go through the streaming preprocessor (`src/stream.py`: causal band-pass with
state kept between windows, 8 s ring buffers), then one fully-INT8 TFLite
inference.

| | Measured | Budget (D2) |
|---|---|---|
| Model file, fully INT8 (`models/ppg_hr_int8_full.tflite`) | 70.4 KB | ≤ 256 KB flash |
| Activation memory, INT8 | between 12.0 KB (largest single layer, in + out) and 55.3 KB (all activation tensors, no reuse) | ≤ 64 KB RAM |
| Preprocessor state (ring buffers + filter state, float32) | 5,248 B | — |
| Compute per window | 1.61 M MACs | ≤ 5 M |
| Latency per window on a 3.6 GHz desktop core (not device latency) | preprocessing 177 µs + inference 114 µs (median) | ≤ 200 ms |

The real tensor arena lies between the two activation bounds: the memory planner
reuses buffers, but the Python interpreter does not report the planned size.
Even the upper bound fits the 64 KB budget.

**Operators the firmware must register** (from the INT8 file): `CONV_2D` ×4,
`DEPTHWISE_CONV_2D` ×3, `MAX_POOL_2D` ×4, `MEAN` ×1, `FULLY_CONNECTED` ×2,
`RESHAPE` ×1. The fully-INT8 file takes int8 input and gives int8 output; the
scales and zero points are in `models/ppg_hr.json`.

## Next step A: microcontroller (TensorFlow Lite for Microcontrollers)

1. **Pick a board** and check its datasheet against the table above: at least
   ~70 KB of flash for the model plus the firmware, and RAM for the tensor arena
   (≤ 55.3 KB) plus ~5 KB of preprocessing state plus the stack.
2. **Embed the model** as a C array (`xxd -i models/ppg_hr_int8_full.tflite`)
   and create a `MicroMutableOpResolver` with exactly the six operators above.
3. **Port the preprocessing** from `src/stream.py`. The band-pass is a 4th-order
   Butterworth, i.e. two biquad sections per channel; the coefficients come from
   `scipy.signal.butter(4, [0.5, 4.0], btype="band", fs=64 or 32, output="sos")`.
   CMSIS-DSP's biquad cascade functions implement exactly this form.
   - Then: keep every 2nd filtered PPG sample, z-score the PPG window, divide
     ACC (in g) by 0.1, and quantize with the input scale / zero point.
   - **Check against Python:** feed the same recorded samples to the C port and
     to `stream.py`, and compare the windows. They should match to float32
     rounding. This is the single most important test, because Phase 4 showed
     that a preprocessing mismatch costs more accuracy than quantization does
     (D19, D21).
4. **Measure on the device**, in place of the desktop numbers above:
   - latency: cycle counter around `interpreter.Invoke()` and around the
     preprocessing, separately;
   - RAM: `interpreter.arena_used_bytes()` after `AllocateTensors()`;
   - flash: from the linker map file;
   - energy per window: a power profiler on the supply rail, if available.
5. **Accuracy on the device:** replay a held-out subject's recording over serial
   and compare the device's outputs with `results/device_demo_*.csv`.

## Next step B: Android phone or watch (LiteRT)

1. A small Kotlin app loads `models/ppg_hr_int8.tflite` (float input/output) with
   the LiteRT interpreter and runs it on recorded windows bundled as an asset,
   showing heart rate and per-window inference time.
2. The streaming preprocessing from `src/stream.py` must be ported to Kotlin and
   checked window-by-window against Python, exactly as for the microcontroller.
3. **Live sensor input** depends on the device: the phone's accelerometer is
   easy to read, but raw PPG is usually not exposed to apps by phone or watch
   APIs, so live use would need a device or SDK that provides raw PPG at 64 Hz.

## Known limitations that carry over to a device

- **Accuracy:** LOSO estimate for the shipped recipe, INT8, streaming input:
  9.24 ± 4.83 bpm MAE (results/stream_eval_ds_32hz_fixed_causal_summary.json),
  with large differences between people (S5 is hardest).
- **INT8 output ceiling:** the shipped INT8 model cannot output more than
  **177.4 bpm** (`output_max_bpm` in `models/ppg_hr.json`; training data reaches
  187 bpm). Very high heart rates can be clipped (D16).
- **No smoothing across windows** yet. Real wearables smooth the heart-rate
  trace; adding it would be a separate, evaluated change.
- **Not a medical device.**
