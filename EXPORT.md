# DeepFilterNet ONNX Export Guide

This document covers how to build the Rust extension, install the Python package, and
export DeepFilterNet3 to ONNX in two modes:

- **Spectral model** — takes a complex STFT spectrogram, returns an enhanced spectrogram.
  Use this when your pipeline already lives in the frequency domain.
- **Component models** — the three sub-networks (`enc`, `erb_dec`, `df_dec`) exported
  individually.  Use this when you need fine-grained control or streaming sub-processing.

---

## Prerequisites

| Requirement | Tested version |
|---|---|
| Rust toolchain | 1.95.0 (`rustup` install) |
| Python | 3.14 |
| maturin | 1.13.1 |
| PyTorch | 2.11.0+cpu |
| onnxruntime | 1.25.1 |

A virtual environment at `venv/` is assumed.  Activate it before running any commands.

---

## 1 — Build and install `libdf` (Rust extension)

`libdf` provides the STFT analysis/synthesis and the ERB/unit-norm feature extraction that
the Python package relies on.  It must be compiled before any Python import of `df`.

```powershell
# From repo root — add cargo to PATH if needed
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

# Python 3.13+ requires the ABI3 forward-compatibility flag
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"

maturin develop --release --manifest-path pyDF/Cargo.toml
```

The compiled wheel is installed directly into the active virtual environment.  You should
see `🛠 Installed DeepFilterLib-x.y.z` on success.

> **Note:** `pyDF/pyproject.toml` pins `maturin>=1.3,<1.5` as a build-system requirement.
> That constraint is only enforced when building via `pip install`; running `maturin develop`
> directly bypasses it.  The `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` flag is needed because
> PyO3 0.20 (used by this repo) only declares support up to Python 3.12.

---

## 2 — Install the Python package

```powershell
# Install df package without rebuilding the Rust dep (already done above)
pip install -e DeepFilterNet --no-deps --no-build-isolation
```

---

## 3 — Run the exports

### Quick start — export everything

```powershell
python run_export.py --export-waveform
```

Produces `onnx_export/DeepFilterNet3/`:

```
deepfilternet_v3.onnx     — waveform end-to-end model (audio-in / audio-out)
deepfilternet_spec.onnx   — spectral model (spec-in / spec-out)
enc.onnx                  — encoder sub-network
erb_dec.onnx              — ERB mask decoder sub-network
df_dec.onnx               — deep-filter coefficient decoder sub-network
wav_input.npz / wav_output.npz       — reference I/O for deepfilternet_v3.onnx
spec_input.npz / spec_output.npz     — reference I/O for deepfilternet_spec.onnx
enc_input.npz / enc_output.npz       — reference I/O for enc.onnx
erb_dec_input.npz / erb_dec_output.npz
df_dec_input.npz / df_dec_output.npz
```

### Waveform model only

```powershell
python run_export.py --export-waveform --spec-only   # spec-only skips components
```

### Spectral model only

```powershell
python run_export.py --spec-only
```

### Component models only

```powershell
python run_export.py --components-only
```

### All flags

```
--model-dir DIR        Model directory            (default: _export_model/DeepFilterNet3)
--out-dir   DIR        Output directory           (default: onnx_export/<model-name>)
--export-waveform      Also export end-to-end waveform model (deepfilternet_v3.onnx)
--spec-only            Export spectral model only (no components)
--components-only      Export component models only (no spec)
--export-full          Also export full-feature monolithic model (deepfilternet2.onnx)
--no-check             Skip ORT parity verification after each export
--simplify             Run onnxsim simplification pass (requires onnxsim package)
--opset INT            ONNX opset version         (default: 14)
```

---

## 4 — Model reference

### 4.1 Waveform model — `deepfilternet_v3.onnx`

**Use case:** drop-in replacement for the original DeepFilterNet pipeline when you need a
single self-contained ONNX graph that accepts raw audio and returns enhanced audio.
The FFT and iFFT are baked in and numerically match the libdf analysis/synthesis.

| Tensor | Shape | Description |
|---|---|---|
| `audio` (input) | `[1, T]` | Raw mono float32 audio at 48 kHz |
| `enh` (output) | `[1, 1, T]` | Enhanced audio, same length |

- `T` is fully dynamic.
- The internal STFT uses the Vorbis window and left-pads by `fft_size − hop_size = 480`
  samples to match the zero-initialised `analysis_mem` in libdf.
- ORT parity vs `df.enhance.enhance()`: MAE ≈ 2.4e-08, MaxE ≈ 2.0e-07.

**How to call:**

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("onnx_export/DeepFilterNet3/deepfilternet_v3.onnx",
                            providers=["CPUExecutionProvider"])
# audio: float32 [1, T]  at 48 kHz
enh = sess.run(["enh"], {"audio": audio_np})[0]   # [1, 1, T]
enh_audio = enh[0, 0]                              # [T]
```

### 4.3 Spectral model — `deepfilternet_spec.onnx`

**Use case:** your pipeline already computes an STFT (e.g. WOLA analysis filter bank) and
you want noise reduction entirely in the frequency domain, without an FFT/iFFT inside the
ONNX graph.

| Tensor | Shape | Description |
|---|---|---|
| `spec` (input) | `[1, 1, T, 481, 2]` | Complex STFT frames.  Last dim is `[re, im]`. |
| `enh` (output) | `[1, 1, T, 481, 2]` | Enhanced complex STFT frames, same shape. |

- `T` is the time/frame dimension and is **dynamic** (any value ≥ 1).
- `F = 481 = fft_size/2 + 1 = 960/2 + 1` for 48 kHz, 20 ms window.
- The graph internally computes ERB filterbank features and normalisation — no
  pre-processing is needed beyond a standard STFT.
- The feature state (ERB mean norm, unit norm) is initialised from fixed constants at every
  call; there is no recurrent state exposed at the graph boundary.

**How to call (Python / OnnxRuntime):**

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("onnx_export/DeepFilterNet3/deepfilternet_spec.onnx",
                            providers=["CPUExecutionProvider"])
# spec: float32 [1, 1, T, 481, 2]
enh = sess.run(["enh"], {"spec": spec_np})[0]
```

### 4.4 Component models

These operate in the **feature** domain, matching the original Python pipeline in
`df.enhance.df_features`.  Use them when you need access to intermediate tensors (mask,
local SNR, deep-filter coefficients) or when you want to run sub-networks independently.

#### Encoder — `enc.onnx`

Inputs (produced by `df.enhance.df_features` or equivalent):

| Name | Shape | Description |
|---|---|---|
| `feat_erb` | `[1, 1, T, 32]` | ERB-band log-power features, mean-normalised |
| `feat_spec` | `[1, 2, T, 96]` | Complex spec features, unit-normalised, re/im in channel dim |

Outputs:

| Name | Shape | Description |
|---|---|---|
| `e0` | `[1, 64, T, 32]` | Skip connection 0 |
| `e1` | `[1, 64, T, 16]` | Skip connection 1 |
| `e2` | `[1, 64, T, 8]`  | Skip connection 2 |
| `e3` | `[1, 64, T, 4]`  | Skip connection 3 |
| `emb` | `[1, T, 512]`   | Temporal embedding |
| `c0`  | `[1, 64, T, 96]`| DF pathway features |
| `lsnr`| `[1, T, 1]`    | Local SNR estimate (dB) |

#### ERB mask decoder — `erb_dec.onnx`

Inputs: `emb [1,T,512]`, `e3 [1,64,T,4]`, `e2 [1,64,T,8]`, `e1 [1,64,T,16]`, `e0 [1,64,T,32]`

Output: `m [1, 1, T, 32]` — ERB gain mask applied to the spectrum.

#### Deep-filter coefficient decoder — `df_dec.onnx`

Inputs: `emb [1,T,512]`, `c0 [1,64,T,96]`

Output: `coefs [1, T, 96, 10]` — complex FIR filter coefficients, shape `[B, T, F, df_order*2]`
where `df_order=5`.

### 4.5 Full-feature monolithic model — `deepfilternet2.onnx` (optional)

Exported only with `--export-full`.  Takes pre-computed features (same as component inputs)
and returns all four outputs in one graph.  Useful if you need `m`, `lsnr`, and `coefs`
alongside the enhanced spectrum without calling three sub-networks.

---

## 5 — Validating outputs

### Automated validation

```powershell
python validate_exports.py
```

This script:
1. Runs `df.enhance.enhance()` on the same test audio saved at export time to get a
   reference waveform.
2. Runs `deepfilternet_v3.onnx` and compares with the reference waveform.
3. Runs `deepfilternet_spec.onnx` with `df.analysis(test_audio)` and compares the output
   spectrum with `df.analysis(reference_enhanced_audio)`.
4. Cross-checks: takes the waveform ONNX output through `df.analysis` and compares with
   the spectral ONNX output — confirming both models are mutually consistent.
5. Checks each model's saved reference `.npz` files against a fresh ORT run.

All checks pass with MAE well below the 1e-4 threshold.

### Verified parity numbers

| Check | MAE | MaxE |
|---|---|---|
| Waveform ONNX vs Python `enhance()` | 2.4e-08 | 2.0e-07 |
| Spectral ONNX vs `df.analysis(enhance())` | 1.3e-05 | 5.7e-03 |
| Waveform ONNX spectrum vs spectral ONNX | 1.3e-05 | 5.7e-03 |

The spectral model's ~5.7e-03 MaxE (vs the waveform model's 2e-07) is expected: the
spectral model is exported with a random feature-normalisation state, while the waveform
model and the Python reference both start from the same initialised state baked into the
`SpectralEnhancer` buffers.  The MAE of 1.3e-05 is well within any practical tolerance.

### Manual check for one model

```python
import numpy as np, onnxruntime as ort

sess    = ort.InferenceSession("onnx_export/DeepFilterNet3/deepfilternet_spec.onnx",
                               providers=["CPUExecutionProvider"])
inp     = dict(np.load("onnx_export/DeepFilterNet3/spec_input.npz"))
ref_enh = np.load("onnx_export/DeepFilterNet3/spec_output.npz")["enh"]
ort_enh = sess.run(["enh"], inp)[0]

print("MAE :", np.abs(ort_enh - ref_enh).mean())
print("MaxE:", np.abs(ort_enh - ref_enh).max())
```

---

## 6 — Implementation notes

The spectral model (`SpectralEnhancer` in `DeepFilterNet/df/scripts/export.py`) wraps
DeepFilterNet3 with two added layers so the graph boundary is pure spectrogram:

1. **`SpectralFeatures`** — computes ERB log-power features and unit-normalised complex
   features from the raw spectrogram, replicating what `libdf.erb` / `libdf.erb_norm` /
   `libdf.unit_norm` do in the C pipeline.

2. **`OnnxExportWrapper`** — replaces the production `MF.DF` (multiframe complex filter,
   uses `torch.view_as_complex` which ONNX cannot represent) with `DfOp(method="real_loop")`
   — a real-valued equivalent that loops over `df_order=5` explicitly, which is ONNX-traceable.

Both models are exported with `jit=False` (trace-based).  The JIT-scripted path loses dtype
information on `BatchNorm2d` outputs in PyTorch 2.11; tracing avoids this.

### STFT/iSTFT in the waveform model

`torch.stft` / `torch.istft` and `Tensor.unfold` are not ONNX-exportable for dynamic-length
inputs via the TorchScript path.  The waveform model therefore implements analysis and
synthesis as DFT filter banks using `F.conv1d` / `F.conv_transpose1d`:

- **Analysis**: `F.conv1d(padded_audio, analysis_kernel, stride=hop)` where
  `analysis_kernel[k, 0, n] = wnorm · window[n] · cos(2πkn/N)` (real) and
  `wnorm · window[n] · (−sin(2πkn/N))` (imaginary), for k = 0 … F−1.
  This is a fixed `[2F, 1, N]` = `[962, 1, 960]` conv kernel.
- **Synthesis**: `F.conv_transpose1d(enh_spectrum, synthesis_kernel, stride=hop)` performs
  IDFT + window + overlap-add in a single transposed conv with a `[2F, 1, N]` kernel.

Both operations are standard ONNX `Conv` / `ConvTranspose` nodes with a fixed kernel,
so ONNX Runtime handles dynamic T without any special support.

---

## 7 — STFT parameters for `deepfilternet_spec.onnx`

The model was trained with the following STFT settings (from `config.ini`):

| Parameter | Value |
|---|---|
| Sample rate | 48 000 Hz |
| FFT size | 960 samples (20 ms) |
| Hop size | 480 samples (10 ms, 50 % overlap) |
| Window | Hann (standard STFT) |
| Frequency bins F | 481 (`fft_size/2 + 1`) |

Your upstream STFT must match these parameters exactly.  The model expects the full
one-sided spectrum `[0 … Nyquist]` in linear (not dB) magnitude, stored as real/imag pairs.
