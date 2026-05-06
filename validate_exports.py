"""
Validate the three ONNX export artefacts against the reference Python pipeline.

What is checked
---------------
1. Waveform model (deepfilternet_v3.onnx)
   Input : audio [1, T]   (same random test signal used at export time)
   Output: enh   [1, 1, T]
   Expected: numerically identical to df.enhance.enhance() on the same audio.

2. Spectral model (deepfilternet_spec.onnx)
   Input : spec  [1, 1, T_frames, 481, 2]  (STFT of the test audio, df-scale)
   Output: enh   [1, 1, T_frames, 481, 2]  (enhanced spectrum, df-scale)
   Expected: matches df.analysis(enhanced_reference_audio).

3. Cross-check: waveform vs spec
   The waveform model output, when run through df.analysis, should match the
   spectral model output directly.  This confirms both models are consistent.

Usage
-----
    python validate_exports.py

All comparisons print MAE and MaxE.  Success threshold is MAE < 1e-4.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepFilterNet"))

import numpy as np
import torch
import onnxruntime as ort

from libdf import DF
from df.enhance import init_df, enhance, df_features
from df.utils import get_norm_alpha
from df.modules import erb_fb

MODEL_DIR = os.path.join(os.path.dirname(__file__), "_export_model", "DeepFilterNet3")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "onnx_export", "DeepFilterNet3")

MAE_THRESHOLD = 1e-4


# ── helpers ──────────────────────────────────────────────────────────────────

def _sess(name):
    path = os.path.join(EXPORT_DIR, name)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _stats(label, a, b):
    diff = np.abs(a - b)
    mae  = diff.mean()
    mxe  = diff.max()
    ok   = "PASS" if mae < MAE_THRESHOLD else "FAIL"
    print(f"  {label:50s}  MAE={mae:.2e}  MaxE={mxe:.2e}  {ok}")
    return mae < MAE_THRESHOLD


# ── reference pipeline ───────────────────────────────────────────────────────

print("Loading model ...")
model, df_state, _, epoch = init_df(MODEL_DIR, log_level="WARNING", config_allow_defaults=True)
print(f"  Epoch {epoch}")

# Use the same random test audio that was saved at export time
audio_np = np.load(os.path.join(EXPORT_DIR, "wav_input.npz"))["audio"]   # [1, T]
audio_t  = torch.from_numpy(audio_np)                                      # [1, T]

print("\nRunning reference Python pipeline (df.enhance.enhance) ...")
# enhance() expects [C, T]; audio_np is [1, T]
with torch.no_grad():
    ref_audio_t = enhance(model, df_state, audio_t, pad=False)              # [C, T']
ref_audio_np = ref_audio_t.numpy()                                          # [C, T'] — C=1

# Reference spectrum (df.analysis of the enhanced output) [C, T_frames, F] complex
df_state.reset()
ref_spec_np = df_state.analysis(ref_audio_np)                               # [1, T_frames, F] complex
ref_spec_ri = np.stack([ref_spec_np.real, ref_spec_np.imag], axis=-1)       # [1, T_frames, F, 2]
ref_spec_5d = ref_spec_ri[:, np.newaxis].astype(np.float32)                 # [1, 1, T_frames, F, 2]


# ── 1. Waveform model ────────────────────────────────────────────────────────

print("\n[1] deepfilternet_v3.onnx — waveform in / waveform out")
wav_sess = _sess("deepfilternet_v3.onnx")
ort_enh = wav_sess.run(["enh"], {"audio": audio_np})[0]                    # [1, 1, T]
ort_enh_1d = ort_enh[0, 0]                                                  # [T]

# enhance() with pad=False outputs T' < T due to STFT delay; trim to match
T_out = ref_audio_np.shape[-1]
ok1 = _stats("ORT waveform vs Python enhance()", ort_enh_1d[:T_out], ref_audio_np[0])


# ── 2. Spectral model ─────────────────────────────────────────────────────────

print("\n[2] deepfilternet_spec.onnx — spectrum in / spectrum out")

# Build the input spectrum in df.analysis format.
df_state.reset()
input_spec_np = df_state.analysis(audio_np)                                # [1, T_frames, F] complex
input_spec_ri = np.stack([input_spec_np.real, input_spec_np.imag], axis=-1)  # [1, T_frames, F, 2]
input_spec_5d = input_spec_ri[:, np.newaxis].astype(np.float32)             # [1, 1, T_frames, F, 2]

spec_sess = _sess("deepfilternet_spec.onnx")
ort_enh_spec = spec_sess.run(["enh"], {"spec": input_spec_5d})[0]           # [1, 1, T_frames, F, 2]

# Compare against df.analysis of the reference audio
ok2 = _stats("ORT spec output vs df.analysis(enhance())", ort_enh_spec, ref_spec_5d.astype(np.float32))


# ── 3. Cross-check: waveform model → spectrum vs spectral model ───────────────

print("\n[3] Cross-check: waveform ONNX output spectrum vs spectral ONNX output")

# Run df.analysis on the waveform ONNX output to get its spectrum
df_state.reset()
ort_enh_spec_from_wav = df_state.analysis(ort_enh_1d[np.newaxis])         # [1, T_frames, F] complex
ort_enh_5d = np.stack([ort_enh_spec_from_wav.real, ort_enh_spec_from_wav.imag], axis=-1)[:, np.newaxis].astype(np.float32)

ok3 = _stats("Waveform ONNX spectrum vs spectral ONNX", ort_enh_5d, ort_enh_spec)


# ── 4. ORT self-consistency: saved reference npz ──────────────────────────────

print("\n[4] ORT self-consistency: npz reference files")
wav_ref  = np.load(os.path.join(EXPORT_DIR, "wav_output.npz"))["enh"]      # [1, T]
spec_ref = np.load(os.path.join(EXPORT_DIR, "spec_output.npz"))["enh"]     # [1, 1, T_frames, F, 2]

ok4a = _stats("wav_output.npz vs ORT re-run", wav_ref, ort_enh_1d)
ok4b = _stats("spec_output.npz vs ORT re-run", spec_ref, ort_enh_spec)


# ── summary ──────────────────────────────────────────────────────────────────

all_ok = ok1 and ok2 and ok3 and ok4a and ok4b
print("\n" + ("All checks PASSED." if all_ok else "Some checks FAILED — see above."))
sys.exit(0 if all_ok else 1)
