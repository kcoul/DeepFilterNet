"""
Validate the streaming DeepFilterNet ONNX exports against the reference Torch pipeline.

Checks:
1. Waveform export preserves the original streaming ABI and matches the checked-in source model.
2. Streaming spectral export rolled frame-by-frame matches the batch spectral Torch wrapper.
3. Streaming component exports rolled frame-by-frame match the batch encoder / decoder modules.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepFilterNet"))

import numpy as np
import onnxruntime as ort
import torch

from df.enhance import df_features, init_df
from df.scripts.export import ModelParams, SpectralEnhancer
from df.utils import get_norm_alpha

MODEL_DIR = os.path.join(os.path.dirname(__file__), "_export_model", "DeepFilterNet3")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "onnx_export", "DeepFilterNet3")
ORIGINAL_WAV_MODEL = os.path.join(os.path.dirname(__file__), "onnx_test", "original", "deepfilternet_v3.onnx")

WAVEFORM_FRAME = 480
WAVEFORM_STATE = 45304
SPEC_STATE = 6666
ENC_STATE = 704
DEC_STATE = 512

MAE_THRESHOLD = 1e-4
MAX_THRESHOLD = 1e-2


def _sess(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _stats(label, a, b, mae_threshold=MAE_THRESHOLD, max_threshold=MAX_THRESHOLD):
    diff = np.abs(a - b)
    mae = float(diff.mean())
    mxe = float(diff.max())
    ok = mae < mae_threshold and mxe < max_threshold
    print(f"  {label:52s}  MAE={mae:.2e}  MaxE={mxe:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _run_waveform_stream(sess, audio_np):
    state = np.zeros((WAVEFORM_STATE,), dtype=np.float32)
    atten = np.array(0.0, dtype=np.float32)
    outs = []
    for start in range(0, audio_np.shape[1], WAVEFORM_FRAME):
        chunk = np.zeros((WAVEFORM_FRAME,), dtype=np.float32)
        src = audio_np[:, start : start + WAVEFORM_FRAME].reshape(-1)
        chunk[: src.shape[0]] = src
        enh, state = sess.run(
            ["enhanced_audio_frame", "new_states"],
            {"input_frame": chunk, "states": state, "atten_lim_db": atten},
        )
        outs.append(np.asarray(enh, dtype=np.float32).reshape(-1)[: src.shape[0]])
    return np.concatenate(outs, axis=0)[np.newaxis, :]


def _run_spectral_stream(sess, spec_np):
    state = np.zeros((SPEC_STATE,), dtype=np.float32)
    enh_frames = []
    lsnr_frames = []
    for idx in range(spec_np.shape[2]):
        enh, state, lsnr = sess.run(
            ["enh", "new_state", "lsnr"],
            {"spec": spec_np[:, :, idx : idx + 1], "state": state},
        )
        enh_frames.append(enh)
        lsnr_frames.append(lsnr)
    return np.concatenate(enh_frames, axis=2), state, np.concatenate(lsnr_frames, axis=1)


def _run_component_streams(enc_sess, erb_sess, df_sess, feat_erb_np, feat_spec_np):
    enc_state = np.zeros((ENC_STATE,), dtype=np.float32)
    erb_state = np.zeros((DEC_STATE,), dtype=np.float32)
    df_state = np.zeros((DEC_STATE,), dtype=np.float32)

    e0s = []
    e1s = []
    e2s = []
    e3s = []
    embs = []
    c0s = []
    lsnrs = []
    ms = []
    coefs = []

    for idx in range(feat_erb_np.shape[2]):
        e0, e1, e2, e3, emb, c0, lsnr, enc_state = enc_sess.run(
            ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr", "new_state"],
            {
                "feat_erb": feat_erb_np[:, :, idx : idx + 1],
                "feat_spec": feat_spec_np[:, :, idx : idx + 1],
                "state": enc_state,
            },
        )
        m, erb_state = erb_sess.run(
            ["m", "new_state"],
            {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0, "state": erb_state},
        )
        coef, df_state = df_sess.run(
            ["coefs", "new_state"], {"emb": emb, "c0": c0, "state": df_state}
        )
        e0s.append(e0)
        e1s.append(e1)
        e2s.append(e2)
        e3s.append(e3)
        embs.append(emb)
        c0s.append(c0)
        lsnrs.append(lsnr)
        ms.append(m)
        coefs.append(coef)

    return {
        "e0": np.concatenate(e0s, axis=2),
        "e1": np.concatenate(e1s, axis=2),
        "e2": np.concatenate(e2s, axis=2),
        "e3": np.concatenate(e3s, axis=2),
        "emb": np.concatenate(embs, axis=1),
        "c0": np.concatenate(c0s, axis=2),
        "lsnr": np.concatenate(lsnrs, axis=1),
        "m": np.concatenate(ms, axis=2),
        "coefs": np.concatenate(coefs, axis=1),
    }


print("Loading model ...")
model, df_state, _, epoch = init_df(MODEL_DIR, log_level="WARNING", config_allow_defaults=True)
model = model.to("cpu").eval()
print(f"  Epoch {epoch}")

p = ModelParams()
audio_np = np.load(os.path.join(EXPORT_DIR, "wav_input.npz"))["audio"].astype(np.float32)
audio_t = torch.from_numpy(audio_np)
spec_t, feat_erb_t, feat_spec_t = df_features(audio_t, df_state, p.nb_df, device="cpu")
feat_spec_ch_t = feat_spec_t.transpose(1, 4).squeeze(4)

with torch.no_grad():
    batch_spec = SpectralEnhancer(
        model, df_state.erb_widths(), p.sr, p.nb_df, get_norm_alpha(log=False)
    ).to("cpu")
    ref_spec = batch_spec(spec_t)[0].numpy()
    ref_e0, ref_e1, ref_e2, ref_e3, ref_emb, ref_c0, ref_lsnr = model.enc(feat_erb_t, feat_spec_ch_t)
    ref_m = model.erb_dec(ref_emb, ref_e3, ref_e2, ref_e1, ref_e0).numpy()
    ref_coefs = model.df_dec(ref_emb, ref_c0).numpy()
    ref_e0 = ref_e0.numpy()
    ref_e1 = ref_e1.numpy()
    ref_e2 = ref_e2.numpy()
    ref_e3 = ref_e3.numpy()
    ref_emb = ref_emb.numpy()
    ref_c0 = ref_c0.numpy()
    ref_lsnr = ref_lsnr.numpy()

spec_np = spec_t.numpy().astype(np.float32)
feat_erb_np = feat_erb_t.numpy().astype(np.float32)
feat_spec_ch_np = feat_spec_ch_t.numpy().astype(np.float32)

print("\n[1] Waveform streaming export")
wav_export = _sess(os.path.join(EXPORT_DIR, "deepfilternet_v3.onnx"))
wav_original = _sess(ORIGINAL_WAV_MODEL)
export_audio = _run_waveform_stream(wav_export, audio_np)
original_audio = _run_waveform_stream(wav_original, audio_np)
ok1 = _stats("exported waveform vs original waveform model", export_audio, original_audio, 1e-6, 1e-6)

print("\n[2] Spectral streaming export")
spec_export = _sess(os.path.join(EXPORT_DIR, "deepfilternet_spec.onnx"))
ort_spec, final_spec_state, ort_lsnr = _run_spectral_stream(spec_export, spec_np)
ok2 = _stats(
    "streaming spectral ONNX vs batch spectral Torch",
    ort_spec,
    ref_spec,
    mae_threshold=5e-3,
    max_threshold=5e-2,
)

print("\n[3] Component streaming exports")
enc_export = _sess(os.path.join(EXPORT_DIR, "enc.onnx"))
erb_export = _sess(os.path.join(EXPORT_DIR, "erb_dec.onnx"))
df_export = _sess(os.path.join(EXPORT_DIR, "df_dec.onnx"))
stream_parts = _run_component_streams(enc_export, erb_export, df_export, feat_erb_np, feat_spec_ch_np)
ok3 = True
ok3 &= _stats("enc.e0 stream vs batch", stream_parts["e0"], ref_e0)
ok3 &= _stats("enc.e1 stream vs batch", stream_parts["e1"], ref_e1)
ok3 &= _stats("enc.e2 stream vs batch", stream_parts["e2"], ref_e2)
ok3 &= _stats("enc.e3 stream vs batch", stream_parts["e3"], ref_e3)
ok3 &= _stats("enc.emb stream vs batch", stream_parts["emb"], ref_emb)
ok3 &= _stats("enc.c0 stream vs batch", stream_parts["c0"], ref_c0)
ok3 &= _stats("enc.lsnr stream vs batch", stream_parts["lsnr"], ref_lsnr)
ok3 &= _stats("erb_dec stream vs batch", stream_parts["m"], ref_m)
ok3 &= _stats(
    "df_dec stream vs batch",
    stream_parts["coefs"],
    ref_coefs,
    mae_threshold=5e-4,
    max_threshold=2e-1,
)

all_ok = ok1 and ok2 and ok3
print("\n" + ("All checks PASSED." if all_ok else "Some checks FAILED - see above."))
sys.exit(0 if all_ok else 1)
