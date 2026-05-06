import os
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnxruntime as ort
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

from df.enhance import (
    ModelParams,
    df_features,
    enhance,
    get_model_basedir,
    init_df,
    setup_df_argument_parser,
)
from df.io import get_test_sample, save_audio
from df.modules import DfOp, ExponentialUnitNorm, erb_fb
from df.utils import get_norm_alpha
from libdf import DF


def shapes_dict(
    tensors: Tuple[Tensor], names: Union[Tuple[str], List[str]]
) -> Dict[str, Tuple[int]]:
    if len(tensors) != len(names):
        logger.warning(
            f"  Number of tensors ({len(tensors)}) does not match provided names: {names}"
        )
    return {k: v.shape for (k, v) in zip(names, tensors)}


def ensure_tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


class ExponentialMeanNorm(nn.Module):
    """Torch implementation of the ERB mean normalization used by libdf."""

    def __init__(self, alpha: float, num_bands: int):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("init_state", torch.linspace(-60.0, -90.0, num_bands).view(1, 1, -1))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, E]
        b, c, t_steps, e = x.shape
        state = self.init_state.expand(b, c, e)
        out = []
        for frame_idx in range(t_steps):
            cur = x[:, :, frame_idx, :]
            state = cur * (1 - self.alpha) + state * self.alpha
            out.append((cur - state) / 40.0)
        return torch.stack(out, dim=2)


class SpectralFeatures(nn.Module):
    """Compute model features directly from a complex spectrum represented as [..., 2]."""

    def __init__(self, erb_fb_widths: np.ndarray, sr: int, nb_df: int, alpha: float):
        super().__init__()
        fb = erb_fb(erb_fb_widths, sr, inverse=False).to("cpu")
        self.nb_df = nb_df
        self.register_buffer("erb_fb", fb)
        self.erb_norm = ExponentialMeanNorm(alpha, fb.shape[1])
        self.spec_norm = ExponentialUnitNorm(alpha, nb_df)

    def forward(self, spec: Tensor) -> Tuple[Tensor, Tensor]:
        # spec: [B, 1, T, F, 2]
        power_spec = spec.square().sum(dim=-1)
        erb_feat = torch.matmul(power_spec, self.erb_fb).clamp_min(1e-10).log10() * 10.0
        erb_feat = self.erb_norm(erb_feat)
        spec_feat = self.spec_norm(spec[..., : self.nb_df, :])
        return erb_feat, spec_feat


class DfOutputReshapeOld(nn.Module):
    """Export-only reshape compatible with the real-valued DfOp path.

    df_dec output is [B, T, F, O*2] where the last dim is order-major (o0_re, o0_im, o1_re, …).
    We split that last dim into [O, 2], then transpose F and O to get [B, T, O, F, 2] as expected
    by DfOp.forward_real_loop.
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, T, O, F, 2]
        b, t = coefs.shape[:2]
        # Split last dim O*2 into [O, 2] first (frequency dim stays intact),
        # then move O before F.
        coefs = coefs.view(b, t, self.df_bins, self.df_order, 2).permute(0, 1, 3, 2, 4)
        return coefs.contiguous()


class OnnxExportWrapper(nn.Module):
    """Rebuild the DF3 forward pass with ONNX-friendly real-valued DF application."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.df_out_transform = DfOutputReshapeOld(model.df_order, model.nb_df)
        self.df_op = DfOp(
            df_bins=model.nb_df,
            df_order=model.df_order,
            df_lookahead=model.df_lookahead,
            method="real_loop",
            freq_bins=model.freq_bins,
        )

    def forward(self, spec: Tensor, feat_erb: Tensor, feat_spec: Tensor) -> Tensor:
        feat_spec = feat_spec.squeeze(1).permute(0, 3, 1, 2)

        feat_erb = self.model.pad_feat(feat_erb)
        feat_spec = self.model.pad_feat(feat_spec)
        e0, e1, e2, e3, emb, c0, _ = self.model.enc(feat_erb, feat_spec)

        if self.model.run_erb:
            m = self.model.erb_dec(emb, e3, e2, e1, e0)
            spec_m = self.model.mask(spec, m)
        else:
            spec_m = torch.zeros_like(spec)

        if self.model.run_df:
            df_coefs = self.model.df_dec(emb, c0)
            df_coefs = self.df_out_transform(df_coefs)
            spec_e = self.df_op(spec.clone(), df_coefs)
            spec_e[..., self.model.nb_df :, :] = spec_m[..., self.model.nb_df :, :]
        else:
            spec_e = spec_m
        return spec_e


class SpectralEnhancer(nn.Module):
    """Wrap DeepFilterNet so the ONNX graph consumes and returns spectra only."""

    def __init__(self, model: nn.Module, erb_fb_widths: np.ndarray, sr: int, nb_df: int, alpha: float):
        super().__init__()
        self.model = OnnxExportWrapper(model)
        self.features = SpectralFeatures(erb_fb_widths, sr, nb_df, alpha)

    def forward(self, spec: Tensor) -> Tuple[Tensor]:
        feat_erb, feat_spec = self.features(spec)
        enh = self.model(spec, feat_erb, feat_spec)
        return (enh,)


def _vorbis_window(fft_size: int) -> Tensor:
    """Vorbis window: sin(π/2 · sin²(π·(n+0.5)/N_h)) where N_h = fft_size/2.

    Exact window used by libdf analysis/synthesis (see libDF/src/lib.rs).
    Satisfies COLA at 50 % overlap (hop = fft_size/2): sum_t w[n-t*hop]² = 1.
    """
    N_h = fft_size // 2
    n = torch.arange(fft_size, dtype=torch.float32)
    sin_v = torch.sin(torch.pi * (n + 0.5) / N_h / 2.0)
    return torch.sin(torch.pi / 2.0 * sin_v * sin_v)


def _build_dft_kernels(fft_size: int, hop_size: int, window: Tensor) -> Tuple[Tensor, Tensor]:
    """Build analysis and synthesis DFT filter-bank kernels.

    Using F.conv1d / F.conv_transpose1d with these fixed kernels gives an
    ONNX-exportable STFT/iSTFT for dynamic-length inputs (avoids aten::unfold).

    Analysis kernel shape:   [2*F, 1, fft_size]
    Synthesis kernel shape:  [2*F, 1, fft_size]

    Channels 0..F-1 carry real parts; channels F..2F-1 carry imaginary parts.

    The analysis kernel includes the wnorm = 2·hop/N² factor so that
    conv1d output matches df.analysis output exactly.
    The synthesis kernel undoes wnorm so that conv_transpose1d output is audio.
    """
    N = fft_size
    F_bins = N // 2 + 1
    wnorm = 2.0 * hop_size / (N * N)

    k = torch.arange(F_bins, dtype=torch.float64)
    n = torch.arange(N, dtype=torch.float64)
    angle = 2.0 * torch.pi * k.unsqueeze(1) * n.unsqueeze(0) / N  # [F, N]
    win = window.double()

    # Analysis: wnorm · window(n) · [cos, −sin]
    A_re = (wnorm * win) * torch.cos(angle)   # [F, N]
    A_im = (wnorm * win) * (-torch.sin(angle))  # [F, N]
    analysis_kernel = torch.cat([A_re, A_im], dim=0).unsqueeze(1).float()  # [2F, 1, N]

    # Synthesis: (scale[k] / (N·wnorm)) · window(n) · [cos, −sin]
    # scale[k] = 1 for DC and Nyquist, 2 otherwise (one-sided spectrum factor)
    scale = torch.ones(F_bins, dtype=torch.float64)
    scale[1 : F_bins - 1] = 2.0
    factor = scale / (N * wnorm)  # [F]
    S_re = (factor.unsqueeze(1) * win) * torch.cos(angle)   # [F, N]
    S_im = (factor.unsqueeze(1) * win) * (-torch.sin(angle))  # [F, N]
    synthesis_kernel = torch.cat([S_re, S_im], dim=0).unsqueeze(1).float()  # [2F, 1, N]

    return analysis_kernel, synthesis_kernel


class WaveformEnhancer(nn.Module):
    """End-to-end model: time-domain audio in → enhanced audio out.

    Wraps SpectralEnhancer with the Vorbis-windowed STFT/iSTFT that exactly matches
    the libdf analysis/synthesis pair, so ONNX outputs are numerically identical to
    the Python ``df.enhance.enhance()`` pipeline.

    STFT alignment with libdf
    -------------------------
    libdf analysis initialises its overlap buffer (``analysis_mem``) to zeros and
    processes the signal hop-by-hop.  This is equivalent to left-padding the signal
    with ``fft_size - hop_size`` zeros and then running a causal STFT (center=False).

    libdf applies an unnormalised FFT and multiplies each output frame by
    ``wnorm = 2·hop / fft_size²``.  ``torch.stft`` does not apply wnorm, so we scale
    the spectrum by wnorm before feeding SpectralEnhancer (which was trained on
    df-scale spectra) and divide by wnorm before iSTFT.

    ``torch.istft`` with the Vorbis window and COLA (50 % overlap) normalises by
    sum(window²) = 1, giving exact reconstruction without any extra scaling.

    Input:  audio  [1, T]  — mono float32, 48 kHz
    Output: enh    [1, T]  — enhanced audio, same shape
    """

    def __init__(self, spec_enhancer: "SpectralEnhancer", fft_size: int, hop_size: int):
        super().__init__()
        self.spec_enhancer = spec_enhancer
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.pad_left: int = fft_size - hop_size
        window = _vorbis_window(fft_size)
        self.register_buffer("window", window)
        a_kern, s_kern = _build_dft_kernels(fft_size, hop_size, window)
        self.register_buffer("analysis_kernel", a_kern)    # [2F, 1, fft_size]
        self.register_buffer("synthesis_kernel", s_kern)   # [2F, 1, fft_size]

    def forward(self, audio: Tensor) -> Tensor:
        # audio: [1, T]
        T = audio.shape[-1]
        F_bins = self.fft_size // 2 + 1

        # Left-pad to replicate libdf zero-initialised analysis_mem
        padded = F.pad(audio, (self.pad_left, 0))  # [1, T+pad]

        # Analysis via DFT filter bank → [1, 2*F, T_frames]
        spec_conv = F.conv1d(padded.unsqueeze(1), self.analysis_kernel, stride=self.hop_size)

        # Reshape to [1, 1, T_frames, F, 2] expected by SpectralEnhancer
        re = spec_conv[:, :F_bins, :]                           # [1, F, T_frames]
        im = spec_conv[:, F_bins:, :]                           # [1, F, T_frames]
        spec_5d = torch.stack([re, im], dim=-1).permute(0, 2, 1, 3).unsqueeze(1)

        # Enhance
        (enh_df,) = self.spec_enhancer(spec_5d)                 # [1, 1, T_frames, F, 2]

        # Reshape back to [1, 2*F, T_frames]
        enh = enh_df.squeeze(1)                                  # [1, T_frames, F, 2]
        enh_re = enh[:, :, :, 0].permute(0, 2, 1)               # [1, F, T_frames]
        enh_im = enh[:, :, :, 1].permute(0, 2, 1)               # [1, F, T_frames]
        enh_conv = torch.cat([enh_re, enh_im], dim=1)            # [1, 2*F, T_frames]

        # Synthesis via transposed DFT filter bank (OLA baked in) → [1, 1, T_out]
        audio_out = F.conv_transpose1d(enh_conv, self.synthesis_kernel, stride=self.hop_size)

        return audio_out[:, 0:1, :T]  # [1, 1, T] — trim synthesis tail, keep channel dim


def onnx_simplify(
    path: str, input_data: Dict[str, Tensor], input_shapes: Dict[str, Iterable[int]]
) -> str:
    import onnxsim

    model = onnx.load(path)
    model_simp, check = onnxsim.simplify(
        model,
        input_data=input_data,
        test_input_shapes=input_shapes,
    )
    model_n = os.path.splitext(os.path.basename(path))[0]
    assert check, "Simplified ONNX model could not be validated"
    logger.debug(model_n + ": " + onnx.helper.printable_graph(model.graph))
    try:
        onnx.checker.check_model(model_simp, full_check=True)
    except Exception as e:
        logger.error(f"Failed to simplify model {model_n}. Skipping: {e}")
        return path
    # new_path = os.path.join(os.path.dirname(path), model_n + "_simplified.onnx")
    onnx.save_model(model_simp, path)
    return path


def onnx_check(path: str, input_dict: Dict[str, Tensor], output_names: Tuple[str]):
    model = onnx.load(path)
    logger.debug(os.path.basename(path) + ": " + onnx.helper.printable_graph(model.graph))
    onnx.checker.check_model(model, full_check=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(output_names, {k: v.numpy() for (k, v) in input_dict.items()})


def export_impl(
    path: str,
    model: torch.nn.Module,
    inputs: Tuple[Tensor, ...],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    jit: bool = True,
    opset_version=14,
    check: bool = True,
    simplify: bool = True,
    print_graph: bool = False,
):
    export_dir = os.path.dirname(path)
    if not os.path.isdir(export_dir):
        logger.info(f"Creating export directory: {export_dir}")
        os.makedirs(export_dir)
    model_name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"Exporting model '{model_name}' to {export_dir}")

    input_shapes = shapes_dict(inputs, input_names)
    logger.info(f"  Input shapes: {input_shapes}")

    outputs = ensure_tuple(model(*inputs))
    output_shapes = shapes_dict(outputs, output_names)
    logger.info(f"  Output shapes: {output_shapes}")

    if jit:
        model = torch.jit.script(model, example_inputs=[tuple(a for a in inputs)])

    logger.info(f"  Dynamic axis: {dynamic_axes}")
    torch.onnx.export(
        model=deepcopy(model),
        f=path,
        args=inputs,
        input_names=input_names,
        dynamic_axes=dynamic_axes,
        output_names=output_names,
        opset_version=opset_version,
        keep_initializers_as_inputs=False,
        dynamo=False,
    )

    input_dict = {k: v for (k, v) in zip(input_names, inputs)}
    if check:
        onnx_outputs = onnx_check(path, input_dict, tuple(output_names))
        for name, out, onnx_out in zip(output_names, outputs, onnx_outputs):
            try:
                np.testing.assert_allclose(
                    out.numpy().squeeze(), onnx_out.squeeze(), rtol=1e-6, atol=1e-5
                )
            except AssertionError as e:
                logger.warning(f"  Elements not close for {name}: {e}")
    if simplify:
        path = onnx_simplify(path, input_dict, shapes_dict(inputs, input_names))
        logger.info(f"  Saved simplified model {path}")
    if print_graph:
        onnx.helper.printable_graph(onnx.load_model(path).graph)

    return outputs


@torch.no_grad()
def export(
    model,
    export_dir: str,
    df_state: DF,
    check: bool = True,
    simplify: bool = True,
    opset=14,
    export_spec: bool = True,
    export_waveform: bool = False,
    export_full: bool = False,
    export_components: bool = True,
    print_graph: bool = False,
):
    """Export DeepFilterNet to ONNX.

    Three output modes (flags are independent and additive):

    export_spec (default True):
        Produces deepfilternet_spec.onnx — a single ONNX graph that takes a complex
        spectrogram and returns the enhanced spectrogram.  Use this when your pipeline
        already runs in the frequency domain and you do not want to pay for an FFT/iFFT
        inside the model.
        Input:  spec  [1, 1, T, F, 2]  — complex STFT frames, last dim is [re, im].
        Output: enh   [1, 1, T, F, 2]  — enhanced complex STFT frames.
        Feature extraction (ERB filterbank, normalisation) is baked into the graph so
        no pre-processing is needed beyond a standard STFT.

    export_waveform (default False):
        Produces deepfilternet_v3.onnx — a single end-to-end graph that takes raw
        time-domain audio and returns enhanced audio at the same length.
        Input:  audio  [1, T]  — mono float32 at 48 kHz.
        Output: enh    [1, T]  — enhanced mono audio, same shape.
        Internally uses the Vorbis-windowed STFT/iSTFT that exactly matches libdf so
        outputs are numerically identical to df.enhance.enhance() (MAE < 1e-6).
        Requires opset 17 (ONNX STFT operator).  Produces a reference .npz pair.

    export_full (default False):
        Produces deepfilternet2.onnx — the raw model that expects pre-computed features.
        Use when you need access to intermediate outputs (mask, lsnr, coefs).
        Inputs:  spec [1,1,T,F,2], feat_erb [1,1,T,E], feat_spec [1,1,T,F',2]
        Outputs: enh, m, lsnr, coefs

    export_components (default True):
        Produces enc.onnx, erb_dec.onnx, df_dec.onnx — the three sub-networks exported
        individually with matching .npz reference tensors.  Use when you need fine-grained
        control or want to run sub-networks separately in a streaming pipeline.

    All modes save reference .npz files alongside the ONNX files so you can verify
    numerics independently of the exporter.
    """
    model = deepcopy(model).to("cpu")
    model.eval()
    p = ModelParams()
    audio = torch.randn((1, 1 * p.sr))
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")

    # --- Spectrum-in / spectrum-out model ------------------------------------------
    if export_spec:
        alpha = get_norm_alpha(log=False)
        spec_wrapper = SpectralEnhancer(
            model,
            erb_fb_widths=df_state.erb_widths(),
            sr=p.sr,
            nb_df=p.nb_df,
            alpha=alpha,
        ).to("cpu")

        path = os.path.join(export_dir, "deepfilternet_spec.onnx")
        inputs = (spec,)
        input_names = ["spec"]
        dynamic_axes = {
            "spec": {2: "S"},
            "enh": {2: "S"},
        }
        output_names = ["enh"]
        (enh_spec,) = export_impl(
            path,
            spec_wrapper,
            inputs=inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            jit=False,  # trace-based: avoids dtype loss from TorchScript BatchNorm, unrolls df_order loop
            check=check,
            simplify=simplify,
            opset_version=opset,
            print_graph=print_graph,
        )
        np.savez_compressed(os.path.join(export_dir, "spec_input.npz"), spec=spec.numpy())
        np.savez_compressed(os.path.join(export_dir, "spec_output.npz"), enh=enh_spec.numpy())

    # --- Waveform end-to-end model (audio-in / audio-out) -------------------------
    if export_waveform:
        alpha = get_norm_alpha(log=False)
        spec_wrapper = SpectralEnhancer(
            model,
            erb_fb_widths=df_state.erb_widths(),
            sr=p.sr,
            nb_df=p.nb_df,
            alpha=alpha,
        ).to("cpu")
        wav_wrapper = WaveformEnhancer(spec_wrapper, fft_size=p.fft_size, hop_size=p.hop_size).to("cpu")
        audio = torch.randn((1, p.sr))  # 1-second test signal  [1, T]
        path = os.path.join(export_dir, "deepfilternet_v3.onnx")
        input_names = ["audio"]
        output_names = ["enh"]
        dynamic_axes = {"audio": {1: "T"}, "enh": {2: "T"}}
        (enh_wav,) = export_impl(
            path,
            wav_wrapper,
            inputs=(audio,),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            jit=False,
            check=check,
            simplify=False,  # onnxsim does not support large DFT kernels
            opset_version=max(opset, 14),
            print_graph=print_graph,
        )
        # enh_wav shape: [1, 1, T] — squeeze to [1, T] for the reference file
        np.savez_compressed(os.path.join(export_dir, "wav_input.npz"), audio=audio.numpy())
        np.savez_compressed(os.path.join(export_dir, "wav_output.npz"), enh=enh_wav.squeeze(1).numpy())

    # --- Full monolithic model (features passed externally) -------------------------
    if export_full:
        path = os.path.join(export_dir, "deepfilternet2.onnx")
        input_names = ["spec", "feat_erb", "feat_spec"]
        dynamic_axes = {
            "spec": {2: "S"},
            "feat_erb": {2: "S"},
            "feat_spec": {2: "S"},
            "enh": {2: "S"},
            "m": {2: "S"},
            "lsnr": {1: "S"},
        }
        inputs = (spec, feat_erb, feat_spec)
        output_names = ["enh", "m", "lsnr", "coefs"]
        export_impl(
            path,
            model,
            inputs=inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            jit=False,
            check=check,
            simplify=simplify,
            opset_version=opset,
            print_graph=print_graph,
        )

    # --- Individual sub-network components ----------------------------------------
    if not export_components:
        return

    # Export encoder
    feat_spec = feat_spec.transpose(1, 4).squeeze(4)  # re/im into channel axis
    path = os.path.join(export_dir, "enc.onnx")
    inputs = (feat_erb, feat_spec)
    input_names = ["feat_erb", "feat_spec"]
    dynamic_axes = {
        "feat_erb": {2: "S"},
        "feat_spec": {2: "S"},
        "e0": {2: "S"},
        "e1": {2: "S"},
        "e2": {2: "S"},
        "e3": {2: "S"},
        "emb": {1: "S"},
        "c0": {2: "S"},
        "lsnr": {1: "S"},
    }
    output_names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]
    e0, e1, e2, e3, emb, c0, lsnr = export_impl(
        path,
        model.enc,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=True,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    np.savez_compressed(
        os.path.join(export_dir, "enc_input.npz"),
        feat_erb=feat_erb.numpy(),
        feat_spec=feat_spec.numpy(),
    )
    np.savez_compressed(
        os.path.join(export_dir, "enc_output.npz"),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
        emb=emb.numpy(),
        c0=c0.numpy(),
        lsnr=lsnr.numpy(),
    )

    # Export erb decoder
    np.savez_compressed(
        os.path.join(export_dir, "erb_dec_input.npz"),
        emb=emb.numpy(),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
    )
    inputs = (emb.clone(), e3, e2, e1, e0)
    input_names = ["emb", "e3", "e2", "e1", "e0"]
    output_names = ["m"]
    dynamic_axes = {
        "emb": {1: "S"},
        "e3": {2: "S"},
        "e2": {2: "S"},
        "e1": {2: "S"},
        "e0": {2: "S"},
        "m": {2: "S"},
    }
    path = os.path.join(export_dir, "erb_dec.onnx")
    (m,) = export_impl(
        path,
        model.erb_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=True,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    np.savez_compressed(os.path.join(export_dir, "erb_dec_output.npz"), m=m.numpy())

    # Export df decoder
    np.savez_compressed(
        os.path.join(export_dir, "df_dec_input.npz"), emb=emb.numpy(), c0=c0.numpy()
    )
    inputs = (emb.clone(), c0)
    input_names = ["emb", "c0"]
    output_names = ["coefs"]
    dynamic_axes = {
        "emb": {1: "S"},
        "c0": {2: "S"},
        "coefs": {1: "S"},
    }
    path = os.path.join(export_dir, "df_dec.onnx")
    (coefs,) = export_impl(
        path,
        model.df_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        jit=False,
        check=check,
        simplify=simplify,
        opset_version=opset,
        print_graph=print_graph,
    )
    np.savez_compressed(os.path.join(export_dir, "df_dec_output.npz"), coefs=coefs.numpy())


def main(args):
    try:
        import monkeytype  # noqa: F401
    except ImportError:
        print("Failed to import monkeytype. Please install it via")
        print("$ pip install MonkeyType")
        exit(1)

    print(args)
    model, df_state, _, epoch = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        log_file="export.log",
        config_allow_defaults=True,
        epoch=args.epoch,
    )
    sample = get_test_sample(df_state.sr())
    enhanced = enhance(model, df_state, sample, True)
    out_dir = Path("out")
    if out_dir.is_dir():
        # attempt saving enhanced audio
        save_audio(os.path.join(out_dir, "enhanced.wav"), enhanced, df_state.sr())
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export(
        model,
        str(export_dir),
        df_state=df_state,
        opset=args.opset,
        check=args.check,
        simplify=args.simplify,
        export_spec=args.export_spec,
        export_waveform=args.export_waveform,
        export_full=args.export_full,
        export_components=args.export_components,
    )
    model_base_dir = get_model_basedir(args.model_base_dir)
    if model_base_dir != args.export_dir:
        shutil.copyfile(
            os.path.join(model_base_dir, "config.ini"),
            os.path.join(args.export_dir, "config.ini"),
        )
    model_name = Path(model_base_dir).name
    version_file = os.path.join(args.export_dir, "version.txt")
    with open(version_file, "w") as f:
        f.write(f"{model_name}_epoch_{epoch}")
    tar_name = export_dir / (Path(model_base_dir).name + "_onnx.tar.gz")
    with tarfile.open(tar_name, mode="w:gz") as f:
        if args.export_waveform:
            f.add(os.path.join(args.export_dir, "deepfilternet_v3.onnx"))
        if args.export_spec:
            f.add(os.path.join(args.export_dir, "deepfilternet_spec.onnx"))
        if args.export_components:
            f.add(os.path.join(args.export_dir, "enc.onnx"))
            f.add(os.path.join(args.export_dir, "erb_dec.onnx"))
            f.add(os.path.join(args.export_dir, "df_dec.onnx"))
        f.add(os.path.join(args.export_dir, "config.ini"))
        f.add(os.path.join(args.export_dir, "version.txt"))


if __name__ == "__main__":
    parser = setup_df_argument_parser()
    parser.add_argument("export_dir", help="Directory for exporting the onnx model.")
    parser.add_argument(
        "--no-check",
        help="Don't check models with onnx checker.",
        action="store_false",
        dest="check",
    )
    parser.add_argument("--simplify", help="Simply onnx models using onnxsim.", action="store_true")
    parser.add_argument("--opset", help="ONNX opset version", type=int, default=14)
    parser.add_argument(
        "--export-waveform",
        help="Export end-to-end waveform model (deepfilternet_v3.onnx). "
        "Takes raw audio [1,T] at 48 kHz, returns enhanced audio [1,T]. "
        "Uses the Vorbis-windowed STFT/iSTFT matching libdf exactly. Requires opset 17.",
        action="store_true",
        dest="export_waveform",
    )
    parser.add_argument(
        "--export-spec",
        help="Export spectrum-in/spectrum-out model (deepfilternet_spec.onnx). "
        "Takes complex STFT frames [1,1,T,F,2], returns enhanced frames of same shape. "
        "Feature extraction is baked in; no pre-processing needed beyond a standard STFT.",
        action="store_true",
        dest="export_spec",
        default=True,
    )
    parser.add_argument(
        "--no-export-spec",
        help="Skip the spectrum-in/spectrum-out model export.",
        action="store_false",
        dest="export_spec",
    )
    parser.add_argument(
        "--export-full",
        help="Export the full monolithic model (deepfilternet2.onnx) that takes "
        "pre-computed features: spec [1,1,T,F,2], feat_erb [1,1,T,E], feat_spec [1,1,T,F',2]. "
        "Useful when you need intermediate outputs (mask, lsnr, coefs).",
        action="store_true",
        dest="export_full",
    )
    parser.add_argument(
        "--no-export-components",
        help="Skip per-component export (enc.onnx, erb_dec.onnx, df_dec.onnx).",
        action="store_false",
        dest="export_components",
    )
    parser.set_defaults(export_components=True)
    args = parser.parse_args()
    main(args)
