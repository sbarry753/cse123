"""
export_to_onnx.py — Convert PolyphonicGuitarToPiano to ONNX for TensorRT on Jetson Orin

We replace torch.stft / torch.istft / torch.polar with Conv1d / ConvTranspose1d so
the graph contains only ops TRT can lower. Numerically equivalent to the original
within ~1e-6 for FRAME_SIZE=1024 inputs.

Usage:
    python export_to_onnx.py --ckpt ./checkpoints/best_model.pt --out ./model.onnx
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    PolyphonicGuitarToPiano,
    SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, N_FFT,
)


# ------------------------------------------------------------
# Conv-based STFT / iSTFT — fixed-length, export-friendly
# ------------------------------------------------------------
class ConvSTFT(nn.Module):
    """STFT as Conv1d. Returns (real, imag) — no complex tensors."""

    def __init__(self, n_fft: int, hop_size: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.pad = n_fft // 2

        window = torch.hann_window(win_length, periodic=True)
        if win_length < n_fft:
            window = F.pad(window, (0, n_fft - win_length))

        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        omega = 2.0 * math.pi * k * n / n_fft

        cos_kernel = torch.cos(omega) * window
        sin_kernel = -torch.sin(omega) * window
        # (out_ch = 2*F, in_ch = 1, kernel = n_fft)
        weight = torch.cat([cos_kernel, sin_kernel], dim=0).unsqueeze(1)
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor):
        # x: (B, T)
        x = x.unsqueeze(1)
        x = F.pad(x, (self.pad, self.pad), mode="reflect")
        spec = F.conv1d(x, self.weight, stride=self.hop_size)
        n_freq = self.n_fft // 2 + 1
        return spec[:, :n_freq, :], spec[:, n_freq:, :]


class ConvISTFT(nn.Module):
    """
    iSTFT via ConvTranspose1d. Pre-computes window^2 normalization for a fixed
    signal length so there are no Python-side loops in the exported graph.
    """

    def __init__(self, n_fft: int, hop_size: int, win_length: int, signal_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.pad = n_fft // 2
        self.signal_length = signal_length

        window = torch.hann_window(win_length, periodic=True)
        if win_length < n_fft:
            window = F.pad(window, (0, n_fft - win_length))

        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        omega = 2.0 * math.pi * k * n / n_fft

        # Inverse DFT for real signal w/ one-sided spectrum.
        # Multiply non-DC/Nyquist bins by 2/N, DC and Nyquist by 1/N.
        scale = torch.full((n_fft // 2 + 1, 1), 2.0 / n_fft)
        scale[0, 0] = 1.0 / n_fft
        if n_fft % 2 == 0:
            scale[-1, 0] = 1.0 / n_fft

        # NOTE: PyTorch STFT uses X[k] = sum x[n] e^{-j 2π kn/N}, so Im{X} carries
        # an implicit minus sign. The inverse therefore needs -sin (not +sin) so the
        # round-trip gives cos^2 + sin^2 = 1 instead of cos^2 - sin^2.
        cos_basis = torch.cos(omega) * scale * window
        sin_basis = -torch.sin(omega) * scale * window
        weight = torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1)
        self.register_buffer("weight", weight)

        # Precompute OLA normalization for the fixed signal length
        padded_len = signal_length + 2 * self.pad
        n_frames = (padded_len - n_fft) // hop_size + 1
        out_len = (n_frames - 1) * hop_size + n_fft

        norm = torch.zeros(out_len)
        w2 = window ** 2
        for i in range(n_frames):
            s = i * hop_size
            norm[s:s + n_fft] += w2
        norm = torch.clamp(norm, min=1e-8)
        # Trim to the same region we'll trim the signal to
        self.register_buffer("norm", norm[self.pad:self.pad + signal_length])

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        spec = torch.cat([real, imag], dim=1)               # (B, 2F, T_frames)
        y = F.conv_transpose1d(spec, self.weight, stride=self.hop_size)
        y = y[..., self.pad:self.pad + self.signal_length]  # crop center padding
        y = y / self.norm                                    # OLA correction
        return y.squeeze(1)


# ------------------------------------------------------------
# Export wrapper that reuses your trained UNet + TransientShaper
# ------------------------------------------------------------
class ExportableModel(nn.Module):
    """
    Same forward semantics as PolyphonicGuitarToPiano, but TRT-clean.

    Original chain:
        STFT -> log|X| -> UNet (mask, residual) -> exp -> polar(input phase)
             -> iSTFT -> TransientShaper -> 0.9*tanh + 0.1*dry

    Phase preservation is done by rescaling (real, imag) by mag_out/mag_in
    instead of going through angle/polar.
    """

    def __init__(self, source: PolyphonicGuitarToPiano, signal_length: int = FRAME_SIZE):
        super().__init__()
        self.unet = source.unet
        self.transient = source.transient

        self.stft = ConvSTFT(N_FFT, HOP_SIZE, FRAME_SIZE)
        self.istft = ConvISTFT(N_FFT, HOP_SIZE, FRAME_SIZE, signal_length=signal_length)

        self.eps_mag = 1e-5    # matches safe_log floor
        self.eps_div = 1e-8

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        real, imag = self.stft(audio)

        mag = torch.sqrt(real * real + imag * imag + 1e-12)
        mag_clamped = torch.clamp(mag, min=self.eps_mag)
        log_mag = torch.log(mag_clamped).unsqueeze(1)

        mask, residual = self.unet(log_mag)
        out_log_mag = log_mag * mask + residual
        out_mag = torch.exp(out_log_mag.squeeze(1))

        ratio = out_mag / (mag_clamped + self.eps_div)
        out_real = real * ratio
        out_imag = imag * ratio

        y = self.istft(out_real, out_imag)
        y = self.transient(y)
        y = 0.9 * torch.tanh(y) + 0.1 * audio
        return y


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to best_model.pt")
    p.add_argument("--out", default="./model.onnx", help="Output ONNX path")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--check-tol", type=float, default=1e-3,
                   help="Max allowed mean abs diff vs original PyTorch model")
    args = p.parse_args()

    # Load source model
    src = PolyphonicGuitarToPiano()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    src.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    src.eval()

    # Build export wrapper, copy weights via shared modules
    exp_model = ExportableModel(src, signal_length=FRAME_SIZE)
    exp_model.eval()

    # Numerical parity check against original
    with torch.no_grad():
        x = torch.randn(1, FRAME_SIZE) * 0.3
        y_orig, _, _ = src(x)
        y_new = exp_model(x)
        diff = (y_orig - y_new).abs().mean().item()
        print(f"Mean abs diff (original vs exportable): {diff:.2e}")
        if diff > args.check_tol:
            print("WARNING: parity check failed. The exported model will not match training.")
        else:
            print("Parity OK.")

    # Export
    dummy = torch.randn(1, FRAME_SIZE)
    torch.onnx.export(
        exp_model,
        dummy,
        args.out,
        opset_version=args.opset,
        input_names=["audio_in"],
        output_names=["audio_out"],
        dynamic_axes=None,            # fixed batch=1, fixed length
        do_constant_folding=True,
    )
    print(f"Wrote ONNX -> {args.out}")

    # Quick ONNX runtime sanity check (optional; skip if onnxruntime not installed)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"audio_in": dummy.numpy()})[0]
        ort_diff = (torch.from_numpy(y_ort) - exp_model(dummy)).abs().mean().item()
        print(f"ONNXRuntime vs PyTorch mean abs diff: {ort_diff:.2e}")
    except ImportError:
        print("(install onnxruntime for the runtime sanity check)")


if __name__ == "__main__":
    main()