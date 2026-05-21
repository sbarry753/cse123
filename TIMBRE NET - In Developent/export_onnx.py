"""
Export a model.py checkpoint to fixed-shape ONNX for TensorRT conversion.

Example:
  python export_onnx.py \
    --checkpoint checkpoints_teach/best_model.pt \
    --output model.onnx

Then build a TensorRT engine on Orin, for example:
  trtexec --onnx=model.onnx --saveEngine=model_fp16.plan --fp16
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
import onnxruntime as ort

from model import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, N_FFT

def checkpoint_training_args(payload):
    if not isinstance(payload, dict):
        return {}
    training_args = payload.get("training_args")
    if isinstance(training_args, dict):
        return training_args
    return payload

def checkpoint_value(payload, name, default=None):
    training_args = checkpoint_training_args(payload)
    if name in training_args and training_args[name] is not None:
        return training_args[name]
    if isinstance(payload, dict):
        return payload.get(name, default)
    return default

def checkpoint_state(payload):
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return {key: value for key, value in state.items() if key != "window"}

def load_checkpoint_model(path: Path, args):
    payload = torch.load(path, map_location="cpu", weights_only=False)

    frame_size = int(args.frame_size or checkpoint_value(payload, "frame_size", FRAME_SIZE))
    output_size = args.output_size if args.output_size is not None else checkpoint_value(payload, "output_size", None)
    output_size = int(output_size) if output_size is not None else None
    hop_size = int(args.hop_size or checkpoint_value(payload, "hop_size", HOP_SIZE))
    n_fft = int(args.n_fft or checkpoint_value(payload, "n_fft", N_FFT))
    win_length = int(args.win_length or checkpoint_value(payload, "win_length", n_fft))
    hidden_size = int(args.hidden_size or checkpoint_value(payload, "hidden_size", 256))
    base_ch = int(args.base_ch or checkpoint_value(payload, "base_ch", 64))
    phase_tcn_ch = int(args.phase_tcn_ch or checkpoint_value(payload, "phase_tcn_ch", 16))
    phase_tcn_layers = int(args.phase_tcn_layers or checkpoint_value(payload, "phase_tcn_layers", 3))
    phase_max_delta = float(
        args.phase_max_delta
        if args.phase_max_delta is not None
        else checkpoint_value(payload, "phase_max_delta", 0.10)
    )

    model = DDSPGuitarToPiano(
        sample_rate=SAMPLE_RATE,
        frame_size=frame_size,
        output_size=output_size,
        hop_size=hop_size,
        n_fft=n_fft,
        win_length=win_length,
        hidden_size=hidden_size,
        base_ch=base_ch,
        phase_tcn_ch=phase_tcn_ch,
        phase_tcn_layers=phase_tcn_layers,
        phase_max_delta=phase_max_delta,
        transient_max_gain=0.2,
    )

    state = checkpoint_state(payload)
    if args.allow_partial_load:
        current = model.state_dict()
        compatible = {}
        skipped = []
        for key, value in state.items():
            if key in current and current[key].shape != value.shape:
                skipped.append(key)
                continue
            compatible[key] = value
        result = model.load_state_dict(compatible, strict=False)
        if skipped:
            print(f"Skipped incompatible checkpoint tensors: {skipped}")
        if result.missing_keys:
            print(f"Missing model keys initialized from defaults: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Ignored unexpected checkpoint keys: {result.unexpected_keys}")
    else:
        model.load_state_dict(state, strict=True)

    model.eval()
    config = {
        "sample_rate": SAMPLE_RATE,
        "frame_size": frame_size,
        "output_size": int(output_size or frame_size),
        "hop_size": hop_size,
        "n_fft": n_fft,
        "win_length": win_length,
        "hidden_size": hidden_size,
        "base_ch": base_ch,
        "phase_tcn_ch": phase_tcn_ch,
        "phase_tcn_layers": phase_tcn_layers,
        "phase_max_delta": phase_max_delta,
    }
    return model, config


def padded_hann_window(win_length: int, n_fft: int) -> torch.Tensor:
    window = torch.hann_window(win_length, periodic=True)
    if win_length == n_fft:
        return window
    if win_length > n_fft:
        raise ValueError(f"win_length must be <= n_fft, got {win_length} > {n_fft}")
    left = (n_fft - win_length) // 2
    right = n_fft - win_length - left
    return F.pad(window, (left, right))


class ConvSTFT(nn.Module):
    def __init__(self, n_fft: int, hop_size: int, win_length: int):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_size = int(hop_size)
        self.pad = self.n_fft // 2

        window = padded_hann_window(win_length, self.n_fft)
        n = torch.arange(self.n_fft, dtype=torch.float32)
        k = torch.arange(self.n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        omega = 2.0 * math.pi * k * n / float(self.n_fft)

        cos_kernel = torch.cos(omega) * window
        sin_kernel = -torch.sin(omega) * window
        weight = torch.cat([cos_kernel, sin_kernel], dim=0).unsqueeze(1)
        self.register_buffer("weight", weight)

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = audio.unsqueeze(1)
        x = F.pad(x, (self.pad, self.pad), mode="reflect")
        spec = F.conv1d(x, self.weight, stride=self.hop_size)
        n_freq = self.n_fft // 2 + 1
        return spec[:, :n_freq, :], spec[:, n_freq:, :]


class ConvISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_size: int, win_length: int, signal_length: int):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_size = int(hop_size)
        self.pad = self.n_fft // 2
        self.signal_length = int(signal_length)

        window = padded_hann_window(win_length, self.n_fft)
        n = torch.arange(self.n_fft, dtype=torch.float32)
        k = torch.arange(self.n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        omega = 2.0 * math.pi * k * n / float(self.n_fft)

        scale = torch.full((self.n_fft // 2 + 1, 1), 2.0 / float(self.n_fft))
        scale[0, 0] = 1.0 / float(self.n_fft)
        if self.n_fft % 2 == 0:
            scale[-1, 0] = 1.0 / float(self.n_fft)

        cos_basis = torch.cos(omega) * scale * window
        sin_basis = -torch.sin(omega) * scale * window
        weight = torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1)
        self.register_buffer("weight", weight)

        padded_len = self.signal_length + 2 * self.pad
        n_frames = (padded_len - self.n_fft) // self.hop_size + 1
        out_len = (n_frames - 1) * self.hop_size + self.n_fft

        norm = torch.zeros(out_len)
        w2 = window.square()
        for idx in range(n_frames):
            start = idx * self.hop_size
            norm[start : start + self.n_fft] += w2
        norm = torch.clamp(norm, min=1.0e-8)
        self.register_buffer("norm", norm[self.pad : self.pad + self.signal_length])

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        spec = torch.cat([real, imag], dim=1)
        audio = F.conv_transpose1d(spec, self.weight, stride=self.hop_size)
        audio = audio[..., self.pad : self.pad + self.signal_length]
        audio = audio / self.norm
        return audio.squeeze(1)


class ExportableTimbreNet(nn.Module):
    """
    Real-valued export wrapper for DDSPGuitarToPiano.

    It mirrors model.py while replacing torch.stft/istft/complex polar ops with
    Conv1d/ConvTranspose1d and real-valued phase arithmetic for TensorRT.
    """

    def __init__(self, source: DDSPGuitarToPiano):
        super().__init__()
        self.frame_size = int(source.frame_size)
        self.output_size = int(source.output_size)
        self.n_fft = int(source.n_fft)
        self.hop_size = int(source.hop_size)
        self.win_length = int(source.win_length)
        self.unet = source.unet
        self.phase_tcn = source.phase_tcn
        self.transient_correction = source.transient_correction
        self.stft = ConvSTFT(self.n_fft, self.hop_size, self.win_length)
        self.istft = ConvISTFT(self.n_fft, self.hop_size, self.win_length, self.frame_size)

    def forward(self, audio_frame: torch.Tensor) -> torch.Tensor:
        real, imag = self.stft(audio_frame)
        mag = torch.sqrt(real * real + imag * imag + 1.0e-12)
        mag_clamped = torch.clamp(mag, min=1.0e-5)
        log_mag = torch.log(mag_clamped).unsqueeze(1)

        mask, residual = self.unet(log_mag)
        out_log_mag = log_mag * mask + residual
        out_mag = torch.exp(out_log_mag.squeeze(1))

        cos_phase = real / mag_clamped
        sin_phase = imag / mag_clamped

        sin_t0 = sin_phase[..., 1:] * cos_phase[..., :-1] - cos_phase[..., 1:] * sin_phase[..., :-1]
        cos_t0 = cos_phase[..., 1:] * cos_phase[..., :-1] + sin_phase[..., 1:] * sin_phase[..., :-1]
        sin_phase_dt = F.pad(sin_t0, (1, 0))
        cos_phase_dt = F.pad(cos_t0, (1, 0), value=1.0)

        sin_f0 = sin_phase[:, 1:, :] * cos_phase[:, :-1, :] - cos_phase[:, 1:, :] * sin_phase[:, :-1, :]
        cos_f0 = cos_phase[:, 1:, :] * cos_phase[:, :-1, :] + sin_phase[:, 1:, :] * sin_phase[:, :-1, :]
        sin_phase_df = F.pad(sin_f0, (0, 0, 1, 0))
        cos_phase_df = F.pad(cos_f0, (0, 0, 1, 0), value=1.0)
        phase_input = torch.cat(
            [
                log_mag,
                out_log_mag,
                mask,
                residual,
                sin_phase.unsqueeze(1),
                cos_phase.unsqueeze(1),
                sin_phase_dt.unsqueeze(1),
                cos_phase_dt.unsqueeze(1),
                sin_phase_df.unsqueeze(1),
                cos_phase_df.unsqueeze(1),
            ],
            dim=1,
        )
        phase_delta, _ = self.phase_tcn(phase_input)
        phase_delta = phase_delta.squeeze(1)

        cos_delta = torch.cos(phase_delta)
        sin_delta = torch.sin(phase_delta)
        out_cos = cos_phase * cos_delta - sin_phase * sin_delta
        out_sin = sin_phase * cos_delta + cos_phase * sin_delta
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin

        audio_before_transient = self.istft(out_real, out_imag)
        transient_delta = self.transient_correction(audio_frame, audio_before_transient)
        audio_out = audio_before_transient + transient_delta
        output_size = min(self.output_size, self.frame_size)
        return audio_out[..., -output_size:]


def export_onnx(model: nn.Module, output_path: Path, frame_size: int, opset: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, frame_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["audio_in"],
        output_names=["audio_out"],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    return dummy

def main():
    parser = argparse.ArgumentParser(description="Export model.py checkpoint to ONNX for TensorRT")
    parser.add_argument("--checkpoint", "--ckpt", required=True, type=Path, help="Path to best_model.pt")
    parser.add_argument("--output", "--out", default=Path("model.onnx"), type=Path, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--allow-partial-load", action="store_true", help="Skip checkpoint tensors with mismatched shapes")
    parser.add_argument("--skip-parity", action="store_true", help="Skip PyTorch wrapper parity check")
    parser.add_argument("--skip-onnxruntime-check", action="store_true", help="Skip optional ONNX Runtime check")
    parser.add_argument("--parity-warn", type=float, default=1.0e-3, help="Mean abs diff threshold for warning")
    parser.add_argument("--frame_size", type=int, default=None)
    parser.add_argument("--output_size", type=int, default=None, help="Override checkpoint output_size")
    parser.add_argument("--hop_size", type=int, default=None)
    parser.add_argument("--n_fft", type=int, default=None)
    parser.add_argument("--win_length", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--base_ch", type=int, default=None)
    parser.add_argument("--phase_tcn_ch", type=int, default=None)
    parser.add_argument("--phase_tcn_layers", type=int, default=None)
    parser.add_argument("--phase_max_delta", type=float, default=None)
    args = parser.parse_args()

    source, config = load_checkpoint_model(args.checkpoint, args)
    if args.output_size is not None and int(args.output_size) != config["output_size"]:
        raise ValueError(
            f"CLI --output_size={args.output_size} does not match checkpoint/model output_size={config['output_size']}"
        )

    exportable = ExportableTimbreNet(source).eval()
    dummy = torch.randn(1, config["frame_size"], dtype=torch.float32)

    if not args.skip_parity:
        with torch.no_grad():
            reference, _, _ = source(dummy)
            wrapped = exportable(dummy)
            diff = (reference - wrapped).abs().mean().item()
            max_diff = (reference - wrapped).abs().max().item()
        print(f"PyTorch wrapper parity: mean_abs={diff:.6e} max_abs={max_diff:.6e}")
        if diff > args.parity_warn:
            print(f"WARNING: wrapper parity mean_abs exceeds {args.parity_warn:g}")

    dummy = export_onnx(exportable, args.output, config["frame_size"], args.opset)
    print(f"Wrote ONNX: {args.output}")
    print(f"Input shape : (1, {config['frame_size']})")
    print(f"Output shape: (1, {config['output_size']})")

    meta_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata = {
        "checkpoint": str(args.checkpoint),
        "onnx": str(args.output),
        "opset": args.opset,
        "config": config,
        "trtexec_example": f"trtexec --onnx={args.output} --saveEngine={args.output.with_suffix('.plan')} --fp16",
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metadata: {meta_path}")

    if not args.skip_onnxruntime_check:
        sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        ort_out = sess.run(["audio_out"], {"audio_in": dummy.numpy()})[0]
        with torch.no_grad():
            torch_out = exportable(dummy).numpy()
        ort_diff = abs(ort_out - torch_out).mean()
        print(f"ONNX Runtime check: mean_abs={ort_diff:.6e}")

if __name__ == "__main__":
    main()
