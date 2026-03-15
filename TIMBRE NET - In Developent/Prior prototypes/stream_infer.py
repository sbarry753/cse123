import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# python stream_infer.py \
#   --checkpoint checkpoints_residual/latest.pt \
#   --input dataset_48k/input/example.wav \
#   --output out_piano.wav

# =========================================================
# Constants
# =========================================================
SAMPLE_RATE = 48000


# =========================================================
# Audio helpers
# =========================================================
def load_audio_mono(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.astype(np.float32)
    audio = audio.mean(axis=1)
    audio = torch.from_numpy(audio).unsqueeze(0)  # [1, T]

    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    return audio


def save_audio(path: str, audio: torch.Tensor, sample_rate: int = SAMPLE_RATE):
    """
    audio: [1, T] on CPU
    """
    audio = audio.clamp(-1.0, 1.0)
    sf.write(path, audio.squeeze(0).numpy(), sample_rate)


def compute_onset_envelope(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1, T]
    returns: [T]
    """
    d = torch.zeros_like(x)
    d[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()

    kernel = torch.ones(1, 1, 33, dtype=x.dtype, device=x.device) / 33.0
    d_sm = F.conv1d(d.unsqueeze(0), kernel, padding=16).squeeze(0)  # [1, T]
    return d_sm.squeeze(0)


# =========================================================
# Model blocks
# =========================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class CausalResBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        z = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        z = F.leaky_relu(self.norm2(self.conv2(z)), 0.2)
        return x + z


class CausalEncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=2):
        super().__init__()
        self.down = CausalConv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        self.res1 = CausalResBlock(out_ch, kernel_size=5, dilation=1)
        self.res2 = CausalResBlock(out_ch, kernel_size=5, dilation=2)
        self.res3 = CausalResBlock(out_ch, kernel_size=5, dilation=4)

    def forward(self, x):
        x = F.leaky_relu(self.down(x), 0.2)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class CausalUpsampleStage(nn.Module):
    def __init__(self, in_ch, out_ch, up_factor=2, kernel_size=5):
        super().__init__()
        self.up_factor = up_factor
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.smooth = CausalConv1d(out_ch, out_ch, kernel_size=kernel_size)
        self.res1 = CausalResBlock(out_ch, kernel_size=5, dilation=1)
        self.res2 = CausalResBlock(out_ch, kernel_size=5, dilation=2)
        self.res3 = CausalResBlock(out_ch, kernel_size=5, dilation=4)

    def forward(self, x):
        x = self.proj(x)
        x = x.repeat_interleave(self.up_factor, dim=-1)
        x = F.leaky_relu(self.smooth(x), 0.2)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class LatentProcessor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre = nn.ModuleList([
            CausalResBlock(channels, kernel_size=5, dilation=1),
            CausalResBlock(channels, kernel_size=5, dilation=2),
            CausalResBlock(channels, kernel_size=5, dilation=4),
            CausalResBlock(channels, kernel_size=5, dilation=8),
            CausalResBlock(channels, kernel_size=5, dilation=16),
        ])
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.post = nn.ModuleList([
            CausalResBlock(channels, kernel_size=5, dilation=1),
            CausalResBlock(channels, kernel_size=5, dilation=2),
            CausalResBlock(channels, kernel_size=5, dilation=4),
            CausalResBlock(channels, kernel_size=5, dilation=8),
        ])

    def forward(self, x, h0=None):
        for blk in self.pre:
            x = blk(x)

        x_seq = x.transpose(1, 2)
        x_seq, hN = self.gru(x_seq, h0)
        x = x_seq.transpose(1, 2)

        for blk in self.post:
            x = blk(x)

        return x, hN


# =========================================================
# Residual model
# =========================================================
class ResidualLiveCausalAutoencoder(nn.Module):
    def __init__(self, base_ch=64, block_size=96):
        super().__init__()
        self.block_size = block_size

        self.in_conv = CausalConv1d(2, base_ch, kernel_size=9)

        self.enc1 = CausalEncoderStage(base_ch,     base_ch * 2, kernel_size=7, stride=2)
        self.enc2 = CausalEncoderStage(base_ch * 2, base_ch * 4, kernel_size=7, stride=2)
        self.enc3 = CausalEncoderStage(base_ch * 4, base_ch * 4, kernel_size=7, stride=2)

        self.latent = LatentProcessor(base_ch * 4)

        self.dec1 = CausalUpsampleStage(base_ch * 4, base_ch * 4, up_factor=2)
        self.dec2 = CausalUpsampleStage(base_ch * 4, base_ch * 2, up_factor=2)
        self.dec3 = CausalUpsampleStage(base_ch * 2, base_ch,     up_factor=2)

        self.skip2_proj = nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=1)
        self.skip1_proj = nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=1)
        self.skip0_proj = nn.Conv1d(base_ch,     base_ch,     kernel_size=1)

        self.skip2_gain = nn.Parameter(torch.tensor(0.20))
        self.skip1_gain = nn.Parameter(torch.tensor(0.15))
        self.skip0_gain = nn.Parameter(torch.tensor(0.10))

        self.out_mid = CausalConv1d(base_ch, base_ch, kernel_size=5)
        self.out_conv = CausalConv1d(base_ch, 1, kernel_size=5)

        self.residual_gain = nn.Parameter(torch.tensor(0.25))

    def forward(self, x, onset, h0=None, return_state=False, return_residual=False):
        inp = torch.cat([x, onset], dim=1)

        x0 = F.leaky_relu(self.in_conv(inp), 0.2)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        y, hN = self.latent(x3, h0=h0)

        y = self.dec1(y)
        y = y + self.skip2_gain * self.skip2_proj(x2[:, :, -y.shape[-1]:])

        y = self.dec2(y)
        y = y + self.skip1_gain * self.skip1_proj(x1[:, :, -y.shape[-1]:])

        y = self.dec3(y)
        y = y + self.skip0_gain * self.skip0_proj(x0[:, :, -y.shape[-1]:])

        y = F.leaky_relu(self.out_mid(y), 0.2)
        residual = torch.tanh(self.out_conv(y))

        input_block = x[:, :, -self.block_size:]
        residual_block = residual[:, :, -self.block_size:]

        gain = torch.clamp(self.residual_gain, 0.0, 1.0)
        out = torch.clamp(input_block + gain * residual_block, -1.0, 1.0)

        if return_state and return_residual:
            return out, hN, residual_block
        if return_state:
            return out, hN
        if return_residual:
            return out, residual_block
        return out


# =========================================================
# Checkpoint loading
# =========================================================
def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    block_size = ckpt.get("block_size", 96)
    context_samples = ckpt.get("context_samples", 4096)
    base_channels = ckpt.get("base_channels", 64)

    model = ResidualLiveCausalAutoencoder(
        base_ch=base_channels,
        block_size=block_size,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, context_samples, block_size, ckpt


# =========================================================
# Streaming offline inference
# =========================================================
@torch.no_grad()
def run_inference(
    model: ResidualLiveCausalAutoencoder,
    audio_in: torch.Tensor,
    context_samples: int,
    block_size: int,
    device: torch.device,
):
    """
    audio_in: [1, T]
    returns: [1, T]
    """
    model.eval()

    x = audio_in.to(device)
    onset_full = compute_onset_envelope(x).unsqueeze(0)  # [1, 1, T]

    T = x.shape[-1]

    remainder = T % block_size
    if remainder != 0:
        pad = block_size - remainder
        x = F.pad(x, (0, pad))
        onset_full = F.pad(onset_full, (0, pad))
    else:
        pad = 0

    T_pad = x.shape[-1]

    x_ring = torch.zeros(1, 1, context_samples + block_size, device=device)
    onset_ring = torch.zeros(1, 1, context_samples + block_size, device=device)

    h = None
    out_blocks = []
    residual_blocks = []

    for start in range(0, T_pad, block_size):
        end = start + block_size

        x_block = x[:, start:end].unsqueeze(0)           # [1,1,B]
        onset_block = onset_full[:, start:end].unsqueeze(0)

        x_ring = torch.roll(x_ring, shifts=-block_size, dims=-1)
        onset_ring = torch.roll(onset_ring, shifts=-block_size, dims=-1)

        x_ring[:, :, -block_size:] = x_block
        onset_ring[:, :, -block_size:] = onset_block

        y_block, h, r_block = model(
            x_ring,
            onset_ring,
            h0=h,
            return_state=True,
            return_residual=True,
        )

        out_blocks.append(y_block.squeeze(0).cpu())
        residual_blocks.append(r_block.squeeze(0).cpu())

    y = torch.cat(out_blocks, dim=-1)
    r = torch.cat(residual_blocks, dim=-1)

    if pad > 0:
        y = y[:, :-pad]
        r = r[:, :-pad]

    return y, r


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input guitar wav")
    parser.add_argument("--output", type=str, required=True, help="Output wav")
    parser.add_argument("--save_residual", type=str, default=None, help="Optional path to save residual-only wav")
    parser.add_argument("--save_dry", type=str, default=None, help="Optional path to save resampled dry input wav")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    model, context_samples, block_size, ckpt = load_model_from_checkpoint(args.checkpoint, device)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Epoch:            {ckpt.get('epoch', 'unknown')}")
    print(f"Block size:       {block_size}")
    print(f"Context samples:  {context_samples}")
    print(f"Residual gain:    {float(torch.clamp(model.residual_gain, 0.0, 1.0).item()):.3f}")
    print(f"Device:           {device}")
    print()

    x = load_audio_mono(args.input, SAMPLE_RATE)
    print(f"Input length:     {x.shape[-1]} samples ({x.shape[-1] / SAMPLE_RATE:.2f} sec)")

    y, residual = run_inference(
        model=model,
        audio_in=x,
        context_samples=context_samples,
        block_size=block_size,
        device=device,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_audio(args.output, y.cpu(), SAMPLE_RATE)
    print(f"Saved output:     {args.output}")

    if args.save_residual:
        os.makedirs(os.path.dirname(args.save_residual) or ".", exist_ok=True)
        save_audio(args.save_residual, residual.cpu(), SAMPLE_RATE)
        print(f"Saved residual:   {args.save_residual}")

    if args.save_dry:
        os.makedirs(os.path.dirname(args.save_dry) or ".", exist_ok=True)
        save_audio(args.save_dry, x.cpu(), SAMPLE_RATE)
        print(f"Saved dry input:  {args.save_dry}")


if __name__ == "__main__":
    main()