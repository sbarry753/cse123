import argparse
import math
import os
from typing import Dict

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# python infer_piano_resynth.py \
#   --checkpoint checkpoints_resynth/resynth_epoch_050.pt \
#   --input "dataset_48k/input/Guitar (1).wav" \
#   --output output_piano.wav
# =========================================================
# Constants
# =========================================================
SAMPLE_RATE = 48000
N_NOTES = 88
N_PARTIALS = 8


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


def compute_onset_env(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1, T]
    returns [T]
    """
    d = torch.zeros_like(x)
    d[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
    kernel = torch.ones(1, 1, 33, dtype=x.dtype, device=x.device) / 33.0
    sm = F.conv1d(d.unsqueeze(0), kernel, padding=16).squeeze(0)  # [1, T]
    return sm.squeeze(0)


# =========================================================
# Model blocks
# =========================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class ResBlock(nn.Module):
    def __init__(self, ch, kernel_size=5, dilation=1):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, kernel_size, dilation=dilation)
        self.c2 = CausalConv1d(ch, ch, kernel_size, dilation=1)
        self.n1 = nn.GroupNorm(min(8, ch), ch)
        self.n2 = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x):
        z = F.gelu(self.n1(self.c1(x)))
        z = F.gelu(self.n2(self.c2(z)))
        return x + z


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.down = CausalConv1d(in_ch, out_ch, kernel_size=7, stride=stride)
        self.r1 = ResBlock(out_ch, dilation=1)
        self.r2 = ResBlock(out_ch, dilation=2)
        self.r3 = ResBlock(out_ch, dilation=4)

    def forward(self, x):
        x = F.gelu(self.down(x))
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        return x


class GuitarToPianoControlNet(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.in_conv = CausalConv1d(2, base_ch, kernel_size=9)
        self.e1 = EncoderStage(base_ch, base_ch * 2, stride=2)
        self.e2 = EncoderStage(base_ch * 2, base_ch * 4, stride=2)
        self.e3 = EncoderStage(base_ch * 4, base_ch * 4, stride=2)

        self.mid = nn.ModuleList([
            ResBlock(base_ch * 4, dilation=1),
            ResBlock(base_ch * 4, dilation=2),
            ResBlock(base_ch * 4, dilation=4),
            ResBlock(base_ch * 4, dilation=8),
        ])

        self.gru = nn.GRU(
            input_size=base_ch * 4,
            hidden_size=base_ch * 4,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        feat_dim = base_ch * 4
        self.note_head = nn.Linear(feat_dim, N_NOTES)
        self.vel_head = nn.Linear(feat_dim, N_NOTES)
        self.brightness_head = nn.Linear(feat_dim, N_NOTES)
        self.noise_head = nn.Linear(feat_dim, N_NOTES)

    def forward(self, x, onset, h0=None, return_state=False):
        inp = torch.cat([x, onset], dim=1)
        z = F.gelu(self.in_conv(inp))
        z = self.e1(z)
        z = self.e2(z)
        z = self.e3(z)

        for blk in self.mid:
            z = blk(z)

        z_seq = z.transpose(1, 2)
        z_seq, hN = self.gru(z_seq, h0)

        feat = z_seq[:, -1, :]
        note_logits = self.note_head(feat)
        velocity = torch.sigmoid(self.vel_head(feat))
        brightness = torch.sigmoid(self.brightness_head(feat))
        noise_amt = torch.sigmoid(self.noise_head(feat))

        if return_state:
            return note_logits, velocity, brightness, noise_amt, hN
        return note_logits, velocity, brightness, noise_amt


class DifferentiablePianoSynth(nn.Module):
    def __init__(self, sample_rate=SAMPLE_RATE, block_size=96, n_partials=N_PARTIALS):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.n_partials = n_partials

        midi = torch.arange(21, 109)
        freqs = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
        self.register_buffer("note_freqs", freqs.float())

        partial_nums = torch.arange(1, n_partials + 1).float()
        self.register_buffer("partial_nums", partial_nums)

        self.harmonic_logits = nn.Parameter(torch.randn(N_NOTES, n_partials) * 0.02)
        self.decay_logits = nn.Parameter(torch.zeros(N_NOTES))
        self.noise_decay_logits = nn.Parameter(torch.zeros(N_NOTES))

        self.master_gain = nn.Parameter(torch.tensor(0.2))
        self.noise_gain = nn.Parameter(torch.tensor(0.05))

        kernel = torch.tensor([0.05, 0.15, 0.35, 0.30, 0.15], dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("noise_kernel", kernel)

    def init_state(self, batch_size: int, device: torch.device):
        return {
            "amp": torch.zeros(batch_size, N_NOTES, device=device),
            "phase": torch.zeros(batch_size, N_NOTES, self.n_partials, device=device),
            "noise": torch.zeros(batch_size, N_NOTES, device=device),
            "prev_gate": torch.zeros(batch_size, N_NOTES, device=device),
        }

    def _per_sample_decay(self):
        d = torch.sigmoid(self.decay_logits)
        return 0.9995 + 0.00049 * d

    def _noise_decay(self):
        d = torch.sigmoid(self.noise_decay_logits)
        return 0.90 + 0.09 * d

    def forward_block(
        self,
        note_logits: torch.Tensor,
        velocity: torch.Tensor,
        brightness: torch.Tensor,
        noise_amt: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ):
        device = note_logits.device
        B = note_logits.shape[0]
        N = N_NOTES
        P = self.n_partials

        gate = torch.sigmoid(note_logits)
        onset = torch.relu(gate - state["prev_gate"])

        amp = state["amp"]
        phase = state["phase"]
        noise_state = state["noise"]

        note_decay = self._per_sample_decay().to(device)
        noise_decay = self._noise_decay().to(device)

        amp = torch.maximum(amp * note_decay.unsqueeze(0), gate * velocity)
        noise_state = noise_state * noise_decay.unsqueeze(0) + onset * noise_amt

        harm = F.softmax(self.harmonic_logits, dim=-1).to(device)
        brightness_curve = self.partial_nums.view(1, 1, P).to(device)
        bright = brightness.unsqueeze(-1)

        tilt = torch.exp(-brightness_curve * (1.5 - 1.2 * bright))
        harm = harm.unsqueeze(0) * tilt
        harm = harm / (harm.sum(dim=-1, keepdim=True) + 1e-8)

        freqs = self.note_freqs.view(1, N, 1).to(device) * self.partial_nums.view(1, 1, P).to(device)
        phase_inc = 2.0 * math.pi * freqs / self.sample_rate

        out = []
        for _ in range(self.block_size):
            phase = torch.remainder(phase + phase_inc, 2.0 * math.pi)
            s = torch.sin(phase)
            tonal = (s * (amp.unsqueeze(-1) * harm)).sum(dim=(-1, -2))

            white = torch.randn(B, N, device=device) * noise_state
            white = white.unsqueeze(1)
            filt = F.conv1d(white, self.noise_kernel, padding=2).squeeze(1)
            noisy = filt.sum(dim=-1)

            sample = self.master_gain * tonal + self.noise_gain * noisy
            out.append(sample.unsqueeze(-1))

            amp = amp * note_decay.unsqueeze(0)
            noise_state = noise_state * noise_decay.unsqueeze(0)

        audio = torch.cat(out, dim=-1).unsqueeze(1)

        new_state = {
            "amp": amp,
            "phase": phase,
            "noise": noise_state,
            "prev_gate": gate.detach(),
        }
        aux = {
            "gate": gate,
            "onset": onset,
        }
        return audio, new_state, aux


class GuitarToPianoResynthModel(nn.Module):
    def __init__(self, base_ch=64, block_size=96):
        super().__init__()
        self.analysis = GuitarToPianoControlNet(base_ch=base_ch)
        self.synth = DifferentiablePianoSynth(block_size=block_size)

    def init_stream_state(self, batch_size: int, device: torch.device):
        return {
            "rnn": None,
            "synth": self.synth.init_state(batch_size, device),
        }

    def forward_step(self, x_ctx, onset_ctx, state):
        note_logits, vel, bright, noise_amt, rnn_state = self.analysis(
            x_ctx, onset_ctx, h0=state["rnn"], return_state=True
        )
        y, synth_state, aux = self.synth.forward_block(
            note_logits, vel, bright, noise_amt, state["synth"]
        )

        new_state = {
            "rnn": rnn_state,
            "synth": synth_state,
        }
        aux.update({
            "note_logits": note_logits,
            "velocity": vel,
            "brightness": bright,
            "noise_amt": noise_amt,
        })
        return y, new_state, aux


# =========================================================
# Inference
# =========================================================
@torch.no_grad()
def run_inference(
    model: GuitarToPianoResynthModel,
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
    onset_full = compute_onset_env(x).unsqueeze(0)  # [1, T] -> [1, 1, T]

    T = x.shape[-1]

    # pad input so loop handles tail block cleanly
    remainder = T % block_size
    if remainder != 0:
        pad = block_size - remainder
        x = F.pad(x, (0, pad))
        onset_full = F.pad(onset_full, (0, pad))
    else:
        pad = 0

    T_pad = x.shape[-1]

    # rolling context buffers
    x_ring = torch.zeros(1, 1, context_samples + block_size, device=device)
    onset_ring = torch.zeros(1, 1, context_samples + block_size, device=device)

    state = model.init_stream_state(batch_size=1, device=device)

    out_blocks = []

    for start in range(0, T_pad, block_size):
        end = start + block_size

        x_block = x[:, start:end].unsqueeze(0)          # [1,1,B]
        onset_block = onset_full[:, start:end].unsqueeze(0)

        x_ring = torch.roll(x_ring, shifts=-block_size, dims=-1)
        onset_ring = torch.roll(onset_ring, shifts=-block_size, dims=-1)

        x_ring[:, :, -block_size:] = x_block
        onset_ring[:, :, -block_size:] = onset_block

        y_block, state, _ = model.forward_step(x_ring, onset_ring, state)
        out_blocks.append(y_block.squeeze(0).cpu())

    y = torch.cat(out_blocks, dim=-1)   # [1, T_pad]
    if pad > 0:
        y = y[:, :-pad]

    return y


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    block_size = ckpt.get("block_size", 96)
    base_ch = 64

    model = GuitarToPianoResynthModel(base_ch=base_ch, block_size=block_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    context_samples = ckpt.get("context_samples", 4096)
    return model, context_samples, block_size, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input guitar wav")
    parser.add_argument("--output", type=str, required=True, help="Output piano wav")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    model, context_samples, block_size, ckpt = load_model_from_checkpoint(args.checkpoint, device)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Epoch:            {ckpt.get('epoch', 'unknown')}")
    print(f"Block size:       {block_size}")
    print(f"Context samples:  {context_samples}")
    print(f"Device:           {device}")
    print()

    x = load_audio_mono(args.input, SAMPLE_RATE)
    print(f"Input length:     {x.shape[-1]} samples ({x.shape[-1] / SAMPLE_RATE:.2f} sec)")

    y = run_inference(
        model=model,
        audio_in=x,
        context_samples=context_samples,
        block_size=block_size,
        device=device,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_audio(args.output, y.cpu(), SAMPLE_RATE)

    print(f"Saved output to:  {args.output}")


if __name__ == "__main__":
    main()