"""
test_synth.py - direct DDSP synth capacity test.

This script bypasses the guitar encoder and decoder entirely. It optimizes
free synth parameters directly against a target piano frame or short sequence.
If this cannot fit the target well, the bottleneck is the synth/loss space
rather than the neural encoder.

Examples:
  python3 test_synth.py --target data_small/piano/example.wav --f0 220

  python3 test_synth.py \
      --target data_small/piano/example.wav \
      --f0_cache f0_cache_small/example.npy \
      --mode sequence --sequence_frames 16 --start_frame 20
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from losses import CombinedLoss
from model import (
    AdditiveSynth,
    FRAME_SIZE,
    N_BODY_FILTER_BANDS,
    N_ENVELOPE_POINTS,
    N_HARMONICS,
    N_NOISE_BANDS,
    SAMPLE_RATE,
)


def parse_args():
    p = argparse.ArgumentParser(description="Optimize free synth params directly against piano audio.")
    p.add_argument("--target", required=True, help="Target piano WAV")
    p.add_argument("--output_dir", default="synth_capacity_test")
    p.add_argument("--mode", choices=["frame", "sequence"], default="frame")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--sequence_frames", type=int, default=16)
    p.add_argument("--hop_size", type=int, default=FRAME_SIZE)
    p.add_argument("--f0", type=float, default=220.0, help="Fixed f0 in Hz when --f0_cache is not provided")
    p.add_argument("--f0_cache", default=None, help="Optional .npy f0 labels aligned by hop")
    p.add_argument("--optimize_f0", action="store_true", help="Also optimize a bounded f0 multiplier")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--device", default="auto")
    p.add_argument("--n_harmonics", type=int, default=N_HARMONICS)
    p.add_argument("--n_noise", type=int, default=N_NOISE_BANDS)
    p.add_argument("--n_envelope", type=int, default=N_ENVELOPE_POINTS)
    p.add_argument("--n_body_filter", type=int, default=N_BODY_FILTER_BANDS)
    p.add_argument("--high_freq_excess_weight", type=float, default=0.0)
    p.add_argument("--high_freq_hz", type=float, default=8000.0)
    p.add_argument("--disable_noise", action="store_true", help="Force noise_gain to zero during optimization")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--print_every", type=int, default=100)
    return p.parse_args()


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def load_audio(path: str) -> torch.Tensor:
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = audio.squeeze(0).float()
    peak = audio.abs().max()
    if peak > 1e-8:
        audio = audio / peak
    return audio


def load_f0_labels(args, n_frames: int, device: torch.device) -> torch.Tensor:
    if args.f0_cache:
        f0 = np.asarray(np.load(args.f0_cache), dtype=np.float32)
        if len(f0) < args.start_frame + n_frames:
            f0 = np.pad(f0, (0, args.start_frame + n_frames - len(f0)), mode="constant")
        f0 = f0[args.start_frame:args.start_frame + n_frames]
        f0 = np.where(f0 > 0.0, f0, args.f0).astype(np.float32)
    else:
        f0 = np.full(n_frames, args.f0, dtype=np.float32)
    return torch.from_numpy(f0).to(device)


def target_slice(audio: torch.Tensor, args) -> torch.Tensor:
    n_frames = 1 if args.mode == "frame" else args.sequence_frames
    start = args.start_frame * args.hop_size
    end = start + n_frames * args.hop_size
    if end > audio.numel():
        audio = F.pad(audio, (0, end - audio.numel()))
    return audio[start:end].view(1, -1)


class FreeSynthParams(nn.Module):
    def __init__(
        self,
        n_frames: int,
        n_harmonics: int,
        n_noise: int,
        n_envelope: int,
        n_body_filter: int,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise = n_noise
        self.n_envelope = n_envelope
        self.n_body_filter = n_body_filter
        self.harm_logits = nn.Parameter(torch.full((n_frames, n_harmonics), -4.0))
        self.global_logit = nn.Parameter(torch.full((n_frames,), -2.0))
        self.noise_mag_logits = nn.Parameter(torch.full((n_frames, n_noise), -5.0))
        self.noise_gain_logit = nn.Parameter(torch.full((n_frames,), -6.0))
        self.body_filter_logits = nn.Parameter(torch.zeros(n_frames, n_body_filter))
        self.envelope_logits = nn.Parameter(torch.zeros(n_frames, n_envelope))
        self.f0_delta = nn.Parameter(torch.zeros(n_frames))

    def forward(self, base_f0: torch.Tensor, optimize_f0: bool, disable_noise: bool):
        global_amp = torch.sigmoid(self.global_logit)
        harm_amps = torch.sigmoid(self.harm_logits) * global_amp.unsqueeze(-1) / np.sqrt(self.n_harmonics)
        noise_mags = torch.sigmoid(self.noise_mag_logits) * 0.1
        noise_gain = torch.sigmoid(self.noise_gain_logit) * 0.2
        if disable_noise:
            noise_gain = torch.zeros_like(noise_gain)
        body_filter_db = torch.tanh(self.body_filter_logits) * 24.0
        body_filter = torch.pow(10.0, body_filter_db / 20.0)
        envelope = torch.sigmoid(self.envelope_logits) * 2.0
        if optimize_f0:
            f0 = base_f0 * torch.exp(torch.tanh(self.f0_delta) * np.log(2.0))
        else:
            f0 = base_f0
        return {
            "harm_amps": harm_amps,
            "global_amp": global_amp,
            "noise_mags": noise_mags,
            "noise_gain": noise_gain,
            "body_filter": body_filter,
            "body_filter_db": body_filter_db,
            "envelope": envelope,
            "f0_corrected": f0,
        }


def overlap_add(rendered: torch.Tensor, hop_size: int) -> torch.Tensor:
    """rendered: (T, 2*hop), returns (1, T*hop)."""
    n_frames, window_size = rendered.shape
    total_size = (n_frames - 1) * hop_size + window_size
    out = rendered.new_zeros(total_size)
    win_sum = rendered.new_zeros(total_size)
    window = torch.hann_window(window_size, periodic=False, device=rendered.device, dtype=rendered.dtype)
    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size
        out[start:end] += rendered[i] * window
        win_sum[start:end] += window
    out = out / win_sum.clamp_min(1e-8)
    return out[:n_frames * hop_size].unsqueeze(0)


def render_direct(synth, params, args):
    n_frames = params["f0_corrected"].shape[0]
    if args.mode == "frame":
        pred = synth(
            f0=params["f0_corrected"],
            harm_amps=params["harm_amps"],
            noise_mags=params["noise_mags"],
            noise_gain=params["noise_gain"],
            body_filter=params["body_filter"],
            envelope=params["envelope"],
            render_size=args.hop_size,
            phase_advance_size=args.hop_size,
        )
        return pred

    rendered = []
    render_size = 2 * args.hop_size
    for i in range(n_frames):
        frame = synth(
            f0=params["f0_corrected"][i:i + 1],
            harm_amps=params["harm_amps"][i:i + 1],
            noise_mags=params["noise_mags"][i:i + 1],
            noise_gain=params["noise_gain"][i:i + 1],
            body_filter=params["body_filter"][i:i + 1],
            envelope=params["envelope"][i:i + 1],
            render_size=render_size,
            phase_advance_size=args.hop_size,
        )
        rendered.append(frame.squeeze(0))
    return overlap_add(torch.stack(rendered, dim=0), args.hop_size)


def save_outputs(output_dir: Path, pred: torch.Tensor, target: torch.Tensor, losses: list[float]):
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_np = pred.detach().cpu().squeeze(0).clamp(-1.0, 1.0)
    target_np = target.detach().cpu().squeeze(0).clamp(-1.0, 1.0)
    print(pred_np.size())
    torchaudio.save(output_dir / "synth_fit.wav", pred_np.unsqueeze(0), SAMPLE_RATE)
    torchaudio.save(output_dir / "target.wav", target_np.unsqueeze(0), SAMPLE_RATE)

    np.savetxt(output_dir / "losses.csv", np.asarray(losses), delimiter=",")
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Direct Synth Capacity Test")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    print(f"Device: {device}")

    n_frames = 1 if args.mode == "frame" else args.sequence_frames
    target = target_slice(load_audio(args.target), args).to(device)
    f0 = load_f0_labels(args, n_frames, device)

    synth = AdditiveSynth(
        frame_size=args.hop_size,
        n_harmonics=args.n_harmonics,
        n_noise=args.n_noise,
        n_envelope=args.n_envelope,
        n_body_filter=args.n_body_filter,
    ).to(device)
    free_params = FreeSynthParams(
        n_frames,
        args.n_harmonics,
        args.n_noise,
        args.n_envelope,
        args.n_body_filter,
    ).to(device)
    criterion = CombinedLoss(
        high_freq_excess_weight=args.high_freq_excess_weight,
        high_freq_hz=args.high_freq_hz,
        sample_rate=SAMPLE_RATE,
    ).to(device)
    optimizer = torch.optim.AdamW(free_params.parameters(), lr=args.lr, weight_decay=0.0)

    losses = []
    best_loss = float("inf")
    best_pred = None

    for step in tqdm(range(args.steps), ncols=72):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(args.seed)
        synth.phase = torch.zeros_like(synth.phase)
        params = free_params(f0, args.optimize_f0, args.disable_noise)
        pred = render_direct(synth, params, args)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_pred = pred.detach().clone()
        if args.print_every > 0 and (step == 0 or (step + 1) % args.print_every == 0):
            with torch.no_grad():
                print(
                    f"step={step + 1:5d} loss={loss_value:.5f} "
                    f"global_amp={params['global_amp'].mean().item():.4f} "
                    f"noise_gain={params['noise_gain'].mean().item():.5f}"
                )

    if best_pred is None:
        best_pred = pred.detach()
    save_outputs(output_dir, best_pred, target, losses)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Saved: {output_dir / 'synth_fit.wav'}")
    print(f"Saved: {output_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
