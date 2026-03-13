import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Config
# =========================================================
DATASET_DIR = "dataset_48k"
INPUT_DIR = os.path.join(DATASET_DIR, "input")
TARGET_DIR = os.path.join("dataset_aligned", "target")
CHECKPOINT_DIR = "checkpoints"

SAMPLE_RATE = 48000

BLOCK_SIZE = 128
CONTEXT_SAMPLES = 4096

UNROLL_STEPS = 8
TRAIN_WINDOW = CONTEXT_SAMPLES + UNROLL_STEPS * BLOCK_SIZE

BATCH_SIZE = 8
EPOCHS = 150
LEARNING_RATE = 1e-4
WINDOWS_PER_FILE = 128

TBPTT_DETACH_EVERY = 4
ONSET_BIAS_PROB = 0.90

NUM_WORKERS = 4
PIN_MEMORY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================================================
# Cached transforms
# =========================================================
_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=64,
    power=2.0,
).to(DEVICE)


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


def peak_normalize_pair(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    peak = max(x.abs().max().item(), y.abs().max().item(), eps)
    return x / peak, y / peak


def maybe_pad_to_window(x: torch.Tensor, y: torch.Tensor, window_size: int):
    min_len = min(x.shape[1], y.shape[1])
    x = x[:, :min_len]
    y = y[:, :min_len]

    if min_len < window_size:
        pad = window_size - min_len
        x = F.pad(x, (0, pad))
        y = F.pad(y, (0, pad))

    return x, y


def compute_onset_envelope_cpu(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1, T]
    returns: [T]
    """
    d = torch.zeros_like(x)
    d[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()

    kernel = torch.ones(1, 1, 33, dtype=x.dtype) / 33.0
    d_sm = F.conv1d(d.unsqueeze(0), kernel, padding=16).squeeze(0)  # [1, T]
    return d_sm.squeeze(0)  # [T]


# =========================================================
# Dataset
# =========================================================
class PairedRiffDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        window_size: int,
        windows_per_file: int = 128,
        onset_bias_prob: float = 0.90,
    ):
        self.window_size = window_size
        self.windows_per_file = windows_per_file
        self.onset_bias_prob = onset_bias_prob

        input_dir = Path(input_dir)
        target_dir = Path(target_dir)

        filenames = sorted([p.name for p in input_dir.glob("*.wav")])
        filenames = [f for f in filenames if (target_dir / f).exists()]

        if not filenames:
            raise RuntimeError("No matching .wav pairs found.")

        self.pairs = []
        for filename in filenames:
            x = load_audio_mono(str(input_dir / filename))
            y = load_audio_mono(str(target_dir / filename))

            x, y = peak_normalize_pair(x, y)
            x, y = maybe_pad_to_window(x, y, window_size)

            onset_env = compute_onset_envelope_cpu(x)

            self.pairs.append({
                "x": x,
                "y": y,
                "onset_env": onset_env,
            })

        self.num_files = len(self.pairs)
        self.virtual_len = self.num_files * self.windows_per_file

    def __len__(self):
        return self.virtual_len

    def _sample_start_onset_biased(self, onset_env: torch.Tensor, max_start: int) -> int:
        usable = onset_env[:max_start + 1]
        weights = usable + 1e-6
        weights = weights / weights.sum()
        return int(torch.multinomial(weights, 1).item())

    def __getitem__(self, idx):
        item = self.pairs[idx % self.num_files]
        x = item["x"]
        y = item["y"]
        onset_env = item["onset_env"]

        min_len = min(x.shape[1], y.shape[1])
        x = x[:, :min_len]
        y = y[:, :min_len]
        onset_env = onset_env[:min_len]

        if min_len == self.window_size:
            return x, y, onset_env.unsqueeze(0)

        max_start = min_len - self.window_size

        if random.random() < self.onset_bias_prob:
            start = self._sample_start_onset_biased(onset_env, max_start)
        else:
            start = random.randint(0, max_start)

        end = start + self.window_size
        return x[:, start:end], y[:, start:end], onset_env[start:end].unsqueeze(0)


# =========================================================
# Causal building blocks
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
# Model
# =========================================================
class LiveCausalAutoencoder(nn.Module):
    """
    input waveform + onset envelope -> synthesized output block
    """

    def __init__(self, base_ch=64):
        super().__init__()

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

        # low initial skip gains so model leans toward synthesis
        self.skip2_gain = nn.Parameter(torch.tensor(0.20))
        self.skip1_gain = nn.Parameter(torch.tensor(0.15))
        self.skip0_gain = nn.Parameter(torch.tensor(0.10))

        self.out_mid = CausalConv1d(base_ch, base_ch, kernel_size=5)
        self.out_conv = CausalConv1d(base_ch, 1, kernel_size=5)

    def forward(self, x, onset, h0=None, return_state=False):
        # x, onset: [B, 1, T]
        inp = torch.cat([x, onset], dim=1)

        x0 = F.leaky_relu(self.in_conv(inp), 0.2)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        z, hN = self.latent(x3, h0=h0)

        y = self.dec1(z)
        y = y + self.skip2_gain * self.skip2_proj(x2[:, :, -y.shape[-1]:])

        y = self.dec2(y)
        y = y + self.skip1_gain * self.skip1_proj(x1[:, :, -y.shape[-1]:])

        y = self.dec3(y)
        y = y + self.skip0_gain * self.skip0_proj(x0[:, :, -y.shape[-1]:])

        y = F.leaky_relu(self.out_mid(y), 0.2)
        y = torch.tanh(self.out_conv(y))

        out = y[:, :, -BLOCK_SIZE:]

        if return_state:
            return out, hN
        return out


# =========================================================
# Losses
# =========================================================
def stft_mag(x, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length, device=x.device)
    X = torch.stft(
        x.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    return torch.abs(X)


def multi_resolution_stft_loss(pred, target):
    chunk_len = pred.shape[-1]

    configs = [(n, h, w) for n, h, w in [
        (64, 16, 64),
        (128, 32, 128),
        (256, 64, 256),
        (512, 128, 512),
    ] if n <= chunk_len]

    total = 0.0
    for n_fft, hop, win in configs:
        p = stft_mag(pred, n_fft, hop, win)
        t = stft_mag(target, n_fft, hop, win)

        sc = torch.norm(t - p, p="fro") / (torch.norm(t, p="fro") + 1e-8)
        mag = F.l1_loss(torch.log1p(p), torch.log1p(t))
        total = total + sc + mag

    return total / max(len(configs), 1)


def mel_loss(pred, target):
    p = _MEL(pred.squeeze(1))
    t = _MEL(target.squeeze(1))
    return F.l1_loss(torch.log1p(p), torch.log1p(t))


def envelope_loss(pred, target, frame_size=256):
    if pred.shape[-1] < frame_size:
        return torch.tensor(0.0, device=pred.device)

    step = frame_size // 2
    pred_env = pred.unfold(-1, frame_size, step).pow(2).mean(-1).sqrt()
    targ_env = target.unfold(-1, frame_size, step).pow(2).mean(-1).sqrt()
    return F.l1_loss(pred_env, targ_env)


def onset_weighted_loss(pred, target, onset_env):
    # onset_env: [B, 1, TRAIN_WINDOW]
    T_out = pred.shape[-1]
    oe = onset_env[:, :, -T_out:]

    maxv = oe.amax(dim=-1, keepdim=True).clamp(min=1e-8)
    weights = oe / maxv + 0.3
    return (weights * (pred - target).abs()).mean()


def highpass(x, coeff=0.97):
    return x[:, :, 1:] - coeff * x[:, :, :-1]


def total_loss_fn(pred, target, onset_env=None):
    l_wave = F.l1_loss(pred, target)
    l_stft = multi_resolution_stft_loss(pred, target)
    l_mel = mel_loss(pred, target)
    l_env = envelope_loss(pred, target)
    l_hp = F.l1_loss(highpass(pred), highpass(target))

    # lower waveform weight = less "copy guitar waveform"
    loss = (
        0.10 * l_wave
        + 0.90 * l_stft
        + 1.50 * l_mel
        + 0.80 * l_env
        + 0.60 * l_hp
    )

    if onset_env is not None:
        loss = loss + 1.50 * onset_weighted_loss(pred, target, onset_env)

    return loss


# =========================================================
# Training unroll
# =========================================================
def forward_unrolled(model, x_window, onset_window):
    preds = []
    h = None

    for step in range(UNROLL_STEPS):
        start = step * BLOCK_SIZE
        end = start + CONTEXT_SAMPLES + BLOCK_SIZE

        y_step, h = model(
            x_window[:, :, start:end],
            onset_window[:, :, start:end],
            h0=h,
            return_state=True,
        )
        preds.append(y_step)

        if (step + 1) % TBPTT_DETACH_EVERY == 0:
            h = h.detach()

    return torch.cat(preds, dim=-1)


# =========================================================
# Train
# =========================================================
def train():
    dataset = PairedRiffDataset(
        INPUT_DIR,
        TARGET_DIR,
        TRAIN_WINDOW,
        windows_per_file=WINDOWS_PER_FILE,
        onset_bias_prob=ONSET_BIAS_PROB,
    )

    effective_batch = min(BATCH_SIZE, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    model = LiveCausalAutoencoder(base_ch=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,
        T_mult=2,
        eta_min=1e-6,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Device:            {DEVICE}")
    print(f"Paired files:      {dataset.num_files}")
    print(f"Batches/epoch:     {len(loader)}")
    print(f"Train window:      {TRAIN_WINDOW} samples ({TRAIN_WINDOW / SAMPLE_RATE * 1000:.1f} ms)")
    print(f"Context:           {CONTEXT_SAMPLES} samples ({CONTEXT_SAMPLES / SAMPLE_RATE * 1000:.1f} ms)")
    print(f"Unrolled chunk:    {UNROLL_STEPS * BLOCK_SIZE} samples ({UNROLL_STEPS * BLOCK_SIZE / SAMPLE_RATE * 1000:.1f} ms)")
    print(f"Batch size:        {effective_batch}")
    print(f"Trainable params:  {total_params:,}")
    print()

    print("model device:", next(model.parameters()).device)
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for batch_idx, (x, y, onset_env) in enumerate(loader):
            x = x.to(DEVICE, non_blocking=True)                  # [B, 1, TRAIN_WINDOW]
            y = y.to(DEVICE, non_blocking=True)                  # [B, 1, TRAIN_WINDOW]
            onset_env = onset_env.to(DEVICE, non_blocking=True)  # [B, 1, TRAIN_WINDOW]

            pred_chunk = forward_unrolled(model, x, onset_env)
            target_chunk = y[:, :, CONTEXT_SAMPLES:]

            loss = total_loss_fn(pred_chunk, target_chunk, onset_env)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  [{epoch:03d}] batch {batch_idx + 1:>4}/{len(loader)} | loss={loss.item():.6f}")

        scheduler.step()

        avg = running / max(len(loader), 1)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{EPOCHS} | avg_loss={avg:.6f} | lr={lr:.2e}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"causal_autoenc_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg,
                "sample_rate": SAMPLE_RATE,
                "block_size": BLOCK_SIZE,
                "context_samples": CONTEXT_SAMPLES,
                "unroll_steps": UNROLL_STEPS,
                "model_type": "LiveCausalAutoencoder",
                "base_channels": 64,
            },
            ckpt_path,
        )
        print(f"  Saved {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    train()