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

# Live inference target
BLOCK_SIZE = 128            # 2.67 ms
CONTEXT_SAMPLES = 4096      # past context (~85.3 ms)

# Training unroll: predict several live blocks in a row
UNROLL_STEPS = 8            # predicts 8 * 128 = 1024 samples at once during training
TRAIN_WINDOW = CONTEXT_SAMPLES + UNROLL_STEPS * BLOCK_SIZE

BATCH_SIZE = 4
EPOCHS = 80
LEARNING_RATE = 2e-4
WINDOWS_PER_FILE = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================================================
# Audio helpers
# =========================================================
def load_audio_mono(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    audio, sr = sf.read(path, always_2d=True)   # [T, C]
    audio = audio.astype(np.float32)
    audio = audio.mean(axis=1)                  # mono => [T]
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


# =========================================================
# Dataset
# =========================================================
class PairedRiffDataset(Dataset):
    def __init__(self, input_dir: str, target_dir: str, window_size: int, windows_per_file: int = 256):
        self.window_size = window_size
        self.windows_per_file = windows_per_file

        input_dir = Path(input_dir)
        target_dir = Path(target_dir)

        filenames = sorted([p.name for p in input_dir.glob("*.wav")])
        filenames = [f for f in filenames if (target_dir / f).exists()]

        if not filenames:
            raise RuntimeError("No matching .wav pairs found in dataset/input and dataset/target")

        self.pairs = []
        for filename in filenames:
            x = load_audio_mono(str(input_dir / filename))
            y = load_audio_mono(str(target_dir / filename))

            x, y = peak_normalize_pair(x, y)
            x, y = maybe_pad_to_window(x, y, window_size)

            self.pairs.append((filename, x, y))

        self.num_files = len(self.pairs)
        self.virtual_len = self.num_files * self.windows_per_file

    def __len__(self):
        return self.virtual_len

    def __getitem__(self, idx):
        file_idx = idx % self.num_files
        _, x, y = self.pairs[file_idx]

        min_len = min(x.shape[1], y.shape[1])
        x = x[:, :min_len]
        y = y[:, :min_len]

        if min_len == self.window_size:
            return x, y

        start = random.randint(0, min_len - self.window_size)
        end = start + self.window_size

        return x[:, start:end], y[:, start:end]


# =========================================================
# Model
# =========================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class GatedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.res_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        h_f = self.filter_conv(x)
        h_g = self.gate_conv(x)

        z = torch.tanh(h_f) * torch.sigmoid(h_g)
        z = self.norm(z)

        residual = self.res_conv(z) + x
        skip = self.skip_conv(z)
        return residual, skip


class LiveTimbreNet(nn.Module):
    """
    Runtime model:
      input  = [B, 1, CONTEXT + BLOCK]
      output = [B, 1, BLOCK]

    Training uses this same model multiple times in sequence.
    """
    def __init__(self, channels=64):
        super().__init__()

        self.in_conv = CausalConv1d(1, channels, kernel_size=5, dilation=1)

        dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        self.blocks = nn.ModuleList([GatedResidualBlock(channels, kernel_size=3, dilation=d) for d in dilations])

        self.post_1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.post_2 = nn.Conv1d(channels, 1, kernel_size=1)

        # learnable dry/wet helps avoid pure distortion collapse
        self.dry_gain = nn.Parameter(torch.tensor(0.15))
        self.wet_gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: [B, 1, CONTEXT+BLOCK]
        dry = x[:, :, -BLOCK_SIZE:]

        z = self.in_conv(x)
        z = F.leaky_relu(z, 0.2)

        skip_sum = 0.0
        for block in self.blocks:
            z, skip = block(z)
            skip_sum = skip_sum + skip

        z = F.leaky_relu(skip_sum, 0.2)
        z = F.leaky_relu(self.post_1(z), 0.2)
        wet = torch.tanh(self.post_2(z))[:, :, -BLOCK_SIZE:]

        out = self.dry_gain * dry + self.wet_gain * wet
        return out


# =========================================================
# Losses
# =========================================================
def stft_mag(x, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length, device=x.device)
    X = torch.stft(
        x.squeeze(1),   # [B, T]
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        return_complex=True,
    )
    return torch.abs(X)


def multi_resolution_stft_loss(pred, target):
    """
    pred/target are [B, 1, UNROLL_STEPS * BLOCK_SIZE]
    so we can use larger FFTs than before.
    """
    chunk_len = pred.shape[-1]

    candidate_configs = [
        (64, 16, 64),
        (128, 32, 128),
        (256, 64, 256),
        (512, 128, 512),
    ]
    configs = [cfg for cfg in candidate_configs if cfg[0] <= chunk_len]

    total = 0.0
    for n_fft, hop, win in configs:
        p = stft_mag(pred, n_fft, hop, win)
        t = stft_mag(target, n_fft, hop, win)

        sc = torch.norm(t - p, p="fro") / (torch.norm(t, p="fro") + 1e-8)
        mag = F.l1_loss(torch.log1p(p), torch.log1p(t))
        total = total + sc + mag

    return total / max(len(configs), 1)


def total_loss_fn(pred, target):
    l_wave = F.l1_loss(pred, target)
    l_stft = multi_resolution_stft_loss(pred, target)
    return l_wave + 0.7 * l_stft


# =========================================================
# Training helper: unroll multiple live blocks
# =========================================================
def forward_unrolled(model, x_window):
    """
    x_window: [B, 1, CONTEXT + UNROLL*BLOCK]
    Returns concatenated predictions: [B, 1, UNROLL*BLOCK]

    Each step uses only past/current input, so runtime model shape stays valid.
    """
    preds = []

    for step in range(UNROLL_STEPS):
        start = step * BLOCK_SIZE
        end = start + CONTEXT_SAMPLES + BLOCK_SIZE

        x_step = x_window[:, :, start:end]   # [B, 1, CONTEXT+BLOCK]
        y_step = model(x_step)               # [B, 1, BLOCK]
        preds.append(y_step)

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
    )

    effective_batch_size = min(BATCH_SIZE, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = LiveTimbreNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {DEVICE}")
    print(f"Paired files: {dataset.num_files}")
    print(f"Virtual training windows: {len(dataset)}")
    print(f"Train window: {TRAIN_WINDOW} samples ({TRAIN_WINDOW / SAMPLE_RATE * 1000:.2f} ms)")
    print(f"Context: {CONTEXT_SAMPLES} samples ({CONTEXT_SAMPLES / SAMPLE_RATE * 1000:.2f} ms)")
    print(f"Live block: {BLOCK_SIZE} samples ({BLOCK_SIZE / SAMPLE_RATE * 1000:.2f} ms)")
    print(f"Unrolled target chunk: {UNROLL_STEPS * BLOCK_SIZE} samples ({UNROLL_STEPS * BLOCK_SIZE / SAMPLE_RATE * 1000:.2f} ms)")
    print(f"Batch size: {effective_batch_size}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for x, y in loader:
            x = x.to(DEVICE)  # [B, 1, CONTEXT+UNROLL*BLOCK]
            y = y.to(DEVICE)

            pred_chunk = forward_unrolled(model, x)
            target_chunk = y[:, :, CONTEXT_SAMPLES:]   # [B, 1, UNROLL*BLOCK]

            loss = total_loss_fn(pred_chunk, target_chunk)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()

        avg = running / max(len(loader), 1)
        print(f"Epoch {epoch:03d}/{EPOCHS} | loss={avg:.6f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"live_timbre_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg,
                "sample_rate": SAMPLE_RATE,
                "block_size": BLOCK_SIZE,
                "context_samples": CONTEXT_SAMPLES,
                "unroll_steps": UNROLL_STEPS,
            },
            ckpt_path,
        )

    print("Training complete.")


if __name__ == "__main__":
    train()