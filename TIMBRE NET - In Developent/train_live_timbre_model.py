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
BLOCK_SIZE = 128               # 2.67 ms at 48 kHz
CONTEXT_SAMPLES = 8192         # ~170.7 ms — covers longer piano decays

# Training unroll
UNROLL_STEPS = 16              # predicts 16 * 128 = 2048 samples during training
TRAIN_WINDOW = CONTEXT_SAMPLES + UNROLL_STEPS * BLOCK_SIZE

BATCH_SIZE = 4
EPOCHS = 150
LEARNING_RATE = 1e-4
WINDOWS_PER_FILE = 256

# TBPTT: detach hidden state every N unroll steps
TBPTT_DETACH_EVERY = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================================================
# Audio helpers
# =========================================================
def load_audio_mono(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    audio, sr = sf.read(path, always_2d=True)   # [T, C]
    audio = audio.astype(np.float32)
    audio = audio.mean(axis=1)                  # mono -> [T]
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


def compute_onset_envelope(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1, T]  or  [B, 1, T]
    returns: same leading dims, last dim = T
    Cheap onset proxy using smoothed abs first-difference.
    """
    squeeze_back = False
    if x.dim() == 2:            # [1, T]
        x = x.unsqueeze(0)      # [1, 1, T]
        squeeze_back = True

    B, C, T = x.shape
    d = torch.zeros_like(x)
    d[:, :, 1:] = (x[:, :, 1:] - x[:, :, :-1]).abs()

    kernel = torch.ones(1, 1, 33, dtype=x.dtype, device=x.device) / 33.0
    # Apply per-channel (C=1 always here, but keep it general)
    d_flat = d.view(B * C, 1, T)
    d_sm = F.conv1d(d_flat, kernel, padding=16)       # [B*C, 1, T]
    d_sm = d_sm.view(B, C, T)

    if squeeze_back:
        return d_sm.squeeze(0)   # [1, T]
    return d_sm                  # [B, 1, T]  (used in forward pass)


# =========================================================
# Dataset
# =========================================================
class PairedRiffDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        window_size: int,
        windows_per_file: int = 256,
        onset_bias_prob: float = 0.7,
    ):
        self.window_size = window_size
        self.windows_per_file = windows_per_file
        self.onset_bias_prob = onset_bias_prob

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

            onset_env = compute_onset_envelope(x).squeeze(0)  # [1, T] -> [T]

            self.pairs.append({
                "filename": filename,
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
        start = torch.multinomial(weights, 1).item()
        return start

    def __getitem__(self, idx):
        file_idx = idx % self.num_files
        item = self.pairs[file_idx]

        x = item["x"]
        y = item["y"]
        onset_env = item["onset_env"]

        min_len = min(x.shape[1], y.shape[1])
        x = x[:, :min_len]
        y = y[:, :min_len]
        onset_env = onset_env[:min_len]

        if min_len == self.window_size:
            return x, y, onset_env

        max_start = min_len - self.window_size

        if random.random() < self.onset_bias_prob:
            start = self._sample_start_onset_biased(onset_env, max_start)
        else:
            start = random.randint(0, max_start)

        end = start + self.window_size
        return x[:, start:end], y[:, start:end], onset_env[start:end]


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
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


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
    """Causal downsampling stage using stride in Conv1d."""
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
    """
    Causal upsampling stage:
      - nearest-neighbor repeat in time
      - causal smoothing conv
      - residual refinement
    """
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
    """
    Causal latent processor.
    Combines dilated causal residual conv blocks plus a GRU for memory.
    """
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
            num_layers=2,           # extra layer for more temporal capacity
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
        # x: [B, C, T]
        for blk in self.pre:
            x = blk(x)

        x_seq = x.transpose(1, 2)          # [B, T, C]
        x_seq, hN = self.gru(x_seq, h0)    # [B, T, C]
        x = x_seq.transpose(1, 2)          # [B, C, T]

        for blk in self.post:
            x = blk(x)

        return x, hN


# =========================================================
# Model: Causal Encoder/Decoder
# =========================================================
class LiveCausalAutoencoder(nn.Module):
    """
    Runtime contract:
      input  = [B, 1, CONTEXT + BLOCK]   (raw waveform)
      output = [B, 1, BLOCK]

    Input is 2-channel: [waveform, onset_envelope]
    3-stage encoder/decoder for richer timbre capacity.

    Enc/Dec stride layout:
      enc1: /2   enc2: /4   enc3: /8
      dec1: x2   dec2: x4   dec3: x8
    """

    def __init__(self, base_ch=64):
        super().__init__()

        # 2-channel input stem: waveform + onset envelope
        self.in_conv = CausalConv1d(2, base_ch, kernel_size=9)

        # Encoder
        self.enc1 = CausalEncoderStage(base_ch,     base_ch * 2, kernel_size=7, stride=2)   # /2
        self.enc2 = CausalEncoderStage(base_ch * 2, base_ch * 4, kernel_size=7, stride=2)   # /4
        self.enc3 = CausalEncoderStage(base_ch * 4, base_ch * 4, kernel_size=7, stride=2)   # /8

        # Latent
        self.latent = LatentProcessor(base_ch * 4)

        # Decoder
        self.dec1 = CausalUpsampleStage(base_ch * 4, base_ch * 4, up_factor=2)  # x2
        self.dec2 = CausalUpsampleStage(base_ch * 4, base_ch * 2, up_factor=2)  # x4
        self.dec3 = CausalUpsampleStage(base_ch * 2, base_ch,     up_factor=2)  # x8

        # Skip projections
        self.skip2_proj = nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=1)
        self.skip1_proj = nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=1)
        self.skip0_proj = nn.Conv1d(base_ch,     base_ch,     kernel_size=1)

        # Output head
        self.out_mid = CausalConv1d(base_ch, base_ch, kernel_size=5)
        self.out_conv = CausalConv1d(base_ch, 1, kernel_size=5)

    def forward(self, x, onset=None, h0=None, return_state=False):
        """
        x:     [B, 1, T]
        onset: [B, 1, T] or None  (if None, computed internally)
        """
        if onset is None:
            onset = compute_onset_envelope(x)   # [B, 1, T]

        x_in = torch.cat([x, onset], dim=1)         # [B, 2, T]
        x0 = F.leaky_relu(self.in_conv(x_in), 0.2)  # [B, base, T]

        x1 = self.enc1(x0)   # [B, 2*base, T/2]
        x2 = self.enc2(x1)   # [B, 4*base, T/4]
        x3 = self.enc3(x2)   # [B, 4*base, T/8]

        z, hN = self.latent(x3, h0=h0)

        # --- Decoder with fixed skip connections ---
        y = self.dec1(z)
        y = y + self.skip2_proj(x2[:, :, -y.shape[-1]:])

        y = self.dec2(y)
        y = y + self.skip1_proj(x1[:, :, -y.shape[-1]:])

        y = self.dec3(y)
        y = y + self.skip0_proj(x0[:, :, -y.shape[-1]:])

        y = F.leaky_relu(self.out_mid(y), 0.2)
        y = torch.tanh(self.out_conv(y))

        out = y[:, :, -BLOCK_SIZE:]   # only the newest live block

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

    candidate_configs = [
        (64,  16,  64),
        (128, 32,  128),
        (256, 64,  256),
        (512, 128, 512),
    ]
    configs = [cfg for cfg in candidate_configs if cfg[0] <= chunk_len]

    total = 0.0
    for n_fft, hop, win in configs:
        p = stft_mag(pred,   n_fft, hop, win)
        t = stft_mag(target, n_fft, hop, win)

        sc  = torch.norm(t - p, p="fro") / (torch.norm(t, p="fro") + 1e-8)
        mag = F.l1_loss(torch.log1p(p), torch.log1p(t))
        total = total + sc + mag

    return total / max(len(configs), 1)


def mel_loss(pred, target, sample_rate=SAMPLE_RATE):
    """Perceptually-weighted mel spectrogram loss."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
    ).to(pred.device)

    p = mel_transform(pred.squeeze(1))
    t = mel_transform(target.squeeze(1))
    return F.l1_loss(torch.log1p(p), torch.log1p(t))


def envelope_loss(pred, target, frame_size=256):
    """
    Penalize mismatch in amplitude envelope shape (ADSR).
    Uses RMS over short overlapping frames.
    """
    # Need at least frame_size samples
    if pred.shape[-1] < frame_size:
        return torch.tensor(0.0, device=pred.device)

    step = frame_size // 2
    pred_env = pred.unfold(-1, frame_size, step).pow(2).mean(-1).sqrt()   # [B, 1, frames]
    targ_env = target.unfold(-1, frame_size, step).pow(2).mean(-1).sqrt()
    return F.l1_loss(pred_env, targ_env)


def onset_weighted_loss(pred, target, onset_env):
    """
    Weight waveform L1 by onset strength so the model focuses on
    getting piano attack transients right.

    onset_env: [B, TRAIN_WINDOW]  (full window, from dataset)
    pred/target: [B, 1, UNROLL*BLOCK]
    """
    T_out = pred.shape[-1]
    # Take the tail of onset_env that corresponds to the unrolled output
    oe = onset_env[:, -T_out:].unsqueeze(1)   # [B, 1, T_out]

    # Normalize to [0.3, 1.3] so even non-onset regions still contribute
    oe_max = oe.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    weights = oe / oe_max + 0.3

    return (weights * (pred - target).abs()).mean()


def highpass(x, coeff=0.97):
    return x[:, :, 1:] - coeff * x[:, :, :-1]


def total_loss_fn(pred, target, onset_env=None):
    l_wave = F.l1_loss(pred, target)
    l_stft = multi_resolution_stft_loss(pred, target)
    l_mel  = mel_loss(pred, target)
    l_env  = envelope_loss(pred, target)
    l_hp   = F.l1_loss(highpass(pred), highpass(target))

    loss = (
        0.5 * l_wave
        + 0.7 * l_stft
        + 1.0 * l_mel
        + 0.8 * l_env
        + 0.3 * l_hp
    )

    if onset_env is not None:
        loss = loss + 1.5 * onset_weighted_loss(pred, target, onset_env)

    return loss


# =========================================================
# Training helper — TBPTT unroll
# =========================================================
def forward_unrolled(model, x_window, onset_window):
    """
    x_window:     [B, 1, CONTEXT + UNROLL*BLOCK]
    onset_window: [B, 1, CONTEXT + UNROLL*BLOCK]

    Returns concatenated predictions: [B, 1, UNROLL*BLOCK]

    TBPTT: hidden state is detached every TBPTT_DETACH_EVERY steps so
    gradients don't explode over 16 unroll steps, but still flow
    long enough for the GRU to learn piano sustain/decay.
    """
    preds = []
    h = None

    for step in range(UNROLL_STEPS):
        start = step * BLOCK_SIZE
        end   = start + CONTEXT_SAMPLES + BLOCK_SIZE

        x_step     = x_window[:, :, start:end]      # [B, 1, CONTEXT+BLOCK]
        onset_step = onset_window[:, :, start:end]   # [B, 1, CONTEXT+BLOCK]

        y_step, h = model(x_step, onset=onset_step, h0=h, return_state=True)
        preds.append(y_step)

        if (step + 1) % TBPTT_DETACH_EVERY == 0:
            h = h.detach()

    return torch.cat(preds, dim=-1)   # [B, 1, UNROLL*BLOCK]


# =========================================================
# Train
# =========================================================
def train():
    dataset = PairedRiffDataset(
        INPUT_DIR,
        TARGET_DIR,
        TRAIN_WINDOW,
        windows_per_file=WINDOWS_PER_FILE,
        onset_bias_prob=0.7,
    )

    effective_batch_size = min(BATCH_SIZE, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = LiveCausalAutoencoder(base_ch=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Cosine LR schedule with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Device:               {DEVICE}")
    print(f"Paired files:         {dataset.num_files}")
    print(f"Virtual windows:      {len(dataset)}")
    print(f"Train window:         {TRAIN_WINDOW} samples  ({TRAIN_WINDOW / SAMPLE_RATE * 1000:.1f} ms)")
    print(f"Context:              {CONTEXT_SAMPLES} samples  ({CONTEXT_SAMPLES / SAMPLE_RATE * 1000:.1f} ms)")
    print(f"Live block:           {BLOCK_SIZE} samples  ({BLOCK_SIZE / SAMPLE_RATE * 1000:.2f} ms)")
    print(f"Unrolled chunk:       {UNROLL_STEPS * BLOCK_SIZE} samples  ({UNROLL_STEPS * BLOCK_SIZE / SAMPLE_RATE * 1000:.1f} ms)")
    print(f"TBPTT detach every:   {TBPTT_DETACH_EVERY} steps")
    print(f"Batch size:           {effective_batch_size}")
    print(f"Trainable params:     {total_params:,}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for x, y, onset_env in loader:
            x         = x.to(DEVICE)            # [B, 1, TRAIN_WINDOW]
            y         = y.to(DEVICE)
            onset_env = onset_env.to(DEVICE)    # [B, TRAIN_WINDOW]

            # Pre-compute onset for the whole window [B, 1, TRAIN_WINDOW]
            onset_window = compute_onset_envelope(x)

            pred_chunk   = forward_unrolled(model, x, onset_window)
            target_chunk = y[:, :, CONTEXT_SAMPLES:]

            loss = total_loss_fn(pred_chunk, target_chunk, onset_env)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()

        scheduler.step()

        avg = running / max(len(loader), 1)
        lr  = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{EPOCHS} | loss={avg:.6f} | lr={lr:.2e}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"causal_autoenc_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss":                 avg,
                "sample_rate":          SAMPLE_RATE,
                "block_size":           BLOCK_SIZE,
                "context_samples":      CONTEXT_SAMPLES,
                "unroll_steps":         UNROLL_STEPS,
                "model_type":           "LiveCausalAutoencoder",
                "base_channels":        64,
            },
            ckpt_path,
        )

    print("Training complete.")


if __name__ == "__main__":
    train()