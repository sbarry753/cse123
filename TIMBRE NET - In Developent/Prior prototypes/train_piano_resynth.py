import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
CHECKPOINT_DIR = "checkpoints_resynth"

SAMPLE_RATE = 48000
BLOCK_SIZE = 96                 # 2 ms at 48 kHz
CONTEXT_SAMPLES = 4096          # past-only context available to encoder
UNROLL_STEPS = 16               # 16 * 96 = 1536 samples = 32 ms predicted per train example
TRAIN_WINDOW = CONTEXT_SAMPLES + UNROLL_STEPS * BLOCK_SIZE

BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 2e-4
NUM_WORKERS = 2
PIN_MEMORY = True
WINDOWS_PER_FILE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_NOTES = 88                    # piano notes A0..C8
N_PARTIALS = 8                  # additive harmonics per note
ONSET_BIAS_PROB = 0.8

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
    T = min(x.shape[-1], y.shape[-1])
    x = x[:, :T]
    y = y[:, :T]
    if T < window_size:
        pad = window_size - T
        x = F.pad(x, (0, pad))
        y = F.pad(y, (0, pad))
    return x, y


def compute_onset_env(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1, T]
    returns [T]
    """
    d = torch.zeros_like(x)
    d[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
    kernel = torch.ones(1, 1, 33, dtype=x.dtype) / 33.0
    sm = F.conv1d(d.unsqueeze(0), kernel, padding=16).squeeze(0)  # [1, T]
    return sm.squeeze(0)


# =========================================================
# Dataset
# =========================================================
class PairedWindowDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        window_size: int,
        windows_per_file: int = 64,
        onset_bias_prob: float = 0.8,
    ):
        self.window_size = window_size
        self.windows_per_file = windows_per_file
        self.onset_bias_prob = onset_bias_prob

        input_dir = Path(input_dir)
        target_dir = Path(target_dir)

        filenames = sorted([p.name for p in input_dir.glob("*.wav")])
        filenames = [f for f in filenames if (target_dir / f).exists()]
        if not filenames:
            raise RuntimeError("No matching wav pairs found.")

        self.items = []
        for name in filenames:
            x = load_audio_mono(str(input_dir / name))
            y = load_audio_mono(str(target_dir / name))
            x, y = peak_normalize_pair(x, y)
            x, y = maybe_pad_to_window(x, y, window_size)
            onset = compute_onset_env(x)
            self.items.append({"x": x, "y": y, "onset": onset})

        self.num_files = len(self.items)
        self.virtual_len = self.num_files * self.windows_per_file

    def __len__(self):
        return self.virtual_len

    def _sample_start(self, onset_env: torch.Tensor, max_start: int):
        usable = onset_env[:max_start + 1]
        weights = usable + 1e-6
        weights = weights / weights.sum()
        return int(torch.multinomial(weights, 1).item())

    def __getitem__(self, idx):
        item = self.items[idx % self.num_files]
        x = item["x"]
        y = item["y"]
        onset = item["onset"]

        T = min(x.shape[-1], y.shape[-1])
        x = x[:, :T]
        y = y[:, :T]
        onset = onset[:T]

        if T == self.window_size:
            return x, y, onset.unsqueeze(0)

        max_start = T - self.window_size
        if random.random() < self.onset_bias_prob:
            start = self._sample_start(onset, max_start)
        else:
            start = random.randint(0, max_start)

        end = start + self.window_size
        return x[:, start:end], y[:, start:end], onset[start:end].unsqueeze(0)


# =========================================================
# Causal encoder blocks
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


# =========================================================
# Guitar analysis network
# =========================================================
class GuitarToPianoControlNet(nn.Module):
    """
    Input: waveform + onset envelope over context+block window
    Output for latest block:
      - note_on logits [B, 88]
      - velocity [B, 88]
      - brightness [B, 88]
      - noise_amt [B, 88]
    """
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
        """
        x, onset: [B, 1, T]
        outputs correspond to the latest block only
        """
        inp = torch.cat([x, onset], dim=1)
        z = F.gelu(self.in_conv(inp))
        z = self.e1(z)
        z = self.e2(z)
        z = self.e3(z)

        for blk in self.mid:
            z = blk(z)

        z_seq = z.transpose(1, 2)       # [B, T', C]
        z_seq, hN = self.gru(z_seq, h0)

        feat = z_seq[:, -1, :]          # latest frame only
        note_logits = self.note_head(feat)
        velocity = torch.sigmoid(self.vel_head(feat))
        brightness = torch.sigmoid(self.brightness_head(feat))
        noise_amt = torch.sigmoid(self.noise_head(feat))

        if return_state:
            return note_logits, velocity, brightness, noise_amt, hN
        return note_logits, velocity, brightness, noise_amt


# =========================================================
# Differentiable piano-ish synth
# =========================================================
class DifferentiablePianoSynth(nn.Module):
    """
    Structured polyphonic synth:
      - per-note amplitude state
      - additive harmonics
      - trainable harmonic templates + decay
      - noise burst on note onsets

    Streaming state:
      amp_state   [B, 88]
      phase_state [B, 88, P]
      noise_state [B, 88]
      prev_gate   [B, 88]
    """
    def __init__(self, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, n_partials=N_PARTIALS):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.n_partials = n_partials

        midi = torch.arange(21, 109)  # A0..C8
        freqs = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
        self.register_buffer("note_freqs", freqs.float())  # [88]

        partial_nums = torch.arange(1, n_partials + 1).float()
        self.register_buffer("partial_nums", partial_nums)

        # Trainable harmonic profile template over note + partial
        self.harmonic_logits = nn.Parameter(torch.randn(N_NOTES, n_partials) * 0.02)

        # Per-note decay parameters
        self.decay_logits = nn.Parameter(torch.zeros(N_NOTES))
        self.noise_decay_logits = nn.Parameter(torch.zeros(N_NOTES))

        # Global trainable gains
        self.master_gain = nn.Parameter(torch.tensor(0.2))
        self.noise_gain = nn.Parameter(torch.tensor(0.05))

        # Simple fixed lowpass kernel for transient noise
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
        # Slow note decay, mapped to stable per-sample multiplier
        # about 0.9995 .. 0.99999
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
        """
        Inputs: [B, 88] for current block
        Returns:
          audio_block [B, 1, BLOCK_SIZE]
          updated_state
          aux dict with gate/onset for regularization
        """
        device = note_logits.device
        B = note_logits.shape[0]
        N = N_NOTES
        P = self.n_partials

        gate = torch.sigmoid(note_logits)  # [B, 88]
        onset = torch.relu(gate - state["prev_gate"])  # rising edge proxy

        amp = state["amp"]
        phase = state["phase"]
        noise_state = state["noise"]

        note_decay = self._per_sample_decay().to(device)       # [88]
        noise_decay = self._noise_decay().to(device)           # [88]

        # Excite amplitude state on note on
        amp = torch.maximum(amp * note_decay.unsqueeze(0), gate * velocity)
        noise_state = noise_state * noise_decay.unsqueeze(0) + onset * noise_amt

        # Harmonic distribution
        harm = F.softmax(self.harmonic_logits, dim=-1).to(device)          # [88, P]
        brightness_curve = self.partial_nums.view(1, 1, P).to(device)      # [1,1,P]
        bright = brightness.unsqueeze(-1)                                  # [B,88,1]

        # More brightness => slower rolloff in harmonics
        tilt = torch.exp(-brightness_curve * (1.5 - 1.2 * bright))
        harm = harm.unsqueeze(0) * tilt                                    # [B,88,P]
        harm = harm / (harm.sum(dim=-1, keepdim=True) + 1e-8)

        # Per-partial frequencies
        freqs = self.note_freqs.view(1, N, 1).to(device) * self.partial_nums.view(1, 1, P).to(device)
        phase_inc = 2.0 * math.pi * freqs / self.sample_rate               # [1,88,P]

        # Render block sample-by-sample to preserve streaming phase continuity
        out = []
        for _ in range(self.block_size):
            phase = torch.remainder(phase + phase_inc, 2.0 * math.pi)
            s = torch.sin(phase)                                           # [B,88,P]
            tonal = (s * (amp.unsqueeze(-1) * harm)).sum(dim=(-1, -2))     # [B]

            # note-shaped transient noise
            white = torch.randn(B, N, device=device) * noise_state
            white = white.unsqueeze(1)                                     # [B,1,88]
            filt = F.conv1d(white, self.noise_kernel, padding=2).squeeze(1)  # [B,88]
            noisy = filt.sum(dim=-1)                                       # [B]

            sample = self.master_gain * tonal + self.noise_gain * noisy
            out.append(sample.unsqueeze(-1))

            # within-block decay
            amp = amp * note_decay.unsqueeze(0)
            noise_state = noise_state * noise_decay.unsqueeze(0)

        audio = torch.cat(out, dim=-1).unsqueeze(1)                        # [B,1,T]

        new_state = {
            "amp": amp,
            "phase": phase,
            "noise": noise_state,
            "prev_gate": gate.detach(),  # edge detector memory only
        }

        aux = {
            "gate": gate,
            "onset": onset,
        }
        return audio, new_state, aux


# =========================================================
# Full streaming model
# =========================================================
class GuitarToPianoResynthModel(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.analysis = GuitarToPianoControlNet(base_ch=base_ch)
        self.synth = DifferentiablePianoSynth()

    def init_stream_state(self, batch_size: int, device: torch.device):
        return {
            "rnn": None,
            "synth": self.synth.init_state(batch_size, device),
        }

    def forward_step(self, x_ctx, onset_ctx, state):
        """
        x_ctx/onset_ctx: [B,1,CONTEXT_SAMPLES + BLOCK_SIZE]
        returns latest output block only
        """
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
# Losses
# =========================================================
_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    power=2.0,
).to(DEVICE)


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


def mrstft_loss(pred, target):
    configs = [(64, 16, 64), (128, 32, 128), (256, 64, 256), (512, 128, 512)]
    total = 0.0
    count = 0
    for n_fft, hop, win in configs:
        if pred.shape[-1] < n_fft:
            continue
        p = stft_mag(pred, n_fft, hop, win)
        t = stft_mag(target, n_fft, hop, win)
        sc = torch.norm(t - p, p="fro") / (torch.norm(t, p="fro") + 1e-8)
        mag = F.l1_loss(torch.log1p(p), torch.log1p(t))
        total = total + sc + mag
        count += 1
    return total / max(count, 1)


def mel_loss(pred, target):
    p = _MEL(pred.squeeze(1))
    t = _MEL(target.squeeze(1))
    return F.l1_loss(torch.log1p(p), torch.log1p(t))


def envelope_loss(pred, target, frame_size=192):
    if pred.shape[-1] < frame_size:
        return pred.new_tensor(0.0)
    step = frame_size // 2
    pe = pred.unfold(-1, frame_size, step).pow(2).mean(-1).sqrt()
    te = target.unfold(-1, frame_size, step).pow(2).mean(-1).sqrt()
    return F.l1_loss(pe, te)


def highpass(x, coeff=0.97):
    return x[:, :, 1:] - coeff * x[:, :, :-1]


def control_sparsity_loss(aux_list: List[Dict[str, torch.Tensor]]):
    """
    Mild regularization:
      - encourage note activity sparsity
      - discourage excessive noise
    """
    gate = torch.stack([a["gate"] for a in aux_list], dim=1)          # [B,S,88]
    noise = torch.stack([a["noise_amt"] for a in aux_list], dim=1)    # [B,S,88]
    vel = torch.stack([a["velocity"] for a in aux_list], dim=1)

    l_gate = gate.mean()
    l_noise = noise.mean()
    l_vel = vel.mean()

    return 0.01 * l_gate + 0.005 * l_noise + 0.002 * l_vel


def total_loss_fn(pred, target, aux_list):
    l_stft = mrstft_loss(pred, target)
    l_mel = mel_loss(pred, target)
    l_env = envelope_loss(pred, target)
    l_hp = F.l1_loss(highpass(pred), highpass(target))
    l_wave = F.l1_loss(pred, target)
    l_ctrl = control_sparsity_loss(aux_list)

    loss = (
        1.2 * l_stft +
        1.5 * l_mel +
        0.8 * l_env +
        0.5 * l_hp +
        0.05 * l_wave +
        l_ctrl
    )
    stats = {
        "stft": float(l_stft.item()),
        "mel": float(l_mel.item()),
        "env": float(l_env.item()),
        "hp": float(l_hp.item()),
        "wave": float(l_wave.item()),
        "ctrl": float(l_ctrl.item()),
    }
    return loss, stats


# =========================================================
# Unrolled training forward
# =========================================================
def forward_unrolled(model, x_window, onset_window):
    """
    x_window/onset_window: [B,1,TRAIN_WINDOW]
    Predicts UNROLL_STEPS blocks sequentially using streaming state.
    """
    B = x_window.shape[0]
    state = model.init_stream_state(B, x_window.device)

    preds = []
    aux_list = []

    for step in range(UNROLL_STEPS):
        start = step * BLOCK_SIZE
        end = start + CONTEXT_SAMPLES + BLOCK_SIZE

        x_ctx = x_window[:, :, start:end]
        onset_ctx = onset_window[:, :, start:end]

        y_step, state, aux = model.forward_step(x_ctx, onset_ctx, state)
        preds.append(y_step)
        aux_list.append(aux)

        # detach recurrent state periodically for stability
        if (step + 1) % 4 == 0:
            if state["rnn"] is not None:
                state["rnn"] = state["rnn"].detach()
            state["synth"] = {k: v.detach() for k, v in state["synth"].items()}

    return torch.cat(preds, dim=-1), aux_list


# =========================================================
# Validation rendering
# =========================================================
@torch.no_grad()
def render_example(model, dataset, out_dir, epoch, device):
    model.eval()
    x, y, onset = dataset[0]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    onset = onset.unsqueeze(0).to(device)

    pred, _ = forward_unrolled(model, x, onset)
    target = y[:, :, CONTEXT_SAMPLES:]

    pred = pred.squeeze(0).cpu().clamp(-1, 1)
    target = target.squeeze(0).cpu().clamp(-1, 1)
    inp = x[:, :, CONTEXT_SAMPLES:].squeeze(0).cpu().clamp(-1, 1)

    os.makedirs(out_dir, exist_ok=True)
    torchaudio.save(os.path.join(out_dir, f"epoch_{epoch:03d}_input.wav"), inp, SAMPLE_RATE)
    torchaudio.save(os.path.join(out_dir, f"epoch_{epoch:03d}_target.wav"), target, SAMPLE_RATE)
    torchaudio.save(os.path.join(out_dir, f"epoch_{epoch:03d}_pred.wav"), pred, SAMPLE_RATE)


# =========================================================
# Train
# =========================================================
def train():
    dataset = PairedWindowDataset(
        INPUT_DIR,
        TARGET_DIR,
        TRAIN_WINDOW,
        windows_per_file=WINDOWS_PER_FILE,
        onset_bias_prob=ONSET_BIAS_PROB,
    )

    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

    model = GuitarToPianoResynthModel(base_ch=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Device:          {DEVICE}")
    print(f"Paired files:    {dataset.num_files}")
    print(f"Batches/epoch:   {len(loader)}")
    print(f"Block size:      {BLOCK_SIZE} samples ({BLOCK_SIZE / SAMPLE_RATE * 1000:.2f} ms)")
    print(f"Context:         {CONTEXT_SAMPLES} samples ({CONTEXT_SAMPLES / SAMPLE_RATE * 1000:.2f} ms)")
    print(f"Unroll steps:    {UNROLL_STEPS}")
    print(f"Train window:    {TRAIN_WINDOW} samples ({TRAIN_WINDOW / SAMPLE_RATE * 1000:.2f} ms)")
    print(f"Parameters:      {total_params:,}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0

        for batch_idx, (x, y, onset) in enumerate(loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            onset = onset.to(DEVICE, non_blocking=True)

            pred, aux_list = forward_unrolled(model, x, onset)
            target = y[:, :, CONTEXT_SAMPLES:]

            loss, stats = total_loss_fn(pred, target, aux_list)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[{epoch:03d}] batch {batch_idx + 1:>4}/{len(loader)} "
                    f"loss={loss.item():.5f} "
                    f"stft={stats['stft']:.4f} mel={stats['mel']:.4f} env={stats['env']:.4f}"
                )

        scheduler.step()
        avg = running / max(len(loader), 1)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{EPOCHS} | avg_loss={avg:.6f} | lr={lr:.2e}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "sample_rate": SAMPLE_RATE,
            "block_size": BLOCK_SIZE,
            "context_samples": CONTEXT_SAMPLES,
            "unroll_steps": UNROLL_STEPS,
            "n_notes": N_NOTES,
            "n_partials": N_PARTIALS,
            "loss": avg,
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"resynth_epoch_{epoch:03d}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"Saved {ckpt_path}")

        if epoch % 5 == 0 or epoch == 1:
            render_example(model, dataset, os.path.join(CHECKPOINT_DIR, "renders"), epoch, DEVICE)

    print("Training complete.")


if __name__ == "__main__":
    train()