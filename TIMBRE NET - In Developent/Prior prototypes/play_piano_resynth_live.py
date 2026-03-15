import argparse
import math
import queue
import threading
import time
from typing import Dict

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


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


def compute_onset_env(x: torch.Tensor) -> torch.Tensor:
    """
    x: [1, T]
    returns: [T]
    """
    d = torch.zeros_like(x)
    d[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
    kernel = torch.ones(1, 1, 33, dtype=x.dtype, device=x.device) / 33.0
    sm = F.conv1d(d.unsqueeze(0), kernel, padding=16).squeeze(0)  # [1, T]
    return sm.squeeze(0)  # [T]


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
        T = self.block_size

        gate = torch.sigmoid(note_logits)                  # [B, N]
        onset = torch.relu(gate - state["prev_gate"])      # [B, N]

        amp0 = state["amp"]                                # [B, N]
        phase0 = state["phase"]                            # [B, N, P]
        noise0 = state["noise"]                            # [B, N]

        note_decay = self._per_sample_decay().to(device)   # [N]
        noise_decay = self._noise_decay().to(device)       # [N]

        # excite states at block start
        amp0 = torch.maximum(amp0 * note_decay.unsqueeze(0), gate * velocity)
        noise0 = noise0 * noise_decay.unsqueeze(0) + onset * noise_amt

        # harmonic distribution
        harm = F.softmax(self.harmonic_logits, dim=-1).to(device)      # [N, P]
        bright = brightness.unsqueeze(-1)                              # [B, N, 1]
        partial_idx = self.partial_nums.view(1, 1, P).to(device)      # [1,1,P]

        tilt = torch.exp(-partial_idx * (1.5 - 1.2 * bright))         # [B,N,P]
        harm = harm.unsqueeze(0) * tilt                                # [B,N,P]
        harm = harm / (harm.sum(dim=-1, keepdim=True) + 1e-8)

        # per-partial angular increment
        freqs = self.note_freqs.view(1, N, 1).to(device) * partial_idx # [1,N,P]
        phase_inc = 2.0 * math.pi * freqs / self.sample_rate           # [1,N,P]

        # sample indices for full block
        t_idx = torch.arange(1, T + 1, device=device).view(1, 1, 1, T)  # [1,1,1,T]

        # phase over whole block
        phase = phase0.unsqueeze(-1) + phase_inc.unsqueeze(-1) * t_idx   # [B,N,P,T]
        s = torch.sin(phase)

        # exponential decay envelope inside block
        decay_curve = note_decay.view(1, N, 1, 1) ** (t_idx - 1)         # [1,N,1,T]
        amp_curve = amp0.unsqueeze(-1).unsqueeze(-1) * decay_curve       # [B,N,1,T]

        tonal = (s * harm.unsqueeze(-1) * amp_curve).sum(dim=(1, 2))     # [B,T]

        # cheaper block noise: one filtered burst shape across block
        noise_decay_curve = noise_decay.view(1, N, 1) ** torch.arange(T, device=device).view(1, 1, T)
        noise_block = noise0.unsqueeze(-1) * noise_decay_curve           # [B,N,T]
        white = torch.randn(B, N, T, device=device) * noise_block
        noisy = white.sum(dim=1)                                         # [B,T]

        audio = (self.master_gain * tonal + self.noise_gain * noisy).unsqueeze(1)  # [B,1,T]

        # final states at end of block
        phase_end = torch.remainder(phase0 + phase_inc * T, 2.0 * math.pi)         # [B,N,P]
        amp_end = amp0 * (note_decay.unsqueeze(0) ** T)                             # [B,N]
        noise_end = noise0 * (noise_decay.unsqueeze(0) ** T)                        # [B,N]

        new_state = {
            "amp": amp_end,
            "phase": phase_end,
            "noise": noise_end,
            "prev_gate": gate.detach(),
        }

        aux = {"gate": gate, "onset": onset}
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
# Checkpoint loader
# =========================================================
def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    block_size = ckpt.get("block_size", 96)
    context_samples = ckpt.get("context_samples", 4096)

    model = GuitarToPianoResynthModel(base_ch=64, block_size=block_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, context_samples, block_size, ckpt


# =========================================================
# Streaming file player
# =========================================================
class LiveFileResynthPlayer:
    def __init__(
        self,
        model: GuitarToPianoResynthModel,
        input_audio: torch.Tensor,
        context_samples: int,
        block_size: int,
        device: torch.device,
        output_gain: float = 0.9,
        queue_blocks: int = 64,
    ):
        """
        input_audio: [1, T] CPU tensor
        """
        self.model = model
        self.device = device
        self.context_samples = context_samples
        self.block_size = block_size
        self.output_gain = output_gain

        self.x = input_audio.to(device)  # [1, T]

        onset = compute_onset_env(self.x)   # [T]
        self.onset_full = onset.view(1, 1, -1)  # [1, 1, T]

        T = self.x.shape[-1]
        rem = T % block_size
        self.pad = 0 if rem == 0 else (block_size - rem)
        if self.pad > 0:
            self.x = F.pad(self.x, (0, self.pad))
            self.onset_full = F.pad(self.onset_full, (0, self.pad))

        self.total_samples = self.x.shape[-1]
        self.num_blocks = self.total_samples // block_size

        self.x_ring = torch.zeros(1, 1, context_samples + block_size, device=device)
        self.onset_ring = torch.zeros(1, 1, context_samples + block_size, device=device)
        self.state = model.init_stream_state(batch_size=1, device=device)

        self.read_block_idx = 0
        self.done_generating = False
        self.stop_requested = False

        self.audio_queue = queue.Queue(maxsize=queue_blocks)
        self.generated_audio = []

    @torch.inference_mode()
    def generate_one_block(self):
        if self.read_block_idx >= self.num_blocks:
            self.done_generating = True
            return

        start = self.read_block_idx * self.block_size
        end = start + self.block_size

        x_block = self.x[:, start:end].unsqueeze(0)     # [1, 1, B]
        onset_block = self.onset_full[:, :, start:end]  # [1, 1, B]

        self.x_ring = torch.roll(self.x_ring, shifts=-self.block_size, dims=-1)
        self.onset_ring = torch.roll(self.onset_ring, shifts=-self.block_size, dims=-1)

        self.x_ring[:, :, -self.block_size:] = x_block
        self.onset_ring[:, :, -self.block_size:] = onset_block

        y_block, self.state, _ = self.model.forward_step(self.x_ring, self.onset_ring, self.state)
        y_np = (
            y_block.squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        y_np *= self.output_gain
        y_np = np.clip(y_np, -1.0, 1.0)

        self.audio_queue.put(y_np)
        self.generated_audio.append(y_np.copy())
        self.read_block_idx += 1

    def producer_loop(self):
        while not self.stop_requested and not self.done_generating:
            if not self.audio_queue.full():
                self.generate_one_block()
            else:
                time.sleep(0.001)

    def get_rendered_audio(self):
        if not self.generated_audio:
            return None
        y = np.concatenate(self.generated_audio, axis=0)
        if self.pad > 0:
            y = y[:-self.pad]
        return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--gain", type=float, default=0.9)
    parser.add_argument("--save-output", type=str, default=None)
    parser.add_argument("--prime-blocks", type=int, default=8, help="How many blocks to pre-buffer before playback")
    args = parser.parse_args()

    device = torch.device(args.device)

    model, context_samples, block_size, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    x = load_audio_mono(args.input, SAMPLE_RATE)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Epoch:             {ckpt.get('epoch', 'unknown')}")
    print(f"Input file:        {args.input}")
    print(f"Input duration:    {x.shape[-1] / SAMPLE_RATE:.2f} sec")
    print(f"Block size:        {block_size} samples ({1000 * block_size / SAMPLE_RATE:.2f} ms)")
    print(f"Context:           {context_samples} samples ({1000 * context_samples / SAMPLE_RATE:.2f} ms)")
    print(f"Device:            {device}")
    print()

    player = LiveFileResynthPlayer(
        model=model,
        input_audio=x,
        context_samples=context_samples,
        block_size=block_size,
        device=device,
        output_gain=args.gain,
    )

    producer = threading.Thread(target=player.producer_loop, daemon=True)
    producer.start()

    while player.audio_queue.qsize() < args.prime_blocks and not player.done_generating:
        time.sleep(0.005)

    finished = False
    underruns = 0

    def callback(outdata, frames, time_info, status):
        nonlocal finished, underruns

        if status:
            print(status)

        if frames != block_size:
            block = np.zeros(frames, dtype=np.float32)
            n = min(frames, block_size)
            try:
                chunk = player.audio_queue.get_nowait()
                block[:n] = chunk[:n]
            except queue.Empty:
                underruns += 1
            outdata[:, 0] = block
            if outdata.shape[1] > 1:
                outdata[:, 1:] = block[:, None]
            return

        try:
            chunk = player.audio_queue.get_nowait()
        except queue.Empty:
            if player.done_generating:
                chunk = np.zeros(block_size, dtype=np.float32)
                finished = True
            else:
                chunk = np.zeros(block_size, dtype=np.float32)
                underruns += 1

        outdata[:, 0] = chunk
        if outdata.shape[1] > 1:
            outdata[:, 1:] = chunk[:, None]

    print("Starting live-style playback...")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=block_size,
            dtype="float32",
            channels=1,
            callback=callback,
            device=args.output_device,
            latency="low",
        ):
            while True:
                if finished and player.audio_queue.empty():
                    break
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        player.stop_requested = True

    player.stop_requested = True
    producer.join(timeout=1.0)

    print(f"Playback finished. Queue underruns: {underruns}")

    if args.save_output:
        rendered = player.get_rendered_audio()
        if rendered is not None:
            sf.write(args.save_output, rendered, SAMPLE_RATE)
            print(f"Saved rendered output to: {args.save_output}")


if __name__ == "__main__":
    main()