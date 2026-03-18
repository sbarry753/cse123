"""
realtime.py — Polyphonic Guitar -> Piano
Supports:
  1) live mic/input -> live output
  2) WAV file -> live playback while processing
  3) WAV file -> saved output file

Examples
--------
# List audio devices
python realtime.py --model ./checkpoints/best_model.pt --list-devices

# Live input mode
python realtime.py --model ./checkpoints/best_model.pt

# WAV -> play while processing
python realtime.py --model ./checkpoints/best_model.pt --input ./guitar.wav --play

# WAV -> save processed file
python realtime.py --model ./checkpoints/best_model.pt --input ./guitar.wav --output ./piano_out.wav

# WAV -> play while processing AND save
python realtime.py --model ./checkpoints/best_model.pt --input ./data/guitar/plaz.wav --play --output ./piano_out.wav

export SD_ENABLE_ASIO=1
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torchaudio
from tqdm import tqdm

from model import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE


# ============================================================
# CONFIG
# ============================================================
BLOCKSIZE = HOP_SIZE
DEVICE_IN = None
DEVICE_OUT = None
DTYPE = "float32"
LATENCY = "low"

# Denoiser / cleanup defaults
DENOISE_STRENGTH = 0.12      # 0.0 -> off, try 0.08 to 0.18
LOWPASS_HZ = 12000.0         # 0 or None -> off
PRED_GAIN = 1.0              # scale model output before OLA, try 0.6 to 1.0
SOFTLIMIT_DRIVE = 0.9        # lower if still crunchy
NOISE_GATE_DB = -70.0        # mute near-total silence if desired


# ============================================================
# HELPERS
# ============================================================
def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def load_model(path: str, device: torch.device):
    try:
        model = torch.jit.load(path, map_location="cpu")
        print(f"Loaded TorchScript model from {path}")
    except Exception:
        print(f"Loading state dict from {path}")
        model = DDSPGuitarToPiano()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    model.to(device)
    model.eval()
    return model


def warmup(model, device, n_iters=10):
    print(f"Warming up ({n_iters} iters)...", end="", flush=True)
    dummy = torch.randn(1, FRAME_SIZE, device=device)
    lats = []

    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            if hasattr(model, "infer_frame"):
                _ = model.infer_frame(dummy)
            else:
                _ = model(dummy)[0]
            lats.append((time.perf_counter() - t0) * 1000.0)

    avg = float(np.mean(lats[max(0, n_iters // 2):]))
    print(f" done. avg: {avg:.2f} ms")
    return avg


def _infer(model, buf_tensor, frame_np: np.ndarray) -> np.ndarray:
    buf_tensor[0].copy_(torch.from_numpy(frame_np).to(buf_tensor.device), non_blocking=True)
    with torch.no_grad():
        pred = model.infer_frame(buf_tensor) if hasattr(model, "infer_frame") else model(buf_tensor)[0]
    return pred[0].detach().cpu().numpy().astype(np.float32)


def prepare_audio_file(input_path: str) -> np.ndarray:
    audio, sr = torchaudio.load(input_path)

    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)

    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} -> {SAMPLE_RATE} Hz...")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

    return audio.squeeze(0).numpy().astype(np.float32)


def rms_db(x: np.ndarray, eps: float = 1e-8) -> float:
    rms = np.sqrt(np.mean(np.square(x.astype(np.float32))) + eps)
    return 20.0 * np.log10(rms + eps)


def softlimit(x: np.ndarray, drive: float = 0.9) -> np.ndarray:
    return np.tanh(drive * x).astype(np.float32)


# ============================================================
# LIGHTWEIGHT DENOISER / FILTERS
# ============================================================
class SpectralDenoiser:
    """
    Very lightweight single-frame spectral floor suppressor.

    This is not a full neural denoiser; it just removes a low-level
    spectral floor that often shows up as steady buzz / fizz.
    """

    def __init__(self, n_fft: int = HOP_SIZE, strength: float = DENOISE_STRENGTH):
        self.n_fft = n_fft
        self.strength = float(strength)
        self.window = np.hanning(n_fft).astype(np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if self.strength <= 0.0:
            return audio

        x = audio.astype(np.float32)
        if len(x) != self.n_fft:
            # simple safe path
            x = x[:self.n_fft] if len(x) > self.n_fft else np.pad(x, (0, self.n_fft - len(x)))

        spec = np.fft.rfft(x * self.window)
        mag = np.abs(spec)
        phase = np.angle(spec)

        # noise floor estimate from low-percentile magnitude
        floor = np.percentile(mag, 15.0)
        cleaned_mag = np.maximum(mag - self.strength * floor, 0.0)

        y = np.fft.irfft(cleaned_mag * np.exp(1j * phase), n=self.n_fft).astype(np.float32)

        # compensate for window attenuation a bit
        return y


class OnePoleLowpass:
    """
    Cheap lowpass to shave off top-end fizz without much latency.
    """

    def __init__(self, cutoff_hz: float, sample_rate: float):
        self.cutoff_hz = cutoff_hz
        self.sample_rate = sample_rate
        if cutoff_hz is None or cutoff_hz <= 0:
            self.alpha = None
        else:
            dt = 1.0 / sample_rate
            rc = 1.0 / (2.0 * np.pi * cutoff_hz)
            self.alpha = dt / (rc + dt)
        self.z = 0.0

    def reset(self):
        self.z = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.alpha is None:
            return x

        y = np.empty_like(x, dtype=np.float32)
        z = self.z
        a = self.alpha

        for i, xi in enumerate(x):
            z = z + a * (float(xi) - z)
            y[i] = z

        self.z = z
        return y


# ============================================================
# NORMALIZED OVERLAP-ADD ENGINE
# ============================================================
class OverlapAddEngine:
    """
    Maintains rolling input/output buffers.

    For each HOP_SIZE input chunk:
      - shift new samples into input ring
      - run model on full FRAME_SIZE window
      - apply synthesis window
      - normalized overlap-add into output ring
      - emit next HOP_SIZE samples
    """

    def __init__(self, model, device: torch.device, pred_gain: float = PRED_GAIN):
        self.model = model
        self.device = device
        self.pred_gain = float(pred_gain)

        self.input_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.output_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.norm_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.buf = torch.zeros(1, FRAME_SIZE, device=device)

        # Hann synthesis window for outer OLA
        self.ola_window = np.hanning(FRAME_SIZE).astype(np.float32)

    def reset(self):
        self.input_ring.fill(0.0)
        self.output_ring.fill(0.0)
        self.norm_ring.fill(0.0)
        if hasattr(self.model, "reset_phase"):
            self.model.reset_phase()

    def process_hop(self, in_hop: np.ndarray) -> np.ndarray:
        if len(in_hop) != HOP_SIZE:
            raise ValueError(f"Expected hop of length {HOP_SIZE}, got {len(in_hop)}")

        # Shift in new input
        self.input_ring[:-HOP_SIZE] = self.input_ring[HOP_SIZE:]
        self.input_ring[-HOP_SIZE:] = in_hop

        # Model predicts full FRAME_SIZE window
        pred_frame = _infer(self.model, self.buf, self.input_ring.copy())
        pred_frame *= self.pred_gain

        # Apply outer synthesis window before OLA
        windowed = pred_frame * self.ola_window

        # Accumulate signal + normalization
        self.output_ring += windowed
        self.norm_ring += self.ola_window

        # Emit earliest hop, normalized
        den = np.maximum(self.norm_ring[:HOP_SIZE], 1e-6)
        out_hop = self.output_ring[:HOP_SIZE].copy() / den

        # Shift rings left by one hop
        self.output_ring[:-HOP_SIZE] = self.output_ring[HOP_SIZE:]
        self.output_ring[-HOP_SIZE:] = 0.0

        self.norm_ring[:-HOP_SIZE] = self.norm_ring[HOP_SIZE:]
        self.norm_ring[-HOP_SIZE:] = 0.0

        return out_hop.astype(np.float32)


# ============================================================
# OUTPUT CLEANUP CHAIN
# ============================================================
class OutputCleanup:
    def __init__(
        self,
        denoise_strength: float = DENOISE_STRENGTH,
        lowpass_hz: float = LOWPASS_HZ,
        sample_rate: float = SAMPLE_RATE,
        softlimit_drive: float = SOFTLIMIT_DRIVE,
        gate_db: float = NOISE_GATE_DB,
    ):
        self.denoiser = SpectralDenoiser(n_fft=HOP_SIZE, strength=denoise_strength)
        self.lowpass = OnePoleLowpass(lowpass_hz, sample_rate)
        self.softlimit_drive = float(softlimit_drive)
        self.gate_db = float(gate_db)

    def reset(self):
        self.lowpass.reset()

    def process(self, x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float32)

        # tiny noise gate for near-silence only
        if rms_db(y) < self.gate_db:
            return np.zeros_like(y, dtype=np.float32)

        y = self.denoiser.process(y)
        y = self.lowpass.process(y)
        y = softlimit(y, drive=self.softlimit_drive)
        return y.astype(np.float32)


# ============================================================
# WAV MODE
# ============================================================
def process_wav(
    model_path: str,
    input_path: str,
    output_path: str | None,
    wet: float,
    volume: float,
    device_str: str,
    play: bool,
):
    device = get_device(device_str)
    print(f"Device: {device}")

    model = load_model(model_path, device)
    warmup(model, device)

    audio_np = prepare_audio_file(input_path)
    orig_len = len(audio_np)
    duration = orig_len / SAMPLE_RATE
    print(f"Input: {input_path} ({duration:.2f}s, {orig_len:,} samples)")

    pad = (HOP_SIZE - (orig_len % HOP_SIZE)) % HOP_SIZE
    if pad:
        audio_np = np.concatenate([audio_np, np.zeros(pad, dtype=np.float32)])

    n_steps = len(audio_np) // HOP_SIZE
    engine = OverlapAddEngine(model, device, pred_gain=PRED_GAIN)
    cleanup = OutputCleanup()
    engine.reset()
    cleanup.reset()

    collected = np.zeros_like(audio_np) if output_path is not None else None
    lats = []

    stream = None
    if play:
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,
            device=DEVICE_OUT,
            channels=1,
            dtype=DTYPE,
            latency=LATENCY,
        )
        stream.start()
        print("Playing output while processing...")

    print(f"Processing {n_steps:,} hops...")
    try:
        for i in tqdm(range(n_steps), unit="hop", ncols=72):
            s = i * HOP_SIZE
            e = s + HOP_SIZE
            in_hop = audio_np[s:e]

            t0 = time.perf_counter()
            out_hop = engine.process_hop(in_hop)
            lats.append((time.perf_counter() - t0) * 1000.0)

            mixed = wet * out_hop + (1.0 - wet) * in_hop
            mixed = (mixed * volume).astype(np.float32)
            mixed = cleanup.process(mixed)
            mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)

            if collected is not None:
                collected[s:e] = mixed

            if stream is not None:
                stream.write(mixed.reshape(-1, 1))

        # Flush tail
        if collected is not None:
            tail_hops = FRAME_SIZE // HOP_SIZE
            tail_out = []

            for _ in range(tail_hops):
                zero_hop = np.zeros(HOP_SIZE, dtype=np.float32)
                out_hop = engine.process_hop(zero_hop)
                mixed = (wet * out_hop * volume).astype(np.float32)
                mixed = cleanup.process(mixed)
                mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
                tail_out.append(mixed)

                if stream is not None:
                    stream.write(mixed.reshape(-1, 1))

            if tail_out:
                tail_cat = np.concatenate(tail_out, axis=0)
                collected = np.concatenate([collected, tail_cat], axis=0)

    finally:
        if stream is not None:
            stream.stop()
            stream.close()

    if collected is not None and output_path is not None:
        collected = collected[:orig_len]
        torchaudio.save(
            output_path,
            torch.from_numpy(collected).unsqueeze(0),
            SAMPLE_RATE,
        )
        print(f"Saved to: {output_path}")

    lats = np.array(lats, dtype=np.float32)
    if len(lats):
        print(
            f"Latency avg: {lats.mean():.2f} ms | "
            f"p95: {np.percentile(lats, 95):.2f} ms | "
            f"max: {lats.max():.2f} ms"
        )


# ============================================================
# LIVE MIC MODE
# ============================================================
class RealTimePipeline:
    def __init__(self, model_path: str, device_str: str = "auto"):
        self.device = get_device(device_str)
        print(f"Inference device: {self.device}")

        self.model = load_model(model_path, self.device)
        self.volume = 1.0
        self.wet_mix = 1.0
        self.running = False
        self._lats = []
        self._frames = 0

        self.engine = OverlapAddEngine(self.model, self.device, pred_gain=PRED_GAIN)
        self.cleanup = OutputCleanup()
        self.engine.reset()
        self.cleanup.reset()

        print(
            f"Window: {FRAME_SIZE} samples ({1000 * FRAME_SIZE / SAMPLE_RATE:.1f} ms) | "
            f"Hop: {HOP_SIZE} samples ({1000 * HOP_SIZE / SAMPLE_RATE:.1f} ms)"
        )

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        t0 = time.perf_counter()

        in_hop = indata[:, 0].astype(np.float32)
        out_hop = self.engine.process_hop(in_hop)

        mixed = self.wet_mix * out_hop + (1.0 - self.wet_mix) * in_hop
        mixed = (mixed * self.volume).astype(np.float32)
        mixed = self.cleanup.process(mixed)
        outdata[:, 0] = np.clip(mixed, -1.0, 1.0)

        self._frames += 1
        if len(self._lats) < 500:
            self._lats.append((time.perf_counter() - t0) * 1000.0)

    def run(self):
        warmup(self.model, self.device)

        print("\n--- LIVE MODE --------------------------------")
        print(f"SR: {SAMPLE_RATE} Hz | Hop: {BLOCKSIZE} | Window: {FRAME_SIZE}")
        print(f"Wet: {self.wet_mix:.0%} piano | Volume: {self.volume:.1f}x")
        print("Controls: [q]uit  [+/-] volume  [m]ix toggle")
        print("----------------------------------------------\n")

        self.running = True
        try:
            with sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                device=(DEVICE_IN, DEVICE_OUT),
                channels=1,
                dtype=DTYPE,
                latency=LATENCY,
                callback=self.audio_callback,
            ):
                print("Streaming... (type command + Enter)\n")
                while self.running:
                    try:
                        self._handle(input().strip().lower())
                    except EOFError:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if self._lats:
                lats = np.array(self._lats, dtype=np.float32)
                print("\n--- Stats ------------------------------------")
                print(f"Frames : {self._frames:,}")
                print(
                    f"avg: {lats.mean():.2f} ms  "
                    f"p95: {np.percentile(lats,95):.2f} ms  "
                    f"max: {lats.max():.2f} ms"
                )
                print("----------------------------------------------")

    def _handle(self, cmd):
        if cmd == "q":
            self.running = False
        elif cmd == "+":
            self.volume = min(4.0, self.volume + 0.1)
            print(f"  Volume: {self.volume:.1f}x")
        elif cmd == "-":
            self.volume = max(0.0, self.volume - 0.1)
            print(f"  Volume: {self.volume:.1f}x")
        elif cmd == "m":
            self.wet_mix = 0.0 if self.wet_mix > 0.5 else 1.0
            print(f"  Mix: {'piano' if self.wet_mix > 0.5 else 'dry guitar'}")
        elif cmd:
            print(f"  Unknown: '{cmd}'")


# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Polyphonic Guitar->Piano | live mic or WAV")
    p.add_argument("--model", required=True, help="Model checkpoint (.pt)")
    p.add_argument("--input", default=None, help="[WAV mode] Input WAV file")
    p.add_argument("--output", default=None, help="[WAV mode] Output WAV file")
    p.add_argument("--play", action="store_true", help="[WAV mode] Play output while processing")
    p.add_argument("--wet", type=float, default=1.0, help="Wet mix 0.0-1.0")
    p.add_argument("--volume", type=float, default=1.0, help="Output volume multiplier")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.input:
        if not os.path.isfile(args.input):
            print(f"Error: file not found: {args.input}")
            sys.exit(1)

        if args.output is None and not args.play:
            stem = Path(args.input).stem
            args.output = str(Path(args.input).parent / f"{stem}_piano.wav")

        process_wav(
            model_path=args.model,
            input_path=args.input,
            output_path=args.output,
            wet=args.wet,
            volume=args.volume,
            device_str=args.device,
            play=args.play,
        )
    else:
        pipe = RealTimePipeline(args.model, args.device)
        pipe.wet_mix = args.wet
        pipe.volume = args.volume
        pipe.run()


if __name__ == "__main__":
    main()