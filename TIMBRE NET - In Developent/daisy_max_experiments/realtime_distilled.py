"""
realtime_distilled.py - Real-time runner for the MAX78000 distilled student.

The distilled TimbreStudent predicts a normalized piano/guitar magnitude mask
from a guitar log-magnitude spectrogram. This script wraps that student with the
same streaming overlap-add interface used by realtime.py:

  audio frame -> STFT -> normalized log-mag -> student mask -> ISTFT -> audio

Examples
--------
# List audio devices
python realtime_distilled.py --model ./checkpoints_distilled/best_model.pt --list-devices

# Live input mode
python realtime_distilled.py --model ./checkpoints_distilled/best_model.pt

# WAV -> saved output file
python realtime_distilled.py --model ./checkpoints_distilled/best_model.pt --input ./guitar.wav --output ./piano_out.wav

# WAV -> play while processing and save
python realtime_distilled.py --model ./checkpoints_distilled/best_model.pt --input ./guitar.wav --play --output ./piano_out.wav
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from model import FRAME_SIZE, HOP_SIZE, N_FFT, SAMPLE_RATE
from unet_distilled import TimbreUNetStudent
import ai8x


BLOCKSIZE = HOP_SIZE
DEVICE_IN = None
DEVICE_OUT = None
DTYPE = "float32"
LATENCY = "low"


def get_sounddevice():
    import sounddevice as sd

    return sd


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


class DistilledStudentRealtime(torch.nn.Module):
    """
    Waveform-facing wrapper around TimbreUNetStudent.

    TimbreUNetStudent is trained against target_mask = clamp(piano_mag/guitar_mag,
    0, 2) / 2, so inference maps the predicted [0, 1] mask back to [0, 2]
    before applying it to the guitar magnitude.
    """

    def __init__(
        self,
        student: TimbreUNetStudent,
        frame_size: int = FRAME_SIZE,
        hop_size: int = HOP_SIZE,
        n_fft: int = N_FFT,
        log_scale: float = 6.0,
        mask_gain: float = 2.0,
        dry_blend: float = 0.0,
    ):
        super().__init__()
        self.student = student
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.log_scale = log_scale
        self.mask_gain = mask_gain
        self.dry_blend = dry_blend
        self.register_buffer("window", torch.hann_window(frame_size))

    @staticmethod
    def _pad_to_multiple_2d(x: torch.Tensor, multiple: int = 4) -> torch.Tensor:
        freq_bins, time_frames = x.shape[-2:]
        pad_freq = (multiple - freq_bins % multiple) % multiple
        pad_time = (multiple - time_frames % multiple) % multiple
        if pad_freq == 0 and pad_time == 0:
            return x
        return torch.nn.functional.pad(x, (0, pad_time, 0, pad_freq))

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            audio.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.frame_size,
            window=self.window.to(audio.device),
            return_complex=True,
            center=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.frame_size,
            window=self.window.to(spec.device),
            center=True,
            length=length,
        )

    def forward(self, audio_frame: torch.Tensor) -> torch.Tensor:
        length = audio_frame.shape[-1]
        spec = self._stft(audio_frame)
        mag = torch.abs(spec)
        phase = torch.angle(spec)

        student_input = torch.clamp(torch.log1p(mag) / self.log_scale, 0.0, 1.0).unsqueeze(1)
        student_input = self._pad_to_multiple_2d(student_input)
        pred_mask = torch.clamp(self.student(student_input), 0.0, 1.0).squeeze(1)
        pred_mask = pred_mask[..., :mag.shape[-2], :mag.shape[-1]]
        out_mag = mag * (pred_mask * self.mask_gain)

        out_spec = torch.polar(out_mag, phase)
        audio_out = self._istft(out_spec, length=length)
        audio_out = torch.tanh(audio_out)

        if self.dry_blend > 0.0:
            audio_out = (1.0 - self.dry_blend) * audio_out + self.dry_blend * audio_frame
        return audio_out

    @torch.no_grad()
    def infer_frame(self, audio_frame: torch.Tensor) -> torch.Tensor:
        return self.forward(audio_frame)

    def reset_phase(self):
        pass


def _checkpoint_state(payload):
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def padded_spectrogram_dimensions(n_fft: int, frame_size: int, hop_size: int, multiple: int = 4):
    freq_bins = n_fft // 2 + 1
    time_frames = frame_size // hop_size + 1
    padded_freq = freq_bins + (multiple - freq_bins % multiple) % multiple
    padded_time = time_frames + (multiple - time_frames % multiple) % multiple
    return padded_freq, padded_time


def load_model(path: str, device: torch.device, args):
    if path.endswith(".pt") or path.endswith(".pth"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        frame_size = int(payload.get("frame_size", args.frame_size)) if isinstance(payload, dict) else args.frame_size
        hop_size = int(payload.get("hop_size", args.hop_size)) if isinstance(payload, dict) else args.hop_size
        n_fft = int(payload.get("n_fft", args.n_fft)) if isinstance(payload, dict) else args.n_fft
        log_scale = float(payload.get("log_scale", args.log_scale)) if isinstance(payload, dict) else args.log_scale
        base_ch = int(payload.get("base_ch", args.base_ch)) if isinstance(payload, dict) else args.base_ch

        if (frame_size, hop_size, n_fft) != (FRAME_SIZE, HOP_SIZE, N_FFT):
            print(
                "Warning: checkpoint dimensions differ from model.py constants; "
                f"using checkpoint frame={frame_size}, hop={hop_size}, n_fft={n_fft}"
            )

        ai8x.set_device(
            device=args.ai8x_device,
            simulate=args.simulate,
            round_avg=args.avg_pool_rounding,
        )
        student = TimbreUNetStudent(
            num_classes=1,
            num_channels=1,
            dimensions=padded_spectrogram_dimensions(n_fft, frame_size, hop_size),
            base_ch=base_ch,
        )
        student.load_state_dict(_checkpoint_state(payload))
        model = DistilledStudentRealtime(
            student,
            frame_size=frame_size,
            hop_size=hop_size,
            n_fft=n_fft,
            log_scale=log_scale,
            mask_gain=args.mask_gain,
            dry_blend=args.dry_blend,
        )
        print(f"Loaded distilled student checkpoint from {path}")
    else:
        model = torch.jit.load(path, map_location="cpu")
        print(f"Loaded TorchScript model from {path}")

    model.to(device)
    model.eval()
    return model


def warmup(model, device, frame_size: int, n_iters=10):
    print(f"Warming up ({n_iters} iters)...", end="", flush=True)
    dummy = torch.randn(1, frame_size, device=device)
    lats = []

    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model.infer_frame(dummy) if hasattr(model, "infer_frame") else model(dummy)
            lats.append((time.perf_counter() - t0) * 1000.0)

    avg = float(np.mean(lats[max(0, n_iters // 2):]))
    print(f" done. avg: {avg:.2f} ms")
    return avg


def _infer(model, buf_tensor, frame_np: np.ndarray) -> np.ndarray:
    buf_tensor[0].copy_(torch.from_numpy(frame_np).to(buf_tensor.device), non_blocking=True)
    with torch.no_grad():
        pred = model.infer_frame(buf_tensor) if hasattr(model, "infer_frame") else model(buf_tensor)
    return pred[0].detach().cpu().numpy().astype(np.float32)


def prepare_audio_file(input_path: str) -> np.ndarray:
    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} -> {SAMPLE_RATE} Hz...")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.squeeze(0).numpy().astype(np.float32)


class OverlapAddEngine:
    def __init__(self, model, device: torch.device, frame_size: int, hop_size: int):
        self.model = model
        self.device = device
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.input_ring = np.zeros(frame_size, dtype=np.float32)
        self.output_ring = np.zeros(frame_size, dtype=np.float32)
        self.buf = torch.zeros(1, frame_size, device=device)

    def reset(self):
        self.input_ring.fill(0.0)
        self.output_ring.fill(0.0)
        if hasattr(self.model, "reset_phase"):
            self.model.reset_phase()

    def process_hop(self, in_hop: np.ndarray) -> np.ndarray:
        if len(in_hop) != self.hop_size:
            raise ValueError(f"Expected hop of length {self.hop_size}, got {len(in_hop)}")

        self.input_ring[:-self.hop_size] = self.input_ring[self.hop_size:]
        self.input_ring[-self.hop_size:] = in_hop

        pred_frame = _infer(self.model, self.buf, self.input_ring.copy())

        self.output_ring += pred_frame
        out_hop = self.output_ring[:self.hop_size].copy()
        self.output_ring[:-self.hop_size] = self.output_ring[self.hop_size:]
        self.output_ring[-self.hop_size:] = 0.0
        return out_hop


def model_sizes(model) -> tuple[int, int]:
    return int(getattr(model, "frame_size", FRAME_SIZE)), int(getattr(model, "hop_size", HOP_SIZE))


def process_wav(
    model_path: str,
    input_path: str,
    output_path: str | None,
    wet: float,
    volume: float,
    device_str: str,
    play: bool,
    args,
):
    device = get_device(device_str)
    print(f"Device: {device}")

    model = load_model(model_path, device, args)
    frame_size, hop_size = model_sizes(model)
    warmup(model, device, frame_size)

    audio_np = prepare_audio_file(input_path)
    orig_len = len(audio_np)
    duration = orig_len / SAMPLE_RATE
    print(f"Input: {input_path} ({duration:.2f}s, {orig_len:,} samples)")

    pad = (hop_size - (orig_len % hop_size)) % hop_size
    if pad:
        audio_np = np.concatenate([audio_np, np.zeros(pad, dtype=np.float32)])

    n_steps = len(audio_np) // hop_size
    engine = OverlapAddEngine(model, device, frame_size, hop_size)
    engine.reset()

    collected = np.zeros_like(audio_np) if output_path is not None else None
    lats = []

    stream = None
    if play:
        sd = get_sounddevice()
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=hop_size,
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
            s = i * hop_size
            e = s + hop_size
            in_hop = audio_np[s:e]

            t0 = time.perf_counter()
            out_hop = engine.process_hop(in_hop)
            lats.append((time.perf_counter() - t0) * 1000.0)

            mixed = wet * out_hop + (1.0 - wet) * in_hop
            mixed = np.clip(mixed * volume, -1.0, 1.0).astype(np.float32)

            if collected is not None:
                collected[s:e] = mixed
            if stream is not None:
                stream.write(mixed.reshape(-1, 1))

        if collected is not None:
            tail_hops = frame_size // hop_size
            tail_out = []
            for _ in range(tail_hops):
                zero_hop = np.zeros(hop_size, dtype=np.float32)
                out_hop = engine.process_hop(zero_hop)
                mixed = np.clip((wet * out_hop) * volume, -1.0, 1.0).astype(np.float32)
                tail_out.append(mixed)
                if stream is not None:
                    stream.write(mixed.reshape(-1, 1))
            if tail_out:
                collected = np.concatenate([collected, np.concatenate(tail_out, axis=0)], axis=0)
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

    if collected is not None and output_path is not None:
        collected = collected[:orig_len]
        torchaudio.save(output_path, torch.from_numpy(collected).unsqueeze(0), SAMPLE_RATE)
        print(f"Saved to: {output_path}")

    lats = np.array(lats, dtype=np.float32)
    if len(lats):
        print(
            f"Latency avg: {lats.mean():.2f} ms | "
            f"p95: {np.percentile(lats, 95):.2f} ms | "
            f"max: {lats.max():.2f} ms"
        )


class RealTimePipeline:
    def __init__(self, model_path: str, args):
        self.device = get_device(args.device)
        print(f"Inference device: {self.device}")

        self.model = load_model(model_path, self.device, args)
        self.frame_size, self.hop_size = model_sizes(self.model)
        self.volume = args.volume
        self.wet_mix = args.wet
        self.running = False
        self._lats = []
        self._frames = 0

        self.engine = OverlapAddEngine(self.model, self.device, self.frame_size, self.hop_size)
        self.engine.reset()

        print(
            f"Window: {self.frame_size} samples ({1000 * self.frame_size / SAMPLE_RATE:.1f} ms) | "
            f"Hop: {self.hop_size} samples ({1000 * self.hop_size / SAMPLE_RATE:.1f} ms)"
        )

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        t0 = time.perf_counter()
        in_hop = indata[:, 0].astype(np.float32)
        out_hop = self.engine.process_hop(in_hop)

        mixed = self.wet_mix * out_hop + (1.0 - self.wet_mix) * in_hop
        outdata[:, 0] = np.clip(mixed * self.volume, -1.0, 1.0)

        self._frames += 1
        if len(self._lats) < 500:
            self._lats.append((time.perf_counter() - t0) * 1000.0)

    def run(self):
        warmup(self.model, self.device, self.frame_size)

        print("\n--- LIVE DISTILLED MODE ----------------------")
        print(f"SR: {SAMPLE_RATE} Hz | Hop: {self.hop_size} | Window: {self.frame_size}")
        print(f"Wet: {self.wet_mix:.0%} piano | Volume: {self.volume:.1f}x")
        print("Controls: [q]uit  [+/-] volume  [m]ix toggle")
        print("----------------------------------------------\n")

        self.running = True
        try:
            sd = get_sounddevice()
            with sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=self.hop_size,
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


def main():
    p = argparse.ArgumentParser(description="Distilled Guitar->Piano student | live mic or WAV")
    p.add_argument("--model", required=True, help="Distilled model checkpoint (.pt) or TorchScript file")
    p.add_argument("--input", default=None, help="[WAV mode] Input WAV file")
    p.add_argument("--output", default=None, help="[WAV mode] Output WAV file")
    p.add_argument("--play", action="store_true", help="[WAV mode] Play output while processing")
    p.add_argument("--wet", type=float, default=1.0, help="Wet mix 0.0-1.0")
    p.add_argument("--volume", type=float, default=1.0, help="Output volume multiplier")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--frame_size", type=int, default=FRAME_SIZE)
    p.add_argument("--hop_size", type=int, default=HOP_SIZE)
    p.add_argument("--n_fft", type=int, default=N_FFT)
    p.add_argument("--base_ch", type=int, default=8)
    p.add_argument("--log_scale", type=float, default=6.0)
    p.add_argument("--mask_gain", type=float, default=2.0, help="Scale normalized student mask back to magnitude ratio")
    p.add_argument("--dry_blend", type=float, default=0.0, help="Small dry blend inside the student wrapper")
    p.add_argument("--ai8x_device", type=int, default=85, help="ai8x hardware device code, 85 for MAX78000")
    p.add_argument("--simulate", action="store_true", help="Use ai8x hardware-simulation quantization behavior")
    p.add_argument("--avg_pool_rounding", action="store_true", help="Use ai8x average-pooling rounding mode")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = p.parse_args()

    if args.list_devices:
        sd = get_sounddevice()
        print(sd.query_devices())
        return

    if args.input:
        if not os.path.isfile(args.input):
            print(f"Error: file not found: {args.input}")
            sys.exit(1)

        if args.output is None and not args.play:
            stem = Path(args.input).stem
            args.output = str(Path(args.input).parent / f"{stem}_piano_distilled.wav")

        process_wav(
            model_path=args.model,
            input_path=args.input,
            output_path=args.output,
            wet=args.wet,
            volume=args.volume,
            device_str=args.device,
            play=args.play,
            args=args,
        )
    else:
        RealTimePipeline(args.model, args).run()


if __name__ == "__main__":
    main()
