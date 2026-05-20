"""
realtime.py — Polyphonic Guitar -> Piano
Supports:
  1) live mic/input -> live output
  2) WAV file -> live playback while processing
  3) WAV file -> saved output file

  python realtime.py --model checkpoints_teach_tcn1D_p/best_model.pt \
  --base_ch 64 --frame_size 2048 --output audio_out/tcn_1d_phase.wav

Examples
--------
# List audio devices
python realtime.py --model ./checkpoints/model_scripted.pt --list-devices

# Live input mode
python realtime.py --model ./checkpoints/model_scripted.pt

# WAV -> play while processing
python realtime.py --model ./checkpoints/model_scripted.pt --input ./guitar.wav --play

# WAV -> save processed file
python realtime.py --model ./checkpoints/model_scripted.pt --input ./guitar.wav --output ./piano_out.wav

# WAV -> play while processing AND save
python realtime.py --model ./checkpoints/model_scripted.pt --input ./guitar.wav --play --output ./piano_out.wav
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

from model import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, N_FFT


# ============================================================
# CONFIG
# ============================================================
BLOCKSIZE = HOP_SIZE          # callback/output chunk size
DEVICE_IN = None              # set to input device index/name if needed
DEVICE_OUT = None             # set to output device index/name if needed
DTYPE = "float32"
LATENCY = "low"
sd = None


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


def get_sounddevice():
    global sd
    if sd is None:
        import sounddevice as sounddevice_module

        sd = sounddevice_module
    return sd


def explicit_cli_args(parser, argv):
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit = set()
    for token in argv:
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit.add(dest)
    return explicit


def checkpoint_training_args(payload):
    if not isinstance(payload, dict):
        return {}
    training_args = payload.get("training_args")
    if isinstance(training_args, dict):
        return training_args
    return payload


def checkpoint_value(payload, name, default=None):
    training_args = checkpoint_training_args(payload)
    if name in training_args and training_args[name] is not None:
        return training_args[name]
    if isinstance(payload, dict):
        return payload.get(name, default)
    return default


def apply_checkpoint_config(args, payload):
    explicit = getattr(args, "_explicit_args", set())
    names = (
        "frame_size",
        "hop_size",
        "n_fft",
        "win_length",
        "hidden_size",
        "base_ch",
        "phase_tcn_ch",
        "phase_tcn_layers",
        "phase_max_delta",
    )
    for name in names:
        if name in explicit:
            continue
        value = checkpoint_value(payload, name)
        if value is not None:
            setattr(args, name, value)


def load_model(path: str, device: torch.device, args):
    try:
        model = torch.jit.load(path, map_location="cpu")
        print(f"Loaded TorchScript model from {path}")
    except Exception:
        print(f"Loading state dict from {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        apply_checkpoint_config(args, ckpt)
        args.win_length = int(args.win_length or checkpoint_value(ckpt, "win_length", args.n_fft))
        args.phase_tcn_ch = int(args.phase_tcn_ch or checkpoint_value(ckpt, "phase_tcn_ch", 16))
        args.phase_tcn_layers = int(args.phase_tcn_layers or checkpoint_value(ckpt, "phase_tcn_layers", 3))
        args.phase_max_delta = float(
            args.phase_max_delta
            if args.phase_max_delta is not None
            else checkpoint_value(ckpt, "phase_max_delta", 0.5)
        )
        model = DDSPGuitarToPiano(
            sample_rate=SAMPLE_RATE,
            frame_size=args.frame_size,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_size=args.hop_size,
            hidden_size=args.hidden_size,
            base_ch=args.base_ch,
            phase_tcn_ch=args.phase_tcn_ch,
            phase_tcn_layers=args.phase_tcn_layers,
            phase_max_delta=args.phase_max_delta,
        )
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    model.to(device)
    model.eval()
    return model


def warmup(model, device, frame_size, n_iters=10):
    print(f"Warming up ({n_iters} iters)...", end="", flush=True)
    dummy = torch.randn(1, frame_size, device=device)
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


# ============================================================
# OVERLAP-ADD ENGINE
# ============================================================
class OverlapAddEngine:
    """
    Maintains rolling input/output buffers.

    For each hop-size input chunk:
      - shift new samples into input ring
      - run model on full frame window
      - render the full predicted frame according to render_mode
      - emit next hop-size samples
    """

    MODES = {"windowed_ola", "last_hop", "legacy_ola"}

    def __init__(
        self,
        model,
        device: torch.device,
        frame_size: int,
        hop_size: int,
        render_mode: str = "windowed_ola",
    ):
        self.model = model
        self.device = device
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        if render_mode not in self.MODES:
            raise ValueError(f"Unsupported render_mode={render_mode!r}; expected one of {sorted(self.MODES)}")
        self.render_mode = render_mode
        self.input_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.output_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.norm_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.synthesis_window = np.hanning(self.frame_size).astype(np.float32)
        self.buf = torch.zeros(1, self.frame_size, device=device)

    def reset(self):
        self.input_ring.fill(0.0)
        self.output_ring.fill(0.0)
        self.norm_ring.fill(0.0)
        if hasattr(self.model, "reset_phase"):
            self.model.reset_phase()

    @property
    def wet_path_delay_samples(self) -> int:
        if self.render_mode == "last_hop":
            return 0
        return self.frame_size - self.hop_size

    def process_hop(self, in_hop: np.ndarray) -> np.ndarray:
        if len(in_hop) != self.hop_size:
            raise ValueError(f"Expected hop of length {self.hop_size}, got {len(in_hop)}")

        # Shift in new input
        self.input_ring[:-self.hop_size] = self.input_ring[self.hop_size:]
        self.input_ring[-self.hop_size:] = in_hop

        # Model predicts full frame window
        pred_frame = _infer(self.model, self.buf, self.input_ring.copy())

        if self.render_mode == "last_hop":
            return pred_frame[-self.hop_size:].copy()

        if self.render_mode == "legacy_ola":
            # Historical behavior retained for A/B comparisons. This applies
            # overlap-count gain when predictions are locally consistent.
            self.output_ring += pred_frame
            out_hop = self.output_ring[:self.hop_size].copy()
        else:
            self.output_ring += pred_frame * self.synthesis_window
            self.norm_ring += self.synthesis_window

            denom = np.maximum(self.norm_ring[:self.hop_size], 1e-6)
            out_hop = (self.output_ring[:self.hop_size] / denom).astype(np.float32)

        # Shift output ring left by one hop
        self.output_ring[:-self.hop_size] = self.output_ring[self.hop_size:]
        self.output_ring[-self.hop_size:] = 0.0
        self.norm_ring[:-self.hop_size] = self.norm_ring[self.hop_size:]
        self.norm_ring[-self.hop_size:] = 0.0

        return out_hop


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
    args
):
    device = get_device(device_str)
    print(f"Device: {device}")

    model = load_model(model_path, device, args)
    warmup(model, device, args.frame_size)

    audio_np = prepare_audio_file(input_path)
    orig_len = len(audio_np)
    duration = orig_len / SAMPLE_RATE
    print(f"Input: {input_path} ({duration:.2f}s, {orig_len:,} samples)")

    pad = (args.hop_size - (orig_len % args.hop_size)) % args.hop_size
    if pad:
        audio_np = np.concatenate([audio_np, np.zeros(pad, dtype=np.float32)])

    n_steps = len(audio_np) // args.hop_size
    engine = OverlapAddEngine(model, device, args.frame_size, args.hop_size, args.render_mode)
    engine.reset()
    print(
        f"Render mode: {args.render_mode} | "
        f"Wet-path delay: {engine.wet_path_delay_samples} samples "
        f"({1000 * engine.wet_path_delay_samples / SAMPLE_RATE:.1f} ms)"
    )
    if args.render_mode == "legacy_ola":
        overlap = args.frame_size / args.hop_size
        print(f"Warning: legacy_ola can apply about {overlap:.1f}x overlap gain and clip.")
    if 0.0 < wet < 1.0 and engine.wet_path_delay_samples:
        print("Note: partial wet mixes combine dry input with delayed wet output; use --wet 1.0 for artifact checks.")

    collected = np.zeros_like(audio_np) if output_path is not None else None
    lats = []

    stream = None
    if play:
        sounddevice = get_sounddevice()
        stream = sounddevice.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=args.hop_size,
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
            s = i * args.hop_size
            e = s + args.hop_size
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

        # Flush the tail so saved file captures final overlap-add decay
        if collected is not None:
            tail_hops = args.frame_size // args.hop_size
            tail_start = len(audio_np)
            tail_out = []

            for _ in range(tail_hops):
                zero_hop = np.zeros(args.hop_size, dtype=np.float32)
                out_hop = engine.process_hop(zero_hop)
                mixed = np.clip((wet * out_hop) * volume, -1.0, 1.0).astype(np.float32)
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
    def __init__(self, args):
        self.device = get_device(args.device)
        print(f"Inference device: {self.device}")

        self.model = load_model(args.model, self.device, args)
        self.volume = 1.0
        self.wet_mix = 1.0
        self.running = False
        self._lats = []
        self._frames = 0

        self.engine = OverlapAddEngine(
            self.model,
            self.device,
            args.frame_size,
            args.hop_size,
            args.render_mode,
        )
        self.engine.reset()
        self.frame_size = int(args.frame_size)
        self.hop_size = int(args.hop_size)
        self.render_mode = args.render_mode

        print(
            f"Window: {self.frame_size} samples ({1000 * self.frame_size / SAMPLE_RATE:.1f} ms) | "
            f"Hop: {self.hop_size} samples ({1000 * self.hop_size / SAMPLE_RATE:.1f} ms)"
        )
        print(
            f"Render mode: {self.render_mode} | "
            f"Wet-path delay: {self.engine.wet_path_delay_samples} samples "
            f"({1000 * self.engine.wet_path_delay_samples / SAMPLE_RATE:.1f} ms)"
        )
        if self.render_mode == "legacy_ola":
            overlap = self.frame_size / self.hop_size
            print(f"Warning: legacy_ola can apply about {overlap:.1f}x overlap gain and clip.")

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

        print("\n--- LIVE MODE --------------------------------")
        print(
            f"SR: {SAMPLE_RATE} Hz | Hop: {self.hop_size} | "
            f"Window: {self.frame_size} | Render: {self.render_mode}"
        )
        print(f"Wet: {self.wet_mix:.0%} piano | Volume: {self.volume:.1f}x")
        if 0.0 < self.wet_mix < 1.0 and self.engine.wet_path_delay_samples:
            print("Partial wet mixes combine dry input with delayed wet output.")
        print("Controls: [q]uit  [+/-] volume  [m]ix toggle")
        print("----------------------------------------------\n")

        self.running = True
        try:
            sounddevice = get_sounddevice()
            with sounddevice.Stream(
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


# ============================================================
# MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Polyphonic Guitar->Piano | live mic or WAV")
    p.add_argument("--model", required=True, help="Model checkpoint (.pt)")
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--phase_tcn_ch", type=int, default=None)
    p.add_argument("--phase_tcn_layers", type=int, default=None)
    p.add_argument("--phase_max_delta", type=float, default=None)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--n_fft", type=int, default=N_FFT)
    p.add_argument("--win_length", type=int, default=None)
    p.add_argument("--hop_size", type=int, default=HOP_SIZE)
    p.add_argument("--frame_size", type=int, default=FRAME_SIZE)
    p.add_argument("--input", default="overfit/guitar/plaz.wav", help="[WAV mode] Input WAV file")
    p.add_argument("--output", default=None, help="[WAV mode] Output WAV file")
    p.add_argument("--play", action="store_true", help="[WAV mode] Play output while processing")
    p.add_argument("--wet", type=float, default=1.0, help="Wet mix 0.0-1.0")
    p.add_argument("--volume", type=float, default=1.0, help="Output volume multiplier")
    p.add_argument(
        "--render-mode",
        choices=sorted(OverlapAddEngine.MODES),
        default="windowed_ola",
        help=(
            "Streaming renderer: windowed_ola fixes overlap gain with normalized Hann OLA; "
            "last_hop emits pred_frame[-hop_size:]; legacy_ola preserves the old unnormalized OLA."
        ),
    )
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = p.parse_args()
    args._explicit_args = explicit_cli_args(p, sys.argv[1:])

    if args.list_devices:
        print(get_sounddevice().query_devices())
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
            args=args
        )
    else:
        pipe = RealTimePipeline(args)
        pipe.wet_mix = args.wet
        pipe.volume = args.volume
        pipe.run()


if __name__ == "__main__":
    main()
