"""
realtime_jetson_live_threaded.py — Jetson-friendly live Polyphonic Guitar -> Piano

Key live-mode change:
  The PortAudio/sounddevice callback NEVER runs CUDA inference.
  It only moves audio hops through small queues. A worker thread runs the model.

This prevents JACK/ALSA xruns caused by CUDA synchronizing inside the realtime callback.

Examples
--------
# List audio devices
python realtime_jetson_live_threaded.py --model best_model.pt --list-devices

# Live ALSA/JACK using selected devices
python realtime_jetson_live_threaded.py --model best_model.pt --input-device 12 --output-device 12 --device cuda

# More safety buffer if glitches
python realtime_jetson_live_threaded.py --model best_model.pt --device cuda --queue-hops 8 --latency 0.10

# WAV offline/playback modes are preserved from the original script
python realtime_jetson_live_threaded.py --model best_model.pt --input guitar.wav --output piano.wav
python realtime_jetson_live_threaded.py --model best_model.pt --input guitar.wav --play
"""

import argparse
import os
import sys
import time
import threading
import queue
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
DTYPE = "float32"


def configure_jetson_torch(device: torch.device, fp16: bool = False):
    """Safe PyTorch flags that usually help NVIDIA Jetson/Orin."""
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # Avoid PyTorch using too many CPU threads and starving audio callback.
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)




def configure_jetson_runtime(device: torch.device, fp16: bool = False):
    """Enable CUDA settings that help Jetson Orin avoid slow default paths."""
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # Make timing more honest after warmup and avoid first-use hiccups.
        torch.cuda.empty_cache()
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark} | TF32: enabled | fp16: {fp16}")


def _model_forward(model, x: torch.Tensor) -> torch.Tensor:
    y = model.infer_frame(x) if hasattr(model, "infer_frame") else model(x)[0]
    if y.dim() == 1:
        y = y.unsqueeze(0)
    return y


def load_model(path: str, device: torch.device, fp16: bool = False):
    try:
        model = torch.jit.load(path, map_location="cpu")
        print(f"Loaded TorchScript model from {path}")
    except Exception:
        print(f"Loading state dict from {path}")
        model = DDSPGuitarToPiano()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)

    model.to(device)
    if fp16 and device.type == "cuda":
        model.half()
    model.eval()
    return model


def warmup(model, device, n_iters=30):
    print(f"Warming up ({n_iters} iters)...", end="", flush=True)
    dtype = next(model.parameters()).dtype if hasattr(model, "parameters") else torch.float32
    dummy = torch.randn(1, FRAME_SIZE, device=device, dtype=dtype)
    lats = []

    with torch.inference_mode():
        for _ in range(n_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _model_forward(model, dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)

    avg = float(np.mean(lats[max(0, n_iters // 2):]))
    print(f" done. avg: {avg:.2f} ms")
    return avg


def _infer(model, buf_tensor, frame_np: np.ndarray) -> np.ndarray:
    src = torch.from_numpy(frame_np)
    if buf_tensor.dtype != src.dtype:
        src = src.to(dtype=buf_tensor.dtype)
    buf_tensor[0].copy_(src.to(buf_tensor.device), non_blocking=True)
    with torch.inference_mode():
        pred = _model_forward(model, buf_tensor)
    return pred[0].detach().to(dtype=torch.float32, device="cpu").numpy()


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
    def __init__(self, model, device: torch.device, fp16: bool = False):
        self.model = model
        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        self.input_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.output_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        dtype = next(model.parameters()).dtype if hasattr(model, "parameters") else torch.float32
        self.buf = torch.zeros(1, FRAME_SIZE, device=device, dtype=dtype)

    def reset(self):
        self.input_ring.fill(0.0)
        self.output_ring.fill(0.0)
        if hasattr(self.model, "reset_phase"):
            self.model.reset_phase()

    def process_hop(self, in_hop: np.ndarray) -> np.ndarray:
        if len(in_hop) != HOP_SIZE:
            raise ValueError(f"Expected hop of length {HOP_SIZE}, got {len(in_hop)}")

        self.input_ring[:-HOP_SIZE] = self.input_ring[HOP_SIZE:]
        self.input_ring[-HOP_SIZE:] = in_hop

        pred_frame = _infer(self.model, self.buf, self.input_ring, self.fp16)

        self.output_ring += pred_frame
        out_hop = self.output_ring[:HOP_SIZE].copy()
        self.output_ring[:-HOP_SIZE] = self.output_ring[HOP_SIZE:]
        self.output_ring[-HOP_SIZE:] = 0.0
        return out_hop


# ============================================================
# WAV MODE — kept simple/original-style
# ============================================================


def process_wav_fast_batched(
    model,
    audio_np: np.ndarray,
    wet: float,
    volume: float,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Jetson-fast WAV path:
    - creates all FRAME_SIZE windows with torch.unfold
    - runs model in batches on CUDA
    - overlap-adds with torch.nn.functional.fold

    This removes the old per-hop CPU->GPU->CPU round trip, which is the main reason
    file processing was crawling on Orin.
    """
    import torch.nn.functional as F

    orig_len = len(audio_np)
    pad = (HOP_SIZE - (orig_len % HOP_SIZE)) % HOP_SIZE
    if pad:
        audio_np = np.concatenate([audio_np, np.zeros(pad, dtype=np.float32)])

    n_steps = len(audio_np) // HOP_SIZE
    model_dtype = next(model.parameters()).dtype if hasattr(model, "parameters") else torch.float32

    audio_t = torch.from_numpy(audio_np).to(device=device, dtype=model_dtype, non_blocking=True)
    left_pad = torch.zeros(FRAME_SIZE - HOP_SIZE, device=device, dtype=model_dtype)
    framed_src = torch.cat([left_pad, audio_t], dim=0)
    frames = framed_src.unfold(0, FRAME_SIZE, HOP_SIZE)[:n_steps].contiguous()

    preds = []
    print(f"Fast CUDA batched inference: {n_steps:,} frames, batch={batch_size}")
    with torch.inference_mode():
        for start in tqdm(range(0, n_steps, batch_size), unit="batch", ncols=72):
            x = frames[start:start + batch_size]
            y = _model_forward(model, x)
            preds.append(y.to(dtype=torch.float32))

    pred_frames = torch.cat(preds, dim=0)  # [n_steps, FRAME_SIZE]

    # Fold performs vectorized overlap-add.
    cols = pred_frames.T.unsqueeze(0)  # [1, FRAME_SIZE, n_steps]
    out_len = n_steps * HOP_SIZE + FRAME_SIZE
    ola = F.fold(
        cols,
        output_size=(1, out_len),
        kernel_size=(1, FRAME_SIZE),
        stride=(1, HOP_SIZE),
    ).view(-1)

    wet_out = ola[:len(audio_np)]
    dry = torch.from_numpy(audio_np).to(device=device, dtype=torch.float32)
    mixed = wet * wet_out + (1.0 - wet) * dry
    mixed = torch.clamp(mixed * volume, -1.0, 1.0)
    return mixed[:orig_len].detach().cpu().numpy().astype(np.float32)

def process_wav(
    model_path: str,
    input_path: str,
    output_path: str | None,
    wet: float,
    volume: float,
    device_str: str,
    play: bool,
    batch_size: int = 256,
    fp16: bool = False,
    fast_wav: bool = True,
):
    device = get_device(device_str)
    configure_jetson_torch(device, fp16)
    print(f"Device: {device}")
    configure_jetson_runtime(device, fp16=fp16)

    model = load_model(model_path, device, fp16=fp16)
    warmup(model, device)

    audio_np = prepare_audio_file(input_path)
    orig_len = len(audio_np)
    duration = orig_len / SAMPLE_RATE
    print(f"Input: {input_path} ({duration:.2f}s, {orig_len:,} samples)")

    if fast_wav and not play and output_path is not None and device.type == "cuda":
        t0 = time.perf_counter()
        collected = process_wav_fast_batched(model, audio_np, wet, volume, device, batch_size=batch_size)
        elapsed = time.perf_counter() - t0
        print(f"Fast path elapsed: {elapsed:.2f}s for {duration:.2f}s audio ({duration / max(elapsed, 1e-9):.2f}x realtime)")
        torchaudio.save(output_path, torch.from_numpy(collected).unsqueeze(0), SAMPLE_RATE)
        print(f"Saved to: {output_path}")
        return

    pad = (HOP_SIZE - (orig_len % HOP_SIZE)) % HOP_SIZE
    if pad:
        audio_np = np.concatenate([audio_np, np.zeros(pad, dtype=np.float32)])

    n_steps = len(audio_np) // HOP_SIZE
    engine = OverlapAddEngine(model, device, fp16=fp16)
    engine.reset()

    collected = np.zeros_like(audio_np) if output_path is not None else None
    lats = []

    stream = None
    if play:
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,
            device=output_device,
            channels=1,
            dtype=DTYPE,
            latency=latency,
        )
        stream.start()
        print("Playing output while processing. This is intentionally realtime-limited.")

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
            mixed = np.clip(mixed * volume, -1.0, 1.0).astype(np.float32)

            if collected is not None:
                collected[s:e] = mixed
            if stream is not None:
                stream.write(mixed.reshape(-1, 1))
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
        print(f"Latency avg: {lats.mean():.2f} ms | p95: {np.percentile(lats,95):.2f} ms | max: {lats.max():.2f} ms")


# ============================================================
# LIVE MODE — threaded for Jetson
# ============================================================
class ThreadedRealTimePipeline:
    """
    Callback thread:
      - receive input hop
      - enqueue input hop, nonblocking
      - dequeue latest processed output hop, nonblocking
      - if output not ready, output dry/zero fallback

    Worker thread:
      - dequeue input hop
      - run CUDA inference and overlap-add
      - enqueue output hop
    """

    def __init__(self, model_path, device_str="auto", fp16=False, wet=1.0, volume=1.0,
                 input_device=None, output_device=None, latency="low", queue_hops=6,
                 fallback="dry"):
        self.device = get_device(device_str)
        configure_jetson_torch(self.device, fp16)
        print(f"Inference device: {self.device}")

        configure_jetson_runtime(self.device, fp16=False)
        self.model = load_model(model_path, self.device, fp16=False)
        self.volume = 1.0
        self.wet_mix = 1.0
        self.running = False
        self.input_device = input_device
        self.output_device = output_device
        self.latency = latency
        self.queue_hops = max(2, int(queue_hops))
        self.fallback = fallback

        self.in_q = queue.Queue(maxsize=self.queue_hops)
        self.out_q = queue.Queue(maxsize=self.queue_hops)
        self.engine = OverlapAddEngine(self.model, self.device, fp16=self.fp16)
        self.engine.reset()

        self._worker = None
        self._frames = 0
        self._xruns = 0
        self._dropped_in = 0
        self._missed_out = 0
        self._worker_lats = []

        print(
            f"Window: {FRAME_SIZE} samples ({1000 * FRAME_SIZE / SAMPLE_RATE:.1f} ms) | "
            f"Hop: {HOP_SIZE} samples ({1000 * HOP_SIZE / SAMPLE_RATE:.1f} ms) | "
            f"Queue: {self.queue_hops} hops"
        )

    def _drop_oldest_and_put(self, q, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                _ = q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def _worker_loop(self):
        while self.running:
            try:
                in_hop = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                out_hop = self.engine.process_hop(in_hop)
                if len(self._worker_lats) < 1000:
                    self._worker_lats.append((time.perf_counter() - t0) * 1000.0)
                self._drop_oldest_and_put(self.out_q, out_hop)
            except Exception as exc:
                print(f"Worker error: {exc}", file=sys.stderr)

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            self._xruns += 1
            # Do not print every callback; printing can cause more xruns.

        if frames != HOP_SIZE:
            # This code is intentionally fixed-hop. Keep sounddevice blocksize == HOP_SIZE.
            outdata[:, 0] = 0.0
            return

        in_hop = indata[:, 0].copy().astype(np.float32, copy=False)

        try:
            self.in_q.put_nowait(in_hop)
        except queue.Full:
            self._dropped_in += 1
            # Drop oldest input so latency doesn't grow forever.
            try:
                _ = self.in_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.in_q.put_nowait(in_hop)
            except queue.Full:
                pass

        try:
            out_hop = self.out_q.get_nowait()
        except queue.Empty:
            self._missed_out += 1
            if self.fallback == "dry":
                out_hop = in_hop
            else:
                out_hop = np.zeros(HOP_SIZE, dtype=np.float32)

        mixed = self.wet_mix * out_hop + (1.0 - self.wet_mix) * in_hop
        outdata[:, 0] = np.clip(mixed * self.volume, -1.0, 1.0)
        self._frames += 1

    def run(self):
        warmup(self.model, self.device, fp16=self.fp16)

        print("\n--- THREADED LIVE MODE -----------------------")
        print(f"SR: {SAMPLE_RATE} Hz | Hop/blocksize: {HOP_SIZE} | Latency: {self.latency}")
        print(f"Input device: {self.input_device} | Output device: {self.output_device}")
        print(f"Wet: {self.wet_mix:.0%} piano | Volume: {self.volume:.1f}x | fallback: {self.fallback}")
        print("Controls: [q]uit  [+/-] volume  [m]ix toggle")
        print("----------------------------------------------\n")

        self.running = True
        self._worker = threading.Thread(target=self._worker_loop, name="cuda-audio-worker", daemon=True)
        self._worker.start()

        try:
            with sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=HOP_SIZE,
                device=(self.input_device, self.output_device),
                channels=1,
                dtype=DTYPE,
                latency=self.latency,
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
            if self._worker is not None:
                self._worker.join(timeout=1.0)

            print("\n--- Stats ------------------------------------")
            print(f"Callbacks       : {self._frames:,}")
            print(f"PortAudio status: {self._xruns:,}")
            print(f"Dropped input   : {self._dropped_in:,}")
            print(f"Missed output   : {self._missed_out:,}")
            if self._worker_lats:
                lats = np.array(self._worker_lats, dtype=np.float32)
                print(f"Worker avg/p95/max: {lats.mean():.2f} / {np.percentile(lats,95):.2f} / {lats.max():.2f} ms")
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
def parse_device_arg(x):
    if x is None or x == "":
        return None
    try:
        return int(x)
    except ValueError:
        return x


def main():
    p = argparse.ArgumentParser(description="Jetson threaded realtime Polyphonic Guitar->Piano")
    p.add_argument("--model", required=True, help="Model checkpoint (.pt)")
    p.add_argument("--input", default=None, help="[WAV mode] Input WAV file")
    p.add_argument("--output", default=None, help="[WAV mode] Output WAV file")
    p.add_argument("--play", action="store_true", help="[WAV mode] Play output while processing")
    p.add_argument("--wet", type=float, default=1.0, help="Wet mix 0.0-1.0")
    p.add_argument("--volume", type=float, default=1.0, help="Output volume multiplier")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--fp16", action="store_true", help="Use half precision on CUDA")
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    p.add_argument("--batch-size", type=int, default=256, help="CUDA batch size for fast WAV export")
    p.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA. Fast on Orin; disable if output quality changes.")
    p.add_argument("--no-fast-wav", action="store_true", help="Disable batched CUDA WAV export path")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    input_device = parse_device_arg(args.input_device)
    output_device = parse_device_arg(args.output_device)
    latency = args.latency
    try:
        latency = float(latency)
    except ValueError:
        pass

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
            batch_size=args.batch_size,
            fp16=args.fp16,
            fast_wav=not args.no_fast_wav,
        )
    else:
        pipe = ThreadedRealTimePipeline(
            model_path=args.model,
            device_str=args.device,
            fp16=args.fp16,
            wet=args.wet,
            volume=args.volume,
            input_device=input_device,
            output_device=output_device,
            latency=latency,
            queue_hops=args.queue_hops,
            fallback=args.fallback,
        )
        pipe.run()


if __name__ == "__main__":
    main()
