"""Real-time / WAV temporal-context guitar -> piano transfer."""

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

from model import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, CONTEXT_FRAMES

BLOCKSIZE = HOP_SIZE
DEVICE_IN = None
DEVICE_OUT = None
DTYPE = "float32"
LATENCY = "low"


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


def warmup(model, device, context_frames=CONTEXT_FRAMES, n_iters=10):
    print(f"Warming up ({n_iters} iters).", end="", flush=True)
    dummy = torch.randn(1, context_frames, FRAME_SIZE, device=device)
    lats = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model.infer_frame(dummy) if hasattr(model, "infer_frame") else model(dummy)[0]
            lats.append((time.perf_counter() - t0) * 1000)
    avg = float(np.mean(lats[max(0, n_iters // 2):]))
    print(f" done. avg: {avg:.2f}ms")
    return avg


def _infer(model, ctx_tensor, frame_ctx_np):
    ctx_tensor[0].copy_(torch.from_numpy(frame_ctx_np).to(ctx_tensor.device), non_blocking=True)
    with torch.no_grad():
        pred = model.infer_frame(ctx_tensor) if hasattr(model, "infer_frame") else model(ctx_tensor)[0]
    return pred[0].cpu().numpy()


def process_wav(model_path, input_path, output_path, wet, volume, device_str):
    device = get_device(device_str)
    print(f"Device : {device}")
    model = load_model(model_path, device)
    context_frames = int(getattr(model, "context_frames", CONTEXT_FRAMES))
    warmup(model, device, context_frames=context_frames)

    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} -> {SAMPLE_RATE} Hz.")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

    audio_np = audio.squeeze(0).numpy().astype(np.float32)
    audio_np = audio_np - np.mean(audio_np)

    in_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
    out_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
    ctx_ring = np.zeros((context_frames, FRAME_SIZE), dtype=np.float32)
    norm_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
    ola_win = np.hanning(FRAME_SIZE).astype(np.float32)

    rendered = []
    ctx_tensor = torch.zeros(1, context_frames, FRAME_SIZE, device=device)

    n_hops = int(np.ceil(len(audio_np) / BLOCKSIZE))
    for i in tqdm(range(n_hops), desc="Rendering"):
        start = i * BLOCKSIZE
        end = min(start + BLOCKSIZE, len(audio_np))
        in_hop = np.zeros(BLOCKSIZE, dtype=np.float32)
        in_hop[: end - start] = audio_np[start:end]

        in_ring[:-BLOCKSIZE] = in_ring[BLOCKSIZE:]
        in_ring[-BLOCKSIZE:] = in_hop

        ctx_ring[:-1] = ctx_ring[1:]
        ctx_ring[-1] = in_ring.copy()

        pred_frame = _infer(model, ctx_tensor, ctx_ring)
        out_ring += pred_frame.astype(np.float32) * ola_win
        norm_ring += ola_win

        out_hop = out_ring[:BLOCKSIZE] / np.maximum(norm_ring[:BLOCKSIZE], 1e-4)
        out_ring[:-BLOCKSIZE] = out_ring[BLOCKSIZE:]
        out_ring[-BLOCKSIZE:] = 0.0
        norm_ring[:-BLOCKSIZE] = norm_ring[BLOCKSIZE:]
        norm_ring[-BLOCKSIZE:] = 0.0

        mixed = wet * out_hop + (1.0 - wet) * in_hop
        rendered.append(np.clip(mixed * volume, -1.0, 1.0))

    rendered = np.concatenate(rendered, axis=0)[: len(audio_np)]
    out_tensor = torch.from_numpy(rendered).unsqueeze(0)
    torchaudio.save(output_path, out_tensor, SAMPLE_RATE)
    print(f"Saved -> {output_path}")


class RealTimePipeline:
    def __init__(self, model_path, device_str):
        self.device = get_device(device_str)
        print(f"Device : {self.device}")
        self.model = load_model(model_path, self.device)
        self.context_frames = int(getattr(self.model, "context_frames", CONTEXT_FRAMES))
        self.ctx_tensor = torch.zeros(1, self.context_frames, FRAME_SIZE, device=self.device)
        self.input_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.context_ring = np.zeros((self.context_frames, FRAME_SIZE), dtype=np.float32)
        self.output_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.norm_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.ola_win = np.hanning(FRAME_SIZE).astype(np.float32)
        self.running = False
        self.wet_mix = 1.0
        self.volume = 1.0
        self._frames = 0
        self._lats = []

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        t0 = time.perf_counter()

        in_hop = indata[:, 0].astype(np.float32, copy=False)
        self.input_ring[:-BLOCKSIZE] = self.input_ring[BLOCKSIZE:]
        self.input_ring[-BLOCKSIZE:] = in_hop

        self.context_ring[:-1] = self.context_ring[1:]
        self.context_ring[-1] = self.input_ring.copy()

        pred_frame = _infer(self.model, self.ctx_tensor, self.context_ring)
        self.output_ring += pred_frame.astype(np.float32) * self.ola_win
        self.norm_ring += self.ola_win

        out_hop = self.output_ring[:BLOCKSIZE] / np.maximum(self.norm_ring[:BLOCKSIZE], 1e-4)
        self.output_ring[:-BLOCKSIZE] = self.output_ring[BLOCKSIZE:]
        self.output_ring[-BLOCKSIZE:] = 0.0
        self.norm_ring[:-BLOCKSIZE] = self.norm_ring[BLOCKSIZE:]
        self.norm_ring[-BLOCKSIZE:] = 0.0

        mixed = self.wet_mix * out_hop + (1.0 - self.wet_mix) * in_hop
        outdata[:, 0] = np.clip(mixed * self.volume, -1.0, 1.0)

        self._frames += 1
        if len(self._lats) < 500:
            self._lats.append((time.perf_counter() - t0) * 1000)

    def run(self):
        warmup(self.model, self.device, context_frames=self.context_frames)
        print("\n--- LIVE MODE --------------------------------")
        print(f"SR: {SAMPLE_RATE} Hz | Hop: {BLOCKSIZE} | Window: {FRAME_SIZE} | Context: {self.context_frames}")
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
                lats = np.array(self._lats)
                print("\n--- Stats ------------------------------------")
                print(f"Frames : {self._frames:,}")
                print(f"avg: {lats.mean():.2f}ms  p95: {np.percentile(lats,95):.2f}ms  max: {lats.max():.2f}ms")
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
    p = argparse.ArgumentParser(description="Temporal-context Guitar->Piano | live mic or WAV file")
    p.add_argument("--model", required=True, help="Model checkpoint (.pt)")
    p.add_argument("--input", default=None, help="[WAV mode] Input guitar WAV")
    p.add_argument("--output", default=None, help="[WAV mode] Output path")
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
        if args.output is None:
            stem = Path(args.input).stem
            args.output = str(Path(args.input).parent / f"{stem}_piano.wav")
        process_wav(args.model, args.input, args.output, args.wet, args.volume, args.device)
    else:
        pipe = RealTimePipeline(args.model, args.device)
        pipe.wet_mix = args.wet
        pipe.volume = args.volume
        pipe.run()


if __name__ == "__main__":
    main()
