"""
Jetson Orin realtime/playback runner for the pruned Guitar->Piano model.

Works like the original realtime.py for WAV playback:
  python3 jetson_realtime.py --model ./model_pruned_scripted.pt --input ./guitar.wav --play
  python3 jetson_realtime.py --model ./model_pruned_scripted.pt --input ./guitar.wav --play --output ./piano_out.wav

Also supports live input:
  python3 jetson_realtime.py --model ./model_pruned_scripted.pt --live
"""

from __future__ import annotations

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

from model import DDSPGuitarToPiano, FRAME_SIZE, HOP_SIZE, SAMPLE_RATE

DTYPE = "float32"
DEFAULT_LATENCY = "low"


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(path: str, device: torch.device):
    try:
        model = torch.jit.load(path, map_location=device)
        model = torch.jit.optimize_for_inference(model)
        print(f"Loaded TorchScript model: {path}")
    except Exception as e:
        print(f"TorchScript load failed ({e}); trying checkpoint/state_dict...")
        model = DDSPGuitarToPiano()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        model.to(device)
    model.eval()
    return model


def warmup(model, device: torch.device, n_iters: int = 20) -> None:
    dummy = torch.zeros(1, FRAME_SIZE, device=device)
    lats = []
    with torch.inference_mode():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            if hasattr(model, "infer_frame"):
                _ = model.infer_frame(dummy)
            else:
                out = model(dummy)
                _ = out[0] if isinstance(out, tuple) else out
            if device.type == "cuda":
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)
    tail = lats[n_iters // 2:]
    print(f"Warmup avg inference: {np.mean(tail):.2f} ms | p95: {np.percentile(tail, 95):.2f} ms")


def prepare_audio_file(path: str) -> np.ndarray:
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} -> {SAMPLE_RATE} Hz")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = audio.squeeze(0).float()
    peak = audio.abs().max()
    if peak > 1.0:
        audio = audio / (peak + 1e-8)
    return audio.numpy().astype(np.float32)


class OverlapAddEngine:
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.input_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.output_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.norm_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.window = np.hanning(FRAME_SIZE).astype(np.float32)
        self.window = np.maximum(self.window, 1e-4)
        self.buf = torch.zeros(1, FRAME_SIZE, device=device)

    def reset(self) -> None:
        self.input_ring.fill(0.0)
        self.output_ring.fill(0.0)
        self.norm_ring.fill(0.0)
        if hasattr(self.model, "reset_phase"):
            self.model.reset_phase()

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        self.buf[0].copy_(torch.from_numpy(frame).to(self.device), non_blocking=True)
        with torch.inference_mode():
            if hasattr(self.model, "infer_frame"):
                pred = self.model.infer_frame(self.buf)
            else:
                out = self.model(self.buf)
                pred = out[0] if isinstance(out, tuple) else out
        return pred[0].detach().cpu().numpy().astype(np.float32)

    def process_hop(self, in_hop: np.ndarray) -> np.ndarray:
        if in_hop.shape[0] != HOP_SIZE:
            raise ValueError(f"Expected {HOP_SIZE} samples, got {in_hop.shape[0]}")

        self.input_ring[:-HOP_SIZE] = self.input_ring[HOP_SIZE:]
        self.input_ring[-HOP_SIZE:] = in_hop

        pred_frame = self._infer(self.input_ring.copy())
        self.output_ring += pred_frame * self.window
        self.norm_ring += self.window

        out = self.output_ring[:HOP_SIZE] / np.maximum(self.norm_ring[:HOP_SIZE], 1e-6)
        out = out.copy()

        self.output_ring[:-HOP_SIZE] = self.output_ring[HOP_SIZE:]
        self.output_ring[-HOP_SIZE:] = 0.0
        self.norm_ring[:-HOP_SIZE] = self.norm_ring[HOP_SIZE:]
        self.norm_ring[-HOP_SIZE:] = 0.0
        return out


def run_wav(args) -> None:
    device = get_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_model(args.model, device)
    warmup(model, device)

    audio = prepare_audio_file(args.input)
    orig_len = len(audio)
    pad = (HOP_SIZE - (orig_len % HOP_SIZE)) % HOP_SIZE
    if pad:
        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

    engine = OverlapAddEngine(model, device)
    engine.reset()

    output = np.zeros_like(audio) if args.output else None
    stream = None
    if args.play:
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,
            device=args.output_device,
            channels=1,
            dtype=DTYPE,
            latency=args.latency,
        )
        stream.start()

    lats = []
    n_hops = len(audio) // HOP_SIZE
    print(f"Processing {n_hops:,} hops | frame={FRAME_SIZE} | hop={HOP_SIZE}")
    try:
        for i in tqdm(range(n_hops), unit="hop", ncols=72):
            s = i * HOP_SIZE
            e = s + HOP_SIZE
            in_hop = audio[s:e]
            t0 = time.perf_counter()
            out_hop = engine.process_hop(in_hop)
            if device.type == "cuda" and args.sync_timing:
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)

            mixed = args.wet * out_hop + (1.0 - args.wet) * in_hop
            mixed = np.clip(mixed * args.volume, -1.0, 1.0).astype(np.float32)

            if output is not None:
                output[s:e] = mixed
            if stream is not None:
                stream.write(mixed.reshape(-1, 1))
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

    if output is not None:
        output = output[:orig_len]
        torchaudio.save(args.output, torch.from_numpy(output).unsqueeze(0), SAMPLE_RATE)
        print(f"Saved: {args.output}")

    lats_np = np.asarray(lats, dtype=np.float32)
    if len(lats_np):
        print(f"Hop latency avg={lats_np.mean():.2f} ms | p95={np.percentile(lats_np, 95):.2f} ms | max={lats_np.max():.2f} ms")


def run_live(args) -> None:
    device = get_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = load_model(args.model, device)
    warmup(model, device)
    engine = OverlapAddEngine(model, device)
    engine.reset()
    lats = []

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        t0 = time.perf_counter()
        in_hop = indata[:, 0].astype(np.float32)
        out_hop = engine.process_hop(in_hop)
        mixed = args.wet * out_hop + (1.0 - args.wet) * in_hop
        outdata[:, 0] = np.clip(mixed * args.volume, -1.0, 1.0)
        if len(lats) < 1000:
            lats.append((time.perf_counter() - t0) * 1000.0)

    print(f"Live mode | SR={SAMPLE_RATE} | frame={FRAME_SIZE} | hop={HOP_SIZE}")
    print("Press Ctrl+C to stop.")
    try:
        with sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=HOP_SIZE,
            device=(args.input_device, args.output_device),
            channels=1,
            dtype=DTYPE,
            latency=args.latency,
            callback=callback,
        ):
            while True:
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    if lats:
        arr = np.asarray(lats, dtype=np.float32)
        print(f"Live callback avg={arr.mean():.2f} ms | p95={np.percentile(arr,95):.2f} ms | max={arr.max():.2f} ms")


def main() -> None:
    p = argparse.ArgumentParser(description="Jetson Orin Guitar->Piano realtime/playback runner")
    p.add_argument("--model", required=True)
    p.add_argument("--input", default=None, help="Input WAV for file playback mode")
    p.add_argument("--output", default=None, help="Optional output WAV")
    p.add_argument("--play", action="store_true", help="Play processed WAV while processing")
    p.add_argument("--live", action="store_true", help="Use live audio input instead of WAV input")
    p.add_argument("--wet", type=float, default=1.0)
    p.add_argument("--volume", type=float, default=0.9)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--latency", default=DEFAULT_LATENCY)
    p.add_argument("--input_device", default=None)
    p.add_argument("--output_device", default=None)
    p.add_argument("--sync_timing", action="store_true", help="More accurate CUDA timing, slightly slower")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.live:
        run_live(args)
        return

    if args.input is None:
        raise SystemExit("Use --input guitar.wav for WAV mode, or --live for live input.")
    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")
    if args.output is None and not args.play:
        stem = Path(args.input).stem
        args.output = str(Path(args.input).with_name(f"{stem}_piano_jetson.wav"))

    run_wav(args)


if __name__ == "__main__":
    main()
