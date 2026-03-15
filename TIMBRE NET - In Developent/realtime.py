"""
realtime.py — Live Real-Time Guitar → Piano  +  WAV File Mode

─── Live mic mode (default) ────────────────────────────────────────────────
  python realtime.py --model ./checkpoints/model_scripted.pt

─── WAV file mode ───────────────────────────────────────────────────────────
  python realtime.py --model  ./checkpoints/model_scripted.pt \
                     --input  ./guitar.wav \
                     --output ./piano_out.wav

  Processes the input WAV through the exact same frame pipeline as live mode
  and writes the result to --output (default: <input stem>_piano.wav).
  Prints per-frame latency stats when done so you can verify the 12ms budget.

─── Live mode controls (keyboard) ──────────────────────────────────────────
  q     — quit
  r     — reset phase accumulator (clears phase continuity artefacts)
  +/-   — adjust output volume
  m     — toggle dry/wet mix

Latency budget breakdown at 44100 Hz, 256-sample frame:
  Audio buffer fill:        ~5.8ms
  Feature extraction:       ~2.0ms
  MLP inference:            ~0.5ms
  Additive synthesis:       ~2.5ms
  Overlap-add output:       ~0.5ms
  ─────────────────────────────────
  Total algorithmic:        ~11.3ms  ✓ (under 12ms)

Note: Audio driver round-trip (ASIO/CoreAudio) adds 2–8ms separately.
Use ASIO on Windows, CoreAudio on macOS for minimum driver latency.
"""

import argparse
import time
import sys
import os
import numpy as np
import torch
import torchaudio
import sounddevice as sd
from pathlib import Path
from tqdm import tqdm

from model import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BLOCKSIZE = FRAME_SIZE   # 256 samples = 5.8ms
DEVICE_IN  = None        # None = system default; set to device name/index if needed
DEVICE_OUT = None
DTYPE      = 'float32'
LATENCY    = 'low'       # sounddevice latency hint


# ─────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────
def get_device(preference: str) -> torch.device:
    if preference == 'auto':
        if torch.cuda.is_available():         return torch.device('cuda')
        if torch.backends.mps.is_available(): return torch.device('mps')
        return torch.device('cpu')
    return torch.device(preference)


def load_model(path: str, device: torch.device):
    try:
        model = torch.jit.load(path, map_location='cpu')
        print(f"Loaded TorchScript model from {path}")
    except Exception:
        print(f"Loading state dict from {path}")
        model = DDSPGuitarToPiano()
        ckpt  = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.to(device)
    model.eval()
    return model


def warmup(model, device, n_iters=20):
    print(f"Warming up ({n_iters} iters)...", end='', flush=True)
    dummy = torch.randn(1, BLOCKSIZE, device=device)
    lats = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model.infer_frame(dummy) if hasattr(model, 'infer_frame') else model(dummy)
            lats.append((time.perf_counter() - t0) * 1000)
    avg = float(np.mean(lats[5:]))
    status = "✓" if avg <= 8.0 else "⚠"
    print(f" done.  avg: {avg:.2f}ms  {status}")
    return avg


def _infer(model, buf, frame_np):
    """Copy frame into pre-allocated tensor and run one forward pass."""
    buf[0].copy_(torch.from_numpy(frame_np).to(buf.device), non_blocking=True)
    with torch.no_grad():
        pred = model.infer_frame(buf) if hasattr(model, 'infer_frame') else model(buf)[0]
    return pred[0].cpu().numpy()


# ─────────────────────────────────────────────
#  WAV FILE MODE
# ─────────────────────────────────────────────
def process_wav(model_path, input_path, output_path, wet, volume, device_str):
    device = get_device(device_str)
    print(f"Device : {device}")
    model  = load_model(model_path, device)
    warmup(model, device)

    # Load + prep
    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)         # stereo → mono
    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} → {SAMPLE_RATE} Hz...")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

    audio_np  = audio.squeeze(0).numpy().astype(np.float32)
    orig_len  = len(audio_np)
    duration  = orig_len / SAMPLE_RATE
    print(f"Input  : {input_path}  ({duration:.2f}s, {orig_len:,} samples)")

    # Pad to frame boundary
    pad = (BLOCKSIZE - orig_len % BLOCKSIZE) % BLOCKSIZE
    if pad:
        audio_np = np.concatenate([audio_np, np.zeros(pad, dtype=np.float32)])
    n_frames  = len(audio_np) // BLOCKSIZE
    output_np = np.zeros_like(audio_np)
    buf       = torch.zeros(1, BLOCKSIZE, device=device)
    lats      = []

    if hasattr(model, 'reset_phase'):
        model.reset_phase()

    print(f"Processing {n_frames:,} frames  (wet={wet:.0%}, vol={volume:.1f}x)...")
    for i in tqdm(range(n_frames), unit='frame', ncols=72):
        s, e  = i * BLOCKSIZE, (i + 1) * BLOCKSIZE
        frame = audio_np[s:e].copy()

        t0              = time.perf_counter()
        pred            = _infer(model, buf, frame)
        lats.append((time.perf_counter() - t0) * 1000)

        mixed           = wet * pred + (1.0 - wet) * frame
        output_np[s:e]  = np.clip(mixed * volume, -1.0, 1.0)

    # Trim padding and save
    output_np = output_np[:orig_len]
    torchaudio.save(output_path, torch.from_numpy(output_np).unsqueeze(0), SAMPLE_RATE)

    # Stats
    lats = np.array(lats)
    over = (lats > 12.0).sum()
    print(f"\n─── Complete ─────────────────────────────")
    print(f"  Saved to    : {output_path}")
    print(f"  Duration    : {orig_len / SAMPLE_RATE:.2f}s  ({n_frames:,} frames)")
    print(f"  Latency avg : {lats.mean():.2f}ms  p95: {np.percentile(lats,95):.2f}ms  max: {lats.max():.2f}ms")
    if over:
        print(f"  ⚠ {over} frames exceeded 12ms budget ({100*over/n_frames:.1f}%)")
    else:
        print(f"  ✓ All frames within 12ms budget")
    print("──────────────────────────────────────────")


# ─────────────────────────────────────────────
#  LIVE MIC MODE
# ─────────────────────────────────────────────
class RealTimePipeline:
    def __init__(self, model_path: str, device_str: str = 'auto'):
        self.device   = get_device(device_str)
        print(f"Inference device: {self.device}")
        self.model    = load_model(model_path, self.device)
        self.volume   = 1.0
        self.wet_mix  = 1.0
        self.running  = False
        self._lats    = []
        self._frames  = 0
        self._buf     = torch.zeros(1, BLOCKSIZE, device=self.device)
        print(f"Frame: {BLOCKSIZE} samples ({1000*BLOCKSIZE/SAMPLE_RATE:.1f}ms)")

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        t0 = time.perf_counter()

        self._buf[0].copy_(
            torch.from_numpy(indata[:, 0]).to(self.device), non_blocking=True
        )
        with torch.no_grad():
            pred = self.model.infer_frame(self._buf) if hasattr(self.model, 'infer_frame') \
                   else self.model(self._buf)[0]

        mixed = self.wet_mix * pred[0].cpu().numpy() + (1.0 - self.wet_mix) * indata[:, 0]
        outdata[:, 0] = np.clip(mixed * self.volume, -1.0, 1.0)

        self._frames += 1
        if len(self._lats) < 200:
            self._lats.append((time.perf_counter() - t0) * 1000)

    def run(self):
        warmup(self.model, self.device)
        print("\n─── LIVE MODE ───────────────────────────────")
        print(f"SR: {SAMPLE_RATE} Hz | Buffer: {BLOCKSIZE} samples ({1000*BLOCKSIZE/SAMPLE_RATE:.1f}ms)")
        print(f"Wet: {self.wet_mix:.0%} piano | Volume: {self.volume:.1f}x")
        print("Controls: [q]uit  [r]eset phase  [+/-] volume  [m]ix toggle")
        print("─────────────────────────────────────────────\n")
        self.running = True
        try:
            with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                           device=(DEVICE_IN, DEVICE_OUT), channels=1,
                           dtype=DTYPE, latency=LATENCY, callback=self.audio_callback):
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
                print(f"\n─── Stats ────────────────────────────────")
                print(f"  Frames : {self._frames:,}")
                print(f"  avg: {lats.mean():.2f}ms  p95: {np.percentile(lats,95):.2f}ms  max: {lats.max():.2f}ms")
                print("──────────────────────────────────────────")

    def _handle(self, cmd):
        if   cmd == 'q':  self.running = False
        elif cmd == 'r':
            if hasattr(self.model, 'reset_phase'): self.model.reset_phase()
            print("  Phase reset.")
        elif cmd == '+':
            self.volume = min(4.0, self.volume + 0.1); print(f"  Volume: {self.volume:.1f}x")
        elif cmd == '-':
            self.volume = max(0.0, self.volume - 0.1); print(f"  Volume: {self.volume:.1f}x")
        elif cmd == 'm':
            self.wet_mix = 0.0 if self.wet_mix > 0.5 else 1.0
            print(f"  Mix: {'piano' if self.wet_mix > 0.5 else 'dry guitar'}")
        elif cmd:
            print(f"  Unknown: '{cmd}'")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='DDSP Guitar→Piano | live mic or WAV file')
    p.add_argument('--model',        required=True,      help='Model checkpoint (.pt)')
    p.add_argument('--input',        default=None,       help='[WAV mode] Input guitar WAV')
    p.add_argument('--output',       default=None,       help='[WAV mode] Output path (default: <stem>_piano.wav)')
    p.add_argument('--wet',          type=float, default=1.0,   help='Wet mix 0.0–1.0 (default: 1.0)')
    p.add_argument('--volume',       type=float, default=1.0,   help='Output volume multiplier')
    p.add_argument('--device',       default='auto',     help='auto | cuda | mps | cpu')
    p.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.input:
        # ── WAV file mode ──────────────────────
        if not os.path.isfile(args.input):
            print(f"Error: file not found: {args.input}"); sys.exit(1)
        if args.output is None:
            stem = Path(args.input).stem
            args.output = str(Path(args.input).parent / f"{stem}_piano.wav")
        process_wav(args.model, args.input, args.output, args.wet, args.volume, args.device)
    else:
        # ── Live mic mode ──────────────────────
        pipe = RealTimePipeline(args.model, args.device)
        pipe.wet_mix = args.wet
        pipe.volume  = args.volume
        pipe.run()


if __name__ == '__main__':
    main()