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

from model import DDSPGuitarToPiano, OverlapAddRenderer, SAMPLE_RATE, FRAME_SIZE, N_MELS


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
def checkpoint_state_dict(ckpt):
    return ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    config = {}
    first_weight = state_dict.get('decoder.net.0.weight')
    if first_weight is not None:
        config['hidden_size'] = int(first_weight.shape[0])
        decoder_input_size = int(first_weight.shape[1])
        z_latent_size = decoder_input_size - (2 + N_MELS)
        config['use_z'] = z_latent_size > 0
        config['z_latent_size'] = max(0, z_latent_size)

    harm_weight = state_dict.get('decoder.head_harmonic_amps.weight')
    if harm_weight is not None:
        config['n_harmonics'] = int(harm_weight.shape[0])

    return config


def get_device(preference: str) -> torch.device:
    if preference == 'auto':
        if torch.cuda.is_available():         return torch.device('cuda')
        if torch.backends.mps.is_available(): return torch.device('mps')
        return torch.device('cpu')
    return torch.device(preference)


def load_model(path: str, params: dict[str, int], device: torch.device):
    try:
        model = torch.jit.load(path, map_location='cpu')
        print(f"Loaded TorchScript model from {path}")
    except Exception:
        print(f"Loading state dict from {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint_state_dict(ckpt)
        inferred = infer_model_config(state_dict)
        hidden_size = inferred.get('hidden_size', params['hidden_size'])
        n_harmonics = inferred.get('n_harmonics', params['n_harmonics'])
        use_z = inferred.get('use_z', params.get('use_z', True))
        z_latent_size = inferred.get('z_latent_size', params.get('z_latent_size', 64))
        if inferred:
            print(
                "Inferred checkpoint config: "
                f"hidden_size={hidden_size}, n_harmonics={n_harmonics}, "
                f"use_z={use_z}, z_latent_size={z_latent_size}"
            )
        model = DDSPGuitarToPiano(
                    hidden_size=hidden_size,
                    n_harmonics=n_harmonics,
                    context_size=params['context_size'], 
                    hop_size=params['hop_size'],
                    use_z=use_z,
                    z_latent_size=z_latent_size,
                )
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def warmup(model, device, context_size, n_iters=20):
    print(f"Warming up ({n_iters} iters)...", end='', flush=True)
    dummy = torch.randn(1, context_size, device=device)
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


def _push_context(context_buf, frame_np, hop_size):
    """Append one hop of audio to the rolling model context."""
    frame = torch.from_numpy(frame_np).to(context_buf.device)
    context_buf[:, :-hop_size] = context_buf[:, hop_size:].clone()
    context_buf[:, -hop_size:] = frame


def _infer(model, context_buf, frame_np, hop_size):
    """Push one hop into the rolling context and run one forward pass."""
    _push_context(context_buf, frame_np, hop_size)
    with torch.no_grad():
        pred = model.infer_frame(context_buf) if hasattr(model, 'infer_frame') else model(context_buf)[0]
    return pred[0].cpu().numpy()


def _can_overlap_add(model) -> bool:
    return hasattr(model, 'predict_params') and hasattr(model, 'render_params')


# ─────────────────────────────────────────────
#  WAV FILE MODE
# ─────────────────────────────────────────────
def process_wav(
        model_path: str,
        model_params: dict[str, int],
        input_path: str, 
        output_path: str, 
        wet: float, 
        volume: float,
        device_str: str = 'auto',
        overlap_add: bool = True,
    ):
    
    device = get_device(device_str)
    print(f"Device : {device}")
    model  = load_model(model_path, model_params, device)
    context_size = int(model_params['context_size'])
    hop_size = int(model_params['hop_size'])
    warmup(model, device, context_size)

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
    pad = (hop_size - orig_len % hop_size) % hop_size
    if pad:
        audio_np = np.concatenate([audio_np, np.zeros(pad, dtype=np.float32)])
    n_frames  = len(audio_np) // hop_size
    output_np = np.zeros_like(audio_np)
    context_buf = torch.zeros(1, context_size, device=device)
    ola_renderer = (
        OverlapAddRenderer(model, context_size, hop_size, device)
        if overlap_add and _can_overlap_add(model) else None
    )
    lats      = []

    if ola_renderer is not None:
        ola_renderer.reset()
    elif hasattr(model, 'reset_phase'):
        model.reset_phase()

    print(
        f"Processing {n_frames:,} frames  "
        f"(context={context_size}, hop={hop_size}, wet={wet:.0%}, "
        f"vol={volume:.1f}x, overlap_add={ola_renderer is not None})..."
    )
    for i in tqdm(range(n_frames), unit='frame', ncols=72):
        s, e  = i * hop_size, (i + 1) * hop_size
        frame = audio_np[s:e].copy()

        t0              = time.perf_counter()
        pred            = (
            ola_renderer.process_frame(frame).cpu().numpy()
            if ola_renderer is not None
            else _infer(model, context_buf, frame, hop_size)
        )
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
    def __init__(
        self,
        model_path: str,
        model_params: dict[str, int],
        device_str: str = 'auto',
        overlap_add: bool = True,
    ):
        self.device   = get_device(device_str)
        print(f"Inference device: {self.device}")
        self.model    = load_model(model_path, model_params, self.device)
        self.context_size = int(model_params['context_size'])
        self.hop_size = int(model_params['hop_size'])
        self.volume   = 1.0
        self.wet_mix  = 1.0
        self.running  = False
        self._lats    = []
        self._frames  = 0
        self._context_buf = torch.zeros(1, self.context_size, device=self.device)
        self._ola_renderer = (
            OverlapAddRenderer(self.model, self.context_size, self.hop_size, self.device)
            if overlap_add and _can_overlap_add(self.model) else None
        )
        print(
            f"Context: {self.context_size} samples | "
            f"Hop: {self.hop_size} samples ({1000*self.hop_size/SAMPLE_RATE:.1f}ms) | "
            f"Overlap-add: {self._ola_renderer is not None}"
        )

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        t0 = time.perf_counter()

        if self._ola_renderer is not None:
            pred_np = self._ola_renderer.process_frame(indata[:, 0].copy()).cpu().numpy()
        else:
            _push_context(self._context_buf, indata[:, 0].copy(), self.hop_size)
            with torch.no_grad():
                pred = self.model.infer_frame(self._context_buf) if hasattr(self.model, 'infer_frame') \
                       else self.model(self._context_buf)[0]
            pred_np = pred[0].cpu().numpy()

        mixed = self.wet_mix * pred_np + (1.0 - self.wet_mix) * indata[:, 0]
        outdata[:, 0] = np.clip(mixed * self.volume, -1.0, 1.0)

        self._frames += 1
        if len(self._lats) < 200:
            self._lats.append((time.perf_counter() - t0) * 1000)

    def run(self):
        warmup(self.model, self.device, self.context_size)
        print("\n─── LIVE MODE ───────────────────────────────")
        print(
            f"SR: {SAMPLE_RATE} Hz | Context: {self.context_size} samples | "
            f"Buffer: {self.hop_size} samples ({1000*self.hop_size/SAMPLE_RATE:.1f}ms)"
        )
        print(f"Wet: {self.wet_mix:.0%} piano | Volume: {self.volume:.1f}x")
        print("Controls: [q]uit  [r]eset phase  [+/-] volume  [m]ix toggle")
        print("─────────────────────────────────────────────\n")
        self.running = True
        try:
            with sd.Stream(samplerate=SAMPLE_RATE, blocksize=self.hop_size,
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
            if self._ola_renderer is not None:
                self._ola_renderer.reset()
                print("  Phase and OLA buffers reset.")
            else:
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
    p.add_argument('--hidden_size', type=int,   default=512)
    p.add_argument('--n_harmonics', type=int,   default=64)
    p.add_argument('--context_size', type=int, default=2048,
                   help='Number of guitar samples the encoder sees for each prediction')
    p.add_argument('--hop_size', type=int, default=FRAME_SIZE,
                   help='Number of target/output samples predicted per step')
    p.add_argument('--input',        default=None,       help='[WAV mode] Input guitar WAV')
    p.add_argument('--output',       default=None,       help='[WAV mode] Output path (default: <stem>_piano.wav)')
    p.add_argument('--wet',          type=float, default=1.0,   help='Wet mix 0.0–1.0 (default: 1.0)')
    p.add_argument('--volume',       type=float, default=1.0,   help='Output volume multiplier')
    p.add_argument('--device',       default='auto',     help='auto | cuda | mps | cpu')
    p.add_argument('--list-devices', action='store_true', help='List audio devices and exit')
    p.add_argument(
        '--overlap_add',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use inference-time Hann overlap-add smoothing when supported',
    )
    args = p.parse_args()

    model_params = {
        'hidden_size': args.hidden_size, 
        'n_harmonics': args.n_harmonics,
        'context_size': args.context_size, 
        'hop_size': args.hop_size
    }

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
        process_wav(
            args.model,
            model_params,
            args.input,
            args.output,
            args.wet,
            args.volume,
            args.device,
            overlap_add=args.overlap_add,
        )
    else:
        # ── Live mic mode ──────────────────────
        pipe = RealTimePipeline(args.model, model_params, args.device, overlap_add=args.overlap_add)
        pipe.wet_mix = args.wet
        pipe.volume  = args.volume
        pipe.run()


if __name__ == '__main__':
    main()
