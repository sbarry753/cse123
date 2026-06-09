"""
realtime_onnx_asio.py — Live guitar -> piano via exported ONNX, ASIO drivers (Windows).

Mirrors the OverlapAddEngine from realtime.py but runs the ONNX graph through
ONNX Runtime. Audio I/O is forced onto the ASIO host API for minimum driver
latency.

Setup
-----
pip install onnxruntime sounddevice numpy
   (or onnxruntime-gpu instead of onnxruntime if you have CUDA + the matching
    cuDNN/CUDA versions installed and want GPU inference)

Install an ASIO driver:
    - Your interface's native ASIO driver (best — Focusrite, RME, MOTU, etc.)
    - ASIO4ALL (universal wrapper, works with any WDM device)

Examples
--------
# List ASIO devices and exit
python realtime_onnx_asio.py --model model.onnx --list-devices

# Run with the first ASIO device found
python realtime_onnx_asio.py --model model.onnx

# Pick a specific interface by partial name match, route in ch 0 -> out ch 0
python realtime_onnx_asio.py --model model.onnx --device "Focusrite" --in-chan 0 --out-chan 0

# Force CPU even if onnxruntime-gpu is installed
python realtime_onnx_asio.py --model model.onnx --cpu
"""

import argparse
import sys
import time
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import sounddevice as sd


# Must match what the model was exported with.
SAMPLE_RATE = 48000
FRAME_SIZE = 1024
HOP_SIZE = 256


# ------------------------------------------------------------
# ASIO device discovery
# ------------------------------------------------------------
def _asio_hostapi_index() -> Optional[int]:
    for i, ha in enumerate(sd.query_hostapis()):
        if "ASIO" in ha["name"]:
            return i
    return None


def list_asio_devices() -> None:
    asio_idx = _asio_hostapi_index()
    if asio_idx is None:
        print("No ASIO host API available.")
        print("Install your interface's ASIO driver, or ASIO4ALL for a generic wrapper.")
        return

    devices = sd.query_devices()
    print(f"\nASIO host API: '{sd.query_hostapis()[asio_idx]['name']}'")
    print(f"{'idx':>4}  {'in':>3}  {'out':>3}  {'sr':>6}  name")
    print("-" * 60)
    for i, d in enumerate(devices):
        if d["hostapi"] == asio_idx:
            print(
                f"{i:>4}  {d['max_input_channels']:>3}  "
                f"{d['max_output_channels']:>3}  "
                f"{int(d['default_samplerate']):>6}  {d['name']}"
            )
    print()


def find_asio_device(name_substr: Optional[str]) -> Optional[int]:
    asio_idx = _asio_hostapi_index()
    if asio_idx is None:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] != asio_idx:
            continue
        if name_substr is None or name_substr.lower() in d["name"].lower():
            return i
    return None


# ------------------------------------------------------------
# ONNX inference + overlap-add
# ------------------------------------------------------------
class OnnxOLAEngine:
    """
    Overlap-add engine driven by ONNX Runtime.

    Same shape contract as OverlapAddEngine in realtime.py:
        process_hop(in_hop[HOP_SIZE]) -> out_hop[HOP_SIZE]

    The model is queried once per hop with the latest FRAME_SIZE samples; the
    full predicted frame is OLA'd into the output ring and the leading hop is
    emitted.
    """

    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        if providers is None:
            avail = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            elif "DmlExecutionProvider" in avail:
                providers.append("DmlExecutionProvider")
            providers.append("CPUExecutionProvider")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Single-threaded inference avoids contention with the audio callback
        # thread, which matters far more than peak throughput at this model size.
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.enable_mem_pattern = True

        self.sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
        print(f"ONNX Runtime providers (in order): {self.sess.get_providers()}")

        in_meta = self.sess.get_inputs()[0]
        out_meta = self.sess.get_outputs()[0]
        self.in_name = in_meta.name
        self.out_name = out_meta.name
        print(f"Input  : {in_meta.name} {in_meta.shape} {in_meta.type}")
        print(f"Output : {out_meta.name} {out_meta.shape} {out_meta.type}")

        # Sanity check: model must take (1, FRAME_SIZE) float32 audio.
        if list(in_meta.shape) not in ([1, FRAME_SIZE], ["batch", FRAME_SIZE]):
            print(
                f"  WARNING: model input shape {in_meta.shape} doesn't match "
                f"(1, {FRAME_SIZE}). Output may be wrong."
            )

        self.input_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.output_ring = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.in_tensor = np.zeros((1, FRAME_SIZE), dtype=np.float32)

    def reset(self) -> None:
        self.input_ring.fill(0.0)
        self.output_ring.fill(0.0)

    def process_hop(self, in_hop: np.ndarray) -> np.ndarray:
        # Slide new samples into the input ring.
        self.input_ring[:-HOP_SIZE] = self.input_ring[HOP_SIZE:]
        self.input_ring[-HOP_SIZE:] = in_hop

        # Run the model on the full FRAME_SIZE window.
        np.copyto(self.in_tensor[0], self.input_ring)
        pred = self.sess.run([self.out_name], {self.in_name: self.in_tensor})[0][0]

        # Overlap-add the prediction.
        self.output_ring += pred
        out_hop = self.output_ring[:HOP_SIZE].copy()
        self.output_ring[:-HOP_SIZE] = self.output_ring[HOP_SIZE:]
        self.output_ring[-HOP_SIZE:] = 0.0
        return out_hop


def warmup(engine: OnnxOLAEngine, n: int = 64) -> None:
    print(f"Warming up ({n} iters)...", end="", flush=True)
    dummy = (np.random.randn(HOP_SIZE) * 0.05).astype(np.float32)
    lats = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = engine.process_hop(dummy)
        lats.append((time.perf_counter() - t0) * 1000.0)
    avg = float(np.mean(lats[len(lats) // 2:]))
    print(f" done. avg infer: {avg:.2f} ms")
    engine.reset()


# ------------------------------------------------------------
# Live audio pipeline
# ------------------------------------------------------------
class LivePipeline:
    def __init__(
        self,
        model_path: str,
        device_substr: Optional[str],
        in_chan: int,
        out_chan: int,
        providers: Optional[List[str]] = None,
    ):
        self.engine = OnnxOLAEngine(model_path, providers=providers)
        warmup(self.engine)

        self.device_idx = find_asio_device(device_substr)
        if self.device_idx is None:
            if _asio_hostapi_index() is None:
                raise RuntimeError("No ASIO host API on this system. Install an ASIO driver.")
            raise RuntimeError(
                f"No ASIO device matching '{device_substr}'. "
                f"Run with --list-devices to see options."
            )

        d = sd.query_devices(self.device_idx)
        print(f"\nASIO device [{self.device_idx}]: {d['name']}")
        print(f"  channels: {d['max_input_channels']} in / {d['max_output_channels']} out")
        print(f"  default sr: {int(d['default_samplerate'])} Hz")

        if in_chan >= d["max_input_channels"]:
            raise ValueError(f"in-chan {in_chan} >= device max input {d['max_input_channels']}")
        if out_chan >= d["max_output_channels"]:
            raise ValueError(f"out-chan {out_chan} >= device max output {d['max_output_channels']}")

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.wet = 1.0
        self.volume = 1.0
        self.running = False

        self._lats: List[float] = []
        self._frames = 0
        self._xruns = 0

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            self._xruns += 1
            print(status, file=sys.stderr)

        t0 = time.perf_counter()
        in_hop = indata[:, 0]
        out_hop = self.engine.process_hop(in_hop)
        mixed = self.wet * out_hop + (1.0 - self.wet) * in_hop
        np.clip(mixed * self.volume, -1.0, 1.0, out=outdata[:, 0])

        self._frames += 1
        if len(self._lats) < 4000:
            self._lats.append((time.perf_counter() - t0) * 1000.0)

    def run(self) -> None:
        # ASIO channel selection — both directions can pick a single channel
        # off a multi-channel interface.
        in_settings = sd.AsioSettings(channel_selectors=[self.in_chan])
        out_settings = sd.AsioSettings(channel_selectors=[self.out_chan])

        block_ms = 1000.0 * HOP_SIZE / SAMPLE_RATE
        win_ms = 1000.0 * FRAME_SIZE / SAMPLE_RATE
        print(f"\nSR     : {SAMPLE_RATE} Hz")
        print(f"Block  : {HOP_SIZE} samples ({block_ms:.2f} ms)")
        print(f"Window : {FRAME_SIZE} samples ({win_ms:.2f} ms)")
        print(f"Wet={self.wet:.0%}  Vol={self.volume:.1f}x  in_ch={self.in_chan}  out_ch={self.out_chan}")
        print("Controls: q=quit  +/-=volume  m=toggle mix  r=reset rings\n")
        print("NOTE: set your ASIO buffer to <= 256 samples in the driver's control panel")
        print("      for the lowest end-to-end latency.\n")

        self.running = True
        try:
            with sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=HOP_SIZE,
                device=(self.device_idx, self.device_idx),
                channels=(1, 1),
                dtype="float32",
                latency="low",
                callback=self._audio_callback,
                extra_settings=(in_settings, out_settings),
            ):
                print("Streaming. Type a command + Enter:\n")
                while self.running:
                    try:
                        self._handle_cmd(input().strip().lower())
                    except EOFError:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self._print_stats()

    def _handle_cmd(self, cmd: str) -> None:
        if cmd == "q":
            self.running = False
        elif cmd == "+":
            self.volume = min(4.0, self.volume + 0.1)
            print(f"  vol = {self.volume:.1f}")
        elif cmd == "-":
            self.volume = max(0.0, self.volume - 0.1)
            print(f"  vol = {self.volume:.1f}")
        elif cmd == "m":
            self.wet = 0.0 if self.wet > 0.5 else 1.0
            print(f"  mix = {'piano' if self.wet > 0.5 else 'dry'}")
        elif cmd == "r":
            self.engine.reset()
            print("  rings reset")
        elif cmd:
            print(f"  unknown: {cmd!r}")

    def _print_stats(self) -> None:
        if not self._lats:
            return
        lats = np.array(self._lats, dtype=np.float32)
        budget_ms = 1000.0 * HOP_SIZE / SAMPLE_RATE
        print("\n--- Stats ----------------------------------------")
        print(f"frames     : {self._frames:,}")
        print(f"xruns      : {self._xruns}")
        print(
            f"infer (ms) : avg={lats.mean():.2f}  "
            f"p50={np.percentile(lats,50):.2f}  "
            f"p95={np.percentile(lats,95):.2f}  "
            f"max={lats.max():.2f}"
        )
        print(f"budget/blk : {budget_ms:.2f} ms  ({100*lats.mean()/budget_ms:.0f}% used)")
        print("--------------------------------------------------")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ONNX guitar->piano on ASIO (Windows)")
    ap.add_argument("--model", default="./model.onnx", help="Path to exported ONNX file")
    ap.add_argument("--list-devices", action="store_true", help="List ASIO devices and exit")
    ap.add_argument("--device", default=None,
                    help="Substring of ASIO device name (case-insensitive). "
                         "Picks first ASIO device if omitted.")
    ap.add_argument("--in-chan", type=int, default=0, help="ASIO input channel index")
    ap.add_argument("--out-chan", type=int, default=0, help="ASIO output channel index")
    ap.add_argument("--wet", type=float, default=1.0, help="Wet mix 0-1")
    ap.add_argument("--volume", type=float, default=1.0, help="Output gain")
    ap.add_argument("--cpu", action="store_true", help="Force CPU EP")
    args = ap.parse_args()

    if args.list_devices:
        list_asio_devices()
        return

    providers = ["CPUExecutionProvider"] if args.cpu else None
    pipe = LivePipeline(
        model_path=args.model,
        device_substr=args.device,
        in_chan=args.in_chan,
        out_chan=args.out_chan,
        providers=providers,
    )
    pipe.wet = args.wet
    pipe.volume = args.volume
    pipe.run()


if __name__ == "__main__":
    main()