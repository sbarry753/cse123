"""
realtime_onnx_orin.py - Live guitar -> piano via ONNX Runtime or TensorRT on Jetson Orin.

This version is intended for Linux audio backends such as JACK/ALSA through
PortAudio/sounddevice. The audio callback never runs ONNX inference; it only
moves fixed-size hops through queues. A worker thread owns inference calls so
GPU synchronization does not block the realtime audio callback.

Examples
--------
# List devices, including host API names such as JACK or ALSA
python realtime_onnx_orin.py --list-devices

# Run through JACK devices selected by index or partial device name
python realtime_onnx_orin.py --model model.onnx --host-api JACK \
    --input-device system:capture_1 --output-device system:playback_1

# Use a larger safety queue if the worker misses output
python realtime_onnx_orin.py --model model.onnx --host-api JACK --queue-hops 8 --latency 0.10

# Run a TensorRT engine built with trtexec
python realtime_onnx_orin.py --engine model_fp16.plan --host-api JACK

# Run a 2048-sample model/window with the fixed 256-sample realtime hop
python realtime_onnx_orin.py --engine model_fp16.plan --frame_size 2048 --host-api JACK

# WAV -> save processed output
python realtime_onnx_orin.py --engine model_fp16.plan --input guitar.wav --output piano.wav

# WAV -> play processed output
python realtime_onnx_orin.py --engine model_fp16.plan --input guitar.wav --play --host-api JACK

# Force CPU if the Jetson ONNX Runtime build does not expose CUDA/TensorRT
python realtime_onnx_orin.py --model model.onnx --provider cpu
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
import torch
import torchaudio

import onnxruntime as ort
import tensorrt as trt
from cuda import cudart


SAMPLE_RATE = 48000
FRAME_SIZE = 1024
HOP_SIZE = 256
DTYPE = "float32"

# ------------------------------------------------------------
# Device discovery
# ------------------------------------------------------------
def parse_device_arg(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def parse_latency(value):
    try:
        return float(value)
    except ValueError:
        return value


def hostapi_name(device) -> str:
    return sd.query_hostapis()[device["hostapi"]]["name"]


def list_devices(host_api: Optional[str] = None) -> None:
    host_filter = host_api.lower() if host_api else None
    print(f"{'idx':>4}  {'host':<12}  {'in':>3}  {'out':>3}  {'sr':>6}  name")
    print("-" * 80)
    for idx, dev in enumerate(sd.query_devices()):
        host = hostapi_name(dev)
        if host_filter and host_filter not in host.lower():
            continue
        print(
            f"{idx:>4}  {host:<12.12}  {dev['max_input_channels']:>3}  "
            f"{dev['max_output_channels']:>3}  {int(dev['default_samplerate']):>6}  "
            f"{dev['name']}"
        )


def find_device(device_arg, host_api: Optional[str], want_input: bool):
    parsed = parse_device_arg(device_arg)
    if isinstance(parsed, int):
        return parsed

    needle = parsed.lower() if parsed is not None else None
    host_filter = host_api.lower() if host_api else None
    channel_key = "max_input_channels" if want_input else "max_output_channels"
    for idx, dev in enumerate(sd.query_devices()):
        if dev[channel_key] <= 0:
            continue
        host = hostapi_name(dev)
        if host_filter and host_filter not in host.lower():
            continue
        if needle is None or needle in dev["name"].lower():
            return idx

    kind = "input" if want_input else "output"
    if host_api:
        raise RuntimeError(f"No {kind} device found for host API {host_api!r}")
    raise RuntimeError(f"No {kind} device matching {device_arg!r}")


def validate_device(idx, want_input: bool) -> None:
    if idx is None:
        return
    dev = sd.query_devices(idx)
    key = "max_input_channels" if want_input else "max_output_channels"
    if dev[key] <= 0:
        kind = "input" if want_input else "output"
        raise RuntimeError(f"Device {idx} ({dev['name']}) has no {kind} channels")


# ------------------------------------------------------------
# ONNX inference + overlap-add
# ------------------------------------------------------------
def choose_providers(provider: str) -> List[str]:
    available = ort.get_available_providers()
    if provider == "cpu":
        return ["CPUExecutionProvider"]

    preferred = {
        "auto": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }[provider]
    selected = [p for p in preferred if p in available]
    if not selected:
        raise RuntimeError(
            f"No requested ONNX Runtime providers available. requested={preferred}, available={available}"
        )
    return selected


class OnnxOLAEngine:
    """
    Overlap-add engine driven by ONNX Runtime.

    process_hop(in_hop[hop_size]) -> out_hop[hop_size]
    """

    def __init__(
        self,
        model_path: str,
        providers: List[str],
        frame_size: int = FRAME_SIZE,
        hop_size: int = HOP_SIZE,
    ):
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        if self.frame_size <= 0:
            raise ValueError(f"frame_size must be positive, got {self.frame_size}")
        if self.hop_size <= 0:
            raise ValueError(f"hop_size must be positive, got {self.hop_size}")
        if self.frame_size < self.hop_size:
            raise ValueError(
                f"frame_size must be >= hop_size, got frame_size={self.frame_size}, "
                f"hop_size={self.hop_size}"
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.enable_mem_pattern = True

        self.sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
        print(f"ONNX Runtime providers: {self.sess.get_providers()}")

        in_meta = self.sess.get_inputs()[0]
        out_meta = self.sess.get_outputs()[0]
        self.in_name = in_meta.name
        self.out_name = out_meta.name
        print(f"Input : {in_meta.name} {in_meta.shape} {in_meta.type}")
        print(f"Output: {out_meta.name} {out_meta.shape} {out_meta.type}")

        expected_shape = [1, self.frame_size]
        if list(in_meta.shape) != expected_shape:
            print(
                f"WARNING: model input shape {in_meta.shape} does not match "
                f"(1, {self.frame_size}). Output may be wrong.",
                file=sys.stderr,
            )

        self.input_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.output_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.norm_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.synthesis_window = np.hanning(self.frame_size).astype(np.float32)
        self.in_tensor = np.zeros((1, self.frame_size), dtype=np.float32)

    def reset(self) -> None:
        self.input_ring.fill(0.0)
        self.output_ring.fill(0.0)
        self.norm_ring.fill(0.0)

    def process_hop(self, in_hop: np.ndarray) -> np.ndarray:
        if len(in_hop) != self.hop_size:
            raise ValueError(f"Expected hop of length {self.hop_size}, got {len(in_hop)}")

        self.input_ring[:-self.hop_size] = self.input_ring[self.hop_size:]
        self.input_ring[-self.hop_size:] = in_hop

        np.copyto(self.in_tensor[0], self.input_ring)
        pred = self.sess.run([self.out_name], {self.in_name: self.in_tensor})[0][0]

        if len(pred) != self.frame_size:
            raise ValueError(f"Expected model output length {self.frame_size}, got {len(pred)}")

        self.output_ring += pred.astype(np.float32, copy=False) * self.synthesis_window
        self.norm_ring += self.synthesis_window

        denom = np.maximum(self.norm_ring[:self.hop_size], 1e-6)
        out_hop = (self.output_ring[:self.hop_size] / denom).astype(np.float32)

        self.output_ring[:-self.hop_size] = self.output_ring[self.hop_size:]
        self.output_ring[-self.hop_size:] = 0.0
        self.norm_ring[:-self.hop_size] = self.norm_ring[self.hop_size:]
        self.norm_ring[-self.hop_size:] = 0.0
        return out_hop


# ------------------------------------------------------------
# TensorRT inference + overlap-add
# ------------------------------------------------------------
def _check_cuda(result, label: str):
    err = result[0]
    code = getattr(err, "value", err)
    if code != 0:
        raise RuntimeError(f"{label} failed with CUDA error {err}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


class TrtOLAEngine:
    """
    Overlap-add engine driven directly by a serialized TensorRT engine.

    The TensorRT engine must have one input and one output with shape compatible
    with (1, frame_size). FP32 and FP16 TensorRT I/O tensors are supported.
    """

    def __init__(
        self,
        engine_path: str,
        frame_size: int = FRAME_SIZE,
        hop_size: int = HOP_SIZE,
    ):
        self.frame_size = int(frame_size)
        self.hop_size = int(hop_size)
        if self.frame_size <= 0:
            raise ValueError(f"frame_size must be positive, got {self.frame_size}")
        if self.hop_size <= 0:
            raise ValueError(f"hop_size must be positive, got {self.hop_size}")
        if self.frame_size < self.hop_size:
            raise ValueError(
                f"frame_size must be >= hop_size, got frame_size={self.frame_size}, "
                f"hop_size={self.hop_size}"
            )

        self.trt = trt
        self.cudart = cudart
        self.stream = None
        self.d_input = None
        self.d_output = None

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.in_name, self.out_name = self._find_io_tensors()

        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        if any(dim < 0 for dim in in_shape):
            self.context.set_input_shape(self.in_name, (1, self.frame_size))
            in_shape = tuple(self.context.get_tensor_shape(self.in_name))
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        if any(dim < 0 for dim in out_shape):
            out_shape = tuple(self.engine.get_tensor_shape(self.out_name))

        expected_shape = (1, self.frame_size)
        if in_shape != expected_shape:
            raise RuntimeError(f"TensorRT input shape {in_shape} does not match {expected_shape}")
        if out_shape != expected_shape:
            raise RuntimeError(f"TensorRT output shape {out_shape} does not match {expected_shape}")

        self.input_dtype = self._trt_dtype_to_np(self.engine.get_tensor_dtype(self.in_name))
        self.output_dtype = self._trt_dtype_to_np(self.engine.get_tensor_dtype(self.out_name))
        self.in_tensor = np.zeros(in_shape, dtype=self.input_dtype)
        self.out_tensor = np.zeros(out_shape, dtype=self.output_dtype)

        self.d_input = _check_cuda(cudart.cudaMalloc(self.in_tensor.nbytes), "cudaMalloc(input)")
        self.d_output = _check_cuda(cudart.cudaMalloc(self.out_tensor.nbytes), "cudaMalloc(output)")
        self.stream = _check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")

        if not self.context.set_tensor_address(self.in_name, int(self.d_input)):
            raise RuntimeError(f"Failed to bind TensorRT input tensor {self.in_name!r}")
        if not self.context.set_tensor_address(self.out_name, int(self.d_output)):
            raise RuntimeError(f"Failed to bind TensorRT output tensor {self.out_name!r}")

        print(f"TensorRT engine: {engine_path}")
        print(f"Input : {self.in_name} {in_shape} {self.input_dtype}")
        print(f"Output: {self.out_name} {out_shape} {self.output_dtype}")

        self.input_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.output_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.norm_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.synthesis_window = np.hanning(self.frame_size).astype(np.float32)

    def _find_io_tensors(self) -> Tuple[str, str]:
        trt = self.trt
        inputs = []
        outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"Expected one TensorRT input and one output, got {inputs=} {outputs=}")
        return inputs[0], outputs[0]

    def _trt_dtype_to_np(self, dtype):
        trt = self.trt
        if dtype == trt.float32:
            return np.float32
        if dtype == trt.float16:
            return np.float16
        raise RuntimeError(f"Unsupported TensorRT tensor dtype: {dtype}")

    def close(self) -> None:
        if self.stream is not None:
            _check_cuda(self.cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
            self.stream = None
        if self.d_input is not None:
            _check_cuda(self.cudart.cudaFree(self.d_input), "cudaFree(input)")
            self.d_input = None
        if self.d_output is not None:
            _check_cuda(self.cudart.cudaFree(self.d_output), "cudaFree(output)")
            self.d_output = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def reset(self) -> None:
        self.input_ring.fill(0.0)
        self.output_ring.fill(0.0)
        self.norm_ring.fill(0.0)

    def process_hop(self, in_hop: np.ndarray) -> np.ndarray:
        if len(in_hop) != self.hop_size:
            raise ValueError(f"Expected hop of length {self.hop_size}, got {len(in_hop)}")

        cudart = self.cudart
        self.input_ring[:-self.hop_size] = self.input_ring[self.hop_size:]
        self.input_ring[-self.hop_size:] = in_hop

        np.copyto(self.in_tensor[0], self.input_ring, casting="same_kind")
        _check_cuda(
            cudart.cudaMemcpyAsync(
                self.d_input,
                self.in_tensor.ctypes.data,
                self.in_tensor.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync(host-to-device)",
        )
        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        _check_cuda(
            cudart.cudaMemcpyAsync(
                self.out_tensor.ctypes.data,
                self.d_output,
                self.out_tensor.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "cudaMemcpyAsync(device-to-host)",
        )
        _check_cuda(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

        pred = self.out_tensor[0].astype(np.float32, copy=False)
        if len(pred) != self.frame_size:
            raise ValueError(f"Expected TensorRT output length {self.frame_size}, got {len(pred)}")

        self.output_ring += pred * self.synthesis_window
        self.norm_ring += self.synthesis_window

        denom = np.maximum(self.norm_ring[:self.hop_size], 1e-6)
        out_hop = (self.output_ring[:self.hop_size] / denom).astype(np.float32)

        self.output_ring[:-self.hop_size] = self.output_ring[self.hop_size:]
        self.output_ring[-self.hop_size:] = 0.0
        self.norm_ring[:-self.hop_size] = self.norm_ring[self.hop_size:]
        self.norm_ring[-self.hop_size:] = 0.0
        return out_hop


def warmup(engine, n: int = 64) -> None:
    print(f"Warming up ({n} iters)...", end="", flush=True)
    hop_size = getattr(engine, "hop_size", HOP_SIZE)
    dummy = (np.random.randn(hop_size) * 0.05).astype(np.float32)
    lats = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = engine.process_hop(dummy)
        lats.append((time.perf_counter() - t0) * 1000.0)
    avg = float(np.mean(lats[len(lats) // 2:]))
    print(f" done. avg worker inference: {avg:.2f} ms")
    engine.reset()


# ------------------------------------------------------------
# WAV file mode
# ------------------------------------------------------------
def prepare_audio_file(input_path: str) -> np.ndarray:
    audio, sr = torchaudio.load(input_path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} -> {SAMPLE_RATE} Hz...")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.squeeze(0).numpy().astype(np.float32)


def save_audio_file(output_path: str, audio: np.ndarray) -> None:
    torchaudio.save(output_path, torch.from_numpy(audio).unsqueeze(0), SAMPLE_RATE)


def process_wav(
    engine,
    input_path: str,
    output_path: Optional[str],
    wet: float,
    volume: float,
    play: bool,
    output_device=None,
    host_api: Optional[str] = "JACK",
    latency="low",
) -> None:
    hop_size = getattr(engine, "hop_size", HOP_SIZE)
    warmup(engine)
    audio_np = prepare_audio_file(input_path)
    orig_len = len(audio_np)
    duration = orig_len / SAMPLE_RATE
    print(f"Input: {input_path} ({duration:.2f}s, {orig_len:,} samples)")

    pad = (hop_size - (orig_len % hop_size)) % hop_size
    if pad:
        audio_np = np.concatenate([audio_np, np.zeros(pad, dtype=np.float32)])

    n_steps = len(audio_np) // hop_size
    engine.reset()

    collected = np.zeros_like(audio_np) if output_path is not None else None
    lats = []
    stream = None

    try:
        if play:
            play_device = find_device(output_device, host_api, want_input=False)
            validate_device(play_device, want_input=False)
            stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                blocksize=hop_size,
                device=play_device,
                channels=1,
                dtype=DTYPE,
                latency=latency,
            )
            stream.start()
            print(f"Playing processed output on device: {play_device}")

        print(f"Processing {n_steps:,} hops...")
        for i in range(n_steps):
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
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
        close = getattr(engine, "close", None)
        if close is not None:
            close()

    if collected is not None and output_path is not None:
        collected = collected[:orig_len]
        save_audio_file(output_path, collected)
        print(f"Saved to: {output_path}")

    if lats:
        arr = np.array(lats, dtype=np.float32)
        print(
            f"Latency ms: avg={arr.mean():.2f} "
            f"p95={np.percentile(arr, 95):.2f} max={arr.max():.2f}"
        )


def create_engine(args):
    frame_size = int(args.frame_size)
    hop_size = int(args.hop_size)
    if args.engine:
        return TrtOLAEngine(args.engine, frame_size=frame_size, hop_size=hop_size), "TensorRT"
    providers = choose_providers(args.provider)
    return OnnxOLAEngine(args.model, providers=providers, frame_size=frame_size), "ONNX Runtime"


# ------------------------------------------------------------
# Threaded live audio pipeline
# ------------------------------------------------------------
class ThreadedLivePipeline:
    def __init__(
        self,
        engine,
        mode_name: str,
        input_device=None,
        output_device=None,
        host_api: Optional[str] = "JACK",
        latency="low",
        queue_hops: int = 6,
        wet: float = 1.0,
        volume: float = 1.0,
        fallback: str = "dry",
    ):
        self.input_device = find_device(input_device, host_api, want_input=True)
        self.output_device = find_device(output_device, host_api, want_input=False)
        validate_device(self.input_device, want_input=True)
        validate_device(self.output_device, want_input=False)

        self.latency = latency
        self.queue_hops = max(2, int(queue_hops))
        self.wet = wet
        self.volume = volume
        self.fallback = fallback
        self.running = False

        self.in_q = queue.Queue(maxsize=self.queue_hops)
        self.out_q = queue.Queue(maxsize=self.queue_hops)
        self.engine = engine
        self.mode_name = mode_name
        self.frame_size = getattr(engine, "frame_size", FRAME_SIZE)
        self.hop_size = getattr(engine, "hop_size", HOP_SIZE)

        self._worker = None
        self._frames = 0
        self._xruns = 0
        self._dropped_in = 0
        self._missed_out = 0
        self._worker_lats = []

    def _drop_oldest_and_put(self, q, item) -> None:
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

    def _worker_loop(self) -> None:
        while self.running:
            try:
                in_hop = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                out_hop = self.engine.process_hop(in_hop)
                if len(self._worker_lats) < 4000:
                    self._worker_lats.append((time.perf_counter() - t0) * 1000.0)
                self._drop_oldest_and_put(self.out_q, out_hop)
            except Exception as exc:
                print(f"Worker error: {exc}", file=sys.stderr)

    def _audio_callback(self, indata, outdata, frames, time_info, status) -> None:
        if status:
            self._xruns += 1

        if frames != self.hop_size:
            outdata[:, 0] = 0.0
            return

        in_hop = indata[:, 0].copy().astype(np.float32, copy=False)
        try:
            self.in_q.put_nowait(in_hop)
        except queue.Full:
            self._dropped_in += 1
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
            out_hop = in_hop if self.fallback == "dry" else np.zeros(self.hop_size, dtype=np.float32)

        mixed = self.wet * out_hop + (1.0 - self.wet) * in_hop
        outdata[:, 0] = np.clip(mixed * self.volume, -1.0, 1.0)
        self._frames += 1

    def run(self) -> None:
        warmup(self.engine)

        block_ms = 1000.0 * self.hop_size / SAMPLE_RATE
        win_ms = 1000.0 * self.frame_size / SAMPLE_RATE
        print(f"\n--- Jetson {self.mode_name} Live Mode ---------------------")
        print(f"SR: {SAMPLE_RATE} Hz | block: {self.hop_size} samples ({block_ms:.2f} ms)")
        print(f"Window: {self.frame_size} samples ({win_ms:.2f} ms) | queue: {self.queue_hops} hops")
        print(f"Input device: {self.input_device} | Output device: {self.output_device}")
        print(f"Latency: {self.latency} | Wet: {self.wet:.0%} | Volume: {self.volume:.1f}x")
        print(f"Fallback when worker is late: {self.fallback}")
        print("Controls: q=quit  +/-=volume  m=toggle mix  r=reset rings")
        print("-----------------------------------------------\n")

        self.running = True
        self._worker = threading.Thread(target=self._worker_loop, name="inference-audio-worker", daemon=True)
        self._worker.start()

        try:
            with sd.Stream(
                samplerate=SAMPLE_RATE,
                blocksize=self.hop_size,
                device=(self.input_device, self.output_device),
                channels=1,
                dtype=DTYPE,
                latency=self.latency,
                callback=self._audio_callback,
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
            if self._worker is not None:
                self._worker.join(timeout=1.0)
            close = getattr(self.engine, "close", None)
            if close is not None:
                close()
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
        print("\n--- Stats ------------------------------------")
        print(f"Callbacks       : {self._frames:,}")
        print(f"PortAudio status: {self._xruns:,}")
        print(f"Dropped input   : {self._dropped_in:,}")
        print(f"Missed output   : {self._missed_out:,}")
        if self._worker_lats:
            lats = np.array(self._worker_lats, dtype=np.float32)
            budget_ms = 1000.0 * self.hop_size / SAMPLE_RATE
            print(
                f"Worker ms       : avg={lats.mean():.2f} "
                f"p95={np.percentile(lats, 95):.2f} max={lats.max():.2f}"
            )
            print(f"Budget/block    : {budget_ms:.2f} ms")
        print("----------------------------------------------")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Guitar->piano on Jetson Orin via JACK/ALSA")
    ap.add_argument("--model", default="./model.onnx", help="Path to exported ONNX file")
    ap.add_argument("--engine", default=None, help="Path to serialized TensorRT engine (.plan)")
    ap.add_argument("--input", default=None, help="[WAV mode] Input WAV file")
    ap.add_argument("--output", default=None, help="[WAV mode] Output WAV file")
    ap.add_argument("--play", action="store_true", help="[WAV mode] Play output while processing")
    ap.add_argument("--frame_size", type=int, default=FRAME_SIZE, help="Model input/output frame size")
    ap.add_argument("--hop_size", type=int, default=HOP_SIZE, help="Model input/output hop size")
    ap.add_argument("--list-devices", action="store_true", help="List PortAudio devices and exit")
    ap.add_argument("--host-api", default="JACK", help="Host API filter for device names, e.g. JACK or ALSA")
    ap.add_argument("--input-device", default=None, help="Input device index or partial device name")
    ap.add_argument("--output-device", default=None, help="Output device index or partial device name")
    ap.add_argument("--latency", default="low", help="sounddevice latency: low, high, or seconds like 0.08")
    ap.add_argument("--queue-hops", type=int, default=6, help="Worker safety queue size in hops")
    ap.add_argument("--fallback", choices=["dry", "zero"], default="dry", help="Output when worker is late")
    ap.add_argument("--wet", type=float, default=1.0, help="Wet mix 0-1")
    ap.add_argument("--volume", type=float, default=1.0, help="Output gain")
    ap.add_argument(
        "--provider",
        choices=["auto", "tensorrt", "cuda", "cpu"],
        default="auto",
        help="ONNX Runtime execution provider preference",
    )
    args = ap.parse_args()

    if args.frame_size <= 0:
        print(f"Error: --frame_size must be positive, got {args.frame_size}", file=sys.stderr)
        sys.exit(1)
    if args.frame_size < HOP_SIZE:
        print(
            f"Error: --frame_size must be at least the hop size ({HOP_SIZE}), got {args.frame_size}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list_devices:
        list_devices(args.host_api)
        return

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            print(f"Error: file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        if args.output is None and not args.play:
            args.output = str(input_path.with_name(f"{input_path.stem}_piano.wav"))

        engine, mode_name = create_engine(args)
        print(f"Mode: {mode_name} WAV processing")
        process_wav(
            engine=engine,
            input_path=args.input,
            output_path=args.output,
            wet=args.wet,
            volume=args.volume,
            play=args.play,
            output_device=args.output_device,
            host_api=args.host_api,
            latency=parse_latency(args.latency),
        )
        return

    engine, mode_name = create_engine(args)

    pipe = ThreadedLivePipeline(
        engine=engine,
        mode_name=mode_name,
        input_device=args.input_device,
        output_device=args.output_device,
        host_api=args.host_api,
        latency=parse_latency(args.latency),
        queue_hops=args.queue_hops,
        wet=args.wet,
        volume=args.volume,
        fallback=args.fallback,
    )
    pipe.run()


if __name__ == "__main__":
    main()
