#!/usr/bin/env python3

"""Export quantized TFLite Micro submodels and benchmark data for Daisy Seed."""
import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf
import torch

ROOT = Path(__file__).resolve().parent
SEED_PROTOTYPE_DIR = ROOT / "seed_prototype"
HEADER_PATH = SEED_PROTOTYPE_DIR / "generated_tflm_model_data.h"
DEFAULT_CHECKPOINT = ROOT / "checkpoints_temporal" / "best_model.pt"
DEFAULT_CACHE_DIR = ROOT / "data" / ".g2p_temporal_cache"
REPORT_PATH = SEED_PROTOTYPE_DIR / "generated_tflm_report.txt"

MODEL_SEED = 42
CASE_RANDOM_SEED = 43
CALIBRATION_SAMPLES = 64

@dataclass(frozen=True)
class QuantInfo:
    scale: float
    zero_point: int

@dataclass
class TfliteRunResult:
    input_quant: QuantInfo
    output_quants: list[QuantInfo]
    input_int8: np.ndarray
    output_int8: list[np.ndarray]
    output_float: list[np.ndarray]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export quantized TFLite Micro submodels for the Daisy Seed"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path. Defaults to checkpoints_temporal/best_model.pt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Path to the temporal cache directory.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=CALIBRATION_SAMPLES,
        help="Maximum number of calibration windows to use.",
    )
    return parser.parse_args()


def gelu_approx(x: tf.Tensor) -> tf.Tensor:
    return x * tf.nn.sigmoid(tf.constant(1.702, dtype=x.dtype) * x)


class InstanceNorm2D(tf.keras.layers.Layer):
    def __init__(self, channels: int, eps: float = 1.0e-5, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.eps = eps

    def build(self, input_shape) -> None:
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.channels,),
            initializer="ones",
            trainable=False,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(self.channels,),
            initializer="zeros",
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        y = (x - mean) * tf.math.rsqrt(var + self.eps)
        gamma = tf.reshape(self.gamma, (1, 1, 1, self.channels))
        beta = tf.reshape(self.beta, (1, 1, 1, self.channels))
        return y * gamma + beta

class ConvBlock2D(tf.keras.layers.Layer):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1), name: str = "block"):
        super().__init__(name=name)
        self.conv0 = tf.keras.layers.Conv2D(
            out_ch,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=True,
            name=f"{name}_conv0",
        )
        self.norm0 = InstanceNorm2D(out_ch, name=f"{name}_norm0")
        self.act0 = tf.keras.layers.Activation(gelu_approx, name=f"{name}_gelu0")
        self.conv1 = tf.keras.layers.Conv2D(
            out_ch,
            kernel_size=3,
            padding="same",
            use_bias=True,
            name=f"{name}_conv1",
        )
        self.norm1 = InstanceNorm2D(out_ch, name=f"{name}_norm1")
        self.act1 = tf.keras.layers.Activation(gelu_approx, name=f"{name}_gelu1")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.act0(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        return x

class UpBlock2D(tf.keras.layers.Layer):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, name: str = "up"):
        super().__init__(name=name)
        _ = in_ch, skip_ch
        self.block = ConvBlock2D(in_ch + skip_ch, out_ch, name=f"{name}_block")

    def call(self, x: tf.Tensor, skip: tf.Tensor) -> tf.Tensor:
        target_hw = tf.shape(skip)[1:3]
        x = tf.image.resize(x, size=target_hw, method="bilinear")
        x = tf.concat([x, skip], axis=-1)
        return self.block(x)

class SpectralTemporalUNetTF(tf.keras.Model):
    def __init__(self):
        super().__init__(name="spectral_temporal_unet")
        self.enc1 = ConvBlock2D(9, 32, name="enc1")
        self.enc2 = ConvBlock2D(32, 64, stride=(2, 2), name="enc2")
        self.enc3 = ConvBlock2D(64, 128, stride=(2, 2), name="enc3")
        self.bottleneck = ConvBlock2D(128, 128, name="bottleneck")
        self.dec3 = UpBlock2D(128, 128, 64, name="dec3")
        self.dec2 = UpBlock2D(64, 64, 32, name="dec2")
        self.dec1 = UpBlock2D(32, 32, 32, name="dec1")
        self.out_mask = tf.keras.layers.Conv2D(1, kernel_size=1, name="out_mask")
        self.out_res = tf.keras.layers.Conv2D(1, kernel_size=1, name="out_res")
        self.out_phase = tf.keras.layers.Conv2D(1, kernel_size=1, name="out_phase")

    def call(self, x: tf.Tensor):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        z = self.bottleneck(s3)
        x = self.dec3(z, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        mask = 0.5 + 2.5 * tf.nn.sigmoid(self.out_mask(x))
        residual = self.out_res(x)
        phase_delta = 0.45 * tf.nn.tanh(self.out_phase(x))
        return [mask, residual, phase_delta]

class TransientShaperTF(tf.keras.Model):
    def __init__(self):
        super().__init__(name="transient_shaper")
        self.delta0 = tf.keras.layers.Conv1D(32, kernel_size=9, padding="same", name="delta0")
        self.delta1 = tf.keras.layers.Activation(gelu_approx, name="delta_gelu0")
        self.delta2 = tf.keras.layers.Conv1D(32, kernel_size=9, padding="same", name="delta2")
        self.delta3 = tf.keras.layers.Activation(gelu_approx, name="delta_gelu1")
        self.delta4 = tf.keras.layers.Conv1D(32, kernel_size=5, padding="same", name="delta4")
        self.delta5 = tf.keras.layers.Activation(gelu_approx, name="delta_gelu2")
        self.delta6 = tf.keras.layers.Conv1D(1, kernel_size=1, padding="same", name="delta6")

        self.gate0 = tf.keras.layers.Conv1D(8, kernel_size=7, padding="same", name="gate0")
        self.gate1 = tf.keras.layers.Activation(gelu_approx, name="gate_gelu0")
        self.gate2 = tf.keras.layers.Conv1D(1, kernel_size=1, padding="same", name="gate2")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        delta = self.delta0(x)
        delta = self.delta1(delta)
        delta = self.delta2(delta)
        delta = self.delta3(delta)
        delta = self.delta4(delta)
        delta = self.delta5(delta)
        delta = self.delta6(delta)

        gate = self.gate0(tf.abs(x))
        gate = self.gate1(gate)
        gate = tf.nn.sigmoid(self.gate2(gate))
        return x + 0.25 * gate * delta

def format_float(value: float) -> str:
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    text = f"{value:.9e}f"
    if text == "-0.000000000e+00f":
        return "0.000000000e+00f"
    return text

def emit_float_array(lines: list[str], name: str, array: np.ndarray, *, shape_comment: bool = True) -> None:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if shape_comment:
        lines.append(f"// {name}: shape={tuple(array.shape)}")
    lines.append(f"alignas(16) static const float {name}[{flat.size}] = {{")
    row = []
    for idx, value in enumerate(flat, start=1):
        row.append(format_float(float(value)))
        if len(row) == 6 or idx == flat.size:
            suffix = "," if idx != flat.size else ""
            lines.append("    " + ", ".join(row) + suffix)
            row = []
    lines.append("};")
    lines.append("")

def emit_uint8_array(lines: list[str], name: str, data: bytes) -> None:
    lines.append(f"alignas(16) static const unsigned char {name}[{len(data)}] = {{")
    row = []
    for idx, value in enumerate(data, start=1):
        row.append(f"0x{value:02x}")
        if len(row) == 12 or idx == len(data):
            suffix = "," if idx != len(data) else ""
            lines.append("    " + ", ".join(row) + suffix)
            row = []
    lines.append("};")
    lines.append("")

def max_abs_and_rmse(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff, dtype=np.float32)))
    return max_abs, rmse

def make_interpreter(model_content: bytes) -> tf.lite.Interpreter:
    return tf.lite.Interpreter(
        model_content=model_content,
        experimental_delegates=[],
        num_threads=1,
    )

def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    return payload

def build_case_tensors(context_frames: int, frame_size: int) -> list[tuple[str, torch.Tensor]]:
    cases = []
    zero = torch.zeros(1, context_frames, frame_size, dtype=torch.float32)
    cases.append(("case_zero", zero))

    impulse = torch.zeros(1, context_frames, frame_size, dtype=torch.float32)
    impulse[0, context_frames - 1, frame_size // 2] = 1.0
    cases.append(("case_impulse", impulse))

    generator = torch.Generator().manual_seed(CASE_RANDOM_SEED)
    random_case = 0.15 * torch.randn(
        (1, context_frames, frame_size), generator=generator, dtype=torch.float32
    )
    random_case = torch.clamp(random_case, -1.0, 1.0)
    cases.append(("case_random", random_case))
    return cases

def set_conv2d_weights(layer: tf.keras.layers.Conv2D, weight: torch.Tensor, bias: torch.Tensor) -> None:
    kernel = weight.detach().cpu().numpy().transpose(2, 3, 1, 0).astype(np.float32)
    bias_np = bias.detach().cpu().numpy().astype(np.float32)
    layer.set_weights([kernel, bias_np])

def set_conv1d_weights(layer: tf.keras.layers.Conv1D, weight: torch.Tensor, bias: torch.Tensor) -> None:
    kernel = weight.detach().cpu().numpy().transpose(2, 1, 0).astype(np.float32)
    bias_np = bias.detach().cpu().numpy().astype(np.float32)
    layer.set_weights([kernel, bias_np])

def set_norm_weights(layer: InstanceNorm2D, weight: torch.Tensor, bias: torch.Tensor) -> None:
    gamma = weight.detach().cpu().numpy().astype(np.float32)
    beta = bias.detach().cpu().numpy().astype(np.float32)
    layer.set_weights([gamma, beta])

def load_unet_weights(unet_tf: SpectralTemporalUNetTF, state: dict[str, torch.Tensor]) -> None:
    mapping = [
        (unet_tf.enc1, "unet.enc1.block"),
        (unet_tf.enc2, "unet.enc2.block"),
        (unet_tf.enc3, "unet.enc3.block"),
        (unet_tf.bottleneck, "unet.bottleneck.block"),
        (unet_tf.dec3.block, "unet.dec3.block"),
        (unet_tf.dec2.block, "unet.dec2.block"),
        (unet_tf.dec1.block, "unet.dec1.block"),
    ]
    for block, prefix in mapping:
        set_conv2d_weights(block.conv0, state[f"{prefix}.0.weight"], state[f"{prefix}.0.bias"])
        set_norm_weights(block.norm0, state[f"{prefix}.1.weight"], state[f"{prefix}.1.bias"])
        set_conv2d_weights(block.conv1, state[f"{prefix}.3.weight"], state[f"{prefix}.3.bias"])
        set_norm_weights(block.norm1, state[f"{prefix}.4.weight"], state[f"{prefix}.4.bias"])

    set_conv2d_weights(unet_tf.out_mask, state["unet.out_mask.weight"], state["unet.out_mask.bias"])
    set_conv2d_weights(unet_tf.out_res, state["unet.out_res.weight"], state["unet.out_res.bias"])
    set_conv2d_weights(unet_tf.out_phase, state["unet.out_phase.weight"], state["unet.out_phase.bias"])

def load_transient_weights(transient_tf: TransientShaperTF, state: dict[str, torch.Tensor]) -> None:
    set_conv1d_weights(
        transient_tf.delta0,
        state["transient.delta_net.0.weight"],
        state["transient.delta_net.0.bias"],
    )
    set_conv1d_weights(
        transient_tf.delta2,
        state["transient.delta_net.2.weight"],
        state["transient.delta_net.2.bias"],
    )
    set_conv1d_weights(
        transient_tf.delta4,
        state["transient.delta_net.4.weight"],
        state["transient.delta_net.4.bias"],
    )
    set_conv1d_weights(
        transient_tf.delta6,
        state["transient.delta_net.6.weight"],
        state["transient.delta_net.6.bias"],
    )
    set_conv1d_weights(
        transient_tf.gate0,
        state["transient.gate_net.0.weight"],
        state["transient.gate_net.0.bias"],
    )
    set_conv1d_weights(
        transient_tf.gate2,
        state["transient.gate_net.2.weight"],
        state["transient.gate_net.2.bias"],
    )

def collect_calibration_inputs(model: torch.nn.Module, cache_dir: Path, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    contexts = []
    for cache_path in sorted(cache_dir.glob("*.pt")):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        guitar_ctx = payload["guitar_ctx"]
        if guitar_ctx.numel() == 0:
            continue
        stride = max(1, guitar_ctx.shape[0] // 4)
        for idx in range(0, guitar_ctx.shape[0], stride):
            contexts.append(guitar_ctx[idx])
            if len(contexts) >= max_samples:
                break
        if len(contexts) >= max_samples:
            break

    if not contexts:
        raise RuntimeError(f"No calibration samples found in {cache_dir}")

    batch = torch.stack(contexts, dim=0)

    with torch.no_grad():
        feat_tf, current_log_mag, _current_mag, current_phase = model._prepare_features(batch)
        mask, residual, phase_delta = model.unet(feat_tf)
        out_log_mag = current_log_mag.unsqueeze(1) * mask + residual
        out_mag = torch.exp(out_log_mag.squeeze(1))
        out_phase = current_phase + phase_delta.squeeze(1)
        out_spec = torch.polar(out_mag, out_phase)
        pre_transient = model._istft(out_spec, length=model.frame_size)

    feat_nhwc = feat_tf.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    transient_inputs = pre_transient.unsqueeze(-1).cpu().numpy().astype(np.float32)
    return feat_nhwc, transient_inputs

def representative_dataset(samples: np.ndarray) -> Iterable[list[np.ndarray]]:
    for sample in samples:
        yield [sample[np.newaxis, ...].astype(np.float32)]

def export_quantized_model(keras_model: tf.keras.Model, sample_input: np.ndarray, calibration_inputs: np.ndarray) -> tuple[bytes, dict, list[dict]]:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calibration_inputs)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    interpreter = make_interpreter(tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    ops = interpreter._get_ops_details()

    detail_summary = {
        "input_shape": tuple(int(v) for v in input_detail["shape"]),
        "input_scale": float(input_detail["quantization"][0]),
        "input_zero_point": int(input_detail["quantization"][1]),
        "tensor_dtypes": sorted({str(d["dtype"]) for d in interpreter.get_tensor_details()}),
        "output_shapes": [tuple(int(v) for v in detail["shape"]) for detail in output_details],
        "output_quantization": [
            (
                float(detail["quantization"][0]),
                int(detail["quantization"][1]),
            )
            for detail in output_details
        ],
        "ops": [op["op_name"] for op in ops if op["op_name"] != "DELEGATE"],
        "model_bytes": len(tflite_model),
        "delegate_mode": "builtin_without_default_delegates",
    }
    _ = sample_input
    return tflite_model, detail_summary, output_details

def quantize_array(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError(f"Quantization scale must be positive, got {scale}")
    scaled = np.asarray(x, dtype=np.float32) / np.float32(scale)
    rounded = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
    q = rounded + np.float32(zero_point)
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)

def dequantize_array(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (x.astype(np.float32) - zero_point) * scale

def run_tflite_model(tflite_model: bytes, input_array: np.ndarray) -> TfliteRunResult:
    interpreter = make_interpreter(tflite_model)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    in_scale, in_zero_point = input_detail["quantization"]
    quantized_input = quantize_array(input_array.astype(np.float32), float(in_scale), int(in_zero_point))
    interpreter.set_tensor(input_detail["index"], quantized_input)
    interpreter.invoke()

    outputs_float = []
    outputs_int8 = []
    output_quants = []

    for detail in interpreter.get_output_details():
        tensor = interpreter.get_tensor(detail["index"]).astype(np.int8, copy=True)
        scale, zero_point = detail["quantization"]
        quant = QuantInfo(scale=float(scale), zero_point=int(zero_point))
        outputs_int8.append(tensor)
        outputs_float.append(dequantize_array(tensor, quant.scale, quant.zero_point))
        output_quants.append(quant)

    return TfliteRunResult(
        input_quant=QuantInfo(scale=float(in_scale), zero_point=int(in_zero_point)),
        output_quants=output_quants,
        input_int8=quantized_input.astype(np.int8, copy=True),
        output_int8=outputs_int8,
        output_float=outputs_float,
    )

def build_keras_submodels(
    state: dict[str, torch.Tensor],
    freq_bins: int,
    stft_frames: int,
    frame_size: int,
) -> tuple[SpectralTemporalUNetTF, TransientShaperTF]:
    unet_tf = SpectralTemporalUNetTF()
    transient_tf = TransientShaperTF()

    _ = unet_tf(tf.zeros((1, freq_bins, stft_frames, 9), dtype=tf.float32))
    _ = transient_tf(tf.zeros((1, frame_size, 1), dtype=tf.float32))

    load_unet_weights(unet_tf, state)
    load_transient_weights(transient_tf, state)
    return unet_tf, transient_tf

def assert_float_parity(
    model: torch.nn.Module,
    unet_tf: SpectralTemporalUNetTF,
    transient_tf: TransientShaperTF,
    case: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        feat_tf, current_log_mag, _current_mag, current_phase = model._prepare_features(case)
        pt_mask, pt_residual, pt_phase = model.unet(feat_tf)

        tf_mask, tf_residual, tf_phase = unet_tf(
            feat_tf.permute(0, 2, 3, 1).cpu().numpy(), training=False
        )

        tf_mask = torch.from_numpy(np.transpose(tf_mask.numpy(), (0, 3, 1, 2)))
        tf_residual = torch.from_numpy(np.transpose(tf_residual.numpy(), (0, 3, 1, 2)))
        tf_phase = torch.from_numpy(np.transpose(tf_phase.numpy(), (0, 3, 1, 2)))

        unet_diff = max(
            (pt_mask - tf_mask).abs().max().item(),
            (pt_residual - tf_residual).abs().max().item(),
            (pt_phase - tf_phase).abs().max().item(),
        )

        out_log_mag = current_log_mag.unsqueeze(1) * pt_mask + pt_residual
        out_mag = torch.exp(out_log_mag.squeeze(1))
        out_phase = current_phase + pt_phase.squeeze(1)
        out_spec = torch.polar(out_mag, out_phase)
        pre_transient = model._istft(out_spec, length=model.frame_size)

        pt_transient = model.transient(pre_transient)
        tf_transient = transient_tf(pre_transient.unsqueeze(-1).cpu().numpy(), training=False)
        tf_transient = torch.from_numpy(tf_transient.numpy().squeeze(-1))
        transient_diff = (pt_transient - tf_transient).abs().max().item()

    return unet_diff, transient_diff

def run_quantized_case(
    model: torch.nn.Module,
    unet_tflite: bytes,
    transient_tflite: bytes,
    case: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        feat_tf, current_log_mag, _current_mag, current_phase = model._prepare_features(case)

    feat_nhwc = feat_tf.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    unet_run = run_tflite_model(unet_tflite, feat_nhwc)

    mask = torch.from_numpy(np.transpose(unet_run.output_float[0], (0, 3, 1, 2))).to(torch.float32)
    residual = torch.from_numpy(np.transpose(unet_run.output_float[1], (0, 3, 1, 2))).to(torch.float32)
    phase_delta = torch.from_numpy(np.transpose(unet_run.output_float[2], (0, 3, 1, 2))).to(torch.float32)

    with torch.no_grad():
        out_log_mag = current_log_mag.unsqueeze(1) * mask + residual
        out_mag = torch.exp(out_log_mag.squeeze(1))
        out_phase = current_phase + phase_delta.squeeze(1)
        out_spec = torch.polar(out_mag, out_phase)
        pre_transient = model._istft(out_spec, length=model.frame_size)

    transient_input = pre_transient.unsqueeze(-1).cpu().numpy().astype(np.float32)
    transient_run = run_tflite_model(transient_tflite, transient_input)

    shaped = torch.from_numpy(transient_run.output_float[0].squeeze(-1)).to(torch.float32)
    current_audio = case[:, -1]
    final = torch.clamp(0.985 * shaped + 0.015 * current_audio, -1.0, 1.0)

    return {
        "mask_dequant": unet_run.output_float[0].reshape(-1).astype(np.float32, copy=True),
        "residual_dequant": unet_run.output_float[1].reshape(-1).astype(np.float32, copy=True),
        "phase_delta_dequant": unet_run.output_float[2].reshape(-1).astype(np.float32, copy=True),
        "pre_transient": pre_transient.squeeze(0).cpu().numpy().astype(np.float32),
        "transient_output_dequant": transient_run.output_float[0].reshape(-1).astype(np.float32, copy=True),
        "final_output": final.squeeze(0).cpu().numpy().astype(np.float32),
    }

def compute_mirror_quant_metrics(
    model: torch.nn.Module,
    unet_tf: SpectralTemporalUNetTF,
    transient_tf: TransientShaperTF,
    transient_tflite: bytes,
    case: torch.Tensor,
    quantized_case: dict[str, np.ndarray],
) -> dict[str, float]:
    with torch.no_grad():
        feat_tf, current_log_mag, _current_mag, current_phase = model._prepare_features(case)

    feat_nhwc = feat_tf.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    tf_mask, tf_residual, tf_phase = unet_tf(feat_nhwc, training=False)
    tf_mask_np = tf_mask.numpy().astype(np.float32, copy=False)
    tf_residual_np = tf_residual.numpy().astype(np.float32, copy=False)
    tf_phase_np = tf_phase.numpy().astype(np.float32, copy=False)

    mask_max_abs = float(np.max(np.abs(tf_mask_np.reshape(-1) - quantized_case["mask_dequant"])))
    residual_max_abs = float(
        np.max(np.abs(tf_residual_np.reshape(-1) - quantized_case["residual_dequant"]))
    )
    phase_delta_max_abs = float(
        np.max(np.abs(tf_phase_np.reshape(-1) - quantized_case["phase_delta_dequant"]))
    )

    tf_mask_t = torch.from_numpy(np.transpose(tf_mask_np, (0, 3, 1, 2))).to(torch.float32)
    tf_residual_t = torch.from_numpy(np.transpose(tf_residual_np, (0, 3, 1, 2))).to(torch.float32)
    tf_phase_t = torch.from_numpy(np.transpose(tf_phase_np, (0, 3, 1, 2))).to(torch.float32)

    with torch.no_grad():
        out_log_mag = current_log_mag.unsqueeze(1) * tf_mask_t + tf_residual_t
        out_mag = torch.exp(out_log_mag.squeeze(1))
        out_phase = current_phase + tf_phase_t.squeeze(1)
        out_spec = torch.polar(out_mag, out_phase)
        pre_transient_float = model._istft(out_spec, length=model.frame_size)

    pre_transient_np = pre_transient_float.squeeze(0).cpu().numpy().astype(np.float32)
    pre_transient_max_abs = float(
        np.max(np.abs(pre_transient_np - quantized_case["pre_transient"]))
    )

    transient_float_input = pre_transient_float.unsqueeze(-1).cpu().numpy().astype(np.float32)
    transient_float_out = transient_tf(transient_float_input, training=False).numpy()
    transient_quant_run = run_tflite_model(transient_tflite, transient_float_input)
    transient_quant_max_abs = float(np.max(np.abs(transient_float_out.reshape(-1)
                            - transient_quant_run.output_float[0].reshape(-1).astype(np.float32, copy=False)
                            ))
    )

    current_audio = case[:, -1].squeeze(0).cpu().numpy().astype(np.float32)
    final_float = np.clip(0.985 * transient_float_out.squeeze(0).squeeze(-1) + 0.015 * current_audio, -1.0, 1.0)
    final_max_abs, final_rmse = max_abs_and_rmse(final_float, quantized_case["final_output"])

    return {
        "mask_max_abs": mask_max_abs,
        "residual_max_abs": residual_max_abs,
        "phase_delta_max_abs": phase_delta_max_abs,
        "pre_transient_max_abs": pre_transient_max_abs,
        "transient_max_abs": transient_quant_max_abs,
        "final_max_abs": final_max_abs,
        "final_rmse": final_rmse,
    }

def write_report(
    checkpoint: dict,
    unet_summary: dict,
    transient_summary: dict,
    unet_diff: float,
    transient_diff: float,
    mirror_quant_metrics: dict[str, dict[str, float]],
) -> None:
    lines = [
        "Daisy TFLM export report",
        "========================",
        "",
        "activation_note=quick_gelu_sigmoid_approx",
        "interpreter_delegate_mode=builtin_without_default_delegates",
        f"checkpoint_epoch={checkpoint.get('epoch', 'n/a')}",
        f"checkpoint_val_loss={checkpoint.get('val_loss', 'n/a')}",
        f"float_parity_unet_max_abs={unet_diff:.6e}",
        f"float_parity_transient_max_abs={transient_diff:.6e}",
        "",
        "[unet]",
        f"model_bytes={unet_summary['model_bytes']}",
        f"input_shape={unet_summary['input_shape']}",
        f"input_quant={unet_summary['input_scale']}, {unet_summary['input_zero_point']}",
        f"output_quant={unet_summary['output_quantization']}",
        f"tensor_dtypes={unet_summary['tensor_dtypes']}",
        f"ops={unet_summary['ops']}",
        "",
        "[transient]",
        f"model_bytes={transient_summary['model_bytes']}",
        f"input_shape={transient_summary['input_shape']}",
        f"input_quant={transient_summary['input_scale']}, {transient_summary['input_zero_point']}",
        f"output_quant={transient_summary['output_quantization']}",
        f"tensor_dtypes={transient_summary['tensor_dtypes']}",
        f"ops={transient_summary['ops']}",
        "",
    ]

    lines.extend(
        [
            "[host_float_vs_keras]",
            f"unet_max_abs={unet_diff:.6e}",
            f"transient_max_abs={transient_diff:.6e}",
            "",
        ]
    )

    for case_name, metrics in mirror_quant_metrics.items():
        lines.append(f"[host_keras_vs_quantized_tflite.{case_name}]")
        lines.append(f"mask_max_abs={metrics['mask_max_abs']:.6e}")
        lines.append(f"residual_max_abs={metrics['residual_max_abs']:.6e}")
        lines.append(f"phase_delta_max_abs={metrics['phase_delta_max_abs']:.6e}")
        lines.append(f"pre_transient_max_abs={metrics['pre_transient_max_abs']:.6e}")
        lines.append(f"transient_max_abs={metrics['transient_max_abs']:.6e}")
        lines.append(f"final_max_abs={metrics['final_max_abs']:.6e}")
        lines.append(f"final_rmse={metrics['final_rmse']:.6e}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    args = parse_args()

    from model import (
        CONTEXT_FRAMES,
        FRAME_SIZE,
        HOP_SIZE,
        N_FFT,
        N_FREQ_BINS,
        PolyphonicGuitarToPianoTemporal,
        SAMPLE_RATE,
    )

    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    tf.random.set_seed(MODEL_SEED)

    model = PolyphonicGuitarToPianoTemporal().eval()
    checkpoint = load_checkpoint(model, args.checkpoint.resolve())
    state = model.state_dict()

    cases = build_case_tensors(CONTEXT_FRAMES, FRAME_SIZE)
    feat_calibration, transient_calibration = collect_calibration_inputs(
        model, args.cache_dir.resolve(), max_samples=args.calibration_samples
    )

    unet_tf, transient_tf = build_keras_submodels(
        state, freq_bins=N_FREQ_BINS, stft_frames=5, frame_size=FRAME_SIZE
    )
    unet_diff, transient_diff = assert_float_parity(model, unet_tf, transient_tf, cases[2][1])

    unet_tflite, unet_summary, _ = export_quantized_model(
        unet_tf, feat_calibration[:1], feat_calibration
    )
    transient_tflite, transient_summary, _ = export_quantized_model(
        transient_tf, transient_calibration[:1], transient_calibration
    )

    final_outputs = {}
    mirror_quant_metrics = {}
    for name, case in cases:
        quantized_case = run_quantized_case(model, unet_tflite, transient_tflite, case)
        final_outputs[name] = quantized_case["final_output"]
        mirror_quant_metrics[name] = compute_mirror_quant_metrics(
            model,
            unet_tf,
            transient_tf,
            transient_tflite,
            case,
            quantized_case,
        )

    lines = []
    lines.append("// Auto-generated by export_seed_tflm.py. Do not edit by hand.")
    if "epoch" in checkpoint:
        lines.append(f"// Checkpoint epoch: {checkpoint['epoch']}")
    if "val_loss" in checkpoint:
        lines.append(f"// Checkpoint val_loss: {checkpoint['val_loss']}")
    lines.append(f"// UNet ops: {', '.join(unet_summary['ops'])}")
    lines.append(f"// Transient ops: {', '.join(transient_summary['ops'])}")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("")
    lines.append(f"static constexpr int kTflmModelSeed = {MODEL_SEED};")
    lines.append(f"static constexpr int kTflmCaseRandomSeed = {CASE_RANDOM_SEED};")
    lines.append(f"static constexpr int kSampleRate = {SAMPLE_RATE};")
    lines.append(f"static constexpr int kFrameSize = {FRAME_SIZE};")
    lines.append(f"static constexpr int kHopSize = {HOP_SIZE};")
    lines.append(f"static constexpr int kContextFrames = {CONTEXT_FRAMES};")
    lines.append(f"static constexpr int kFftSize = {N_FFT};")
    lines.append(f"static constexpr int kFreqBins = {N_FREQ_BINS};")
    lines.append("static constexpr int kStftFrames = 5;")
    lines.append("static constexpr int kFeatureChannels = 9;")
    lines.append(f"static constexpr std::size_t kUnetModelBytes = {len(unet_tflite)}u;")
    lines.append(
        f"static constexpr std::size_t kTransientModelBytes = {len(transient_tflite)}u;"
    )
    lines.append("")

    emit_float_array(lines, "g_window", state["window"].detach().cpu().numpy())
    emit_uint8_array(lines, "g_unet_int8_model_data", unet_tflite)
    emit_uint8_array(lines, "g_transient_int8_model_data", transient_tflite)

    for name, case in cases:
        emit_float_array(lines, f"{name}_input", case.squeeze(0).cpu().numpy(), shape_comment=False)
        emit_float_array(lines, f"{name}_golden", final_outputs[name], shape_comment=False)

    HEADER_PATH.write_text("\n".join(lines), encoding="utf-8")
    write_report(
        checkpoint,
        unet_summary,
        transient_summary,
        unet_diff,
        transient_diff,
        mirror_quant_metrics,
    )
    print(f"Wrote {HEADER_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
