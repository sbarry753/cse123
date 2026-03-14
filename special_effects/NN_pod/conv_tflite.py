#!/usr/bin/env python3

"""
conv_tflite.py

Converts a trained pytorch model into a quantized tflite micro model to run on the Daisy Seed.
Uses a representative sample from the training data to calibrate the quantization.

To turn a tflite model to a C header to use on the Daisy Seed, run:
xxd -i onset_best.tflite > tflite_onset.h

usage: python3 conv_tflite.py --model-path MODEL PATH --manifest-path MANIFEST PATH --output-path onset_best.tflite
"""

import argparse
import json
import tempfile
from pathlib import Path

import litert_torch
import numpy as np
import soundfile as sf
import tensorflow as tf
import torch
import torch.nn as nn

CALIBRATION_SAMPLES = 128
RNG = np.random.default_rng(0)

class OnsetNet(nn.Module):
    """
    Tiny 1D conv with four heads.

    Accepts variable input length due to AdaptiveAvgPool1d(1).
    """

    def __init__(self, note_vocab_size: int, width: int = 64, n_strings: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, width // 2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(width // 2, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.note_head = nn.Linear(width, note_vocab_size)
        self.string_head = nn.Linear(width, n_strings)
        self.picked_head = nn.Linear(width, 1)
        self.sustain_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor):
        z = self.conv(x).squeeze(-1)
        note_logits = self.note_head(z)
        string_logits = self.string_head(z)
        picked_logit = self.picked_head(z).squeeze(-1)
        sustain_logit = self.sustain_head(z).squeeze(-1)
        return note_logits, string_logits, picked_logit, sustain_logit


def load_checkpoint(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    state = checkpoint["model_state"]
    note_vocab_size = state["note_head.weight"].shape[0]
    width = state["note_head.weight"].shape[1]
    export_window_samples = int(checkpoint["export_window_samples"])
    preemph_coef = float(checkpoint.get("args", {}).get("preemph_coef", 0.0))
    sample_rate = int(checkpoint.get("sr", 48000))
    return checkpoint, state, note_vocab_size, width, export_window_samples, preemph_coef, sample_rate


def preprocess_window(x: np.ndarray, preemph_coef: float) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if preemph_coef > 0.0:
        y = np.empty_like(x)
        y[..., 0] = x[..., 0]
        y[..., 1:] = x[..., 1:] - preemph_coef * x[..., :-1]
        x = y
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-12)
    x = x / np.maximum(rms, 1e-4)
    
    return x

def calibration_manifest_entries(manifest_path: str) -> tuple[dict[str, str], ...]:
    manifest = Path(manifest_path)
    if not manifest.exists():
        return ()

    entries = []
    with manifest.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "audio" in payload and "label" in payload:
                entries.append(payload)
    return tuple(entries)


def load_label(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as label_file:
        return json.load(label_file)


def load_wav_mono(path: str) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def sample_window_from_audio(audio: np.ndarray, window_samples: int) -> np.ndarray:
    if audio.shape[0] <= window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[: audio.shape[0]] = audio
        return padded

    max_start = audio.shape[0] - window_samples
    start_candidates = [
        int(RNG.integers(0, min(max_start, max_start // 4) + 1)),
        int(RNG.integers(0, max_start + 1)),
        int(RNG.integers(0, min(max_start, max_start // 2) + 1)),
    ]

    for start in start_candidates:
        window = audio[start : start + window_samples]
        if np.sqrt(np.mean(window * window) + 1e-12) > 1e-4:
            return window.astype(np.float32)

    start = int(RNG.integers(0, max_start + 1))
    return audio[start : start + window_samples].astype(np.float32)


def pad_or_trim_window(window: np.ndarray, window_samples: int) -> np.ndarray:
    if window.shape[0] >= window_samples:
        return window[:window_samples].astype(np.float32, copy=False)

    padded = np.zeros(window_samples, dtype=np.float32)
    padded[: window.shape[0]] = window
    return padded


def sample_window_from_labeled_audio(
    audio: np.ndarray,
    label: dict,
    window_samples: int,
    index: int,
) -> np.ndarray:
    if audio.shape[0] <= window_samples:
        return pad_or_trim_window(audio, window_samples)

    max_start = audio.shape[0] - window_samples
    onset = int(label.get("onset_sample", 0))

    if "pick_region" in label:
        pick_start, pick_end = (int(label["pick_region"][0]), int(label["pick_region"][1]))
    else:
        pick_start = int(label.get("transient_window_start", onset))
        pick_end = int(label.get("transient_window_end", onset + window_samples))

    if "sustain_region" in label:
        sustain_start, sustain_end = (
            int(label["sustain_region"][0]),
            int(label["sustain_region"][1]),
        )
    else:
        sustain_start, sustain_end = pick_start, pick_end

    mode = index % 3
    candidate_starts: list[int] = []

    if mode == 0:
        onset_offset = int(
            RNG.integers(int(0.10 * window_samples), int(0.90 * window_samples) + 1)
        )
        candidate_starts.append(int(np.clip(pick_start - onset_offset, 0, max_start)))
    elif mode == 1:
        sustain_window_max = max(sustain_start, sustain_end - window_samples)
        sustain_start_clamped = int(np.clip(sustain_start, 0, max_start))
        sustain_window_max = int(np.clip(sustain_window_max, sustain_start_clamped, max_start))
        if sustain_start_clamped <= sustain_window_max:
            candidate_starts.append(
                int(RNG.integers(sustain_start_clamped, sustain_window_max + 1))
            )
        candidate_starts.append(int(np.clip(sustain_start, 0, max_start)))
    else:
        pre_end = int(min(max_start, onset - window_samples))
        if pre_end > 0:
            candidate_starts.append(int(RNG.integers(0, pre_end + 1)))

    for start in candidate_starts:
        window = pad_or_trim_window(audio[start : start + window_samples], window_samples)
        if mode == 2:
            return window
        if np.sqrt(np.mean(window * window) + 1e-12) > 1e-4:
            return window

    return sample_window_from_audio(audio, window_samples)

def make_manifest_sample(
    manifest_path: Path,
    window_samples: int,
    preemph_coef: float,
    index: int,
) -> np.ndarray | None:
    entries = calibration_manifest_entries(str(manifest_path))
    if not entries:
        return None

    entry = entries[index % len(entries)]
    dataset_root = manifest_path.parent
    audio_path = dataset_root / entry["audio"]
    label_path = dataset_root / entry["label"]

    audio = load_wav_mono(str(audio_path))
    label = load_label(str(label_path))

    window = sample_window_from_labeled_audio(audio, label, window_samples, index)
    return preprocess_window(window.reshape(1, 1, window_samples), preemph_coef)


def representative_dataset(manifest_path: Path, window_samples: int, preemph_coef: float):
    for index in range(CALIBRATION_SAMPLES):
        sample = make_manifest_sample(manifest_path, window_samples, preemph_coef, index)
        if sample is None:
            raise RuntimeError(f"No representative samples found in manifest: {manifest_path}")
        yield [sample]

def quantize_array(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(x / scale) + zero_point
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)


def dequantize_array(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (x.astype(np.float32) - zero_point) * scale

def convert_model(args):
    _, state, note_vocab_size, width, window_samples, preemph_coef, _sample_rate = load_checkpoint(
        args.model_path
    )

    model = OnsetNet(note_vocab_size, width=width)
    model.load_state_dict(state)
    model.eval()

    sample_input = (torch.randn(1, 1, window_samples),)

    with tempfile.TemporaryDirectory() as saved_model_dir:
        litert_torch.convert(model, sample_input, _saved_model_dir=saved_model_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(
            args.manifest_path, window_samples, preemph_coef
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

    args.output_path.write_bytes(tflite_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OnsetNet checkpoint to fully int8 TFLite.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the trained PyTorch model (.pt).",
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to the calibration manifest.jsonl.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("onset_best.tflite"),
        help="Path to write the converted .tflite model.",
    )
    
    args = parser.parse_args()

    convert_model(args)
