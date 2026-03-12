#!/usr/bin/env python3
"""
export_onset_header.py

Export a trained onset_best.pt checkpoint to a C/C++ header usable by a JUCE
plugin or other native runtime.

This exporter matches the 5-conv OnsetNet used in your Visualizer.py:

    feat.0   Conv1d(1 -> W, 63, stride=2, padding=31, bias=False)
    feat.1   BatchNorm1d(W)
    feat.3   Conv1d(W -> W, 31, stride=2, padding=15, dilation=1, bias=False)
    feat.4   BatchNorm1d(W)
    feat.6   Conv1d(W -> W, 15, stride=1, padding=14, dilation=2, bias=False)
    feat.7   BatchNorm1d(W)
    feat.9   Conv1d(W -> W, 15, stride=1, padding=28, dilation=4, bias=False)
    feat.10  BatchNorm1d(W)
    feat.12  Conv1d(W -> W, 15, stride=1, padding=56, dilation=8, bias=False)
    feat.13  BatchNorm1d(W)
    note_head
    string_head

Usage:
    python export_onset_header.py \
        --ckpt stage1_dist_finetune_v7/onset_best.pt \
        --metadata dataset/metadata.json \
        --out onset_weights.h
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


N_STRINGS = 6


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def note_name_to_midi(name: str) -> Optional[int]:
    if not name or not isinstance(name, str):
        return None

    s = name.strip().replace("♯", "#").replace("♭", "b")
    pitch_map = {
        "C": 0, "C#": 1, "Db": 1,
        "D": 2, "D#": 3, "Eb": 3,
        "E": 4,
        "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8,
        "A": 9, "A#": 10, "Bb": 10,
        "B": 11,
    }

    if len(s) >= 3 and s[1] in ("#", "b"):
        pc = s[:2]
        octv = s[2:]
    else:
        pc = s[:1]
        octv = s[1:]

    if pc not in pitch_map:
        return None

    try:
        octave = int(octv)
    except ValueError:
        return None

    return (octave + 1) * 12 + pitch_map[pc]


def ascii_pitch(name: str) -> str:
    return name.replace("♯", "#").replace("♭", "b")


def c_float(v: float) -> str:
    if v != v:
        return "0.0f"
    if v == float("inf"):
        return "3.402823466e+38f"
    if v == float("-inf"):
        return "-3.402823466e+38f"
    return f"{float(v):.8g}f"


def emit_float_array(lines: List[str], name: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().cpu().float().contiguous().view(-1).tolist()
    lines.append(f"// shape: {list(tensor.shape)}")
    lines.append(f"static const float {name}[{len(flat)}] = {{")
    chunk: List[str] = []
    for i, v in enumerate(flat, start=1):
        chunk.append(c_float(v))
        if len(chunk) >= 8:
            lines.append("    " + ", ".join(chunk) + ",")
            chunk = []
    if chunk:
        lines.append("    " + ", ".join(chunk) + ",")
    lines.append("};")
    lines.append("")


def emit_int_array(lines: List[str], name: str, values: List[int]) -> None:
    lines.append(f"static const int {name}[{len(values)}] = {{")
    chunk: List[str] = []
    for v in values:
        chunk.append(str(int(v)))
        if len(chunk) >= 16:
            lines.append("    " + ", ".join(chunk) + ",")
            chunk = []
    if chunk:
        lines.append("    " + ", ".join(chunk) + ",")
    lines.append("};")
    lines.append("")


def emit_string_array(lines: List[str], name: str, values: List[str]) -> None:
    lines.append(f"static const char* {name}[{len(values)}] = {{")
    for v in values:
        safe = ascii_pitch(v).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    "{safe}",')
    lines.append("};")
    lines.append("")


def require_key(sd: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in sd:
        available = "\n".join(sorted(sd.keys()))
        raise KeyError(
            f"Missing checkpoint key: {key}\n\n"
            f"Available keys:\n{available}"
        )
    return sd[key]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to onset_best.pt")
    ap.add_argument("--metadata", required=True, help="Path to metadata.json")
    ap.add_argument("--out", required=True, help="Output .h path")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    metadata_path = Path(args.metadata)
    out_path = Path(args.out)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    meta = load_json(str(metadata_path))
    sd = ckpt["model_state"]

    sr = int(ckpt.get("sr", meta.get("sr", 48000)))
    width = int(ckpt.get("width", ckpt.get("args", {}).get("width", 128)))
    vocab_size = int(ckpt.get("vocab_size", len(meta.get("index_to_pitch", []))))
    export_window = int(
        ckpt.get(
            "export_window_samples",
            round(sr * float(ckpt.get("args", {}).get("export_ms", 10.0)) / 1000.0),
        )
    )
    thresh = float(ckpt.get("thresh", 0.2))
    stage2 = int(bool(ckpt.get("stage2", False)))
    preemph_coef = float(ckpt.get("args", {}).get("preemph_coef", 0.0))

    index_to_pitch = list(meta.get("index_to_pitch", []))
    if not index_to_pitch:
        index_to_pitch = [str(i) for i in range(vocab_size)]
    if len(index_to_pitch) < vocab_size:
        index_to_pitch.extend(str(i) for i in range(len(index_to_pitch), vocab_size))
    index_to_pitch = index_to_pitch[:vocab_size]

    index_to_midi: List[int] = []
    for i, p in enumerate(index_to_pitch):
        m = note_name_to_midi(p)
        index_to_midi.append(i if m is None else int(m))

    lines: List[str] = []
    lines.extend(
        [
            "// Auto-generated - DO NOT EDIT",
            f"// Source checkpoint: {ckpt_path.name}",
            f"// Source metadata: {metadata_path.name}",
            "// Exported for native inference / JUCE use",
            "",
            "#pragma once",
            "#include <stdint.h>",
            "",
            f"#define ONSET_SR {sr}",
            f"#define ONSET_NOTE_VOCAB_SIZE {vocab_size}",
            f"#define ONSET_WINDOW {export_window}",
            f"#define ONSET_WIDTH {width}",
            f"#define ONSET_N_STRINGS {N_STRINGS}",
            f"#define ONSET_STAGE2 {stage2}",
            f"#define ONSET_THRESH {thresh:.8g}f",
            f"#define ONSET_PREEMPH {preemph_coef:.8g}f",
            "",
        ]
    )

    emit_string_array(lines, "onset_pitch_names", index_to_pitch)
    emit_int_array(lines, "onset_index_to_midi", index_to_midi)

    layer_specs = [
        ("feat.0.weight",  "feat_0_conv_weight"),
        ("feat.1.weight",  "feat_1_bn_weight"),
        ("feat.1.bias",    "feat_1_bn_bias"),
        ("feat.1.running_mean", "feat_1_bn_running_mean"),
        ("feat.1.running_var",  "feat_1_bn_running_var"),

        ("feat.3.weight",  "feat_3_conv_weight"),
        ("feat.4.weight",  "feat_4_bn_weight"),
        ("feat.4.bias",    "feat_4_bn_bias"),
        ("feat.4.running_mean", "feat_4_bn_running_mean"),
        ("feat.4.running_var",  "feat_4_bn_running_var"),

        ("feat.6.weight",  "feat_6_conv_weight"),
        ("feat.7.weight",  "feat_7_bn_weight"),
        ("feat.7.bias",    "feat_7_bn_bias"),
        ("feat.7.running_mean", "feat_7_bn_running_mean"),
        ("feat.7.running_var",  "feat_7_bn_running_var"),

        ("feat.9.weight",  "feat_9_conv_weight"),
        ("feat.10.weight", "feat_10_bn_weight"),
        ("feat.10.bias",   "feat_10_bn_bias"),
        ("feat.10.running_mean", "feat_10_bn_running_mean"),
        ("feat.10.running_var",  "feat_10_bn_running_var"),

        ("feat.12.weight", "feat_12_conv_weight"),
        ("feat.13.weight", "feat_13_bn_weight"),
        ("feat.13.bias",   "feat_13_bn_bias"),
        ("feat.13.running_mean", "feat_13_bn_running_mean"),
        ("feat.13.running_var",  "feat_13_bn_running_var"),

        ("note_head.weight",   "note_head_weight"),
        ("note_head.bias",     "note_head_bias"),
        ("string_head.weight", "string_head_weight"),
        ("string_head.bias",   "string_head_bias"),
    ]

    for sd_key, out_name in layer_specs:
        emit_float_array(lines, out_name, require_key(sd, sd_key))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote header: {out_path}")
    print(f"[INFO] sr={sr} vocab={vocab_size} width={width} window={export_window}")
    print(f"[INFO] stage2={bool(stage2)} thresh={thresh:.4f} preemph={preemph_coef:.4f}")


if __name__ == "__main__":
    main()