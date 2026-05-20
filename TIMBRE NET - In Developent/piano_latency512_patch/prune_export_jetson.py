"""
Prune + export Guitar->Piano model for Jetson Orin.

This does magnitude pruning on Conv/Linear weights, bakes the pruning masks
into the weights, strips optimizer state, and exports a TorchScript model that
jetson_realtime.py can load.

Important: unstructured pruning mostly reduces model size / regularizes the
network. It may not make PyTorch dense CUDA inference faster by itself. The
Jetson speed wins here mostly come from the 512-sample window, TorchScript,
CUDA, inference_mode, and the optimized realtime loop.

Examples
--------
# From a full checkpoint saved by train.py:
python prune_export_jetson.py \
  --checkpoint ./checkpoints/best_model.pt \
  --output_dir ./jetson_export \
  --prune_amount 0.35 \
  --device cuda

# Export without pruning, just strip/trace:
python prune_export_jetson.py \
  --checkpoint ./checkpoints/best_model.pt \
  --output_dir ./jetson_export \
  --prune_amount 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from model import DDSPGuitarToPiano, FRAME_SIZE, HOP_SIZE, SAMPLE_RATE


PRUNABLE_TYPES = (nn.Conv1d, nn.Conv2d, nn.Linear)


class InferenceOnlyWrapper(nn.Module):
    """
    Clean wrapper used only for TorchScript export.

    The training model returns (audio, features, params) and also exposes
    @torch.jit.export infer_frame(). Tracing that full class can fail on some
    PyTorch/Windows setups with: forward already defined. This wrapper has one
    simple forward that returns only audio, so TorchScript gets a clean graph.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, audio_frame: torch.Tensor) -> torch.Tensor:
        y = self.core(audio_frame)
        if isinstance(y, tuple):
            return y[0]
        return y


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> DDSPGuitarToPiano:
    model = DDSPGuitarToPiano().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

    model.eval()
    return model


def prunable_parameters(model: nn.Module) -> list[tuple[nn.Module, str]]:
    params: list[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, PRUNABLE_TYPES):
            params.append((module, "weight"))
    return params


def tensor_sparsity(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    return float((t == 0).sum().item()) / float(t.numel())


def model_sparsity(model: nn.Module) -> dict[str, float]:
    total = 0
    zeros = 0
    by_module: dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, PRUNABLE_TYPES):
            w = module.weight.detach()
            total += w.numel()
            zeros += int((w == 0).sum().item())
            by_module[name or "root"] = tensor_sparsity(w)
    return {
        "global": (zeros / total) if total else 0.0,
        "modules": by_module,
    }


def apply_global_pruning(model: nn.Module, amount: float) -> None:
    if amount <= 0:
        return
    params = prunable_parameters(model)
    if not params:
        print("No Conv/Linear weights found to prune.")
        return

    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Bake masks into normal parameters so the exported model has no pruning wrappers.
    for module, param_name in params:
        prune.remove(module, param_name)


def save_compact_checkpoint(model: nn.Module, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def trace_torchscript(model: nn.Module, output_path: Path, device: torch.device) -> None:
    model.eval()
    dummy = torch.zeros(1, FRAME_SIZE, device=device)

    # Trace a clean inference-only wrapper instead of the full training class.
    # This avoids PyTorch trying to compile both forward() and exported
    # infer_frame(), which can raise "forward already defined".
    wrapper = InferenceOnlyWrapper(model).to(device).eval()

    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, dummy, strict=False)
        try:
            traced = torch.jit.freeze(traced)
        except Exception as e:
            print(f"Warning: torch.jit.freeze skipped: {e}")
        try:
            traced = torch.jit.optimize_for_inference(traced)
        except Exception as e:
            print(f"Warning: optimize_for_inference skipped: {e}")
        traced.save(str(output_path))


def verify_export(scripted_path: Path, device: torch.device) -> None:
    model = torch.jit.load(str(scripted_path), map_location=device)
    model.eval()
    x = torch.randn(1, FRAME_SIZE, device=device) * 0.05
    with torch.inference_mode():
        if hasattr(model, "infer_frame"):
            y = model.infer_frame(x)
        else:
            out = model(x)
            y = out[0] if isinstance(out, tuple) else out
    if y.shape[-1] != FRAME_SIZE:
        raise RuntimeError(f"Bad output shape: {tuple(y.shape)}")
    if not torch.isfinite(y).all():
        raise RuntimeError("Export verification failed: NaN/Inf in output")
    print(f"Verified TorchScript export: input {tuple(x.shape)} -> output {tuple(y.shape)}")


def make_deploy_tar(output_dir: Path, files: Iterable[Path]) -> Path:
    tar_path = output_dir / "jetson_deploy_bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in files:
            if f.exists():
                tar.add(f, arcname=f.name)
    return tar_path


def main() -> None:
    p = argparse.ArgumentParser(description="Prune/export Guitar->Piano model for Jetson Orin")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt or model_weights.pt")
    p.add_argument("--output_dir", default="./jetson_export")
    p.add_argument("--prune_amount", type=float, default=0.35, help="0.0-0.9 global L1 unstructured pruning")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--no_verify", action="store_true")
    args = p.parse_args()

    if not (0.0 <= args.prune_amount < 1.0):
        raise ValueError("--prune_amount must be in [0.0, 1.0)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Frame: {FRAME_SIZE} samples | Hop: {HOP_SIZE} samples | SR: {SAMPLE_RATE} Hz")

    model = load_checkpoint_model(args.checkpoint, device)
    before = model_sparsity(model)
    print(f"Initial Conv/Linear sparsity: {before['global'] * 100:.2f}%")

    apply_global_pruning(model, args.prune_amount)
    after = model_sparsity(model)
    print(f"After pruning Conv/Linear sparsity: {after['global'] * 100:.2f}%")

    metadata = {
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "sample_rate": SAMPLE_RATE,
        "frame_size": FRAME_SIZE,
        "hop_size": HOP_SIZE,
        "prune_amount": args.prune_amount,
        "conv_linear_sparsity": after,
        "note": "Unstructured pruning reduces stored nonzero weights but may not speed up dense PyTorch CUDA kernels.",
    }

    compact_ckpt = output_dir / "model_pruned_checkpoint.pt"
    scripted_path = output_dir / "model_pruned_scripted.pt"
    metadata_path = output_dir / "export_metadata.json"

    save_compact_checkpoint(model.cpu(), compact_ckpt, metadata)
    model.to(device)
    trace_torchscript(model, scripted_path, device)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Copy runtime files next to the export for scp/rsync to Jetson.
    here = Path(__file__).resolve().parent
    for name in ["model.py", "jetson_realtime.py"]:
        src = here / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    readme = output_dir / "JETSON_README.txt"
    readme.write_text(
        "Jetson Orin deployment\n"
        "======================\n\n"
        "Copy this folder to the Jetson, then run:\n\n"
        "  python3 jetson_realtime.py --model ./model_pruned_scripted.pt --input ./guitar.wav --play\n\n"
        "Optional save while playing:\n\n"
        "  python3 jetson_realtime.py --model ./model_pruned_scripted.pt --input ./guitar.wav --play --output ./piano_out.wav\n\n"
        "For live input instead of WAV:\n\n"
        "  python3 jetson_realtime.py --model ./model_pruned_scripted.pt --live\n\n"
        "Useful Jetson performance settings:\n\n"
        "  sudo nvpmodel -m 0\n"
        "  sudo jetson_clocks\n\n"
        "If sounddevice cannot find the right output, run:\n\n"
        "  python3 jetson_realtime.py --list-devices\n\n"
        "Then pass --output_device INDEX.\n",
        encoding="utf-8",
    )

    if not args.no_verify:
        verify_export(scripted_path, device)

    bundle = make_deploy_tar(output_dir, [scripted_path, compact_ckpt, metadata_path, output_dir / "model.py", output_dir / "jetson_realtime.py", readme])

    print("\nExport complete:")
    print(f"  TorchScript: {scripted_path}")
    print(f"  Compact checkpoint: {compact_ckpt}")
    print(f"  Deploy tarball: {bundle}")


if __name__ == "__main__":
    main()
