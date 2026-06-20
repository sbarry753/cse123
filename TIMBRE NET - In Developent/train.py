"""
Train polyphonic Guitar -> Piano timbre transfer

Usage:
  python train.py --data_dir ./data --epochs 100 --batch_size 16

  python train.py --data_dir overfit --output_dir checkpoints_teach_tcn1D_p2 \
  --split_manifest overfit/splits.json --win_length 1024 --frame_size 2048 \
  --phase_delta_l2_weight 0.01 --phase_max_delta 0.25 --waveform_weight 0.3 --onset_weight 0.8
"""

import os
import argparse
import shlex
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, N_FFT
from dataset import GuitarPianoDataset, load_split_manifest
from losses import CombinedLoss, AttackLoss


def serializable_training_args(args):
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
        and isinstance(value, (str, int, float, bool, type(None)))
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train polyphonic Guitar -> Piano model")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--split_manifest", type=str, default=None)
    p.add_argument("--eval_split", choices=["val", "test"], default="val")
    p.add_argument("--n_fft", type=int, default=N_FFT)
    p.add_argument("--win_length", type=int, default=FRAME_SIZE)
    p.add_argument("--hop_size", type=int, default=HOP_SIZE)
    p.add_argument("--frame_size", type=int, default=FRAME_SIZE)
    p.add_argument("--output_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--phase_tcn_ch", type=int, default=16)
    p.add_argument("--phase_tcn_layers", type=int, default=3)
    p.add_argument("--phase_max_delta", type=float, default=0.10)
    p.add_argument("--spectral_weight", type=float, default=1.0)
    p.add_argument("--waveform_weight", type=float, default=0.05)
    p.add_argument("--envelope_weight", type=float, default=0.4)
    p.add_argument("--onset_weight", type=float, default=1.0)
    p.add_argument("--spectral_convergence_weight", type=float, default=0.25)
    p.add_argument("--log_stft_weight", type=float, default=0.25)
    p.add_argument("--plain_log_stft_weight", type=float, default=0.1)
    p.add_argument("--energy_weight_floor", type=float, default=0.1)
    p.add_argument("--energy_weight_ceiling", type=float, default=5.0)
    p.add_argument("--intended_log_mag_weight", type=float, default=0.2)
    p.add_argument("--mask_reg_weight", type=float, default=0.01)
    p.add_argument("--phase_delta_l2_weight", type=float, default=0.04)
    p.add_argument("--phase_delta_df_l1_weight", type=float, default=0.015)
    p.add_argument("--phase_delta_dt_l1_weight", type=float, default=0.01)
    p.add_argument("--phase_saturation_weight", type=float, default=0.1)
    p.add_argument("--phase_saturation_threshold", type=float, default=0.095)
    p.add_argument("--attack_envelope_weight", type=float, default=0.0)
    p.add_argument("--attack_hf_over_weight", type=float, default=0.0)
    p.add_argument("--attack_hf_flux_weight", type=float, default=0.0)
    p.add_argument("--attack_contrast_logmag_weight", type=float, default=0.0)
    p.add_argument("--attack_flux_low_weight", type=float, default=0.0)
    p.add_argument("--attack_flux_mid_weight", type=float, default=0.0)
    p.add_argument("--attack_flux_high_weight", type=float, default=0.0)
    p.add_argument("--attack_flux_low_l1_weight", type=float, default=0.0)
    p.add_argument("--attack_flux_mid_l1_weight", type=float, default=0.0)
    p.add_argument("--attack_flux_high_l1_weight", type=float, default=0.0)
    p.add_argument("--attack_contrast_margin", type=float, default=0.0)
    p.add_argument("--attack_loss_ms", type=float, default=20.0)
    p.add_argument("--attack_gate_threshold", type=float, default=0.0075)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    if args.output_size is not None and (args.output_size <= 0 or args.output_size > args.frame_size):
        p.error("--output_size must satisfy 0 < output_size <= frame_size")
    return args


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def new_loss_totals():
    keys = [
        "total",
        "spectral",
        "waveform",
        "envelope",
        "onset",
        "spectral_mel",
        "spectral_convergence",
        "spectral_log_stft",
        "spectral_plain_log_stft",
        "weighted_spectral_mel",
        "weighted_spectral_convergence",
        "weighted_spectral_log_stft",
        "weighted_spectral_plain_log_stft",
        "weighted_spectral",
        "weighted_waveform",
        "weighted_envelope",
        "weighted_onset",
        "intended_log_mag",
        "weighted_intended_log_mag",
        "mask_reg",
        "weighted_mask_reg",
        "phase_delta_l2",
        "weighted_phase_delta_l2",
        "phase_delta_df_l1",
        "weighted_phase_delta_df_l1",
        "phase_delta_dt_l1",
        "weighted_phase_delta_dt_l1",
        "phase_saturation_excess",
        "weighted_phase_saturation",
        "phase_delta_abs_mean",
        "phase_delta_abs_max",
        "phase_delta_saturation_frac",
        "attack_envelope",
        "weighted_attack_envelope",
        "attack_hf_over",
        "weighted_attack_hf_over",
        "attack_hf_flux",
        "weighted_attack_hf_flux",
        "attack_teacher_piano_logmag_l1",
        "attack_teacher_guitar_logmag_l1",
        "attack_contrast_logmag",
        "weighted_attack_contrast_logmag",
        "attack_teacher_piano_flux_l1",
        "attack_teacher_guitar_flux_l1",
        "attack_contrast_flux",
        "attack_closer_to_piano_logmag_frac",
        "attack_closer_to_piano_flux_frac",
        "attack_flux_low_teacher_piano_l1",
        "attack_flux_low_teacher_guitar_l1",
        "attack_flux_low_l1",
        "weighted_attack_flux_low_l1",
        "attack_flux_low_contrast",
        "weighted_attack_flux_low_contrast",
        "attack_flux_low_closer_to_piano_frac",
        "attack_flux_mid_teacher_piano_l1",
        "attack_flux_mid_teacher_guitar_l1",
        "attack_flux_mid_l1",
        "weighted_attack_flux_mid_l1",
        "attack_flux_mid_contrast",
        "weighted_attack_flux_mid_contrast",
        "attack_flux_mid_closer_to_piano_frac",
        "attack_flux_high_teacher_piano_l1",
        "attack_flux_high_teacher_guitar_l1",
        "attack_flux_high_l1",
        "weighted_attack_flux_high_l1",
        "attack_flux_high_contrast",
        "weighted_attack_flux_high_contrast",
        "attack_flux_high_closer_to_piano_frac",
        "attack_gate_frac",
        "attack_onset_mean_ms",
        "attack_onset_std_ms",
        "residual_reg",
    ]
    return {key: 0.0 for key in keys}


def average_loss_totals(totals, n_batches):
    denom = max(1, n_batches)
    return {key: value / denom for key, value in totals.items()}


def should_log_loss_component(key):
    return key not in {"total", "spectral_hf_artifact", "weighted_spectral_hf_artifact"}


def format_loss_components(prefix, metrics):
    return (
        f"{prefix}: total={metrics['total']:.4f} "
        f"spec={metrics['weighted_spectral']:.4f}({metrics['spectral']:.4f}) "
        f"wave={metrics['weighted_waveform']:.4f}({metrics['waveform']:.4f}) "
        f"env={metrics['weighted_envelope']:.4f}({metrics['envelope']:.4f}) "
        f"onset={metrics['weighted_onset']:.4f}({metrics['onset']:.4f}) "
        f"mel={metrics['weighted_spectral_mel']:.4f}({metrics['spectral_mel']:.4f}) "
        f"sc={metrics['weighted_spectral_convergence']:.4f}({metrics['spectral_convergence']:.4f}) "
        f"log_stft={metrics['weighted_spectral_log_stft']:.4f}({metrics['spectral_log_stft']:.4f}) "
        f"plain_log={metrics['weighted_spectral_plain_log_stft']:.4f}({metrics['spectral_plain_log_stft']:.4f}) "
        f"intended_log={metrics['weighted_intended_log_mag']:.4f}({metrics['intended_log_mag']:.4f}) "
        f"mask_reg={metrics['weighted_mask_reg']:.6f}({metrics['mask_reg']:.4f}) "
        f"phase_l2={metrics['weighted_phase_delta_l2']:.6f}({metrics['phase_delta_l2']:.4f}) "
        f"phase_df={metrics['weighted_phase_delta_df_l1']:.6f}({metrics['phase_delta_df_l1']:.4f}) "
        f"phase_dt={metrics['weighted_phase_delta_dt_l1']:.6f}({metrics['phase_delta_dt_l1']:.4f}) "
        f"phase_abs={metrics['phase_delta_abs_mean']:.4f}/{metrics['phase_delta_abs_max']:.4f} "
        f"phase_sat={metrics['weighted_phase_saturation']:.6f}({metrics['phase_delta_saturation_frac']:.4f}/{metrics['phase_saturation_excess']:.4f}) "
        f"atk_env={metrics['weighted_attack_envelope']:.6f}({metrics['attack_envelope']:.4f}) "
        f"atk_hf_over={metrics['weighted_attack_hf_over']:.6f}({metrics['attack_hf_over']:.4f}) "
        f"atk_hf_flux={metrics['weighted_attack_hf_flux']:.6f}({metrics['attack_hf_flux']:.4f}) "
        f"atk_con_log={metrics['weighted_attack_contrast_logmag']:.6f}({metrics['attack_contrast_logmag']:.4f}) "
        f"atk_flux_bands="
        f"{metrics['weighted_attack_flux_low_contrast']:.6f}({metrics['attack_flux_low_contrast']:.4f})/"
        f"{metrics['weighted_attack_flux_mid_contrast']:.6f}({metrics['attack_flux_mid_contrast']:.4f})/"
        f"{metrics['weighted_attack_flux_high_contrast']:.6f}({metrics['attack_flux_high_contrast']:.4f}) "
        f"atk_flux_l1_bands="
        f"{metrics['weighted_attack_flux_low_l1']:.6f}({metrics['attack_flux_low_l1']:.4f})/"
        f"{metrics['weighted_attack_flux_mid_l1']:.6f}({metrics['attack_flux_mid_l1']:.4f})/"
        f"{metrics['weighted_attack_flux_high_l1']:.6f}({metrics['attack_flux_high_l1']:.4f}) "
        f"atk_close={metrics['attack_closer_to_piano_logmag_frac']:.4f}/{metrics['attack_closer_to_piano_flux_frac']:.4f} "
        f"atk_band_close="
        f"{metrics['attack_flux_low_closer_to_piano_frac']:.4f}/"
        f"{metrics['attack_flux_mid_closer_to_piano_frac']:.4f}/"
        f"{metrics['attack_flux_high_closer_to_piano_frac']:.4f} "
        f"atk_gate={metrics['attack_gate_frac']:.4f} "
        f"atk_pos={metrics['attack_onset_mean_ms']:.2f}±{metrics['attack_onset_std_ms']:.2f}ms "
        f"reg={metrics['residual_reg']:.6f}"
    )


def intended_log_mag_loss(model, features, params, piano_frames):
    input_log_mag = features["input_log_mag"]
    intended_log_mag = input_log_mag * params["mask"] + params["residual"]
    piano_spec = model._stft(piano_frames)
    piano_mag = torch.abs(piano_spec)
    piano_log_mag = torch.log(torch.clamp(piano_mag, min=1.0e-5))
    return F.l1_loss(intended_log_mag, piano_log_mag)


def prediction_aligned_frames(pred, piano_frames, guitar_frames):
    # Allows to keep a large context size but only output a single hop at a time
    pred_len = pred.shape[-1]
    target_len = piano_frames.shape[-1]
    if pred_len > target_len:
        raise ValueError(f"Prediction length {pred_len} exceeds target length {target_len}")
    if pred_len == target_len:
        return piano_frames, guitar_frames
    return piano_frames[..., -pred_len:], guitar_frames[..., -pred_len:]


def phase_delta_metrics(params, args):
    phase_delta = params["phase_delta"]
    phase_delta_abs = phase_delta.abs()
    phase_delta_l2 = phase_delta.square().mean()
    weighted_phase_delta_l2 = args.phase_delta_l2_weight * phase_delta_l2
    if phase_delta.shape[-2] > 1:
        phase_delta_df_l1 = torch.abs(phase_delta[..., 1:, :] - phase_delta[..., :-1, :]).mean()
    else:
        phase_delta_df_l1 = phase_delta.new_tensor(0.0)
    if phase_delta.shape[-1] > 1:
        phase_delta_dt_l1 = torch.abs(phase_delta[..., 1:] - phase_delta[..., :-1]).mean()
    else:
        phase_delta_dt_l1 = phase_delta.new_tensor(0.0)
    weighted_phase_delta_df_l1 = args.phase_delta_df_l1_weight * phase_delta_df_l1
    weighted_phase_delta_dt_l1 = args.phase_delta_dt_l1_weight * phase_delta_dt_l1
    saturation_frac = (phase_delta_abs >= args.phase_saturation_threshold).float().mean()
    phase_saturation_excess = F.relu(
        phase_delta_abs - float(args.phase_saturation_threshold)
    ).mean()
    weighted_phase_saturation = args.phase_saturation_weight * phase_saturation_excess
    return {
        "phase_delta_l2": phase_delta_l2,
        "weighted_phase_delta_l2": weighted_phase_delta_l2,
        "phase_delta_df_l1": phase_delta_df_l1,
        "weighted_phase_delta_df_l1": weighted_phase_delta_df_l1,
        "phase_delta_dt_l1": phase_delta_dt_l1,
        "weighted_phase_delta_dt_l1": weighted_phase_delta_dt_l1,
        "phase_saturation_excess": phase_saturation_excess,
        "weighted_phase_saturation": weighted_phase_saturation,
        "phase_delta_abs_mean": phase_delta_abs.mean(),
        "phase_delta_abs_max": phase_delta_abs.max(),
        "phase_delta_saturation_frac": saturation_frac,
    }


def train_epoch(model, loader, optimizer, criterion, attack_criterion, device, args, epoch=0):
    model.train()
    totals = new_loss_totals()
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Train", leave=False):
        guitar_frames = guitar_frames.to(device)
        piano_frames = piano_frames.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred, features, params = model(guitar_frames)
        target_frames, source_frames = prediction_aligned_frames(pred, piano_frames, guitar_frames)
        loss_components = criterion.components(pred, target_frames)
        
        loss = loss_components["total"]
        log_mag_loss = intended_log_mag_loss(model, features, params, piano_frames)
        weighted_log_mag_loss = args.intended_log_mag_weight * log_mag_loss

        loss = loss + weighted_log_mag_loss

        mask_reg = (params["mask"] - 1.0).abs().mean()
        weighted_mask_reg = args.mask_reg_weight * mask_reg
        loss = loss + weighted_mask_reg
        phase_metrics = phase_delta_metrics(params, args)
        loss = loss + phase_metrics["weighted_phase_delta_l2"]
        loss = loss + phase_metrics["weighted_phase_delta_df_l1"]
        loss = loss + phase_metrics["weighted_phase_delta_dt_l1"]
        loss = loss + phase_metrics["weighted_phase_saturation"]
        attack_components = attack_criterion.components(pred, target_frames, source=source_frames)
        loss = loss + attack_components["total"]

        residual_reg = loss.new_tensor(0.0)

        # Mild regularization on very large residual outputs
        if "residual" in params:
            residual_reg = 1e-4 * params["residual"].abs().mean()
            loss = loss + residual_reg

        if torch.isnan(loss) or torch.isinf(loss):
            print("  ⚠ NaN/Inf loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        totals["total"] += float(loss.item())
        totals["intended_log_mag"] += float(log_mag_loss.item())
        totals["weighted_intended_log_mag"] += float(weighted_log_mag_loss.item())
        totals["mask_reg"] += float(mask_reg.item())
        totals["weighted_mask_reg"] += float(weighted_mask_reg.item())

        for key, value in phase_metrics.items():
            totals[key] += float(value.item())

        for key, value in attack_components.items():
            if key == "total":
                continue
            totals[key] += float(value.item())

        totals["residual_reg"] += float(residual_reg.item())

        for key, value in loss_components.items():
            if should_log_loss_component(key):
                totals[key] += float(value.item())
        n_batches += 1

    return average_loss_totals(totals, n_batches)


@torch.no_grad()
def val_epoch(model, loader, criterion, attack_criterion, device, args):
    if loader is None:
        return None

    model.eval()
    totals = new_loss_totals()
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Val  ", leave=False):
        guitar_frames = guitar_frames.to(device)
        piano_frames = piano_frames.to(device)

        pred, features, params = model(guitar_frames)
        target_frames, source_frames = prediction_aligned_frames(pred, piano_frames, guitar_frames)
        loss_components = criterion.components(pred, target_frames)
        loss = loss_components["total"]
        log_mag_loss = intended_log_mag_loss(model, features, params, piano_frames)
        weighted_log_mag_loss = args.intended_log_mag_weight * log_mag_loss
        loss = loss + weighted_log_mag_loss
        mask_reg = (params["mask"] - 1.0).abs().mean()
        weighted_mask_reg = args.mask_reg_weight * mask_reg
        loss = loss + weighted_mask_reg
        phase_metrics = phase_delta_metrics(params, args)
        loss = loss + phase_metrics["weighted_phase_delta_l2"]
        loss = loss + phase_metrics["weighted_phase_delta_df_l1"]
        loss = loss + phase_metrics["weighted_phase_delta_dt_l1"]
        loss = loss + phase_metrics["weighted_phase_saturation"]
        attack_components = attack_criterion.components(pred, target_frames, source=source_frames)
        loss = loss + attack_components["total"]
        residual_reg = loss.new_tensor(0.0)

        if "residual" in params:
            residual_reg = 1e-4 * params["residual"].abs().mean()
            loss = loss + residual_reg

        totals["total"] += float(loss.item())
        totals["intended_log_mag"] += float(log_mag_loss.item())
        totals["weighted_intended_log_mag"] += float(weighted_log_mag_loss.item())
        totals["mask_reg"] += float(mask_reg.item())
        totals["weighted_mask_reg"] += float(weighted_mask_reg.item())
        for key, value in phase_metrics.items():
            totals[key] += float(value.item())
        for key, value in attack_components.items():
            if key == "total":
                continue
            totals[key] += float(value.item())
        totals["residual_reg"] += float(residual_reg.item())
        for key, value in loss_components.items():
            if should_log_loss_component(key):
                totals[key] += float(value.item())
        n_batches += 1

    return average_loss_totals(totals, n_batches)


def checkpoint_state(payload, model=None):
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    state = {key: value for key, value in state.items() if key != "window"}
    if model is None:
        return state

    current = model.state_dict()
    compatible = {}
    for key, value in state.items():
        if key in current and current[key].shape != value.shape:
            print(
                f"Ignoring checkpoint tensor with incompatible shape: "
                f"{key} checkpoint={tuple(value.shape)} model={tuple(current[key].shape)}"
            )
            continue
        compatible[key] = value
    return compatible


def save_checkpoint(model, optimizer, epoch, val_loss, path, args):
    training_args = serializable_training_args(args)
    torch.save(
        {
            "checkpoint_format_version": 2,
            "epoch": epoch,
            "val_loss": val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_args": training_args,
            "training_command": " ".join(shlex.quote(part) for part in sys.argv),
            "frame_size": args.frame_size,
            "output_size": args.output_size,
            "hop_size": args.hop_size,
            "n_fft": args.n_fft,
            "win_length": args.win_length,
            "base_ch": args.base_ch,
            "phase_tcn_ch": args.phase_tcn_ch,
            "phase_tcn_layers": args.phase_tcn_layers,
            "phase_max_delta": args.phase_max_delta,
            "spectral_weight": args.spectral_weight,
            "waveform_weight": args.waveform_weight,
            "envelope_weight": args.envelope_weight,
            "onset_weight": args.onset_weight,
            "spectral_convergence_weight": args.spectral_convergence_weight,
            "log_stft_weight": args.log_stft_weight,
            "plain_log_stft_weight": args.plain_log_stft_weight,
            "intended_log_mag_weight": args.intended_log_mag_weight,
            "mask_reg_weight": args.mask_reg_weight,
            "phase_delta_l2_weight": args.phase_delta_l2_weight,
            "phase_delta_df_l1_weight": args.phase_delta_df_l1_weight,
            "phase_delta_dt_l1_weight": args.phase_delta_dt_l1_weight,
            "phase_saturation_weight": args.phase_saturation_weight,
            "phase_saturation_threshold": args.phase_saturation_threshold,
            "attack_envelope_weight": args.attack_envelope_weight,
            "attack_hf_over_weight": args.attack_hf_over_weight,
            "attack_hf_flux_weight": args.attack_hf_flux_weight,
            "attack_contrast_logmag_weight": args.attack_contrast_logmag_weight,
            "attack_flux_low_weight": args.attack_flux_low_weight,
            "attack_flux_mid_weight": args.attack_flux_mid_weight,
            "attack_flux_high_weight": args.attack_flux_high_weight,
            "attack_flux_low_l1_weight": args.attack_flux_low_l1_weight,
            "attack_flux_mid_l1_weight": args.attack_flux_mid_l1_weight,
            "attack_flux_high_l1_weight": args.attack_flux_high_l1_weight,
            "attack_contrast_margin": args.attack_contrast_margin,
            "attack_loss_ms": args.attack_loss_ms,
            "attack_gate_threshold": args.attack_gate_threshold,
        },
        path,
    )


def plot_loss_curves(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train", color="steelblue")
    plt.plot(val_losses, label="Val", color="coral")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Polyphonic Guitar->Piano Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close()
    print(f"  Loss curve saved to {output_dir}/loss_curves.png")


def export_torchscript(model, output_dir, frame_size):
    model.eval()
    model.cpu()
    dummy = torch.randn(1, frame_size)
    try:
        scripted = torch.jit.trace(model, dummy, strict=False)
        path = os.path.join(output_dir, "model_scripted.pt")
        scripted.save(path)
        print(f"  TorchScript model saved -> {path}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")
        torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pt"))

def make_loader(
    data_dir: str,
    stems: list[str],
    batch_size: int,
    frame_size: int,
    hop_size: int,
    augment: bool,
    shuffle: bool,
):
    dataset = GuitarPianoDataset(
        data_dir=data_dir,
        stems=stems,
        sample_rate=SAMPLE_RATE,
        frame_size=frame_size,
        hop_size=hop_size,
        augment=augment,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=shuffle and len(dataset) >= batch_size,
    )

def make_dataloaders(args):
    splits = load_split_manifest(args.data_dir, args.split_manifest)
    if splits is None:
        raise ValueError(
            "Create a split manifest first with data_splits.py, or pass --split_manifest."
        )
    eval_stems = splits[args.eval_split]
    if not eval_stems:
        raise ValueError(f"Split manifest has no stems for eval split: {args.eval_split}")

    train_loader = make_loader(
        args.data_dir,
        splits["train"],
        args.batch_size,
        args.frame_size,
        args.hop_size,
        augment=False,
        shuffle=True,
    )
    eval_loader = make_loader(
        args.data_dir,
        eval_stems,
        args.batch_size * 2,
        args.frame_size,
        args.hop_size,
        augment=False,
        shuffle=False,
    )
    return train_loader, eval_loader


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    train_loader, val_loader = make_dataloaders(
        args
    )

    model = DDSPGuitarToPiano(
        hidden_size=args.hidden_size,
        sample_rate=SAMPLE_RATE,
        frame_size=args.frame_size,
        output_size=args.output_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        win_length=args.win_length,
        base_ch=args.base_ch,
        phase_tcn_ch=args.phase_tcn_ch,
        phase_tcn_layers=args.phase_tcn_layers,
        phase_max_delta=args.phase_max_delta,
        transient_max_gain=0.2
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = CombinedLoss(
        spectral_weight=args.spectral_weight,
        waveform_weight=args.waveform_weight,
        envelope_weight=args.envelope_weight,
        onset_weight=args.onset_weight,
        spectral_convergence_weight=args.spectral_convergence_weight,
        log_stft_weight=args.log_stft_weight,
        plain_log_stft_weight=args.plain_log_stft_weight,
        hf_artifact_weight=0.0,
        energy_weight_floor=args.energy_weight_floor,
        energy_weight_ceiling=args.energy_weight_ceiling,
    ).to(device)
    attack_n_fft = min(args.n_fft, int(args.output_size or args.frame_size))
    
    attack_criterion = AttackLoss(
        sample_rate=SAMPLE_RATE,
        n_fft=attack_n_fft,
        hop_size=args.hop_size,
        attack_loss_ms=args.attack_loss_ms,
        attack_envelope_weight=args.attack_envelope_weight,
        attack_hf_over_weight=args.attack_hf_over_weight,
        attack_hf_flux_weight=args.attack_hf_flux_weight,
        attack_contrast_logmag_weight=args.attack_contrast_logmag_weight,
        attack_flux_low_weight=args.attack_flux_low_weight,
        attack_flux_mid_weight=args.attack_flux_mid_weight,
        attack_flux_high_weight=args.attack_flux_high_weight,
        attack_flux_low_l1_weight=args.attack_flux_low_l1_weight,
        attack_flux_mid_l1_weight=args.attack_flux_mid_l1_weight,
        attack_flux_high_l1_weight=args.attack_flux_high_l1_weight,
        attack_contrast_margin=args.attack_contrast_margin,
        hf_artifact_start_hz=8000.0,
        onset_gate_threshold=args.attack_gate_threshold,
    ).to(device)

    warmup_epochs = 5
    start_epoch = 0
    best_val = float("inf")
    train_losses, val_losses = [], []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        load_result = model.load_state_dict(checkpoint_state(ckpt, model), strict=False)
        if load_result.missing_keys:
            print(f"Resume missing model keys initialized from defaults: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Resume ignored unexpected model keys: {load_result.unexpected_keys}")
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["val_loss"]
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val:.4f}")

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            warmup_lr = args.lr * float(epoch + 1) / float(warmup_epochs)
            set_lr(optimizer, warmup_lr)

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, attack_criterion, device, args, epoch=epoch
        )
        val_metrics = val_epoch(model, val_loader, criterion, attack_criterion, device, args)

        if epoch >= warmup_epochs:
            scheduler.step()

        if val_metrics is None:
            val_metrics = train_metrics

        train_loss = train_metrics["total"]
        val_loss = val_metrics["total"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
        )
        print("  " + format_loss_components("train", train_metrics))
        print()
        print("  " + format_loss_components("val", val_metrics))

        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pt")
        save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path, args)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, best_path, args)
            print(f"   New best model saved -> {best_path}")
        print()

    print("\nExporting for real-time inference...")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(checkpoint_state(best_ckpt, model), strict=False)
    export_torchscript(model, args.output_dir, args.frame_size)

    plot_loss_curves(train_losses, val_losses, args.output_dir)
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
