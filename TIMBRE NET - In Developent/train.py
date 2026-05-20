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
    p.add_argument("--hf_artifact_weight", type=float, default=0.0)
    p.add_argument("--hf_artifact_start_hz", type=float, default=8000.0)
    p.add_argument("--hf_artifact_margin", type=float, default=0.0)
    p.add_argument("--hf_artifact_topk_frac", type=float, default=0.25)
    p.add_argument("--energy_weight_floor", type=float, default=0.1)
    p.add_argument("--energy_weight_ceiling", type=float, default=5.0)
    p.add_argument("--intended_log_mag_weight", type=float, default=0.2)
    p.add_argument("--mask_reg_weight", type=float, default=0.01)
    p.add_argument("--phase_delta_l2_weight", type=float, default=0.04)
    p.add_argument("--phase_delta_df_l1_weight", type=float, default=0.015)
    p.add_argument("--phase_delta_dt_l1_weight", type=float, default=0.01)
    p.add_argument("--phase_saturation_weight", type=float, default=0.1)
    p.add_argument("--phase_saturation_threshold", type=float, default=0.095)
    p.add_argument("--low_energy_spectral_weight", type=float, default=0.008)
    p.add_argument("--low_energy_spectral_quantile", type=float, default=0.25)
    p.add_argument("--low_energy_spectral_margin", type=float, default=0.05)
    p.add_argument("--low_energy_spectral_hf_boost", type=float, default=1.5)
    p.add_argument("--low_energy_sustain_only", action="store_true",)
    p.add_argument("--low_energy_onset_flux_std", type=float, default=1.5)
    p.add_argument("--low_energy_onset_pre_ms", type=float, default=5.0)
    p.add_argument("--low_energy_onset_post_ms", type=float, default=35.0)
    p.add_argument("--low_energy_band_low_weight", type=float, default=0.0)
    p.add_argument("--low_energy_band_low_mid_weight", type=float, default=0.5)
    p.add_argument("--low_energy_band_mid_weight", type=float, default=1.0)
    p.add_argument("--low_energy_band_high_weight", type=float, default=1.0)
    p.add_argument("--low_energy_low_note_threshold_hz", type=float, default=500.0)
    p.add_argument("--low_energy_low_note_ratio_threshold", type=float, default=0.45)
    p.add_argument("--low_energy_harmonic_protect", action="store_true")
    p.add_argument("--low_energy_harmonic_peak_margin", type=float, default=0.10)
    p.add_argument("--low_energy_harmonic_peak_prominence", type=float, default=0.20)
    p.add_argument("--sustain_shimmer_weight", type=float, default=0.03)
    p.add_argument("--sustain_shimmer_margin", type=float, default=0.05)
    p.add_argument("--sustain_shimmer_low_weight", type=float, default=0.0)
    p.add_argument("--sustain_shimmer_low_mid_weight", type=float, default=0.5)
    p.add_argument("--sustain_shimmer_mid_weight", type=float, default=1.0)
    p.add_argument("--sustain_shimmer_high_weight", type=float, default=1.0)
    p.add_argument("--render_shimmer_weight", type=float, default=0.02)
    p.add_argument("--render_shimmer_margin", type=float, default=0.05)
    p.add_argument("--render_shimmer_low_weight", type=float, default=0.0)
    p.add_argument("--render_shimmer_low_mid_weight", type=float, default=0.5)
    p.add_argument("--render_shimmer_mid_weight", type=float, default=1.0)
    p.add_argument("--render_shimmer_high_weight", type=float, default=1.25)
    p.add_argument("--phase_oracle_render_weight", type=float, default=0.005)
    p.add_argument("--phase_oracle_render_margin", type=float, default=0.0)
    p.add_argument("--phase_oracle_render_low_weight", type=float, default=0.0)
    p.add_argument("--phase_oracle_render_low_mid_weight", type=float, default=0.5)
    p.add_argument("--phase_oracle_render_mid_weight", type=float, default=1.0)
    p.add_argument("--phase_oracle_render_high_weight", type=float, default=1.0)
    p.add_argument("--interharmonic_sustain_weight", type=float, default=0.02)
    p.add_argument("--interharmonic_peak_prominence", type=float, default=0.20)
    p.add_argument("--interharmonic_peak_radius_bins", type=int, default=1)
    p.add_argument("--interharmonic_margin", type=float, default=0.05)
    p.add_argument("--interharmonic_low_weight", type=float, default=0.0)
    p.add_argument("--interharmonic_low_mid_weight", type=float, default=0.5)
    p.add_argument("--interharmonic_mid_weight", type=float, default=1.0)
    p.add_argument("--interharmonic_high_weight", type=float, default=0.75)
    p.add_argument("--high_energy_interharmonic_weight", type=float, default=0.0)
    p.add_argument("--high_energy_interharmonic_quantile", type=float, default=0.75)
    p.add_argument("--high_energy_interharmonic_margin", type=float, default=0.05)
    p.add_argument("--high_energy_interharmonic_peak_prominence", type=float, default=0.20)
    p.add_argument("--high_energy_interharmonic_peak_radius_bins", type=int, default=1)
    p.add_argument("--high_energy_interharmonic_low_weight", type=float, default=0.0)
    p.add_argument("--high_energy_interharmonic_low_mid_weight", type=float, default=0.0)
    p.add_argument("--high_energy_interharmonic_mid_weight", type=float, default=1.0)
    p.add_argument("--high_energy_interharmonic_high_weight", type=float, default=1.0)
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
        "spectral_hf_artifact",
        "weighted_spectral_mel",
        "weighted_spectral_convergence",
        "weighted_spectral_log_stft",
        "weighted_spectral_plain_log_stft",
        "weighted_spectral_hf_artifact",
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
        "low_energy_spectral",
        "weighted_low_energy_spectral",
        "sustain_shimmer",
        "weighted_sustain_shimmer",
        "render_shimmer",
        "weighted_render_shimmer",
        "phase_oracle_render",
        "weighted_phase_oracle_render",
        "interharmonic_sustain",
        "weighted_interharmonic_sustain",
        "high_energy_interharmonic",
        "weighted_high_energy_interharmonic",
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
        f"hf_art={metrics['weighted_spectral_hf_artifact']:.4f}({metrics['spectral_hf_artifact']:.4f}) "
        f"intended_log={metrics['weighted_intended_log_mag']:.4f}({metrics['intended_log_mag']:.4f}) "
        f"mask_reg={metrics['weighted_mask_reg']:.6f}({metrics['mask_reg']:.4f}) "
        f"phase_l2={metrics['weighted_phase_delta_l2']:.6f}({metrics['phase_delta_l2']:.4f}) "
        f"phase_df={metrics['weighted_phase_delta_df_l1']:.6f}({metrics['phase_delta_df_l1']:.4f}) "
        f"phase_dt={metrics['weighted_phase_delta_dt_l1']:.6f}({metrics['phase_delta_dt_l1']:.4f}) "
        f"phase_abs={metrics['phase_delta_abs_mean']:.4f}/{metrics['phase_delta_abs_max']:.4f} "
        f"phase_sat={metrics['weighted_phase_saturation']:.6f}({metrics['phase_delta_saturation_frac']:.4f}/{metrics['phase_saturation_excess']:.4f}) "
        f"low_e_spec={metrics['weighted_low_energy_spectral']:.6f}({metrics['low_energy_spectral']:.4f}) "
        f"shim={metrics['weighted_sustain_shimmer']:.6f}({metrics['sustain_shimmer']:.4f}) "
        f"render_shim={metrics['weighted_render_shimmer']:.6f}({metrics['render_shimmer']:.4f}) "
        f"phase_oracle={metrics['weighted_phase_oracle_render']:.6f}({metrics['phase_oracle_render']:.4f}) "
        f"interharm={metrics['weighted_interharmonic_sustain']:.6f}({metrics['interharmonic_sustain']:.4f}) "
        f"hi_e_interharm={metrics['weighted_high_energy_interharmonic']:.6f}({metrics['high_energy_interharmonic']:.4f}) "
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


def stft_bin_frequencies(freq_bins, n_fft, device, dtype):
    freqs = torch.arange(freq_bins, device=device, dtype=dtype)
    return freqs * (float(SAMPLE_RATE) / 2.0) / max(1, freq_bins - 1)


def onset_exclusion_mask(target_log_mag, args):
    time_bins = target_log_mag.shape[-1]
    if time_bins <= 1:
        return torch.zeros(target_log_mag.shape[0], 1, time_bins, device=target_log_mag.device, dtype=torch.bool)

    flux = F.relu(target_log_mag[..., 1:] - target_log_mag[..., :-1]).mean(dim=1)
    flux = F.pad(flux, (1, 0))
    flux_mean = flux.mean(dim=1, keepdim=True)
    flux_std = flux.std(dim=1, keepdim=True, unbiased=False)
    threshold = flux_mean + float(args.low_energy_onset_flux_std) * flux_std
    onset = flux > threshold

    pre_cols = int(float(args.low_energy_onset_pre_ms) * SAMPLE_RATE / 1000.0 / max(1, args.hop_size))
    post_cols = int(float(args.low_energy_onset_post_ms) * SAMPLE_RATE / 1000.0 / max(1, args.hop_size))
    if pre_cols <= 0 and post_cols <= 0:
        return onset.unsqueeze(1)

    excluded = torch.zeros_like(onset)
    for shift in range(-pre_cols, post_cols + 1):
        if shift < 0:
            excluded[:, :shift] |= onset[:, -shift:]
        elif shift > 0:
            excluded[:, shift:] |= onset[:, :-shift]
        else:
            excluded |= onset
    return excluded.unsqueeze(1)


def low_energy_band_weights(freqs, args):
    weights = torch.zeros_like(freqs)
    weights = torch.where(
        freqs < 500.0,
        torch.full_like(weights, float(args.low_energy_band_low_weight)),
        weights,
    )
    weights = torch.where(
        (freqs >= 500.0) & (freqs < 2000.0),
        torch.full_like(weights, float(args.low_energy_band_low_mid_weight)),
        weights,
    )
    weights = torch.where(
        (freqs >= 2000.0) & (freqs < 8000.0),
        torch.full_like(weights, float(args.low_energy_band_mid_weight)),
        weights,
    )
    weights = torch.where(
        freqs >= 8000.0,
        torch.full_like(weights, float(args.low_energy_band_high_weight)),
        weights,
    )
    return weights.view(1, -1, 1)


def artifact_band_weights(freqs, low_weight, low_mid_weight, mid_weight, high_weight):
    weights = torch.zeros_like(freqs)
    weights = torch.where(freqs < 500.0, torch.full_like(weights, float(low_weight)), weights)
    weights = torch.where(
        (freqs >= 500.0) & (freqs < 2000.0),
        torch.full_like(weights, float(low_mid_weight)),
        weights,
    )
    weights = torch.where(
        (freqs >= 2000.0) & (freqs < 8000.0),
        torch.full_like(weights, float(mid_weight)),
        weights,
    )
    weights = torch.where(freqs >= 8000.0, torch.full_like(weights, float(high_weight)), weights)
    return weights.view(1, -1, 1)


def dilate_frequency_mask(mask, radius_bins):
    radius_bins = max(0, int(radius_bins))
    if radius_bins == 0 or mask.shape[-2] <= 1:
        return mask
    dilated = mask.clone()
    for shift in range(1, radius_bins + 1):
        dilated[..., shift:, :] |= mask[..., :-shift, :]
        dilated[..., :-shift, :] |= mask[..., shift:, :]
    return dilated


def harmonic_region_mask(target_log_mag, prominence, radius_bins):
    if target_log_mag.shape[1] < 3:
        return torch.ones_like(target_log_mag, dtype=torch.bool)

    center = target_log_mag[:, 1:-1, :]
    left = target_log_mag[:, :-2, :]
    right = target_log_mag[:, 2:, :]
    peak = (center > left + float(prominence)) & (center > right + float(prominence))
    peak = F.pad(peak, (0, 0, 1, 1))
    return dilate_frequency_mask(peak, radius_bins)


def harmonic_protection_margin(target_log_mag, args):
    if not args.low_energy_harmonic_protect or target_log_mag.shape[1] < 3:
        return torch.zeros_like(target_log_mag)

    center = target_log_mag[:, 1:-1, :]
    left = target_log_mag[:, :-2, :]
    right = target_log_mag[:, 2:, :]
    prominence = float(args.low_energy_harmonic_peak_prominence)
    peak = (center > left + prominence) & (center > right + prominence)
    protected = F.pad(peak, (0, 0, 1, 1))
    return protected.to(target_log_mag.dtype) * float(args.low_energy_harmonic_peak_margin)


def low_energy_spectral_loss(model, pred, target, args):
    pred_spec = model._stft(pred)
    target_spec = model._stft(target)
    pred_mag = torch.abs(pred_spec)
    target_mag = torch.abs(target_spec)
    pred_log_mag = torch.log(torch.clamp(pred_mag, min=1.0e-5))
    target_log_mag = torch.log(torch.clamp(target_mag, min=1.0e-5))

    quantile = max(0.0, min(1.0, float(args.low_energy_spectral_quantile)))
    threshold = torch.quantile(target_mag.detach().flatten(1), quantile, dim=1).view(-1, 1, 1)
    low_energy_mask = target_mag.detach() <= threshold
    if not low_energy_mask.any():
        return pred.new_tensor(0.0)

    extra_margin = harmonic_protection_margin(target_log_mag.detach(), args)
    overprediction = F.relu(pred_log_mag - target_log_mag - args.low_energy_spectral_margin - extra_margin)
    weights = torch.ones_like(overprediction)
    freq_bins = overprediction.shape[-2]
    freqs = stft_bin_frequencies(freq_bins, args.n_fft, overprediction.device, overprediction.dtype)

    if args.low_energy_sustain_only:
        sustain_mask = ~onset_exclusion_mask(target_log_mag.detach(), args)
        low_energy_mask = low_energy_mask & sustain_mask
        weights = weights * low_energy_band_weights(freqs, args)
    elif args.low_energy_spectral_hf_boost != 1.0:
        hf_mask = freqs.view(1, -1, 1) >= float(args.hf_artifact_start_hz)
        weights = torch.where(hf_mask, weights * args.low_energy_spectral_hf_boost, weights)

    if not low_energy_mask.any():
        return pred.new_tensor(0.0)

    masked_overprediction = overprediction.masked_select(low_energy_mask)
    masked_weights = weights.masked_select(low_energy_mask)
    return (masked_overprediction * masked_weights).sum() / masked_weights.sum().clamp_min(1.0e-8)


def sustain_shimmer_loss(model, pred, target, args):
    pred_spec = model._stft(pred)
    target_spec = model._stft(target)
    pred_log_mag = torch.log(torch.clamp(torch.abs(pred_spec), min=1.0e-5))
    target_log_mag = torch.log(torch.clamp(torch.abs(target_spec), min=1.0e-5))
    if pred_log_mag.shape[-1] <= 1:
        return pred.new_tensor(0.0)

    pred_dt = torch.abs(pred_log_mag[..., 1:] - pred_log_mag[..., :-1])
    target_dt = torch.abs(target_log_mag[..., 1:] - target_log_mag[..., :-1])
    excess = F.relu(pred_dt - target_dt - float(args.sustain_shimmer_margin))

    sustain_mask = (~onset_exclusion_mask(target_log_mag.detach(), args))[..., 1:]
    freq_bins = excess.shape[-2]
    freqs = stft_bin_frequencies(freq_bins, args.n_fft, excess.device, excess.dtype)
    weights = artifact_band_weights(
        freqs,
        args.sustain_shimmer_low_weight,
        args.sustain_shimmer_low_mid_weight,
        args.sustain_shimmer_mid_weight,
        args.sustain_shimmer_high_weight,
    )
    mask = sustain_mask & (weights > 0.0)
    if not mask.any():
        return pred.new_tensor(0.0)

    masked_excess = excess.masked_select(mask)
    masked_weights = weights.expand_as(excess).masked_select(mask)
    return (masked_excess * masked_weights).sum() / masked_weights.sum().clamp_min(1.0e-8)


def render_shimmer_loss(model, features, params, pred, target, args):
    input_log_mag = features["input_log_mag"]
    intended_log_mag = input_log_mag * params["mask"] + params["residual"]

    pred_spec = model._stft(pred)
    target_spec = model._stft(target)
    rendered_log_mag = torch.log(torch.clamp(torch.abs(pred_spec), min=1.0e-5))
    target_log_mag = torch.log(torch.clamp(torch.abs(target_spec), min=1.0e-5))

    min_freq = min(intended_log_mag.shape[-2], rendered_log_mag.shape[-2], target_log_mag.shape[-2])
    min_time = min(intended_log_mag.shape[-1], rendered_log_mag.shape[-1], target_log_mag.shape[-1])
    intended_log_mag = intended_log_mag[..., :min_freq, :min_time]
    rendered_log_mag = rendered_log_mag[..., :min_freq, :min_time]
    target_log_mag = target_log_mag[..., :min_freq, :min_time]

    if rendered_log_mag.shape[-1] <= 1:
        return pred.new_tensor(0.0)

    intended_dt = torch.abs(intended_log_mag[..., 1:] - intended_log_mag[..., :-1])
    rendered_dt = torch.abs(rendered_log_mag[..., 1:] - rendered_log_mag[..., :-1])
    excess = F.relu(rendered_dt - intended_dt - float(args.render_shimmer_margin))

    sustain_mask = (~onset_exclusion_mask(target_log_mag.detach(), args))[..., 1:]
    freq_bins = excess.shape[-2]
    freqs = stft_bin_frequencies(freq_bins, args.n_fft, excess.device, excess.dtype)
    weights = artifact_band_weights(
        freqs,
        args.render_shimmer_low_weight,
        args.render_shimmer_low_mid_weight,
        args.render_shimmer_mid_weight,
        args.render_shimmer_high_weight,
    )

    mask = sustain_mask & (weights > 0.0)
    if not mask.any():
        return pred.new_tensor(0.0)

    masked_excess = excess.masked_select(mask)
    masked_weights = weights.expand_as(excess).masked_select(mask)
    return (masked_excess * masked_weights).sum() / masked_weights.sum().clamp_min(1.0e-8)


def phase_oracle_render_loss(model, features, params, target, args):
    input_log_mag = features["input_log_mag"]
    input_phase = features["input_phase"]
    phase_delta = params["phase_delta"]

    intended_log_mag = input_log_mag * params["mask"] + params["residual"]
    intended_mag = torch.exp(intended_log_mag).detach()

    target_spec = model._stft(target)
    target_phase = torch.angle(target_spec).detach()
    target_log_mag = torch.log(torch.clamp(torch.abs(target_spec), min=1.0e-5))

    min_freq = min(
        intended_mag.shape[-2],
        input_phase.shape[-2],
        phase_delta.shape[-2],
        target_phase.shape[-2],
        target_log_mag.shape[-2],
    )
    min_time = min(
        intended_mag.shape[-1],
        input_phase.shape[-1],
        phase_delta.shape[-1],
        target_phase.shape[-1],
        target_log_mag.shape[-1],
    )
    intended_mag = intended_mag[..., :min_freq, :min_time]
    input_phase = input_phase[..., :min_freq, :min_time]
    phase_delta = phase_delta[..., :min_freq, :min_time]
    target_phase = target_phase[..., :min_freq, :min_time]
    target_log_mag = target_log_mag[..., :min_freq, :min_time]

    phase_tcn_spec = torch.polar(intended_mag, input_phase + phase_delta)
    target_phase_spec = torch.polar(intended_mag, target_phase)
    phase_tcn_audio = model._istft(phase_tcn_spec, length=target.shape[-1])
    target_phase_audio = model._istft(target_phase_spec, length=target.shape[-1]).detach()

    phase_tcn_log_mag = torch.log(torch.clamp(torch.abs(model._stft(phase_tcn_audio)), min=1.0e-5))
    target_phase_log_mag = torch.log(torch.clamp(torch.abs(model._stft(target_phase_audio)), min=1.0e-5)).detach()

    min_freq = min(phase_tcn_log_mag.shape[-2], target_phase_log_mag.shape[-2], target_log_mag.shape[-2])
    min_time = min(phase_tcn_log_mag.shape[-1], target_phase_log_mag.shape[-1], target_log_mag.shape[-1])
    phase_tcn_log_mag = phase_tcn_log_mag[..., :min_freq, :min_time]
    target_phase_log_mag = target_phase_log_mag[..., :min_freq, :min_time]
    target_log_mag = target_log_mag[..., :min_freq, :min_time]

    diff = F.relu(
        torch.abs(phase_tcn_log_mag - target_phase_log_mag)
        - float(args.phase_oracle_render_margin)
    )
    sustain_mask = ~onset_exclusion_mask(target_log_mag.detach(), args)
    freq_bins = diff.shape[-2]
    freqs = stft_bin_frequencies(freq_bins, args.n_fft, diff.device, diff.dtype)
    weights = artifact_band_weights(
        freqs,
        args.phase_oracle_render_low_weight,
        args.phase_oracle_render_low_mid_weight,
        args.phase_oracle_render_mid_weight,
        args.phase_oracle_render_high_weight,
    )

    mask = sustain_mask & (weights > 0.0)
    if not mask.any():
        return target.new_tensor(0.0)

    masked_diff = diff.masked_select(mask)
    masked_weights = weights.expand_as(diff).masked_select(mask)
    return (masked_diff * masked_weights).sum() / masked_weights.sum().clamp_min(1.0e-8)


def interharmonic_sustain_loss(model, features, params, target, args):
    input_log_mag = features["input_log_mag"]
    intended_log_mag = input_log_mag * params["mask"] + params["residual"]
    target_spec = model._stft(target)
    target_log_mag = torch.log(torch.clamp(torch.abs(target_spec), min=1.0e-5))

    sustain_mask = ~onset_exclusion_mask(target_log_mag.detach(), args)
    harmonic_mask = harmonic_region_mask(
        target_log_mag.detach(),
        args.interharmonic_peak_prominence,
        args.interharmonic_peak_radius_bins,
    )
    interharmonic_mask = sustain_mask & (~harmonic_mask)

    overprediction = F.relu(intended_log_mag - target_log_mag - float(args.interharmonic_margin))
    freq_bins = overprediction.shape[-2]
    freqs = stft_bin_frequencies(freq_bins, args.n_fft, overprediction.device, overprediction.dtype)
    weights = artifact_band_weights(
        freqs,
        args.interharmonic_low_weight,
        args.interharmonic_low_mid_weight,
        args.interharmonic_mid_weight,
        args.interharmonic_high_weight,
    )
    mask = interharmonic_mask & (weights > 0.0)
    if not mask.any():
        return target.new_tensor(0.0)

    masked_overprediction = overprediction.masked_select(mask)
    masked_weights = weights.expand_as(overprediction).masked_select(mask)
    return (masked_overprediction * masked_weights).sum() / masked_weights.sum().clamp_min(1.0e-8)


def high_energy_interharmonic_sustain_loss(model, features, params, target, args):
    input_log_mag = features["input_log_mag"]
    intended_log_mag = input_log_mag * params["mask"] + params["residual"]
    target_spec = model._stft(target)
    target_mag = torch.abs(target_spec)
    target_log_mag = torch.log(torch.clamp(target_mag, min=1.0e-5))

    quantile = max(0.0, min(1.0, float(args.high_energy_interharmonic_quantile)))
    target_time_energy = target_mag.detach().mean(dim=-2, keepdim=True)
    threshold = torch.quantile(target_time_energy.flatten(1), quantile, dim=1).view(-1, 1, 1)
    high_energy_time_mask = target_time_energy >= threshold
    if not high_energy_time_mask.any():
        return target.new_tensor(0.0)

    sustain_mask = ~onset_exclusion_mask(target_log_mag.detach(), args)
    harmonic_mask = harmonic_region_mask(
        target_log_mag.detach(),
        args.high_energy_interharmonic_peak_prominence,
        args.high_energy_interharmonic_peak_radius_bins,
    )
    interharmonic_mask = sustain_mask & high_energy_time_mask & (~harmonic_mask)

    overprediction = F.relu(
        intended_log_mag - target_log_mag - float(args.high_energy_interharmonic_margin)
    )
    freq_bins = overprediction.shape[-2]
    freqs = stft_bin_frequencies(freq_bins, args.n_fft, overprediction.device, overprediction.dtype)
    weights = artifact_band_weights(
        freqs,
        args.high_energy_interharmonic_low_weight,
        args.high_energy_interharmonic_low_mid_weight,
        args.high_energy_interharmonic_mid_weight,
        args.high_energy_interharmonic_high_weight,
    )
    mask = interharmonic_mask & (weights > 0.0)
    if not mask.any():
        return target.new_tensor(0.0)

    masked_overprediction = overprediction.masked_select(mask)
    masked_weights = weights.expand_as(overprediction).masked_select(mask)
    return (masked_overprediction * masked_weights).sum() / masked_weights.sum().clamp_min(1.0e-8)


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
        loss_components = criterion.components(pred, piano_frames)
        
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
        low_energy_loss = low_energy_spectral_loss(model, pred, piano_frames, args)
        weighted_low_energy_loss = args.low_energy_spectral_weight * low_energy_loss
        loss = loss + weighted_low_energy_loss
        shimmer_loss = sustain_shimmer_loss(model, pred, piano_frames, args)
        weighted_shimmer_loss = args.sustain_shimmer_weight * shimmer_loss
        loss = loss + weighted_shimmer_loss
        render_shim_loss = render_shimmer_loss(model, features, params, pred, piano_frames, args)
        weighted_render_shim_loss = args.render_shimmer_weight * render_shim_loss
        loss = loss + weighted_render_shim_loss
        phase_oracle_loss = phase_oracle_render_loss(model, features, params, piano_frames, args)
        weighted_phase_oracle_loss = args.phase_oracle_render_weight * phase_oracle_loss
        loss = loss + weighted_phase_oracle_loss
        interharmonic_loss = interharmonic_sustain_loss(model, features, params, piano_frames, args)
        weighted_interharmonic_loss = args.interharmonic_sustain_weight * interharmonic_loss
        loss = loss + weighted_interharmonic_loss
        high_energy_interharmonic_loss = high_energy_interharmonic_sustain_loss(
            model, features, params, piano_frames, args
        )
        weighted_high_energy_interharmonic_loss = (
            args.high_energy_interharmonic_weight * high_energy_interharmonic_loss
        )
        loss = loss + weighted_high_energy_interharmonic_loss
        attack_components = attack_criterion.components(pred, piano_frames, source=guitar_frames)
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
        totals["low_energy_spectral"] += float(low_energy_loss.item())
        totals["weighted_low_energy_spectral"] += float(weighted_low_energy_loss.item())
        totals["sustain_shimmer"] += float(shimmer_loss.item())
        totals["weighted_sustain_shimmer"] += float(weighted_shimmer_loss.item())
        totals["render_shimmer"] += float(render_shim_loss.item())
        totals["weighted_render_shimmer"] += float(weighted_render_shim_loss.item())
        totals["phase_oracle_render"] += float(phase_oracle_loss.item())
        totals["weighted_phase_oracle_render"] += float(weighted_phase_oracle_loss.item())
        totals["interharmonic_sustain"] += float(interharmonic_loss.item())
        totals["weighted_interharmonic_sustain"] += float(weighted_interharmonic_loss.item())
        totals["high_energy_interharmonic"] += float(high_energy_interharmonic_loss.item())
        totals["weighted_high_energy_interharmonic"] += float(weighted_high_energy_interharmonic_loss.item())
        for key, value in phase_metrics.items():
            totals[key] += float(value.item())
        for key, value in attack_components.items():
            if key == "total":
                continue
            totals[key] += float(value.item())
        totals["residual_reg"] += float(residual_reg.item())
        for key, value in loss_components.items():
            if key != "total":
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
        loss_components = criterion.components(pred, piano_frames)
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
        low_energy_loss = low_energy_spectral_loss(model, pred, piano_frames, args)
        weighted_low_energy_loss = args.low_energy_spectral_weight * low_energy_loss
        loss = loss + weighted_low_energy_loss
        shimmer_loss = sustain_shimmer_loss(model, pred, piano_frames, args)
        weighted_shimmer_loss = args.sustain_shimmer_weight * shimmer_loss
        loss = loss + weighted_shimmer_loss
        render_shim_loss = render_shimmer_loss(model, features, params, pred, piano_frames, args)
        weighted_render_shim_loss = args.render_shimmer_weight * render_shim_loss
        loss = loss + weighted_render_shim_loss
        phase_oracle_loss = phase_oracle_render_loss(model, features, params, piano_frames, args)
        weighted_phase_oracle_loss = args.phase_oracle_render_weight * phase_oracle_loss
        loss = loss + weighted_phase_oracle_loss
        interharmonic_loss = interharmonic_sustain_loss(model, features, params, piano_frames, args)
        weighted_interharmonic_loss = args.interharmonic_sustain_weight * interharmonic_loss
        loss = loss + weighted_interharmonic_loss
        high_energy_interharmonic_loss = high_energy_interharmonic_sustain_loss(
            model, features, params, piano_frames, args
        )
        weighted_high_energy_interharmonic_loss = (
            args.high_energy_interharmonic_weight * high_energy_interharmonic_loss
        )
        loss = loss + weighted_high_energy_interharmonic_loss
        attack_components = attack_criterion.components(pred, piano_frames, source=guitar_frames)
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
        totals["low_energy_spectral"] += float(low_energy_loss.item())
        totals["weighted_low_energy_spectral"] += float(weighted_low_energy_loss.item())
        totals["sustain_shimmer"] += float(shimmer_loss.item())
        totals["weighted_sustain_shimmer"] += float(weighted_shimmer_loss.item())
        totals["render_shimmer"] += float(render_shim_loss.item())
        totals["weighted_render_shimmer"] += float(weighted_render_shim_loss.item())
        totals["phase_oracle_render"] += float(phase_oracle_loss.item())
        totals["weighted_phase_oracle_render"] += float(weighted_phase_oracle_loss.item())
        totals["interharmonic_sustain"] += float(interharmonic_loss.item())
        totals["weighted_interharmonic_sustain"] += float(weighted_interharmonic_loss.item())
        totals["high_energy_interharmonic"] += float(high_energy_interharmonic_loss.item())
        totals["weighted_high_energy_interharmonic"] += float(weighted_high_energy_interharmonic_loss.item())
        for key, value in phase_metrics.items():
            totals[key] += float(value.item())
        for key, value in attack_components.items():
            if key == "total":
                continue
            totals[key] += float(value.item())
        totals["residual_reg"] += float(residual_reg.item())
        for key, value in loss_components.items():
            if key != "total":
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
            "hf_artifact_weight": args.hf_artifact_weight,
            "intended_log_mag_weight": args.intended_log_mag_weight,
            "mask_reg_weight": args.mask_reg_weight,
            "phase_delta_l2_weight": args.phase_delta_l2_weight,
            "phase_delta_df_l1_weight": args.phase_delta_df_l1_weight,
            "phase_delta_dt_l1_weight": args.phase_delta_dt_l1_weight,
            "phase_saturation_weight": args.phase_saturation_weight,
            "phase_saturation_threshold": args.phase_saturation_threshold,
            "low_energy_spectral_weight": args.low_energy_spectral_weight,
            "low_energy_spectral_quantile": args.low_energy_spectral_quantile,
            "low_energy_spectral_margin": args.low_energy_spectral_margin,
            "low_energy_spectral_hf_boost": args.low_energy_spectral_hf_boost,
            "low_energy_sustain_only": args.low_energy_sustain_only,
            "low_energy_onset_flux_std": args.low_energy_onset_flux_std,
            "low_energy_onset_pre_ms": args.low_energy_onset_pre_ms,
            "low_energy_onset_post_ms": args.low_energy_onset_post_ms,
            "low_energy_band_low_weight": args.low_energy_band_low_weight,
            "low_energy_band_low_mid_weight": args.low_energy_band_low_mid_weight,
            "low_energy_band_mid_weight": args.low_energy_band_mid_weight,
            "low_energy_band_high_weight": args.low_energy_band_high_weight,
            "low_energy_low_note_threshold_hz": args.low_energy_low_note_threshold_hz,
            "low_energy_low_note_ratio_threshold": args.low_energy_low_note_ratio_threshold,
            "low_energy_harmonic_protect": args.low_energy_harmonic_protect,
            "low_energy_harmonic_peak_margin": args.low_energy_harmonic_peak_margin,
            "low_energy_harmonic_peak_prominence": args.low_energy_harmonic_peak_prominence,
            "sustain_shimmer_weight": args.sustain_shimmer_weight,
            "sustain_shimmer_margin": args.sustain_shimmer_margin,
            "sustain_shimmer_low_weight": args.sustain_shimmer_low_weight,
            "sustain_shimmer_low_mid_weight": args.sustain_shimmer_low_mid_weight,
            "sustain_shimmer_mid_weight": args.sustain_shimmer_mid_weight,
            "sustain_shimmer_high_weight": args.sustain_shimmer_high_weight,
            "render_shimmer_weight": args.render_shimmer_weight,
            "render_shimmer_margin": args.render_shimmer_margin,
            "render_shimmer_low_weight": args.render_shimmer_low_weight,
            "render_shimmer_low_mid_weight": args.render_shimmer_low_mid_weight,
            "render_shimmer_mid_weight": args.render_shimmer_mid_weight,
            "render_shimmer_high_weight": args.render_shimmer_high_weight,
            "phase_oracle_render_weight": args.phase_oracle_render_weight,
            "phase_oracle_render_margin": args.phase_oracle_render_margin,
            "phase_oracle_render_low_weight": args.phase_oracle_render_low_weight,
            "phase_oracle_render_low_mid_weight": args.phase_oracle_render_low_mid_weight,
            "phase_oracle_render_mid_weight": args.phase_oracle_render_mid_weight,
            "phase_oracle_render_high_weight": args.phase_oracle_render_high_weight,
            "interharmonic_sustain_weight": args.interharmonic_sustain_weight,
            "interharmonic_peak_prominence": args.interharmonic_peak_prominence,
            "interharmonic_peak_radius_bins": args.interharmonic_peak_radius_bins,
            "interharmonic_margin": args.interharmonic_margin,
            "interharmonic_low_weight": args.interharmonic_low_weight,
            "interharmonic_low_mid_weight": args.interharmonic_low_mid_weight,
            "interharmonic_mid_weight": args.interharmonic_mid_weight,
            "interharmonic_high_weight": args.interharmonic_high_weight,
            "high_energy_interharmonic_weight": args.high_energy_interharmonic_weight,
            "high_energy_interharmonic_quantile": args.high_energy_interharmonic_quantile,
            "high_energy_interharmonic_margin": args.high_energy_interharmonic_margin,
            "high_energy_interharmonic_peak_prominence": args.high_energy_interharmonic_peak_prominence,
            "high_energy_interharmonic_peak_radius_bins": args.high_energy_interharmonic_peak_radius_bins,
            "high_energy_interharmonic_low_weight": args.high_energy_interharmonic_low_weight,
            "high_energy_interharmonic_low_mid_weight": args.high_energy_interharmonic_low_mid_weight,
            "high_energy_interharmonic_mid_weight": args.high_energy_interharmonic_mid_weight,
            "high_energy_interharmonic_high_weight": args.high_energy_interharmonic_high_weight,
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
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        win_length=args.win_length,
        base_ch=args.base_ch,
        phase_tcn_ch=args.phase_tcn_ch,
        phase_tcn_layers=args.phase_tcn_layers,
        phase_max_delta=args.phase_max_delta,
        transient_max_gain=0.0
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
        hf_artifact_weight=args.hf_artifact_weight,
        hf_artifact_start_hz=args.hf_artifact_start_hz,
        hf_artifact_margin=args.hf_artifact_margin,
        hf_artifact_topk_frac=args.hf_artifact_topk_frac,
        energy_weight_floor=args.energy_weight_floor,
        energy_weight_ceiling=args.energy_weight_ceiling,
    ).to(device)
    attack_criterion = AttackLoss(
        sample_rate=SAMPLE_RATE,
        n_fft=args.n_fft,
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
        hf_artifact_start_hz=args.hf_artifact_start_hz,
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
