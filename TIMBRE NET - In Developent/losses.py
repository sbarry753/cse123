"""
Losses for polyphonic direct timbre transfer

Emphasis:
- multi-scale spectral matching
- waveform stability
- onset / transient matching for pick -> hammer behavior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleSpectralLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=(256, 512, 1024, 2048),
        hop_fractions=0.25,
        n_mels=80,
        sample_rate=48000,
        mel_weight=1.0,
        spectral_convergence_weight=0.25,
        log_stft_weight=0.25,
        plain_log_stft_weight=0.1,
        hf_artifact_weight=0.05,
        hf_artifact_start_hz=8000.0,
        hf_artifact_margin=0.0,
        hf_artifact_topk_frac=0.25,
        energy_weight_floor=0.1,
        energy_weight_ceiling=5.0,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_frac = hop_fractions
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mel_weight = mel_weight
        self.spectral_convergence_weight = spectral_convergence_weight
        self.log_stft_weight = log_stft_weight
        self.plain_log_stft_weight = plain_log_stft_weight
        self.hf_artifact_weight = hf_artifact_weight
        self.hf_artifact_start_hz = hf_artifact_start_hz
        self.hf_artifact_margin = hf_artifact_margin
        self.hf_artifact_topk_frac = hf_artifact_topk_frac
        self.energy_weight_floor = energy_weight_floor
        self.energy_weight_ceiling = energy_weight_ceiling

        for fft_size in fft_sizes:
            fb = self._mel_filterbank(n_mels, fft_size, sample_rate)
            self.register_buffer(f"mel_fb_{fft_size}", fb)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(pred, target)["total"]

    def components(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        mel_loss = pred.new_tensor(0.0)
        spectral_convergence_loss = pred.new_tensor(0.0)
        log_stft_loss = pred.new_tensor(0.0)
        plain_log_stft_loss = pred.new_tensor(0.0)
        hf_artifact_loss = pred.new_tensor(0.0)

        for fft_size in self.fft_sizes:
            hop_size = max(1, int(fft_size * self.hop_frac))
            window = torch.hann_window(fft_size, device=pred.device)
            mel_fb = getattr(self, f"mel_fb_{fft_size}")

            pred_mag = self._stft_mag(pred, fft_size, hop_size, window)
            target_mag = self._stft_mag(target, fft_size, hop_size, window)

            pred_mel = self._log_mel_from_power(pred_mag.square(), mel_fb)
            target_mel = self._log_mel_from_power(target_mag.square(), mel_fb)

            mel_loss = mel_loss + F.l1_loss(pred_mel, target_mel)
            pred_mag_flat = pred_mag.flatten(1)
            target_mag_flat = target_mag.flatten(1)
            spectral_convergence_loss = spectral_convergence_loss + (
                torch.linalg.vector_norm(pred_mag_flat - target_mag_flat, dim=1)
                / torch.linalg.vector_norm(target_mag_flat, dim=1).clamp_min(1.0e-8)
            ).mean()

            pred_log_mag = torch.log(torch.clamp(pred_mag, min=1.0e-5))
            target_log_mag = torch.log(torch.clamp(target_mag, min=1.0e-5))
            denom = target_mag.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
            energy_weight = torch.clamp(
                target_mag / denom,
                min=self.energy_weight_floor,
                max=self.energy_weight_ceiling,
            )
            log_stft_loss = log_stft_loss + (
                energy_weight * torch.abs(pred_log_mag - target_log_mag)
            ).mean()
            plain_log_stft_loss = plain_log_stft_loss + F.l1_loss(pred_log_mag, target_log_mag)
            hf_artifact_loss = hf_artifact_loss + self._hf_artifact_loss(
                pred_log_mag,
                target_log_mag,
                fft_size,
            )

        denom = float(len(self.fft_sizes))
        mel_loss = mel_loss / denom
        spectral_convergence_loss = spectral_convergence_loss / denom
        log_stft_loss = log_stft_loss / denom
        plain_log_stft_loss = plain_log_stft_loss / denom
        hf_artifact_loss = hf_artifact_loss / denom

        total_loss = (
            self.mel_weight * mel_loss
            + self.spectral_convergence_weight * spectral_convergence_loss
            + self.log_stft_weight * log_stft_loss
            + self.plain_log_stft_weight * plain_log_stft_loss
            + self.hf_artifact_weight * hf_artifact_loss
        )

        return {
            "mel": mel_loss,
            "spectral_convergence": spectral_convergence_loss,
            "log_stft": log_stft_loss,
            "plain_log_stft": plain_log_stft_loss,
            "hf_artifact": hf_artifact_loss,
            "total": total_loss,
        }

    def _hf_artifact_loss(self, pred_log_mag, target_log_mag, fft_size):
        if self.hf_artifact_weight <= 0.0:
            return pred_log_mag.new_tensor(0.0)

        freq_bins = pred_log_mag.shape[-2]
        bin_hz = torch.arange(
            freq_bins,
            device=pred_log_mag.device,
            dtype=pred_log_mag.dtype,
        ) * (self.sample_rate / float(fft_size))
        high_freq_mask = (bin_hz >= self.hf_artifact_start_hz).view(1, freq_bins, 1)
        overprediction = F.relu(pred_log_mag - target_log_mag - self.hf_artifact_margin)
        scores = overprediction.masked_select(high_freq_mask.expand_as(overprediction))
        if scores.numel() == 0:
            return pred_log_mag.new_tensor(0.0)

        topk_frac = min(max(float(self.hf_artifact_topk_frac), 0.0), 1.0)
        if topk_frac <= 0.0:
            return pred_log_mag.new_tensor(0.0)
        k = max(1, int(torch.ceil(scores.new_tensor(scores.numel() * topk_frac)).item()))
        return torch.topk(scores, k).values.mean()

    def _stft_mag(self, audio, fft_size, hop_size, window):
        audio = audio.float()
        window = window.float()

        pad = fft_size // 2
        audio_padded = F.pad(audio, (pad, pad))

        stft = torch.stft(
            audio_padded,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window=window,
            return_complex=True,
        )
        return torch.abs(stft)

    def _log_mel_spec(self, audio, fft_size, hop_size, window, mel_fb):
        mag = self._stft_mag(audio, fft_size, hop_size, window)
        return self._log_mel_from_power(mag.square(), mel_fb)

    def _log_mel_from_power(self, power, mel_fb):
        mel = torch.einsum("mf,bft->bmt", mel_fb, power)
        return torch.log(mel + 1e-7)

    def _mel_filterbank(self, n_mels, n_fft, sample_rate):
        freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        mel_lo = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
        mel_hi = 2595.0 * np.log10(1.0 + (sample_rate / 2) / 700.0)
        mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)

        f = freqs[np.newaxis, :]
        lo = hz_pts[:-2, np.newaxis]
        c = hz_pts[1:-1, np.newaxis]
        hi = hz_pts[2:, np.newaxis]

        rising = np.clip((f - lo) / (c - lo + 1e-8), 0.0, 1.0)
        falling = np.clip((hi - f) / (hi - c + 1e-8), 0.0, 1.0)
        fb = np.where(f <= c, rising, falling).astype(np.float32)
        return torch.from_numpy(fb)


class OnsetLoss(nn.Module):
    """
    Matches transient shape by comparing first differences
    and short-window onset energy.
    """

    def __init__(self, diff_weight=1.0, envelope_weight=1.0):
        super().__init__()
        self.diff_weight = diff_weight
        self.envelope_weight = envelope_weight

    def _smooth_abs(self, x: torch.Tensor, window: int = 64) -> torch.Tensor:
        env = F.avg_pool1d(
            x.abs().unsqueeze(1),
            kernel_size=window,
            stride=max(1, window // 4),
            padding=window // 2,
        ).squeeze(1)
        return env

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        diff_loss = F.l1_loss(pred_diff, target_diff)

        pred_env = self._smooth_abs(pred)
        target_env = self._smooth_abs(target)
        env_diff = pred_env[:, 1:] - pred_env[:, :-1]
        target_env_diff = target_env[:, 1:] - target_env[:, :-1]
        env_loss = F.l1_loss(env_diff, target_env_diff)

        return self.diff_weight * diff_loss + self.envelope_weight * env_loss


class AttackLoss(nn.Module):
    """
    Onset-gated attack losses for reducing pick-like transients.
    """

    def __init__(
        self,
        sample_rate=48000,
        n_fft=1024,
        hop_size=256,
        attack_loss_ms=20.0,
        attack_envelope_weight=0.0,
        attack_hf_over_weight=0.0,
        attack_hf_flux_weight=0.0,
        attack_contrast_logmag_weight=0.0,
        attack_flux_low_weight=0.0,
        attack_flux_mid_weight=0.0,
        attack_flux_high_weight=0.0,
        attack_flux_low_l1_weight=0.0,
        attack_flux_mid_l1_weight=0.0,
        attack_flux_high_l1_weight=0.0,
        attack_contrast_margin=0.0,
        hf_artifact_start_hz=8000.0,
        onset_gate_threshold=0.0075,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.attack_loss_ms = attack_loss_ms
        self.attack_envelope_weight = attack_envelope_weight
        self.attack_hf_over_weight = attack_hf_over_weight
        self.attack_hf_flux_weight = attack_hf_flux_weight
        self.attack_contrast_logmag_weight = attack_contrast_logmag_weight
        self.attack_flux_low_weight = attack_flux_low_weight
        self.attack_flux_mid_weight = attack_flux_mid_weight
        self.attack_flux_high_weight = attack_flux_high_weight
        self.attack_flux_low_l1_weight = attack_flux_low_l1_weight
        self.attack_flux_mid_l1_weight = attack_flux_mid_l1_weight
        self.attack_flux_high_l1_weight = attack_flux_high_l1_weight
        self.attack_contrast_margin = attack_contrast_margin
        self.hf_artifact_start_hz = hf_artifact_start_hz
        self.onset_gate_threshold = onset_gate_threshold
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(pred, target)["total"]

    def components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        gate, onset_positions = self._onset_gate(target)
        gate_frac = gate.float().mean()
        zero = pred.new_tensor(0.0)

        if not gate.any():
            return self._zero_components(zero, gate_frac)

        pred_gated = pred[gate]
        target_gated = target[gate]
        source_gated = source[gate] if source is not None else None
        gated_onset_ms = onset_positions[gate].float() * (1000.0 / float(self.sample_rate))
        attack_envelope = self._attack_envelope_loss(pred_gated, target_gated)
        attack_hf_over, attack_hf_flux = self._attack_hf_losses(pred_gated, target_gated)
        contrast = self._attack_contrastive_losses(pred_gated, target_gated, source_gated)

        weighted_attack_envelope = self.attack_envelope_weight * attack_envelope
        weighted_attack_hf_over = self.attack_hf_over_weight * attack_hf_over
        weighted_attack_hf_flux = self.attack_hf_flux_weight * attack_hf_flux
        weighted_attack_contrast_logmag = (
            self.attack_contrast_logmag_weight * contrast["attack_contrast_logmag"]
        )
        weighted_attack_flux_low_contrast = (
            self.attack_flux_low_weight * contrast["attack_flux_low_contrast"]
        )
        weighted_attack_flux_mid_contrast = (
            self.attack_flux_mid_weight * contrast["attack_flux_mid_contrast"]
        )
        weighted_attack_flux_high_contrast = (
            self.attack_flux_high_weight * contrast["attack_flux_high_contrast"]
        )
        weighted_attack_flux_low_l1 = self.attack_flux_low_l1_weight * contrast["attack_flux_low_l1"]
        weighted_attack_flux_mid_l1 = self.attack_flux_mid_l1_weight * contrast["attack_flux_mid_l1"]
        weighted_attack_flux_high_l1 = self.attack_flux_high_l1_weight * contrast["attack_flux_high_l1"]
        total = (
            weighted_attack_envelope
            + weighted_attack_hf_over
            + weighted_attack_hf_flux
            + weighted_attack_contrast_logmag
            + weighted_attack_flux_low_contrast
            + weighted_attack_flux_mid_contrast
            + weighted_attack_flux_high_contrast
            + weighted_attack_flux_low_l1
            + weighted_attack_flux_mid_l1
            + weighted_attack_flux_high_l1
        )

        return {
            "attack_envelope": attack_envelope,
            "weighted_attack_envelope": weighted_attack_envelope,
            "attack_hf_over": attack_hf_over,
            "weighted_attack_hf_over": weighted_attack_hf_over,
            "attack_hf_flux": attack_hf_flux,
            "weighted_attack_hf_flux": weighted_attack_hf_flux,
            "weighted_attack_contrast_logmag": weighted_attack_contrast_logmag,
            "weighted_attack_flux_low_contrast": weighted_attack_flux_low_contrast,
            "weighted_attack_flux_mid_contrast": weighted_attack_flux_mid_contrast,
            "weighted_attack_flux_high_contrast": weighted_attack_flux_high_contrast,
            "weighted_attack_flux_low_l1": weighted_attack_flux_low_l1,
            "weighted_attack_flux_mid_l1": weighted_attack_flux_mid_l1,
            "weighted_attack_flux_high_l1": weighted_attack_flux_high_l1,
            "attack_gate_frac": gate_frac,
            "attack_onset_mean_ms": gated_onset_ms.mean(),
            "attack_onset_std_ms": gated_onset_ms.std(unbiased=False),
            "total": total,
            **contrast,
        }

    def _zero_components(self, zero: torch.Tensor, gate_frac: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
                "attack_envelope": zero,
                "weighted_attack_envelope": zero,
                "attack_hf_over": zero,
                "weighted_attack_hf_over": zero,
                "attack_hf_flux": zero,
                "weighted_attack_hf_flux": zero,
                "attack_teacher_piano_logmag_l1": zero,
                "attack_teacher_guitar_logmag_l1": zero,
                "attack_contrast_logmag": zero,
                "weighted_attack_contrast_logmag": zero,
                "attack_teacher_piano_flux_l1": zero,
                "attack_teacher_guitar_flux_l1": zero,
                "attack_contrast_flux": zero,
                "attack_closer_to_piano_logmag_frac": zero,
                "attack_closer_to_piano_flux_frac": zero,
                "attack_flux_low_teacher_piano_l1": zero,
                "attack_flux_low_teacher_guitar_l1": zero,
                "attack_flux_low_l1": zero,
                "weighted_attack_flux_low_l1": zero,
                "attack_flux_low_contrast": zero,
                "weighted_attack_flux_low_contrast": zero,
                "attack_flux_low_closer_to_piano_frac": zero,
                "attack_flux_mid_teacher_piano_l1": zero,
                "attack_flux_mid_teacher_guitar_l1": zero,
                "attack_flux_mid_l1": zero,
                "weighted_attack_flux_mid_l1": zero,
                "attack_flux_mid_contrast": zero,
                "weighted_attack_flux_mid_contrast": zero,
                "attack_flux_mid_closer_to_piano_frac": zero,
                "attack_flux_high_teacher_piano_l1": zero,
                "attack_flux_high_teacher_guitar_l1": zero,
                "attack_flux_high_l1": zero,
                "weighted_attack_flux_high_l1": zero,
                "attack_flux_high_contrast": zero,
                "weighted_attack_flux_high_contrast": zero,
                "attack_flux_high_closer_to_piano_frac": zero,
                "attack_gate_frac": gate_frac,
                "attack_onset_mean_ms": zero,
                "attack_onset_std_ms": zero,
                "total": zero,
            }

    def _attack_samples(self, n_samples: int) -> int:
        attack_samples = int(round(self.attack_loss_ms * self.sample_rate / 1000.0))
        return min(attack_samples, n_samples)

    def _onset_gate(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attack_samples = self._attack_samples(target.shape[-1])
        if attack_samples <= 1:
            gate = torch.zeros(target.shape[0], device=target.device, dtype=torch.bool)
            positions = torch.zeros(target.shape[0], device=target.device, dtype=torch.long)
            return gate, positions

        window = min(64, max(1, attack_samples // 4))
        envelope = F.avg_pool1d(
            target.abs().unsqueeze(1),
            kernel_size=window,
            stride=1,
            padding=window // 2,
        ).squeeze(1)
        envelope = envelope[:, :target.shape[-1]]
        attack_envelope = envelope[:, :attack_samples]
        positive_slope = F.relu(attack_envelope[:, 1:] - attack_envelope[:, :-1])
        max_slope, max_idx = positive_slope.max(dim=1)
        return max_slope > self.onset_gate_threshold, max_idx + 1

    def _attack_envelope_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        attack_samples = self._attack_samples(pred.shape[-1])
        if attack_samples <= 0:
            return pred.new_tensor(0.0)

        pred_env = pred[..., :attack_samples].abs()
        target_env = target[..., :attack_samples].abs()
        return F.l1_loss(pred_env, target_env)

    def _attack_hf_losses(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_spec = self._stft(pred)
        target_spec = self._stft(target)
        pred_log_mag = torch.log(torch.clamp(torch.abs(pred_spec), min=1.0e-5)).unsqueeze(1)
        target_log_mag = torch.log(torch.clamp(torch.abs(target_spec), min=1.0e-5)).unsqueeze(1)

        freq_bins = pred_log_mag.shape[-2]
        frames = pred_log_mag.shape[-1]
        bin_hz = torch.arange(freq_bins, device=pred_log_mag.device, dtype=pred_log_mag.dtype)
        bin_hz = bin_hz * (self.sample_rate / float(self.n_fft))
        hf_mask = (bin_hz >= self.hf_artifact_start_hz).view(1, 1, freq_bins, 1)

        attack_end = self._attack_samples(pred.shape[-1])
        frame_centers = torch.arange(frames, device=pred_log_mag.device) * int(self.hop_size)
        attack_time = (frame_centers < attack_end).view(1, 1, 1, frames)
        attack_mask = hf_mask & attack_time

        if attack_mask.any():
            attack_hf_over = F.relu(pred_log_mag - target_log_mag).masked_select(
                attack_mask.expand_as(pred_log_mag)
            ).mean()
        else:
            attack_hf_over = pred_log_mag.new_tensor(0.0)

        pred_flux = F.relu(pred_log_mag[..., 1:] - pred_log_mag[..., :-1])
        target_flux = F.relu(target_log_mag[..., 1:] - target_log_mag[..., :-1])
        flux_frames = pred_flux.shape[-1]
        flux_centers = torch.arange(flux_frames, device=pred_log_mag.device) * int(self.hop_size)
        flux_centers = flux_centers + int(self.hop_size)
        attack_flux_time = (flux_centers < attack_end).view(1, 1, 1, flux_frames)
        attack_flux_mask = hf_mask & attack_flux_time

        if attack_flux_mask.any():
            attack_hf_flux = torch.abs(pred_flux - target_flux).masked_select(
                attack_flux_mask.expand_as(pred_flux)
            ).mean()
        else:
            attack_hf_flux = pred_log_mag.new_tensor(0.0)

        return attack_hf_over, attack_hf_flux

    def _attack_contrastive_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        zero = pred.new_tensor(0.0)
        if source is None:
            return {
                "attack_teacher_piano_logmag_l1": zero,
                "attack_teacher_guitar_logmag_l1": zero,
                "attack_contrast_logmag": zero,
                "attack_teacher_piano_flux_l1": zero,
                "attack_teacher_guitar_flux_l1": zero,
                "attack_contrast_flux": zero,
                "attack_closer_to_piano_logmag_frac": zero,
                "attack_closer_to_piano_flux_frac": zero,
                **self._zero_banded_flux_components(zero),
            }

        pred_log_mag = self._log_mag(pred)
        target_log_mag = self._log_mag(target)
        source_log_mag = self._log_mag(source)

        frames = pred_log_mag.shape[-1]
        attack_end = self._attack_samples(pred.shape[-1])
        frame_centers = torch.arange(frames, device=pred_log_mag.device) * int(self.hop_size)
        attack_time = (frame_centers < attack_end).view(1, 1, frames)
        if not attack_time.any():
            return {
                "attack_teacher_piano_logmag_l1": zero,
                "attack_teacher_guitar_logmag_l1": zero,
                "attack_contrast_logmag": zero,
                "attack_teacher_piano_flux_l1": zero,
                "attack_teacher_guitar_flux_l1": zero,
                "attack_contrast_flux": zero,
                "attack_closer_to_piano_logmag_frac": zero,
                "attack_closer_to_piano_flux_frac": zero,
                **self._zero_banded_flux_components(zero),
            }

        pred_target_log_dist = self._masked_example_l1(pred_log_mag, target_log_mag, attack_time)
        pred_source_log_dist = self._masked_example_l1(pred_log_mag, source_log_mag, attack_time)
        attack_contrast_logmag = F.relu(
            self.attack_contrast_margin + pred_target_log_dist - pred_source_log_dist
        ).mean()

        pred_flux = F.relu(pred_log_mag[..., 1:] - pred_log_mag[..., :-1])
        target_flux = F.relu(target_log_mag[..., 1:] - target_log_mag[..., :-1])
        source_flux = F.relu(source_log_mag[..., 1:] - source_log_mag[..., :-1])
        flux_frames = pred_flux.shape[-1]
        flux_centers = torch.arange(flux_frames, device=pred_log_mag.device) * int(self.hop_size)
        flux_centers = flux_centers + int(self.hop_size)
        attack_flux_time = (flux_centers < attack_end).view(1, 1, flux_frames)

        if attack_flux_time.any():
            pred_target_flux_dist = self._masked_example_l1(pred_flux, target_flux, attack_flux_time)
            pred_source_flux_dist = self._masked_example_l1(pred_flux, source_flux, attack_flux_time)
            attack_contrast_flux = F.relu(
                self.attack_contrast_margin + pred_target_flux_dist - pred_source_flux_dist
            ).mean()
            closer_flux = (pred_target_flux_dist < pred_source_flux_dist).float().mean()
            banded_flux = self._banded_flux_contrast_components(
                pred_flux,
                target_flux,
                source_flux,
                attack_flux_time,
            )
        else:
            pred_target_flux_dist = pred.new_zeros(pred.shape[0])
            pred_source_flux_dist = pred.new_zeros(pred.shape[0])
            attack_contrast_flux = zero
            closer_flux = zero
            banded_flux = self._zero_banded_flux_components(zero)

        return {
            "attack_teacher_piano_logmag_l1": pred_target_log_dist.mean(),
            "attack_teacher_guitar_logmag_l1": pred_source_log_dist.mean(),
            "attack_contrast_logmag": attack_contrast_logmag,
            "attack_teacher_piano_flux_l1": pred_target_flux_dist.mean(),
            "attack_teacher_guitar_flux_l1": pred_source_flux_dist.mean(),
            "attack_contrast_flux": attack_contrast_flux,
            "attack_closer_to_piano_logmag_frac": (pred_target_log_dist < pred_source_log_dist).float().mean(),
            "attack_closer_to_piano_flux_frac": closer_flux,
            **banded_flux,
        }

    def _zero_banded_flux_components(self, zero: torch.Tensor) -> dict[str, torch.Tensor]:
        components = {}
        for name in ("low", "mid", "high"):
            components[f"attack_flux_{name}_teacher_piano_l1"] = zero
            components[f"attack_flux_{name}_teacher_guitar_l1"] = zero
            components[f"attack_flux_{name}_l1"] = zero
            components[f"attack_flux_{name}_contrast"] = zero
            components[f"attack_flux_{name}_closer_to_piano_frac"] = zero
        return components

    def _banded_flux_contrast_components(
        self,
        pred_flux: torch.Tensor,
        target_flux: torch.Tensor,
        source_flux: torch.Tensor,
        attack_flux_time: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        freq_bins = pred_flux.shape[-2]
        bin_hz = torch.arange(freq_bins, device=pred_flux.device, dtype=pred_flux.dtype)
        bin_hz = bin_hz * (self.sample_rate / float(self.n_fft))

        components = {}
        for name, low_hz, high_hz in (
            ("low", 0.0, 2000.0),
            ("mid", 2000.0, 8000.0),
            ("high", 8000.0, None),
        ):
            if high_hz is None:
                freq_mask = bin_hz >= low_hz
            else:
                freq_mask = (bin_hz >= low_hz) & (bin_hz < high_hz)
            band_time_mask = freq_mask.view(1, -1, 1) & attack_flux_time
            if band_time_mask.any():
                target_dist = self._masked_example_l1(pred_flux, target_flux, band_time_mask)
                source_dist = self._masked_example_l1(pred_flux, source_flux, band_time_mask)
                contrast = F.relu(self.attack_contrast_margin + target_dist - source_dist).mean()
                closer = (target_dist < source_dist).float().mean()
            else:
                target_dist = pred_flux.new_zeros(pred_flux.shape[0])
                source_dist = pred_flux.new_zeros(pred_flux.shape[0])
                contrast = pred_flux.new_tensor(0.0)
                closer = pred_flux.new_tensor(0.0)

            components[f"attack_flux_{name}_teacher_piano_l1"] = target_dist.mean()
            components[f"attack_flux_{name}_teacher_guitar_l1"] = source_dist.mean()
            components[f"attack_flux_{name}_l1"] = target_dist.mean()
            components[f"attack_flux_{name}_contrast"] = contrast
            components[f"attack_flux_{name}_closer_to_piano_frac"] = closer

        return components

    def _log_mag(self, audio: torch.Tensor) -> torch.Tensor:
        spec = self._stft(audio)
        return torch.log(torch.clamp(torch.abs(spec), min=1.0e-5))

    def _masked_example_l1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        abs_error = torch.abs(pred - target)
        masked = abs_error.masked_select(mask.expand_as(abs_error))
        return masked.view(pred.shape[0], -1).mean(dim=1)

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        window = self.window.to(device=audio.device, dtype=audio.dtype)
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )


class CombinedLoss(nn.Module):
    def __init__(
        self,
        spectral_weight=1.0,
        waveform_weight=0.25,
        envelope_weight=0.10,
        onset_weight=0.35,
        spectral_convergence_weight=0.25,
        log_stft_weight=0.25,
        plain_log_stft_weight=0.1,
        hf_artifact_weight=0.05,
        hf_artifact_start_hz=8000.0,
        hf_artifact_margin=0.0,
        hf_artifact_topk_frac=0.25,
        energy_weight_floor=0.1,
        energy_weight_ceiling=5.0,
    ):
        super().__init__()
        self.spectral_loss = MultiScaleSpectralLoss(
            spectral_convergence_weight=spectral_convergence_weight,
            log_stft_weight=log_stft_weight,
            plain_log_stft_weight=plain_log_stft_weight,
            hf_artifact_weight=hf_artifact_weight,
            hf_artifact_start_hz=hf_artifact_start_hz,
            hf_artifact_margin=hf_artifact_margin,
            hf_artifact_topk_frac=hf_artifact_topk_frac,
            energy_weight_floor=energy_weight_floor,
            energy_weight_ceiling=energy_weight_ceiling,
        )
        self.spectral_weight = spectral_weight
        self.waveform_weight = waveform_weight
        self.envelope_weight = envelope_weight
        self.onset_weight = onset_weight
        self.onset_loss = OnsetLoss()

    def _smooth_rms(self, audio, window=128):
        audio_sq = audio ** 2
        rms = F.avg_pool1d(
            audio_sq.unsqueeze(1),
            kernel_size=window,
            stride=max(1, window // 2),
            padding=window // 4,
        ).squeeze(1)
        return torch.sqrt(rms + 1e-8)

    def components(self, pred, target):
        spectral_components = self.spectral_loss.components(pred, target)
        spec_loss = spectral_components["total"]
        wave_loss = F.l1_loss(pred, target)

        pred_rms = self._smooth_rms(pred)
        target_rms = self._smooth_rms(target)
        env_loss = F.l1_loss(pred_rms, target_rms)

        onset_loss = self.onset_loss(pred, target)

        weighted_spec = self.spectral_weight * spec_loss
        weighted_wave = self.waveform_weight * wave_loss
        weighted_env = self.envelope_weight * env_loss
        weighted_onset = self.onset_weight * onset_loss
        weighted_spectral_mel = (
            self.spectral_weight
            * self.spectral_loss.mel_weight
            * spectral_components["mel"]
        )
        weighted_spectral_convergence = (
            self.spectral_weight
            * self.spectral_loss.spectral_convergence_weight
            * spectral_components["spectral_convergence"]
        )
        weighted_spectral_log_stft = (
            self.spectral_weight
            * self.spectral_loss.log_stft_weight
            * spectral_components["log_stft"]
        )
        weighted_spectral_plain_log_stft = (
            self.spectral_weight
            * self.spectral_loss.plain_log_stft_weight
            * spectral_components["plain_log_stft"]
        )
        weighted_spectral_hf_artifact = (
            self.spectral_weight
            * self.spectral_loss.hf_artifact_weight
            * spectral_components["hf_artifact"]
        )

        return {
            "spectral": spec_loss,
            "spectral_mel": spectral_components["mel"],
            "spectral_convergence": spectral_components["spectral_convergence"],
            "spectral_log_stft": spectral_components["log_stft"],
            "spectral_plain_log_stft": spectral_components["plain_log_stft"],
            "spectral_hf_artifact": spectral_components["hf_artifact"],
            "weighted_spectral_mel": weighted_spectral_mel,
            "weighted_spectral_convergence": weighted_spectral_convergence,
            "weighted_spectral_log_stft": weighted_spectral_log_stft,
            "weighted_spectral_plain_log_stft": weighted_spectral_plain_log_stft,
            "weighted_spectral_hf_artifact": weighted_spectral_hf_artifact,
            "waveform": wave_loss,
            "envelope": env_loss,
            "onset": onset_loss,
            "weighted_spectral": weighted_spec,
            "weighted_waveform": weighted_wave,
            "weighted_envelope": weighted_env,
            "weighted_onset": weighted_onset,
            "total": weighted_spec + weighted_wave + weighted_env + weighted_onset,
        }

    def forward(self, pred, target):
        return self.components(pred, target)["total"]
