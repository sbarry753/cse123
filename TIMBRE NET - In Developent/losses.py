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
