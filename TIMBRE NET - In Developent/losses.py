"""
Losses for polyphonic direct timbre transfer

Updated emphasis:
- multi-scale spectral matching
- exact-bin linear STFT matching for upper harmonics
- mel spectral matching for broad timbre
- stronger high-frequency weighting
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
        fft_sizes=(256, 512, 1024, 2048, 4096),
        hop_fractions=0.25,
        n_mels=96,
        sample_rate=48000,
        hf_emphasis=4.0,
        mel_mix=0.40,
        lin_mix=0.60,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_frac = hop_fractions
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hf_emphasis = hf_emphasis
        self.mel_mix = mel_mix
        self.lin_mix = lin_mix

        for fft_size in fft_sizes:
            fb = self._mel_filterbank(n_mels, fft_size, sample_rate)
            self.register_buffer(f"mel_fb_{fft_size}", fb)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0

        for fft_size in self.fft_sizes:
            hop_size = max(1, int(fft_size * self.hop_frac))
            window = torch.hann_window(fft_size, device=pred.device, dtype=pred.dtype)
            mel_fb = getattr(self, f"mel_fb_{fft_size}").to(device=pred.device, dtype=pred.dtype)

            pred_stft = torch.stft(
                F.pad(pred.float(), (fft_size // 2, fft_size // 2)),
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=fft_size,
                window=window,
                return_complex=True,
            )
            target_stft = torch.stft(
                F.pad(target.float(), (fft_size // 2, fft_size // 2)),
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=fft_size,
                window=window,
                return_complex=True,
            )

            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)

            pred_log_mag = torch.log(pred_mag + 1e-7)
            target_log_mag = torch.log(target_mag + 1e-7)

            n_freq = pred_log_mag.shape[1]
            freq_weights = torch.linspace(
                1.0,
                self.hf_emphasis,
                n_freq,
                device=pred.device,
                dtype=pred.dtype,
            ).view(1, n_freq, 1)

            # exact-bin spectral match: helps upper harmonics a lot
            lin_loss = F.l1_loss(pred_log_mag * freq_weights, target_log_mag * freq_weights)

            pred_power = pred_stft.real ** 2 + pred_stft.imag ** 2
            target_power = target_stft.real ** 2 + target_stft.imag ** 2

            pred_mel = torch.einsum("mf,bft->bmt", mel_fb, pred_power)
            target_mel = torch.einsum("mf,bft->bmt", mel_fb, target_power)

            pred_log_mel = torch.log(pred_mel + 1e-7)
            target_log_mel = torch.log(target_mel + 1e-7)

            n_mels = pred_log_mel.shape[1]
            mel_weights = torch.linspace(
                1.0,
                self.hf_emphasis,
                n_mels,
                device=pred.device,
                dtype=pred.dtype,
            ).view(1, n_mels, 1)

            mel_loss = F.l1_loss(pred_log_mel * mel_weights, target_log_mel * mel_weights)

            total_loss = total_loss + self.lin_mix * lin_loss + self.mel_mix * mel_loss

        return total_loss / len(self.fft_sizes)

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
        waveform_weight=0.20,
        envelope_weight=0.10,
        onset_weight=0.30,
    ):
        super().__init__()
        self.spectral_loss = MultiScaleSpectralLoss()
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

    def forward(self, pred, target):
        spec_loss = self.spectral_loss(pred, target)
        wave_loss = F.l1_loss(pred, target)

        pred_rms = self._smooth_rms(pred)
        target_rms = self._smooth_rms(target)
        env_loss = F.l1_loss(pred_rms, target_rms)

        onset_loss = self.onset_loss(pred, target)

        return (
            self.spectral_weight * spec_loss
            + self.waveform_weight * wave_loss
            + self.envelope_weight * env_loss
            + self.onset_weight * onset_loss
        )