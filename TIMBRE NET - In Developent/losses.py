"""
losses.py — Spectral Reconstruction Losses

Multi-scale spectral loss is the standard for audio synthesis models.
We compare the model output to the target piano audio in the spectral
domain, which is perceptually meaningful and avoids phase alignment issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleSpectralLoss(nn.Module):
    """
    Computes L1 loss on log-mel spectrograms at multiple FFT sizes.
    This is the standard DDSP training objective — it's perceptually
    meaningful and doesn't require phase alignment between prediction
    and target.
    """

    def __init__(
        self,
        fft_sizes    = (64, 128, 256, 512, 1024, 2048),
        hop_fractions = 0.25,     # hop = fft_size * this
        n_mels       = 64,
        sample_rate  = 44100,
    ):
        super().__init__()
        self.fft_sizes  = fft_sizes
        self.hop_frac   = hop_fractions
        self.n_mels     = n_mels
        self.sample_rate = sample_rate

        # Precompute mel filterbanks for each FFT size
        self.mel_fbs = nn.ParameterDict()
        for fft_size in fft_sizes:
            fb = self._mel_filterbank(n_mels, fft_size, sample_rate)
            self.register_buffer(f'mel_fb_{fft_size}', fb)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (batch, n_samples)
        target: (batch, n_samples)
        returns: scalar loss
        """
        total_loss = 0.0

        for fft_size in self.fft_sizes:
            hop_size = max(1, int(fft_size * self.hop_frac))
            window   = torch.hann_window(fft_size, device=pred.device)

            mel_fb = getattr(self, f'mel_fb_{fft_size}')

            pred_spec   = self._log_mel_spec(pred,   fft_size, hop_size, window, mel_fb)
            target_spec = self._log_mel_spec(target, fft_size, hop_size, window, mel_fb)

            # L1 on log-mel (perceptual)
            loss = F.l1_loss(pred_spec, target_spec)
            total_loss = total_loss + loss

        return total_loss / len(self.fft_sizes)

    def _log_mel_spec(self, audio, fft_size, hop_size, window, mel_fb):
        """Compute log-mel spectrogram."""
        # torch.stft does NOT support float16 — always run in float32
        audio = audio.float()
        window = window.float()

        # Pad audio to fit STFT
        pad = fft_size // 2
        audio_padded = F.pad(audio, (pad, pad))

        # STFT
        stft = torch.stft(
            audio_padded,
            n_fft       = fft_size,
            hop_length  = hop_size,
            win_length  = fft_size,
            window      = window,
            return_complex = True,
        )
        power = stft.real ** 2 + stft.imag ** 2   # (B, F, T)

        # Mel filterbank: (B, F, T) → (B, n_mels, T)
        # mel_fb: (n_mels, F)
        mel = torch.einsum('mf,bft->bmt', mel_fb, power)

        return torch.log(mel + 1e-7)

    def _mel_filterbank(self, n_mels, n_fft, sample_rate):
        freqs   = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        mel_lo  = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
        mel_hi  = 2595.0 * np.log10(1.0 + (sample_rate / 2) / 700.0)
        mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
        hz_pts  = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)

        # Vectorised triangle filters — no Python loops
        f      = freqs[np.newaxis, :]           # (1, F)
        lo     = hz_pts[:-2, np.newaxis]        # (n_mels, 1)
        c      = hz_pts[1:-1, np.newaxis]       # (n_mels, 1)
        hi     = hz_pts[2:,   np.newaxis]       # (n_mels, 1)
        rising  = np.clip((f - lo) / (c  - lo + 1e-8), 0.0, 1.0)
        falling = np.clip((hi - f) / (hi - c  + 1e-8), 0.0, 1.0)
        fb = np.where(f <= c, rising, falling).astype(np.float32)
        return torch.from_numpy(fb)


class CombinedLoss(nn.Module):
    """Spectral + time-domain amplitude envelope loss."""

    def __init__(self, spectral_weight=1.0, envelope_weight=0.1):
        super().__init__()
        self.spectral_loss    = MultiScaleSpectralLoss()
        self.spectral_weight  = spectral_weight
        self.envelope_weight  = envelope_weight

    def forward(self, pred, target):
        spec_loss = self.spectral_loss(pred, target)

        # Loudness envelope loss (smooth RMS over 64-sample windows)
        pred_rms   = self._smooth_rms(pred)
        target_rms = self._smooth_rms(target)
        env_loss   = F.l1_loss(pred_rms, target_rms)

        return self.spectral_weight * spec_loss + self.envelope_weight * env_loss

    def _smooth_rms(self, audio, window=64):
        audio_sq = audio ** 2
        # 1D average pooling
        rms = F.avg_pool1d(
            audio_sq.unsqueeze(1),
            kernel_size = window,
            stride      = window // 2,
            padding     = window // 4,
        ).squeeze(1)
        return torch.sqrt(rms + 1e-8)