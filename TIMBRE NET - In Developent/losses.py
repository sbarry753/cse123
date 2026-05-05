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
        fft_sizes = (64, 128, 256, 512, 1024, 2048),
        # fft_sizes    = (64, 128, 256, 512),
        hop_fractions = 0.25,     # hop = fft_size * this
        n_mels: int       = 64,
        sample_rate: int  = 48000,
        eps: float = 1e-7
    ):
        super().__init__()
        self.fft_sizes  = fft_sizes
        self.hop_frac   = hop_fractions
        self.n_mels     = n_mels
        self.sample_rate = sample_rate
        self.eps = eps

        # Precompute mel filterbanks and Hann window for each FFT size
        for fft_size in fft_sizes:
            fb = self._mel_filterbank(n_mels, fft_size, sample_rate)
            window = torch.hann_window(fft_size)
            self.register_buffer(f'window_{fft_size}', window)
            self.register_buffer(f'mel_fb_{fft_size}', fb)


    def forward(self, pred, target):
        total_loss = 0.0

        pred = pred.float()
        target = target.float()

        for fft_size in self.fft_sizes:
            hop_size = max(1, int(fft_size * self.hop_frac))
            window = getattr(self, f"window_{fft_size}")
            mel_fb = getattr(self, f'mel_fb_{fft_size}')

            pred_spec   = self._log_mel_spec(pred,   fft_size, hop_size, window, mel_fb)
            target_spec = self._log_mel_spec(target, fft_size, hop_size, window, mel_fb)

            pred_mag = self._mag_spec(pred, fft_size, hop_size, window)
            target_mag = self._mag_spec(target, fft_size, hop_size, window)

            mel_loss = F.l1_loss(pred_spec, target_spec)

            linear_loss = F.l1_loss(pred_mag, target_mag)
            log_loss = F.l1_loss(
                torch.log(pred_mag + self.eps),
                torch.log(target_mag + self.eps),
            )

            total_loss += linear_loss + log_loss + (0.25 * mel_loss)

        return total_loss / len(self.fft_sizes)

    def _mag_spec(self, audio, fft_size, hop_size, window):
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
        # power = stft.real ** 2 + stft.imag ** 2   # (B, F, T)
        mag = torch.abs(stft)
        mel = torch.einsum("mf,bft->bmt", mel_fb, mag)
    
        # Mel filterbank: (B, F, T) → (B, n_mels, T)
        # mel_fb: (n_mels, F)
        # mel = torch.einsum('mf,bft->bmt', mel_fb, power)

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

    def __init__(
        self,
        spectral_weight: float = 1.0,
        envelope_weight: float = 0.5,
        log_rms_weight: float = 0.5,
        high_freq_excess_weight: float = 0.0,
        high_freq_hz: float = 8000.0,
        high_freq_fft_size: int = 1024,
        sample_rate: int = 48000,
        eps: float = 1e-7
    ):
        super().__init__()
        self.spectral_loss    = MultiScaleSpectralLoss(eps=eps)
        self.spectral_weight  = spectral_weight
        self.envelope_weight  = envelope_weight
        self.log_rms_weight = log_rms_weight
        self.high_freq_excess_weight = high_freq_excess_weight
        self.high_freq_hz = high_freq_hz
        self.high_freq_fft_size = high_freq_fft_size
        self.sample_rate = sample_rate
        self.eps = eps
        self.register_buffer(
            "high_freq_window",
            torch.hann_window(high_freq_fft_size),
        )
        freqs = torch.linspace(0.0, sample_rate / 2.0, high_freq_fft_size // 2 + 1)
        self.register_buffer("high_freq_mask", freqs >= high_freq_hz)

    def forward(self, pred, target):
        spec_loss = self.spectral_loss(pred, target)

        # Loudness envelope loss (smooth RMS over 64-sample windows)
        pred_rms   = self._smooth_rms(pred)
        target_rms = self._smooth_rms(target)
        env_loss   = F.l1_loss(pred_rms, target_rms)

        pred_frame_rms = torch.sqrt(torch.mean(pred.float() ** 2, dim=-1) + self.eps)
        target_frame_rms = torch.sqrt(torch.mean(target.float() ** 2, dim=-1) + self.eps)
        log_rms_loss = F.l1_loss(
            torch.log(pred_frame_rms + self.eps),
            torch.log(target_frame_rms + self.eps),
        )

        hf_excess_loss = pred.sum() * 0.0
        if self.high_freq_excess_weight > 0.0:
            hf_excess_loss = self._high_freq_excess_loss(pred, target)

        loss = (
            self.spectral_weight * spec_loss
            + self.envelope_weight * env_loss
            + self.log_rms_weight * log_rms_loss
            + self.high_freq_excess_weight * hf_excess_loss
        )

        return loss

    # Penalize very high freqencies
    def _high_freq_excess_loss(self, pred, target):
        hop_size = max(1, self.high_freq_fft_size // 4)
        pred_mag = self._mag_spec(pred, self.high_freq_fft_size, hop_size, self.high_freq_window)
        target_mag = self._mag_spec(target, self.high_freq_fft_size, hop_size, self.high_freq_window)
        pred_db = 20.0 * torch.log10(pred_mag + self.eps)
        target_db = 20.0 * torch.log10(target_mag + self.eps)
        excess_db = F.relu(pred_db[:, self.high_freq_mask, :] - target_db[:, self.high_freq_mask, :])
        return excess_db.mean()
    
    def _mag_spec(self, audio, fft_size, hop_size, window):
        pad = fft_size // 2
        audio_padded = F.pad(audio.float(), (pad, pad))
        return torch.abs(torch.stft(
            audio_padded,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window=window.float(),
            return_complex=True,
        ))

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
