"""
model.py — DDSP Guitar-to-Piano Timbre Transfer

Architecture:
  Guitar audio → Feature Encoder (f0 + loudness + spectral)
              → MLP Decoder (guitar features → piano synth params)
              → Additive Synthesizer (harmonics + filtered noise)
              → Piano-like audio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
SAMPLE_RATE     = 48000         # match your audio interface — change to 44100 if needed
FRAME_SIZE      = 256          # ~5.3ms at 48kHz — still well under 12ms budget
HOP_SIZE        = 256          # non-overlapping frames for minimum latency
N_HARMONICS     = 64           # piano has rich harmonic content
N_NOISE_BANDS   = 65           # noise filter bands (FFT size // 2 + 1)
N_FFT           = 1024
HIDDEN_SIZE     = 512
N_MFCC          = 20


# ─────────────────────────────────────────────
#  FEATURE ENCODER
#  Extracts f0, loudness, and MFCC from audio
#  frames. All operations are fast DSP — no
#  neural network here, keeping latency minimal.
# ─────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    def __init__(self, sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE, n_mfcc=N_MFCC):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size  = frame_size
        self.n_mfcc      = n_mfcc

        # Mel filterbank for MFCC (precomputed, stored as buffer)
        mel_fb = self._build_mel_filterbank(n_mels=64, n_fft=N_FFT)
        self.register_buffer('mel_fb', mel_fb)

        # DCT matrix for MFCC (precomputed)
        dct = self._build_dct_matrix(64, n_mfcc)
        self.register_buffer('dct', dct)

    # ---- Pitch (f0) via autocorrelation (YIN-lite) ----
    def estimate_f0(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Fast autocorrelation-based f0 estimate.
        frame: (batch, frame_size) mono audio
        returns: (batch,) f0 in Hz, 0.0 if unvoiced
        """
        # torch.fft does not support float16 — cast to float32
        frame = frame.float()
        B, N = frame.shape

        # DC removal
        frame = frame - frame.mean(dim=-1, keepdim=True)

        # ── Amplitude-normalise before YIN ────────────────────────────────
        # YIN's voiced/unvoiced decision is amplitude-independent once we
        # normalise, so quiet guitar recordings don't get silently flagged
        # as unvoiced. We track original RMS separately for loudness feature.
        rms = torch.sqrt((frame ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        frame_norm = frame / (rms + 1e-8)   # unit-RMS for pitch detection only

        # Autocorrelation via FFT
        fft_size = 2 * N
        X = torch.fft.rfft(frame_norm, n=fft_size)
        acf = torch.fft.irfft(X * X.conj(), n=fft_size)[..., :N]
        acf = acf / (acf[..., :1] + 1e-8)

        # Search tau in range corresponding to 60–1200 Hz
        tau_min = max(1, int(self.sample_rate / 1200))
        tau_max = min(N - 1, int(self.sample_rate / 60))

        # Difference function from YIN
        diff = 1.0 - acf[..., tau_min:tau_max]
        tau_hat = diff.argmin(dim=-1) + tau_min          # (B,)

        f0 = self.sample_rate / tau_hat.float()          # (B,)

        # Voiced / unvoiced threshold — raised from 0.15 → 0.35 so quiet
        # but periodic guitar notes aren't incorrectly flagged as unvoiced.
        min_diff = diff.min(dim=-1).values
        f0 = torch.where(min_diff < 0.35, f0, torch.zeros_like(f0))
        return f0

    # ---- Loudness (RMS in dB) ----
    def compute_loudness(self, frame: torch.Tensor) -> torch.Tensor:
        """frame: (batch, N) → (batch,) loudness in dB"""
        rms = torch.sqrt((frame ** 2).mean(dim=-1) + 1e-8)
        return 20.0 * torch.log10(rms + 1e-8)

    # ---- MFCC ----
    def compute_mfcc(self, frame: torch.Tensor) -> torch.Tensor:
        """frame: (batch, N) → (batch, n_mfcc)"""
        # torch.fft does not support float16 — cast to float32
        frame = frame.float()
        # Hann window
        window = torch.hann_window(self.frame_size, device=frame.device)
        windowed = frame * window
        # Power spectrum
        X = torch.fft.rfft(windowed, n=N_FFT)
        power = X.real ** 2 + X.imag ** 2          # (B, N_FFT//2+1)
        # Mel filterbank
        mel = power @ self.mel_fb.T                 # (B, 64)
        log_mel = torch.log(mel + 1e-8)
        # DCT
        mfcc = log_mel @ self.dct.T                 # (B, n_mfcc)
        return mfcc

    def forward(self, frame: torch.Tensor):
        """
        frame: (batch, frame_size)
        returns dict with f0, loudness, mfcc, feature_vec
        """
        f0       = self.estimate_f0(frame)           # (B,)
        loudness = self.compute_loudness(frame)       # (B,)
        mfcc     = self.compute_mfcc(frame)           # (B, n_mfcc)

        # Normalise for MLP input
        f0_norm  = f0 / 1000.0                                   # Hz → ~[0,1] for guitar range
        ld_norm  = torch.clamp((loudness + 80.0) / 80.0, 0.0, 1.0)  # dB → [0,1], clamped
        f0_norm  = f0_norm.unsqueeze(-1)
        ld_norm  = ld_norm.unsqueeze(-1)

        feature_vec = torch.cat([f0_norm, ld_norm, mfcc], dim=-1)  # (B, 2 + n_mfcc)
        return {
            'f0':          f0,
            'loudness':    loudness,
            'mfcc':        mfcc,
            'feature_vec': feature_vec,
        }

    # ── helpers ──────────────────────────────
    def _build_mel_filterbank(self, n_mels: int, n_fft: int) -> torch.Tensor:
        freqs  = np.linspace(0, self.sample_rate / 2, n_fft // 2 + 1)
        mel_lo = self._hz_to_mel(0.0)
        mel_hi = self._hz_to_mel(self.sample_rate / 2)
        mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
        hz_pts  = self._mel_to_hz(mel_pts)
        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            f_lo, f_c, f_hi = hz_pts[m-1], hz_pts[m], hz_pts[m+1]
            for k, f in enumerate(freqs):
                if f_lo <= f <= f_c:
                    fb[m-1, k] = (f - f_lo) / (f_c - f_lo + 1e-8)
                elif f_c < f <= f_hi:
                    fb[m-1, k] = (f_hi - f) / (f_hi - f_c + 1e-8)
        return torch.from_numpy(fb)

    def _build_dct_matrix(self, n_mels: int, n_mfcc: int) -> torch.Tensor:
        n = np.arange(n_mels)
        k = np.arange(n_mfcc).reshape(-1, 1)
        dct = np.cos(np.pi / n_mels * (n + 0.5) * k).astype(np.float32)
        return torch.from_numpy(dct)

    @staticmethod
    def _hz_to_mel(hz):  return 2595.0 * np.log10(1.0 + hz / 700.0)
    @staticmethod
    def _mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ─────────────────────────────────────────────
#  MLP DECODER
#  Maps guitar feature vector → piano synth
#  parameters. This is where all the "learning"
#  from your paired dataset lives.
# ─────────────────────────────────────────────
class MLPDecoder(nn.Module):
    def __init__(
        self,
        input_size  = 2 + N_MFCC,
        hidden_size = HIDDEN_SIZE,
        n_harmonics = N_HARMONICS,
        n_noise     = N_NOISE_BANDS,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise     = n_noise

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),

            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
        )

        self.head_harmonic_amps = nn.Linear(hidden_size // 2, n_harmonics)
        self.head_global_amp    = nn.Linear(hidden_size // 2, 1)
        self.head_noise_mags    = nn.Linear(hidden_size // 2, n_noise)
        self.head_f0_correction = nn.Linear(hidden_size // 2, 1)

    def forward(self, feature_vec: torch.Tensor, f0: torch.Tensor):
        h = self.net(feature_vec)

        # allow true near-zero harmonic output
        harm_raw = self.head_harmonic_amps(h)
        harm_amps = torch.relu(harm_raw)

        # learned loudness gate
        global_amp = torch.sigmoid(self.head_global_amp(h))
        harm_amps = harm_amps * global_amp

        # normalize only when there is actual harmonic energy
        harm_sum = harm_amps.sum(dim=-1, keepdim=True)
        harm_amps = torch.where(
            harm_sum > 1e-6,
            harm_amps / (harm_sum + 1e-8) * global_amp,
            harm_amps
        )

        # keep noise weak
        noise_mags = torch.sigmoid(self.head_noise_mags(h)) * 0.01

        # small pitch correction only
        f0_correction = torch.tanh(self.head_f0_correction(h)).squeeze(-1) * 3.0
        f0_corrected = torch.clamp(f0 + f0_correction, min=0.0)

        return {
            'harm_amps':    harm_amps,
            'global_amp':   global_amp.squeeze(-1),
            'noise_mags':   noise_mags,
            'f0_corrected': f0_corrected,
        }
    
# ─────────────────────────────────────────────
#  ADDITIVE SYNTHESIZER
#  Generates audio from harmonic parameters.
#  Pure DSP — fast and deterministic.
# ─────────────────────────────────────────────
class AdditiveSynth(nn.Module):
    def __init__(
        self,
        sample_rate = SAMPLE_RATE,
        frame_size  = FRAME_SIZE,
        n_harmonics = N_HARMONICS,
        n_noise     = N_NOISE_BANDS,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size  = frame_size
        self.n_harmonics = n_harmonics
        self.n_noise     = n_noise

        # Phase accumulator for real-time inference (B=1 only).
        # During training frames are shuffled so continuity is meaningless —
        # we start each frame at phase=0 in that case.
        self.register_buffer('phase', torch.zeros(1, n_harmonics))

    def forward(self, f0: torch.Tensor, harm_amps: torch.Tensor, noise_mags: torch.Tensor):
        """
        f0:         (batch,) fundamental frequency in Hz
        harm_amps:  (batch, n_harmonics)
        noise_mags: (batch, n_noise)
        returns:    (batch, frame_size) audio
        """
        B = f0.shape[0]
        device = f0.device

        # ── Harmonic component ───────────────
        # Harmonic frequencies: f0 * [1, 2, 3, ..., n_harmonics]
        harmonic_idx = torch.arange(1, self.n_harmonics + 1, device=device).float()
        freqs = f0.unsqueeze(-1) * harmonic_idx              # (B, n_harmonics)

        # Phase increment per sample
        phase_increment = 2.0 * np.pi * freqs / self.sample_rate   # (B, n_harmonics)

        # Starting phase:
        #   • B == 1  → use persistent accumulator (real-time, frame-to-frame continuity)
        #   • B  > 1  → start at zero (training — frames are shuffled, continuity is irrelevant)
        if B == 1:
            start_phase = self.phase                              # (1, n_harmonics)
        else:
            start_phase = torch.zeros(B, self.n_harmonics, device=device)  # (B, n_harmonics)

        # Generate phase ramp for this frame
        t = torch.arange(self.frame_size, device=device).float()   # (frame_size,)
        # (B, 1, n_harmonics) + (1, frame_size, 1) → (B, frame_size, n_harmonics)
        phase_ramp = (
            start_phase.unsqueeze(1) +
            phase_increment.unsqueeze(1) * t.unsqueeze(0).unsqueeze(-1)
        )

        # Advance persistent phase only in real-time mode (B == 1)
        if B == 1:
            self.phase = (start_phase + phase_increment * self.frame_size) % (2 * np.pi)

        # Weighted sum of harmonics
        harm_signal = (torch.sin(phase_ramp) * harm_amps.unsqueeze(1)).sum(dim=-1)  # (B, frame_size)

        # Zero out if f0 = 0 (unvoiced)
        voiced = (f0 > 0).float().unsqueeze(-1)
        harm_signal = harm_signal * voiced

        # ── Noise component (shaped noise) ───
        noise = torch.randn(B, self.frame_size, device=device)
        noise_filtered = self._filter_noise(noise, noise_mags)

        # Mix: piano is dominated by harmonics, subtle noise for breathiness/transient
        signal = harm_signal + noise_filtered * 0.05

        # Soft clip to avoid digital clipping
        signal = torch.tanh(signal * 0.9)
        return signal

    def _filter_noise(self, noise: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        """Apply frequency-domain shaping to noise."""
        N = self.frame_size
        # FFT noise
        noise_fft = torch.fft.rfft(noise, n=N)                          # (B, N//2+1)
        # Interpolate magnitude envelope to match FFT bins
        n_bins = N // 2 + 1
        mags_interp = F.interpolate(
            magnitudes.unsqueeze(1),
            size=n_bins,
            mode='linear',
            align_corners=False
        ).squeeze(1)                                                      # (B, n_bins)
        # Apply shaping
        noise_fft = noise_fft * mags_interp
        return torch.fft.irfft(noise_fft, n=N)                           # (B, N)


# ─────────────────────────────────────────────
#  FULL DDSP MODEL
#  Encoder → MLP Decoder → Additive Synth
# ─────────────────────────────────────────────
class DDSPGuitarToPiano(nn.Module):
    def __init__(
        self,
        sample_rate = SAMPLE_RATE,
        frame_size  = FRAME_SIZE,
        n_harmonics = N_HARMONICS,
        n_noise     = N_NOISE_BANDS,
        hidden_size = HIDDEN_SIZE,
        n_mfcc      = N_MFCC,
    ):
        super().__init__()
        self.encoder  = FeatureEncoder(sample_rate, frame_size, n_mfcc)
        self.decoder  = MLPDecoder(
            input_size  = 2 + n_mfcc,
            hidden_size = hidden_size,
            n_harmonics = n_harmonics,
            n_noise     = n_noise,
        )
        self.synth    = AdditiveSynth(sample_rate, frame_size, n_harmonics, n_noise)
        self.frame_size = frame_size

    def forward(self, audio_frame: torch.Tensor):
        """
        audio_frame: (batch, frame_size) — mono, normalised to [-1, 1]
        returns:     (batch, frame_size) resynthesised piano-like audio
        """
        features  = self.encoder(audio_frame)
        params    = self.decoder(features['feature_vec'], features['f0'])
        output    = self.synth(
            f0         = params['f0_corrected'],
            harm_amps  = params['harm_amps'],
            noise_mags = params['noise_mags'],
        )
        return output, features, params

    def reset_phase(self):
        """Call between songs / phrases to reset phase accumulator."""
        self.synth.phase = torch.zeros_like(self.synth.phase)

    @torch.jit.export
    def infer_frame(self, audio_frame: torch.Tensor) -> torch.Tensor:
        """Minimal inference path — used in real-time loop."""
        with torch.no_grad():
            output, _, _ = self.forward(audio_frame)
        return output