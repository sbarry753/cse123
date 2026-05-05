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
# from einops import rearrange


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
N_MELS          = 64
N_ENVELOPE_POINTS = 16
N_BODY_FILTER_BANDS = 64
Z_LATENT_SIZE   = 64


# ─────────────────────────────────────────────
#  FEATURE ENCODER
#  Extracts f0, loudness, and log-mel features from audio
#  frames. All operations are fast DSP — no
#  neural network here, keeping latency minimal.
# ─────────────────────────────────────────────
class FeatureEncoder(nn.Module):
    def __init__(self, sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE, n_mels=N_MELS):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size  = frame_size
        self.n_mels      = n_mels

        # Mel filterbank for log-mel features (precomputed, stored as buffer)
        mel_fb = self._build_mel_filterbank(n_mels=n_mels, n_fft=N_FFT)
        self.register_buffer('mel_fb', mel_fb)

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

    # ---- Log-mel spectral envelope ----
    def compute_log_mel(self, frame: torch.Tensor) -> torch.Tensor:
        """frame: (batch, N) → (batch, n_mels)"""
        # torch.fft does not support float16 — cast to float32
        frame = frame.float()
        # Hann window
        window = torch.hann_window(self.frame_size, device=frame.device)
        windowed = frame * window
        # Power spectrum
        X = torch.fft.rfft(windowed, n=N_FFT)
        mag = torch.abs(X)     # (B, N_FFT//2+1)
        # Mel filterbank
        mel = mag @ self.mel_fb.T                 # (B, n_mels)
        log_mel = torch.log(mel + 1e-8)
        return log_mel

    def forward(self, frame: torch.Tensor):
        """
        frame: (batch, frame_size)
        returns dict with f0, loudness, log_mel, feature_vec
        """
        f0       = self.estimate_f0(frame)           # (B,)
        loudness = self.compute_loudness(frame)       # (B,)
        log_mel  = self.compute_log_mel(frame)        # (B, n_mels)

        # Normalise for MLP input
        f0_norm  = f0 / 1000.0                                   # Hz → ~[0,1] for guitar range
        ld_norm  = torch.clamp((loudness + 80.0) / 80.0, 0.0, 1.0)  # dB → [0,1], clamped
        f0_norm  = f0_norm.unsqueeze(-1)
        ld_norm  = ld_norm.unsqueeze(-1)

        feature_vec = torch.cat([f0_norm, ld_norm, log_mel], dim=-1)  # (B, 2 + n_mels)
        return {
            'f0':          f0,
            'loudness':    loudness,
            'log_mel':     log_mel,
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

    @staticmethod
    def _hz_to_mel(hz):  return 2595.0 * np.log10(1.0 + hz / 700.0)
    @staticmethod
    def _mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ─────────────────────────────────────────────
#  Z ENCODER
#  Small CNN encoder to capture residual audio
#  information not captured by the DSP encoder
# ─────────────────────────────────────────────
class ZEncoder(nn.Module):
    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        z_latent_size=Z_LATENT_SIZE,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.z_latent_size = z_latent_size

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(1, 16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, z_latent_size),
        )

    def forward(self, audio_frame: torch.Tensor) -> torch.Tensor:
        return self.net(audio_frame.float().unsqueeze(1))

# ─────────────────────────────────────────────
#  MLP DECODER
#  Maps guitar feature vector → piano synth
#  parameters. This is where all the "learning"
#  from your paired dataset lives.
# ─────────────────────────────────────────────
class MLPDecoder(nn.Module):
    def __init__(
        self,
        input_size  = 2 + N_MELS,       # f0 + loudness + log-mel
        hidden_size = HIDDEN_SIZE,
        n_harmonics = N_HARMONICS,
        n_noise     = N_NOISE_BANDS,
        n_envelope  = N_ENVELOPE_POINTS,
        n_body_filter = N_BODY_FILTER_BANDS,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise     = n_noise
        self.n_envelope  = n_envelope
        self.n_body_filter = n_body_filter

        self.global_amp_scale   = 1.0
        self.noise_gain_scale   = 0.2
        self.body_filter_max_db = 24.0
        self.f0_min_hz          = 50.0
        self.f0_max_hz          = 1200.0
        self.f0_correction_octaves = 2.0

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
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

            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
        )

        # Separate output heads
        self.head_harmonic_amps = nn.Linear(hidden_size // 2, n_harmonics)   # per-harmonic amplitudes
        self.head_global_amp    = nn.Linear(hidden_size // 2, 1)             # overall amplitude
        self.head_noise_mags    = nn.Linear(hidden_size // 2, n_noise)       # noise filter magnitudes
        self.head_noise_gain    = nn.Linear(hidden_size // 2, 1)             # learned scalar noise/transient amount
        self.head_body_filter   = nn.Linear(hidden_size // 2, n_body_filter) # post-source piano body filter
        self.head_f0            = nn.Linear(hidden_size // 2, 1)             # bounded log-f0 correction
        self.head_voicing       = nn.Linear(hidden_size // 2, 1)             # voiced/unvoiced probability
        self.head_envelope      = nn.Linear(hidden_size // 2, n_envelope)    # intra-frame amplitude curve

        # A zero-biased sigmoid starts at 0.5, which is very loud for this
        # additive synth. Start quieter so training has to add energy when the
        # piano target needs it instead of fighting an always-hot default.
        nn.init.constant_(self.head_global_amp.bias, -3.0)
        nn.init.constant_(self.head_noise_gain.bias, -5.0)
        nn.init.constant_(self.head_harmonic_amps.bias, -4.0)
        nn.init.zeros_(self.head_body_filter.weight)
        nn.init.zeros_(self.head_body_filter.bias)

    def forward(self, feature_vec: torch.Tensor, f0: torch.Tensor):
        """
        feature_vec: (batch, input_size)
        f0:          (batch,) in Hz, included in feature_vec for conditioning
        returns dict with synth params
        """
        h = self.net(feature_vec)

        # Harmonic amplitude distribution (softmax = relative weights)
        # harm_dist  = torch.softmax(self.head_harmonic_amps(h), dim=-1)   # (B, n_harmonics)
        global_amp = torch.sigmoid(self.head_global_amp(h)) * self.global_amp_scale  # (B, 1)
        # harm_amps  = harm_dist * global_amp                               # (B, n_harmonics)
        harm_amps = torch.sigmoid(self.head_harmonic_amps(h)) * global_amp / np.sqrt(self.n_harmonics)

        # Noise component (sigmoid → magnitude envelope per band)
        noise_mags = torch.sigmoid(self.head_noise_mags(h)) * 0.1       # (B, n_noise) keep noise subtle
        noise_gain = (
            torch.sigmoid(self.head_noise_gain(h)).squeeze(-1)
            * self.noise_gain_scale
        )                                                              # (B,) low-initialized transient/noise gain
        body_filter_db = (
            torch.tanh(self.head_body_filter(h))
            * self.body_filter_max_db
        )                                                              # (B, n_body_filter)
        body_filter = torch.pow(10.0, body_filter_db / 20.0)            # neutral init = 1.0
        # Intra-frame envelope. Scaling sigmoid by 2.0 starts near unity
        # at initialization while still allowing attack/decay shaping.
        envelope = torch.sigmoid(self.head_envelope(h)) * 2.0           # (B, n_envelope)

        # Bounded correction in log-Hz around the encoder f0. This keeps the
        # prediction anchored to the audio-derived pitch while allowing 
        # greater octave fixes
        fallback_f0 = torch.full_like(f0, 100.0)
        base_f0 = torch.where(f0 > 0.0, f0, fallback_f0).clamp(self.f0_min_hz, self.f0_max_hz)
        base_log_f0 = torch.log(base_f0)
        delta_log_f0 = (
            torch.tanh(self.head_f0(h)).squeeze(-1)
            * self.f0_correction_octaves
            * np.log(2.0)
        )
        corrected_log_f0 = (base_log_f0 + delta_log_f0).clamp(
            np.log(self.f0_min_hz),
            np.log(self.f0_max_hz),
        )
        f0_corrected = torch.exp(corrected_log_f0)
        voicing_logit = self.head_voicing(h).squeeze(-1)
        voicing_prob = torch.sigmoid(voicing_logit)

        return {
            'harm_amps':     harm_amps,
            'global_amp':    global_amp.squeeze(-1),
            'noise_mags':    noise_mags,
            'noise_gain':    noise_gain,
            'body_filter':   body_filter,
            'body_filter_db': body_filter_db,
            'f0_corrected':  f0_corrected,
            'voicing_logit': voicing_logit,
            'voicing_prob':  voicing_prob,
            'envelope':      envelope,
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
        n_envelope  = N_ENVELOPE_POINTS,
        n_body_filter = N_BODY_FILTER_BANDS,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size  = frame_size
        self.n_harmonics = n_harmonics
        self.n_noise     = n_noise
        self.n_envelope  = n_envelope
        self.n_body_filter = n_body_filter

        # Phase accumulator for real-time inference (B=1 only).
        # During training frames are shuffled so continuity is meaningless —
        # we start each frame at phase=0 in that case.
        self.register_buffer('phase', torch.zeros(1, n_harmonics))

    def forward(
        self,
        f0: torch.Tensor,
        harm_amps: torch.Tensor,
        noise_mags: torch.Tensor,
        noise_gain: torch.Tensor | None = None,
        body_filter: torch.Tensor | None = None,
        envelope: torch.Tensor | None = None,
        render_size: int | None = None,
        phase_advance_size: int | None = None,
    ):
        """
        f0:         (batch,) fundamental frequency in Hz
        harm_amps:  (batch, n_harmonics)
        noise_mags: (batch, n_noise)
        noise_gain: (batch,) optional scalar gain for shaped noise
        body_filter: (batch, n_body_filter) optional post-source magnitude filter
        envelope:   (batch, n_envelope) optional intra-frame amplitude curve
        render_size: optional number of samples to render
        phase_advance_size: optional number of samples to advance realtime phase
        returns:    (batch, render_size or frame_size) audio
        """
        B = f0.shape[0]
        device = f0.device
        render_size = int(render_size or self.frame_size)
        phase_advance_size = int(phase_advance_size or render_size)

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
        t = torch.arange(render_size, device=device).float()   # (render_size,)
        # (B, 1, n_harmonics) + (1, frame_size, 1) → (B, frame_size, n_harmonics)
        phase_ramp = (
            start_phase.unsqueeze(1) +
            phase_increment.unsqueeze(1) * t.unsqueeze(0).unsqueeze(-1)
        )

        # Advance persistent phase only in real-time mode (B == 1)
        if B == 1:
            self.phase = (start_phase + phase_increment * phase_advance_size) % (2 * np.pi)

        # Weighted sum of harmonics
        harm_signal = (torch.sin(phase_ramp) * harm_amps.unsqueeze(1)).sum(dim=-1)  # (B, frame_size)

        # Zero out if f0 = 0 (unvoiced)
        voiced = (f0 > 0).float().unsqueeze(-1)
        harm_signal = harm_signal * voiced

        # ── Noise component (shaped noise) ───
        noise = torch.randn(B, render_size, device=device)
        noise_filtered = self._filter_noise(noise, noise_mags)
        if noise_gain is None:
            noise_gain = torch.zeros(B, device=device, dtype=noise_filtered.dtype)
        else:
            noise_gain = noise_gain.to(device=device, dtype=noise_filtered.dtype)

        # Keep the stochastic branch opt-in and voiced-gated so it can model
        # piano attacks/transients without becoming constant background hiss.
        signal = harm_signal + noise_filtered * noise_gain.unsqueeze(-1) * voiced

        if body_filter is not None:
            signal = self._apply_body_filter(signal, body_filter)

        if envelope is not None:
            frame_envelope = F.interpolate(
                envelope.unsqueeze(1),
                size=render_size,
                mode='linear',
                align_corners=False,
            ).squeeze(1)
            signal = signal * frame_envelope

        # Soft clip to avoid digital clipping
        signal = torch.tanh(signal * 0.9)
        return signal

    def _filter_noise(self, noise: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        """Apply frequency-domain shaping to noise."""
        N = noise.shape[-1]
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

    def _apply_body_filter(self, signal: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        """Apply a learned post-source piano-body spectral envelope."""
        N = signal.shape[-1]
        signal_fft = torch.fft.rfft(signal, n=N)
        n_bins = N // 2 + 1
        mags_interp = F.interpolate(
            magnitudes.unsqueeze(1),
            size=n_bins,
            mode='linear',
            align_corners=False,
        ).squeeze(1)
        signal_fft = signal_fft * mags_interp
        return torch.fft.irfft(signal_fft, n=N)


# ─────────────────────────────────────────────
#  FULL DDSP MODEL
#  Encoder → MLP Decoder → Additive Synth
# ─────────────────────────────────────────────
class DDSPGuitarToPiano(nn.Module):
    def __init__(
        self,
        sample_rate = SAMPLE_RATE,
        frame_size  = FRAME_SIZE,
        context_size = None,
        hop_size = None,
        n_harmonics = N_HARMONICS,
        n_noise     = N_NOISE_BANDS,
        n_envelope  = N_ENVELOPE_POINTS,
        n_body_filter = N_BODY_FILTER_BANDS,
        hidden_size = HIDDEN_SIZE,
        n_mels      = N_MELS,
        n_mfcc      = None,
        use_z       = True,
        z_latent_size = Z_LATENT_SIZE,
    ):
        super().__init__()
        self.context_size = context_size or frame_size
        self.hop_size = hop_size or frame_size
        self.use_z = bool(use_z)
        self.z_latent_size = int(z_latent_size) if self.use_z else 0

        if n_mfcc is not None:
            n_mels = n_mfcc

        self.encoder  = FeatureEncoder(sample_rate, self.context_size, n_mels)
        self.z_encoder = (
            ZEncoder(sample_rate, self.context_size, self.z_latent_size)
            if self.use_z else None
        )
        self.decoder  = MLPDecoder(
            input_size  = 2 + n_mels + self.z_latent_size,
            hidden_size = hidden_size,
            n_harmonics = n_harmonics,
            n_noise     = n_noise,
            n_envelope  = n_envelope,
            n_body_filter = n_body_filter,
        )
        self.synth    = AdditiveSynth(sample_rate, self.hop_size, n_harmonics, n_noise, n_envelope, n_body_filter)
        self.frame_size = self.hop_size

    def render_params(
        self,
        params: dict[str, torch.Tensor],
        detach_f0: bool = False,
        f0_override: torch.Tensor | None = None,
        voicing_override: torch.Tensor | None = None,
        render_size: int | None = None,
        phase_advance_size: int | None = None,
    ) -> torch.Tensor:
        """Render decoder parameters, optionally blocking audio-loss gradients into f0."""
        f0 = params['f0_corrected'] if f0_override is None else f0_override
        if detach_f0:
            f0 = f0.detach()
        if voicing_override is None:
            voicing = params.get('voicing_prob')
            if voicing is None:
                voicing = (f0 > 0.0).to(f0.dtype)
        else:
            voicing = voicing_override.to(f0.dtype)
        return self.synth(
            f0         = f0,
            harm_amps  = params['harm_amps'] * voicing.unsqueeze(-1),
            noise_mags = params['noise_mags'],
            noise_gain = params.get('noise_gain'),
            body_filter = params.get('body_filter'),
            envelope   = params.get('envelope'),
            render_size = render_size,
            phase_advance_size = phase_advance_size,
        )

    def predict_params(self, audio_frame: torch.Tensor):
        features = self.encoder(audio_frame)
        decoder_input = features['feature_vec']
        if self.z_encoder is not None:
            z = self.z_encoder(audio_frame)
            features['z'] = z
            decoder_input = torch.cat([decoder_input, z], dim=-1)
        params = self.decoder(decoder_input, features['f0'])
        return features, params

    def forward(self, audio_frame: torch.Tensor):
        """
        audio_frame: (batch, context_size) — mono, normalised to [-1, 1]
        returns:     (batch, hop_size) resynthesised piano-like audio
        """
        features, params = self.predict_params(audio_frame)
        output    = self.render_params(params)
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


class OverlapAddRenderer:
    """Streaming inference-only overlap-add renderer for frame-boundary smoothing.

    True OLA-aware training needs sequential or clip-level batches; the current
    training loader shuffles independent hops, so this helper is intentionally
    used only by evaluation and realtime inference.
    """

    def __init__(
        self,
        model: DDSPGuitarToPiano,
        context_size: int,
        hop_size: int,
        device: torch.device,
        window_size: int | None = None,
    ):
        self.model = model
        self.context_size = int(context_size)
        self.hop_size = int(hop_size)
        self.window_size = int(window_size or (2 * self.hop_size))
        self.device = device

        if self.window_size < self.hop_size:
            raise ValueError("window_size must be >= hop_size")

        self.window = torch.hann_window(
            self.window_size,
            periodic=False,
            device=self.device,
        ).view(1, -1)
        self.last_params = None
        self.reset()

    def reset(self):
        self.context_buf = torch.zeros(1, self.context_size, device=self.device)
        self.output_accum = torch.zeros(1, self.window_size, device=self.device)
        self.window_accum = torch.zeros(1, self.window_size, device=self.device)
        self.last_params = None
        if hasattr(self.model, "reset_phase"):
            self.model.reset_phase()

    def _push_context(self, frame: torch.Tensor):
        self.context_buf[:, :-self.hop_size] = self.context_buf[:, self.hop_size:].clone()
        self.context_buf[:, -self.hop_size:] = frame

    def process_frame(self, frame) -> torch.Tensor:
        frame = torch.as_tensor(frame, dtype=torch.float32, device=self.device).view(1, self.hop_size)
        self._push_context(frame)

        with torch.no_grad():
            if hasattr(self.model, "predict_params") and hasattr(self.model, "render_params"):
                _, params = self.model.predict_params(self.context_buf)
                self.last_params = params
                rendered = self.model.render_params(
                    params,
                    render_size=self.window_size,
                    phase_advance_size=self.hop_size,
                )
            else:
                self.last_params = None
                rendered = self.model.infer_frame(self.context_buf) if hasattr(self.model, "infer_frame") else self.model(self.context_buf)[0]
                if rendered.shape[-1] != self.window_size:
                    rendered = F.pad(rendered, (0, max(0, self.window_size - rendered.shape[-1])))
                    rendered = rendered[:, :self.window_size]

        self.output_accum[:, :self.window_size] += rendered * self.window
        self.window_accum[:, :self.window_size] += self.window

        denom = self.window_accum[:, :self.hop_size].clamp_min(1e-8)
        out = self.output_accum[:, :self.hop_size] / denom

        self.output_accum[:, :-self.hop_size] = self.output_accum[:, self.hop_size:].clone()
        self.window_accum[:, :-self.hop_size] = self.window_accum[:, self.hop_size:].clone()
        self.output_accum[:, -self.hop_size:] = 0.0
        self.window_accum[:, -self.hop_size:] = 0.0
        return out.squeeze(0)

    def flush(self) -> torch.Tensor:
        denom = self.window_accum.clamp_min(1e-8)
        tail = self.output_accum / denom
        self.output_accum.zero_()
        self.window_accum.zero_()
        return tail.squeeze(0)
