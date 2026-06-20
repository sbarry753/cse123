import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Polyphonic real-time guitar -> piano timbre transfer
# Spectral U-Net + transient shaper
# No pitch detection
# ============================================================

SAMPLE_RATE = 48000
FRAME_SIZE = 1024               # audio chunk length
HOP_SIZE = 256                  # 75% overlap
N_FFT = 1024
WIN_LENGTH = N_FFT
N_FREQ_BINS = N_FFT // 2 + 1

# Kept for compatibility with old scripts / imports
N_HARMONICS = 64
N_NOISE_BANDS = 65
HIDDEN_SIZE = 128
N_MFCC = 20


def safe_log(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock2d(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SpectralUNet(nn.Module):
    """
    Input : log-magnitude spectrogram patch  (B, 1, F, T)
    Output: multiplicative mask + additive residual in log-mag domain
    """

    def __init__(self, base_ch: int = 12):
        super().__init__()
        self.enc1 = ConvBlock2d(1, base_ch)
        self.enc2 = ConvBlock2d(base_ch, base_ch * 2, stride=(2, 2))
        self.enc3 = ConvBlock2d(base_ch * 2, base_ch * 4, stride=(2, 2))

        self.bottleneck = ConvBlock2d(base_ch * 4, base_ch * 4)

        self.dec3 = UpBlock2d(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec2 = UpBlock2d(base_ch * 2, base_ch * 2, base_ch)
        self.dec1 = UpBlock2d(base_ch, base_ch, base_ch)

        self.out_mask = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.out_res = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, log_mag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s1 = self.enc1(log_mag)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        z = self.bottleneck(s3)

        x = self.dec3(z, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        mask = torch.sigmoid(self.out_mask(x)) * 2.0
        residual = self.out_res(x)
        return mask, residual


class PhaseResidualTCN(nn.Module):
    """
    Predicts a bounded phase residual after magnitude prediction.
    Operates over STFT time frames with frequency bins folded into channels.
    """

    def __init__(
        self,
        in_ch: int = 4,
        n_freq_bins: int = N_FREQ_BINS,
        hidden_ch: int = 16,
        layers: int = 3,
        max_delta: float = 0.5,
    ):
        super().__init__()
        if in_ch <= 0:
            raise ValueError(f"in_ch must be > 0, got {in_ch}")
        if n_freq_bins <= 0:
            raise ValueError(f"n_freq_bins must be > 0, got {n_freq_bins}")
        if hidden_ch <= 0:
            raise ValueError(f"phase_tcn_ch must be > 0, got {hidden_ch}")
        if layers <= 0:
            raise ValueError(f"phase_tcn_layers must be > 0, got {layers}")
        if max_delta < 0.0:
            raise ValueError(f"phase_max_delta must be >= 0, got {max_delta}")

        self.in_ch = in_ch
        self.n_freq_bins = n_freq_bins
        self.input_proj = nn.Sequential(
            nn.Conv1d(self.in_ch * self.n_freq_bins, hidden_ch, kernel_size=1),
            nn.GELU(),
        )

        blocks = []
        for idx in range(layers):
            dilation = 2 ** idx
            blocks.extend(
                [
                    nn.Conv1d(
                        hidden_ch,
                        hidden_ch,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1),
                    nn.GELU(),
                ]
            )
        self.blocks = nn.Sequential(*blocks)

        self.out = nn.Conv1d(hidden_ch, self.n_freq_bins, kernel_size=1)
        self.max_delta = float(max_delta)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("PhaseResidualTCN expects a 4D tensor shaped (B, C, F, T)")
        batch, channels, freq_bins, frames = x.shape
        if channels != self.in_ch:
            raise ValueError(f"PhaseResidualTCN expected {self.in_ch} channels, got {channels}")
        if freq_bins != self.n_freq_bins:
            raise ValueError(f"PhaseResidualTCN expected {self.n_freq_bins} freq bins, got {freq_bins}")

        y = x.reshape(batch, channels * freq_bins, frames)
        y = self.input_proj(y)
        y = self.blocks(y)
        raw = self.out(y).reshape(batch, 1, freq_bins, frames)
        delta = self.max_delta * torch.tanh(raw)
        return delta, raw


class TransientShaper(nn.Module):
    """
    Learns onset reshaping so pick transients become more hammer-like.
    """

    def __init__(self, channels: int = 16):
        super().__init__()
        # Lighter transient shaper for real-time use.
        # This keeps the same input/output behavior but uses fewer channels
        # and one fewer convolution than the original version.
        self.delta_net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )
        self.gate_net = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)
        delta = self.delta_net(x)
        gate = self.gate_net(torch.abs(x))
        y = x + 0.30 * gate * delta
        return y.squeeze(1)


class TransientCorrection(nn.Module):
    """
    Small post-ISTFT residual branch for attack-local waveform correction.
    Starts as a no-op because the final projection is zero-initialized.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = 16,
        transient_ms: float = 30.0,
        max_gain: float = 0.20,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"transient_ch must be > 0, got {channels}")
        if transient_ms <= 0.0:
            raise ValueError(f"transient_ms must be > 0, got {transient_ms}")
        if max_gain < 0.0:
            raise ValueError(f"transient_max_gain must be >= 0, got {max_gain}")
        self.sample_rate = int(sample_rate)
        self.transient_ms = float(transient_ms)
        self.max_gain = float(max_gain)
        self.net = nn.Sequential(
            nn.Conv1d(3, channels, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _window(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_attack = min(length, max(1, int(round(self.transient_ms * self.sample_rate / 1000.0))))
        window = torch.zeros(length, device=device, dtype=dtype)
        if n_attack == 1:
            window[0] = 1.0
            return window.view(1, length)
        fade = torch.hann_window(n_attack * 2, periodic=False, device=device, dtype=dtype)[n_attack:]
        window[:n_attack] = fade
        return window.view(1, length)

    def forward(self, guitar_audio: torch.Tensor, base_audio: torch.Tensor) -> torch.Tensor:
        if guitar_audio.shape != base_audio.shape:
            raise ValueError("TransientCorrection inputs must have matching shapes")
        x = torch.stack([guitar_audio, base_audio, base_audio - guitar_audio], dim=1)
        residual = torch.tanh(self.net(x)).squeeze(1)
        window = self._window(base_audio.shape[-1], base_audio.device, base_audio.dtype)
        return self.max_gain * window * residual


class PolyphonicGuitarToPiano(nn.Module):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        n_fft: int = N_FFT,
        win_length: int | None = None,
        hop_size: int = HOP_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        base_ch: int = 32,
        phase_tcn_ch: int = 16,
        phase_tcn_layers: int = 3,
        phase_max_delta: float = 0.5,
        output_size: int | None = None,
        transient_ch: int = 16,
        transient_ms: float = 30.0,
        transient_max_gain: float = 0.20,
        **kwargs,
    ):
        super().__init__()
        win_length = int(win_length or n_fft)
        if frame_size <= 0:
            raise ValueError(f"frame_size must be > 0, got {frame_size}")
        if hop_size <= 0:
            raise ValueError(f"hop_size must be > 0, got {hop_size}")
        output_size = int(frame_size if output_size is None else output_size)
        if output_size <= 0 or output_size > frame_size:
            raise ValueError(f"output_size must satisfy 0 < output_size <= frame_size, got {output_size}")
        if win_length <= 0 or win_length > n_fft:
            raise ValueError(f"win_length must satisfy 0 < win_length <= n_fft, got {win_length} and n_fft={n_fft}")
        self.sample_rate = sample_rate
        self.frame_size = int(frame_size)
        self.output_size = int(output_size)
        self.n_fft = int(n_fft)
        self.win_length = win_length
        self.hop_size = int(hop_size)
        self.hidden_size = hidden_size
        self.n_freq_bins = self.n_fft // 2 + 1

        self.unet = SpectralUNet(base_ch)
        self.phase_tcn = PhaseResidualTCN(
            in_ch=10,
            n_freq_bins=self.n_freq_bins,
            hidden_ch=phase_tcn_ch,
            layers=phase_tcn_layers,
            max_delta=phase_max_delta,
        )
        self.transient = TransientShaper(base_ch)
        self.transient_correction = TransientCorrection(
            sample_rate=sample_rate,
            channels=transient_ch,
            transient_ms=transient_ms,
            max_gain=transient_max_gain,
        )

        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    @staticmethod
    def _wrapped_phase_delta(delta: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    def _phase_context(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase_dt = self._wrapped_phase_delta(phase[:, :, 1:] - phase[:, :, :-1])
        phase_dt = F.pad(phase_dt, (1, 0))
        phase_df = self._wrapped_phase_delta(phase[:, 1:, :] - phase[:, :-1, :])
        phase_df = F.pad(phase_df, (0, 0, 1, 0))
        return phase_dt, phase_df

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            audio.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window.to(audio.device),
            return_complex=True,
            center=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window.to(spec.device),
            center=True,
            length=length,
        )

    def forward(self, audio_frame: torch.Tensor):
        """
        audio_frame: (B, FRAME_SIZE)
        returns: audio_out, features, params
        """
        length = audio_frame.shape[-1]

        spec = self._stft(audio_frame)         # (B, F, T)
        mag = torch.abs(spec)
        phase = torch.angle(spec)

        log_mag = safe_log(mag).unsqueeze(1)   # (B, 1, F, T)

        mask, residual = self.unet(log_mag)
        out_log_mag = log_mag * mask + residual
        out_mag = torch.exp(out_log_mag.squeeze(1))

        phase_dt, phase_df = self._phase_context(phase)
        phase_input = torch.cat(
            [
                log_mag,
                out_log_mag,
                mask,
                residual,
                torch.sin(phase).unsqueeze(1),
                torch.cos(phase).unsqueeze(1),
                torch.sin(phase_dt).unsqueeze(1),
                torch.cos(phase_dt).unsqueeze(1),
                torch.sin(phase_df).unsqueeze(1),
                torch.cos(phase_df).unsqueeze(1),
            ],
            dim=1,
        )
        phase_delta, phase_delta_raw = self.phase_tcn(phase_input)
        phase_delta = phase_delta.squeeze(1)
        phase_delta_raw = phase_delta_raw.squeeze(1)
        out_phase = phase + phase_delta

        out_spec = torch.polar(out_mag, out_phase)

        audio_before_transient = self._istft(out_spec, length=length)
        transient_delta = self.transient_correction(audio_frame, audio_before_transient)
        full_audio_out = audio_before_transient + transient_delta
        #audio_out = self.transient(audio_out)
        # audio_out = torch.tanh(audio_out)
        output_size = min(self.output_size, length)
        audio_out = full_audio_out[..., -output_size:]

        features = {
            "input_mag": mag,
            "input_log_mag": log_mag.squeeze(1),
            "input_phase": phase,
            "input_phase_dt": phase_dt,
            "input_phase_df": phase_df,
            "out_log_mag": out_log_mag.squeeze(1),
            "out_phase": out_phase,
            "before_transient_audio": audio_before_transient,
            "after_transient_audio": full_audio_out,
            "cropped_audio_out": audio_out,
        }
        params = {
            "mask": mask.squeeze(1),
            "residual": residual.squeeze(1),
            "phase_delta": phase_delta,
            "phase_delta_raw": phase_delta_raw,
            "transient_delta": transient_delta,
            "transient_delta_abs": torch.abs(transient_delta),
        }
        return audio_out, features, params

    def reset_phase(self):
        # Kept for compatibility with the old real-time interface
        pass

    @torch.jit.export
    def infer_frame(self, audio_frame: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out, _, _ = self.forward(audio_frame)
        return out


# Backward-compatible name for old imports
DDSPGuitarToPiano = PolyphonicGuitarToPiano
