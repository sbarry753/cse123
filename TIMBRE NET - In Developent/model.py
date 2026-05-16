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
HIDDEN_SIZE = 256
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

    def __init__(self, base_ch: int = 24):
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

        self.blocks = nn.Sequential(
                    nn.Conv1d(
                        hidden_ch, hidden_ch, kernel_size=3, padding=1, dilation=1,
                    ),
                    nn.GELU(),
                    nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1),
                    nn.GELU(),
                    
                    nn.Conv1d(
                        hidden_ch, hidden_ch, kernel_size=3, padding=2, dilation=2,
                    ),
                    nn.GELU(),
                    nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1),
                    nn.GELU(),
                    
                    nn.Conv1d(
                        hidden_ch, hidden_ch, kernel_size=3, padding=4, dilation=4,
                    ),
                    nn.GELU(),
                    nn.Conv1d(hidden_ch, hidden_ch, kernel_size=1),
                    nn.GELU(),
                )

        self.out = nn.Conv1d(hidden_ch, self.n_freq_bins, kernel_size=1)
        self.max_delta = float(max_delta)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"PhaseResidualTCN expects (B, C, F, T), got shape {tuple(x.shape)}")
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

    def __init__(self, channels: int = 32):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )
        self.gate_net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)
        delta = self.delta_net(x)
        gate = self.gate_net(torch.abs(x))
        y = x + 0.30 * gate * delta
        return y.squeeze(1)


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
        **kwargs,
    ):
        super().__init__()
        win_length = int(win_length or n_fft)
        if frame_size <= 0:
            raise ValueError(f"frame_size must be > 0, got {frame_size}")
        if hop_size <= 0:
            raise ValueError(f"hop_size must be > 0, got {hop_size}")
        if win_length <= 0 or win_length > n_fft:
            raise ValueError(f"win_length must satisfy 0 < win_length <= n_fft, got {win_length} and n_fft={n_fft}")
        self.sample_rate = sample_rate
        self.frame_size = int(frame_size)
        self.n_fft = int(n_fft)
        self.win_length = win_length
        self.hop_size = int(hop_size)
        self.hidden_size = hidden_size
        self.n_freq_bins = self.n_fft // 2 + 1

        self.unet = SpectralUNet(base_ch)
        self.phase_tcn = PhaseResidualTCN(
            in_ch=4,
            n_freq_bins=self.n_freq_bins,
            hidden_ch=phase_tcn_ch,
            layers=phase_tcn_layers,
            max_delta=phase_max_delta,
        )
        self.transient = TransientShaper(base_ch)

        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

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

        phase_input = torch.cat([log_mag, out_log_mag, mask, residual], dim=1)
        phase_delta, phase_delta_raw = self.phase_tcn(phase_input)
        phase_delta = phase_delta.squeeze(1)
        phase_delta_raw = phase_delta_raw.squeeze(1)
        out_phase = phase + phase_delta

        out_spec = torch.polar(out_mag, out_phase)

        audio_out = self._istft(out_spec, length=length)
        #audio_out = self.transient(audio_out)
        # audio_out = torch.tanh(audio_out)

        features = {
            "input_mag": mag,
            "input_log_mag": log_mag.squeeze(1),
            "input_phase": phase,
            "out_log_mag": out_log_mag.squeeze(1),
            "out_phase": out_phase,
        }
        params = {
            "mask": mask.squeeze(1),
            "residual": residual.squeeze(1),
            "phase_delta": phase_delta,
            "phase_delta_raw": phase_delta_raw,
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
