import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Polyphonic real-time guitar -> piano timbre transfer
# Spectral U-Net + transient shaper + resonance block
# No pitch detection
# ============================================================

SAMPLE_RATE = 48000
FRAME_SIZE = 1024               # analysis window
HOP_SIZE = 256                  # 75% overlap
N_FFT = 1024
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

    def __init__(self, base_ch: int = 32):
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


class TransientShaper(nn.Module):
    """
    Learns onset reshaping so pick transients become more hammer-like.
    """

    def __init__(self, channels: int = 48):
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
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = audio.unsqueeze(1)
        delta = self.delta_net(x)
        gate = self.gate_net(torch.abs(x))

        # stronger transient shaping than before
        y = x + 0.40 * gate * delta
        return y.squeeze(1), gate.squeeze(1)


class ResonanceBlock(nn.Module):
    """
    Short learned resonant response to add piano-body ring/bloom.
    """

    def __init__(self, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=33, padding=16),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=65, padding=32),
            nn.GELU(),
            nn.Conv1d(hidden, 1, kernel_size=33, padding=16),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, audio: torch.Tensor, excite: torch.Tensor | None = None) -> torch.Tensor:
        x = audio.unsqueeze(1)
        res = self.net(x)

        if excite is None:
            g = self.gate(torch.abs(x))
        else:
            g = self.gate(excite.unsqueeze(1))

        # subtle body resonance
        y = x + 0.18 * g * res
        return y.squeeze(1)


class PolyphonicGuitarToPiano(nn.Module):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        n_fft: int = N_FFT,
        hop_size: int = HOP_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.hidden_size = hidden_size

        self.unet = SpectralUNet(base_ch=32)
        self.transient = TransientShaper(channels=48)
        self.resonance = ResonanceBlock(hidden=24)

        self.register_buffer("window", torch.hann_window(frame_size))

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            audio.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.frame_size,
            window=self.window.to(audio.device),
            return_complex=True,
            center=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.frame_size,
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

        # clamp helps keep extreme spectral explosions down
        out_log_mag = torch.clamp(out_log_mag, min=-12.0, max=8.0)
        out_mag = torch.exp(out_log_mag.squeeze(1))

        # Reuse input phase for low-latency polyphonic consistency
        out_spec = torch.polar(out_mag, phase)

        audio_out = self._istft(out_spec, length=length)

        # hammer-like attack shaping
        audio_out, transient_gate = self.transient(audio_out)

        # piano body / bloom
        audio_out = self.resonance(audio_out, excite=transient_gate)

        # less dry blend than before so more guitar identity is removed
        audio_out = 0.97 * torch.tanh(audio_out) + 0.03 * audio_frame

        features = {
            "input_mag": mag,
            "input_log_mag": log_mag.squeeze(1),
        }
        params = {
            "mask": mask.squeeze(1),
            "residual": residual.squeeze(1),
            "transient_gate": transient_gate,
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


DDSPGuitarToPiano = PolyphonicGuitarToPiano