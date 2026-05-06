import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Polyphonic real-time guitar -> piano timbre transfer
# Spectral U-Net + transient shaper
# No pitch detection
# ============================================================

SAMPLE_RATE = 48000
FRAME_SIZE = 512                # was 1024 — halves analysis window
HOP_SIZE = 128                  # was 256 — keeps 75% overlap
N_FFT = 512                     # match frame size
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
    Predicts a bounded log-magnitude DELTA per (frequency, time) bin.

    Input : log-magnitude spectrogram patch  (B, 1, F, T)
    Output: log-mag delta in [-MAX_DELTA, +MAX_DELTA] (B, 1, F, T)

    Adding this delta in log space is equivalent to a per-bin multiplicative
    correction in linear space — but, unlike a sigmoid mask multiplied onto
    log_mag, the gradients stay sane near silence.
    """

    MAX_DELTA = 4.0  # ~ ×exp(4) = ×54 boost or ÷54 cut per bin

    def __init__(self, base_ch: int = 24):
        super().__init__()
        self.enc1 = ConvBlock2d(1, base_ch)
        self.enc2 = ConvBlock2d(base_ch, base_ch * 2, stride=(2, 2))
        self.enc3 = ConvBlock2d(base_ch * 2, base_ch * 4, stride=(2, 2))

        self.bottleneck = ConvBlock2d(base_ch * 4, base_ch * 4)

        self.dec3 = UpBlock2d(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec2 = UpBlock2d(base_ch * 2, base_ch * 2, base_ch)
        self.dec1 = UpBlock2d(base_ch, base_ch, base_ch)

        self.out_delta = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, log_mag: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(log_mag)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        z = self.bottleneck(s3)

        x = self.dec3(z, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        delta = torch.tanh(self.out_delta(x)) * self.MAX_DELTA
        return delta


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

        self.unet = SpectralUNet(base_ch=24)
        self.transient = TransientShaper(channels=32)

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

        spec = self._stft(audio_frame)             # (B, F, T)
        mag = torch.abs(spec)
        phase = torch.angle(spec)

        log_mag = safe_log(mag).unsqueeze(1)       # (B, 1, F, T)

        # Per-bin log-mag delta. Additive in log == multiplicative in linear.
        log_mag_delta = self.unet(log_mag)         # (B, 1, F, T) in [-4, 4]
        out_log_mag = log_mag + log_mag_delta
        out_mag = torch.exp(out_log_mag.squeeze(1))

        # Reuse input phase for low-latency reconstruction.
        # (This caps achievable quality; complex masking is the next step.)
        out_spec = torch.polar(out_mag, phase)

        audio_out = self._istft(out_spec, length=length)
        audio_out = self.transient(audio_out)

        # Hard limit only — NO dry blend, NO tanh distortion.
        audio_out = torch.clamp(audio_out, -1.0, 1.0)

        features = {
            "input_mag": mag,
            "input_log_mag": log_mag.squeeze(1),
        }
        params = {
            "log_mag_delta": log_mag_delta.squeeze(1),
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