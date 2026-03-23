import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal-context version of the real-time guitar -> piano model.
# The model sees a short history of past frames as separate channels,
# but still predicts only the current output frame, so runtime latency
# stays tied to FRAME_SIZE / HOP_SIZE.

SAMPLE_RATE = 48000
FRAME_SIZE = 512
HOP_SIZE = 128
N_FFT = 512
N_FREQ_BINS = N_FFT // 2 + 1
CONTEXT_FRAMES = 4

# Compatibility constants used elsewhere.
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


class SpectralTemporalUNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock2d(in_ch, base_ch)
        self.enc2 = ConvBlock2d(base_ch, base_ch * 2, stride=(2, 2))
        self.enc3 = ConvBlock2d(base_ch * 2, base_ch * 4, stride=(2, 2))

        self.bottleneck = ConvBlock2d(base_ch * 4, base_ch * 4)

        self.dec3 = UpBlock2d(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec2 = UpBlock2d(base_ch * 2, base_ch * 2, base_ch)
        self.dec1 = UpBlock2d(base_ch, base_ch, base_ch)

        self.out_mask = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.out_res = nn.Conv2d(base_ch, 1, kernel_size=1)
        self.out_phase = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        z = self.bottleneck(s3)
        x = self.dec3(z, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        mask = 0.5 + 2.5 * torch.sigmoid(self.out_mask(x))
        residual = self.out_res(x)
        phase_delta = 0.45 * torch.tanh(self.out_phase(x))
        return mask, residual, phase_delta


class TransientShaper(nn.Module):
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
        return (x + 0.25 * gate * delta).squeeze(1)


class PolyphonicGuitarToPianoTemporal(nn.Module):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        n_fft: int = N_FFT,
        hop_size: int = HOP_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        context_frames: int = CONTEXT_FRAMES,
        **kwargs,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.hidden_size = hidden_size
        self.context_frames = context_frames

        # Per-context-channel features:
        #   log magnitude for each context frame
        #   delta-to-current log magnitude for older frames
        #   current-frame envelope repeated over TF plane
        #   current-frame signed waveform repeated over TF plane
        in_channels = context_frames + max(0, context_frames - 1) + 2
        self.unet = SpectralTemporalUNet(in_ch=in_channels, base_ch=32)
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

    def _prepare_features(self, audio_ctx: torch.Tensor):
        # audio_ctx: (B, C, FRAME_SIZE)
        B, C, T = audio_ctx.shape
        flat = audio_ctx.reshape(B * C, T)
        spec_all = self._stft(flat)  # (B*C, F, TT)
        Freq, Time = spec_all.shape[-2], spec_all.shape[-1]
        spec_all = spec_all.reshape(B, C, Freq, Time)

        mag_all = torch.abs(spec_all)
        log_mag_all = safe_log(mag_all)

        current_spec = spec_all[:, -1]
        current_mag = mag_all[:, -1]
        current_phase = torch.angle(current_spec)
        current_log_mag = log_mag_all[:, -1]

        feat_list = [log_mag_all]
        if C > 1:
            past_delta = log_mag_all[:, :-1] - current_log_mag.unsqueeze(1)
            feat_list.append(past_delta)

        env = torch.sqrt((audio_ctx[:, -1] ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        env_plane = env.view(B, 1, 1, 1).expand(B, 1, Freq, Time)

        current_wave = audio_ctx[:, -1]
        signed_mean = current_wave.mean(dim=-1, keepdim=True)
        signed_plane = signed_mean.view(B, 1, 1, 1).expand(B, 1, Freq, Time)

        feat_list.append(env_plane)
        feat_list.append(signed_plane)
        features = torch.cat(feat_list, dim=1)

        return features, current_log_mag, current_mag, current_phase

    def forward(self, audio_ctx: torch.Tensor):
        # audio_ctx: (B, C, FRAME_SIZE) or legacy (B, FRAME_SIZE)
        if audio_ctx.dim() == 2:
            audio_ctx = audio_ctx.unsqueeze(1)
        if audio_ctx.shape[1] != self.context_frames:
            if audio_ctx.shape[1] == 1:
                audio_ctx = audio_ctx.repeat(1, self.context_frames, 1)
            else:
                raise ValueError(
                    f"Expected {self.context_frames} context frames, got {audio_ctx.shape[1]}"
                )

        current_audio = audio_ctx[:, -1]
        length = current_audio.shape[-1]

        feat_tf, current_log_mag, current_mag, current_phase = self._prepare_features(audio_ctx)
        mask, residual, phase_delta = self.unet(feat_tf)

        out_log_mag = current_log_mag.unsqueeze(1) * mask + residual
        out_mag = torch.exp(out_log_mag.squeeze(1))
        out_phase = current_phase + phase_delta.squeeze(1)
        out_spec = torch.polar(out_mag, out_phase)

        audio_out = self._istft(out_spec, length=length)
        audio_out = self.transient(audio_out)

        # Keep a tiny dry skip from the current frame for stability.
        audio_out = 0.985 * audio_out + 0.015 * current_audio
        audio_out = torch.clamp(audio_out, -1.0, 1.0)

        features = {
            "input_mag": current_mag,
            "input_log_mag": current_log_mag,
            "context_frames": audio_ctx,
        }
        params = {
            "mask": mask.squeeze(1),
            "residual": residual.squeeze(1),
            "phase_delta": phase_delta.squeeze(1),
        }
        return audio_out, features, params

    def reset_phase(self):
        pass

    @torch.jit.export
    def infer_frame(self, audio_ctx: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out, _, _ = self.forward(audio_ctx)
        return out


DDSPGuitarToPiano = PolyphonicGuitarToPianoTemporal
