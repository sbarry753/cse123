import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


SAMPLE_RATE = 48000
BLOCK_SIZE = 128
CONTEXT_SAMPLES = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "checkpoints/live_timbre_epoch_023.pt"
INPUT_WAV = "dataset/input/plaz.wav"
OUTPUT_WAV = "streamed_output.wav"


def load_audio_mono(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.astype(np.float32)
    audio = audio.mean(axis=1)
    audio = torch.from_numpy(audio).unsqueeze(0)

    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    return audio


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class GatedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.res_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        h_f = self.filter_conv(x)
        h_g = self.gate_conv(x)
        z = torch.tanh(h_f) * torch.sigmoid(h_g)
        z = self.norm(z)
        residual = self.res_conv(z) + x
        skip = self.skip_conv(z)
        return residual, skip


class LiveTimbreNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.in_conv = CausalConv1d(1, channels, kernel_size=5, dilation=1)
        dilations = [1, 2, 4, 8, 16, 32, 64, 128]
        self.blocks = nn.ModuleList([GatedResidualBlock(channels, kernel_size=3, dilation=d) for d in dilations])
        self.post_1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.post_2 = nn.Conv1d(channels, 1, kernel_size=1)
        self.dry_gain = nn.Parameter(torch.tensor(0.15))
        self.wet_gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        dry = x[:, :, -BLOCK_SIZE:]
        z = F.leaky_relu(self.in_conv(x), 0.2)

        skip_sum = 0.0
        for block in self.blocks:
            z, skip = block(z)
            skip_sum = skip_sum + skip

        z = F.leaky_relu(skip_sum, 0.2)
        z = F.leaky_relu(self.post_1(z), 0.2)
        wet = torch.tanh(self.post_2(z))[:, :, -BLOCK_SIZE:]
        return self.dry_gain * dry + self.wet_gain * wet


@torch.no_grad()
def run_streaming(model, waveform: torch.Tensor) -> torch.Tensor:
    model.eval()

    x = waveform.clone()
    x = F.pad(x, (CONTEXT_SAMPLES, 0))

    usable_len = x.shape[1] - CONTEXT_SAMPLES
    num_blocks = (usable_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    outs = []

    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        end = start + CONTEXT_SAMPLES + BLOCK_SIZE

        if end > x.shape[1]:
            x_pad = F.pad(x, (0, end - x.shape[1]))
        else:
            x_pad = x

        window = x_pad[:, start:end].unsqueeze(0).to(DEVICE)  # [1,1,CONTEXT+BLOCK]
        yb = model(window).cpu().squeeze(0)                   # [1,BLOCK]
        outs.append(yb)

    y = torch.cat(outs, dim=-1)
    return y[:, :waveform.shape[1]]


def main():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model = LiveTimbreNet().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = load_audio_mono(INPUT_WAV)
    y = run_streaming(model, x)

    peak = max(y.abs().max().item(), 1e-8)
    if peak > 0.99:
        y = y / peak * 0.99

    sf.write(OUTPUT_WAV, y.squeeze(0).numpy(), SAMPLE_RATE)
    print(f"Saved {OUTPUT_WAV}")


if __name__ == "__main__":
    main()