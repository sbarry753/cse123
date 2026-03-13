import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


SAMPLE_RATE = 48000
BLOCK_SIZE = 128
CONTEXT_SAMPLES = 4096   # should match training / checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "checkpoints/causal_autoenc_epoch_029.pt"
INPUT_WAV = "dataset_48k/input/Guitar (1).wav"
OUTPUT_WAV = "streamed_output.wav"


# =========================================================
# Audio helpers
# =========================================================
def load_audio_mono(path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.astype(np.float32)
    audio = audio.mean(axis=1)
    audio = torch.from_numpy(audio).unsqueeze(0)  # [1, T]

    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    return audio


def compute_onset_envelope(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, 1, T]
    returns: [B, 1, T]
    """
    d = torch.zeros_like(x)
    d[:, :, 1:] = (x[:, :, 1:] - x[:, :, :-1]).abs()

    kernel = torch.ones(1, 1, 33, dtype=x.dtype, device=x.device) / 33.0
    return F.conv1d(d, kernel, padding=16)


# =========================================================
# Model (must match training script)
# =========================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, bias=True):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class CausalResBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        z = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        z = F.leaky_relu(self.norm2(self.conv2(z)), 0.2)
        return x + z


class CausalEncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=2):
        super().__init__()
        self.down = CausalConv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        self.res1 = CausalResBlock(out_ch, kernel_size=5, dilation=1)
        self.res2 = CausalResBlock(out_ch, kernel_size=5, dilation=2)
        self.res3 = CausalResBlock(out_ch, kernel_size=5, dilation=4)

    def forward(self, x):
        x = F.leaky_relu(self.down(x), 0.2)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class CausalUpsampleStage(nn.Module):
    def __init__(self, in_ch, out_ch, up_factor=2, kernel_size=5):
        super().__init__()
        self.up_factor = up_factor
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.smooth = CausalConv1d(out_ch, out_ch, kernel_size=kernel_size)
        self.res1 = CausalResBlock(out_ch, kernel_size=5, dilation=1)
        self.res2 = CausalResBlock(out_ch, kernel_size=5, dilation=2)
        self.res3 = CausalResBlock(out_ch, kernel_size=5, dilation=4)

    def forward(self, x):
        x = self.proj(x)
        x = x.repeat_interleave(self.up_factor, dim=-1)
        x = F.leaky_relu(self.smooth(x), 0.2)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x


class LatentProcessor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre = nn.ModuleList([
            CausalResBlock(channels, kernel_size=5, dilation=1),
            CausalResBlock(channels, kernel_size=5, dilation=2),
            CausalResBlock(channels, kernel_size=5, dilation=4),
            CausalResBlock(channels, kernel_size=5, dilation=8),
            CausalResBlock(channels, kernel_size=5, dilation=16),
        ])
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.post = nn.ModuleList([
            CausalResBlock(channels, kernel_size=5, dilation=1),
            CausalResBlock(channels, kernel_size=5, dilation=2),
            CausalResBlock(channels, kernel_size=5, dilation=4),
            CausalResBlock(channels, kernel_size=5, dilation=8),
        ])

    def forward(self, x, h0=None):
        for blk in self.pre:
            x = blk(x)

        x_seq = x.transpose(1, 2)
        x_seq, hN = self.gru(x_seq, h0)
        x = x_seq.transpose(1, 2)

        for blk in self.post:
            x = blk(x)

        return x, hN


class LiveCausalAutoencoder(nn.Module):
    """
    Must match training script.
    input waveform + onset envelope -> synthesized output
    """

    def __init__(self, base_ch=64):
        super().__init__()

        self.in_conv = CausalConv1d(2, base_ch, kernel_size=9)

        self.enc1 = CausalEncoderStage(base_ch,     base_ch * 2, kernel_size=7, stride=2)
        self.enc2 = CausalEncoderStage(base_ch * 2, base_ch * 4, kernel_size=7, stride=2)
        self.enc3 = CausalEncoderStage(base_ch * 4, base_ch * 4, kernel_size=7, stride=2)

        self.latent = LatentProcessor(base_ch * 4)

        self.dec1 = CausalUpsampleStage(base_ch * 4, base_ch * 4, up_factor=2)
        self.dec2 = CausalUpsampleStage(base_ch * 4, base_ch * 2, up_factor=2)
        self.dec3 = CausalUpsampleStage(base_ch * 2, base_ch,     up_factor=2)

        self.skip2_proj = nn.Conv1d(base_ch * 4, base_ch * 4, kernel_size=1)
        self.skip1_proj = nn.Conv1d(base_ch * 2, base_ch * 2, kernel_size=1)
        self.skip0_proj = nn.Conv1d(base_ch,     base_ch,     kernel_size=1)

        self.skip2_gain = nn.Parameter(torch.tensor(0.20))
        self.skip1_gain = nn.Parameter(torch.tensor(0.15))
        self.skip0_gain = nn.Parameter(torch.tensor(0.10))

        self.out_mid = CausalConv1d(base_ch, base_ch, kernel_size=5)
        self.out_conv = CausalConv1d(base_ch, 1, kernel_size=5)

    def forward(self, x, onset, h0=None, return_state=False):
        # x, onset: [B, 1, T]
        inp = torch.cat([x, onset], dim=1)

        x0 = F.leaky_relu(self.in_conv(inp), 0.2)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        z, hN = self.latent(x3, h0=h0)

        y = self.dec1(z)
        y = y + self.skip2_gain * self.skip2_proj(x2[:, :, -y.shape[-1]:])

        y = self.dec2(y)
        y = y + self.skip1_gain * self.skip1_proj(x1[:, :, -y.shape[-1]:])

        y = self.dec3(y)
        y = y + self.skip0_gain * self.skip0_proj(x0[:, :, -y.shape[-1]:])

        y = F.leaky_relu(self.out_mid(y), 0.2)
        y = torch.tanh(self.out_conv(y))

        out = y[:, :, -BLOCK_SIZE:]

        if return_state:
            return out, hN
        return out


# =========================================================
# Streaming inference
# =========================================================
@torch.no_grad()
def run_streaming(model, waveform: torch.Tensor, context_samples: int, block_size: int) -> torch.Tensor:
    """
    waveform: [1, T] CPU tensor
    returns:  [1, T] CPU tensor
    """
    model.eval()

    x = F.pad(waveform, (context_samples, 0))
    usable_len = x.shape[1] - context_samples
    num_blocks = (usable_len + block_size - 1) // block_size

    outs = []
    h = None
    x_work = x

    for i in range(num_blocks):
        start = i * block_size
        end = start + context_samples + block_size

        if end > x_work.shape[1]:
            x_work = F.pad(x_work, (0, end - x_work.shape[1]))

        window = x_work[:, start:end].unsqueeze(0).to(DEVICE)   # [1, 1, CONTEXT+BLOCK]
        onset = compute_onset_envelope(window)

        yb, h = model(window, onset, h0=h, return_state=True)
        h = h.detach()

        outs.append(yb.cpu().squeeze(0))  # [1, BLOCK]

    y = torch.cat(outs, dim=-1)
    return y[:, :waveform.shape[1]]


# =========================================================
# Main
# =========================================================
def main():
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    base_ch = ckpt.get("base_channels", 64)
    ckpt_sr = ckpt.get("sample_rate", SAMPLE_RATE)
    ckpt_block = ckpt.get("block_size", BLOCK_SIZE)
    ckpt_context = ckpt.get("context_samples", CONTEXT_SAMPLES)
    model_type = ckpt.get("model_type", "unknown")

    print(f"  model_type={model_type}")
    print(f"  base_ch={base_ch}  sr={ckpt_sr}  block={ckpt_block}  context={ckpt_context}")

    model = LiveCausalAutoencoder(base_ch=base_ch).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    print(f"Loading input: {INPUT_WAV}")
    x = load_audio_mono(INPUT_WAV, sample_rate=ckpt_sr)
    print(f"  Duration: {x.shape[1] / ckpt_sr:.2f}s ({x.shape[1]} samples)")

    print("Running streaming inference...")
    y = run_streaming(model, x, context_samples=ckpt_context, block_size=ckpt_block)

    peak = y.abs().max().item()
    if peak > 1e-8:
        y = y / peak * 0.99

    sf.write(OUTPUT_WAV, y.squeeze(0).numpy(), ckpt_sr)
    print(f"Saved: {OUTPUT_WAV}")


if __name__ == "__main__":
    main()