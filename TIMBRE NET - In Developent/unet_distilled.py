"""
MAX78000-compatible small U-Net student for timbre-transfer KD.

The network itself uses ai8x-supported modules: fused convolution, fused
pool-convolution, fused transposed convolution, and element-wise add. Inputs
with odd spectrogram sizes should be padded by the caller before entering the
network, then cropped back after inference.
"""
from pathlib import Path
import sys

from torch import nn

ai8x_dir = str(Path(__file__).resolve().parent.parent / "lib" / "ai8x-training")
sys.path.insert(0, ai8x_dir)
import ai8x


class TimbreUNetStudent(nn.Module):
    """
    Compact additive-skip U-Net for mask + residual spectrogram prediction.

    Expected input shape is (N, 1, padded_freq_bins, padded_time_frames), where
    both spatial dimensions are divisible by 4. The output preserves that padded
    shape. The caller crops it to the original spectrogram dimensions.
    """

    def __init__(
        self,
        num_classes=2,
        num_channels=1,
        dimensions=(516, 8),
        base_ch=8,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        freq_bins, time_frames = dimensions
        if freq_bins % 4 != 0 or time_frames % 4 != 0:
            raise ValueError(
                "TimbreUNetStudent dimensions must be divisible by 4; "
                f"got {dimensions}. Pad before calling the model."
            )

        self.enc0 = ai8x.FusedConv2dBNReLU(
            num_channels, base_ch, 3, stride=1, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )
        self.enc0_refine = ai8x.FusedConv2dBNReLU(
            base_ch, base_ch, 3, stride=1, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )

        self.down1 = ai8x.FusedMaxPoolConv2dBNReLU(
            base_ch, base_ch * 2, 3, pool_size=2, pool_stride=2, stride=1, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )
        self.enc1_refine = ai8x.FusedConv2dBNReLU(
            base_ch * 2, base_ch * 2, 3, stride=1, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )

        self.down2 = ai8x.FusedMaxPoolConv2dBNReLU(
            base_ch * 2, base_ch * 4, 3, pool_size=2, pool_stride=2, stride=1, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )
        self.bottleneck = ai8x.FusedConv2dBNReLU(
            base_ch * 4, base_ch * 4, 3, stride=1, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )

        self.up1 = ai8x.FusedConvTranspose2dBNReLU(
            base_ch * 4, base_ch * 2, 3, stride=2, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )
        self.skip1 = ai8x.Add()
        self.dec1 = ai8x.FusedConv2dBNReLU(
            base_ch * 2, base_ch * 2, 3, stride=1, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )

        self.up0 = ai8x.FusedConvTranspose2dBNReLU(
            base_ch * 2, base_ch, 3, stride=2, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )
        self.skip0 = ai8x.Add()
        self.dec0 = ai8x.FusedConv2dBNReLU(
            base_ch, base_ch, 3, stride=1, padding=1,
            bias=bias, batchnorm="Affine", **kwargs
        )

        self.out = ai8x.FusedConv2dBN(
            base_ch, num_classes, 1, stride=1, padding=0,
            bias=bias, batchnorm="Affine", **kwargs
        )

    def forward(self, x):
        enc0 = self.enc0_refine(self.enc0(x))
        enc1 = self.enc1_refine(self.down1(enc0))
        x = self.bottleneck(self.down2(enc1))

        x = self.up1(x)
        x = self.skip1(x, enc1)
        x = self.dec1(x)

        x = self.up0(x)
        x = self.skip0(x, enc0)
        x = self.dec0(x)
        return self.out(x)


def timbreunetstudent(pretrained=False, **kwargs):
    assert not pretrained
    return TimbreUNetStudent(**kwargs)


models = [
    {
        "name": "timbreunetstudent",
        "min_input": 1,
        "dim": 2,
    },
]
