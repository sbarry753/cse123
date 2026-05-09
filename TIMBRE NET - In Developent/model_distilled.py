"""
MAX78000 student for guitar-to-piano timbre transfer.

This is written in the model format expected by ai8x-training's train.py.
Place it in lib/ai8x-training/models and select it with:

    --model ai85timbrestudent
"""
from torch import nn
import sys
from pathlib import Path

ai8x_dir = str(Path(__file__).resolve().parent.parent / "lib" / "ai8x-training")
sys.path.insert(0, str(ai8x_dir))
import ai8x

class TimbreStudent(nn.Module):
    """
    Same-resolution CNN that maps a log-magnitude spectrogram patch to mask logits.

    Expected input shape from the dataset entry:
        (num_channels, freq_bins, time_frames)

    The output keeps the same spatial shape:
        (1, freq_bins, time_frames)
    """

    def __init__(
            self,
            num_classes=1,
            num_channels=1,
            dimensions=(64, 64),
            base_ch=8,
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.l1 = ai8x.FusedConv2dBNReLU(
            num_channels, base_ch, 3, stride=1, padding=1,
            bias=bias, batchnorm='NoAffine', **kwargs
        )
        self.l2 = ai8x.FusedConv2dBNReLU(
            base_ch, base_ch * 2, 3, stride=1, padding=1,
            bias=bias, batchnorm='NoAffine', **kwargs
        )
        self.l3 = ai8x.FusedConv2dBNReLU(
            base_ch * 2, base_ch * 2, 3, stride=1, padding=1,
            bias=bias, batchnorm='NoAffine', **kwargs
        )
        self.l4 = ai8x.FusedConv2dBNReLU(
            base_ch * 2, base_ch, 3, stride=1, padding=1,
            bias=bias, batchnorm='NoAffine', **kwargs
        )
        self.out = ai8x.FusedConv2dBN(
            base_ch, num_classes, 1, stride=1, padding=0,
            bias=bias, batchnorm='NoAffine', **kwargs
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.out(x)


def timbrestudent(pretrained=False, **kwargs):
    """
    Factory used by train.py.

    pretrained is accepted for ai8x-training compatibility; this project does
    not provide pretrained weights through the model definition.
    """
    assert not pretrained
    return TimbreStudent(**kwargs)


models = [
    {
        'name': 'timbrestudent',
        'min_input': 1,
        'dim': 2,
    },
]