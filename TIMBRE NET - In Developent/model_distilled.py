"""
MAX78000 student for guitar-to-piano timbre transfer.

This is written in the model format expected by ai8x-training's train.py.
Place it in lib/ai8x-training/models and select it with:

    --model ai85timbrestudent
"""
import torch
from torch import nn
import sys
from pathlib import Path

ai8x_dir = str(Path(__file__).resolve().parent.parent / "lib" / "ai8x-training")
sys.path.insert(0, str(ai8x_dir))
import ai8x

class TimbreStudent(nn.Module):
    """
    Same-resolution CNN that maps a log-magnitude spectrogram patch to spectral params.

    Expected input shape from the dataset entry:
        (num_channels, freq_bins, time_frames)

    The output keeps the same spatial shape. Use num_classes=1 for mask-only
    output, or num_classes=2 for mask + normalized-log residual output.
    """

    def __init__(
            self,
            num_classes=2,
            num_channels=1,
            dimensions=(64, 64),
            base_ch=8,
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.l1 = ai8x.FusedConv2dBNReLU(
            num_channels, base_ch, 3, stride=1, padding=1,
            bias=bias, batchnorm='Affine', **kwargs
        )
        self.l2 = ai8x.FusedConv2dBNReLU(
            base_ch, base_ch * 2, 3, stride=1, padding=1,
            bias=bias, batchnorm='Affine', **kwargs
        )
        self.l3 = ai8x.FusedConv2dBNReLU(
            base_ch * 2, base_ch * 2, 3, stride=1, padding=1,
            bias=bias, batchnorm='Affine', **kwargs
        )
        self.l4 = ai8x.FusedConv2dBNReLU(
            base_ch * 2, base_ch, 3, stride=1, padding=1,
            bias=bias, batchnorm='Affine', **kwargs
        )
        self.out = ai8x.FusedConv2dBN(
            base_ch, num_classes, 1, stride=1, padding=0,
            bias=bias, batchnorm='Affine', **kwargs
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


class TransientShaperStudent(nn.Module):
    """
    1D waveform transient shaper for the post-spectral DSP audio frame.

    Expected input shape is (N, 1, frame_size) by default:
        channel 0: pre-transient waveform

    Output shape is (N, 2, frame_size):
        channel 0: delta
        channel 1: gate
    """

    def __init__(
            self,
            num_classes=1,
            num_channels=1,
            dimensions=(1024, 1),
            base_ch=8,
            bias=True,
            **kwargs
        ):

        super().__init__()

        self.delta1 = ai8x.FusedConv1dReLU(
            num_channels, base_ch, 5, padding=2,
            bias=bias, **kwargs
        )
        self.delta2 = ai8x.FusedConv1dReLU(
            base_ch, base_ch, 5, padding=2,
            bias=bias, **kwargs
        )
        self.delta3 = ai8x.FusedConv1dReLU(
            base_ch, base_ch, 3, padding=1,
            bias=bias, **kwargs
        )
        self.delta_out = ai8x.Conv1d(
            base_ch, num_classes, 1, padding=0,
            bias=bias, **kwargs
        )

        gate_ch = max(4, base_ch // 2)
        self.gate1 = ai8x.FusedConv1dReLU(
            num_channels, gate_ch, 5, padding=2,
            bias=bias, **kwargs
        )
        self.gate_out = ai8x.Conv1d(
            gate_ch, num_classes, 1, padding=0,
            bias=bias, **kwargs
        )

        nn.init.zeros_(self.delta_out.op.weight)
        nn.init.zeros_(self.delta_out.op.bias)

    def forward(self, x):

        delta = self.delta1(x)
        delta = self.delta2(delta)
        delta = self.delta3(delta)
        delta = self.delta_out(delta)

        gate = self.gate1(torch.abs(x))
        gate = self.gate_out(gate)

        return torch.cat((delta, gate), dim=1)

def transientshaperstudent(pretrained=False, **kwargs):
    assert not pretrained
    return TransientShaperStudent(**kwargs)


models = [
    {
        'name': 'timbrestudent',
        'min_input': 1,
        'dim': 2,
    },
    {
        'name': 'transientshaperstudent',
        'min_input': 1,
        'dim': 1,
    },
]
