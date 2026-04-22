"""
dataset.py — Paired Guitar / Piano Dataset

Expects a folder layout like:

  data/
    guitar/
      clip_001.wav
      clip_002.wav
      ...
    piano/
      clip_001.wav   ← same filenames, same duration, time-aligned
      clip_002.wav
      ...

Audio is chunked into FRAME_SIZE frames for training.
"""

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from model import FRAME_SIZE, SAMPLE_RATE


class GuitarPianoDataset(Dataset):
    """
    Loads matched guitar/piano pairs and returns (guitar_frame, piano_frame) tuples.
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = SAMPLE_RATE,
        frame_size:  int = FRAME_SIZE,
        augment:     bool = True,
    ):
        self.data_dir    = Path(data_dir)
        self.sample_rate = sample_rate
        self.frame_size  = frame_size
        self.augment     = augment

        guitar_dir = self.data_dir / 'guitar'
        piano_dir  = self.data_dir / 'piano'

        if not guitar_dir.exists() or not piano_dir.exists():
            raise FileNotFoundError(
                f"Expected {guitar_dir} and {piano_dir} to exist.\n"
                "Place your paired audio files there with matching filenames."
            )

        guitar_files = sorted(guitar_dir.glob('*.wav')) + sorted(guitar_dir.glob('*.flac'))
        piano_files  = sorted(piano_dir.glob('*.wav'))  + sorted(piano_dir.glob('*.flac'))

        # Match by stem name
        guitar_map = {f.stem: f for f in guitar_files}
        piano_map  = {f.stem: f for f in piano_files}
        common     = sorted(set(guitar_map) & set(piano_map))

        if not common:
            raise ValueError("No matching guitar/piano file pairs found (match by filename stem).")

        print(f"Found {len(common)} paired clips.")

        # Load all clips into memory (chunked into frames)
        self.frames = []
        for stem in common:
            g_path = guitar_map[stem]
            p_path = piano_map[stem]
            guitar_frames, piano_frames = self._load_pair(g_path, p_path)
            self.frames.extend(zip(guitar_frames, piano_frames))

        print(f"Total training frames: {len(self.frames):,}")

    def _load_pair(self, guitar_path: Path, piano_path: Path):
        g_audio, g_sr = torchaudio.load(str(guitar_path))
        p_audio, p_sr = torchaudio.load(str(piano_path))

        # Resample if needed
        if g_sr != self.sample_rate:
            g_audio = torchaudio.functional.resample(g_audio, g_sr, self.sample_rate)
        if p_sr != self.sample_rate:
            p_audio = torchaudio.functional.resample(p_audio, p_sr, self.sample_rate)

        # Mono
        if g_audio.shape[0] > 1:
            g_audio = g_audio.mean(dim=0, keepdim=True)
        if p_audio.shape[0] > 1:
            p_audio = p_audio.mean(dim=0, keepdim=True)

        g_audio = g_audio.squeeze(0)
        p_audio = p_audio.squeeze(0)

        # Trim/pad to same length
        min_len = min(g_audio.shape[0], p_audio.shape[0])
        g_audio = g_audio[:min_len]
        p_audio = p_audio[:min_len]

        # Normalise to [-1, 1]
        g_audio = g_audio / (g_audio.abs().max() + 1e-8)
        p_audio = p_audio / (p_audio.abs().max() + 1e-8)

        # Chunk into frames (drop last incomplete frame)
        n_frames = min_len // self.frame_size
        g_frames = g_audio[:n_frames * self.frame_size].reshape(n_frames, self.frame_size)
        p_frames = p_audio[:n_frames * self.frame_size].reshape(n_frames, self.frame_size)

        return list(g_frames), list(p_frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        guitar_frame, piano_frame = self.frames[idx]

        if self.augment:
            # Random amplitude scaling ±6dB
            scale = 10 ** (torch.empty(1).uniform_(-0.5, 0.5).item())
            guitar_frame = guitar_frame * scale
            piano_frame  = piano_frame  * scale
            # Small DC offset removal (common in real recordings)
            guitar_frame = guitar_frame - guitar_frame.mean()
            piano_frame  = piano_frame  - piano_frame.mean()

        return guitar_frame, piano_frame


def make_dataloaders(data_dir: str, batch_size: int = 64, val_split: float = 0.1):
    """Create train/val DataLoaders from a data directory."""
    full_dataset = GuitarPianoDataset(data_dir, augment=True)

    n_val   = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    # Disable augmentation for val
    val_set.dataset.augment = False

    train_loader = DataLoader(
        train_set,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

    return train_loader, val_loader
