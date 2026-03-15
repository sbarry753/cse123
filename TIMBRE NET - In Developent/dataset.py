"""
dataset.py — Paired Guitar / Piano Dataset with auto-alignment

Fixes for polyphonic-converted piano targets:
- estimates small timing lag between guitar and piano per clip
- trims/pads after alignment
- uses overlapping frames
- drops mostly silent frames
- splits by CLIP, not by frame
- keeps augmentation only on train split
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import FRAME_SIZE, HOP_SIZE, SAMPLE_RATE


def _to_mono_resampled(path: Path, sample_rate: int) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))

    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    audio = audio.squeeze(0).float()
    audio = audio - audio.mean()

    peak = audio.abs().max()
    if peak > 0:
        audio = audio / (peak + 1e-8)

    return audio


def _trim_shared_silence(
    guitar: torch.Tensor,
    piano: torch.Tensor,
    threshold: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trim leading/trailing regions where BOTH signals are basically silent.
    Keeps note tails if either side still has energy.
    """
    energy = torch.maximum(guitar.abs(), piano.abs())
    idx = torch.nonzero(energy > threshold, as_tuple=False).flatten()

    if len(idx) == 0:
        return guitar, piano

    start = int(idx[0].item())
    end = int(idx[-1].item()) + 1
    return guitar[start:end], piano[start:end]


def _frame_rms(x: torch.Tensor, win: int = 1024, hop: int = 256) -> torch.Tensor:
    """
    Cheap amplitude envelope for alignment.
    """
    if x.numel() < win:
        x = F.pad(x, (0, win - x.numel()))

    frames = x.unfold(0, win, hop)
    rms = torch.sqrt((frames ** 2).mean(dim=-1) + 1e-8)
    return rms


def _estimate_lag_samples(
    guitar: torch.Tensor,
    piano: torch.Tensor,
    sample_rate: int,
    max_shift_ms: float = 120.0,
) -> int:
    """
    Estimate lag using cross-correlation of RMS envelopes.

    Positive lag means piano starts later than guitar
    and piano should be shifted LEFT by that many samples.
    """
    env_win = 1024
    env_hop = 256

    g_env = _frame_rms(guitar, env_win, env_hop)
    p_env = _frame_rms(piano, env_win, env_hop)

    g_env = g_env - g_env.mean()
    p_env = p_env - p_env.mean()

    max_shift_samples = int(sample_rate * max_shift_ms / 1000.0)
    max_shift_frames = max(1, max_shift_samples // env_hop)

    best_lag = 0
    best_score = -float("inf")

    g_len = g_env.numel()
    p_len = p_env.numel()

    for lag in range(-max_shift_frames, max_shift_frames + 1):
        if lag >= 0:
            g_start = 0
            p_start = lag
        else:
            g_start = -lag
            p_start = 0

        overlap = min(g_len - g_start, p_len - p_start)
        if overlap < 8:
            continue

        g_seg = g_env[g_start:g_start + overlap]
        p_seg = p_env[p_start:p_start + overlap]

        score = torch.dot(g_seg, p_seg).item()
        if score > best_score:
            best_score = score
            best_lag = lag

    return best_lag * env_hop

def _apply_lag(
    guitar: torch.Tensor,
    piano: torch.Tensor,
    lag_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Positive lag => piano is delayed, so trim piano front.
    Negative lag => guitar is delayed, so trim guitar front.
    """
    if lag_samples > 0:
        piano = piano[lag_samples:]
    elif lag_samples < 0:
        guitar = guitar[-lag_samples:]

    n = min(len(guitar), len(piano))
    return guitar[:n], piano[:n]


def _chunk_audio(
    guitar: torch.Tensor,
    piano: torch.Tensor,
    frame_size: int,
    hop_size: int,
    min_rms: float = 0.01,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Overlapping framed pairs, dropping mostly silent regions.
    """
    n = min(len(guitar), len(piano))
    guitar = guitar[:n]
    piano = piano[:n]

    if n < frame_size:
        return []

    pairs = []
    for start in range(0, n - frame_size + 1, hop_size):
        g = guitar[start:start + frame_size]
        p = piano[start:start + frame_size]

        # Drop frames where both sides are near silent
        g_rms = torch.sqrt((g ** 2).mean() + 1e-8).item()
        p_rms = torch.sqrt((p ** 2).mean() + 1e-8).item()
        if max(g_rms, p_rms) < min_rms:
            continue

        pairs.append((g, p))

    return pairs


class GuitarPianoDataset(Dataset):
    """
    Loads matched guitar/piano pairs and returns aligned frame pairs.
    """

    def __init__(
        self,
        data_dir: str,
        stems: List[str] | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        hop_size: int | None = None,
        augment: bool = True,
        max_shift_ms: float = 120.0,
        min_rms: float = 0.01,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else max(1, frame_size // 4)
        self.augment = augment
        self.max_shift_ms = max_shift_ms
        self.min_rms = min_rms

        guitar_dir = self.data_dir / "guitar"
        piano_dir = self.data_dir / "piano"

        if not guitar_dir.exists() or not piano_dir.exists():
            raise FileNotFoundError(
                f"Expected {guitar_dir} and {piano_dir} to exist.\n"
                "Place paired audio there with matching filenames."
            )

        guitar_files = sorted(guitar_dir.glob("*.wav")) + sorted(guitar_dir.glob("*.flac"))
        piano_files = sorted(piano_dir.glob("*.wav")) + sorted(piano_dir.glob("*.flac"))

        guitar_map = {f.stem: f for f in guitar_files}
        piano_map = {f.stem: f for f in piano_files}

        common = sorted(set(guitar_map) & set(piano_map))
        if stems is not None:
            common = [s for s in common if s in set(stems)]

        if not common:
            raise ValueError("No matching guitar/piano file pairs found.")

        self.frames: List[Tuple[torch.Tensor, torch.Tensor]] = []

        print(f"Found {len(common)} paired clips.")
        for stem in common:
            g_path = guitar_map[stem]
            p_path = piano_map[stem]

            clip_pairs = self._load_pair(g_path, p_path)
            self.frames.extend(clip_pairs)

        print(f"Total aligned training frames: {len(self.frames):,}")

    def _load_pair(self, guitar_path: Path, piano_path: Path):
        guitar = _to_mono_resampled(guitar_path, self.sample_rate)
        piano = _to_mono_resampled(piano_path, self.sample_rate)

        lag = _estimate_lag_samples(
            guitar,
            piano,
            sample_rate=self.sample_rate,
            max_shift_ms=self.max_shift_ms,
        )

        guitar, piano = _apply_lag(guitar, piano, lag)
        guitar, piano = _trim_shared_silence(guitar, piano, threshold=1e-3)

        # Re-normalize after trimming/alignment
        g_peak = guitar.abs().max()
        p_peak = piano.abs().max()
        if g_peak > 0:
            guitar = guitar / (g_peak + 1e-8)
        if p_peak > 0:
            piano = piano / (p_peak + 1e-8)

        print(
            f"  {guitar_path.stem}: estimated lag = {lag} samples "
            f"({1000.0 * lag / self.sample_rate:+.1f} ms)"
        )

        return _chunk_audio(
            guitar,
            piano,
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            min_rms=self.min_rms,
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        guitar_frame, piano_frame = self.frames[idx]

        # clone so augmentation never mutates stored tensors
        guitar_frame = guitar_frame.clone()
        piano_frame = piano_frame.clone()

        if self.augment:
            # Shared gain jitter so mapping stays consistent
            gain = 10 ** random.uniform(-0.2, 0.2)
            guitar_frame *= gain
            piano_frame *= gain

            # Tiny input noise only on guitar to improve robustness
            guitar_frame += 0.001 * torch.randn_like(guitar_frame)

            # Remove tiny DC offsets
            guitar_frame = guitar_frame - guitar_frame.mean()
            piano_frame = piano_frame - piano_frame.mean()

            # Clamp
            guitar_frame = torch.clamp(guitar_frame, -1.0, 1.0)
            piano_frame = torch.clamp(piano_frame, -1.0, 1.0)

        return guitar_frame, piano_frame


def make_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    sample_rate: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE,
    hop_size: int | None = None,
    max_shift_ms: float = 120.0,
    min_rms: float = 0.01,
    seed: int = 42,
):
    """
    Split by clip stem, not by frame, so validation is honest.
    """
    data_dir = Path(data_dir)
    guitar_dir = data_dir / "guitar"
    piano_dir = data_dir / "piano"

    guitar_files = sorted(guitar_dir.glob("*.wav")) + sorted(guitar_dir.glob("*.flac"))
    piano_files = sorted(piano_dir.glob("*.wav")) + sorted(piano_dir.glob("*.flac"))

    guitar_stems = {f.stem for f in guitar_files}
    piano_stems = {f.stem for f in piano_files}
    common = sorted(guitar_stems & piano_stems)

    if not common:
        raise ValueError("No matching guitar/piano stems found.")

    rng = random.Random(seed)
    rng.shuffle(common)

    n_val = max(2, int(round(len(common) * val_split))) if len(common) > 2 else 1
    val_stems = common[:n_val]
    train_stems = common[n_val:] if n_val > 0 else common

    train_set = GuitarPianoDataset(
        data_dir=str(data_dir),
        stems=train_stems,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        augment=True,
        max_shift_ms=max_shift_ms,
        min_rms=min_rms,
    )

    val_set = GuitarPianoDataset(
        data_dir=str(data_dir),
        stems=val_stems,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        augment=False,
        max_shift_ms=max_shift_ms,
        min_rms=min_rms,
    ) if len(val_stems) > 0 else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=len(train_set) >= batch_size,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    ) if val_set is not None else None

    return train_loader, val_loader