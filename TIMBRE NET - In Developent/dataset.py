"""
dataset.py — Paired Guitar / Piano Dataset with auto-alignment

Changes:
- gain jitter applies ONLY to guitar input
- keeps silent / near-silent frames so the model can learn silence
- still estimates small timing lag between guitar and piano per clip
- still trims/pads after alignment
- still uses overlapping frames
- still splits by CLIP, not by frame
- augmentation remains train-only
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import FRAME_SIZE, HOP_SIZE, SAMPLE_RATE
from data_splits import load_split_manifest, collect_stems, find_split_manifest

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
    min_rms: float = 0.002,
    keep_silence_prob: float = 1.0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Overlapping framed pairs.

    Keeps silent / near-silent regions so the model can learn:
      quiet guitar -> quiet piano
      silence -> silence

    If you ever want to thin quiet frames later, set
    keep_silence_prob to something below 1.0.
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

        g_rms = torch.sqrt((g ** 2).mean() + 1e-8).item()
        p_rms = torch.sqrt((p ** 2).mean() + 1e-8).item()

        if max(g_rms, p_rms) < min_rms:
            if random.random() > keep_silence_prob:
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
        cache_dir: str | None = None,
        stems: List[str] | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        hop_size: int | None = None,
        augment: bool = True,
        max_shift_ms: float = 120.0,
        min_rms: float = 0.002,
        keep_silence_prob: float = 1.0,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else max(1, frame_size // 4)
        self.augment = augment
        self.max_shift_ms = max_shift_ms
        self.min_rms = min_rms
        self.keep_silence_prob = keep_silence_prob

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
        self.clips: List[dict] = []
        self.cache_path: Path | None = None
        self.cache = None
        self.cache_len = 0

        print(f"Found {len(common)} paired clips.")

        for stem in common:
            g_path = guitar_map[stem]
            p_path = piano_map[stem]

            clip = self._make_clip_index(stem, g_path, p_path)
            if clip["n_frames"] == 0:
                continue
            self.clips.append(clip)

        if self.cache_dir is not None:
            self._prepare_cache()
        else:
            for clip in self.clips:
                clip_pairs = self._load_pair(clip["guitar_path"], clip["piano_path"])
                self.frames.extend(clip_pairs)

        print(f"Total aligned training frames: {len(self.frames):,}")
    
    # Indexes all clips with additional metadata to build cache
    def _make_clip_index(self, stem: str, guitar_path: Path, piano_path: Path) -> dict:
        
        def _target_sample_count(num_frames: int, sample_rate: int) -> int:
            if sample_rate == self.sample_rate:
                return num_frames
            return int(num_frames * self.sample_rate / sample_rate)
    
        g_info = torchaudio.info(str(guitar_path))
        p_info = torchaudio.info(str(piano_path))

        g_len = _target_sample_count(g_info.num_frames, g_info.sample_rate)
        p_len = _target_sample_count(p_info.num_frames, p_info.sample_rate)

        n = min(g_len, p_len)
        if n < self.frame_size:
            n_frames = 0
        else:
            n_frames = (n - self.frame_size) // self.hop_size + 1

        return {
            "stem": stem,
            "guitar_path": guitar_path,
            "piano_path": piano_path,
            "guitar_sr": g_info.sample_rate,
            "piano_sr": p_info.sample_rate,
            "n_frames": n_frames,
        }
    
    # Uses existing cache or builds data cache
    def _prepare_cache(self) -> None:
        assert self.cache_dir is not None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.cache_dir / "metadata.json"
        data_path = self.cache_dir / "frames.dat"
        expected_config = self._metadata_payload()

        if meta_path.exists() and data_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if cached_meta.get("config") == expected_config:
                self.cache_path = data_path
                self._open_cache(cached_meta["num_frames"])
                print(f"Using frame cache: {self.cache_dir}")
                return
            
        print(f"Building frame cache: {self.cache_dir}")

        max_frames = sum(clip["n_frames"] for clip in self.clips)
        if max_frames == 0:
            raise ValueError("No frames available to cache.")

        cache = np.memmap(
            data_path,
            dtype=np.float32,
            mode="w+",
            shape=(max_frames, 2, self.frame_size),
        )
        cursor = 0

        for clip in self.clips:
            clip_pairs = self._load_pair(clip["guitar_path"], clip["piano_path"])
            for guitar_frame, piano_frame in clip_pairs:
                if cursor >= max_frames:
                    raise RuntimeError(
                        "Cache frame estimate was too small. "
                        "Check _make_clip_index() frame counting."
                    )
                cache[cursor, 0, :] = guitar_frame.numpy()
                cache[cursor, 1, :] = piano_frame.numpy()
                cursor += 1

        cache.flush()

        metadata = {
            "config": expected_config,
            "num_frames": cursor,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.cache_path = data_path
        self._open_cache(cursor)
    
    # Cache metadata
    def _metadata_payload(self):
        return {
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
            "max_shift_ms": self.max_shift_ms,
            "min_rms": self.min_rms,
            "keep_silence_prob": self.keep_silence_prob,
            "clips": [
                {
                    "stem": clip["stem"],
                    "guitar_path": str(clip["guitar_path"]),
                    "piano_path": str(clip["piano_path"]),
                    "guitar_sr": clip["guitar_sr"],
                    "piano_sr": clip["piano_sr"],
                    "n_frames": clip["n_frames"],
                }
                for clip in self.clips
            ],
        }

    def _open_cache(self, num_frames: int):
        assert self.cache_path is not None

        self.cache_len = num_frames
        self.cache = np.memmap(
            self.cache_path,
            dtype=np.float32,
            mode="r",
            shape=(num_frames, 2, self.frame_size),
        )
        self.frames = [None] * num_frames


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
            keep_silence_prob=self.keep_silence_prob,
        )

    def __len__(self):
        return self.cache_len if self.cache is not None else len(self.frames)

    def __getitem__(self, idx):
        if self.cache is not None:
            guitar_frame = torch.from_numpy(self.cache[idx, 0].copy())
            piano_frame = torch.from_numpy(self.cache[idx, 1].copy())
        else:
            guitar_frame, piano_frame = self.frames[idx]

        # clone so augmentation never mutates stored tensors
        guitar_frame = guitar_frame.clone()
        piano_frame = piano_frame.clone()

        if self.augment:
            # Gain jitter ONLY on guitar input.
            # Do not scale the piano target.
            gain = 10 ** random.uniform(-0.2, 0.2)
            guitar_frame *= gain

            # Tiny input noise only on guitar to improve robustness
            guitar_frame += 0.0002 * torch.randn_like(guitar_frame)

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
    min_rms: float = 0.002,
    keep_silence_prob: float = 1.0,
    seed: int = 22,
    split_manifest: str | Path | None = None,
):
    """
    Split by clip stem, not by frame, so validation is honest.
    """
    data_dir = Path(data_dir)
    manifest_splits = load_split_manifest(data_dir, split_manifest)

    if manifest_splits is not None:
        train_stems = manifest_splits["train"]
        val_stems = manifest_splits["val"]
        if not val_stems:
            raise ValueError("Split manifest must include a non-empty val split for make_dataloaders().")
    else:
        guitar_stems = collect_stems(data_dir / "guitar")
        piano_stems = collect_stems(data_dir / "piano")
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
        keep_silence_prob=keep_silence_prob,
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
        keep_silence_prob=keep_silence_prob,
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