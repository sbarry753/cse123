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

import torch
import torchaudio
import json
import numpy as np
from bisect import bisect_right
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path

from model import FRAME_SIZE, SAMPLE_RATE


class AugmentedSubset(Dataset):
    """
    Wraps a Subset so augmentation state belongs to only this split.
    random_split() shares the original Dataset object between train/val,
    so toggling dataset.augment on one split would affect both.
    """

    def __init__(self, subset: Subset, augment: bool, virtual_copies: int = 1):
        self.subset = subset
        self.augment = augment
        self.virtual_copies = max(1, int(virtual_copies))

    def __len__(self):
        return len(self.subset) * self.virtual_copies

    def __getitem__(self, idx):
        guitar_frame, piano_frame = self.subset[idx % len(self.subset)]
        guitar_frame = guitar_frame.clone()
        piano_frame = piano_frame.clone()

        if self.augment:
            guitar_frame, piano_frame = augment_pair(guitar_frame, piano_frame)

        return guitar_frame, piano_frame


def augment_pair(guitar_frame: torch.Tensor, piano_frame: torch.Tensor):
    """
    Pair-preserving frame augmentation.

    Shared transforms keep guitar and piano time-aligned. Guitar-only
    perturbations make the encoder a little more robust without changing
    the piano target.
    """

    # Shared gain keeps relative guitar/piano dynamics intact while exposing
    # the model to quieter/louder recordings. +/-6 dB.
    gain_db = torch.empty(1).uniform_(-6.0, 6.0).item()
    gain = 10 ** (gain_db / 20.0)
    guitar_frame = guitar_frame * gain
    piano_frame = piano_frame * gain

    # Shared polarity flip is physically equivalent for raw waveforms and
    # doubles phase orientation examples without breaking the pair.
    if torch.rand(()) < 0.5:
        guitar_frame = -guitar_frame
        piano_frame = -piano_frame

    # Tiny shared circular shift simulates slightly different frame boundaries.
    # Keep it small so the target still corresponds to the same local event.
    max_shift = min(16, guitar_frame.numel() // 16)
    if max_shift > 0:
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        if shift:
            guitar_frame = torch.roll(guitar_frame, shifts=shift, dims=0)
            piano_frame = torch.roll(piano_frame, shifts=shift, dims=0)

    # Guitar-only low-level input noise improves robustness to pickup/interface
    # noise while preserving the desired clean piano target.
    if torch.rand(()) < 0.35:
        rms = torch.sqrt((guitar_frame ** 2).mean() + 1e-8)
        noise_db = torch.empty(1).uniform_(-42.0, -30.0).item()
        noise_scale = rms * (10 ** (noise_db / 20.0))
        guitar_frame = guitar_frame + torch.randn_like(guitar_frame) * noise_scale

    # DC offset removal after perturbations.
    guitar_frame = guitar_frame - guitar_frame.mean()
    piano_frame = piano_frame - piano_frame.mean()

    # Stay in the model's expected input/output range.
    guitar_frame = torch.clamp(guitar_frame, -1.0, 1.0)
    piano_frame = torch.clamp(piano_frame, -1.0, 1.0)
    return guitar_frame, piano_frame


class GuitarPianoDataset(Dataset):
    """
    Lazily loads matched guitar/piano frames from disk.

    The original implementation decoded every audio file and stored every
    frame in RAM during __init__. That is fine for small datasets, but large
    corpora can exceed system memory before training begins. This dataset only
    stores file paths and frame offsets up front, then loads a single frame pair
    in __getitem__.
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = SAMPLE_RATE,
        frame_size:  int = FRAME_SIZE,
        cache_dir: str | None = None,
        cache_dtype: str = "float16",
        f0_cache_dir: str | None = None,
        require_f0_cache: bool = False,
    ):
        self.data_dir    = Path(data_dir)
        self.sample_rate = sample_rate
        self.frame_size  = frame_size
        self.cache_dir   = Path(cache_dir) if cache_dir else None
        self.cache_dtype = np.dtype(cache_dtype)
        self.cache       = None
        self.cache_path  = None
        self.f0_cache_dir = Path(f0_cache_dir) if f0_cache_dir else None
        self.require_f0_cache = require_f0_cache

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

        self.clips = []
        self.cumulative_frames = []
        total_frames = 0
        for stem in common:
            g_path = guitar_map[stem]
            p_path = piano_map[stem]
            clip = self._make_clip_index(stem, g_path, p_path)
            if clip["n_frames"] == 0:
                continue
            if self.f0_cache_dir is not None:
                clip["f0_track"] = self._load_f0_track(stem, clip["n_frames"])
            self.clips.append(clip)
            total_frames += clip["n_frames"]
            self.cumulative_frames.append(total_frames)

        if self.cache_dir is not None:
            self._prepare_cache()

        print(f"Total training frames: {len(self):,}")

    def _make_clip_index(self, stem: str, guitar_path: Path, piano_path: Path) -> dict:
        g_info = torchaudio.info(str(guitar_path))
        p_info = torchaudio.info(str(piano_path))

        g_len = self._target_sample_count(g_info.num_frames, g_info.sample_rate)
        p_len = self._target_sample_count(p_info.num_frames, p_info.sample_rate)
        guitar_peak = self._measure_clip_peak(guitar_path)
        piano_peak = self._measure_clip_peak(piano_path)
        return {
            "stem": stem,
            "guitar_path": guitar_path,
            "piano_path": piano_path,
            "guitar_sr": g_info.sample_rate,
            "piano_sr": p_info.sample_rate,
            "guitar_peak": guitar_peak,
            "piano_peak": piano_peak,
            "n_frames": min(g_len, p_len) // self.frame_size,
        }

    def _load_f0_track(self, stem: str, n_frames: int) -> np.ndarray:
        f0_path = self.f0_cache_dir / f"{stem}.npy"
        if not f0_path.exists():
            if self.require_f0_cache:
                raise FileNotFoundError(f"Missing f0 cache for {stem}: {f0_path}")
            return np.zeros(n_frames, dtype=np.float32)

        f0 = np.asarray(np.load(f0_path), dtype=np.float32)
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        if f0.shape[0] < n_frames:
            f0 = np.pad(f0, (0, n_frames - f0.shape[0]), mode="constant")
        elif f0.shape[0] > n_frames:
            f0 = f0[:n_frames]
        return f0.astype(np.float32, copy=False)

    def _target_sample_count(self, num_frames: int, sample_rate: int) -> int:
        if sample_rate == self.sample_rate:
            return num_frames
        return int(num_frames * self.sample_rate / sample_rate)

    def _measure_clip_peak(self, path: Path) -> float:
        audio = self._load_clip(path, normalize=False)
        return float(audio.abs().max().item()) if audio.numel() else 0.0

    def _load_frame(
        self,
        path: Path,
        source_sample_rate: int,
        frame_idx: int,
        clip_peak: float,
    ) -> torch.Tensor:
        if source_sample_rate == self.sample_rate:
            audio, sr = torchaudio.load(
                str(path),
                frame_offset=frame_idx * self.frame_size,
                num_frames=self.frame_size,
            )
        else:
            # Resampling needs context from the original sample-rate timeline.
            # This fallback stays memory-bounded to one clip rather than the
            # whole dataset, and most prepared datasets should already match.
            audio, sr = torchaudio.load(str(path))

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        # Mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.squeeze(0)

        if sr != self.sample_rate:
            start = frame_idx * self.frame_size
            audio = audio[start:start + self.frame_size]

        if audio.numel() < self.frame_size:
            audio = torch.nn.functional.pad(audio, (0, self.frame_size - audio.numel()))

        # Match the cached path: every frame is scaled by the source clip peak,
        # preserving frame-to-frame dynamics for loudness learning.
        if clip_peak > 1e-8:
            audio = audio / clip_peak

        return audio

    def _metadata_payload(self) -> dict:
        files = []
        for clip in self.clips:
            for key in ("guitar_path", "piano_path"):
                path = clip[key]
                stat = path.stat()
                files.append({
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })

        return {
            "version": 1,
            "data_dir": str(self.data_dir.resolve()),
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "dtype": self.cache_dtype.name,
            "total_frames": len(self),
            "files": files,
        }

    def _prepare_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.cache_dir / "metadata.json"
        data_path = self.cache_dir / "frames.dat"
        expected_meta = self._metadata_payload()

        if meta_path.exists() and data_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if cached_meta == expected_meta:
                self.cache_path = data_path
                self._open_cache()
                print(f"Using frame cache: {self.cache_dir}")
                return

        print(f"Building frame cache: {self.cache_dir}")

        cache = np.memmap(
            data_path,
            dtype=self.cache_dtype,
            mode="w+",
            shape=(len(self), 2, self.frame_size),
        )

        cursor = 0
        for clip in self.clips:
            guitar_audio = self._load_clip(clip["guitar_path"],)
            piano_audio = self._load_clip(clip["piano_path"])

            n_samples = clip["n_frames"] * self.frame_size
            guitar_frames = guitar_audio[:n_samples].reshape(clip["n_frames"], self.frame_size)
            piano_frames = piano_audio[:n_samples].reshape(clip["n_frames"], self.frame_size)

            next_cursor = cursor + clip["n_frames"]
            cache[cursor:next_cursor, 0, :] = guitar_frames.numpy().astype(self.cache_dtype, copy=False)
            cache[cursor:next_cursor, 1, :] = piano_frames.numpy().astype(self.cache_dtype, copy=False)
            cursor = next_cursor

        cache.flush()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(expected_meta, f, indent=2)

        self.cache_path = data_path
        self._open_cache()

    def _open_cache(self) -> None:
        self.cache = np.memmap(
            self.cache_path,
            dtype=self.cache_dtype,
            mode="r",
            shape=(len(self), 2, self.frame_size),
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["cache"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self.cache_path is not None:
            self._open_cache()

    def _load_clip(self, path: Path, normalize: bool = True) -> torch.Tensor:
        audio, sr = torchaudio.load(str(path))

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.squeeze(0)
        peak = audio.abs().max()
        if normalize and peak > 1e-8:
            audio = audio / peak

        return audio

    def __len__(self):
        return self.cumulative_frames[-1] if self.cumulative_frames else 0

    def __getitem__(self, idx):
        if self.cache is not None:
            frames = self.cache[idx]
            guitar_frame = torch.from_numpy(np.array(frames[0], dtype=np.float32, copy=True))
            piano_frame = torch.from_numpy(np.array(frames[1], dtype=np.float32, copy=True))
            return guitar_frame, piano_frame

        clip_idx = bisect_right(self.cumulative_frames, idx)
        previous_total = 0 if clip_idx == 0 else self.cumulative_frames[clip_idx - 1]
        frame_idx = idx - previous_total
        clip = self.clips[clip_idx]

        guitar_frame = self._load_frame(
            clip["guitar_path"],
            clip["guitar_sr"],
            frame_idx,
            clip["guitar_peak"],
        )
        piano_frame = self._load_frame(
            clip["piano_path"],
            clip["piano_sr"],
            frame_idx,
            clip["piano_peak"],
        )
        return guitar_frame, piano_frame


class ContextFrameDataset(Dataset):
    """
    Returns a rolling guitar context plus the current piano target frame.

    The base dataset still stores/loads hop-sized frames. For item i, this
    wrapper concatenates the current frame and preceding frames from the same
    clip until context_size samples are available. Clip starts are left-padded
    with zeros instead of leaking context from the previous clip.
    """

    def __init__(self, frame_dataset: GuitarPianoDataset, context_size: int, hop_size: int):
        self.frame_dataset = frame_dataset
        self.context_size = int(context_size)
        self.hop_size = int(hop_size)

        if self.context_size < self.hop_size:
            raise ValueError("context_size must be >= hop_size")
        if self.context_size % self.hop_size != 0:
            raise ValueError("context_size must be an integer multiple of hop_size")
        if self.frame_dataset.frame_size != self.hop_size:
            raise ValueError(
                f"Base dataset frame_size={self.frame_dataset.frame_size} must match hop_size={self.hop_size}"
            )

        self.context_frames = self.context_size // self.hop_size

    def __len__(self):
        return len(self.frame_dataset)

    def _clip_position_for_index(self, idx: int) -> tuple[int, int, int]:
        clip_idx = bisect_right(self.frame_dataset.cumulative_frames, idx)
        clip_start = 0 if clip_idx == 0 else self.frame_dataset.cumulative_frames[clip_idx - 1]
        frame_idx = idx - clip_start
        return clip_idx, clip_start, frame_idx

    def __getitem__(self, idx):
        idx = int(idx)
        clip_idx, clip_start, local_frame_idx = self._clip_position_for_index(idx)
        first_idx = idx - self.context_frames + 1
        context_chunks = []

        for context_idx in range(first_idx, idx + 1):
            if context_idx < clip_start:
                context_chunks.append(torch.zeros(self.hop_size, dtype=torch.float32))
            else:
                guitar_frame, _ = self.frame_dataset[context_idx]
                context_chunks.append(guitar_frame)

        _, piano_frame = self.frame_dataset[idx]
        guitar_context = torch.cat(context_chunks, dim=0)
        clip = self.frame_dataset.clips[clip_idx]
        if "f0_track" in clip:
            f0_label = torch.tensor(float(clip["f0_track"][local_frame_idx]), dtype=torch.float32)
            return guitar_context, piano_frame, f0_label
        return guitar_context, piano_frame


class ContextAugmentedSubset(Dataset):
    """Augments context/target pairs without changing their alignment."""

    def __init__(self, subset: Subset, augment: bool, virtual_copies: int = 1):
        self.subset = subset
        self.augment = augment
        self.virtual_copies = max(1, int(virtual_copies))

    def __len__(self):
        return len(self.subset) * self.virtual_copies

    def __getitem__(self, idx):
        item = self.subset[idx % len(self.subset)]
        if len(item) == 3:
            guitar_context, piano_frame, f0_label = item
        else:
            guitar_context, piano_frame = item
            f0_label = None
        guitar_context = guitar_context.clone()
        piano_frame = piano_frame.clone()

        if self.augment:
            gain_db = torch.empty(1).uniform_(-6.0, 6.0).item()
            gain = 10 ** (gain_db / 20.0)
            guitar_context = guitar_context * gain
            piano_frame = piano_frame * gain

            if torch.rand(()) < 0.5:
                guitar_context = -guitar_context
                piano_frame = -piano_frame

            if torch.rand(()) < 0.35:
                rms = torch.sqrt((guitar_context ** 2).mean() + 1e-8)
                noise_db = torch.empty(1).uniform_(-42.0, -30.0).item()
                noise_scale = rms * (10 ** (noise_db / 20.0))
                guitar_context = guitar_context + torch.randn_like(guitar_context) * noise_scale

        guitar_context = guitar_context - guitar_context.mean()
        piano_frame = piano_frame - piano_frame.mean()
        guitar_context = torch.clamp(guitar_context, -1.0, 1.0)
        piano_frame = torch.clamp(piano_frame, -1.0, 1.0)
        if f0_label is not None:
            return guitar_context, piano_frame, f0_label
        return guitar_context, piano_frame


def make_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    augment: bool = True,
    augment_copies: int = 4,
    cache_dir: str | None = None,
    cache_dtype: str = "float16",
):
    """Create train/val DataLoaders from a data directory."""
    full_dataset = GuitarPianoDataset(
        data_dir,
        cache_dir=cache_dir,
        cache_dtype=cache_dtype,
    )

    n_val   = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_set = AugmentedSubset(train_set, augment=augment, virtual_copies=augment_copies)
    val_set = AugmentedSubset(val_set, augment=False)

    train_workers = 8
    val_workers = 4

    train_loader = DataLoader(
        train_set,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = train_workers,
        persistent_workers = True if train_workers > 0 else False,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = val_workers,
        persistent_workers = True if val_workers > 0 else False,
        pin_memory  = True,
    )

    return train_loader, val_loader


def make_context_dataloaders(
    data_dir: str,
    batch_size: int,
    context_size: int,
    hop_size: int,
    val_split: float = 0.1,
    augment: bool = True,
    augment_copies: int = 4,
    cache_dir: str | None = None,
    cache_dtype: str = "float16",
    f0_cache_dir: str | None = None,
    require_f0_cache: bool = False,
):
    """Create train/val DataLoaders that return guitar context and piano hop frames."""
    base_dataset = GuitarPianoDataset(
        data_dir,
        frame_size=hop_size,
        cache_dir=cache_dir,
        cache_dtype=cache_dtype,
        f0_cache_dir=f0_cache_dir,
        require_f0_cache=require_f0_cache,
    )
    full_dataset = ContextFrameDataset(base_dataset, context_size=context_size, hop_size=hop_size)

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_set = ContextAugmentedSubset(train_set, augment=augment, virtual_copies=augment_copies)
    val_set = ContextAugmentedSubset(val_set, augment=False)

    train_workers = 16
    val_workers = 8

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        persistent_workers=True if train_workers > 0 else False,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=val_workers,
        persistent_workers=True if val_workers > 0 else False,
        pin_memory=True,
    )

    return train_loader, val_loader
