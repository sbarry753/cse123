"""
max_dataset.py - Guitar-piano dataset for MAX78000
"""

import random
from pathlib import Path
from typing import List
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import GuitarPianoDataset, SAMPLE_RATE, FRAME_SIZE

class MAXGuitarPianoDataset(Dataset):
    """
    MAX78000 spectrogram dataset.

    Returns guitar log-magnitude spectrograms as inputs and clipped piano/guitar
    magnitude masks as regression labels. Both are stored in [0, 1].
    """

    def __init__(
            self,
            root_dir: str,
            d_type: str,
            transform=None,
            sample_rate: int = SAMPLE_RATE,
            frame_size: int = FRAME_SIZE,
            hop_size: int | None = None,
            n_fft: int = FRAME_SIZE,
            cache_dir: str | None = None,
            val_split: float = 0.1,
            seed: int = 22,
            max_shift_ms: float = 120.0,
            min_rms: float = 0.002,
            keep_silence_prob: float = 1.0,
            log_scale: float = 6.0,
            truncate_testset: bool = False,
        ):
        self.root_dir = Path(root_dir)
        self.d_type = d_type
        self.transform = transform
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else max(1, frame_size // 4)
        self.n_fft = n_fft
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_dir / "processed" / "max78000"
        self.val_split = val_split
        self.seed = seed
        self.max_shift_ms = max_shift_ms
        self.min_rms = min_rms
        self.keep_silence_prob = keep_silence_prob
        self.log_scale = log_scale
        self.truncate_testset = truncate_testset
        self.window = torch.hann_window(frame_size)

        if d_type not in {"train", "test"}:
            raise ValueError(f"Unknown dataset type: {d_type}")

        split = self._split_stems()
        self.stems = split["train"] if d_type == "train" else split["test"]
        self.freq_bins = self.n_fft // 2 + 1
        self.time_frames = self._spectrogram_mag(torch.zeros(self.frame_size)).shape[-1]

        self.cache_dir = self.cache_dir / d_type
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "metadata.json"
        self.data_path = self.cache_dir / "spectrograms.dat"
        self.data = None
        self.num_frames = 0

        self._prepare_cache()

        if self.truncate_testset and self.d_type == "test":
            self.num_frames = min(self.num_frames, 1)

    def _split_stems(self) -> dict[str, List[str]]:
        guitar_dir = self.root_dir / "guitar"
        piano_dir = self.root_dir / "piano"

        guitar_files = sorted(guitar_dir.glob("*.wav")) + sorted(guitar_dir.glob("*.flac"))
        piano_files = sorted(piano_dir.glob("*.wav")) + sorted(piano_dir.glob("*.flac"))
        common = sorted({f.stem for f in guitar_files} & {f.stem for f in piano_files})

        if not common:
            raise ValueError("No matching guitar/piano stems found.")

        rng = random.Random(self.seed)
        rng.shuffle(common)

        n_val = max(2, int(round(len(common) * self.val_split))) if len(common) > 2 else 1
        return {
            "test": common[:n_val],
            "train": common[n_val:] if n_val > 0 else common,
        }

    def _metadata_payload(self) -> dict:
        return {
            "stems": self.stems,
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
            "n_fft": self.n_fft,
            "freq_bins": self.freq_bins,
            "time_frames": self.time_frames,
            "max_shift_ms": self.max_shift_ms,
            "min_rms": self.min_rms,
            "keep_silence_prob": self.keep_silence_prob,
            "log_scale": self.log_scale,
        }

    def _prepare_cache(self) -> None:
        expected_meta = self._metadata_payload()

        if self.meta_path.exists() and self.data_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if cached_meta.get("config") == expected_meta:
                self._open_cache(cached_meta["num_frames"])
                print(f"Using MAX78000 spectrogram cache: {self.cache_dir}")
                return

        print(f"Building MAX78000 spectrogram cache: {self.cache_dir}")
        waveform_ds = GuitarPianoDataset(
            data_dir=str(self.root_dir),
            cache_dir=str(self.cache_dir / "waveforms"),
            stems=self.stems,
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            augment=False,
            max_shift_ms=self.max_shift_ms,
            min_rms=self.min_rms,
            keep_silence_prob=self.keep_silence_prob,
        )

        cache = np.memmap(
            self.data_path,
            dtype=np.float32,
            mode="w+",
            shape=(len(waveform_ds), 2, self.freq_bins, self.time_frames),
        )

        for idx in range(len(waveform_ds)):
            guitar_frame, piano_frame = waveform_ds[idx]
            guitar_mag = self._spectrogram_mag(guitar_frame)
            piano_mag = self._spectrogram_mag(piano_frame)
            input_spec = self._normalize_log_mag(guitar_mag)
            target_mask = torch.clamp(piano_mag / (guitar_mag + 1e-5), 0.0, 2.0) / 2.0

            cache[idx, 0, :, :] = input_spec.numpy()
            cache[idx, 1, :, :] = target_mask.numpy()

        cache.flush()
        metadata = {
            "config": expected_meta,
            "num_frames": len(waveform_ds),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._open_cache(len(waveform_ds))

    def _open_cache(self, num_frames: int) -> None:
        self.num_frames = num_frames
        self.data = np.memmap(
            self.data_path,
            dtype=np.float32,
            mode="r",
            shape=(num_frames, 2, self.freq_bins, self.time_frames),
        )

    def _spectrogram_mag(self, frame: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            frame.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.frame_size,
            window=self.window,
            return_complex=True,
            center=True,
        )
        return torch.abs(spec)

    def _normalize_log_mag(self, mag: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.log1p(mag) / self.log_scale, 0.0, 1.0)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        input_spec = torch.from_numpy(self.data[idx, 0].copy()).unsqueeze(0).float()
        target_mask = torch.from_numpy(self.data[idx, 1].copy()).unsqueeze(0).float()

        if self.transform is not None:
            input_spec = self.transform(input_spec)

        return input_spec, target_mask

def MAXGuitarPiano_get_datasets(data, load_train=True, load_test=True):
    """Load MAX78000 guitar-to-piano spectrogram regression datasets."""
    import ai8x 

    data_dir, args = data
    transform = ai8x.normalize(args=args)

    if load_train:
        train_dataset = MAXGuitarPianoDataset(
            root_dir=data_dir,
            d_type="train",
            transform=transform,
        )
        print(f"Train dataset length: {len(train_dataset)}\n")
    else:
        train_dataset = None

    if load_test:
        test_dataset = MAXGuitarPianoDataset(
            root_dir=data_dir,
            d_type="test",
            transform=transform,
            truncate_testset=getattr(args, "truncate_testset", False),
        )
        print(f"Test dataset length: {len(test_dataset)}\n")
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        "name": "MAXGuitarPiano",
        "input": (1, FRAME_SIZE // 2 + 1, 5),
        "output": ("mask",),
        "loader": MAXGuitarPiano_get_datasets,
        "regression": True,
    },
]