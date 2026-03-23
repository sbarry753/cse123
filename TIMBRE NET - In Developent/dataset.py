from __future__ import annotations

"""
Temporal-context + onset-aligned + cached dataset for direct guitar -> piano transfer.

What this version does:
- preserves the quiet-DI / loud-piano relationship using shared clip scaling
- finds a clip-level lag using onset envelopes
- locally refines target timing around onset-heavy windows
- feeds the model a short history of past guitar frames as temporal context
- caches the final contextual tensors so retraining skips rescanning/alignment
"""

import hashlib
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import FRAME_SIZE, HOP_SIZE, SAMPLE_RATE, CONTEXT_FRAMES

CACHE_VERSION = 3


def _to_mono_resampled_raw(path: Path, sample_rate: int) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0).float()
    audio = audio - audio.mean()
    return audio.contiguous()


def _trim_shared_silence(guitar: torch.Tensor, piano: torch.Tensor, threshold: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    energy = torch.maximum(guitar.abs(), piano.abs())
    idx = torch.nonzero(energy > threshold, as_tuple=False).flatten()
    if idx.numel() == 0:
        return guitar, piano
    start = int(idx[0].item())
    end = int(idx[-1].item()) + 1
    return guitar[start:end].contiguous(), piano[start:end].contiguous()


def _moving_average(x: torch.Tensor, win: int) -> torch.Tensor:
    if win <= 1 or x.numel() <= 2:
        return x
    pad = win // 2
    x_pad = F.pad(x[None, None], (pad, pad), mode="replicate")
    kernel = torch.ones(1, 1, win, device=x.device, dtype=x.dtype) / float(win)
    y = F.conv1d(x_pad, kernel)
    return y[0, 0, : x.numel()]


def _onset_envelope(x: torch.Tensor, sample_rate: int) -> torch.Tensor:
    # raw-waveform onset proxy: remove slow trend, half-wave rectify derivative, smooth
    slow = _moving_average(x, max(5, int(0.020 * sample_rate)))
    hp = x - slow
    dx = hp[1:] - hp[:-1]
    dx = F.pad(dx, (1, 0))
    onset = torch.relu(dx)
    onset = _moving_average(onset, max(3, int(0.003 * sample_rate)))
    onset = onset / (onset.max() + 1e-8)
    return onset.contiguous()


def _frame_feature(x: torch.Tensor, win: int, hop: int, reduce: str = "mean") -> torch.Tensor:
    if x.numel() < win:
        x = F.pad(x, (0, win - x.numel()))
    frames = x.unfold(0, win, hop)
    if reduce == "mean":
        return frames.mean(dim=-1)
    if reduce == "max":
        return frames.max(dim=-1).values
    raise ValueError(f"Unknown reduce={reduce}")


def _peak_pick(x: torch.Tensor, min_distance: int, threshold_rel: float = 0.20) -> torch.Tensor:
    if x.numel() < 3:
        return torch.empty(0, dtype=torch.long)
    thr = float(x.max().item()) * threshold_rel
    peaks: List[int] = []
    last = -min_distance
    for i in range(1, x.numel() - 1):
        xi = float(x[i].item())
        if xi < thr:
            continue
        if xi >= float(x[i - 1].item()) and xi > float(x[i + 1].item()):
            if i - last >= min_distance:
                peaks.append(i)
                last = i
            elif peaks and xi > float(x[peaks[-1]].item()):
                peaks[-1] = i
                last = i
    return torch.tensor(peaks, dtype=torch.long)


def _estimate_global_lag_samples(guitar: torch.Tensor, piano: torch.Tensor, sample_rate: int, max_shift_ms: float) -> int:
    env_hop = 64
    env_win = 256
    g_env = _frame_feature(_onset_envelope(guitar, sample_rate), env_win, env_hop, reduce="mean")
    p_env = _frame_feature(_onset_envelope(piano, sample_rate), env_win, env_hop, reduce="mean")
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
            g_start, p_start = 0, lag
        else:
            g_start, p_start = -lag, 0
        overlap = min(g_len - g_start, p_len - p_start)
        if overlap < 8:
            continue
        score = torch.dot(g_env[g_start:g_start + overlap], p_env[p_start:p_start + overlap]).item()
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag * env_hop


def _apply_lag(guitar: torch.Tensor, piano: torch.Tensor, lag_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if lag_samples > 0:
        piano = piano[lag_samples:]
    elif lag_samples < 0:
        guitar = guitar[-lag_samples:]
    n = min(guitar.numel(), piano.numel())
    return guitar[:n].contiguous(), piano[:n].contiguous()


def _matched_onset_lags(guitar: torch.Tensor, piano: torch.Tensor, sample_rate: int, search_ms: float = 80.0, peak_rel: float = 0.20) -> torch.Tensor:
    hop = 64
    win = 256
    g_env = _frame_feature(_onset_envelope(guitar, sample_rate), win, hop, reduce="mean")
    p_env = _frame_feature(_onset_envelope(piano, sample_rate), win, hop, reduce="mean")

    min_distance_frames = max(1, int(0.030 * sample_rate / hop))
    g_peaks = _peak_pick(g_env, min_distance=min_distance_frames, threshold_rel=peak_rel)
    p_peaks = _peak_pick(p_env, min_distance=min_distance_frames, threshold_rel=peak_rel)
    if g_peaks.numel() == 0 or p_peaks.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    max_delta_frames = max(1, int(search_ms * sample_rate / 1000.0 / hop))
    lags: List[int] = []
    p_list = p_peaks.tolist()
    for gp in g_peaks.tolist():
        best = None
        best_dist = 10**9
        for pp in p_list:
            d = pp - gp
            ad = abs(d)
            if ad <= max_delta_frames and ad < best_dist:
                best = d
                best_dist = ad
        if best is not None:
            lags.append(best * hop)
    if not lags:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(lags, dtype=torch.long)


def _robust_lag_from_onsets(guitar: torch.Tensor, piano: torch.Tensor, sample_rate: int, fallback_ms: float) -> int:
    coarse = _estimate_global_lag_samples(guitar, piano, sample_rate, max_shift_ms=fallback_ms)
    g2, p2 = _apply_lag(guitar, piano, coarse)
    matched = _matched_onset_lags(g2, p2, sample_rate)
    if matched.numel() == 0:
        return coarse
    return coarse + int(torch.median(matched).item())


def _collect_start_positions(guitar: torch.Tensor, piano: torch.Tensor, frame_size: int, hop_size: int, sample_rate: int) -> List[Tuple[int, bool]]:
    n = min(guitar.numel(), piano.numel())
    if n < frame_size:
        return []

    positions: List[Tuple[int, bool]] = [(start, False) for start in range(0, n - frame_size + 1, hop_size)]

    env = _frame_feature(_onset_envelope(guitar, sample_rate), 256, 64, reduce="mean")
    peaks = _peak_pick(env, min_distance=max(1, int(0.030 * sample_rate / 64)), threshold_rel=0.18)
    pre_roll = int(0.008 * sample_rate)

    for pk in peaks.tolist():
        sample_pos = pk * 64
        start = max(0, min(sample_pos - pre_roll, n - frame_size))
        positions.append((start, True))
        for extra in (-hop_size, hop_size):
            s2 = max(0, min(start + extra, n - frame_size))
            positions.append((s2, True))

    return sorted(set(positions), key=lambda x: (x[0], x[1]))


def _best_local_delay_fast(guitar_env: torch.Tensor, piano_env: torch.Tensor, center: int, frame_size: int, max_local_shift: int, step: int) -> int:
    g_seg = guitar_env[center:center + frame_size]
    if g_seg.numel() != frame_size:
        return 0
    g_seg = g_seg - g_seg.mean()
    g_norm = torch.linalg.norm(g_seg) + 1e-8

    best_delay = 0
    best_score = -float("inf")
    low = max(-max_local_shift, -center)
    high = min(max_local_shift, piano_env.numel() - frame_size - center)
    if high < low:
        return 0

    for d in range(low, high + 1, step):
        s = center + d
        p_seg = piano_env[s:s + frame_size]
        p_seg = p_seg - p_seg.mean()
        denom = g_norm * (torch.linalg.norm(p_seg) + 1e-8)
        score = torch.dot(g_seg, p_seg).item() / float(denom)
        if score > best_score:
            best_score = score
            best_delay = d

    fine_low = max(low, best_delay - step)
    fine_high = min(high, best_delay + step)
    for d in range(fine_low, fine_high + 1):
        s = center + d
        p_seg = piano_env[s:s + frame_size]
        p_seg = p_seg - p_seg.mean()
        denom = g_norm * (torch.linalg.norm(p_seg) + 1e-8)
        score = torch.dot(g_seg, p_seg).item() / float(denom)
        if score > best_score:
            best_score = score
            best_delay = d
    return best_delay


def _bright_repeats(p: torch.Tensor, frame_size: int, hop_size: int) -> int:
    win = torch.hann_window(frame_size, device=p.device, dtype=p.dtype)
    p_spec = torch.stft(
        p,
        n_fft=frame_size,
        hop_length=max(1, hop_size),
        win_length=frame_size,
        window=win,
        return_complex=True,
        center=True,
    )
    p_mag = torch.abs(p_spec).mean(dim=-1)
    split_bin = int(0.45 * p_mag.shape[0])
    low_e = p_mag[:split_bin].mean().item()
    high_e = p_mag[split_bin:].mean().item()
    bright_ratio = high_e / (low_e + 1e-8)
    if bright_ratio > 1.20:
        return 3
    if bright_ratio > 0.90:
        return 2
    return 1


def _build_contextual_pairs(
    guitar: torch.Tensor,
    piano: torch.Tensor,
    frame_size: int,
    hop_size: int,
    context_frames: int,
    sample_rate: int,
    min_rms: float,
    local_refine_ms: float,
    local_refine_step: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    n = min(guitar.numel(), piano.numel())
    guitar = guitar[:n].contiguous()
    piano = piano[:n].contiguous()
    if n < frame_size:
        return []

    positions = _collect_start_positions(guitar, piano, frame_size, hop_size, sample_rate)
    if not positions:
        return []

    starts_regular = list(range(0, n - frame_size + 1, hop_size))
    start_to_regular_idx = {s: i for i, s in enumerate(starts_regular)}
    max_local_shift = int(local_refine_ms * sample_rate / 1000.0)

    guitar_env = _onset_envelope(guitar, sample_rate)
    piano_env = _onset_envelope(piano, sample_rate)

    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx, (start, is_onset) in enumerate(positions):
        g_cur = guitar[start:start + frame_size]
        g_rms = torch.sqrt((g_cur ** 2).mean() + 1e-8).item()
        if g_rms < min_rms and not is_onset:
            continue

        do_refine = is_onset or (idx % 4 == 0)
        delay = _best_local_delay_fast(guitar_env, piano_env, start, frame_size, max_local_shift, local_refine_step) if do_refine else 0
        p_start = max(0, min(start + delay, piano.numel() - frame_size))
        p_cur = piano[p_start:p_start + frame_size]
        p_rms = torch.sqrt((p_cur ** 2).mean() + 1e-8).item()
        if max(g_rms, p_rms) < min_rms:
            continue

        base_idx = start_to_regular_idx.get(start)
        if base_idx is None:
            base_idx = int(round(start / max(1, hop_size)))
        ctx: List[torch.Tensor] = []
        for k in range(context_frames):
            src_i = max(0, base_idx - (context_frames - 1 - k))
            src_i = min(src_i, len(starts_regular) - 1)
            src_start = starts_regular[src_i]
            frame = guitar[src_start:src_start + frame_size]
            if frame.numel() < frame_size:
                frame = F.pad(frame, (0, frame_size - frame.numel()))
            ctx.append(frame)
        g_ctx = torch.stack(ctx, dim=0)

        repeats = _bright_repeats(p_cur, frame_size, hop_size)
        if is_onset:
            repeats += 2

        g_ctx = g_ctx.clone()
        p_cur = p_cur.clone()
        for _ in range(repeats):
            pairs.append((g_ctx, p_cur))
    return pairs


class GuitarPianoContextDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        stems: List[str] | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        hop_size: int | None = None,
        context_frames: int = CONTEXT_FRAMES,
        augment: bool = True,
        max_shift_ms: float = 140.0,
        min_rms: float = 0.008,
        noise_std: float = 1e-4,
        cache: bool = True,
        local_refine_ms: float = 12.0,
        local_refine_step: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else HOP_SIZE
        self.context_frames = context_frames
        self.augment = augment
        self.max_shift_ms = max_shift_ms
        self.min_rms = min_rms
        self.noise_std = noise_std
        self.cache = cache
        self.local_refine_ms = local_refine_ms
        self.local_refine_step = local_refine_step

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
            stem_set = set(stems)
            common = [s for s in common if s in stem_set]
        if not common:
            raise ValueError("No matching guitar/piano file pairs found.")

        self.frames: List[Tuple[torch.Tensor, torch.Tensor]] = []
        print(f"Found {len(common)} paired clips.")
        for idx, stem in enumerate(common, start=1):
            g_path = guitar_map[stem]
            p_path = piano_map[stem]
            print(f"  [{idx}/{len(common)}] {stem}")
            clip_pairs = self._load_pair(g_path, p_path)
            self.frames.extend(clip_pairs)
        print(f"Total onset-aligned contextual training frames: {len(self.frames):,}")

    def _cache_dir(self) -> Path:
        d = self.data_dir / ".g2p_temporal_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cache_key(self, guitar_path: Path, piano_path: Path) -> str:
        payload = {
            "cache_version": CACHE_VERSION,
            "guitar_path": str(guitar_path.resolve()),
            "piano_path": str(piano_path.resolve()),
            "guitar_mtime": guitar_path.stat().st_mtime_ns,
            "piano_mtime": piano_path.stat().st_mtime_ns,
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
            "context_frames": self.context_frames,
            "max_shift_ms": self.max_shift_ms,
            "min_rms": self.min_rms,
            "local_refine_ms": self.local_refine_ms,
            "local_refine_step": self.local_refine_step,
        }
        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _load_pair(self, guitar_path: Path, piano_path: Path) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        cache_path = self._cache_dir() / f"{guitar_path.stem}_{self._cache_key(guitar_path, piano_path)}.pt"
        if self.cache and cache_path.exists():
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            lag = int(blob.get("lag", 0))
            clip_seconds = float(blob.get("clip_seconds", 0.0))
            print(
                f"      cache hit | onset lag={lag} samples ({1000.0 * lag / self.sample_rate:+.1f} ms), "
                f"len={clip_seconds:.2f}s"
            )
            g_ctx = blob["guitar_ctx"].float()
            p = blob["piano"].float()
            return [(g_ctx[i], p[i]) for i in range(g_ctx.shape[0])]

        guitar = _to_mono_resampled_raw(guitar_path, self.sample_rate)
        piano = _to_mono_resampled_raw(piano_path, self.sample_rate)

        lag = _robust_lag_from_onsets(guitar, piano, sample_rate=self.sample_rate, fallback_ms=self.max_shift_ms)
        guitar, piano = _apply_lag(guitar, piano, lag)
        guitar, piano = _trim_shared_silence(guitar, piano, threshold=1e-4)

        # Shared scaling preserves the natural quiet-DI / loud-piano relationship.
        shared_peak = torch.maximum(guitar.abs().max(), piano.abs().max())
        if shared_peak > 0:
            scale = 0.98 / (shared_peak + 1e-8)
            guitar = guitar * scale
            piano = piano * scale

        clip_seconds = guitar.numel() / self.sample_rate
        print(
            f"      cache miss | onset lag={lag} samples ({1000.0 * lag / self.sample_rate:+.1f} ms), "
            f"len={clip_seconds:.2f}s"
        )

        pairs = _build_contextual_pairs(
            guitar,
            piano,
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            context_frames=self.context_frames,
            sample_rate=self.sample_rate,
            min_rms=self.min_rms,
            local_refine_ms=self.local_refine_ms,
            local_refine_step=self.local_refine_step,
        )

        if self.cache:
            if pairs:
                g_ctx = torch.stack([x[0] for x in pairs], dim=0)
                p = torch.stack([x[1] for x in pairs], dim=0)
            else:
                g_ctx = torch.empty(0, self.context_frames, self.frame_size)
                p = torch.empty(0, self.frame_size)
            torch.save({
                "lag": lag,
                "clip_seconds": clip_seconds,
                "guitar_ctx": g_ctx.cpu(),
                "piano": p.cpu(),
            }, cache_path)
            print(f"      saved cache -> {cache_path}")
        return pairs

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        guitar_ctx, piano_frame = self.frames[idx]
        guitar_ctx = guitar_ctx.clone()
        piano_frame = piano_frame.clone()

        if self.augment:
            gain = 10 ** random.uniform(-0.15, 0.15)
            guitar_ctx *= gain
            piano_frame *= gain
            if self.noise_std > 0:
                guitar_ctx += self.noise_std * torch.randn_like(guitar_ctx)
            guitar_ctx = torch.clamp(guitar_ctx, -1.0, 1.0)
            piano_frame = torch.clamp(piano_frame, -1.0, 1.0)

        return guitar_ctx, piano_frame


def make_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    val_split: float = 0.1,
    sample_rate: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE,
    hop_size: int | None = None,
    max_shift_ms: float = 140.0,
    min_rms: float = 0.008,
    noise_std: float = 1e-4,
    seed: int = 22,
    context_frames: int = CONTEXT_FRAMES,
    cache: bool = True,
    local_refine_ms: float = 12.0,
    local_refine_step: int = 8,
):
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

    train_set = GuitarPianoContextDataset(
        data_dir=str(data_dir),
        stems=train_stems,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size if hop_size is not None else HOP_SIZE,
        context_frames=context_frames,
        augment=True,
        max_shift_ms=max_shift_ms,
        min_rms=min_rms,
        noise_std=noise_std,
        cache=cache,
        local_refine_ms=local_refine_ms,
        local_refine_step=local_refine_step,
    )
    val_set = GuitarPianoContextDataset(
        data_dir=str(data_dir),
        stems=val_stems,
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size if hop_size is not None else HOP_SIZE,
        context_frames=context_frames,
        augment=False,
        max_shift_ms=max_shift_ms,
        min_rms=min_rms,
        noise_std=0.0,
        cache=cache,
        local_refine_ms=local_refine_ms,
        local_refine_step=local_refine_step,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader
