"""
precompute_f0.py - Offline pYIN f0 labels for TIMBRE NET.

Example:
  python3 precompute_f0.py --data_dir data --f0_cache_dir f0_cache
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Precompute per-hop guitar f0 labels with librosa.pyin")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--f0_cache_dir", type=str, default="./f0_cache")
    p.add_argument("--source", type=str, default="guitar", choices=["guitar", "piano"])
    p.add_argument("--sample_rate", type=int, default=48000)
    p.add_argument("--hop_size", type=int, default=256)
    p.add_argument("--frame_length", type=int, default=4096)
    p.add_argument("--fmin", type=str, default="E2")
    p.add_argument("--fmax", type=str, default="E6")
    p.add_argument("--max_hold_frames", type=int, default=32)
    p.add_argument("--rms_db_threshold", type=float, default=-60.0)
    return p.parse_args()


def _import_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise SystemExit(
            "librosa is required for f0 precomputation. Install project dependencies first."
        ) from exc
    return librosa


def _db_to_amplitude(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _rms_by_hop(y: np.ndarray, hop_size: int, n_hops: int) -> np.ndarray:
    rms = np.zeros(n_hops, dtype=np.float32)
    for i in range(n_hops):
        start = i * hop_size
        frame = y[start:start + hop_size]
        if frame.size:
            rms[i] = np.sqrt(np.mean(frame.astype(np.float32) ** 2) + 1e-12)
    return rms


def _fill_short_gaps(f0: np.ndarray, active: np.ndarray, max_hold_frames: int) -> np.ndarray:
    if max_hold_frames <= 0:
        return np.where(active, f0, 0.0).astype(np.float32)

    filled = f0.copy()
    valid = (filled > 0.0) & active
    n = len(filled)
    i = 0
    while i < n:
        if valid[i]:
            i += 1
            continue

        start = i
        while i < n and not valid[i]:
            i += 1
        end = i

        gap_active = active[start:end].all()
        gap_len = end - start
        prev_f0 = filled[start - 1] if start > 0 and valid[start - 1] else 0.0
        next_f0 = filled[end] if end < n and valid[end] else 0.0

        if gap_active and gap_len <= max_hold_frames:
            if prev_f0 > 0.0 and next_f0 > 0.0:
                filled[start:end] = np.linspace(prev_f0, next_f0, gap_len + 2, dtype=np.float32)[1:-1]
            elif prev_f0 > 0.0:
                filled[start:end] = prev_f0
            elif next_f0 > 0.0:
                filled[start:end] = next_f0

    filled[~active] = 0.0
    filled[~np.isfinite(filled)] = 0.0
    return filled.astype(np.float32)


def compute_f0_track(
    librosa,
    path: Path,
    sample_rate: int,
    hop_size: int,
    frame_length: int,
    fmin_hz: float,
    fmax_hz: float,
    max_hold_frames: int,
    rms_db_threshold: float,
) -> np.ndarray:
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    y = np.asarray(y, dtype=np.float32)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak

    n_hops = len(y) // hop_size
    if n_hops == 0:
        return np.zeros(0, dtype=np.float32)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_size,
        center=False,
        fill_na=np.nan,
    )

    f0 = np.asarray(f0, dtype=np.float32)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    if f0.shape[0] < n_hops:
        f0 = np.pad(f0, (0, n_hops - f0.shape[0]), constant_values=np.nan)
        voiced_flag = np.pad(voiced_flag, (0, n_hops - voiced_flag.shape[0]), constant_values=False)
    else:
        f0 = f0[:n_hops]
        voiced_flag = voiced_flag[:n_hops]

    rms = _rms_by_hop(y, hop_size, n_hops)
    active = rms >= _db_to_amplitude(rms_db_threshold)
    f0 = np.where(voiced_flag & active & np.isfinite(f0), f0, 0.0).astype(np.float32)
    return _fill_short_gaps(f0, active=active, max_hold_frames=max_hold_frames)


def main():
    args = parse_args()
    librosa = _import_librosa()

    data_dir = Path(args.data_dir)
    source_dir = data_dir / args.source
    f0_cache_dir = Path(args.f0_cache_dir)
    f0_cache_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    fmin_hz = float(librosa.note_to_hz(args.fmin))
    fmax_hz = float(librosa.note_to_hz(args.fmax))
    files = sorted(source_dir.glob("*.wav")) + sorted(source_dir.glob("*.flac"))

    metadata = {
        "version": 1,
        "data_dir": str(data_dir.resolve()),
        "source": args.source,
        "sample_rate": args.sample_rate,
        "hop_size": args.hop_size,
        "frame_length": args.frame_length,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "fmin_hz": fmin_hz,
        "fmax_hz": fmax_hz,
        "max_hold_frames": args.max_hold_frames,
        "rms_db_threshold": args.rms_db_threshold,
        "files": [],
    }

    for path in tqdm(files, desc="Precompute f0"):
        track = compute_f0_track(
            librosa,
            path=path,
            sample_rate=args.sample_rate,
            hop_size=args.hop_size,
            frame_length=args.frame_length,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            max_hold_frames=args.max_hold_frames,
            rms_db_threshold=args.rms_db_threshold,
        )
        out_path = f0_cache_dir / f"{path.stem}.npy"
        np.save(out_path, track.astype(np.float32, copy=False))

        stat = path.stat()
        metadata["files"].append({
            "stem": path.stem,
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "n_hops": int(track.shape[0]),
            "voiced_fraction": float(np.mean(track > 0.0)) if track.size else 0.0,
        })

    with open(f0_cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(files)} f0 tracks to {f0_cache_dir}")


if __name__ == "__main__":
    main()
