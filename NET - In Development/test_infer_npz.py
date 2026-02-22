#!/usr/bin/env python3
"""
test_infer_npz.py

Quick test/inference script for your updated dataset + model with count head.

What it does:
- Loads metadata.json (for bands/ctx info + normalization details if you need them later)
- Loads a checkpoint (.pt) produced by training
- Runs inference on:
    (A) a random batch from val shards, OR
    (B) a provided .wav file (optional) using the SAME feature pipeline as dataset maker
- Prints:
    - predicted polyphony count (0..6)
    - top-K predicted notes (clamped to predicted count)
    - onset/active probabilities for those notes
- Optional: simple hysteresis tracker for continuous wav inference.

Dependencies:
  pip install numpy torch librosa scipy

Examples:
  # 1) Sanity check on validation shards:
  python test_infer_npz.py --dataset_dir gs_fb --ckpt fb_model_torch/best.pt

  # 2) Run on a wav file (DI recording):
  python test_infer_npz.py --dataset_dir gs_fb --ckpt fb_model_torch/best.pt --wav myriff.wav

  # 3) Run wav with hysteresis (more stable note sets):
  python test_infer_npz.py --dataset_dir gs_fb --ckpt fb_model_torch/best.pt --wav myriff.wav --hysteresis
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import librosa
from scipy.io import wavfile


# -------------------------
# Constants / note mapping
# -------------------------
MIDI_MIN = 40  # E2
MIDI_MAX = 88  # E6
N_NOTES = MIDI_MAX - MIDI_MIN + 1

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(m: int) -> str:
    pc = m % 12
    octv = m // 12 - 1
    return f"{NOTE_NAMES_SHARP[pc]}{octv}"


def idx_to_midi(i: int) -> int:
    return MIDI_MIN + int(i)


# -------------------------
# Model (must match training)
# -------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.pad = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=self.kernel_size, dilation=self.dilation,
                              padding=0, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class GuitarPolyTCN(nn.Module):
    def __init__(self, bands: int, notes: int = 49, channels: int = 64, count_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(bands, channels, kernel_size=3, dilation=1),
            CausalConv1d(channels, channels, kernel_size=3, dilation=2),
            CausalConv1d(channels, channels, kernel_size=3, dilation=4),
            CausalConv1d(channels, channels, kernel_size=3, dilation=8),
            CausalConv1d(channels, channels, kernel_size=3, dilation=16),
        )
        self.active_head = nn.Linear(channels, notes)
        self.onset_head = nn.Linear(channels, notes)
        self.count_head = nn.Linear(channels, count_classes)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        h_last = h[:, :, -1]
        return self.active_head(h_last), self.onset_head(h_last), self.count_head(h_last)


# -------------------------
# Feature pipeline (same as dataset maker)
# -------------------------
def make_log_spaced_edges(fmin: float, fmax: float, n_bands: int) -> np.ndarray:
    fmin = max(10.0, float(fmin))
    fmax = max(fmin * 1.01, float(fmax))
    return np.geomspace(fmin, fmax, int(n_bands) + 1).astype(np.float64)


def stft_power(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
    S = librosa.stft(
        y=y.astype(np.float32),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        window="hann",
        center=False,
    )
    P = (np.abs(S) ** 2).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=int(n_fft)).astype(np.float64)
    return freqs, P


def band_energy_from_power(freqs: np.ndarray, P: np.ndarray, edges: np.ndarray, agg: str = "mean") -> np.ndarray:
    B = len(edges) - 1
    out = np.zeros((B, P.shape[1]), dtype=np.float32)
    for b in range(B):
        lo, hi = edges[b], edges[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        band = P[mask, :]
        out[b] = np.max(band, axis=0) if agg == "max" else np.mean(band, axis=0)
    return out


def log1p_feat(band_energy: np.ndarray) -> np.ndarray:
    return np.log1p(band_energy).astype(np.float32)


def zscore_feat(x_log: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    y = (x_log - mu[:, None]) / (sigma[:, None] + 1e-6)
    return np.clip(y, -6.0, 6.0).astype(np.float32)


def load_wav_mono_resample(path: Path, sr_target: int) -> np.ndarray:
    sr, x = wavfile.read(str(path))
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / (float(np.iinfo(x.dtype).max) + 1e-12)
    else:
        x = x.astype(np.float32)
    if int(sr) != int(sr_target):
        x = librosa.resample(x, orig_sr=int(sr), target_sr=int(sr_target)).astype(np.float32)
    return x


# -------------------------
# Dataset shard loader (for quick sanity check)
# -------------------------
def list_shards(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.glob("shard_*.npz") if p.is_file()])


def load_random_val_batch(dataset_dir: Path, batch: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    val_shards = list_shards(dataset_dir / "val")
    if not val_shards:
        raise RuntimeError("No val shards found.")
    shard_path = rng.choice(val_shards)
    d = np.load(shard_path)
    X = d["X"]   # (N,B,ctx)
    YA = d["YA"]
    YO = d["YO"]
    YC = d["YC"]
    n = X.shape[0]
    idx = rng.choice(n, size=min(batch, n), replace=False)
    return X[idx], YA[idx], YO[idx], YC[idx]


# -------------------------
# Post-processing helpers
# -------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def topk_notes(active_p: np.ndarray, onset_p: np.ndarray, k: int) -> List[int]:
    """
    Choose k notes by a simple confidence score:
      score = 0.8*active + 0.2*onset
    """
    if k <= 0:
        return []
    score = 0.8 * active_p + 0.2 * onset_p
    idx = np.argsort(score)[::-1]
    return idx[:k].tolist()


@dataclass
class HysteresisState:
    on_th: float = 0.55
    off_th: float = 0.35
    max_notes: int = 6
    active_mask: np.ndarray = None  # (49,) bool

    def __post_init__(self):
        if self.active_mask is None:
            self.active_mask = np.zeros((N_NOTES,), dtype=bool)

    def step(self, active_p: np.ndarray, onset_p: np.ndarray, k: int) -> List[int]:
        """
        Hysteresis:
          - a note turns on if onset_p>on_th OR active_p>on_th
          - stays on while active_p>off_th
        Then clamp to k notes by confidence.
        """
        turn_on = (onset_p >= self.on_th) | (active_p >= self.on_th)
        stay_on = (active_p >= self.off_th)

        self.active_mask = (self.active_mask & stay_on) | turn_on

        # clamp to k
        idx_on = np.where(self.active_mask)[0].tolist()
        if k <= 0:
            self.active_mask[:] = False
            return []
        if len(idx_on) <= k:
            return idx_on

        score = 0.8 * active_p + 0.2 * onset_p
        idx_sorted = sorted(idx_on, key=lambda i: float(score[i]), reverse=True)
        keep = idx_sorted[:k]
        new_mask = np.zeros_like(self.active_mask)
        new_mask[keep] = True
        self.active_mask = new_mask
        return keep


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Folder containing metadata.json and train/val shards")
    ap.add_argument("--ckpt", required=True, help="Path to best.pt or last.pt")
    ap.add_argument("--batch", type=int, default=8, help="Batch size for val-shard test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wav", type=str, default=None, help="Optional: run on a wav file instead of val shards")
    ap.add_argument("--hysteresis", action="store_true", help="Use hysteresis tracker for wav inference")
    ap.add_argument("--active_th", type=float, default=0.45)
    ap.add_argument("--onset_th", type=float, default=0.55)
    args = ap.parse_args()

    ds_dir = Path(args.dataset_dir)
    meta = json.loads((ds_dir / "metadata.json").read_text(encoding="utf-8"))

    sr = int(meta["sr"])
    hop = int(meta["hop_length"])
    n_fft = int(meta["n_fft"])
    bands = int(meta["bands"])
    ctx = int(meta["ctx_frames"])
    fmin = float(meta["fmin"])
    fmax = float(meta["fmax"])
    agg = str(meta.get("band_agg", "mean"))

    mu = np.asarray(meta["norm"]["band_mu"], dtype=np.float32)
    sigma = np.asarray(meta["norm"]["band_sigma"], dtype=np.float32)

    edges = make_log_spaced_edges(fmin, fmax, bands)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ck = torch.load(args.ckpt, map_location=device)
    channels = int(ck.get("channels", 64))
    model = GuitarPolyTCN(bands=bands, notes=N_NOTES, channels=channels, count_classes=7).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()

    print(f"Loaded model: bands={bands} ctx={ctx} channels={channels} device={device}")
    print(f"Feature config: sr={sr} hop={hop} n_fft={n_fft} bands={bands} fmin={fmin} fmax={fmax} agg={agg}")

    if args.wav is None:
        # ---- A) Test on validation shards
        X, YA, YO, YC = load_random_val_batch(ds_dir, batch=args.batch, seed=args.seed)
        x_t = torch.from_numpy(X.astype(np.float32)).to(device)

        with torch.no_grad():
            a_logits, o_logits, c_logits = model(x_t)

        a_p = torch.sigmoid(a_logits).cpu().numpy()
        o_p = torch.sigmoid(o_logits).cpu().numpy()
        c_p = torch.softmax(c_logits, dim=-1).cpu().numpy()

        for i in range(X.shape[0]):
            pred_k = int(np.argmax(c_p[i]))
            gt_k = int(YC[i])
            sel = topk_notes(a_p[i], o_p[i], pred_k)

            gt_notes = np.where(YA[i] > 0.5)[0].tolist()
            gt_names = [midi_to_name(idx_to_midi(j)) for j in gt_notes[:12]]
            pred_names = [midi_to_name(idx_to_midi(j)) for j in sel]

            print("\n--- sample", i, "---")
            print("GT count:", gt_k, "| Pred count:", pred_k)
            print("GT notes (up to 12):", gt_names)
            print("Pred notes:", pred_names)
            for j in sel:
                print(f"  {midi_to_name(idx_to_midi(j)):>4}  active={a_p[i][j]:.3f}  onset={o_p[i][j]:.3f}")

        return

    # ---- B) Run on WAV
    wav_path = Path(args.wav)
    x = load_wav_mono_resample(wav_path, sr_target=sr)

    freqs, P = stft_power(x, sr=sr, n_fft=n_fft, hop_length=hop)
    band = band_energy_from_power(freqs, P, edges, agg=agg)
    xlog = log1p_feat(band)
    feat = zscore_feat(xlog, mu=mu, sigma=sigma)  # (B,T)
    B, T = feat.shape

    if T < ctx:
        raise RuntimeError(f"WAV too short after framing: T={T} < ctx={ctx}")

    print(f"WAV frames: T={T} (ctx={ctx})")

    hyst = HysteresisState(on_th=args.onset_th, off_th=args.active_th) if args.hysteresis else None

    # iterate frame by frame (causal), matching your on-device use
    for t in range(ctx - 1, T):
        x_win = feat[:, t - ctx + 1:t + 1][None, :, :]  # (1,B,ctx)
        x_t = torch.from_numpy(x_win.astype(np.float32)).to(device)

        with torch.no_grad():
            a_logits, o_logits, c_logits = model(x_t)

        a_p = torch.sigmoid(a_logits).cpu().numpy()[0]
        o_p = torch.sigmoid(o_logits).cpu().numpy()[0]
        c_p = torch.softmax(c_logits, dim=-1).cpu().numpy()[0]
        pred_k = int(np.argmax(c_p))

        if args.hysteresis:
            sel = hyst.step(a_p, o_p, pred_k)
        else:
            # simple thresholding then clamp
            mask = (a_p >= args.active_th) | (o_p >= args.onset_th)
            idx = np.where(mask)[0].tolist()
            # clamp to pred_k by confidence
            if pred_k <= 0:
                sel = []
            elif len(idx) <= pred_k:
                sel = idx
            else:
                score = 0.8 * a_p + 0.2 * o_p
                idx_sorted = sorted(idx, key=lambda i: float(score[i]), reverse=True)
                sel = idx_sorted[:pred_k]

        # Print occasionally (every ~10 frames) to avoid spam
        if (t % 10) == 0:
            t_sec = (t * hop) / float(sr)
            names = [midi_to_name(idx_to_midi(j)) for j in sel]
            print(f"{t_sec:7.3f}s  k={pred_k}  notes={names}")

    print("Done.")


if __name__ == "__main__":
    main()