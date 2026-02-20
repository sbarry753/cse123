#!/usr/bin/env python3
# chord_match.py
"""
Match a WAV against a CHORD LUT created by chord_lut_build.py.

This is a fast direct matcher:
- Compute the live log-band template (same params as LUT meta.band_template)
- Cosine score against every chord template in chord_lut.json
- Return top-N chords

Usage:
  python chord_match.py --lut chord_lut.json --wav chord.wav --start 0.10 --dur 0.25 --top 8 --plot
Optional:
  --highpass --highpass_hz 80
  --window hann
  --min_sounding 1 --max_sounding 6   (filter results to certain chord sizes)

Notes:
- This does NOT do mixture/NNLS. It assumes the chord LUT already encodes combos.
- For big chord LUTs, this can be heavy but still workable if you keep it under ~200k entries.
"""

import argparse
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, butter, sosfilt
import matplotlib.pyplot as plt


# -----------------------------
# Audio + spectrum
# -----------------------------
def read_wav_mono_float(wav_path: str) -> Tuple[int, np.ndarray]:
    fs, x = wavfile.read(wav_path)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype.kind in "iu":
        maxv = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / maxv
    else:
        x = x.astype(np.float64)
    return int(fs), x


def normalize_peak(x: np.ndarray) -> np.ndarray:
    return x / (float(np.max(np.abs(x)) + 1e-12))


def pick_segment(x: np.ndarray, fs: int, start_sec: float, dur_sec: float) -> np.ndarray:
    start = int(round(start_sec * fs))
    length = int(round(dur_sec * fs))
    seg = x[start:start + length]
    if len(seg) < length:
        seg = np.pad(seg, (0, length - len(seg)))
    return seg


def highpass_filter(x: np.ndarray, fs: int, cutoff_hz: float = 80.0, order: int = 4) -> np.ndarray:
    if cutoff_hz <= 0:
        return x
    sos = butter(order, cutoff_hz / (fs * 0.5), btype="highpass", output="sos")
    return sosfilt(sos, x)


def spectrum_mag(seg: np.ndarray, fs: int, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    N = len(seg)
    w = get_window(window, N, fftbins=True)
    X = np.fft.rfft(seg * w)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    mag = np.abs(X)
    return freqs, mag


# -----------------------------
# Log-band template (must match LUT meta)
# -----------------------------
def make_log_band_edges(fmin_hz: float, fmax_hz: float, n_bands: int) -> np.ndarray:
    fmin_hz = max(1e-3, float(fmin_hz))
    fmax_hz = max(fmin_hz * 1.001, float(fmax_hz))
    n_bands = max(8, int(n_bands))
    return np.geomspace(fmin_hz, fmax_hz, n_bands + 1).astype(np.float64)


def pool_to_log_bands(freqs: np.ndarray, mag: np.ndarray, edges: np.ndarray, agg: str = "max") -> np.ndarray:
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    agg = (agg or "max").lower()
    for i in range(len(out)):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            out[i] = 0.0
        else:
            band = mag[mask]
            out[i] = float(np.max(band)) if agg == "max" else float(np.mean(band))
    return out


def build_live_band(seg: np.ndarray, fs: int, band_meta: Dict[str, Any], window: str) -> np.ndarray:
    fmin = float(band_meta.get("fmin_hz", 40.0))
    fmax = float(band_meta.get("fmax_hz", 8000.0))
    nb = int(band_meta.get("n_bands", 480))
    agg = str(band_meta.get("agg", "max"))
    logmag = bool(band_meta.get("logmag", True))

    edges = make_log_band_edges(fmin, fmax, nb)
    freqs, mag = spectrum_mag(seg, fs, window=window)
    y = mag.astype(np.float64)
    if logmag:
        y = np.log1p(y)
    band = pool_to_log_bands(freqs, y, edges, agg=agg)
    band /= (np.linalg.norm(band) + 1e-12)
    return band


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


# -----------------------------
# LUT load + fast matrix scoring
# -----------------------------
def load_chord_lut(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("meta", {})
    data.setdefault("entries", [])
    return data


def build_matrix(lut_entries: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Build A where each column is a chord template (L2 normalized already).
    Returns (A, kept_entries).
    """
    cols = []
    kept = []
    for e in lut_entries:
        bt = e.get("band_template", None)
        if bt is None:
            continue
        v = np.array(bt, dtype=np.float64)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-9:
            continue
        cols.append(v / (nrm + 1e-12))
        kept.append(e)
    if not cols:
        return np.zeros((0, 0), dtype=np.float64), []
    A = np.column_stack(cols)  # shape [bands, num_chords]
    return A, kept


# -----------------------------
# Main match
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut", required=True, help="Chord LUT json from chord_lut_build.py")
    ap.add_argument("--wav", required=True, help="WAV to classify")
    ap.add_argument("--start", type=float, default=0.10, help="Segment start sec")
    ap.add_argument("--dur", type=float, default=0.25, help="Segment duration sec")
    ap.add_argument("--window", type=str, default="hann", help="FFT window")
    ap.add_argument("--highpass", action="store_true", help="Apply highpass")
    ap.add_argument("--highpass_hz", type=float, default=80.0, help="Highpass cutoff Hz")
    ap.add_argument("--top", type=int, default=8, help="Top N chords to print")
    ap.add_argument("--plot", action="store_true", help="Plot spectrum + top scores")

    ap.add_argument("--min_sounding", type=int, default=0, help="Filter results (min sounding strings)")
    ap.add_argument("--max_sounding", type=int, default=6, help="Filter results (max sounding strings)")

    args = ap.parse_args()

    lut = load_chord_lut(args.lut)
    band_meta = (lut.get("meta", {}) or {}).get("band_template", {}) or {}

    entries_all = lut.get("entries", [])
    # optional filter by size
    entries = []
    for e in entries_all:
        ns = int(e.get("num_sounding", 0))
        if ns < int(args.min_sounding) or ns > int(args.max_sounding):
            continue
        entries.append(e)

    if not entries:
        raise RuntimeError("No chord entries available after filtering.")

    A, kept = build_matrix(entries)
    if A.size == 0:
        raise RuntimeError("Chord LUT has no usable band_template vectors.")

    fs, x = read_wav_mono_float(args.wav)
    x = normalize_peak(x)

    seg = pick_segment(x, fs, args.start, args.dur)
    if args.highpass:
        seg = highpass_filter(seg, fs, cutoff_hz=args.highpass_hz)
    seg = normalize_peak(seg)

    live_band = build_live_band(seg, fs, band_meta, window=args.window)
    y = live_band / (np.linalg.norm(live_band) + 1e-12)

    # Fast cosine with matrix: scores = A^T y
    scores = A.T @ y  # shape [num_chords]
    idxs = np.argsort(scores)[::-1]

    top = int(args.top)
    print(f"\nTop {top} chord matches:")
    for rank, j in enumerate(idxs[:top], start=1):
        e = kept[int(j)]
        s = float(scores[int(j)])
        cid = str(e.get("chord_id", "<no_id>"))
        ns = int(e.get("num_sounding", 0))
        notes_by_string = e.get("notes_by_string", {})
        # pretty string order 0..5
        snotes = []
        for si in range(6):
            v = notes_by_string.get(str(si), None)
            snotes.append(v if v is not None else "--")
        print(f"{rank:2d}. score={s:.4f}  sounding={ns}  [{', '.join(snotes)}]")
        # uncomment if you want the verbose id
        # print(f"    {cid}")

    if args.plot:
        freqs, mag = spectrum_mag(seg, fs, window=args.window)

        plt.figure(figsize=(12, 4))
        plt.plot(freqs, mag)
        plt.title("Live Segment Magnitude Spectrum")
        plt.xlabel("Hz")
        plt.ylabel("|X(f)|")
        plt.xlim(0, min(8000, freqs[-1]))
        plt.tight_layout()
        plt.show()

        show = min(12, len(idxs))
        labels = []
        vals = []
        for j in idxs[:show]:
            e = kept[int(j)]
            notes_by_string = e.get("notes_by_string", {})
            snotes = []
            for si in range(6):
                v = notes_by_string.get(str(si), None)
                snotes.append(v if v is not None else "--")
            labels.append("|".join(snotes))
            vals.append(float(scores[int(j)]))

        plt.figure(figsize=(12, 4))
        plt.bar(range(len(vals)), vals)
        plt.title(f"Top chord scores (showing {show})")
        plt.xlabel("Chord rank")
        plt.ylabel("Cosine score")
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())