#!/usr/bin/env python3
"""
wav_to_lut.py

Load a WAV file (single note or chord), approximate a selected segment as a sum of sines,
plot original vs reconstructed, and append a mathematical LUT entry (JSON) you can use later.

Model:
  x[n] ≈ Σ_k A_k * sin(2π f_k n/fs + φ_k)

Usage examples:
  # Single note (1 sine)
  python wav_to_lut.py samples/E2_pick.wav --label E2_pick --k 1 --out_wav recreated_E2.wav

  # Chord (multiple sines)
  python wav_to_lut.py samples/Am_chord.wav --label Am_chord --k 4 --out_wav recreated_Am.wav
"""

import json
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, find_peaks
import matplotlib.pyplot as plt


def to_mono(x: np.ndarray) -> np.ndarray:
    """Convert mono/stereo to mono float64."""
    if x.ndim == 1:
        return x.astype(np.float64)
    return x.mean(axis=1).astype(np.float64)


def normalize_audio(x: np.ndarray) -> np.ndarray:
    """Normalize to peak 1.0 (safe for visualization/fitting)."""
    m = float(np.max(np.abs(x)) + 1e-12)
    return x / m


def pick_analysis_segment(x: np.ndarray, fs: int, start_sec: float, dur_sec: float) -> np.ndarray:
    """Slice a segment from start_sec for dur_sec. Zero-pad if needed."""
    start = int(start_sec * fs)
    length = int(dur_sec * fs)
    seg = x[start:start + length]
    if len(seg) < length:
        seg = np.pad(seg, (0, length - len(seg)))
    return seg


def fft_peak_frequencies(x: np.ndarray, fs: int, k: int, fmin: float, fmax: float) -> np.ndarray:
    """
    Return up to k peak frequencies from the Hann-windowed magnitude spectrum
    in [fmin, fmax].
    """
    N = len(x)
    w = get_window("hann", N, fftbins=True)
    xw = x * w

    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mag = np.abs(X)

    # band-limit
    band = (freqs >= fmin) & (freqs <= fmax)
    freqs_b = freqs[band]
    mag_b = mag[band]

    if len(freqs_b) == 0:
        raise ValueError("Frequency band is empty. Check fmin/fmax and sample rate.")

    # find peaks
    prom = float(np.max(mag_b) * 0.02)  # tweakable: 2% of max magnitude
    peaks, _ = find_peaks(mag_b, prominence=prom)

    if len(peaks) == 0:
        # fallback: just take max bin
        idx = int(np.argmax(mag_b))
        return np.array([float(freqs_b[idx])], dtype=np.float64)

    # sort peaks by magnitude desc
    peak_mags = mag_b[peaks]
    order = np.argsort(peak_mags)[::-1]
    peaks = peaks[order]

    freqs_out = freqs_b[peaks[:k]]
    freqs_out = np.sort(freqs_out)

    # de-duplicate very close peaks
    dedup = []
    for f in freqs_out:
        if not dedup or abs(f - dedup[-1]) > 2.0:  # Hz spacing threshold
            dedup.append(float(f))
    return np.array(dedup[:k], dtype=np.float64)


def fit_sum_of_sines(x: np.ndarray, fs: int, freqs: np.ndarray):
    """
    Fit:
      x[n] ≈ Σ (a_k cos(w_k n) + b_k sin(w_k n))
    via least squares, then convert to:
      A_k, φ_k in A_k sin(w_k n + φ_k)
    """
    N = len(x)
    n = np.arange(N, dtype=np.float64)

    # Build design matrix M: [cos(w1 n), sin(w1 n), cos(w2 n), sin(w2 n), ...]
    cols = []
    for f in freqs:
        w = 2.0 * np.pi * f / fs
        cols.append(np.cos(w * n))
        cols.append(np.sin(w * n))
    M = np.stack(cols, axis=1)  # (N, 2K)

    # Least squares solve
    theta, *_ = np.linalg.lstsq(M, x, rcond=None)

    params = []
    yhat = np.zeros_like(x, dtype=np.float64)

    for i, f in enumerate(freqs):
        a = float(theta[2 * i + 0])  # cos coeff
        b = float(theta[2 * i + 1])  # sin coeff

        # a cos + b sin = A sin(w n + φ)
        # A = sqrt(a^2 + b^2)
        # φ = atan2(a, b)
        A = float(np.sqrt(a * a + b * b))
        phi = float(np.arctan2(a, b))

        params.append({"freq_hz": float(f), "amp": A, "phase_rad": phi})

        w = 2.0 * np.pi * f / fs
        yhat += A * np.sin(w * n + phi)

    return params, yhat


def append_to_lut(lut_path: str, entry: dict):
    """Append an entry to a LUT JSON file."""
    try:
        with open(lut_path, "r", encoding="utf-8") as f:
            lut = json.load(f)
    except FileNotFoundError:
        lut = {"entries": []}

    if "entries" not in lut or not isinstance(lut["entries"], list):
        lut = {"entries": []}

    lut["entries"].append(entry)

    with open(lut_path, "w", encoding="utf-8") as f:
        json.dump(lut, f, indent=2)


def write_wav_i16(path: str, fs: int, x: np.ndarray):
    """Write a float signal in [-1,1] as int16 wav."""
    x = normalize_audio(x)
    x_i16 = np.int16(np.clip(x, -1.0, 1.0) * 32767)
    wavfile.write(path, fs, x_i16)


def plot_waveforms(seg: np.ndarray, yhat: np.ndarray, fs: int, show_spectrum: bool, fmax_plot: float):
    """Plot original and reconstructed waveforms, plus optional spectrum."""
    t = np.arange(len(seg), dtype=np.float64) / fs

    plt.figure(figsize=(12, 8))

    # 1) Original
    plt.subplot(4, 1, 1)
    plt.plot(t, seg)
    plt.title("Original Waveform (Segment)")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")

    # 2) Recreated
    plt.subplot(4, 1, 2)
    plt.plot(t, yhat)
    plt.title("Recreated Sine Wave (Sum of Sines)")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")

    # 3) Overlay
    plt.subplot(4, 1, 3)
    plt.plot(t, seg, label="Original", alpha=0.7)
    plt.plot(t, yhat, label="Recreated", alpha=0.7)
    plt.title("Overlay: Original vs Recreated")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.legend()

    # 4) Zoomed view (first 10 ms)
    zoom_samples = int(0.010 * fs)
    zoom_samples = max(16, min(zoom_samples, len(seg)))
    plt.subplot(4, 1, 4)
    plt.plot(t[:zoom_samples], seg[:zoom_samples], label="Original", alpha=0.8)
    plt.plot(t[:zoom_samples], yhat[:zoom_samples], label="Recreated", alpha=0.8)
    plt.title("Zoomed View (First 10 ms)")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    if show_spectrum:
        # Spectrum of original segment
        N = len(seg)
        w = get_window("hann", N, fftbins=True)
        X = np.fft.rfft(seg * w)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)

        plt.figure(figsize=(10, 4))
        plt.plot(freqs, np.abs(X))
        plt.title("Magnitude Spectrum (Original Segment)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.xlim(0, float(fmax_plot))
        plt.tight_layout()
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_path", type=str, help="Path to input wav file")

    # LUT
    ap.add_argument("--lut", type=str, default="lut.json", help="Path to LUT JSON file")
    ap.add_argument("--label", type=str, default=None, help="Label for this entry (e.g., E2_pick, Am_chord)")

    # Segment selection
    ap.add_argument("--start", type=float, default=0.05, help="Analysis start time (sec)")
    ap.add_argument("--dur", type=float, default=0.25, help="Analysis duration (sec)")

    # Model settings
    ap.add_argument("--k", type=int, default=1, help="Number of sine components (1=note, >1=chord)")
    ap.add_argument("--fmin", type=float, default=50.0, help="Min freq for peak search (Hz)")
    ap.add_argument("--fmax", type=float, default=2000.0, help="Max freq for peak search (Hz)")

    # Outputs
    ap.add_argument("--out_wav", type=str, default=None, help="Optional output wav path for recreated signal")
    ap.add_argument("--show_spectrum", action="store_true", help="Also plot the spectrum of the segment")
    ap.add_argument("--fmax_plot", type=float, default=1000.0, help="Max freq shown in spectrum plot (Hz)")

    args = ap.parse_args()

    # Load WAV
    fs, x = wavfile.read(args.wav_path)
    x = to_mono(x)

    # Convert ints to float [-1,1]
    if x.dtype.kind in "iu":
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float64) / float(maxv)

    x = normalize_audio(x)

    # Segment
    seg = pick_analysis_segment(x, fs, args.start, args.dur)
    seg = normalize_audio(seg)

    # Get peak freqs + fit
    freqs = fft_peak_frequencies(seg, fs, k=args.k, fmin=args.fmin, fmax=args.fmax)
    params, yhat = fit_sum_of_sines(seg, fs, freqs)

    # Print components (your math representation)
    print("Components (math LUT params):")
    for p in params:
        print(f"  f={p['freq_hz']:.2f} Hz, A={p['amp']:.6f}, phi={p['phase_rad']:.6f} rad")

    # Save LUT entry
    entry = {
        "label": args.label or args.wav_path,
        "sample_rate_hz": int(fs),
        "analysis": {"start_sec": float(args.start), "dur_sec": float(args.dur)},
        "model": {
            "type": "sum_of_sines",
            "equation": "x[n] = sum_k A_k * sin(2*pi*f_k*n/fs + phi_k)",
            "components": params
        }
    }
    append_to_lut(args.lut, entry)
    print(f"Saved LUT entry to: {args.lut}")

    # Optional recreated wav output
    if args.out_wav:
        write_wav_i16(args.out_wav, fs, yhat)
        print(f"Wrote recreated wav to: {args.out_wav}")

    # Plots: original vs sine recreation
    plot_waveforms(seg, yhat, fs, show_spectrum=args.show_spectrum, fmax_plot=args.fmax_plot)


if __name__ == "__main__":
    main()
