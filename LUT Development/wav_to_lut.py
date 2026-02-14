#!/usr/bin/env python3
"""
wav_to_lut.py
A simple harmonic fingerprint LUT builder + matcher.

- For every candidate note in the LUT with known f0(note),
  measure energy in the live spectrum near k*f0(note) for k=1..K.
- Normalize those harmonic energies -> fingerprint vector.
- Compare that vector to the stored template fingerprint for that note.
- Choose best similarity.

Commands:
  1) Build LUT from labeled recordings (NOTE=path):
    python wav_to_lut.py build --out lut.json --k 60 --tol 15 --start 0.12 --dur 0.18 \
        E6=samples/E6.wav F4=samples/F4.wav

  2) Build LUT from a directory (auto-note from filename):
    python wav_to_lut.py build_dir --out lut.json --dir samples --k 60 --tol 15 --start 0.12 --dur 0.18

  3) Match an unknown recording:
    python wav_to_lut.py match --lut lut.json --wav unknown.wav --start 0.12 --dur 0.18 --plot

Optional:
  --plot : show spectrum + note scores
  --top  : show top N matches
  --highpass : apply simple highpass to reduce rumble (optional)

Requirements:
  pip install numpy scipy matplotlib
"""

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, butter, sosfilt
import matplotlib.pyplot as plt


# -----------------------------
# Note utilities: "E6", "F#4", "Bb3"
# -----------------------------
_NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11,
}

# Match note tokens in filenames, e.g. "E6", "F#4", "Bb3"
# Accept either '#' or 'b' (case-insensitive), with optional separators after it.
_NOTE_TOKEN_RE = re.compile(r"(?i)\b([A-G])([#b]?)(-?\d+)\b")


def normalize_note_token(letter: str, accidental: str, octave_str: str) -> str:
    """
    Convert letter/accidental/octave into canonical form like:
      Bb3, F#4, E6
    """
    letter = letter.upper()
    accidental = accidental.strip()
    if accidental == "b":
        accidental = "b"
    elif accidental == "#":
        accidental = "#"
    else:
        accidental = ""
    return f"{letter}{accidental}{octave_str}"


def note_to_midi(note: str) -> int:
    s = note.strip().upper().replace("♯", "#").replace("♭", "B")
    m = re.match(r"^([A-G])([#B]?)(-?\d+)$", s)
    if not m:
        raise ValueError(f"Bad note format '{note}'. Use like E6, F#4, Bb3.")
    letter, accidental, octave_str = m.group(1), m.group(2), m.group(3)
    pitch = letter + accidental
    if pitch not in _NOTE_TO_SEMITONE:
        raise ValueError(f"Unknown pitch class '{pitch}' from '{note}'.")
    semitone = _NOTE_TO_SEMITONE[pitch]
    octave = int(octave_str)
    midi = (octave + 1) * 12 + semitone  # C4=60
    return int(midi)


def midi_to_freq_hz(midi: int, a4_hz: float = 440.0) -> float:
    return float(a4_hz * (2.0 ** ((midi - 69) / 12.0)))


def note_from_filename(path: Path) -> str:
    """
    Extract a note token from filename stem.
    Examples:
      E6.wav -> E6
      F#4_take2.wav -> F#4
      Bb3-clean.wav -> Bb3
    """
    stem = path.stem  # without extension
    m = _NOTE_TOKEN_RE.search(stem)
    if not m:
        raise ValueError(f"Could not parse note from filename '{path.name}'. "
                         f"Expected something like E6.wav, F#4.wav, Bb3.wav.")
    letter, accidental, octave_str = m.group(1), m.group(2), m.group(3)
    # Canonicalize to nice form: Bb3 / F#4 / E6
    # Note: note_to_midi expects flats as 'b' or 'B' in original string; we store as 'Bb3' etc.
    return normalize_note_token(letter, accidental.lower(), octave_str)


# -----------------------------
# Audio helpers
# -----------------------------
def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float64)
    return x.mean(axis=1).astype(np.float64)


def normalize_peak(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-12)
    return x / m


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


# -----------------------------
# Spectrum + fingerprint extraction
# -----------------------------
def spectrum_mag(seg: np.ndarray, fs: int, window: str = "hann"):
    N = len(seg)
    w = get_window(window, N, fftbins=True)
    X = np.fft.rfft(seg * w)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    mag = np.abs(X)
    return freqs, mag


def band_energy_max(freqs: np.ndarray, mag: np.ndarray, center_hz: float, tol_hz: float) -> float:
    lo = center_hz - tol_hz
    hi = center_hz + tol_hz
    if hi <= 0:
        return 0.0
    mask = (freqs >= max(0.0, lo)) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.max(mag[mask]))


def fingerprint_for_note(seg: np.ndarray,
                         fs: int,
                         note_f0_hz: float,
                         k: int,
                         tol_hz: float,
                         window: str = "hann") -> np.ndarray:
    """
    Extract a normalized harmonic fingerprint from seg ASSUMING the candidate note's f0.
    """
    freqs, mag = spectrum_mag(seg, fs, window=window)
    nyq = float(freqs[-1])

    amps = []
    for h in range(1, k + 1):
        fh = h * note_f0_hz
        if fh >= nyq:
            break
        amps.append(band_energy_max(freqs, mag, fh, tol_hz))

    v = np.array(amps, dtype=np.float64)
    s = float(np.sum(v))
    if s <= 1e-18:
        return np.zeros_like(v)
    return v / (s + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


# -----------------------------
# LUT format
# -----------------------------
@dataclass
class LutEntry:
    note: str
    midi: int
    f0_hz: float
    k: int
    tol_hz: float
    window: str
    fingerprint: List[float]
    source_wav: str
    analysis_start_sec: float
    analysis_dur_sec: float


def load_lut(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"entries": []}
        if "entries" not in data or not isinstance(data["entries"], list):
            data["entries"] = []
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"LUT file not found: {path}")


def save_lut(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# Build LUT (from labeled list)
# -----------------------------
def build_lut(out_path: str,
              labeled: List[Tuple[str, str]],
              k: int,
              tol_hz: float,
              start: float,
              dur: float,
              a4: float,
              window: str,
              use_highpass: bool,
              highpass_hz: float,
              replace_existing: bool):
    lut = {"meta": {
                "type": "harmonic_fingerprint_lut",
                "normalization": "sum_to_1",
                "similarity": "cosine",
                "a4_hz": float(a4),
            },
           "entries": []
    }

    # If replacing into an existing LUT, load it
    if not replace_existing and Path(out_path).exists():
        try:
            lut = load_lut(out_path)
        except Exception:
            pass

    # Index existing entries by note for easy replacement
    existing_idx = {}
    for i, e in enumerate(lut.get("entries", [])):
        if isinstance(e, dict) and "note" in e:
            existing_idx[str(e["note"]).upper()] = i

    for note, wav_path in labeled:
        fs, x = wavfile.read(wav_path)
        x = to_mono(x)
        if x.dtype.kind in "iu":
            x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
        x = normalize_peak(x)

        seg = pick_segment(x, fs, start, dur)
        if use_highpass:
            seg = highpass_filter(seg, fs, cutoff_hz=highpass_hz)
        seg = normalize_peak(seg)

        midi = note_to_midi(note)
        f0 = midi_to_freq_hz(midi, a4_hz=a4)

        fp = fingerprint_for_note(seg, fs, f0, k=k, tol_hz=tol_hz, window=window)

        entry = LutEntry(
            note=note,
            midi=midi,
            f0_hz=float(f0),
            k=int(len(fp)),
            tol_hz=float(tol_hz),
            window=window,
            fingerprint=[float(v) for v in fp.tolist()],
            source_wav=str(wav_path),
            analysis_start_sec=float(start),
            analysis_dur_sec=float(dur),
        ).__dict__

        key = note.strip().upper()
        if key in existing_idx:
            lut["entries"][existing_idx[key]] = entry
            print(f"[build] replaced {note}: f0={f0:.2f} Hz, fp_len={len(fp)} from {wav_path}")
        else:
            lut["entries"].append(entry)
            print(f"[build] added    {note}: f0={f0:.2f} Hz, fp_len={len(fp)} from {wav_path}")

    save_lut(out_path, lut)
    print(f"Saved LUT -> {out_path} with {len(lut['entries'])} entries")


def gather_wavs_from_dir(dir_path: str, recursive: bool) -> List[Path]:
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    glob = root.rglob("*.wav") if recursive else root.glob("*.wav")
    return sorted([p for p in glob if p.is_file()])


# -----------------------------
# Match
# -----------------------------
def match_note(lut_path: str,
               wav_path: str,
               start: float,
               dur: float,
               tol_hz_live: Optional[float],
               k_live: Optional[int],
               window_live: Optional[str],
               a4_override: Optional[float],
               use_highpass: bool,
               highpass_hz: float,
               top_n: int,
               plot: bool):
    lut = load_lut(lut_path)
    entries = lut.get("entries", [])
    if not entries:
        raise ValueError("LUT has no entries.")

    fs, x = wavfile.read(wav_path)
    x = to_mono(x)
    if x.dtype.kind in "iu":
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    x = normalize_peak(x)

    seg = pick_segment(x, fs, start, dur)
    if use_highpass:
        seg = highpass_filter(seg, fs, cutoff_hz=highpass_hz)
    seg = normalize_peak(seg)

    results = []

    freqs_mag = None
    if plot:
        freqs_mag = spectrum_mag(seg, fs, window=window_live or entries[0].get("window", "hann"))

    a4_lut = float(lut.get("meta", {}).get("a4_hz", 440.0))
    a4 = float(a4_override) if a4_override is not None else a4_lut

    for e in entries:
        note = e["note"]
        midi = int(e["midi"])
        f0 = midi_to_freq_hz(midi, a4_hz=a4)
        k = int(k_live) if k_live is not None else int(e.get("k", 60))
        tol = float(tol_hz_live) if tol_hz_live is not None else float(e.get("tol_hz", 12.0))
        win = window_live or e.get("window", "hann")

        fp_live = fingerprint_for_note(seg, fs, f0, k=k, tol_hz=tol, window=win)
        fp_ref = np.array(e["fingerprint"], dtype=np.float64)

        sim = cosine_similarity(fp_live, fp_ref)
        results.append((note, sim, f0, len(fp_live)))

    results.sort(key=lambda t: t[1], reverse=True)

    best_note, best_sim, best_f0, best_len = results[0]
    print(f"\nBest match: {best_note}  (sim={best_sim:.4f}, f0={best_f0:.2f} Hz, fp_len={best_len})")

    print(f"\nTop {top_n} matches:")
    for note, sim, f0, L in results[:top_n]:
        print(f"  {note:6s}  sim={sim:.4f}   f0={f0:8.2f} Hz   fp_len={L}")

    if plot:
        freqs, mag = freqs_mag

        plt.figure(figsize=(12, 4))
        plt.plot(freqs, mag)
        plt.title("Live Segment Magnitude Spectrum")
        plt.xlabel("Hz")
        plt.ylabel("|X(f)|")
        plt.xlim(0, min(8000, freqs[-1]))
        plt.tight_layout()
        plt.show()

        notes = [r[0] for r in results[:top_n]]
        sims = [r[1] for r in results[:top_n]]

        plt.figure(figsize=(12, 4))
        plt.bar(notes, sims)
        plt.title(f"Top {top_n} LUT Similarities (Cosine)")
        plt.xlabel("Note")
        plt.ylabel("Similarity")
        plt.tight_layout()
        plt.show()

    return best_note, best_sim


# -----------------------------
# CLI parsing
# -----------------------------
def parse_labeled_args(pairs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for s in pairs:
        if "=" not in s:
            raise ValueError(f"Expected NOTE=path, got '{s}'")
        note, path = s.split("=", 1)
        note = note.strip()
        path = path.strip()
        if not note or not path:
            raise ValueError(f"Bad NOTE=path: '{s}'")
        out.append((note, path))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build", help="Build LUT from labeled note recordings (NOTE=wav_path)")
    ap_build.add_argument("--out", required=True, help="Output LUT json")
    ap_build.add_argument("--k", type=int, default=60, help="Max harmonics K")
    ap_build.add_argument("--tol", type=float, default=15.0, help="Tolerance Hz around each harmonic")
    ap_build.add_argument("--start", type=float, default=0.10, help="Segment start sec")
    ap_build.add_argument("--dur", type=float, default=0.20, help="Segment duration sec")
    ap_build.add_argument("--a4", type=float, default=440.0, help="A4 reference")
    ap_build.add_argument("--window", type=str, default="hann", help="FFT window")
    ap_build.add_argument("--highpass", action="store_true", help="Apply a highpass to reduce rumble")
    ap_build.add_argument("--highpass_hz", type=float, default=80.0, help="Highpass cutoff Hz")
    ap_build.add_argument("--replace_existing", action="store_true",
                          help="Overwrite output file instead of merging/replacing by note")
    ap_build.add_argument("labeled", nargs="+", help="Pairs NOTE=wav_path, e.g. E6=E6.wav")

    ap_build_dir = sub.add_parser("build_dir", help="Build LUT from a directory of wavs (note parsed from filename)")
    ap_build_dir.add_argument("--out", required=True, help="Output LUT json")
    ap_build_dir.add_argument("--dir", required=True, help="Directory of wavs (filenames contain notes, e.g. E6.wav)")
    ap_build_dir.add_argument("--recursive", action="store_true", help="Search subfolders too")
    ap_build_dir.add_argument("--k", type=int, default=60, help="Max harmonics K")
    ap_build_dir.add_argument("--tol", type=float, default=15.0, help="Tolerance Hz around each harmonic")
    ap_build_dir.add_argument("--start", type=float, default=0.10, help="Segment start sec")
    ap_build_dir.add_argument("--dur", type=float, default=0.20, help="Segment duration sec")
    ap_build_dir.add_argument("--a4", type=float, default=440.0, help="A4 reference")
    ap_build_dir.add_argument("--window", type=str, default="hann", help="FFT window")
    ap_build_dir.add_argument("--highpass", action="store_true", help="Apply a highpass to reduce rumble")
    ap_build_dir.add_argument("--highpass_hz", type=float, default=80.0, help="Highpass cutoff Hz")
    ap_build_dir.add_argument("--replace_existing", action="store_true",
                              help="Overwrite output file instead of merging/replacing by note")

    ap_match = sub.add_parser("match", help="Match an unknown note recording against LUT")
    ap_match.add_argument("--lut", required=True, help="LUT json")
    ap_match.add_argument("--wav", required=True, help="Unknown wav to classify")
    ap_match.add_argument("--start", type=float, default=0.10, help="Segment start sec")
    ap_match.add_argument("--dur", type=float, default=0.20, help="Segment duration sec")
    ap_match.add_argument("--tol", type=float, default=None, help="Override tolerance Hz (else per-entry)")
    ap_match.add_argument("--k", type=int, default=None, help="Override K (else per-entry)")
    ap_match.add_argument("--window", type=str, default=None, help="Override FFT window (else per-entry)")
    ap_match.add_argument("--a4", type=float, default=None, help="Override A4 reference")
    ap_match.add_argument("--highpass", action="store_true", help="Apply a highpass to reduce rumble")
    ap_match.add_argument("--highpass_hz", type=float, default=80.0, help="Highpass cutoff Hz")
    ap_match.add_argument("--top", type=int, default=8, help="Print top N matches")
    ap_match.add_argument("--plot", action="store_true", help="Plot spectrum and top scores")

    args = ap.parse_args()

    if args.cmd == "build":
        labeled = parse_labeled_args(args.labeled)
        build_lut(
            out_path=args.out,
            labeled=labeled,
            k=args.k,
            tol_hz=args.tol,
            start=args.start,
            dur=args.dur,
            a4=args.a4,
            window=args.window,
            use_highpass=args.highpass,
            highpass_hz=args.highpass_hz,
            replace_existing=args.replace_existing,
        )
        return

    if args.cmd == "build_dir":
        wavs = gather_wavs_from_dir(args.dir, recursive=args.recursive)
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in {args.dir} (recursive={args.recursive})")

        labeled = []
        skipped = 0
        for p in wavs:
            try:
                note = note_from_filename(p)
                labeled.append((note, str(p)))
            except ValueError as ex:
                skipped += 1
                print(f"[skip] {p.name}: {ex}")

        if not labeled:
            raise RuntimeError("No wav files had parseable note names in their filename.")

        print(f"Found {len(labeled)} labeled wavs (skipped {skipped}).")

        build_lut(
            out_path=args.out,
            labeled=labeled,
            k=args.k,
            tol_hz=args.tol,
            start=args.start,
            dur=args.dur,
            a4=args.a4,
            window=args.window,
            use_highpass=args.highpass,
            highpass_hz=args.highpass_hz,
            replace_existing=args.replace_existing,
        )
        return

    if args.cmd == "match":
        match_note(
            lut_path=args.lut,
            wav_path=args.wav,
            start=args.start,
            dur=args.dur,
            tol_hz_live=args.tol,
            k_live=args.k,
            window_live=args.window,
            a4_override=args.a4,
            use_highpass=args.highpass,
            highpass_hz=args.highpass_hz,
            top_n=args.top,
            plot=args.plot
        )
        return


if __name__ == "__main__":
    main()
