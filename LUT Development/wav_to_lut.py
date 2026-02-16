#!/usr/bin/env python3
"""
wav_to_lut.py
Harmonic fingerprint LUT builder + matcher (multi-take per note + richer features).

Core idea:
- For a candidate note with known f0(note), measure harmonic energy near k*f0 for k=1..K.
- Normalize those harmonic energies -> fingerprint vector.
- Store MULTIPLE takes per note (each take is a separate lookup).
- During matching: score live vs ALL takes for each note, and use the best take score.

Upgrades for accuracy (per take, stored in LUT):
- fingerprint: normalized harmonic amplitude vector (sum-to-1)
- peak_freqs: peak frequency near each harmonic
- peak_amps:  peak magnitude near each harmonic
- harm_slope: slope of log(amp_k) vs k (spectral envelope proxy)
- inharm:     average relative deviation of peak_freqs from k*f0
- centroid_hz, rolloff_hz, flatness: simple spectral features

Commands:
  1) Build LUT from labeled recordings (NOTE=path):
    python wav_to_lut.py build --out lut.json --k 60 --tol 15 --start 0.12 --dur 0.18 \
        E3=samples/E3.wav E3=samples/E3_lowstring.wav

  2) Build LUT from a directory (auto-note from filename, supports E3_lowstring.wav):
    python wav_to_lut.py build_dir --out lut.json --dir samples --k 60 --tol 15 --start 0.12 --dur 0.18

  3) Match an unknown recording:
    python wav_to_lut.py match --lut lut.json --wav unknown.wav --start 0.12 --dur 0.18 --plot

Optional:
  --plot : show spectrum + note scores
  --top  : show top N matches
  --highpass : apply simple highpass to reduce rumble (optional)
  --score_mode : how to collapse multiple take scores into a note score: max | mean | topk
  --topk : if score_mode=topk, average the top-k take scores

Requirements:
  pip install numpy scipy matplotlib
"""

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

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

# IMPORTANT: works for "E3_lowstring.wav" and similar (underscore-safe).
# Matches: optional non-alnum boundary, then note letter, optional accidental, octave,
# then requires non-alnum boundary or end.
_NOTE_TOKEN_RE = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9])([A-G])([#b]?)(-?\d+)(?=[^A-Za-z0-9]|$)"
)


def normalize_note_token(letter: str, accidental: str, octave_str: str) -> str:
    """Canonical form like Bb3, F#4, E6."""
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
    # Normalize: allow unicode accidentals; flats as 'b'
    s = note.strip().replace("♯", "#").replace("♭", "b")
    m = re.match(r"^([A-Ga-g])([#bB]?)(-?\d+)$", s)
    if not m:
        raise ValueError(f"Bad note format '{note}'. Use like E6, F#4, Bb3.")
    letter, accidental, octave_str = m.group(1).upper(), m.group(2), m.group(3)

    # Use _NOTE_TO_SEMITONE keys: sharps '#', flats 'B'
    if accidental in ("b", "B"):
        accidental_key = "B"
    elif accidental == "#":
        accidental_key = "#"
    else:
        accidental_key = ""

    pitch = letter + accidental_key
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
      E3_lowstring.wav -> E3
    """
    stem = path.stem
    m = _NOTE_TOKEN_RE.search(stem)
    if not m:
        raise ValueError(
            f"Could not parse note from filename '{path.name}'. "
            f"Expected something like E6.wav, F#4.wav, Bb3.wav, E3_lowstring.wav."
        )
    letter, accidental, octave_str = m.group(1), m.group(2), m.group(3)
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
def spectrum_mag(seg: np.ndarray, fs: int, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
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


def fingerprint_for_note(
    seg: np.ndarray,
    fs: int,
    note_f0_hz: float,
    k: int,
    tol_hz: float,
    window: str = "hann",
) -> np.ndarray:
    """Extract a normalized harmonic fingerprint from seg ASSUMING candidate note's f0."""
    freqs, mag = spectrum_mag(seg, fs, window=window)
    nyq = float(freqs[-1])

    amps: List[float] = []
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
# Extra features for accuracy
# -----------------------------
def harmonic_peaks(freqs: np.ndarray,
                   mag: np.ndarray,
                   f0: float,
                   k: int,
                   tol_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For k harmonics, find the peak frequency + peak magnitude near k*f0 within +/- tol_hz.
    Returns:
      peak_freqs[k_used], peak_amps[k_used]
    """
    nyq = float(freqs[-1])
    peak_fs: List[float] = []
    peak_as: List[float] = []
    for h in range(1, k + 1):
        center = h * f0
        if center >= nyq:
            break
        lo = max(0.0, center - tol_hz)
        hi = center + tol_hz
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            peak_fs.append(float(center))
            peak_as.append(0.0)
            continue
        f_band = freqs[mask]
        m_band = mag[mask]
        idx = int(np.argmax(m_band))
        peak_fs.append(float(f_band[idx]))
        peak_as.append(float(m_band[idx]))
    return np.array(peak_fs, dtype=np.float64), np.array(peak_as, dtype=np.float64)


def harmonic_slope(peak_amps: np.ndarray) -> float:
    """Fit log(amp) vs harmonic index. Returns slope (negative for typical signals)."""
    if len(peak_amps) < 3:
        return 0.0
    amps = np.maximum(peak_amps, 1e-12)
    y = np.log(amps)
    x = np.arange(1, len(amps) + 1, dtype=np.float64)
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = float(np.dot(x0, x0)) + 1e-12
    return float(np.dot(x0, y0) / denom)


def inharmonicity(peak_freqs: np.ndarray, f0: float) -> float:
    """Average relative deviation of measured harmonic peaks from k*f0."""
    if len(peak_freqs) == 0 or f0 <= 0:
        return 0.0
    k = np.arange(1, len(peak_freqs) + 1, dtype=np.float64)
    ideal = k * float(f0)
    rel = np.abs((peak_freqs / (ideal + 1e-12)) - 1.0)
    return float(np.mean(rel))


def spectral_features(freqs: np.ndarray, mag: np.ndarray) -> Dict[str, float]:
    """Simple spectral stats computed from magnitude spectrum."""
    m = mag.astype(np.float64)
    s = float(np.sum(m)) + 1e-12

    centroid = float(np.sum(freqs * m) / s)

    cumsum = np.cumsum(m)
    roll_p = 0.85
    roll_idx = int(np.searchsorted(cumsum, roll_p * cumsum[-1]))
    rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])

    # flatness: geometric mean / arithmetic mean on power-ish (use m^2)
    p = np.maximum(m * m, 1e-24)
    flatness = float(np.exp(np.mean(np.log(p))) / (np.mean(p) + 1e-12))

    return {"centroid_hz": centroid, "rolloff_hz": rolloff, "flatness": flatness}


def features_for_candidate(seg: np.ndarray,
                           fs: int,
                           f0: float,
                           k: int,
                           tol_hz: float,
                           window: str) -> Dict[str, Any]:
    freqs, mag = spectrum_mag(seg, fs, window=window)

    pk_f, pk_a = harmonic_peaks(freqs, mag, f0, k=k, tol_hz=tol_hz)
    fp = fingerprint_for_note(seg, fs, f0, k=k, tol_hz=tol_hz, window=window)

    slope = harmonic_slope(pk_a)
    inharm = inharmonicity(pk_f, f0)
    spec = spectral_features(freqs, mag)

    return {
        "fingerprint": [float(v) for v in fp.tolist()],
        "peak_freqs": [float(v) for v in pk_f.tolist()],
        "peak_amps": [float(v) for v in pk_a.tolist()],
        "harm_slope": float(slope),
        "inharm": float(inharm),
        "centroid_hz": float(spec["centroid_hz"]),
        "rolloff_hz": float(spec["rolloff_hz"]),
        "flatness": float(spec["flatness"]),
    }


def combined_score(live_take: Dict[str, Any],
                   ref_take: Dict[str, Any],
                   w_inharm: float,
                   w_centroid: float,
                   w_slope: float,
                   w_rolloff: float,
                   w_flatness: float) -> float:
    """
    Combine cosine similarity of harmonic fingerprint with small penalties on other features.
    Higher is better.
    """
    fp_live = np.array(live_take.get("fingerprint", []), dtype=np.float64)
    fp_ref = np.array(ref_take.get("fingerprint", []), dtype=np.float64)
    sim = cosine_similarity(fp_live, fp_ref)

    # penalties
    inharm_pen = abs(float(live_take.get("inharm", 0.0)) - float(ref_take.get("inharm", 0.0)))

    c_live = float(live_take.get("centroid_hz", 0.0)) + 1e-12
    c_ref = float(ref_take.get("centroid_hz", 0.0)) + 1e-12
    centroid_pen = abs(float(np.log(c_live / c_ref)))

    r_live = float(live_take.get("rolloff_hz", 0.0)) + 1e-12
    r_ref = float(ref_take.get("rolloff_hz", 0.0)) + 1e-12
    rolloff_pen = abs(float(np.log(r_live / r_ref)))

    slope_pen = abs(float(live_take.get("harm_slope", 0.0)) - float(ref_take.get("harm_slope", 0.0)))

    f_live = float(live_take.get("flatness", 0.0))
    f_ref = float(ref_take.get("flatness", 0.0))
    flat_pen = abs(f_live - f_ref)

    return float(
        sim
        - w_inharm * inharm_pen
        - w_centroid * centroid_pen
        - w_rolloff * rolloff_pen
        - w_slope * slope_pen
        - w_flatness * flat_pen
    )


def collapse_take_scores(scores: List[float], mode: str, topk: int) -> float:
    if not scores:
        return 0.0
    mode = mode.lower()
    if mode == "max":
        return float(max(scores))
    if mode == "mean":
        return float(np.mean(scores))
    if mode == "topk":
        k = max(1, int(topk))
        ss = sorted(scores, reverse=True)[:k]
        return float(np.mean(ss))
    raise ValueError(f"Unknown score_mode '{mode}'. Use max|mean|topk.")


# -----------------------------
# LUT I/O
# -----------------------------
def load_lut(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"entries": [], "meta": {}}
        if "entries" not in data or not isinstance(data["entries"], list):
            data["entries"] = []
        if "meta" not in data or not isinstance(data["meta"], dict):
            data["meta"] = {}
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"LUT file not found: {path}")


def save_lut(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# Build LUT
# -----------------------------
@dataclass
class LutEntry:
    note: str
    midi: int
    f0_hz: float
    k: int
    tol_hz: float
    window: str
    takes: List[Dict[str, Any]]
    source_wavs: List[str]
    analysis_start_sec: float
    analysis_dur_sec: float


def build_lut(
    out_path: str,
    labeled: List[Tuple[str, str]],
    k: int,
    tol_hz: float,
    start: float,
    dur: float,
    a4: float,
    window: str,
    use_highpass: bool,
    highpass_hz: float,
    replace_existing: bool,
):
    # Start new or merge into existing
    if not replace_existing and Path(out_path).exists():
        lut = load_lut(out_path)
    else:
        lut = {"meta": {}, "entries": []}

    lut["meta"].setdefault("type", "harmonic_fingerprint_lut")
    lut["meta"].setdefault("normalization", "sum_to_1")
    lut["meta"].setdefault("similarity", "cosine+features")
    lut["meta"]["a4_hz"] = float(a4)
    lut["meta"]["multi_take"] = {
        "enabled": True,
        "merge": "append_per_take",
        "note_score_default": "max",
    }

    # Index existing entries by note
    existing_idx: Dict[str, int] = {}
    for i, e in enumerate(lut.get("entries", [])):
        if isinstance(e, dict) and "note" in e:
            existing_idx[str(e["note"]).strip().upper()] = i

    # Group inputs by note key
    grouped: Dict[str, List[str]] = defaultdict(list)
    note_canonical: Dict[str, str] = {}
    for note, wav_path in labeled:
        key = note.strip().upper()
        grouped[key].append(wav_path)
        note_canonical.setdefault(key, note.strip())

    for key, wav_paths in grouped.items():
        note = note_canonical[key]
        midi = note_to_midi(note)
        f0 = midi_to_freq_hz(midi, a4_hz=a4)

        new_takes: List[Dict[str, Any]] = []
        new_sources: List[str] = []

        for wav_path in wav_paths:
            fs, x = wavfile.read(wav_path)
            x = to_mono(x)
            if x.dtype.kind in "iu":
                x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
            x = normalize_peak(x)

            seg = pick_segment(x, fs, start, dur)
            if use_highpass:
                seg = highpass_filter(seg, fs, cutoff_hz=highpass_hz)
            seg = normalize_peak(seg)

            take = features_for_candidate(seg, fs, f0, k=k, tol_hz=tol_hz, window=window)
            new_takes.append(take)
            new_sources.append(str(wav_path))

        if key in existing_idx and not replace_existing:
            old = lut["entries"][existing_idx[key]]

            old_takes = old.get("takes", [])
            old_sources = old.get("source_wavs", [])
            if isinstance(old_sources, str):
                old_sources = [old_sources]

            # Backward compatibility: if old LUT used fingerprints only
            if not old_takes and "fingerprints" in old:
                fps = old.get("fingerprints", [])
                srcs = old_sources if old_sources else ["<unknown_take>"] * len(fps)
                rebuilt = []
                for fp, src in zip(fps, srcs):
                    rebuilt.append({
                        "fingerprint": fp,
                        "peak_freqs": [],
                        "peak_amps": [],
                        "harm_slope": 0.0,
                        "inharm": 0.0,
                        "centroid_hz": 0.0,
                        "rolloff_hz": 0.0,
                        "flatness": 0.0,
                    })
                old_takes = rebuilt

            old["note"] = note
            old["midi"] = midi
            old["f0_hz"] = float(f0)
            old["k"] = int(k)
            old["tol_hz"] = float(tol_hz)
            old["window"] = window
            old["takes"] = list(old_takes) + new_takes
            old["source_wavs"] = list(old_sources) + new_sources
            old["analysis_start_sec"] = float(start)
            old["analysis_dur_sec"] = float(dur)

            lut["entries"][existing_idx[key]] = old
            print(f"[build] merged  {note}: added_takes={len(new_sources)} total_takes={len(old['takes'])}")

        elif key in existing_idx and replace_existing:
            entry = LutEntry(
                note=note,
                midi=midi,
                f0_hz=float(f0),
                k=int(k),
                tol_hz=float(tol_hz),
                window=window,
                takes=new_takes,
                source_wavs=new_sources,
                analysis_start_sec=float(start),
                analysis_dur_sec=float(dur),
            ).__dict__
            lut["entries"][existing_idx[key]] = entry
            print(f"[build] replaced {note}: takes={len(new_sources)}")

        else:
            entry = LutEntry(
                note=note,
                midi=midi,
                f0_hz=float(f0),
                k=int(k),
                tol_hz=float(tol_hz),
                window=window,
                takes=new_takes,
                source_wavs=new_sources,
                analysis_start_sec=float(start),
                analysis_dur_sec=float(dur),
            ).__dict__
            lut["entries"].append(entry)
            print(f"[build] added   {note}: takes={len(new_sources)}")

    save_lut(out_path, lut)
    print(f"Saved LUT -> {out_path} with {len(lut['entries'])} note entries")


def gather_wavs_from_dir(dir_path: str, recursive: bool) -> List[Path]:
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    glob = root.rglob("*.wav") if recursive else root.glob("*.wav")
    return sorted([p for p in glob if p.is_file()])


# -----------------------------
# Match
# -----------------------------
def match_note(
    lut_path: str,
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
    plot: bool,
    score_mode: str,
    topk: int,
    w_inharm: float,
    w_centroid: float,
    w_slope: float,
    w_rolloff: float,
    w_flatness: float,
):
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

    a4_lut = float(lut.get("meta", {}).get("a4_hz", 440.0))
    a4 = float(a4_override) if a4_override is not None else a4_lut

    freqs_mag = None
    if plot:
        freqs_mag = spectrum_mag(seg, fs, window=window_live or entries[0].get("window", "hann"))

    results = []
    for e in entries:
        note = e.get("note", "?")
        midi = int(e.get("midi", 0))
        f0 = midi_to_freq_hz(midi, a4_hz=a4)

        k = int(k_live) if k_live is not None else int(e.get("k", 60))
        tol = float(tol_hz_live) if tol_hz_live is not None else float(e.get("tol_hz", 12.0))
        win = window_live or e.get("window", "hann")

        # Live features *for this candidate f0*
        live_take = features_for_candidate(seg, fs, f0, k=k, tol_hz=tol, window=win)

        takes = e.get("takes", [])
        # Backward compatibility: older LUTs might have "fingerprints"
        if not takes and "fingerprints" in e:
            takes = [{"fingerprint": fp, "peak_freqs": [], "peak_amps": [], "harm_slope": 0.0,
                      "inharm": 0.0, "centroid_hz": 0.0, "rolloff_hz": 0.0, "flatness": 0.0}
                     for fp in e.get("fingerprints", [])]
        if not takes and "fingerprint" in e:
            takes = [{"fingerprint": e.get("fingerprint", []), "peak_freqs": [], "peak_amps": [], "harm_slope": 0.0,
                      "inharm": 0.0, "centroid_hz": 0.0, "rolloff_hz": 0.0, "flatness": 0.0}]

        take_scores = [
            combined_score(
                live_take, t,
                w_inharm=w_inharm,
                w_centroid=w_centroid,
                w_slope=w_slope,
                w_rolloff=w_rolloff,
                w_flatness=w_flatness
            )
            for t in takes
        ]

        note_score = collapse_take_scores(take_scores, mode=score_mode, topk=topk)
        best_take_idx = int(np.argmax(take_scores)) if take_scores else -1

        results.append((note, note_score, f0, len(live_take.get("fingerprint", [])), len(takes), best_take_idx))

    results.sort(key=lambda t: t[1], reverse=True)

    best_note, best_sim, best_f0, best_len, best_num_takes, best_take_idx = results[0]
    print(
        f"\nBest match: {best_note}  (score={best_sim:.4f}, f0={best_f0:.2f} Hz, "
        f"fp_len={best_len}, takes={best_num_takes}, best_take={best_take_idx})"
    )

    print(f"\nTop {top_n} matches:")
    for note, sim, f0, L, takes, take_idx in results[:top_n]:
        print(f"  {note:6s}  score={sim:.4f}   f0={f0:8.2f} Hz   fp_len={L}   takes={takes} best_take={take_idx}")

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
        sims_plot = [r[1] for r in results[:top_n]]

        plt.figure(figsize=(12, 4))
        plt.bar(notes, sims_plot)
        plt.title(f"Top {top_n} Note Scores ({score_mode})")
        plt.xlabel("Note")
        plt.ylabel("Score")
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
    ap_build.add_argument(
        "--replace_existing",
        action="store_true",
        help="Overwrite output file instead of merging/appending by note",
    )
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
    ap_build_dir.add_argument(
        "--replace_existing",
        action="store_true",
        help="Overwrite output file instead of merging/appending by note",
    )

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

    # NEW: scoring controls
    ap_match.add_argument("--score_mode", type=str, default="max", choices=["max", "mean", "topk"],
                          help="How to collapse multiple take scores into a note score")
    ap_match.add_argument("--topk", type=int, default=3,
                          help="If score_mode=topk, average the top-k take scores")

    # NEW: feature weights (tweak if needed)
    ap_match.add_argument("--w_inharm", type=float, default=2.0, help="Penalty weight for inharmonicity difference")
    ap_match.add_argument("--w_centroid", type=float, default=0.5, help="Penalty weight for centroid ratio")
    ap_match.add_argument("--w_rolloff", type=float, default=0.25, help="Penalty weight for rolloff ratio")
    ap_match.add_argument("--w_slope", type=float, default=0.5, help="Penalty weight for harmonic slope difference")
    ap_match.add_argument("--w_flatness", type=float, default=0.25, help="Penalty weight for flatness difference")

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
            plot=args.plot,
            score_mode=args.score_mode,
            topk=args.topk,
            w_inharm=args.w_inharm,
            w_centroid=args.w_centroid,
            w_slope=args.w_slope,
            w_rolloff=args.w_rolloff,
            w_flatness=args.w_flatness,
        )
        return


if __name__ == "__main__":
    main()
