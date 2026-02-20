#!/usr/bin/env python3
"""
wav_to_lut_improved_strings.py
Harmonic fingerprint LUT builder + matcher with stronger note distinctiveness AND string-aware classes.

Your filename convention (low E to high e):
  - No suffix => string 0 (low E)
  - _1 => A string
  - _2 => D string
  - _3 => G string
  - _4 => B string
  - _5 => high e string

Examples:
  E2.wav      -> label "E2_0", base_note "E2", string_idx 0
  E4_5.wav    -> label "E4_5", base_note "E4", string_idx 5
  F#3_2.wav   -> label "F#3_2", base_note "F#3", string_idx 2

Key improvements:
- LUT stores per-take log-frequency band templates ("spectral stamps") to separate neighbors and strings.
- LUT entries are STRING-AWARE classes (label includes _idx), so takes do not get merged across strings.
- Monophonic matching uses stored band templates (LUT does heavy lifting).
- Polyphonic matching uses mixture of stored band templates (NNLS or sparse OMP).

Commands:
  1) Build LUT from labeled recordings (LABEL=path):
     python wav_to_lut.py build --out lut.json --k 60 --tol 15 --start 0.12 --dur 0.18 \
         E4_5=samples/E4_5.wav E4_0=samples/E4.wav

  2) Build LUT from directory (auto from filename, supports E3.wav, E3_2.wav):
     python wav_to_lut.py build_dir --out lut.json --dir samples --k 60 --tol 15 --start 0.12 --dur 0.18

  3) Match monophonic:
     python wav_to_lut.py match --lut lut.json --wav unknown.wav --start 0.12 --dur 0.18 --plot

  4) Match polyphonic:
     python wav_to_lut.py match_poly --lut lut.json --wav chord.wav --start 0.12 --dur 0.25 \
         --algo omp --max_notes 6 --thresh 0.25 --prune 40 --plot

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
from scipy.optimize import nnls
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

# Matches a note token in a filename stem (underscore-safe).
_NOTE_TOKEN_RE = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9])([A-G])([#b]?)(-?\d+)(?=[^A-Za-z0-9]|$)"
)

# Matches optional "_<digit 0..5>" string index at end of stem
_STRING_SUFFIX_RE = re.compile(r".*?_([0-5])$")


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


def split_label(label: str) -> Tuple[str, int]:
    """
    'E4_5' -> ('E4', 5)
    'E2'   -> ('E2', 0)
    """
    s = label.strip().replace("♯", "#").replace("♭", "b")
    m = re.match(r"^([A-Ga-g])([#bB]?)(-?\d+)(?:_([0-5]))?$", s)
    if not m:
        raise ValueError(f"Bad label '{label}'. Expected like E2 or E4_5.")
    base = normalize_note_token(m.group(1), (m.group(2) or "").lower(), m.group(3))
    idx = int(m.group(4)) if m.group(4) is not None else 0
    return base, idx


def label_from_filename(path: Path) -> Tuple[str, str, int]:
    """
    Extract (label, base_note, string_idx) from filename.

    E2.wav   -> (E2_0, E2, 0)
    E4_5.wav -> (E4_5, E4, 5)
    """
    stem = path.stem

    m = _NOTE_TOKEN_RE.search(stem)
    if not m:
        raise ValueError(
            f"Could not parse note from filename '{path.name}'. "
            f"Expected E2.wav or E4_5.wav etc."
        )
    base_note = normalize_note_token(m.group(1), m.group(2).lower(), m.group(3))

    string_idx = 0
    ms = _STRING_SUFFIX_RE.match(stem)
    if ms and ms.group(1) is not None:
        string_idx = int(ms.group(1))

    label = f"{base_note}_{string_idx}"
    return label, base_note, string_idx


# -----------------------------
# Audio helpers
# -----------------------------
def read_wav_mono_float(wav_path: str) -> Tuple[int, np.ndarray]:
    """
    Read wav and return (fs, mono_float64) normalized to [-1,1] if integer PCM.
    """
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
# Spectrum + harmonic fingerprint extraction (kept for compatibility + features)
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


def cents_to_hz_width(center_hz: float, tol_cents: float) -> float:
    if center_hz <= 0:
        return 0.0
    r = 2.0 ** (tol_cents / 1200.0)
    return float(center_hz * (r - 1.0))


def get_tol_hz(center_hz: float, tol_value: float, tol_mode: str) -> float:
    tol_mode = (tol_mode or "hz").lower()
    if tol_mode == "hz":
        return float(tol_value)
    if tol_mode == "cents":
        return cents_to_hz_width(center_hz, float(tol_value))
    raise ValueError(f"Unknown tol_mode '{tol_mode}'. Use hz|cents.")


def fingerprint_for_note(
    seg: np.ndarray,
    fs: int,
    note_f0_hz: float,
    k: int,
    tol_value: float,
    tol_mode: str = "hz",
    window: str = "hann",
) -> np.ndarray:
    freqs, mag = spectrum_mag(seg, fs, window=window)
    nyq = float(freqs[-1])

    amps: List[float] = []
    for h in range(1, k + 1):
        fh = h * note_f0_hz
        if fh >= nyq:
            break
        tol_hz = get_tol_hz(fh, tol_value, tol_mode)
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
# NEW: log-frequency band templates (big distinctiveness boost)
# -----------------------------
def make_log_band_edges(fmin_hz: float, fmax_hz: float, n_bands: int) -> np.ndarray:
    fmin_hz = max(1e-3, float(fmin_hz))
    fmax_hz = max(fmin_hz * 1.001, float(fmax_hz))
    n_bands = max(8, int(n_bands))
    return np.geomspace(fmin_hz, fmax_hz, n_bands + 1).astype(np.float64)


def pool_to_log_bands(freqs: np.ndarray, mag: np.ndarray, edges: np.ndarray, agg: str = "mean") -> np.ndarray:
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    agg = (agg or "mean").lower()
    for i in range(len(out)):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            out[i] = 0.0
        else:
            band = mag[mask]
            out[i] = float(np.mean(band)) if agg == "mean" else float(np.max(band))
    return out


def build_harmonic_band_mask(edges: np.ndarray,
                            f0: float,
                            k: int,
                            tol_value: float,
                            tol_mode: str,
                            fmax: float) -> np.ndarray:
    nb = len(edges) - 1
    mask = np.zeros(nb, dtype=np.float64)
    if f0 <= 0:
        return mask

    centers = []
    for h in range(1, k + 1):
        fh = h * f0
        if fh >= fmax:
            break
        tol_hz = get_tol_hz(fh, tol_value, tol_mode)
        centers.append((fh, tol_hz))

    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        mid = 0.5 * (lo + hi)
        for fh, tol_hz in centers:
            if abs(mid - fh) <= tol_hz:
                mask[i] = 1.0
                break
    return mask


def make_band_template(seg: np.ndarray,
                       fs: int,
                       window: str,
                       edges: np.ndarray,
                       f0: float,
                       k_mask: int,
                       mask_tol: float,
                       mask_tol_mode: str,
                       agg: str,
                       logmag: bool) -> Tuple[np.ndarray, np.ndarray]:
    freqs, mag = spectrum_mag(seg, fs, window=window)
    y = mag.astype(np.float64)
    if logmag:
        y = np.log1p(y)

    band = pool_to_log_bands(freqs, y, edges, agg=agg)
    band /= (np.linalg.norm(band) + 1e-12)

    fmax = float(edges[-1])
    mask = build_harmonic_band_mask(edges, f0, k=k_mask, tol_value=mask_tol, tol_mode=mask_tol_mode, fmax=fmax)
    return band, mask


# -----------------------------
# Extra features (optional, stored in takes)
# -----------------------------
def harmonic_peaks(freqs: np.ndarray,
                   mag: np.ndarray,
                   f0: float,
                   k: int,
                   tol_value: float,
                   tol_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    nyq = float(freqs[-1])
    peak_fs: List[float] = []
    peak_as: List[float] = []
    for h in range(1, k + 1):
        center = h * f0
        if center >= nyq:
            break
        tol_hz = get_tol_hz(center, tol_value, tol_mode)
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
    if len(peak_freqs) == 0 or f0 <= 0:
        return 0.0
    k = np.arange(1, len(peak_freqs) + 1, dtype=np.float64)
    ideal = k * float(f0)
    rel = np.abs((peak_freqs / (ideal + 1e-12)) - 1.0)
    return float(np.mean(rel))


def spectral_features(freqs: np.ndarray, mag: np.ndarray) -> Dict[str, float]:
    m = mag.astype(np.float64)
    s = float(np.sum(m)) + 1e-12
    centroid = float(np.sum(freqs * m) / s)

    cumsum = np.cumsum(m)
    roll_p = 0.85
    roll_idx = int(np.searchsorted(cumsum, roll_p * cumsum[-1]))
    rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])

    p = np.maximum(m * m, 1e-24)
    flatness = float(np.exp(np.mean(np.log(p))) / (np.mean(p) + 1e-12))
    return {"centroid_hz": centroid, "rolloff_hz": rolloff, "flatness": flatness}


def features_for_candidate(seg: np.ndarray,
                           fs: int,
                           f0: float,
                           k: int,
                           tol_value: float,
                           tol_mode: str,
                           window: str) -> Dict[str, Any]:
    freqs, mag = spectrum_mag(seg, fs, window=window)
    pk_f, pk_a = harmonic_peaks(freqs, mag, f0, k=k, tol_value=tol_value, tol_mode=tol_mode)
    fp = fingerprint_for_note(seg, fs, f0, k=k, tol_value=tol_value, tol_mode=tol_mode, window=window)
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
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"entries": [], "meta": {}}
    if "entries" not in data or not isinstance(data["entries"], list):
        data["entries"] = []
    if "meta" not in data or not isinstance(data["meta"], dict):
        data["meta"] = {}
    return data


def save_lut(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# Build LUT
# -----------------------------
@dataclass
class LutEntry:
    note: str               # string-aware label, e.g. "E4_5"
    midi: int               # based on base_note only
    f0_hz: float            # based on base_note only
    k: int
    tol_hz: float
    window: str
    takes: List[Dict[str, Any]]
    source_wavs: List[str]
    analysis_start_sec: float
    analysis_dur_sec: float
    base_note: str          # e.g. "E4"
    string_idx: int         # 0..5


def gather_wavs_from_dir(dir_path: str, recursive: bool) -> List[Path]:
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    glob = root.rglob("*.wav") if recursive else root.glob("*.wav")
    return sorted([p for p in glob if p.is_file()])


def build_lut(
    out_path: str,
    labeled: List[Tuple[str, str]],   # label, wav_path
    k: int,
    tol_hz: float,
    start: float,
    dur: float,
    a4: float,
    window: str,
    use_highpass: bool,
    highpass_hz: float,
    replace_existing: bool,

    # band-template params
    band_fmin: float,
    band_fmax: float,
    band_n: int,
    band_agg: str,
    band_logmag: bool,
    mask_k: int,
    mask_tol: float,
    mask_tol_mode: str,
):
    if not replace_existing and Path(out_path).exists():
        lut = load_lut(out_path)
    else:
        lut = {"meta": {}, "entries": []}

    lut["meta"].setdefault("type", "harmonic_fingerprint_lut_v2_stringaware")
    lut["meta"].setdefault("normalization", "sum_to_1 (fingerprint), L2 (band_template)")
    lut["meta"]["a4_hz"] = float(a4)
    lut["meta"]["multi_take"] = {"enabled": True, "merge": "append_per_take", "note_score_default": "max"}

    lut["meta"]["band_template"] = {
        "enabled": True,
        "fmin_hz": float(band_fmin),
        "fmax_hz": float(band_fmax),
        "n_bands": int(band_n),
        "agg": str(band_agg),
        "logmag": bool(band_logmag),
        "harm_mask": {"k": int(mask_k), "tol": float(mask_tol), "tol_mode": str(mask_tol_mode)},
    }

    edges = make_log_band_edges(band_fmin, band_fmax, band_n)

    # Index existing entries by LABEL (string-aware)
    existing_idx: Dict[str, int] = {}
    for i, e in enumerate(lut.get("entries", [])):
        if isinstance(e, dict) and "note" in e:
            existing_idx[str(e["note"]).strip().upper()] = i

    # Group inputs by label key
    grouped: Dict[str, List[str]] = defaultdict(list)
    label_canonical: Dict[str, str] = {}
    for label, wav_path in labeled:
        key = label.strip().upper()
        grouped[key].append(wav_path)
        label_canonical.setdefault(key, label.strip())

    for key, wav_paths in grouped.items():
        label = label_canonical[key]          # e.g. "E4_5"
        base_note, string_idx = split_label(label)

        midi = note_to_midi(base_note)
        f0 = midi_to_freq_hz(midi, a4_hz=a4)

        new_takes: List[Dict[str, Any]] = []
        new_sources: List[str] = []

        for wav_path in wav_paths:
            fs, x = read_wav_mono_float(wav_path)
            x = normalize_peak(x)

            seg = pick_segment(x, fs, start, dur)
            if use_highpass:
                seg = highpass_filter(seg, fs, cutoff_hz=highpass_hz)
            seg = normalize_peak(seg)

            take = features_for_candidate(seg, fs, f0, k=k, tol_value=tol_hz, tol_mode="hz", window=window)

            band_vec, harm_mask = make_band_template(
                seg=seg,
                fs=fs,
                window=window,
                edges=edges,
                f0=f0,
                k_mask=mask_k,
                mask_tol=mask_tol,
                mask_tol_mode=mask_tol_mode,
                agg=band_agg,
                logmag=band_logmag,
            )
            take["band_template"] = [float(v) for v in band_vec.tolist()]
            take["harm_band_mask"] = [float(v) for v in harm_mask.tolist()]

            new_takes.append(take)
            new_sources.append(str(wav_path))

        if key in existing_idx and not replace_existing:
            old = lut["entries"][existing_idx[key]]
            old_takes = old.get("takes", [])
            old_sources = old.get("source_wavs", [])
            if isinstance(old_sources, str):
                old_sources = [old_sources]

            # Backward compat if old entry had fingerprints
            if not old_takes and "fingerprints" in old:
                fps = old.get("fingerprints", [])
                srcs = old_sources if old_sources else ["<unknown_take>"] * len(fps)
                rebuilt = []
                for fp, _src in zip(fps, srcs):
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

            old["note"] = label
            old["base_note"] = base_note
            old["string_idx"] = int(string_idx)
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
            print(f"[build] merged  {label}: added_takes={len(new_sources)} total_takes={len(old['takes'])}")

        elif key in existing_idx and replace_existing:
            entry = LutEntry(
                note=label,
                base_note=base_note,
                string_idx=int(string_idx),
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
            print(f"[build] replaced {label}: takes={len(new_sources)}")

        else:
            entry = LutEntry(
                note=label,
                base_note=base_note,
                string_idx=int(string_idx),
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
            print(f"[build] added   {label}: takes={len(new_sources)}")

    save_lut(out_path, lut)
    print(f"Saved LUT -> {out_path} with {len(lut['entries'])} class entries")


# -----------------------------
# Scoring: primarily band templates
# -----------------------------
def score_take_band(live_band: np.ndarray, take: Dict[str, Any], offharm_alpha: float) -> float:
    tb = take.get("band_template", None)
    if tb is None:
        return -1e9
    take_band = np.array(tb, dtype=np.float64)
    sim = cosine_similarity(live_band, take_band)

    if offharm_alpha <= 0.0:
        return float(sim)

    hm = take.get("harm_band_mask", None)
    if hm is None:
        return float(sim)

    harm_mask = (np.array(hm, dtype=np.float64) > 0.5).astype(np.float64)
    inv = 1.0 - harm_mask

    yin = float(np.sum(np.abs(live_band) * harm_mask)) + 1e-12
    yout = float(np.sum(np.abs(live_band) * inv))
    live_ratio = yout / yin

    tin = float(np.sum(np.abs(take_band) * harm_mask)) + 1e-12
    tout = float(np.sum(np.abs(take_band) * inv))
    take_ratio = tout / tin

    pen = abs(live_ratio - take_ratio)
    return float(sim - offharm_alpha * pen)


def build_live_band_from_lut(seg: np.ndarray, fs: int, lut_meta: Dict[str, Any], window: str) -> np.ndarray:
    bt = lut_meta.get("band_template", {}) if isinstance(lut_meta, dict) else {}
    fmin = float(bt.get("fmin_hz", 40.0))
    fmax = float(bt.get("fmax_hz", 8000.0))
    nb = int(bt.get("n_bands", 360))
    agg = str(bt.get("agg", "mean"))
    logmag = bool(bt.get("logmag", True))

    edges = make_log_band_edges(fmin, fmax, nb)
    freqs, mag = spectrum_mag(seg, fs, window=window)
    y = mag.astype(np.float64)
    if logmag:
        y = np.log1p(y)
    band = pool_to_log_bands(freqs, y, edges, agg=agg)
    band /= (np.linalg.norm(band) + 1e-12)
    return band


# -----------------------------
# Match (monophonic)
# -----------------------------
def match_note(
    lut_path: str,
    wav_path: str,
    start: float,
    dur: float,
    window_live: Optional[str],
    a4_override: Optional[float],
    use_highpass: bool,
    highpass_hz: float,
    top_n: int,
    plot: bool,
    score_mode: str,
    topk: int,
    offharm_alpha: float,
):
    lut = load_lut(lut_path)
    entries = lut.get("entries", [])
    if not entries:
        raise ValueError("LUT has no entries.")

    fs, x = read_wav_mono_float(wav_path)
    x = normalize_peak(x)

    seg = pick_segment(x, fs, start, dur)
    if use_highpass:
        seg = highpass_filter(seg, fs, cutoff_hz=highpass_hz)
    seg = normalize_peak(seg)

    # kept for LUT meta consistency (even though mono doesn't need f0 anymore)
    a4_lut = float(lut.get("meta", {}).get("a4_hz", 440.0))
    _a4 = float(a4_override) if a4_override is not None else a4_lut
    _ = _a4  # silence "unused"

    win = window_live or str(entries[0].get("window", "hann"))

    live_band = build_live_band_from_lut(seg, fs, lut.get("meta", {}), window=win)

    freqs_mag = spectrum_mag(seg, fs, window=win) if plot else None

    results: List[Tuple[str, float, str, int, int]] = []
    for e in entries:
        label = str(e.get("note", "?"))           # string-aware label
        base_note = str(e.get("base_note", ""))   # base note
        string_idx = int(e.get("string_idx", 0))

        takes = e.get("takes", [])
        if not takes and "fingerprints" in e:
            takes = [{"fingerprint": fp} for fp in e.get("fingerprints", [])]
        if not takes and "fingerprint" in e:
            takes = [{"fingerprint": e.get("fingerprint", [])}]

        take_scores: List[float] = []
        for t in takes:
            if "band_template" in t:
                s = score_take_band(live_band, t, offharm_alpha=offharm_alpha)
            else:
                # fallback (rare): harmonic fingerprint similarity if no template present
                fp_live = fingerprint_for_note(seg, fs, float(e.get("f0_hz", 0.0)), k=int(e.get("k", 60)),
                                               tol_value=float(e.get("tol_hz", 15.0)), tol_mode="hz",
                                               window=win)
                fp_ref = np.array(t.get("fingerprint", []), dtype=np.float64)
                s = cosine_similarity(fp_live, fp_ref)
            take_scores.append(float(s))

        note_score = collapse_take_scores(take_scores, mode=score_mode, topk=topk)
        best_take_idx = int(np.argmax(take_scores)) if take_scores else -1

        results.append((label, note_score, base_note, string_idx, best_take_idx))

    results.sort(key=lambda t: t[1], reverse=True)

    best_label, best_score, best_base, best_sidx, best_take_idx = results[0]
    print(f"\nBest match: {best_label}  (score={best_score:.4f}, base={best_base}, string={best_sidx}, best_take={best_take_idx})")

    print(f"\nTop {top_n} matches:")
    for label, s, base, sidx, bti in results[:top_n]:
        print(f"  {label:8s} score={s:.4f}   base={base:4s}  string={sidx}  best_take={bti}")

    if plot and freqs_mag is not None:
        freqs, mag = freqs_mag

        plt.figure(figsize=(12, 4))
        plt.plot(freqs, mag)
        plt.title("Live Segment Magnitude Spectrum")
        plt.xlabel("Hz")
        plt.ylabel("|X(f)|")
        plt.xlim(0, min(8000, freqs[-1]))
        plt.tight_layout()
        plt.show()

        labels = [r[0] for r in results[:top_n]]
        sims_plot = [r[1] for r in results[:top_n]]

        plt.figure(figsize=(12, 4))
        plt.bar(labels, sims_plot)
        plt.title(f"Top {top_n} Class Scores (mono, band-template)  mode={score_mode}")
        plt.xlabel("Class label (note_string)")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.show()

    return best_label, best_score


# -----------------------------
# Match (polyphonic)
# -----------------------------
def quick_note_prune_band(entries: List[Dict[str, Any]], live_band: np.ndarray, prune_top: int) -> List[str]:
    scored: List[Tuple[str, float]] = []
    for e in entries:
        label = str(e.get("note", "?"))
        takes = e.get("takes", [])
        best = -1e9
        for t in takes:
            if "band_template" not in t:
                continue
            tb = np.array(t["band_template"], dtype=np.float64)
            best = max(best, cosine_similarity(live_band, tb))
        scored.append((label, float(best)))

    scored.sort(key=lambda x: x[1], reverse=True)
    keep = [lab for lab, _ in scored[:max(1, int(prune_top))]]
    return keep


def solve_nonneg_omp(A: np.ndarray,
                     y: np.ndarray,
                     max_notes: int,
                     min_corr: float = 1e-3,
                     min_improve: float = 1e-4) -> np.ndarray:
    _m, n = A.shape
    selected: List[int] = []
    r = y.copy()
    prev_norm = float(np.linalg.norm(r))
    x_full = np.zeros(n, dtype=np.float64)

    for _ in range(int(max_notes)):
        scores = A.T @ r
        j = int(np.argmax(scores))
        best = float(scores[j])

        if best < float(min_corr):
            break

        if j not in selected:
            selected.append(j)

        As = A[:, selected]
        x_s, _ = nnls(As, y)
        r_new = y - As @ x_s
        new_norm = float(np.linalg.norm(r_new))

        if (prev_norm - new_norm) < float(min_improve):
            if selected:
                selected.pop()
            break

        r = r_new
        prev_norm = new_norm

    if selected:
        As = A[:, selected]
        x_s, _ = nnls(As, y)
        for idx, val in zip(selected, x_s):
            x_full[idx] = float(val)

    return x_full


def match_poly(
    lut_path: str,
    wav_path: str,
    start: float,
    dur: float,
    algo: str,
    window: Optional[str],
    a4_override: Optional[float],
    use_highpass: bool,
    highpass_hz: float,
    max_notes: int,
    thresh: float,
    prune_top: int,
    plot: bool,
    show_collapsed: bool,
):
    lut = load_lut(lut_path)
    entries = lut.get("entries", [])
    if not entries:
        raise ValueError("LUT has no entries.")

    fs, x = read_wav_mono_float(wav_path)
    x = normalize_peak(x)

    seg = pick_segment(x, fs, start, dur)
    if use_highpass:
        seg = highpass_filter(seg, fs, cutoff_hz=highpass_hz)
    seg = normalize_peak(seg)

    # kept for LUT meta consistency
    a4_lut = float(lut.get("meta", {}).get("a4_hz", 440.0))
    _a4 = float(a4_override) if a4_override is not None else a4_lut
    _ = _a4

    win = window or str(entries[0].get("window", "hann"))

    live_band = build_live_band_from_lut(seg, fs, lut.get("meta", {}), window=win)
    y = live_band / (np.linalg.norm(live_band) + 1e-12)

    keep_labels = quick_note_prune_band(entries, live_band, prune_top=prune_top)
    keep_set = set(keep_labels)

    cols: List[np.ndarray] = []
    col_label: List[str] = []

    for e in entries:
        label = str(e.get("note", "?"))
        if label not in keep_set:
            continue
        takes = e.get("takes", [])
        for t in takes:
            tb = t.get("band_template", None)
            if tb is None:
                continue
            templ = np.array(tb, dtype=np.float64)
            nrm = float(np.linalg.norm(templ))
            if nrm < 1e-9:
                continue
            cols.append(templ / (nrm + 1e-12))
            col_label.append(label)

    if not cols:
        print("No band templates in LUT. Rebuild with this script.")
        return []

    A = np.column_stack(cols)

    algo_l = (algo or "nnls").lower()
    if algo_l == "nnls":
        xw, _ = nnls(A, y)
    elif algo_l == "omp":
        xw = solve_nonneg_omp(A, y, max_notes=max_notes)
    else:
        raise ValueError(f"Unknown algo '{algo}'. Use nnls|omp.")

    # Collapse template weights -> label strength (max over takes)
    label_strength: Dict[str, float] = {}
    for w, lab in zip(xw, col_label):
        label_strength[lab] = max(label_strength.get(lab, 0.0), float(w))

    if not label_strength:
        print("No note strengths found.")
        return []

    # Normalize to [0,1]
    m = max(label_strength.values()) + 1e-12
    items = sorted(((lab, s / m) for lab, s in label_strength.items()), key=lambda p: p[1], reverse=True)

    out: List[Tuple[str, float]] = []
    for lab, s in items:
        if len(out) >= int(max_notes):
            break
        if float(s) < float(thresh):
            break
        out.append((lab, float(s)))

    print("\nDetected classes (polyphonic, string-aware):")
    if not out:
        print("  (none above threshold)")
    else:
        for lab, s in out:
            base, sidx = split_label(lab)
            print(f"  {lab:8s} strength={s:.3f}   base={base:4s} string={sidx}")

    if show_collapsed:
        base_strength: Dict[str, float] = {}
        for lab, s in label_strength.items():
            base, _sidx = split_label(lab)
            base_strength[base] = max(base_strength.get(base, 0.0), float(s))
        m2 = max(base_strength.values()) + 1e-12
        base_items = sorted(((b, v / m2) for b, v in base_strength.items()), key=lambda p: p[1], reverse=True)
        print("\nCollapsed base-note strengths (max over strings):")
        for b, v in base_items[:12]:
            print(f"  {b:4s}  strength={v:.3f}")

    if plot:
        freqs, mag = spectrum_mag(seg, fs, window=win)
        plt.figure(figsize=(12, 4))
        plt.plot(freqs, mag)
        plt.title("Live Segment Magnitude Spectrum")
        plt.xlabel("Hz")
        plt.ylabel("|X(f)|")
        plt.xlim(0, min(8000, freqs[-1]))
        plt.tight_layout()
        plt.show()

        top_show = min(12, len(items))
        labels = [lab for lab, _ in items[:top_show]]
        vals = [s for _lab, s in items[:top_show]]

        plt.figure(figsize=(12, 4))
        plt.bar(labels, vals)
        plt.title(f"Top Class Strengths (poly) algo={algo_l} prune={prune_top} thresh={thresh}")
        plt.xlabel("Class label (note_string)")
        plt.ylabel("Strength (normalized)")
        plt.tight_layout()
        plt.show()

    return out


# -----------------------------
# CLI parsing
# -----------------------------
def parse_labeled_args(pairs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for s in pairs:
        if "=" not in s:
            raise ValueError(f"Expected LABEL=path, got '{s}'")
        label, path = s.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Bad LABEL=path: '{s}'")
        # normalize label by re-parsing it
        base, idx = split_label(label)
        out.append((f"{base}_{idx}", path))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build", help="Build LUT from labeled recordings (LABEL=wav_path)")
    ap_build.add_argument("--out", required=True, help="Output LUT json")
    ap_build.add_argument("--k", type=int, default=60, help="Max harmonics K (fingerprint features)")
    ap_build.add_argument("--tol", type=float, default=15.0, help="Tolerance Hz around each harmonic (fingerprint)")
    ap_build.add_argument("--start", type=float, default=0.10, help="Segment start sec")
    ap_build.add_argument("--dur", type=float, default=0.20, help="Segment duration sec")
    ap_build.add_argument("--a4", type=float, default=440.0, help="A4 reference")
    ap_build.add_argument("--window", type=str, default="hann", help="FFT window")
    ap_build.add_argument("--highpass", action="store_true", help="Apply a highpass to reduce rumble")
    ap_build.add_argument("--highpass_hz", type=float, default=80.0, help="Highpass cutoff Hz")
    ap_build.add_argument("--replace_existing", action="store_true",
                          help="Overwrite output file instead of merging/appending by label")
    ap_build.add_argument("labeled", nargs="+", help="Pairs LABEL=wav_path, e.g. E4_5=E4_5.wav")

    # Band template params
    ap_build.add_argument("--band_fmin", type=float, default=40.0, help="Band template min Hz")
    ap_build.add_argument("--band_fmax", type=float, default=8000.0, help="Band template max Hz")
    ap_build.add_argument("--band_n", type=int, default=480, help="Number of log-frequency bands (more = better separation)")
    ap_build.add_argument("--band_agg", type=str, default="max", choices=["mean", "max"],
                          help="Pooling in bands; max preserves narrow resonances (good for string ID)")
    ap_build.add_argument("--band_logmag", action="store_true",
                          help="Use log1p(mag) for band template (recommended)")
    ap_build.add_argument("--mask_k", type=int, default=24, help="Harmonic mask uses up to this many harmonics")
    ap_build.add_argument("--mask_tol", type=float, default=20.0, help="Mask tolerance value")
    ap_build.add_argument("--mask_tol_mode", type=str, default="cents", choices=["hz", "cents"],
                          help="Interpret --mask_tol as hz or cents")

    ap_build_dir = sub.add_parser("build_dir", help="Build LUT from a directory of wavs (label parsed from filename)")
    ap_build_dir.add_argument("--out", required=True, help="Output LUT json")
    ap_build_dir.add_argument("--dir", required=True, help="Directory of wavs (E2.wav, E4_5.wav, etc.)")
    ap_build_dir.add_argument("--recursive", action="store_true", help="Search subfolders too")
    ap_build_dir.add_argument("--k", type=int, default=60, help="Max harmonics K (fingerprint features)")
    ap_build_dir.add_argument("--tol", type=float, default=15.0, help="Tolerance Hz around each harmonic (fingerprint)")
    ap_build_dir.add_argument("--start", type=float, default=0.10, help="Segment start sec")
    ap_build_dir.add_argument("--dur", type=float, default=0.20, help="Segment duration sec")
    ap_build_dir.add_argument("--a4", type=float, default=440.0, help="A4 reference")
    ap_build_dir.add_argument("--window", type=str, default="hann", help="FFT window")
    ap_build_dir.add_argument("--highpass", action="store_true", help="Apply a highpass to reduce rumble")
    ap_build_dir.add_argument("--highpass_hz", type=float, default=80.0, help="Highpass cutoff Hz")
    ap_build_dir.add_argument("--replace_existing", action="store_true",
                              help="Overwrite output file instead of merging/appending by label")

    ap_build_dir.add_argument("--band_fmin", type=float, default=40.0, help="Band template min Hz")
    ap_build_dir.add_argument("--band_fmax", type=float, default=8000.0, help="Band template max Hz")
    ap_build_dir.add_argument("--band_n", type=int, default=480, help="Number of log-frequency bands")
    ap_build_dir.add_argument("--band_agg", type=str, default="max", choices=["mean", "max"],
                              help="Pooling in bands")
    ap_build_dir.add_argument("--band_logmag", action="store_true",
                              help="Use log1p(mag) for band template")
    ap_build_dir.add_argument("--mask_k", type=int, default=24, help="Harmonic mask harmonics")
    ap_build_dir.add_argument("--mask_tol", type=float, default=20.0, help="Mask tolerance")
    ap_build_dir.add_argument("--mask_tol_mode", type=str, default="cents", choices=["hz", "cents"],
                              help="Mask tol mode")

    ap_match = sub.add_parser("match", help="Match a MONOPHONIC recording against LUT (string-aware classes)")
    ap_match.add_argument("--lut", required=True, help="LUT json")
    ap_match.add_argument("--wav", required=True, help="Unknown wav to classify")
    ap_match.add_argument("--start", type=float, default=0.10, help="Segment start sec")
    ap_match.add_argument("--dur", type=float, default=0.20, help="Segment duration sec")
    ap_match.add_argument("--window", type=str, default=None, help="Override FFT window (else LUT default)")
    ap_match.add_argument("--a4", type=float, default=None, help="Override A4 reference")
    ap_match.add_argument("--highpass", action="store_true", help="Apply highpass")
    ap_match.add_argument("--highpass_hz", type=float, default=80.0, help="Highpass cutoff Hz")
    ap_match.add_argument("--top", type=int, default=8, help="Print top N matches")
    ap_match.add_argument("--plot", action="store_true", help="Plot spectrum and top scores")
    ap_match.add_argument("--score_mode", type=str, default="max", choices=["max", "mean", "topk"],
                          help="Collapse multiple takes into a class score")
    ap_match.add_argument("--topk", type=int, default=3, help="If score_mode=topk, average top-k take scores")
    ap_match.add_argument("--offharm_alpha", type=float, default=1.0,
                          help="Penalty weight for off-harmonic ratio mismatch (helps neighbor bleed)")

    ap_poly = sub.add_parser("match_poly", help="Match a POLYPHONIC chord/strum (string-aware classes)")
    ap_poly.add_argument("--lut", required=True, help="LUT json")
    ap_poly.add_argument("--wav", required=True, help="Unknown wav to classify")
    ap_poly.add_argument("--start", type=float, default=0.10, help="Segment start sec")
    ap_poly.add_argument("--dur", type=float, default=0.25, help="Segment duration sec (longer helps poly)")
    ap_poly.add_argument("--algo", type=str, default="omp", choices=["nnls", "omp"],
                         help="Poly solver: omp is sparser and reduces neighbor bleed")
    ap_poly.add_argument("--window", type=str, default=None, help="FFT window (else LUT default)")
    ap_poly.add_argument("--a4", type=float, default=None, help="Override A4 reference")
    ap_poly.add_argument("--highpass", action="store_true", help="Apply highpass")
    ap_poly.add_argument("--highpass_hz", type=float, default=80.0, help="Highpass cutoff Hz")
    ap_poly.add_argument("--max_notes", type=int, default=6, help="Max classes to output (6 strings)")
    ap_poly.add_argument("--thresh", type=float, default=0.25, help="Strength threshold (normalized 0..1)")
    ap_poly.add_argument("--prune", type=int, default=40, help="Keep top-N classes before solve (smaller = less bleed)")
    ap_poly.add_argument("--plot", action="store_true", help="Plot spectrum + top strengths")
    ap_poly.add_argument("--collapsed", action="store_true", help="Also show collapsed base-note strengths")

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
            band_fmin=args.band_fmin,
            band_fmax=args.band_fmax,
            band_n=args.band_n,
            band_agg=args.band_agg,
            band_logmag=args.band_logmag,
            mask_k=args.mask_k,
            mask_tol=args.mask_tol,
            mask_tol_mode=args.mask_tol_mode,
        )
        return

    if args.cmd == "build_dir":
        wavs = gather_wavs_from_dir(args.dir, recursive=args.recursive)
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in {args.dir} (recursive={args.recursive})")

        labeled: List[Tuple[str, str]] = []
        skipped = 0
        for p in wavs:
            try:
                label, _base, _idx = label_from_filename(p)
                labeled.append((label, str(p)))
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
            band_fmin=args.band_fmin,
            band_fmax=args.band_fmax,
            band_n=args.band_n,
            band_agg=args.band_agg,
            band_logmag=args.band_logmag,
            mask_k=args.mask_k,
            mask_tol=args.mask_tol,
            mask_tol_mode=args.mask_tol_mode,
        )
        return

    if args.cmd == "match":
        match_note(
            lut_path=args.lut,
            wav_path=args.wav,
            start=args.start,
            dur=args.dur,
            window_live=args.window,
            a4_override=args.a4,
            use_highpass=args.highpass,
            highpass_hz=args.highpass_hz,
            top_n=args.top,
            plot=args.plot,
            score_mode=args.score_mode,
            topk=args.topk,
            offharm_alpha=args.offharm_alpha,
        )
        return

    if args.cmd == "match_poly":
        match_poly(
            lut_path=args.lut,
            wav_path=args.wav,
            start=args.start,
            dur=args.dur,
            algo=args.algo,
            window=args.window,
            a4_override=args.a4,
            use_highpass=args.highpass,
            highpass_hz=args.highpass_hz,
            max_notes=args.max_notes,
            thresh=args.thresh,
            prune_top=args.prune,
            plot=args.plot,
            show_collapsed=args.collapsed,
        )
        return


if __name__ == "__main__":
    main()