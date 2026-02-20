#!/usr/bin/env python3
# lut_build.py
"""
LUT builder (string-aware) for guitar notes.

Filename convention:
  E2.wav      -> label "E2_0"
  E4_5.wav    -> label "E4_5"
  F#3_2.wav   -> label "F#3_2"
No suffix => _0.

Build commands:
  1) Build from labeled pairs (LABEL=path):
     python lut_build.py build --out lut.json --k 60 --tol 15 --tol_mode cents \
       --auto_onset --post_onset 0.06 --dur 0.18 \
       --frame_ms 46 --hop_ms 12 \
       E4_5=samples/E4_5.wav E4_0=samples/E4.wav

  2) Build from directory (auto label from filename):
     python lut_build.py build_dir --out lut.json --dir samples --k 60 --tol 15 --tol_mode cents \
       --auto_onset --post_onset 0.06 --dur 0.18 --frame_ms 46 --hop_ms 12

What this stores:
- per-take harmonic fingerprint (compat/debug)
- per-take log-frequency band template (main feature)
- harmonic band mask (for optional off-harmonic penalties at match time)
- per-take harmonic/off-harmonic energy summaries (useful for rejection / penalties)
- entry-level aggregated statistics across takes (median + std) for more stable matching

Added features:
1) Ideal templates per note per string (math-based, not from a single take):
   - Fits each band dimension as a smooth function of log(f0) across ALL notes on that string
   - Stored per entry:
       ideal_band_template_median
       ideal_band_template_std  (fit residual std per band)

2) Per-string band importance weights (what bands separate notes best):
   - Fisher-like ratio per band:
       importance = between_note_var / (within_note_var + eps)
   - Stored in LUT:
       lut["string_band_importance"]["strings"][<string_idx>] with:
         importance_raw, top_raw
         importance_residual, top_residual (if ideals exist)
"""

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, butter, sosfilt


# -----------------------------
# Note + label parsing
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

_NOTE_TOKEN_RE = re.compile(r"(?i)(?:^|[^A-Za-z0-9])([A-G])([#b]?)(-?\d+)(?=[^A-Za-z0-9]|$)")
_STRING_SUFFIX_RE = re.compile(r".*?_([0-5])$")


def normalize_note_token(letter: str, accidental: str, octave_str: str) -> str:
    letter = letter.upper()
    accidental = accidental.strip()
    if accidental == "b":
        accidental = "b"
    elif accidental == "#":
        accidental = "#"
    else:
        accidental = ""
    return f"{letter}{accidental}{octave_str}"


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
    E2.wav   -> (E2_0, E2, 0)
    E4_5.wav -> (E4_5, E4, 5)
    """
    stem = path.stem
    m = _NOTE_TOKEN_RE.search(stem)
    if not m:
        raise ValueError(f"Could not parse note from filename '{path.name}'. Expected E2.wav or E4_5.wav etc.")
    base_note = normalize_note_token(m.group(1), m.group(2).lower(), m.group(3))

    string_idx = 0
    ms = _STRING_SUFFIX_RE.match(stem)
    if ms and ms.group(1) is not None:
        string_idx = int(ms.group(1))

    label = f"{base_note}_{string_idx}"
    return label, base_note, string_idx


def note_to_midi(note: str) -> int:
    s = note.strip().replace("♯", "#").replace("♭", "b")
    m = re.match(r"^([A-Ga-g])([#bB]?)(-?\d+)$", s)
    if not m:
        raise ValueError(f"Bad note format '{note}'. Use like E6, F#4, Bb3.")
    letter, accidental, octave_str = m.group(1).upper(), m.group(2), m.group(3)

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
    return int((octave + 1) * 12 + semitone)


def midi_to_freq_hz(midi: int, a4_hz: float = 440.0) -> float:
    return float(a4_hz * (2.0 ** ((midi - 69) / 12.0)))


# -----------------------------
# Audio helpers
# -----------------------------
def read_wav_mono_float(wav_path: str) -> Tuple[int, np.ndarray]:
    fs, x = wavfile.read(wav_path)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype.kind in "iu":
        maxv = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / (maxv + 1e-12)
    else:
        x = x.astype(np.float64)
    return int(fs), x


def normalize_audio(x: np.ndarray, mode: str = "peak", target_rms: float = 0.1) -> np.ndarray:
    """
    mode:
      - peak: divide by max(abs(x))
      - p95 : divide by 95th percentile abs
      - rms : scale so RMS ~= target_rms
      - none: no normalization
    """
    mode = (mode or "peak").lower()
    if mode == "none":
        return x
    if mode == "peak":
        d = float(np.max(np.abs(x)) + 1e-12)
        return x / d
    if mode in ("p95", "pct", "percentile"):
        d = float(np.percentile(np.abs(x), 95) + 1e-12)
        return x / d
    if mode == "rms":
        rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
        return x * (float(target_rms) / rms)
    raise ValueError("normalize_mode must be one of: peak|p95|rms|none")


def highpass_filter(x: np.ndarray, fs: int, cutoff_hz: float = 80.0, order: int = 4) -> np.ndarray:
    if cutoff_hz <= 0:
        return x
    sos = butter(order, cutoff_hz / (fs * 0.5), btype="highpass", output="sos")
    return sosfilt(sos, x)


def pick_segment(x: np.ndarray, fs: int, start_sec: float, dur_sec: float) -> np.ndarray:
    start = int(round(start_sec * fs))
    length = int(round(dur_sec * fs))
    seg = x[start:start + length]
    if len(seg) < length:
        seg = np.pad(seg, (0, length - len(seg)))
    return seg


# -----------------------------
# Onset detection (energy-based)
# -----------------------------
def frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if frame <= 0 or hop <= 0:
        raise ValueError("frame and hop must be > 0")
    n = len(x)
    if n <= frame:
        return np.zeros((1, frame), dtype=np.float64)
    count = 1 + (n - frame) // hop
    out = np.zeros((count, frame), dtype=np.float64)
    for i in range(count):
        s = i * hop
        out[i] = x[s:s + frame]
    return out


def detect_onset_sec_energy(
    x: np.ndarray,
    fs: int,
    frame: int,
    hop: int,
    thresh_ratio: float = 6.0,
    hold_frames: int = 3,
    search_start_sec: float = 0.0,
    search_end_sec: Optional[float] = None,
    noise_sec: float = 0.12,
    noise_percentile: float = 25.0,
    fallback_to_peak: bool = True,
    peak_backtrack_sec: float = 0.03,
) -> float:
    """
    Energy-based onset:
    - Compute per-frame RMS
    - noise_floor = percentile(noise window RMS)
    - onset = first place RMS >= noise_floor*thresh_ratio for hold_frames
    Fallback:
    - if never crosses, pick the max-RMS frame, optionally backtrack a bit
    """
    n = len(x)
    if search_end_sec is None:
        search_end_sec = n / fs

    start_i = int(max(0, round(search_start_sec * fs)))
    end_i = int(min(n, round(search_end_sec * fs)))
    if end_i <= start_i + frame:
        return float(search_start_sec)

    xs = x[start_i:end_i]
    frames = frame_signal(xs, frame=frame, hop=hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)

    # --- robust noise floor from first noise_sec of search region
    nf_frames = int(round((noise_sec * fs) / hop))
    nf_frames = max(8, min(nf_frames, len(rms)))
    noise_slice = rms[:nf_frames]
    noise_floor = float(np.percentile(noise_slice, noise_percentile) + 1e-12)

    thresh = noise_floor * float(thresh_ratio)
    hold_frames = max(1, int(hold_frames))

    for i in range(0, len(rms) - hold_frames + 1):
        if np.all(rms[i:i + hold_frames] >= thresh):
            onset_sample = start_i + i * hop
            return float(onset_sample / fs)

    # --- fallback: if it never crosses threshold, take peak RMS frame
    if fallback_to_peak and len(rms) > 0:
        i_peak = int(np.argmax(rms))
        back = int(round((peak_backtrack_sec * fs) / hop))
        i0 = max(0, i_peak - back)
        onset_sample = start_i + i0 * hop
        return float(onset_sample / fs)

    return float(search_start_sec)


# -----------------------------
# Spectrum + harmonic fingerprint
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
    mask = (freqs >= max(0.0, lo)) & (freqs <= hi)
    return float(np.max(mag[mask])) if np.any(mask) else 0.0


def cents_to_hz_width(center_hz: float, tol_cents: float) -> float:
    r = 2.0 ** (tol_cents / 1200.0)
    return float(center_hz * (r - 1.0))


def get_tol_hz(center_hz: float, tol_value: float, tol_mode: str) -> float:
    tol_mode = (tol_mode or "cents").lower()
    if tol_mode == "hz":
        return float(tol_value)
    if tol_mode == "cents":
        return cents_to_hz_width(center_hz, float(tol_value))
    raise ValueError("tol_mode must be hz|cents")


def fingerprint_for_note_oneframe(
    seg: np.ndarray,
    fs: int,
    f0: float,
    k: int,
    tol_value: float,
    tol_mode: str,
    window: str,
) -> np.ndarray:
    freqs, mag = spectrum_mag(seg, fs, window=window)
    nyq = float(freqs[-1])
    amps: List[float] = []
    for h in range(1, k + 1):
        fh = h * f0
        if fh >= nyq:
            break
        tol_hz = get_tol_hz(fh, tol_value, tol_mode)
        amps.append(band_energy_max(freqs, mag, fh, tol_hz))
    v = np.array(amps, dtype=np.float64)
    s = float(np.sum(v))
    return (v / (s + 1e-12)) if s > 1e-18 else np.zeros_like(v)


# -----------------------------
# Log-band templates
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


def build_harmonic_band_mask(edges: np.ndarray, f0: float, k: int, tol_value: float, tol_mode: str) -> np.ndarray:
    nb = len(edges) - 1
    mask = np.zeros(nb, dtype=np.float64)
    fmax = float(edges[-1])
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        mid = 0.5 * (lo + hi)
        ok = False
        for h in range(1, k + 1):
            fh = h * f0
            if fh >= fmax:
                break
            tol_hz = get_tol_hz(fh, tol_value, tol_mode)
            if abs(mid - fh) <= tol_hz:
                ok = True
                break
        mask[i] = 1.0 if ok else 0.0
    return mask


def aggregate_feature(frames_feat: np.ndarray, agg: str = "median") -> np.ndarray:
    """
    frames_feat: shape (T, D)
    """
    if frames_feat.ndim != 2:
        raise ValueError("frames_feat must be 2D (T, D)")
    agg = (agg or "median").lower()
    if agg == "mean":
        return np.mean(frames_feat, axis=0)
    return np.median(frames_feat, axis=0)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def analyze_segment_multiframe(
    seg: np.ndarray,
    fs: int,
    *,
    window: str,
    edges: np.ndarray,
    band_logmag: bool,
    band_agg: str,
    feat_agg: str,
    f0: float,
    fp_k: int,
    fp_tol: float,
    fp_tol_mode: str,
    frame_size: int,
    hop_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      band_template (unit norm)
      fingerprint (unit sum)
    """
    if len(seg) < frame_size:
        seg = np.pad(seg, (0, frame_size - len(seg)))

    frames = frame_signal(seg, frame=frame_size, hop=hop_size)
    w = get_window(window, frame_size, fftbins=True).astype(np.float64)
    frames_w = frames * w[None, :]

    bands = []
    fps = []
    for i in range(frames_w.shape[0]):
        f, mag = spectrum_mag(frames_w[i], fs, window="boxcar")  # already windowed
        y = mag.astype(np.float64)
        if band_logmag:
            y = np.log1p(y)
        band_i = pool_to_log_bands(f, y, edges, agg=band_agg)
        bands.append(band_i)

        fp_i = fingerprint_for_note_oneframe(
            frames_w[i], fs, f0=f0, k=fp_k, tol_value=fp_tol, tol_mode=fp_tol_mode, window="boxcar"
        )
        fps.append(fp_i)

    bands = np.stack(bands, axis=0) if bands else np.zeros((1, len(edges) - 1), dtype=np.float64)

    if fps:
        maxlen = max(len(v) for v in fps)
        fp_mat = np.zeros((len(fps), maxlen), dtype=np.float64)
        for i, v in enumerate(fps):
            fp_mat[i, :len(v)] = v
    else:
        fp_mat = np.zeros((1, fp_k), dtype=np.float64)

    band_vec = aggregate_feature(bands, agg=feat_agg)
    band_vec = l2_normalize(band_vec)

    fp_vec = aggregate_feature(fp_mat, agg=feat_agg)
    s = float(np.sum(fp_vec))
    fp_vec = (fp_vec / (s + 1e-12)) if s > 1e-18 else np.zeros_like(fp_vec)

    return band_vec, fp_vec


# -----------------------------
# Ideal per-string per-note model (smooth fit across pitch)
# -----------------------------
def _poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.vstack([x**p for p in range(int(degree) + 1)]).T


def _ridge_fit_predict(X: np.ndarray, y: np.ndarray, lam: float, Xq: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    A = X.T @ X + float(lam) * np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return Xq @ w


def build_string_ideals_from_entries(
    entries: List[Dict[str, Any]],
    *,
    degree: int = 4,
    lam: float = 1e-2,
    min_notes: int = 6,
) -> int:
    """
    For each string, fit each band dimension y_d as a smooth function of x=log(f0_hz) across ALL notes on that string.
    Then for each entry on that string, predict its “ideal” band template at its f0.

    Mutates entries in-place:
      - ideal_band_template_median
      - ideal_band_template_std

    Returns number of entries that received an ideal.
    """
    by_string: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        if not isinstance(e, dict):
            continue
        if "band_template_median" not in e or "f0_hz" not in e:
            continue
        try:
            si = int(e.get("string_idx", -1))
        except Exception:
            continue
        if si < 0:
            continue
        by_string[si].append(e)

    wrote = 0
    for si, group in by_string.items():
        if len(group) < int(min_notes):
            continue

        f0 = np.array([float(e["f0_hz"]) for e in group], dtype=np.float64)
        x = np.log(np.maximum(f0, 1e-9))

        Y = np.stack([np.array(e["band_template_median"], dtype=np.float64) for e in group], axis=0)
        N, D = Y.shape

        X = _poly_design(x, degree=int(degree))
        Xq = X  # predict at same x positions (ideal per note entry)

        Yhat = np.zeros_like(Y)
        resid_std = np.zeros(D, dtype=np.float64)

        for d in range(D):
            y = Y[:, d]
            yhat = _ridge_fit_predict(X, y, lam=float(lam), Xq=Xq)
            Yhat[:, d] = yhat
            resid_std[d] = float(np.std(y - yhat))

        for i, e in enumerate(group):
            v = Yhat[i].copy()
            v[v < 0.0] = 0.0
            v = l2_normalize(v)

            e["ideal_band_template_median"] = [float(z) for z in v.tolist()]
            e["ideal_band_template_std"] = [float(z) for z in resid_std.tolist()]
            wrote += 1

    return wrote


# -----------------------------
# String band importance (what bands are most discriminative)
# -----------------------------
def _normalize_nonneg(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v[v < 0.0] = 0.0
    s = float(np.sum(v))
    return v / (s + 1e-12)


def compute_string_band_importance(
    lut: Dict[str, Any],
    *,
    top_n: int = 24,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Computes per-string band importance weights (Fisher-like):
      importance[d] = between_note_var[d] / (within_note_var[d] + eps)

    between_note_var: var over band_template_median across notes on the same string
    within_note_var : mean(band_template_std^2) across notes on the same string

    Also computes "residual" importance if ideal templates exist:
      residual = band_template_median - ideal_band_template_median
    """
    meta_bt = lut.get("meta", {}).get("band_template", {})
    fmin = float(meta_bt.get("fmin_hz", 40.0))
    fmax = float(meta_bt.get("fmax_hz", 8000.0))
    nb = int(meta_bt.get("n_bands", 480))
    edges = make_log_band_edges(fmin, fmax, nb)

    by_string: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in lut.get("entries", []):
        if not isinstance(e, dict):
            continue
        if "string_idx" not in e:
            continue
        if "band_template_median" not in e or "band_template_std" not in e:
            continue
        try:
            si = int(e["string_idx"])
        except Exception:
            continue
        by_string[si].append(e)

    out: Dict[str, Any] = {"strings": {}}

    def top_list(w: np.ndarray) -> List[Dict[str, Any]]:
        idxs = np.argsort(-w)[: int(top_n)]
        items: List[Dict[str, Any]] = []
        for idx in idxs:
            items.append({
                "band": int(idx),
                "lo_hz": float(edges[idx]),
                "hi_hz": float(edges[idx + 1]),
                "weight": float(w[idx]),
            })
        return items

    for si, group in sorted(by_string.items()):
        if len(group) < 2:
            continue

        # (N, D)
        Y = np.stack([np.array(e["band_template_median"], dtype=np.float64) for e in group], axis=0)
        S = np.stack([np.array(e["band_template_std"], dtype=np.float64) for e in group], axis=0)

        if Y.shape[1] != nb or S.shape[1] != nb:
            # If someone changed band_n between runs, skip importance (inconsistent dims).
            continue

        within_var = np.mean(S * S, axis=0)
        between_var = np.var(Y, axis=0)

        imp_raw = _normalize_nonneg(between_var / (within_var + float(eps)))

        has_ideal = all(("ideal_band_template_median" in e) for e in group)
        if has_ideal:
            Yid = np.stack([np.array(e["ideal_band_template_median"], dtype=np.float64) for e in group], axis=0)
            if Yid.shape[1] == nb:
                R = Y - Yid
                between_var_r = np.var(R, axis=0)
                imp_resid = _normalize_nonneg(between_var_r / (within_var + float(eps)))
            else:
                imp_resid = None
        else:
            imp_resid = None

        out["strings"][str(si)] = {
            "string_idx": int(si),
            "n_notes": int(len(group)),
            "n_bands": int(nb),
            "band_edges_hz": [float(x) for x in edges.tolist()],
            "importance_raw": [float(x) for x in imp_raw.tolist()],
            "top_raw": top_list(imp_raw),
            "importance_residual": None if imp_resid is None else [float(x) for x in imp_resid.tolist()],
            "top_residual": None if imp_resid is None else top_list(imp_resid),
        }

    return out


# -----------------------------
# LUT structure
# -----------------------------
@dataclass
class LutEntry:
    note: str          # label, e.g. "E4_5"
    base_note: str     # e.g. "E4"
    string_idx: int    # 0..5
    midi: int
    f0_hz: float
    k: int
    tol_value: float
    tol_mode: str
    window: str

    takes: List[Dict[str, Any]]
    source_wavs: List[str]

    analysis_start_sec: float
    analysis_dur_sec: float
    auto_onset: bool

    band_template_median: List[float]
    band_template_std: List[float]
    fingerprint_median: List[float]
    fingerprint_std: List[float]


def load_lut(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"entries": [], "meta": {}}
    data.setdefault("entries", [])
    data.setdefault("meta", {})
    return data


def save_lut(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def gather_wavs_from_dir(dir_path: str, recursive: bool) -> List[Path]:
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    glob = root.rglob("*.wav") if recursive else root.glob("*.wav")
    return sorted([p for p in glob if p.is_file()])


def parse_labeled_args(pairs: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in pairs:
        if "=" not in s:
            raise ValueError(f"Expected LABEL=path, got '{s}'")
        label, path = s.split("=", 1)
        base, idx = split_label(label.strip())
        out.append((f"{base}_{idx}", path.strip()))
    return out


def build_lut(
    out_path: str,
    labeled: List[Tuple[str, str]],
    k: int,
    tol_value: float,
    tol_mode: str,
    start: float,
    dur: float,
    a4: float,
    window: str,
    use_highpass: bool,
    highpass_hz: float,
    replace_existing: bool,
    # band template params
    band_fmin: float,
    band_fmax: float,
    band_n: int,
    band_pool_agg: str,
    band_logmag: bool,
    # mask params
    mask_k: int,
    mask_tol: float,
    mask_tol_mode: str,
    # multiframe params
    feat_agg: str,
    frame_ms: float,
    hop_ms: float,
    # onset params
    auto_onset: bool,
    onset_thresh_ratio: float,
    onset_hold_frames: int,
    onset_search_start: float,
    onset_search_end: float,
    post_onset: float,
    onset_frame_ms: float,
    onset_hop_ms: float,
    onset_noise_sec: float,
    onset_noise_pct: float,
    onset_fallback_peak: bool,
    onset_peak_backtrack: float,
    # normalization
    normalize_mode: str,
    target_rms: float,
) -> None:
    if (not replace_existing) and Path(out_path).exists():
        lut = load_lut(out_path)
    else:
        lut = {"meta": {}, "entries": []}

    lut["meta"]["type"] = "harmonic_fingerprint_lut_v3_multiframe_onset_stringaware"
    lut["meta"]["a4_hz"] = float(a4)
    lut["meta"]["analysis"] = {
        "window": str(window),
        "dur_sec": float(dur),
        "start_sec": float(start),
        "auto_onset": bool(auto_onset),
        "post_onset_sec": float(post_onset),
        "onset": {
            "frame_ms": float(frame_ms),
            "hop_ms": float(hop_ms),
            "thresh_ratio": float(onset_thresh_ratio),
            "hold_frames": int(onset_hold_frames),
            "search_start_sec": float(onset_search_start),
            "search_end_sec": float(onset_search_end),
        },
        "normalize": {"mode": str(normalize_mode), "target_rms": float(target_rms)},
        "highpass": {"enabled": bool(use_highpass), "hz": float(highpass_hz)},
        "multiframe": {"feature_agg": str(feat_agg)},
        "fingerprint": {"k": int(k), "tol": float(tol_value), "tol_mode": str(tol_mode)},
    }
    lut["meta"]["band_template"] = {
        "enabled": True,
        "fmin_hz": float(band_fmin),
        "fmax_hz": float(band_fmax),
        "n_bands": int(band_n),
        "pool_agg": str(band_pool_agg),
        "logmag": bool(band_logmag),
        "harm_mask": {"k": int(mask_k), "tol": float(mask_tol), "tol_mode": str(mask_tol_mode)},
    }

    edges = make_log_band_edges(band_fmin, band_fmax, band_n)

    existing_idx: Dict[str, int] = {}
    for i, e in enumerate(lut.get("entries", [])):
        if isinstance(e, dict) and "note" in e:
            existing_idx[str(e["note"]).strip().upper()] = i

    grouped: Dict[str, List[str]] = defaultdict(list)
    for label, wav_path in labeled:
        grouped[label.strip().upper()].append(wav_path)

    for key, paths in grouped.items():
        label = key
        base_note, string_idx = split_label(label)

        midi = note_to_midi(base_note)
        f0 = midi_to_freq_hz(midi, a4_hz=a4)

        takes: List[Dict[str, Any]] = []
        sources: List[str] = []
        band_take_list: List[np.ndarray] = []
        fp_take_list: List[np.ndarray] = []

        harm_mask = build_harmonic_band_mask(edges, f0=f0, k=mask_k, tol_value=mask_tol, tol_mode=mask_tol_mode)

        for wav_path in paths:
            fs, x = read_wav_mono_float(wav_path)

            # normalize early for more stable onset detection
            x = normalize_audio(x, mode=normalize_mode, target_rms=target_rms)

            if use_highpass:
                x = highpass_filter(x, fs, cutoff_hz=highpass_hz)
                x = normalize_audio(x, mode=normalize_mode, target_rms=target_rms)

            # compute frame/hop in samples for this file
            frame_size = max(256, int(round((frame_ms / 1000.0) * fs)))
            hop_size = max(64, int(round((hop_ms / 1000.0) * fs)))

            onset_frame_size = max(128, int(round((onset_frame_ms / 1000.0) * fs)))
            onset_hop_size   = max(32,  int(round((onset_hop_ms   / 1000.0) * fs)))

            onset_sec = detect_onset_sec_energy(
                x, fs,
                frame=onset_frame_size,
                hop=onset_hop_size,
                thresh_ratio=onset_thresh_ratio,
                hold_frames=onset_hold_frames,
                search_start_sec=onset_search_start,
                search_end_sec=onset_search_end if onset_search_end > 0 else None,
                noise_sec=onset_noise_sec,
                noise_percentile=onset_noise_pct,
                fallback_to_peak=onset_fallback_peak,
                peak_backtrack_sec=onset_peak_backtrack,
            )
            seg_start = onset_sec + float(post_onset)
            if auto_onset:
                onset_sec = detect_onset_sec_energy(
                    x, fs,
                    frame=frame_size,
                    hop=hop_size,
                    thresh_ratio=onset_thresh_ratio,
                    hold_frames=onset_hold_frames,
                    search_start_sec=onset_search_start,
                    search_end_sec=onset_search_end if onset_search_end > 0 else None,
                )
                seg_start = onset_sec + float(post_onset)
            else:
                onset_sec = float("nan")
                seg_start = float(start)
            max_start = max(0.0, (len(x) / fs) - float(dur) - 1e-3)
            seg_start = float(min(seg_start, max_start))
            
            seg = pick_segment(x, fs, seg_start, float(dur))
            seg = normalize_audio(seg, mode=normalize_mode, target_rms=target_rms)

            band_vec, fp_vec = analyze_segment_multiframe(
                seg=seg, fs=fs,
                window=window,
                edges=edges,
                band_logmag=band_logmag,
                band_agg=band_pool_agg,
                feat_agg=feat_agg,
                f0=f0,
                fp_k=k,
                fp_tol=tol_value,
                fp_tol_mode=tol_mode,
                frame_size=frame_size,
                hop_size=hop_size,
            )

            harm_energy = float(np.sum(band_vec * harm_mask))
            off_energy = float(np.sum(band_vec * (1.0 - harm_mask)))
            harm_ratio = float(harm_energy / (off_energy + 1e-12))

            takes.append({
                "fingerprint": [float(v) for v in fp_vec.tolist()],
                "band_template": [float(v) for v in band_vec.tolist()],
                "harm_band_mask": [float(v) for v in harm_mask.tolist()],
                "harm_energy": harm_energy,
                "off_energy": off_energy,
                "harm_ratio": harm_ratio,
                "onset_sec": None if not np.isfinite(onset_sec) else float(onset_sec),
                "segment_start_sec": float(seg_start),
                "segment_dur_sec": float(dur),
                "frame_size": int(frame_size),
                "hop_size": int(hop_size),
                "fs": int(fs),
            })
            sources.append(str(wav_path))

            band_take_list.append(band_vec)
            fp_take_list.append(fp_vec)

        # aggregate across takes
        if band_take_list:
            band_mat = np.stack(band_take_list, axis=0)
            band_med = np.median(band_mat, axis=0)
            band_std = np.std(band_mat, axis=0)
            band_med = l2_normalize(band_med)
        else:
            band_med = np.zeros(len(edges) - 1, dtype=np.float64)
            band_std = np.zeros(len(edges) - 1, dtype=np.float64)

        if fp_take_list:
            maxlen = max(len(v) for v in fp_take_list)
            fp_mat = np.zeros((len(fp_take_list), maxlen), dtype=np.float64)
            for i, v in enumerate(fp_take_list):
                fp_mat[i, :len(v)] = v
            fp_med = np.median(fp_mat, axis=0)
            fp_std = np.std(fp_mat, axis=0)
            s = float(np.sum(fp_med))
            fp_med = (fp_med / (s + 1e-12)) if s > 1e-18 else np.zeros_like(fp_med)
        else:
            fp_med = np.zeros(k, dtype=np.float64)
            fp_std = np.zeros(k, dtype=np.float64)

        entry = LutEntry(
            note=f"{base_note}_{string_idx}",
            base_note=base_note,
            string_idx=int(string_idx),
            midi=midi,
            f0_hz=float(f0),
            k=int(k),
            tol_value=float(tol_value),
            tol_mode=str(tol_mode),
            window=str(window),
            takes=takes,
            source_wavs=sources,
            analysis_start_sec=float(start),
            analysis_dur_sec=float(dur),
            auto_onset=bool(auto_onset),
            band_template_median=[float(v) for v in band_med.tolist()],
            band_template_std=[float(v) for v in band_std.tolist()],
            fingerprint_median=[float(v) for v in fp_med.tolist()],
            fingerprint_std=[float(v) for v in fp_std.tolist()],
        ).__dict__

        if key in existing_idx and not replace_existing:
            old = lut["entries"][existing_idx[key]]
            old.setdefault("takes", [])
            old.setdefault("source_wavs", [])
            old["takes"] = list(old["takes"]) + takes
            old["source_wavs"] = list(old["source_wavs"]) + sources

            # refresh meta fields
            for f in [
                "note", "base_note", "string_idx", "midi", "f0_hz", "k",
                "tol_value", "tol_mode", "window",
                "analysis_start_sec", "analysis_dur_sec", "auto_onset",
                "band_template_median", "band_template_std",
                "fingerprint_median", "fingerprint_std",
            ]:
                old[f] = entry[f]

            lut["entries"][existing_idx[key]] = old
            print(f"[build] merged  {entry['note']}: added_takes={len(sources)} total_takes={len(old['takes'])}")
        elif key in existing_idx and replace_existing:
            lut["entries"][existing_idx[key]] = entry
            print(f"[build] replaced {entry['note']}: takes={len(sources)}")
        else:
            lut["entries"].append(entry)
            print(f"[build] added   {entry['note']}: takes={len(sources)}")

    # --- NEW 1: build ideal templates per entry (math model per string)
    wrote_ideals = build_string_ideals_from_entries(
        lut["entries"],
        degree=4,
        lam=1e-2,
        min_notes=6,
    )
    lut["meta"].setdefault("ideals_by_string", {})
    lut["meta"]["ideals_by_string"] = {
        "method": "poly_ridge_fit_per_band_dim",
        "x": "log(f0_hz)",
        "degree": 4,
        "lambda": 1e-2,
        "min_notes": 6,
        "stored_fields": ["ideal_band_template_median", "ideal_band_template_std"],
        "entries_written": int(wrote_ideals),
    }

    # --- NEW 2: compute per-string band importance masks
    lut["string_band_importance"] = compute_string_band_importance(lut, top_n=24)
    lut["meta"].setdefault("string_band_importance", {})
    lut["meta"]["string_band_importance"] = {
        "method": "fisher_ratio_between_over_within",
        "between": "var_across_notes_on_same_string (band_template_median)",
        "within": "mean(band_template_std^2) across notes",
        "stored_at": "lut['string_band_importance']",
        "notes": "importance_residual uses band_template_median - ideal_band_template_median when available.",
    }

    save_lut(out_path, lut)
    print(f"Saved LUT -> {out_path} with {len(lut['entries'])} class entries")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Shared defaults tuned for string separation / live robustness
    def add_common(apx: argparse.ArgumentParser) -> None:
        apx.add_argument("--out", required=True)

        # fingerprint params
        apx.add_argument("--k", type=int, default=60)
        apx.add_argument("--tol", type=float, default=20.0)
        apx.add_argument("--tol_mode", type=str, default="cents", choices=["hz", "cents"])

        # segment params (used if not auto-onset; dur always used)
        apx.add_argument("--start", type=float, default=0.10)
        apx.add_argument("--dur", type=float, default=0.20)

        apx.add_argument("--a4", type=float, default=440.0)
        apx.add_argument("--window", type=str, default="hann")

        apx.add_argument("--highpass", action="store_true")
        apx.add_argument("--highpass_hz", type=float, default=80.0)

        apx.add_argument("--replace_existing", action="store_true")

        # band template params
        apx.add_argument("--band_fmin", type=float, default=40.0)
        apx.add_argument("--band_fmax", type=float, default=8000.0)
        apx.add_argument("--band_n", type=int, default=480)
        apx.add_argument("--band_pool_agg", type=str, default="max", choices=["max", "mean"])
        apx.add_argument("--band_logmag", action="store_true")
        apx.add_argument("--mask_k", type=int, default=24)
        apx.add_argument("--mask_tol", type=float, default=20.0)
        apx.add_argument("--mask_tol_mode", type=str, default="cents", choices=["hz", "cents"])

        # multiframe analysis params
        apx.add_argument("--feat_agg", type=str, default="median", choices=["median", "mean"])
        apx.add_argument("--frame_ms", type=float, default=46.0, help="analysis frame length in ms")
        apx.add_argument("--hop_ms", type=float, default=12.0, help="analysis hop in ms")

        # onset detection params
        apx.add_argument("--auto_onset", action="store_true", help="detect onset and analyze after pick attack")
        apx.add_argument("--onset_thresh_ratio", type=float, default=6.0)
        apx.add_argument("--onset_hold_frames", type=int, default=3)
        apx.add_argument("--onset_search_start", type=float, default=0.0)
        apx.add_argument("--onset_search_end", type=float, default=0.0, help="0 = end of file")
        apx.add_argument("--post_onset", type=float, default=0.06, help="seconds after onset to start analysis")

        apx.add_argument("--onset_frame_ms", type=float, default=10.0, help="onset detection frame length (ms)")
        apx.add_argument("--onset_hop_ms", type=float, default=3.0, help="onset detection hop (ms)")
        apx.add_argument("--onset_noise_sec", type=float, default=0.12, help="seconds used to estimate noise floor")
        apx.add_argument("--onset_noise_pct", type=float, default=25.0, help="percentile for noise floor (lower = safer)")
        apx.add_argument("--onset_fallback_peak", action="store_true", help="if threshold never crossed, fallback to peak-RMS frame")
        apx.add_argument("--onset_peak_backtrack", type=float, default=0.03, help="seconds to backtrack from peak-RMS fallback")

        # normalization
        apx.add_argument("--normalize", type=str, default="p95", choices=["peak", "p95", "rms", "none"])
        apx.add_argument("--target_rms", type=float, default=0.10)

    ap_build = sub.add_parser("build", help="Build LUT from labeled recordings (LABEL=wav_path)")
    add_common(ap_build)
    ap_build.add_argument("labeled", nargs="+")  # LABEL=path

    ap_build_dir = sub.add_parser("build_dir", help="Build LUT from directory (auto label from filename)")
    add_common(ap_build_dir)
    ap_build_dir.add_argument("--dir", required=True)
    ap_build_dir.add_argument("--recursive", action="store_true")

    args = ap.parse_args()

    if args.cmd == "build":
        labeled = parse_labeled_args(args.labeled)
        build_lut(
            out_path=args.out,
            labeled=labeled,
            k=args.k,
            tol_value=args.tol,
            tol_mode=args.tol_mode,
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
            band_pool_agg=args.band_pool_agg,
            band_logmag=args.band_logmag,
            mask_k=args.mask_k,
            mask_tol=args.mask_tol,
            mask_tol_mode=args.mask_tol_mode,
            feat_agg=args.feat_agg,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            auto_onset=args.auto_onset,
            onset_thresh_ratio=args.onset_thresh_ratio,
            onset_hold_frames=args.onset_hold_frames,
            onset_search_start=args.onset_search_start,
            onset_search_end=args.onset_search_end,
            post_onset=args.post_onset,
            normalize_mode=args.normalize,
            target_rms=args.target_rms,
        )
        return

    if args.cmd == "build_dir":
        wavs = gather_wavs_from_dir(args.dir, recursive=args.recursive)
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in {args.dir}")

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
            raise RuntimeError("No wav files had parseable labels in their filename.")

        print(f"Found {len(labeled)} labeled wavs (skipped {skipped}).")

        build_lut(
            out_path=args.out,
            labeled=labeled,
            k=args.k,
            tol_value=args.tol,
            tol_mode=args.tol_mode,
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
            band_pool_agg=args.band_pool_agg,
            band_logmag=args.band_logmag,
            mask_k=args.mask_k,
            mask_tol=args.mask_tol,
            mask_tol_mode=args.mask_tol_mode,
            feat_agg=args.feat_agg,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            auto_onset=args.auto_onset,
            onset_thresh_ratio=args.onset_thresh_ratio,
            onset_hold_frames=args.onset_hold_frames,
            onset_search_start=args.onset_search_start,
            onset_search_end=args.onset_search_end,
            post_onset=args.post_onset,
            normalize_mode=args.normalize,
            target_rms=args.target_rms,
        )
        return


if __name__ == "__main__":
    main()