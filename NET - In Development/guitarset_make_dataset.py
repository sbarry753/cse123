#!/usr/bin/env python3
r"""
guitarset_make_fb_dataset.py  (v3 - GuitarSet HEX DI + robust IDMT + better splits + alignment + streaming shards)

Builds a framewise guitar note detection dataset from:
  - GuitarSet (via mirdata)  ✅ now supports HEX pickup DI (6-ch) and synthesizes 0–6 polyphony by mixing string subsets
  - IDMT-SMT-Guitar          ✅ more robust wav/xml pairing + XML parsing
  - Extra single-note DI folder (C4_2.wav style naming)

Labels per frame:
  - active[49] : note sounding (multi-label sigmoid target)
  - onset[49]  : note attack (multi-label sigmoid target)
  - count[1]   : polyphonic note count, clipped 0-6 (CE target, int)

Features:
  STFT power -> log-spaced band pooling -> log1p -> per-band zscore

Main improvements vs your v2:
  - Deterministic shuffle splits (no "first N tracks" leakage risk)
  - GuitarSet: supports --gs_source hex (recommended) and generates multiple random mixes per track
  - GuitarSet: if notes_string0..notes_string5 exist, labels match the chosen string subset (less label noise)
  - Annotation alignment fix: shift label times by n_fft/2 to match STFT frame energy center (center=False)
  - Robust IDMT wav/xml pairing via indexed stems
  - More robust IDMT pitch parsing (MIDI, pitchname, or Hz)
  - Optional DC blocker preprocessing (helps DI realism)
  - Safer shard streaming: flush when buffer hits shard_size or memory budget

Install:
  pip install numpy librosa mirdata soundfile scipy lxml

Example:
python guitarset_make_fb_dataset.py \
  --out_dir gs_dataset \
  --download \
  --gs_source hex \
  --gs_mixes_per_track 3 \
  --allow_zero_poly \
  --extra_dir ../samples \
  --idmt_dir "C:\Users\...\IDMT-SMT-GUITAR_V2"

Notes:
  - For Daisy Seed DI inference, use --gs_source hex (best domain match).
  - IDMT should point at root containing dataset1/dataset2/dataset3.
"""

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import librosa
import mirdata
import soundfile as sf

from scipy.io import wavfile


# ─────────────────────────────────────────────────
# Label space
# ─────────────────────────────────────────────────

MIDI_MIN = 40   # E2
MIDI_MAX = 88   # E6
N_NOTES  = MIDI_MAX - MIDI_MIN + 1   # 49
MAX_POLY = 6    # clamp polyphony count to this

_NOTE_TO_SEMI = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11,
}

_NOTE_RE = re.compile(r"(?i)^([A-G])([#b]?)(-?\d+)(?:_([0-5]))?$")


# ─────────────────────────────────────────────────
# DSP helpers
# ─────────────────────────────────────────────────

def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def dc_block(x: np.ndarray, r: float = 0.995) -> np.ndarray:
    """Simple DC blocker (1st order highpass-ish). Helps DI realism."""
    if x.size == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float32)
    xm1 = 0.0
    ym1 = 0.0
    rr = float(r)
    for i in range(x.size):
        xi = float(x[i])
        yi = xi - xm1 + rr * ym1
        y[i] = yi
        xm1 = xi
        ym1 = yi
    return y


def safe_peak_norm(x: np.ndarray, target: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(x)) + 1e-12)
    return (target * x / peak).astype(np.float32)


# ─────────────────────────────────────────────────
# Labels helpers
# ─────────────────────────────────────────────────

def note_data_to_active_labels(note_data, n_frames: int, hop_sec: float, frame_time_offset: float) -> np.ndarray:
    """
    Convert mirdata note_data (intervals,pitches) to framewise active labels.
    frame_time_offset shifts label times by ~n_fft/2/sr to align with STFT energy center.
    """
    y = np.zeros((n_frames, N_NOTES), dtype=np.uint8)
    if note_data is None:
        return y
    intervals = np.asarray(note_data.intervals, dtype=np.float64)
    pitches   = np.asarray(note_data.pitches,   dtype=np.float64)
    for (t0, t1), midi in zip(intervals, pitches):
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue
        m = int(round(midi))
        if m < MIDI_MIN or m > MIDI_MAX:
            continue
        t0s = float(t0) + frame_time_offset
        t1s = float(t1) + frame_time_offset
        i0 = int(math.floor(t0s / hop_sec))
        i1 = int(math.ceil(t1s  / hop_sec))
        i0 = max(0, min(n_frames, i0))
        i1 = max(0, min(n_frames, i1))
        if i1 > i0:
            y[i0:i1, m - MIDI_MIN] = 1
    return y


def active_to_onset_labels(active: np.ndarray, onset_width_frames: int = 2) -> np.ndarray:
    T, D = active.shape
    onset = np.zeros((T, D), dtype=np.uint8)
    prev  = np.zeros((D,), dtype=np.uint8)
    for t in range(T):
        cur = active[t]
        rising = (cur == 1) & (prev == 0)
        if np.any(rising):
            t1 = min(T, t + int(onset_width_frames))
            onset[t:t1, rising] = 1
        prev = cur
    return onset


def active_to_count_labels(active: np.ndarray) -> np.ndarray:
    """
    Returns int8 array shape (T,) with polyphony count, clamped to [0, MAX_POLY].
    Adds light temporal smoothing to reduce boundary flicker.
    """
    a = active.astype(np.uint8)
    if a.shape[0] >= 3:
        mid = ((a[:-2] + a[1:-1] + a[2:]) >= 2).astype(np.uint8)
        a_sm = np.zeros_like(a, dtype=np.uint8)
        a_sm[1:-1] = mid
        a_sm[0] = a[0]
        a_sm[-1] = a[-1]
    else:
        a_sm = a
    counts = a_sm.sum(axis=1).astype(np.int8)
    return np.clip(counts, 0, MAX_POLY)


# ─────────────────────────────────────────────────
# IDMT-SMT-Guitar annotation parser
# ─────────────────────────────────────────────────

def _pitchname_to_midi(name: str) -> int:
    """Convert pitchname like 'E4', 'F#3', 'Bb2' to MIDI number. Returns -1 on failure."""
    m = re.match(r"(?i)^([A-G])([#b]?)(-?\d+)$", name.strip())
    if not m:
        return -1
    letter = m.group(1).upper()
    acc    = m.group(2)
    octave = int(m.group(3))
    if acc in ("b", "B"):
        pc = letter + "B"
    elif acc == "#":
        pc = letter + "#"
    else:
        pc = letter
    semi = _NOTE_TO_SEMI.get(pc, -1)
    if semi < 0:
        return -1
    return (octave + 1) * 12 + semi


def parse_idmt_xml(xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse an IDMT-SMT-Guitar XML annotation file.
    Returns (intervals, midis) where intervals is (N,2) float64 and midis is (N,) float64.
    Handles pitch as:
      - <pitch> MIDI int
      - <pitchname> like E4
      - <pitch> numeric Hz (fallback)
    Handles time as:
      - onsetSec/offsetSec
      - onsetSample/offsetSample (+ sampleRate)
      - onsetSample + durationSample/durationSec
    """
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        print(f"[idmt] XML parse error {xml_path.name}: {e}")
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    root = tree.getroot()
    intervals: List[List[float]] = []
    midis: List[float] = []

    sr_el = root.find(".//sampleRate")
    sr = float(sr_el.text) if (sr_el is not None and sr_el.text) else 44100.0

    for event in root.iter("event"):
        # --- time ---
        onset_el  = event.find("onsetSec")
        offset_el = event.find("offsetSec")

        if onset_el is not None and onset_el.text:
            t0 = float(onset_el.text)
            if offset_el is not None and offset_el.text:
                t1 = float(offset_el.text)
            else:
                # durationSec fallback
                dur_sec = event.find("durationSec")
                if dur_sec is not None and dur_sec.text:
                    t1 = t0 + float(dur_sec.text)
                else:
                    t1 = t0 + 0.25
        else:
            onset_samp = event.find("onsetSample")
            offset_samp = event.find("offsetSample")
            if onset_samp is None or (onset_samp.text is None):
                continue
            t0 = float(onset_samp.text) / sr
            if offset_samp is not None and offset_samp.text:
                t1 = float(offset_samp.text) / sr
            else:
                dur_samp = event.find("durationSample")
                dur_sec  = event.find("durationSec")
                if dur_samp is not None and dur_samp.text:
                    t1 = t0 + float(dur_samp.text) / sr
                elif dur_sec is not None and dur_sec.text:
                    t1 = t0 + float(dur_sec.text)
                else:
                    t1 = t0 + 0.25

        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue

        # --- pitch ---
        midi = -1
        pitch_el = event.find("pitch")
        if pitch_el is not None and pitch_el.text:
            text = pitch_el.text.strip()
            # integer midi?
            if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
                midi = int(text)
            else:
                # pitchname?
                midi = _pitchname_to_midi(text)
                if midi < 0:
                    # Hz fallback
                    try:
                        hz = float(text)
                        midi = int(round(69 + 12 * math.log2(hz / 440.0)))
                    except Exception:
                        midi = -1
        else:
            pn_el = event.find("pitchname")
            if pn_el is not None and pn_el.text:
                midi = _pitchname_to_midi(pn_el.text.strip())

        if midi < MIDI_MIN or midi > MIDI_MAX:
            continue

        intervals.append([t0, t1])
        midis.append(float(midi))

    if not intervals:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.array(intervals, dtype=np.float64), np.array(midis, dtype=np.float64)


def build_idmt_active_from_arrays(
    intervals: np.ndarray,
    midis: np.ndarray,
    n_frames: int,
    hop_sec: float,
    frame_time_offset: float,
) -> np.ndarray:
    active = np.zeros((n_frames, N_NOTES), dtype=np.uint8)
    for (t0, t1), midi in zip(intervals, midis):
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue
        m = int(round(midi))
        if m < MIDI_MIN or m > MIDI_MAX:
            continue
        t0s = float(t0) + frame_time_offset
        t1s = float(t1) + frame_time_offset
        i0 = max(0, min(n_frames, int(math.floor(t0s / hop_sec))))
        i1 = max(0, min(n_frames, int(math.ceil(t1s  / hop_sec))))
        if i1 > i0:
            active[i0:i1, m - MIDI_MIN] = 1
    return active


def discover_idmt_pairs(idmt_root: Path) -> List[Tuple[Path, Path]]:
    """
    Robustly find (wav, xml) pairs by indexing all XML stems once.
    """
    wavs = list(idmt_root.rglob("*.wav")) + list(idmt_root.rglob("*.WAV"))
    xmls = list(idmt_root.rglob("*.xml")) + list(idmt_root.rglob("*.XML"))

    xml_map: Dict[str, List[Path]] = {}
    for x in xmls:
        xml_map.setdefault(x.stem.lower(), []).append(x)

    pairs: List[Tuple[Path, Path]] = []
    for w in wavs:
        cands = xml_map.get(w.stem.lower())
        if not cands:
            continue
        # prefer "closest" candidate by path length (heuristic)
        cands_sorted = sorted(cands, key=lambda p: len(p.parts))
        pairs.append((w, cands_sorted[0]))

    print(f"[idmt] discovered {len(pairs)} annotated pairs under {idmt_root}")
    return pairs


# ─────────────────────────────────────────────────
# GuitarSet HEX helpers (DI + 0–6 poly mixing)
# ─────────────────────────────────────────────────

def load_guitarset_hex6(tr, sr_target: int) -> np.ndarray:
    """
    Load GuitarSet hex pickup audio as (T,6) float32 at sr_target.

    mirdata exposes:
      - tr.audio_hex_path (str) and tr.audio_hex -> (audio, sr)
      - optional: tr.audio_hex_cln_path / tr.audio_hex_cln
    """
    # Prefer cleaned if available
    if hasattr(tr, "audio_hex_cln") and tr.audio_hex_cln is not None:
        y, sr = tr.audio_hex_cln
    elif hasattr(tr, "audio_hex") and tr.audio_hex is not None:
        y, sr = tr.audio_hex
    else:
        # as a last resort, try to load from path fields
        path = None
        if hasattr(tr, "audio_hex_cln_path") and tr.audio_hex_cln_path:
            path = tr.audio_hex_cln_path
        elif hasattr(tr, "audio_hex_path") and tr.audio_hex_path:
            path = tr.audio_hex_path
        if path is None:
            raise RuntimeError("No hex pickup audio found (audio_hex/audio_hex_path missing).")
        y, sr = librosa.load(path, sr=None, mono=False)

    y = np.asarray(y)

    # mirdata returns multitrack as (6, T) in many setups (mono=False)
    if y.ndim == 2 and y.shape[0] == 6 and y.shape[1] != 6:
        y = y.T  # -> (T, 6)

    if y.ndim != 2 or y.shape[1] != 6:
        raise RuntimeError(f"Expected hex audio shape (T,6), got {y.shape}")

    y = y.astype(np.float32)

    if int(sr) != int(sr_target):
        y = np.stack(
            [librosa.resample(y[:, c], orig_sr=int(sr), target_sr=int(sr_target)).astype(np.float32)
             for c in range(6)],
            axis=1
        )

    return y

def choose_string_subset(rng: np.random.Generator, allow_zero: bool) -> List[int]:
    kmin = 0 if allow_zero else 1
    k = int(rng.integers(kmin, 7))
    if k == 0:
        return []
    return sorted(rng.choice(6, size=k, replace=False).tolist())


def mix_strings(hex6: np.ndarray, active_strings: List[int]) -> np.ndarray:
    if len(active_strings) == 0:
        return np.zeros((hex6.shape[0],), dtype=np.float32)
    y = np.sum(hex6[:, active_strings], axis=1).astype(np.float32)
    return safe_peak_norm(y, target=0.9)


def guitarset_active_for_strings(tr, strings: List[int], T: int, hop_sec: float, frame_time_offset: float) -> np.ndarray:
    """
    If mirdata exposes per-string note annotations (notes_string0..notes_string5),
    build labels only from active strings. Otherwise return merged notes_all.
    """
    if len(strings) == 0:
        return np.zeros((T, N_NOTES), dtype=np.uint8)

    if all(hasattr(tr, f"notes_string{s}") for s in range(6)):
        parts = []
        for s in strings:
            nd = getattr(tr, f"notes_string{s}", None)
            parts.append(note_data_to_active_labels(nd, n_frames=T, hop_sec=hop_sec, frame_time_offset=frame_time_offset))
        return np.maximum.reduce(parts) if parts else np.zeros((T, N_NOTES), dtype=np.uint8)

    # fallback (label noise possible)
    return note_data_to_active_labels(tr.notes_all, n_frames=T, hop_sec=hop_sec, frame_time_offset=frame_time_offset)


# ─────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────

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
    P     = (np.abs(S) ** 2).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=int(n_fft)).astype(np.float64)
    return freqs, P


def band_energy_from_power(freqs: np.ndarray, P: np.ndarray, edges: np.ndarray, agg: str = "mean") -> np.ndarray:
    B   = len(edges) - 1
    out = np.zeros((B, P.shape[1]), dtype=np.float32)
    for b in range(B):
        lo, hi = edges[b], edges[b + 1]
        mask   = (freqs >= lo) & (freqs < hi)
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


# ─────────────────────────────────────────────────
# Example slicing + shard writing
# ─────────────────────────────────────────────────

def make_examples(
    feat:       np.ndarray,   # (B,T)
    active:     np.ndarray,   # (T,49)
    onset:      np.ndarray,   # (T,49)
    count:      np.ndarray,   # (T,) int8
    ctx_frames: int,
    stride:     int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    B, T = feat.shape
    if T < ctx_frames:
        return (
            np.zeros((0, B, ctx_frames), np.float16),
            np.zeros((0, N_NOTES),       np.uint8),
            np.zeros((0, N_NOTES),       np.uint8),
            np.zeros((0,),               np.int8),
        )
    xs, ya, yo, yc = [], [], [], []
    for t in range(ctx_frames - 1, T, int(stride)):
        xs.append(feat[:, t - ctx_frames + 1:t + 1].astype(np.float16))
        ya.append(active[t].astype(np.uint8))
        yo.append(onset[t].astype(np.uint8))
        yc.append(count[t])
    return np.stack(xs, 0), np.stack(ya, 0), np.stack(yo, 0), np.array(yc, dtype=np.int8)


def save_shard(out_dir: Path, shard_idx: int, X, YA, YO, YC) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{shard_idx:03d}.npz"
    np.savez_compressed(path, X=X, YA=YA, YO=YO, YC=YC)


# ─────────────────────────────────────────────────
# Augmentation (your originals, kept)
# ─────────────────────────────────────────────────

def one_pole_lpf(x: np.ndarray, a: float) -> np.ndarray:
    y  = np.empty_like(x)
    y0 = 0.0
    for i in range(x.size):
        y0   = (1.0 - a) * x[i] + a * y0
        y[i] = y0
    return y


def tilt_eq(x: np.ndarray, sr: int, tilt_db: float) -> np.ndarray:
    t = float(tilt_db)
    if abs(t) < 1e-6:
        return x
    a  = math.exp(-2.0 * math.pi * 3000.0 / float(sr))
    lp = one_pole_lpf(x, a=a)
    hf = x - lp
    g  = db_to_lin(t)
    return (lp + g * hf).astype(np.float32)


def soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    if drive <= 1e-6:
        return x
    k = 1.0 + 8.0 * float(drive)
    return np.tanh(k * x).astype(np.float32)


def add_noise_floor(x: np.ndarray, snr_db: float) -> np.ndarray:
    if snr_db <= 0:
        return x
    rms       = float(np.sqrt(np.mean(x * x)) + 1e-12)
    noise_rms = rms / db_to_lin(snr_db)
    n         = np.random.randn(x.size).astype(np.float32) * float(noise_rms)
    return (x + n).astype(np.float32)


def augment_di(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    x = x * db_to_lin(float(rng.uniform(-12.0, 12.0)))
    x = tilt_eq(x, sr, tilt_db=float(rng.uniform(-10.0, 10.0)))
    x = soft_clip(x, drive=float(rng.uniform(0.0, 0.35)))
    x = add_noise_floor(x, snr_db=float(rng.uniform(30.0, 60.0)))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def augment_guitarset_safe_gain(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x * db_to_lin(float(rng.uniform(-6.0, 6.0)))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def augment_idmt(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    x = x * db_to_lin(float(rng.uniform(-6.0, 6.0)))
    x = tilt_eq(x, sr, tilt_db=float(rng.uniform(-4.0, 4.0)))
    x = add_noise_floor(x, snr_db=float(rng.uniform(45.0, 70.0)))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────
# Extra single-note helpers (yours, kept)
# ─────────────────────────────────────────────────

def note_to_midi_token(token: str) -> int:
    m = _NOTE_RE.match(token.strip())
    if not m:
        raise ValueError(f"Bad note token: '{token}'")
    letter = m.group(1).upper()
    acc    = m.group(2) or ""
    pc     = letter + ("B" if acc in ("b", "B") else ("#" if acc == "#" else ""))
    semi   = _NOTE_TO_SEMI[pc]
    octv   = int(m.group(3))
    return int((octv + 1) * 12 + semi)


def parse_midi_from_filename(p: Path) -> int:
    stem = p.stem
    m    = _NOTE_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse note from filename: {p.name}")
    base = f"{m.group(1)}{m.group(2) or ''}{m.group(3)}"
    return note_to_midi_token(base)


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


def detect_onset_frame_rms(
    x: np.ndarray, *, sr: int, frame: int, hop: int,
    ratio: float = 6.0, hold: int = 2, nf_sec: float = 0.25,
) -> int:
    if x.size < frame:
        return 0
    n_frames  = 1 + (x.size - frame) // hop
    rms       = np.empty((n_frames,), dtype=np.float32)
    for i in range(n_frames):
        seg    = x[i * hop:i * hop + frame]
        rms[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    nf_frames = min(n_frames, max(10, int(round((nf_sec * float(sr)) / float(hop)))) )
    floor     = float(np.median(rms[:nf_frames]) + 1e-12)
    th        = floor * float(ratio)
    hold      = max(1, int(hold))
    for i in range(0, n_frames - hold + 1):
        if np.all(rms[i:i + hold] >= th):
            return i
    return 0


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build framewise guitar note detection dataset.")

    ap.add_argument("--data_home",   type=str,   default=None)
    ap.add_argument("--out_dir",     type=str,   required=True)
    ap.add_argument("--download",    action="store_true")

    # GuitarSet audio selection
    ap.add_argument("--gs_source", choices=["mix", "mic", "hex"], default="hex",
                    help="GuitarSet source: mix/mic or hex (6-ch DI pickup, recommended).")
    ap.add_argument("--gs_mixes_per_track", type=int, default=2,
                    help="For gs_source=hex: number of random string-subset mixes per track.")
    ap.add_argument("--allow_zero_poly", action="store_true",
                    help="For gs_source=hex: allow 0-string mixes (silence) to teach 'no notes'.")

    # audio / feature params
    ap.add_argument("--sr",          type=int,   default=48000)
    ap.add_argument("--hop_length",  type=int,   default=256)
    ap.add_argument("--n_fft",       type=int,   default=1024)
    ap.add_argument("--bands",       type=int,   default=48)
    ap.add_argument("--fmin",        type=float, default=80.0)
    ap.add_argument("--fmax",        type=float, default=6000.0)
    ap.add_argument("--band_agg",    choices=["mean", "max"], default="mean")

    ap.add_argument("--ctx_ms",      type=float, default=60.0)
    ap.add_argument("--stride",      type=int,   default=1)
    ap.add_argument("--onset_width", type=int,   default=2)

    # sharding / split
    ap.add_argument("--shard_size",  type=int,   default=5000,
                    help="Examples per shard (lower = less RAM).")
    ap.add_argument("--flush_mb",    type=int,   default=512,
                    help="Flush shard buffers when buffered arrays exceed this many MB.")
    ap.add_argument("--val_split",   type=float, default=0.10)
    ap.add_argument("--max_tracks",  type=int,   default=0)
    ap.add_argument("--split_seed",  type=int,   default=1337)

    # Extra single-note DI folder
    ap.add_argument("--extra_dir",       type=str,   default=None)
    ap.add_argument("--extra_val_split", type=float, default=0.10)

    # IDMT-SMT-Guitar root (optional)
    ap.add_argument("--idmt_dir",        type=str,   default=None,
                    help="Root of IDMT-SMT-Guitar dataset (contains dataset1/dataset2/dataset3).")
    ap.add_argument("--idmt_val_split",  type=float, default=0.10)

    # preprocessing realism
    ap.add_argument("--dc_block", action="store_true", help="Apply DC blocking filter before feature extraction.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hop_sec    = float(args.hop_length) / float(args.sr)
    ctx_frames = max(4, int(round((args.ctx_ms / 1000.0) / hop_sec)))
    edges      = make_log_spaced_edges(args.fmin, args.fmax, args.bands)

    # Align labels to STFT energy center (center=False)
    frame_time_offset = (float(args.n_fft) / 2.0) / float(args.sr)

    # ── GuitarSet ──────────────────────────────────
    ds = mirdata.initialize("guitarset", data_home=args.data_home)
    if args.download:
        ds.download()
        ds.validate()

    track_ids = sorted(list(ds.track_ids))
    if args.max_tracks and args.max_tracks > 0:
        track_ids = track_ids[:args.max_tracks]

    # deterministic shuffled split
    rng_split = np.random.default_rng(int(args.split_seed))
    ids_arr = np.array(track_ids)
    rng_split.shuffle(ids_arr)
    n_val = int(round(len(ids_arr) * float(args.val_split)))
    val_ids = set(ids_arr[:n_val].tolist())
    train_ids = ids_arr[n_val:].tolist()
    val_ids_list = sorted(list(val_ids))

    # ── IDMT-SMT-Guitar ───────────────────────────
    idmt_pairs: List[Tuple[Path, Path]] = []
    idmt_train: List[Tuple[Path, Path]] = []
    idmt_val:   List[Tuple[Path, Path]] = []
    if args.idmt_dir:
        idmt_pairs = discover_idmt_pairs(Path(args.idmt_dir))
        rng_idmt = np.random.default_rng(42)
        idx = np.arange(len(idmt_pairs))
        rng_idmt.shuffle(idx)
        cut = int(round(len(idmt_pairs) * float(args.idmt_val_split)))
        val_idx = set(idx[:cut].tolist())
        for i in range(len(idmt_pairs)):
            (idmt_val if i in val_idx else idmt_train).append(idmt_pairs[i])
        print(f"[idmt] train={len(idmt_train)}  val={len(idmt_val)}")

    # ── Extra single-note files ────────────────────
    extra_train: List[Path] = []
    extra_val:   List[Path] = []
    if args.extra_dir:
        extra_files = sorted(Path(args.extra_dir).glob("*.wav"))
        print(f"[extra] found {len(extra_files)} wavs in {args.extra_dir}")
        rng_ex = np.random.default_rng(123)
        idx = np.arange(len(extra_files))
        rng_ex.shuffle(idx)
        cut = int(round(len(extra_files) * float(args.extra_val_split)))
        val_set = set(idx[:cut].tolist())
        for i, p in enumerate(extra_files):
            (extra_val if i in val_set else extra_train).append(p)
        print(f"[extra] split -> train={len(extra_train)} val={len(extra_val)}")

    # ─────────────────────────────────────────────
    # PASS 1: compute mu/sigma on TRAIN split
    # ─────────────────────────────────────────────
    band_sum   = np.zeros((args.bands,), dtype=np.float64)
    band_sq    = np.zeros((args.bands,), dtype=np.float64)
    band_count = 0

    print("[pass1] computing per-band mu/sigma on train split...")

    # GuitarSet pass1
    for tid in train_ids:
        tr = ds.track(tid)

        if args.gs_source == "hex":
            try:
                hex6 = load_guitarset_hex6(tr, sr_target=args.sr)
            except Exception as ex:
                print(f"[gs pass1 skip] {tid}: {ex}")
                continue
            rng = np.random.default_rng(seed=hash(tid) & 0xFFFFFFFF)

            for m in range(int(args.gs_mixes_per_track)):
                strings = choose_string_subset(rng, allow_zero=bool(args.allow_zero_poly))
                y_audio = mix_strings(hex6, strings)
                if args.dc_block:
                    y_audio = dc_block(y_audio)
                y_audio = augment_guitarset_safe_gain(y_audio, rng)

                freqs, P = stft_power(y_audio, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
                band = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
                xlog = log1p_feat(band)
                band_sum += xlog.sum(axis=1)
                band_sq  += (xlog * xlog).sum(axis=1)
                band_count += xlog.shape[1]

        else:
            audio = tr.audio_mix if args.gs_source == "mix" else tr.audio_mic
            if audio is None:
                continue
            y_audio, sr = audio
            y_audio = y_audio.astype(np.float32)
            if int(sr) != args.sr:
                y_audio = librosa.resample(y_audio, orig_sr=int(sr), target_sr=args.sr).astype(np.float32)
            if args.dc_block:
                y_audio = dc_block(y_audio)
            rng = np.random.default_rng(seed=hash(tid) & 0xFFFFFFFF)
            y_audio = augment_guitarset_safe_gain(y_audio, rng)

            freqs, P = stft_power(y_audio, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
            band = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
            xlog = log1p_feat(band)
            band_sum += xlog.sum(axis=1)
            band_sq  += (xlog * xlog).sum(axis=1)
            band_count += xlog.shape[1]

    # IDMT pass1
    for wav_path, _ in idmt_train:
        try:
            x = load_wav_mono_resample(wav_path, sr_target=args.sr)
        except Exception as ex:
            print(f"[idmt pass1 skip] {wav_path.name}: {ex}")
            continue
        if args.dc_block:
            x = dc_block(x)
        rng = np.random.default_rng(seed=hash(wav_path.name) & 0xFFFFFFFF)
        x = augment_idmt(x, args.sr, rng)

        freqs, P = stft_power(x, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
        band = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
        xlog = log1p_feat(band)
        band_sum += xlog.sum(axis=1)
        band_sq  += (xlog * xlog).sum(axis=1)
        band_count += xlog.shape[1]

    # extra pass1
    for p in extra_train:
        try:
            x = load_wav_mono_resample(p, sr_target=args.sr)
        except Exception as ex:
            print(f"[extra pass1 skip] {p.name}: {ex}")
            continue
        if args.dc_block:
            x = dc_block(x)
        rng = np.random.default_rng(seed=hash(p.name) & 0xFFFFFFFF)
        x = augment_di(x, args.sr, rng)

        freqs, P = stft_power(x, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
        band = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
        xlog = log1p_feat(band)
        band_sum += xlog.sum(axis=1)
        band_sq  += (xlog * xlog).sum(axis=1)
        band_count += xlog.shape[1]

    mu = (band_sum / max(1, band_count)).astype(np.float32)
    var = (band_sq / max(1, band_count) - (mu.astype(np.float64) ** 2)).astype(np.float32)
    sigma = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)

    meta = {
        "dataset": "GuitarSet + IDMT-SMT-Guitar + extra_single_notes",
        "guitarset": {
            "gs_source": args.gs_source,
            "gs_mixes_per_track": int(args.gs_mixes_per_track),
            "allow_zero_poly": bool(args.allow_zero_poly),
        },
        "sr": args.sr,
        "hop_length": args.hop_length,
        "hop_sec": hop_sec,
        "n_fft": args.n_fft,
        "frame_time_offset": frame_time_offset,
        "bands": args.bands,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "band_agg": args.band_agg,
        "ctx_ms": args.ctx_ms,
        "ctx_frames": ctx_frames,
        "stride": args.stride,
        "onset_width_frames": args.onset_width,
        "label": {
            "midi_min": MIDI_MIN,
            "midi_max": MIDI_MAX,
            "n_notes": N_NOTES,
            "count_max": MAX_POLY,
            "keys": "YA=active(49,), YO=onset(49,), YC=count(int 0-6)",
        },
        "splits": {
            "guitarset_train": train_ids,
            "guitarset_val": val_ids_list,
            "split_seed": int(args.split_seed),
        },
        "idmt": {
            "dir": str(args.idmt_dir) if args.idmt_dir else None,
            "train": len(idmt_train),
            "val": len(idmt_val),
        },
        "extra": {
            "dir": str(args.extra_dir) if args.extra_dir else None,
            "train_files": [p.name for p in extra_train],
            "val_files": [p.name for p in extra_val],
        },
        "norm": {
            "type": "per_band_zscore_log1p",
            "band_mu": mu.tolist(),
            "band_sigma": sigma.tolist(),
        },
        "preproc": {
            "dc_block": bool(args.dc_block),
        },
    }

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[pass1] done. wrote metadata.json")

    # ─────────────────────────────────────────────
    # PASS 2: write shards
    # ─────────────────────────────────────────────

    flush_bytes_limit = int(args.flush_mb) * 1024 * 1024

    def buffered_bytes(X_buf, YA_buf, YO_buf, YC_buf) -> int:
        total = 0
        for arrs in (X_buf, YA_buf, YO_buf, YC_buf):
            for a in arrs:
                total += int(a.nbytes)
        return total

    def flush(split_dir: Path, shard_idx: int, X_buf, YA_buf, YO_buf, YC_buf) -> int:
        Xs  = np.concatenate(X_buf,  axis=0)
        YAs = np.concatenate(YA_buf, axis=0)
        YOs = np.concatenate(YO_buf, axis=0)
        YCs = np.concatenate(YC_buf, axis=0)
        save_shard(split_dir, shard_idx, Xs, YAs, YOs, YCs)
        return Xs.shape[0]

    def process_split(
        split_name:  str,
        gs_ids:      List[str],
        idmt_pairs_: List[Tuple[Path, Path]],
        extras_:     List[Path],
    ) -> None:
        split_dir = out_dir / split_name
        shard_idx = 0
        cur_in_shard = 0
        X_buf: List[np.ndarray] = []
        YA_buf: List[np.ndarray] = []
        YO_buf: List[np.ndarray] = []
        YC_buf: List[np.ndarray] = []

        print(f"\n[pass2] writing {split_name} shards ...")

        # ── GuitarSet ──
        for tid in gs_ids:
            tr = ds.track(tid)

            if args.gs_source == "hex":
                try:
                    hex6 = load_guitarset_hex6(tr, sr_target=args.sr)
                except Exception as ex:
                    print(f"  [gs skip] {tid}: {ex}")
                    continue

                rng = np.random.default_rng(seed=hash(tid) & 0xFFFFFFFF)

                for m in range(int(args.gs_mixes_per_track)):
                    strings = choose_string_subset(rng, allow_zero=bool(args.allow_zero_poly))
                    y_audio = mix_strings(hex6, strings)
                    if args.dc_block:
                        y_audio = dc_block(y_audio)
                    y_audio = augment_guitarset_safe_gain(y_audio, rng)

                    freqs, P = stft_power(y_audio, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
                    band = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
                    feat = zscore_feat(log1p_feat(band), mu=mu, sigma=sigma)
                    T = feat.shape[1]

                    active = guitarset_active_for_strings(tr, strings, T, hop_sec, frame_time_offset)
                    onset = active_to_onset_labels(active, onset_width_frames=args.onset_width)
                    poly_cnt = active_to_count_labels(active)

                    X, YA, YO, YC = make_examples(feat, active, onset, poly_cnt, ctx_frames=ctx_frames, stride=args.stride)
                    if X.shape[0] == 0:
                        continue

                    X_buf.append(X); YA_buf.append(YA); YO_buf.append(YO); YC_buf.append(YC)
                    cur_in_shard += int(X.shape[0])

                    if cur_in_shard >= int(args.shard_size) or buffered_bytes(X_buf, YA_buf, YO_buf, YC_buf) >= flush_bytes_limit:
                        written = flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf)
                        shard_idx += 1
                        cur_in_shard = 0
                        X_buf.clear(); YA_buf.clear(); YO_buf.clear(); YC_buf.clear()

            else:
                audio = tr.audio_mix if args.gs_source == "mix" else tr.audio_mic
                if audio is None:
                    continue
                y_audio, sr = audio
                y_audio = y_audio.astype(np.float32)
                if int(sr) != args.sr:
                    y_audio = librosa.resample(y_audio, orig_sr=int(sr), target_sr=args.sr).astype(np.float32)
                if args.dc_block:
                    y_audio = dc_block(y_audio)
                rng = np.random.default_rng(seed=hash(tid) & 0xFFFFFFFF)
                y_audio = augment_guitarset_safe_gain(y_audio, rng)

                freqs, P = stft_power(y_audio, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
                band = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
                feat = zscore_feat(log1p_feat(band), mu=mu, sigma=sigma)
                T = feat.shape[1]

                active = note_data_to_active_labels(tr.notes_all, n_frames=T, hop_sec=hop_sec, frame_time_offset=frame_time_offset)
                onset = active_to_onset_labels(active, onset_width_frames=args.onset_width)
                poly_cnt = active_to_count_labels(active)

                X, YA, YO, YC = make_examples(feat, active, onset, poly_cnt, ctx_frames=ctx_frames, stride=args.stride)
                if X.shape[0] == 0:
                    continue

                X_buf.append(X); YA_buf.append(YA); YO_buf.append(YO); YC_buf.append(YC)
                cur_in_shard += int(X.shape[0])

                if cur_in_shard >= int(args.shard_size) or buffered_bytes(X_buf, YA_buf, YO_buf, YC_buf) >= flush_bytes_limit:
                    written = flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf)
                    shard_idx += 1
                    cur_in_shard = 0
                    X_buf.clear(); YA_buf.clear(); YO_buf.clear(); YC_buf.clear()

        # ── IDMT-SMT-Guitar ──
        if idmt_pairs_:
            print(f"  [idmt] processing {len(idmt_pairs_)} files ...")
        for wav_path, xml_path in idmt_pairs_:
            try:
                x = load_wav_mono_resample(wav_path, sr_target=args.sr)
            except Exception as ex:
                print(f"  [idmt skip] {wav_path.name}: {ex}")
                continue

            if args.dc_block:
                x = dc_block(x)

            rng = np.random.default_rng(seed=hash(wav_path.name) & 0xFFFFFFFF)
            x = augment_idmt(x, args.sr, rng)

            freqs, P = stft_power(x, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
            band = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
            feat = zscore_feat(log1p_feat(band), mu=mu, sigma=sigma)
            T = feat.shape[1]

            try:
                intervals, midis = parse_idmt_xml(xml_path)
            except Exception as ex:
                print(f"  [idmt xml skip] {xml_path.name}: {ex}")
                continue

            active = build_idmt_active_from_arrays(intervals, midis, n_frames=T, hop_sec=hop_sec, frame_time_offset=frame_time_offset)
            onset = active_to_onset_labels(active, onset_width_frames=args.onset_width)
            poly_cnt = active_to_count_labels(active)

            X, YA, YO, YC = make_examples(feat, active, onset, poly_cnt, ctx_frames=ctx_frames, stride=args.stride)
            if X.shape[0] == 0:
                continue

            X_buf.append(X); YA_buf.append(YA); YO_buf.append(YO); YC_buf.append(YC)
            cur_in_shard += int(X.shape[0])

            if cur_in_shard >= int(args.shard_size) or buffered_bytes(X_buf, YA_buf, YO_buf, YC_buf) >= flush_bytes_limit:
                written = flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf)
                shard_idx += 1
                cur_in_shard = 0
                X_buf.clear(); YA_buf.clear(); YO_buf.clear(); YC_buf.clear()

        # ── Extra single-note wavs ──
        if extras_:
            print(f"  [extra] processing {len(extras_)} single-note wavs ...")
        for p in extras_:
            try:
                midi = parse_midi_from_filename(p)
                if midi < MIDI_MIN or midi > MIDI_MAX:
                    continue
                x = load_wav_mono_resample(p, sr_target=args.sr)
                if args.dc_block:
                    x = dc_block(x)
                rng = np.random.default_rng(seed=hash(p.name) & 0xFFFFFFFF)
                x = augment_di(x, args.sr, rng)
            except Exception as ex:
                print(f"  [extra skip] {p.name}: {ex}")
                continue

            freqs, P = stft_power(x, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
            band = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
            feat = zscore_feat(log1p_feat(band), mu=mu, sigma=sigma)
            T = feat.shape[1]

            onset_fr = detect_onset_frame_rms(
                x, sr=args.sr,
                frame=max(1024, args.n_fft // 2),
                hop=args.hop_length, ratio=6.0, hold=2, nf_sec=0.25,
            )
            onset_fr = int(max(0, min(T - 1, onset_fr)))

            active = np.zeros((T, N_NOTES), dtype=np.uint8)
            onset = np.zeros((T, N_NOTES), dtype=np.uint8)
            k = midi - MIDI_MIN

            # shift by frame_time_offset so extra labels are aligned similarly:
            # (approximate by shifting onset_fr forward a bit)
            onset_fr2 = int(max(0, min(T - 1, onset_fr + round(frame_time_offset / hop_sec))))
            active[onset_fr2:, k] = 1
            onset[onset_fr2:min(T, onset_fr2 + int(args.onset_width)), k] = 1

            poly_cnt = active_to_count_labels(active)

            X, YA, YO, YC = make_examples(feat, active, onset, poly_cnt, ctx_frames=ctx_frames, stride=args.stride)
            if X.shape[0] == 0:
                continue

            X_buf.append(X); YA_buf.append(YA); YO_buf.append(YO); YC_buf.append(YC)
            cur_in_shard += int(X.shape[0])

            if cur_in_shard >= int(args.shard_size) or buffered_bytes(X_buf, YA_buf, YO_buf, YC_buf) >= flush_bytes_limit:
                written = flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf)
                shard_idx += 1
                cur_in_shard = 0
                X_buf.clear(); YA_buf.clear(); YO_buf.clear(); YC_buf.clear()

        # flush remainder
        if cur_in_shard > 0 and X_buf:
            flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf)

    process_split("train", train_ids,    idmt_train, extra_train)
    process_split("val",   val_ids_list, idmt_val,   extra_val)
    print(f"\nDone. Dataset written to: {out_dir}")


if __name__ == "__main__":
    main()