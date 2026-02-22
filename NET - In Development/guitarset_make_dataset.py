# !/usr/bin/env python3
r"""
guitarset_make_fb_dataset.py  (v2 - with IDMT-SMT-Guitar + polyphony count)

Builds framewise guitar note detection dataset from:
  - GuitarSet (via mirdata)
  - IDMT-SMT-Guitar (monophonic + polyphonic DI/clean recordings, XML annotations)
  - Extra single-note DI folder (C4_2.wav style naming)

Labels per frame:
  - active[49]  : note sounding (sigmoid target)
  - onset[49]   : note attack (sigmoid target)
  - count[1]    : polyphonic note count, clipped 0-6 (CE target, int)

Features: STFT power -> log-spaced band pooling -> log1p -> per-band zscore

Install:
  pip install numpy librosa mirdata soundfile scipy lxml
python guitarset_make_fb_dataset.py \
  --out_dir gs_dataset \
  --download \
  --extra_dir ../samples \
  --idmt_dir C:\Users\anwga\Downloads\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2
  C:\Users\anwga\Documents\GitHub\cse123\NET - In Development
  ../../../../Downloads\IDMT-SMT-GUITAR_V2\IDMT-SMT-GUITAR_V2
IDMT-SMT-Guitar:
  Download from https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html
  Point --idmt_dir at the root folder containing "dataset1", "dataset2", "dataset3" subfolders.
"""
import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import librosa
import mirdata
from scipy.io import wavfile

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
# Labels helpers
# ─────────────────────────────────────────────────

def note_data_to_active_labels(note_data, n_frames: int, hop_sec: float) -> np.ndarray:
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
        i0 = int(math.floor(t0 / hop_sec))
        i1 = int(math.ceil(t1  / hop_sec))
        i0 = max(0, min(n_frames, i0))
        i1 = max(0, min(n_frames, i1))
        if i1 > i0:
            y[i0:i1, m - MIDI_MIN] = 1
    return y


def active_to_onset_labels(active: np.ndarray, onset_width_frames: int = 2) -> np.ndarray:
    T, D = active.shape
    onset = np.zeros((T, D), dtype=np.uint8)
    prev  = np.zeros((D,),   dtype=np.uint8)
    for t in range(T):
        cur    = active[t]
        rising = (cur == 1) & (prev == 0)
        if np.any(rising):
            t1 = min(T, t + onset_width_frames)
            onset[t:t1, rising] = 1
        prev = cur
    return onset


def active_to_count_labels(active: np.ndarray) -> np.ndarray:
    """Returns int8 array of shape (T,) with polyphony count, clamped to [0, MAX_POLY]."""
    counts = active.sum(axis=1).astype(np.int8)
    return np.clip(counts, 0, MAX_POLY)


# ─────────────────────────────────────────────────
# IDMT-SMT-Guitar annotation parser
# ─────────────────────────────────────────────────

def parse_idmt_xml(xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse an IDMT-SMT-Guitar XML annotation file.
    Returns (intervals, midis) where intervals is (N,2) float64 and midis is (N,) float64.

    The XML structure looks like:
      <instrumentRecording>
        <globalParameter>
          <sampleRate>44100</sampleRate>
        </globalParameter>
        <transcription>
          <event>
            <onsetSec>0.123</onsetSec>
            <offsetSec>0.456</offsetSec>
            <pitch>64</pitch>   <!-- MIDI number or semitone -->
          </event>
          ...
        </transcription>
      </instrumentRecording>

    Dataset3 uses <pitchname> (e.g. "E4") instead of <pitch> MIDI int; we handle both.
    """
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as e:
        print(f"[idmt] XML parse error {xml_path.name}: {e}")
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    root = tree.getroot()
    intervals, midis = [], []

    for event in root.iter("event"):
        # onset
        onset_el  = event.find("onsetSec")
        offset_el = event.find("offsetSec")
        if onset_el is None or offset_el is None:
            # fallback: some files use startTime/endTime in samples
            onset_el  = event.find("onsetSample")
            offset_el = event.find("offsetSample")
            if onset_el is None:
                continue
            sr_el = root.find(".//sampleRate")
            sr    = float(sr_el.text) if sr_el is not None else 44100.0
            t0 = float(onset_el.text)  / sr
            t1 = float(offset_el.text) / sr if offset_el is not None else t0 + 0.5
        else:
            t0 = float(onset_el.text)
            t1 = float(offset_el.text) if offset_el is not None else t0 + 0.5

        # pitch — try numeric first, then pitchname
        pitch_el = event.find("pitch")
        if pitch_el is not None and pitch_el.text is not None:
            try:
                midi = int(pitch_el.text.strip())
            except ValueError:
                midi = _pitchname_to_midi(pitch_el.text.strip())
        else:
            pn_el = event.find("pitchname")
            if pn_el is None or pn_el.text is None:
                continue
            midi = _pitchname_to_midi(pn_el.text.strip())

        if midi < 0:
            continue

        intervals.append([t0, t1])
        midis.append(float(midi))

    if not intervals:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.array(intervals, dtype=np.float64), np.array(midis, dtype=np.float64)


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


def build_idmt_active_from_arrays(
    intervals: np.ndarray,
    midis: np.ndarray,
    n_frames: int,
    hop_sec: float,
) -> np.ndarray:
    active = np.zeros((n_frames, N_NOTES), dtype=np.uint8)
    for (t0, t1), midi in zip(intervals, midis):
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue
        m = int(round(midi))
        if m < MIDI_MIN or m > MIDI_MAX:
            continue
        i0 = max(0, min(n_frames, int(math.floor(t0 / hop_sec))))
        i1 = max(0, min(n_frames, int(math.ceil(t1  / hop_sec))))
        if i1 > i0:
            active[i0:i1, m - MIDI_MIN] = 1
    return active


def discover_idmt_pairs(idmt_root: Path) -> List[Tuple[Path, Path]]:
    """
    Walk IDMT-SMT-Guitar root and find (wav, xml) pairs.
    The dataset has audio/ and annotation/ (or Annotation/) subfolders at various depths.
    We locate all .wav files then look for a sibling .xml with the same stem.
    """
    pairs = []
    audio_files = list(idmt_root.rglob("*.wav")) + list(idmt_root.rglob("*.WAV"))
    for wav_path in audio_files:
        # Prefer sibling annotation directory at same level
        xml_path = wav_path.with_suffix(".xml")
        if not xml_path.exists():
            # Try annotation subfolder pattern: audio/../annotation/stem.xml
            candidates = list(wav_path.parent.parent.rglob(wav_path.stem + ".xml"))
            if candidates:
                xml_path = candidates[0]
            else:
                continue   # no annotation found; skip
        pairs.append((wav_path, xml_path))
    print(f"[idmt] discovered {len(pairs)} annotated pairs under {idmt_root}")
    return pairs


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


def make_examples(
    feat:       np.ndarray,   # (B,T)
    active:     np.ndarray,   # (T,49)
    onset:      np.ndarray,   # (T,49)
    count:      np.ndarray,   # (T,)   int8
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
    for t in range(ctx_frames - 1, T, stride):
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
# Augmentation
# ─────────────────────────────────────────────────

def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


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
    x = x * db_to_lin(rng.uniform(-12.0, 12.0))
    x = tilt_eq(x, sr, tilt_db=rng.uniform(-10.0, 10.0))
    x = soft_clip(x, drive=rng.uniform(0.0, 0.35))
    x = add_noise_floor(x, snr_db=float(rng.uniform(30.0, 60.0)))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def augment_guitarset_safe_gain(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x * db_to_lin(rng.uniform(-6.0, 6.0))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# Same mild augmentation for IDMT — it's already a controlled studio DI, keep it gentle
def augment_idmt(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    x = x * db_to_lin(rng.uniform(-6.0, 6.0))
    x = tilt_eq(x, sr, tilt_db=rng.uniform(-4.0, 4.0))
    x = add_noise_floor(x, snr_db=float(rng.uniform(45.0, 70.0)))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────
# Extra single-note helpers
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
    nf_frames = min(n_frames, max(10, int(round((nf_sec * float(sr)) / float(hop)))))
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
    ap.add_argument("--audio",       choices=["mix", "mic"], default="mix")
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
    ap.add_argument("--shard_size",  type=int,   default=25000)
    ap.add_argument("--val_split",   type=float, default=0.10)
    ap.add_argument("--max_tracks",  type=int,   default=0)
    # Extra single-note DI folder
    ap.add_argument("--extra_dir",       type=str,   default=None)
    ap.add_argument("--extra_val_split", type=float, default=0.10)
    # IDMT-SMT-Guitar root (optional but recommended)
    ap.add_argument("--idmt_dir",        type=str,   default=None,
                    help="Root of IDMT-SMT-Guitar dataset (contains dataset1/dataset2/dataset3).")
    ap.add_argument("--idmt_val_split",  type=float, default=0.10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── GuitarSet ──────────────────────────────────
    ds       = mirdata.initialize("guitarset", data_home=args.data_home)
    if args.download:
        ds.download()
        ds.validate()
    track_ids = sorted(list(ds.track_ids))
    if args.max_tracks and args.max_tracks > 0:
        track_ids = track_ids[:args.max_tracks]
    n_val      = int(round(len(track_ids) * args.val_split))
    val_ids    = set(track_ids[:n_val])
    train_ids  = [tid for tid in track_ids if tid not in val_ids]
    val_ids_list = sorted(list(val_ids))

    hop_sec    = float(args.hop_length) / float(args.sr)
    ctx_frames = max(4, int(round((args.ctx_ms / 1000.0) / hop_sec)))
    edges      = make_log_spaced_edges(args.fmin, args.fmax, args.bands)

    # ── IDMT-SMT-Guitar ───────────────────────────
    idmt_pairs: List[Tuple[Path, Path]] = []
    idmt_train: List[Tuple[Path, Path]] = []
    idmt_val:   List[Tuple[Path, Path]] = []
    if args.idmt_dir:
        idmt_pairs = discover_idmt_pairs(Path(args.idmt_dir))
        rng_idmt   = np.random.default_rng(42)
        idx        = np.arange(len(idmt_pairs))
        rng_idmt.shuffle(idx)
        cut        = int(round(len(idmt_pairs) * args.idmt_val_split))
        val_set    = set(idx[:cut].tolist())
        for i, pair in enumerate(idmt_pairs):
            (idmt_val if i in val_set else idmt_train).append(pair)
        print(f"[idmt] train={len(idmt_train)}  val={len(idmt_val)}")

    # ── Extra single-note files ────────────────────
    extra_files: List[Path] = []
    extra_train: List[Path] = []
    extra_val:   List[Path] = []
    if args.extra_dir:
        extra_files = sorted(Path(args.extra_dir).glob("*.wav"))
        print(f"[extra] found {len(extra_files)} wavs in {args.extra_dir}")
        rng_ex = np.random.default_rng(123)
        idx    = np.arange(len(extra_files))
        rng_ex.shuffle(idx)
        cut    = int(round(len(extra_files) * float(args.extra_val_split)))
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

    for tid in train_ids:
        tr    = ds.track(tid)
        audio = tr.audio_mix if args.audio == "mix" else tr.audio_mic
        if audio is None:
            continue
        y_audio, sr = audio
        y_audio = y_audio.astype(np.float32)
        if int(sr) != args.sr:
            y_audio = librosa.resample(y_audio, orig_sr=int(sr), target_sr=args.sr).astype(np.float32)
        freqs, P = stft_power(y_audio, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
        band     = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
        xlog     = log1p_feat(band)
        band_sum   += xlog.sum(axis=1)
        band_sq    += (xlog * xlog).sum(axis=1)
        band_count += xlog.shape[1]

    for wav_path, _ in idmt_train:
        try:
            x = load_wav_mono_resample(wav_path, sr_target=args.sr)
        except Exception as ex:
            print(f"[idmt pass1 skip] {wav_path.name}: {ex}")
            continue
        freqs, P = stft_power(x, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
        band     = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
        xlog     = log1p_feat(band)
        band_sum   += xlog.sum(axis=1)
        band_sq    += (xlog * xlog).sum(axis=1)
        band_count += xlog.shape[1]

    for p in extra_train:
        try:
            x = load_wav_mono_resample(p, sr_target=args.sr)
        except Exception as ex:
            print(f"[extra pass1 skip] {p.name}: {ex}")
            continue
        freqs, P = stft_power(x, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
        band     = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
        xlog     = log1p_feat(band)
        band_sum   += xlog.sum(axis=1)
        band_sq    += (xlog * xlog).sum(axis=1)
        band_count += xlog.shape[1]

    mu    = (band_sum / max(1, band_count)).astype(np.float32)
    var   = (band_sq  / max(1, band_count) - (mu.astype(np.float64) ** 2)).astype(np.float32)
    sigma = np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)

    # ── metadata ──────────────────────────────────
    meta = {
        "dataset": "GuitarSet + IDMT-SMT-Guitar + extra_single_notes",
        "audio":   args.audio,
        "sr":      args.sr,
        "hop_length": args.hop_length,
        "hop_sec":    hop_sec,
        "n_fft":      args.n_fft,
        "bands":      args.bands,
        "fmin":       args.fmin,
        "fmax":       args.fmax,
        "band_agg":   args.band_agg,
        "ctx_ms":     args.ctx_ms,
        "ctx_frames": ctx_frames,
        "stride":     args.stride,
        "onset_width_frames": args.onset_width,
        "label": {
            "midi_min": MIDI_MIN, "midi_max": MIDI_MAX, "n_notes": N_NOTES,
            "count_max": MAX_POLY,
            "keys": "YA=active(49,), YO=onset(49,), YC=count(int 0-6)",
        },
        "splits":  {"guitarset_train": train_ids, "guitarset_val": val_ids_list},
        "idmt": {
            "dir":    str(args.idmt_dir) if args.idmt_dir else None,
            "train":  len(idmt_train),
            "val":    len(idmt_val),
        },
        "extra": {
            "dir":         str(args.extra_dir) if args.extra_dir else None,
            "train_files": [p.name for p in extra_train],
            "val_files":   [p.name for p in extra_val],
        },
        "norm": {
            "type":       "per_band_zscore_log1p",
            "band_mu":    mu.tolist(),
            "band_sigma": sigma.tolist(),
        },
        "augmentation": {
            "guitarset": "safe_gain_only",
            "idmt":      "mild gain + tilt + noise",
            "extra_dir": "augment_di (gain+tilt+softclip+noise)",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[pass1] done. wrote metadata.json")

    # ─────────────────────────────────────────────
    # PASS 2: write shards
    # ─────────────────────────────────────────────

    def flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf):
        Xs  = np.concatenate(X_buf,  axis=0)
        YAs = np.concatenate(YA_buf, axis=0)
        YOs = np.concatenate(YO_buf, axis=0)
        YCs = np.concatenate(YC_buf, axis=0)
        save_shard(split_dir, shard_idx, Xs, YAs, YOs, YCs)

    def process_split(
        split_name:  str,
        gs_ids:      List[str],
        idmt_pairs_: List[Tuple[Path, Path]],
        extras_:     List[Path],
    ) -> None:
        split_dir = out_dir / split_name
        shard_idx = 0
        count     = 0
        X_buf, YA_buf, YO_buf, YC_buf = [], [], [], []

        print(f"\n[pass2] writing {split_name} shards ...")

        # ── GuitarSet ──
        for tid in gs_ids:
            tr    = ds.track(tid)
            audio = tr.audio_mix if args.audio == "mix" else tr.audio_mic
            if audio is None:
                continue
            y_audio, sr = audio
            y_audio = y_audio.astype(np.float32)
            if int(sr) != args.sr:
                y_audio = librosa.resample(y_audio, orig_sr=int(sr), target_sr=args.sr).astype(np.float32)
            rng     = np.random.default_rng(seed=hash(tid) & 0xFFFFFFFF)
            y_audio = augment_guitarset_safe_gain(y_audio, rng)
            freqs, P = stft_power(y_audio, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
            band     = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
            xlog     = log1p_feat(band)
            feat     = zscore_feat(xlog, mu=mu, sigma=sigma)
            T        = feat.shape[1]
            active   = note_data_to_active_labels(tr.notes_all, n_frames=T, hop_sec=hop_sec)
            onset    = active_to_onset_labels(active, onset_width_frames=args.onset_width)
            poly_cnt = active_to_count_labels(active)
            X, YA, YO, YC = make_examples(feat, active, onset, poly_cnt, ctx_frames=ctx_frames, stride=args.stride)
            if X.shape[0] == 0:
                continue
            X_buf.append(X); YA_buf.append(YA); YO_buf.append(YO); YC_buf.append(YC)
            count += X.shape[0]
            if count >= args.shard_size:
                flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf)
                shard_idx += 1; count = 0
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
            rng = np.random.default_rng(seed=hash(wav_path.name) & 0xFFFFFFFF)
            x   = augment_idmt(x, args.sr, rng)
            freqs, P = stft_power(x, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
            band     = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
            xlog     = log1p_feat(band)
            feat     = zscore_feat(xlog, mu=mu, sigma=sigma)
            T        = feat.shape[1]
            try:
                intervals, midis = parse_idmt_xml(xml_path)
            except Exception as ex:
                print(f"  [idmt xml skip] {xml_path.name}: {ex}")
                continue
            active   = build_idmt_active_from_arrays(intervals, midis, n_frames=T, hop_sec=hop_sec)
            onset    = active_to_onset_labels(active, onset_width_frames=args.onset_width)
            poly_cnt = active_to_count_labels(active)
            X, YA, YO, YC = make_examples(feat, active, onset, poly_cnt, ctx_frames=ctx_frames, stride=args.stride)
            if X.shape[0] == 0:
                continue
            X_buf.append(X); YA_buf.append(YA); YO_buf.append(YO); YC_buf.append(YC)
            count += X.shape[0]
            if count >= args.shard_size:
                flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf)
                shard_idx += 1; count = 0
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
                rng = np.random.default_rng(seed=hash(p.name) & 0xFFFFFFFF)
                x   = augment_di(x, args.sr, rng)
            except Exception as ex:
                print(f"  [extra skip] {p.name}: {ex}")
                continue
            freqs, P = stft_power(x, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)
            band     = band_energy_from_power(freqs, P, edges, agg=args.band_agg)
            xlog     = log1p_feat(band)
            feat     = zscore_feat(xlog, mu=mu, sigma=sigma)
            T        = feat.shape[1]
            onset_fr = detect_onset_frame_rms(
                x, sr=args.sr,
                frame=max(1024, args.n_fft // 2),
                hop=args.hop_length, ratio=6.0, hold=2, nf_sec=0.25,
            )
            onset_fr = int(max(0, min(T - 1, onset_fr)))
            active   = np.zeros((T, N_NOTES), dtype=np.uint8)
            onset    = np.zeros((T, N_NOTES), dtype=np.uint8)
            k        = midi - MIDI_MIN
            active[onset_fr:, k]                                     = 1
            onset[onset_fr:min(T, onset_fr + args.onset_width), k]  = 1
            poly_cnt = active_to_count_labels(active)      # always 0 or 1
            X, YA, YO, YC = make_examples(feat, active, onset, poly_cnt, ctx_frames=ctx_frames, stride=args.stride)
            if X.shape[0] == 0:
                continue
            X_buf.append(X); YA_buf.append(YA); YO_buf.append(YO); YC_buf.append(YC)
            count += X.shape[0]
            if count >= args.shard_size:
                flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf)
                shard_idx += 1; count = 0
                X_buf.clear(); YA_buf.clear(); YO_buf.clear(); YC_buf.clear()

        # flush remainder
        if count > 0:
            flush(split_dir, shard_idx, X_buf, YA_buf, YO_buf, YC_buf)

    process_split("train", train_ids,     idmt_train, extra_train)
    process_split("val",   val_ids_list,  idmt_val,   extra_val)
    print(f"\nDone. Dataset written to: {out_dir}")


if __name__ == "__main__":
    main()