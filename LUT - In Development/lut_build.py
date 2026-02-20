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
     python lut_build.py build --out lut.json --k 60 --tol 15 --start 0.12 --dur 0.18 \
       E4_5=samples/E4_5.wav E4_0=samples/E4.wav

  2) Build from directory (auto label from filename):
     python lut_build.py build_dir --out lut.json --dir samples --k 60 --tol 15 --start 0.12 --dur 0.18

This builder stores:
- harmonic fingerprint (compat/debug)
- per-take log-frequency band template (main feature for distinctiveness)
- harmonic band mask (for optional off-harmonic penalties at match time)
"""

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
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


# -----------------------------
# Spectrum + harmonic fingerprint (compat/debug)
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


def fingerprint_for_note(seg: np.ndarray, fs: int, f0: float, k: int, tol_hz: float, window: str) -> np.ndarray:
    freqs, mag = spectrum_mag(seg, fs, window=window)
    nyq = float(freqs[-1])
    amps: List[float] = []
    for h in range(1, k + 1):
        fh = h * f0
        if fh >= nyq:
            break
        amps.append(band_energy_max(freqs, mag, fh, tol_hz))
    v = np.array(amps, dtype=np.float64)
    s = float(np.sum(v))
    return (v / (s + 1e-12)) if s > 1e-18 else np.zeros_like(v)


# -----------------------------
# NEW: log-band templates
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


def make_band_template(seg: np.ndarray,
                       fs: int,
                       window: str,
                       edges: np.ndarray,
                       logmag: bool,
                       agg: str,
                       f0: float,
                       mask_k: int,
                       mask_tol: float,
                       mask_tol_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    freqs, mag = spectrum_mag(seg, fs, window=window)
    y = mag.astype(np.float64)
    if logmag:
        y = np.log1p(y)

    band = pool_to_log_bands(freqs, y, edges, agg=agg)
    band /= (np.linalg.norm(band) + 1e-12)

    harm_mask = build_harmonic_band_mask(edges, f0=f0, k=mask_k, tol_value=mask_tol, tol_mode=mask_tol_mode)
    return band, harm_mask


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
    tol_hz: float
    window: str
    takes: List[Dict[str, Any]]
    source_wavs: List[str]
    analysis_start_sec: float
    analysis_dur_sec: float


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
              replace_existing: bool,
              band_fmin: float,
              band_fmax: float,
              band_n: int,
              band_agg: str,
              band_logmag: bool,
              mask_k: int,
              mask_tol: float,
              mask_tol_mode: str) -> None:
    if (not replace_existing) and Path(out_path).exists():
        lut = load_lut(out_path)
    else:
        lut = {"meta": {}, "entries": []}

    lut["meta"]["type"] = "harmonic_fingerprint_lut_v2_stringaware"
    lut["meta"]["a4_hz"] = float(a4)
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

    existing_idx: Dict[str, int] = {}
    for i, e in enumerate(lut.get("entries", [])):
        if isinstance(e, dict) and "note" in e:
            existing_idx[str(e["note"]).strip().upper()] = i

    grouped: Dict[str, List[str]] = defaultdict(list)
    for label, wav_path in labeled:
        grouped[label.strip().upper()].append(wav_path)

    for key, paths in grouped.items():
        label = key  # already upper
        # preserve original casing: labels are canonical anyway
        base_note, string_idx = split_label(label)

        midi = note_to_midi(base_note)
        f0 = midi_to_freq_hz(midi, a4_hz=a4)

        takes: List[Dict[str, Any]] = []
        sources: List[str] = []

        for wav_path in paths:
            fs, x = read_wav_mono_float(wav_path)
            x = normalize_peak(x)
            seg = pick_segment(x, fs, start, dur)
            if use_highpass:
                seg = highpass_filter(seg, fs, cutoff_hz=highpass_hz)
            seg = normalize_peak(seg)

            fp = fingerprint_for_note(seg, fs, f0=f0, k=k, tol_hz=tol_hz, window=window)
            band_vec, harm_mask = make_band_template(
                seg=seg, fs=fs, window=window, edges=edges,
                logmag=band_logmag, agg=band_agg,
                f0=f0, mask_k=mask_k, mask_tol=mask_tol, mask_tol_mode=mask_tol_mode
            )

            takes.append({
                "fingerprint": [float(v) for v in fp.tolist()],
                "band_template": [float(v) for v in band_vec.tolist()],
                "harm_band_mask": [float(v) for v in harm_mask.tolist()],
            })
            sources.append(str(wav_path))

        entry = LutEntry(
            note=f"{base_note}_{string_idx}",
            base_note=base_note,
            string_idx=int(string_idx),
            midi=midi,
            f0_hz=float(f0),
            k=int(k),
            tol_hz=float(tol_hz),
            window=str(window),
            takes=takes,
            source_wavs=sources,
            analysis_start_sec=float(start),
            analysis_dur_sec=float(dur),
        ).__dict__

        if key in existing_idx and not replace_existing:
            old = lut["entries"][existing_idx[key]]
            old.setdefault("takes", [])
            old.setdefault("source_wavs", [])
            old["takes"] = list(old["takes"]) + takes
            old["source_wavs"] = list(old["source_wavs"]) + sources
            # refresh meta fields
            old["note"] = entry["note"]
            old["base_note"] = entry["base_note"]
            old["string_idx"] = entry["string_idx"]
            old["midi"] = entry["midi"]
            old["f0_hz"] = entry["f0_hz"]
            old["k"] = entry["k"]
            old["tol_hz"] = entry["tol_hz"]
            old["window"] = entry["window"]
            old["analysis_start_sec"] = entry["analysis_start_sec"]
            old["analysis_dur_sec"] = entry["analysis_dur_sec"]
            lut["entries"][existing_idx[key]] = old
            print(f"[build] merged  {entry['note']}: added_takes={len(sources)} total_takes={len(old['takes'])}")
        elif key in existing_idx and replace_existing:
            lut["entries"][existing_idx[key]] = entry
            print(f"[build] replaced {entry['note']}: takes={len(sources)}")
        else:
            lut["entries"].append(entry)
            print(f"[build] added   {entry['note']}: takes={len(sources)}")

    save_lut(out_path, lut)
    print(f"Saved LUT -> {out_path} with {len(lut['entries'])} class entries")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build", help="Build LUT from labeled recordings (LABEL=wav_path)")
    ap_build.add_argument("--out", required=True)
    ap_build.add_argument("--k", type=int, default=60)
    ap_build.add_argument("--tol", type=float, default=15.0)
    ap_build.add_argument("--start", type=float, default=0.10)
    ap_build.add_argument("--dur", type=float, default=0.20)
    ap_build.add_argument("--a4", type=float, default=440.0)
    ap_build.add_argument("--window", type=str, default="hann")
    ap_build.add_argument("--highpass", action="store_true")
    ap_build.add_argument("--highpass_hz", type=float, default=80.0)
    ap_build.add_argument("--replace_existing", action="store_true")
    ap_build.add_argument("labeled", nargs="+")  # LABEL=path

    # band template params (defaults tuned for string separation)
    ap_build.add_argument("--band_fmin", type=float, default=40.0)
    ap_build.add_argument("--band_fmax", type=float, default=8000.0)
    ap_build.add_argument("--band_n", type=int, default=480)
    ap_build.add_argument("--band_agg", type=str, default="max", choices=["max", "mean"])
    ap_build.add_argument("--band_logmag", action="store_true")
    ap_build.add_argument("--mask_k", type=int, default=24)
    ap_build.add_argument("--mask_tol", type=float, default=20.0)
    ap_build.add_argument("--mask_tol_mode", type=str, default="cents", choices=["hz", "cents"])

    ap_build_dir = sub.add_parser("build_dir", help="Build LUT from directory (auto label from filename)")
    ap_build_dir.add_argument("--out", required=True)
    ap_build_dir.add_argument("--dir", required=True)
    ap_build_dir.add_argument("--recursive", action="store_true")
    ap_build_dir.add_argument("--k", type=int, default=60)
    ap_build_dir.add_argument("--tol", type=float, default=15.0)
    ap_build_dir.add_argument("--start", type=float, default=0.10)
    ap_build_dir.add_argument("--dur", type=float, default=0.20)
    ap_build_dir.add_argument("--a4", type=float, default=440.0)
    ap_build_dir.add_argument("--window", type=str, default="hann")
    ap_build_dir.add_argument("--highpass", action="store_true")
    ap_build_dir.add_argument("--highpass_hz", type=float, default=80.0)
    ap_build_dir.add_argument("--replace_existing", action="store_true")

    ap_build_dir.add_argument("--band_fmin", type=float, default=40.0)
    ap_build_dir.add_argument("--band_fmax", type=float, default=8000.0)
    ap_build_dir.add_argument("--band_n", type=int, default=480)
    ap_build_dir.add_argument("--band_agg", type=str, default="max", choices=["max", "mean"])
    ap_build_dir.add_argument("--band_logmag", action="store_true")
    ap_build_dir.add_argument("--mask_k", type=int, default=24)
    ap_build_dir.add_argument("--mask_tol", type=float, default=20.0)
    ap_build_dir.add_argument("--mask_tol_mode", type=str, default="cents", choices=["hz", "cents"])

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


if __name__ == "__main__":
    main()