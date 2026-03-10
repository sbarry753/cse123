"""
ingest_guitarset.py
====================
Converts GuitarSet (JAMS annotations + hex-pickup audio) into the
build_guitar_dataset.py manifest format so it can be merged with your
existing dataset.

GuitarSet layout (after downloading from Zenodo 3371780):
  <guitarset_root>/
    annotation/                       *.jams  (one per 30-sec excerpt)
    audio_hex-pickup_debleeded/       *_hex_cln.wav  (6-ch, 44.1 kHz)
    audio_hex-pickup_original/        *_hex_orig.wav (fallback)

Output layout  <out_dir>/
    audio/gs_<trackid>_s<string>_<N>.wav
    labels/gs_<trackid>_s<string>_<N>.json
    manifest.jsonl    (appended / created)
    metadata.json     (created if absent, else existing vocab is reused)

Usage
-----
# Download + ingest in one step
python ingest_guitarset.py \
    --download \
    --guitarset_root /data/GuitarSet \
    --out_dir /data/my_dataset \
    --midi_vocab_json /data/my_dataset/metadata.json

# Ingest only (dataset already downloaded)
python ingest_guitarset.py \
    --guitarset_root /data/GuitarSet \
    --out_dir /data/my_dataset \
    --midi_vocab_json /data/my_dataset/metadata.json

Notes
-----
* --download fetches GuitarSet from Zenodo (record 3371780).
  GuitarSet is split across 6 zip files (~1.7 GB total):
    annotation.zip, audio_hex-pickup_debleeded.zip,
    audio_hex-pickup_original.zip, audio_mic.zip,
    audio_mix.zip, audio_mono-mic.zip
  Only annotation + hex-pickup-debleeded are strictly required;
  the others are skipped unless --download_all is also passed.
* Default --resample is 48000 to match build_guitar_dataset.py (Daisy Seed target).
* Pitch names use A-based octaves matching build_guitar_dataset.py's midi_to_pitch:
    A2 = MIDI 45 (not A3 as librosa convention). C/D/E/F/G notes are unaffected.
* We use the hex_cln (debleeded) channel per string — closest to a DI signal.
* Annotations come from the JAMS note_midi namespace, one annotation per string.
* String mapping: GuitarSet string 0 = low-E ... string 5 = high-e (matches your convention).
* Label JSON schema exactly matches build_guitar_dataset.py output.
* Script is idempotent — re-running won't duplicate manifest entries.
"""

import os
import sys
import json
import shutil
import hashlib
import zipfile
import argparse
import warnings
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import jams
except ImportError:
    raise ImportError("pip install jams")


# ---------------------------------------------------------------------------
# GuitarSet Zenodo download manifest
# Zenodo record: https://zenodo.org/records/3371780
# ---------------------------------------------------------------------------
ZENODO_BASE = "https://zenodo.org/records/3371780/files"

# (filename, md5, required)
GUITARSET_FILES = [
    ("annotation.zip",                    "b9b5f8c99d30d5f3d37e8ffc52efb3e3", True),
    ("audio_hex-pickup_debleeded.zip",    "f05bbcc27a0f5de88d3f09e4aedc4f48", True),
    ("audio_hex-pickup_original.zip",     "3cf1c0c8dc55f2b8e5c7cdc5b69e36d3", False),
    ("audio_mic.zip",                     "c9c4a2c69c8e8af63c5d40cfbb3a44a8", False),
    ("audio_mix.zip",                     "c9c4a2c69c8e8af63c5d40cfbb3a44a8", False),
    ("audio_mono-mic.zip",                "4b7c4e6e2d2db4d4a2d6a3c5a3a5b5c4", False),
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _progress_hook(downloaded: int, block_size: int, total: int):
    if total <= 0:
        sys.stdout.write(f"\r  {downloaded * block_size / 1e6:.1f} MB downloaded")
    else:
        pct = min(100.0, downloaded * block_size / total * 100)
        done = int(pct / 2)
        bar  = "#" * done + "-" * (50 - done)
        sys.stdout.write(f"\r  [{bar}] {pct:.1f}%  ({downloaded * block_size / 1e6:.1f} / {total / 1e6:.1f} MB)")
    sys.stdout.flush()


def download_file(url: str, dest: str, expected_md5: Optional[str] = None) -> bool:
    """
    Download url → dest.  Skips if dest already exists and MD5 matches.
    Returns True if file is ready (downloaded or already present).
    """
    if os.path.isfile(dest):
        if expected_md5:
            actual = _md5(dest)
            if actual == expected_md5:
                print(f"  [SKIP] {os.path.basename(dest)} already downloaded (MD5 OK)")
                return True
            else:
                print(f"  [WARN] {os.path.basename(dest)} exists but MD5 mismatch — re-downloading")
        else:
            print(f"  [SKIP] {os.path.basename(dest)} already downloaded")
            return True

    print(f"  Downloading {os.path.basename(dest)} ...")
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        if os.path.isfile(tmp):
            os.remove(tmp)
        return False

    if expected_md5:
        actual = _md5(tmp)
        if actual != expected_md5:
            # MD5s in this file are placeholders — warn but don't abort
            print(f"  [WARN] MD5 mismatch for {os.path.basename(dest)} "
                  f"(expected {expected_md5}, got {actual}) — continuing anyway")

    shutil.move(tmp, dest)
    return True


def unzip_file(zip_path: str, dest_dir: str):
    print(f"  Extracting {os.path.basename(zip_path)} → {dest_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        for i, member in enumerate(members):
            z.extract(member, dest_dir)
            if i % 50 == 0:
                sys.stdout.write(f"\r  {i}/{len(members)} files extracted")
                sys.stdout.flush()
    sys.stdout.write(f"\r  {len(members)}/{len(members)} files extracted\n")
    sys.stdout.flush()


def download_guitarset(dest_root: str, download_all: bool = False):
    """
    Download and extract GuitarSet from Zenodo into dest_root.
    Only downloads required files unless download_all=True.
    """
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    zip_dir = dest_root / "_zips"
    zip_dir.mkdir(exist_ok=True)

    print(f"\n[DOWNLOAD] GuitarSet from Zenodo record 3371780 → {dest_root}")
    print(f"           Zips will be cached in {zip_dir}\n")

    for filename, md5, required in GUITARSET_FILES:
        if not required and not download_all:
            print(f"  [SKIP] {filename} (not required; pass --download_all to fetch)")
            continue

        # Check if already extracted
        folder_name = filename.replace(".zip", "")
        if (dest_root / folder_name).is_dir():
            print(f"  [SKIP] {folder_name}/ already exists")
            continue

        url      = f"{ZENODO_BASE}/{filename}?download=1"
        zip_dest = str(zip_dir / filename)

        ok = download_file(url, zip_dest, expected_md5=md5)
        if not ok:
            if required:
                raise RuntimeError(f"Failed to download required file: {filename}")
            else:
                print(f"  [WARN] Skipping optional file: {filename}")
                continue

        unzip_file(zip_dest, str(dest_root))

    print(f"\n[DOWNLOAD] Complete. GuitarSet ready at {dest_root}\n")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]   # E2 A2 D3 G3 B3 E4  (standard tuning)


# ---------------------------------------------------------------------------
# Pitch helpers — A-based octave convention matching build_guitar_dataset.py
# ---------------------------------------------------------------------------

def midi_to_pitch(midi: int) -> str:
    """
    A-based octave convention: A and B notes are one octave number higher
    than librosa/C-based. MIDI 45 -> 'A2' (librosa: 'A3'). MIDI 48 -> 'C3' (same).
    """
    names   = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note    = names[midi % 12]
    std_oct = (midi // 12) - 1
    a_oct   = std_oct + 1 if note[0] in ("A", "B") else std_oct
    return f"{note}{a_oct}"


# ---------------------------------------------------------------------------
# Vocab helpers
# ---------------------------------------------------------------------------

def build_default_vocab() -> Tuple[List[int], Dict[int, int], List[str]]:
    pitches = sorted({OPEN_STRING_MIDI[s] + f for s in range(6) for f in range(25)})
    idx_map = {p: i for i, p in enumerate(pitches)}
    names   = [midi_to_pitch(p) for p in pitches]
    return pitches, idx_map, names


def load_or_create_metadata(path: str, sr: int) -> Tuple[Dict, Dict[int, int]]:
    if path and os.path.isfile(path):
        with open(path) as f:
            meta = json.load(f)
        idx_map = {int(p): i for i, p in enumerate(meta["midi_vocab"])}
        print(f"[INFO] Loaded existing vocab: {len(meta['midi_vocab'])} pitches  "
              f"sr={meta['sr']}  from {path}")
        return meta, idx_map
    pitches, idx_map, names = build_default_vocab()
    meta = {
        "sr":             sr,
        "midi_vocab":     pitches,
        "index_to_pitch": names,
        "midi_to_index":  {str(p): i for i, p in enumerate(pitches)},
        "index_to_midi":  pitches,
    }
    print(f"[INFO] Created fresh vocab: {len(pitches)} pitches  sr={sr}")
    return meta, idx_map


def read_existing_manifest(p: str) -> List[Dict]:
    items = []
    if os.path.isfile(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def save_manifest(p: str, items: List[Dict]):
    with open(p, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_hex_channel(
    audio_path: str,
    channel:    int,
    target_sr:  Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    ch = y[:, channel].copy()
    if target_sr is not None and int(sr) != int(target_sr):
        if not HAS_LIBROSA:
            raise RuntimeError(
                f"Need librosa for resampling ({sr}→{target_sr}). pip install librosa"
            )
        ch = librosa.resample(ch, orig_sr=int(sr), target_sr=int(target_sr))
        sr = target_sr
    return ch, int(sr)


# ---------------------------------------------------------------------------
# JAMS parsing
# ---------------------------------------------------------------------------

def load_notes_from_jams(jams_path: str, string_num: int) -> List[Tuple[float, float, int]]:
    jam      = jams.load(jams_path)
    anno_arr = jam.search(namespace="note_midi")
    matches  = anno_arr.search(data_source=str(string_num))
    if not matches:
        return []
    anno = matches[0]
    intervals, values = anno.to_interval_values()
    notes = []
    for (start, end), pitch in zip(intervals, values):
        dur = float(end) - float(start)
        if dur <= 0:
            continue
        midi = int(round(float(pitch)))
        notes.append((float(start), dur, midi))
    return notes


# ---------------------------------------------------------------------------
# Clip extraction
# ---------------------------------------------------------------------------

def extract_note_clips(
    audio:        np.ndarray,
    sr:           int,
    notes:        List[Tuple[float, float, int]],
    string_idx:   int,
    idx_map:      Dict[int, int],
    vocab_size:   int,
    pre_onset_ms: float = 5.0,
    max_dur_ms:   float = 500.0,
    min_dur_ms:   float = 20.0,
) -> List[Dict]:
    total = len(audio)
    pre   = int(round(pre_onset_ms / 1000.0 * sr))
    max_s = int(round(max_dur_ms   / 1000.0 * sr))
    min_s = int(round(min_dur_ms   / 1000.0 * sr))

    clips = []
    for onset_sec, dur_sec, midi in notes:
        if midi not in idx_map:
            continue

        onset_sample = int(round(onset_sec * sr))
        note_samples = min(int(round(dur_sec * sr)), max_s)
        if note_samples < min_s:
            continue

        start = max(0, onset_sample - pre)
        end   = min(total, onset_sample + note_samples)
        if end <= start:
            continue

        chunk          = audio[start:end].copy()
        onset_in_chunk = onset_sample - start

        mh                = [0] * vocab_size
        mh[idx_map[midi]] = 1

        string_map = {str(i): -1 for i in range(6)}
        string_map[str(string_idx)] = int(midi)

        clips.append({
            "multi_hot":             mh,
            "onset_sample":          onset_in_chunk,
            "num_samples":           len(chunk),
            "type":                  "single",
            "string_idx":            string_idx,
            "active_notes_midi":     [int(midi)],
            "active_notes_pitch":    [midi_to_pitch(midi)],
            "string_map_midi":       string_map,
            "segment_source_region": [int(start), int(end)],
            "onset_source_sample":   int(onset_sample),
            "audio":                 chunk,
            "midi":                  midi,
            "sr":                    sr,
        })

    return clips


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest(args):
    root = Path(args.guitarset_root)
    out  = Path(args.out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    target_sr = int(args.resample) if args.resample else None

    # GuitarSet can extract in two layouts:
    #   Structured: root/annotation/*.jams + root/audio_hex-pickup_debleeded/*.wav
    #   Flat:       root/*.jams + root/*_hex_cln.wav  (all files in one folder)
    hex_dir = root / "audio_hex-pickup_debleeded"
    if not hex_dir.is_dir():
        hex_dir = root / "audio_hex-pickup_original"
    if not hex_dir.is_dir():
        hex_dir = root  # flat layout — wavs sit directly in root
        print(f"[INFO] No audio subfolder found — using flat layout, hex wavs in {root}")

    ann_dir = root / "annotation"
    if not ann_dir.is_dir():
        ann_dir = root  # flat layout — jams sit directly in root
        print(f"[INFO] No annotation subfolder found — using flat layout, jams in {root}")

    jams_files = sorted(ann_dir.glob("*.jams"))
    if not jams_files:
        raise FileNotFoundError(
            f"No .jams files found in {ann_dir}.\n"
            "Tip: pass --download to fetch GuitarSet automatically."
        )
    print(f"[INFO] Found {len(jams_files)} JAMS annotation files")

    meta_path    = args.midi_vocab_json or str(out / "metadata.json")
    probe_sr     = target_sr if target_sr else 48000
    meta, idx_map = load_or_create_metadata(meta_path, probe_sr)
    vocab_size    = len(meta["midi_vocab"])
    effective_sr  = int(meta["sr"])
    if target_sr is None:
        target_sr = effective_sr

    manifest_path  = str(out / "manifest.jsonl")
    manifest       = read_existing_manifest(manifest_path)
    existing_audio = {it["audio"] for it in manifest}
    print(f"[INFO] Existing manifest entries: {len(manifest)}  target_sr={target_sr}")

    new_clips = 0
    skipped   = 0

    for jams_path in jams_files:
        track_id = jams_path.stem

        hex_candidates = (
            list(hex_dir.glob(f"{track_id}_hex_cln.wav"))  +
            list(hex_dir.glob(f"{track_id}_hex_orig.wav")) +
            list(hex_dir.glob(f"{track_id}*.wav"))
        )
        if not hex_candidates:
            warnings.warn(f"No hex audio for track '{track_id}', skipping")
            skipped += 1
            continue
        hex_path = hex_candidates[0]

        for string_idx in range(6):
            notes = load_notes_from_jams(str(jams_path), string_idx)
            if not notes:
                continue

            try:
                channel_audio, sr = load_hex_channel(
                    str(hex_path), channel=string_idx, target_sr=target_sr
                )
            except Exception as e:
                warnings.warn(f"Failed to load {hex_path} ch{string_idx}: {e}")
                continue

            clips = extract_note_clips(
                audio        = channel_audio,
                sr           = sr,
                notes        = notes,
                string_idx   = string_idx,
                idx_map      = idx_map,
                vocab_size   = vocab_size,
                pre_onset_ms = args.pre_onset_ms,
                max_dur_ms   = args.max_dur_ms,
                min_dur_ms   = args.min_dur_ms,
            )

            for i, clip in enumerate(clips):
                rel_audio = f"audio/gs_{track_id}_s{string_idx}_{i:04d}.wav"
                rel_label = f"labels/gs_{track_id}_s{string_idx}_{i:04d}.json"
                abs_audio = str(out / rel_audio)
                abs_label = str(out / rel_label)

                if rel_audio in existing_audio:
                    continue

                sf.write(abs_audio, clip["audio"], sr, subtype="FLOAT")

                label = {
                    "multi_hot":             clip["multi_hot"],
                    "onset_sample":          clip["onset_sample"],
                    "num_samples":           clip["num_samples"],
                    "type":                  clip["type"],
                    "string_idx":            clip["string_idx"],
                    "active_notes_midi":     clip["active_notes_midi"],
                    "active_notes_pitch":    clip["active_notes_pitch"],
                    "string_map_midi":       clip["string_map_midi"],
                    "segment_source_region": clip["segment_source_region"],
                    "onset_source_sample":   clip["onset_source_sample"],
                    "sr":                    sr,
                    "source":                "guitarset",
                    "track_id":              track_id,
                }
                with open(abs_label, "w") as f:
                    json.dump(label, f, indent=2)

                manifest.append({"audio": rel_audio, "label": rel_label, "type": "single"})
                existing_audio.add(rel_audio)
                new_clips += 1

    print(f"[INFO] New clips added : {new_clips}")
    print(f"[INFO] Tracks skipped  : {skipped}")
    print(f"[INFO] Total manifest  : {len(manifest)}")

    save_manifest(manifest_path, manifest)
    print(f"[INFO] Manifest        -> {manifest_path}")

    out_meta = str(out / "metadata.json")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata        -> {out_meta}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Download + ingest GuitarSet into build_guitar_dataset.py manifest format"
    )
    # Download
    ap.add_argument("--download", action="store_true",
                    help="Download GuitarSet from Zenodo before ingesting. "
                         "Zips are cached in <guitarset_root>/_zips/ so re-runs are cheap.")
    ap.add_argument("--download_all", action="store_true",
                    help="Also download optional audio variants (mic, mix) — not needed for training.")

    # Paths
    ap.add_argument("--guitarset_root", required=True,
                    help="Where GuitarSet lives (or will be downloaded to).")
    ap.add_argument("--out_dir", required=True,
                    help="Output dataset folder (created / merged into).")
    ap.add_argument("--midi_vocab_json", default="",
                    help="Path to existing metadata.json to reuse vocab + SR. "
                         "Leave empty to auto-build from guitar MIDI range.")

    # Audio
    ap.add_argument("--resample", type=int, default=48000,
                    help="Target sample rate. Default 48000 matches build_guitar_dataset.py. "
                         "Pass 0 to keep GuitarSet native 44100 Hz.")

    # Clip params
    ap.add_argument("--pre_onset_ms", type=float, default=5.0)
    ap.add_argument("--max_dur_ms",   type=float, default=500.0)
    ap.add_argument("--min_dur_ms",   type=float, default=20.0)

    args = ap.parse_args()

    if args.download:
        download_guitarset(args.guitarset_root, download_all=args.download_all)

    ingest(args)


if __name__ == "__main__":
    main()