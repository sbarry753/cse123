"""
ingest_idmt_guitar.py
======================
Converts IDMT-SMT-Guitar (Zenodo 7544110) into the build_guitar_dataset.py
manifest format so it can be merged with your existing dataset.

Real IDMT-SMT-GUITAR_V2 layout (lowercase dataset names, annotation subfolder):

  IDMT-SMT-GUITAR_V2/
    dataset1/
      <Guitar Name>/              e.g. "Fender Strat Clean Neck SC Chords"
        annotation/
          <trackname>.xml
        <trackname>.wav           wav sits alongside the guitar-name folder,
                                  NOT inside annotation/
    dataset2/
      <Guitar Name>/
        annotation/
          <trackname>.xml
        <trackname>.wav
    dataset3/
      <piece name>/
        annotation/
          <trackname>.xml
        <trackname>.wav
    dataset4/  (polyphonic, skipped unless --include_poly)

So for every xml at:
  dataset1/<Guitar>/<annotation>/<stem>.xml
the wav is at:
  dataset1/<Guitar>/<stem>.wav

Usage
-----
# Download + ingest in one step
python ingest_idmt_guitar.py \
    --download \
    --idmt_root /data/IDMT \
    --out_dir   /data/my_dataset \
    --midi_vocab_json /data/my_dataset/metadata.json

# Ingest only (already downloaded)
python ingest_idmt_guitar.py \
    --idmt_root /data/IDMT \
    --out_dir   /data/my_dataset \
    --midi_vocab_json /data/my_dataset/metadata.json

# Skip noisy techniques, drop chords
python ingest_idmt_guitar.py ... \
    --skip_expressions "dead-notes,harmonics" \
    --skip_chords

Notes
-----
* IDMT is licensed CC BY-NC-ND 4.0 (non-commercial use only).
* Default --resample is 48000 to match build_guitar_dataset.py.
* IDMT uses 1-indexed strings; we convert to 0-indexed (0=low-E).
* Pitch names use A-based octaves matching build_guitar_dataset.py.
* Notes within --chord_thresh_ms of each other become type="chord" (string_idx=-1).
* Label JSON schema exactly matches build_guitar_dataset.py.
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
import xml.etree.ElementTree as ET
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


# ---------------------------------------------------------------------------
# Zenodo download
# ---------------------------------------------------------------------------
IDMT_ZENODO_URL = "https://zenodo.org/records/7544110/files/IDMT-SMT-GUITAR_V2.zip?download=1"
IDMT_ZIP_MD5    = "06796e08731bccffaed6ae59361486e4"
IDMT_ZIP_NAME   = "IDMT-SMT-GUITAR_V2.zip"


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
        pct  = min(100.0, downloaded * block_size / total * 100)
        done = int(pct / 2)
        bar  = "#" * done + "-" * (50 - done)
        sys.stdout.write(
            f"\r  [{bar}] {pct:.1f}%  "
            f"({downloaded * block_size / 1e6:.1f} / {total / 1e6:.1f} MB)"
        )
    sys.stdout.flush()


def download_file(url: str, dest: str, expected_md5: Optional[str] = None) -> bool:
    if os.path.isfile(dest):
        if expected_md5 and _md5(dest) == expected_md5:
            print(f"  [SKIP] {os.path.basename(dest)} already downloaded (MD5 OK)")
            return True
        elif expected_md5:
            print(f"  [WARN] {os.path.basename(dest)} MD5 mismatch — re-downloading")
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
            print(f"  [WARN] MD5 mismatch (expected {expected_md5}, got {actual}) — continuing anyway")

    shutil.move(tmp, dest)
    return True


def unzip_file(zip_path: str, dest_dir: str):
    print(f"  Extracting {os.path.basename(zip_path)} → {dest_dir}")
    print("  (1.3 GB archive — may take a minute ...)")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        for i, member in enumerate(members):
            z.extract(member, dest_dir)
            if i % 200 == 0:
                sys.stdout.write(f"\r  {i}/{len(members)} files extracted")
                sys.stdout.flush()
    sys.stdout.write(f"\r  {len(members)}/{len(members)} files extracted\n")
    sys.stdout.flush()


def download_idmt(dest_root: str):
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    extracted = dest_root / "IDMT-SMT-GUITAR_V2"
    if extracted.is_dir() and any((extracted / ds).is_dir() for ds in ("dataset1", "dataset2", "dataset3")):
        print(f"[DOWNLOAD] IDMT already extracted at {extracted} — skipping")
        return

    print(f"\n[DOWNLOAD] IDMT-SMT-Guitar from Zenodo 7544110 → {dest_root}")
    print( "           License: CC BY-NC-ND 4.0 (non-commercial use only)\n")

    zip_path = str(dest_root / IDMT_ZIP_NAME)
    ok = download_file(IDMT_ZENODO_URL, zip_path, expected_md5=IDMT_ZIP_MD5)
    if not ok:
        raise RuntimeError(f"Failed to download IDMT-SMT-Guitar from {IDMT_ZENODO_URL}")

    unzip_file(zip_path, str(dest_root))
    print(f"\n[DOWNLOAD] Complete. IDMT ready at {extracted}\n")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]


# ---------------------------------------------------------------------------
# Pitch helpers — A-based octave convention
# ---------------------------------------------------------------------------

def midi_to_pitch(midi: int) -> str:
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
# Resolve IDMT root (handle auto-extracted subfolder)
# ---------------------------------------------------------------------------

def resolve_idmt_root(path: str) -> Path:
    """
    The zip extracts to IDMT-SMT-GUITAR_V2/ inside the given path.
    If the user points at the parent, navigate in automatically.
    Datasets are lowercase: dataset1, dataset2, dataset3.
    """
    root = Path(path)

    # Already pointing at the right level?
    if any((root / ds).is_dir() for ds in ("dataset1", "dataset2", "dataset3",
                                            "Dataset1", "Dataset2", "Dataset3")):
        return root

    # One level down — the zip extracts a subfolder
    candidate = root / "IDMT-SMT-GUITAR_V2"
    if candidate.is_dir():
        print(f"[INFO] Auto-detected IDMT root at {candidate}")
        return candidate

    return root  # Let find_pairs raise a useful error


# ---------------------------------------------------------------------------
# Find (wav, xml) pairs
#
# Real layout:
#   dataset1/<Guitar Name>/annotation/<stem>.xml
#   dataset1/<Guitar Name>/audio/<stem>.wav
#
# The wav lives in an audio/ subfolder alongside annotation/, both under GuitarName.
# ---------------------------------------------------------------------------

def find_pairs(idmt_root: Path, datasets: List[str]) -> List[Tuple[Path, Path]]:
    pairs = []
    # Try lowercase, Title case, and original for each dataset name
    ds_dirs = []
    for ds in datasets:
        for candidate in (ds, ds.lower(), ds.title()):
            d = idmt_root / candidate
            if d.is_dir():
                ds_dirs.append(d)
                break

    if not ds_dirs:
        # Fallback: search entire tree
        ds_dirs = [idmt_root]

    seen = set()
    for ds_dir in ds_dirs:
        for xml_file in sorted(ds_dir.rglob("*.xml")):
            stem       = xml_file.stem
            ann_dir    = xml_file.parent       # .../GuitarName/annotation/
            guitar_dir = ann_dir.parent        # .../GuitarName/

            # Try candidate locations in order of likelihood
            candidates = [
                guitar_dir / "audio" / (stem + ".wav"),   # real layout
                guitar_dir / (stem + ".wav"),              # flat fallback
                xml_file.with_suffix(".wav"),              # co-located fallback
            ]
            wav_file = next((p for p in candidates if p.is_file()), None)

            if wav_file is not None and str(wav_file) not in seen:
                pairs.append((wav_file, xml_file))
                seen.add(str(wav_file))

    return pairs


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _float_tag(el: ET.Element, tag: str, default=None):
    child = el.find(tag)
    if child is None or not (child.text or "").strip():
        return default
    try:
        return float(child.text.strip())
    except ValueError:
        return default


def _int_tag(el: ET.Element, tag: str, default=None):
    v = _float_tag(el, tag)
    return default if v is None else int(round(v))


def parse_xml_annotations(xml_path: str) -> List[Dict]:
    """
    Parses IDMT-SMT-Guitar XML annotations.

    Real schema (all lowercase, offset not duration):
      <instrumentRecording>
        <transcription>
          <event>
            <pitch>40</pitch>            MIDI pitch
            <onsetSec>0.2</onsetSec>
            <offsetSec>2.5</offsetSec>
            <fretNumber>0</fretNumber>
            <stringNumber>1</stringNumber>   1-indexed
            <excitationStyle>PK</excitationStyle>
            <expressionStyle>NO</expressionStyle>
          </event>
        </transcription>
      </instrumentRecording>
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        warnings.warn(f"XML parse error {xml_path}: {e}")
        return []

    notes = []
    for ev in tree.getroot().findall(".//event"):
        onset  = _float_tag(ev, "onsetSec")
        offset = _float_tag(ev, "offsetSec")
        midi   = _int_tag(ev, "pitch")
        s1idx  = _int_tag(ev, "stringNumber")   # 1-indexed

        if onset is None or offset is None or midi is None:
            continue
        dur = offset - onset
        if dur <= 0:
            continue

        string_idx = (s1idx - 1) if (s1idx is not None and 1 <= s1idx <= 6) else -1

        notes.append({
            "onset_sec":    float(onset),
            "duration_sec": float(dur),
            "midi":         int(midi),
            "string_idx":   string_idx,
            "fret":         _int_tag(ev, "fretNumber", default=0),
            "plucking":     (ev.findtext("excitationStyle") or "").strip(),
            "expression":   (ev.findtext("expressionStyle") or "").strip(),
        })
    return notes


# ---------------------------------------------------------------------------
# Chord grouping
# ---------------------------------------------------------------------------

def group_simultaneous(notes: List[Dict], chord_thresh_ms: float = 20.0) -> List[List[Dict]]:
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n["onset_sec"])
    thresh  = chord_thresh_ms / 1000.0
    groups  = [[sorted_notes[0]]]
    for note in sorted_notes[1:]:
        if note["onset_sec"] - groups[-1][0]["onset_sec"] <= thresh:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


# ---------------------------------------------------------------------------
# Clip extraction
# ---------------------------------------------------------------------------

def extract_clips_from_groups(
    audio:           np.ndarray,
    sr:              int,
    groups:          List[List[Dict]],
    idx_map:         Dict[int, int],
    vocab_size:      int,
    pre_onset_ms:    float,
    max_dur_ms:      float,
    min_dur_ms:      float,
    skip_chords:     bool,
    skip_expression: Optional[List[str]] = None,
) -> List[Dict]:
    total     = len(audio)
    pre       = int(round(pre_onset_ms / 1000.0 * sr))
    max_s     = int(round(max_dur_ms   / 1000.0 * sr))
    min_s     = int(round(min_dur_ms   / 1000.0 * sr))
    skip_expr = set(skip_expression or [])

    clips = []
    for group in groups:
        is_chord = len(group) > 1
        if is_chord and skip_chords:
            continue

        valid = [
            n for n in group
            if n["midi"] in idx_map
            and (not skip_expr or n["expression"] not in skip_expr)
        ]
        if not valid:
            continue

        onset_sec    = valid[0]["onset_sec"]
        dur_sec      = min(max(n["duration_sec"] for n in valid), max_dur_ms / 1000.0)
        onset_sample = int(round(onset_sec * sr))
        note_samples = int(round(dur_sec * sr))

        if note_samples < min_s:
            continue

        start = max(0, onset_sample - pre)
        end   = min(total, onset_sample + note_samples)
        if end <= start:
            continue

        chunk          = audio[start:end].copy()
        onset_in_chunk = onset_sample - start

        mh = [0] * vocab_size
        for n in valid:
            mh[idx_map[n["midi"]]] = 1

        string_map = {str(i): -1 for i in range(6)}

        if is_chord:
            clip_type  = "chord"
            string_idx = -1
            for n in valid:
                si = n["string_idx"]
                if 0 <= si <= 5:
                    string_map[str(si)] = int(n["midi"])
            active_midi = sorted({int(n["midi"]) for n in valid})
        else:
            clip_type  = "single"
            string_idx = valid[0]["string_idx"]
            if 0 <= string_idx <= 5:
                string_map[str(string_idx)] = int(valid[0]["midi"])
            active_midi = [int(valid[0]["midi"])]

        clips.append({
            "multi_hot":             mh,
            "onset_sample":          onset_in_chunk,
            "num_samples":           len(chunk),
            "type":                  clip_type,
            "string_idx":            string_idx,
            "active_notes_midi":     active_midi,
            "active_notes_pitch":    [midi_to_pitch(m) for m in active_midi],
            "string_map_midi":       string_map,
            "segment_source_region": [int(start), int(end)],
            "onset_source_sample":   int(onset_sample),
            "audio":                 chunk,
            "sr":                    sr,
        })

    return clips


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest(args):
    root = resolve_idmt_root(args.idmt_root)
    out  = Path(args.out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    target_sr = int(args.resample) if args.resample else None

    datasets = ["dataset1", "dataset2", "dataset3"]
    if args.include_poly:
        datasets.append("dataset4")

    pairs = find_pairs(root, datasets)
    if not pairs:
        raise FileNotFoundError(
            f"No (wav, xml) pairs found under {root}.\n"
            f"Expected structure:\n"
            f"  dataset1/<Guitar Name>/annotation/<stem>.xml\n"
            f"  dataset1/<Guitar Name>/audio/<stem>.wav\n"
            f"Contents of {root}:\n"
            + "\n".join(f"  {p.name}" for p in sorted(root.iterdir()) if p.is_dir())
        )
    print(f"[INFO] Found {len(pairs)} (wav, xml) pairs  datasets={datasets}")

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

    skip_expr = [e.strip() for e in (args.skip_expressions or "").split(",") if e.strip()]
    if skip_expr:
        print(f"[INFO] Skipping expression styles: {skip_expr}")

    new_clips = 0
    skipped   = 0

    for wav_path, xml_path in pairs:
        notes = parse_xml_annotations(str(xml_path))
        if not notes:
            skipped += 1
            continue

        try:
            y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as e:
            warnings.warn(f"Failed to read {wav_path}: {e}")
            skipped += 1
            continue

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        if target_sr is not None and int(sr) != int(target_sr):
            if not HAS_LIBROSA:
                raise RuntimeError("Need librosa for resampling. pip install librosa")
            y  = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr))
            sr = target_sr

        groups = group_simultaneous(notes, chord_thresh_ms=args.chord_thresh_ms)
        clips  = extract_clips_from_groups(
            audio           = y,
            sr              = int(sr),
            groups          = groups,
            idx_map         = idx_map,
            vocab_size      = vocab_size,
            pre_onset_ms    = args.pre_onset_ms,
            max_dur_ms      = args.max_dur_ms,
            min_dur_ms      = args.min_dur_ms,
            skip_chords     = args.skip_chords,
            skip_expression = skip_expr,
        )

        try:
            track_stem = xml_path.relative_to(root).with_suffix("").as_posix().replace("/", "_")
        except ValueError:
            track_stem = xml_path.stem

        for i, clip in enumerate(clips):
            rel_audio = f"audio/idmt_{track_stem}_{i:04d}.wav"
            rel_label = f"labels/idmt_{track_stem}_{i:04d}.json"
            abs_audio = str(out / rel_audio)
            abs_label = str(out / rel_label)

            if rel_audio in existing_audio:
                continue

            sf.write(abs_audio, clip["audio"], int(sr), subtype="FLOAT")

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
                "sr":                    int(sr),
                "source":                "idmt_smt_guitar",
                "track_id":              track_stem,
            }
            with open(abs_label, "w") as f:
                json.dump(label, f, indent=2)

            manifest.append({"audio": rel_audio, "label": rel_label, "type": clip["type"]})
            existing_audio.add(rel_audio)
            new_clips += 1

    print(f"[INFO] New clips added : {new_clips}")
    print(f"[INFO] Pairs skipped   : {skipped}")
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
        description="Download + ingest IDMT-SMT-Guitar into build_guitar_dataset.py manifest format"
    )
    ap.add_argument("--download", action="store_true",
                    help="Download IDMT-SMT-Guitar from Zenodo before ingesting. "
                         "The zip (~1.3 GB) is cached so re-runs skip the download. "
                         "License: CC BY-NC-ND 4.0 (non-commercial use only).")
    ap.add_argument("--idmt_root", required=True,
                    help="Where IDMT-SMT-GUITAR_V2 lives (or will be downloaded to).")
    ap.add_argument("--out_dir", required=True,
                    help="Output dataset folder (created / merged into).")
    ap.add_argument("--midi_vocab_json", default="",
                    help="Path to existing metadata.json to reuse vocab + SR.")
    ap.add_argument("--resample", type=int, default=48000,
                    help="Target sample rate. Default 48000 matches build_guitar_dataset.py.")
    ap.add_argument("--pre_onset_ms",    type=float, default=5.0)
    ap.add_argument("--max_dur_ms",      type=float, default=500.0)
    ap.add_argument("--min_dur_ms",      type=float, default=20.0)
    ap.add_argument("--chord_thresh_ms", type=float, default=20.0)
    ap.add_argument("--skip_chords",     action="store_true",
                    help="Discard chord groups — keep only single-note clips.")
    ap.add_argument("--include_poly",    action="store_true",
                    help="Also process dataset4 (polyphonic chord excerpts).")
    ap.add_argument("--skip_expressions", type=str, default="",
                    help="Comma-separated expressionStyle codes to skip. "
                         "Real IDMT codes: NO=normal, VI=vibrato, BN=bend, "
                         "SL=slide, HR=harmonics, PM=palm-mute, DN=dead-note. "
                         "E.g. --skip_expressions DN,HR,SL")
    args = ap.parse_args()

    if args.download:
        download_idmt(args.idmt_root)

    ingest(args)


if __name__ == "__main__":
    main()