import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import librosa
import soundfile as sf


# ----------------------------
# Config / constants
# ----------------------------
TRANSIENT_MS = 5.0
PRE_ROLL_MS = 10.0         # keep a little audio before the transient
ONSET_SEARCH_MAX_MS = 80.0 # for scale segments, refine onset within first ~80ms
SILENCE_RMS_DB = -60.0     # used if you later add end-of-note trimming logic (currently not trimming sustain)
DEFAULT_SR = 48000


# ----------------------------
# Helpers
# ----------------------------
NOTE_RE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+)$')

def normalize_pitch_str(p: str) -> str:
    """Normalize pitch tokens to librosa-friendly note names: e.g., 'f#4' -> 'F#4'."""
    p = p.strip()
    if p == "0":
        return p
    m = NOTE_RE.match(p)
    if not m:
        raise ValueError(f"Bad pitch token: {p}")
    letter, accidental, octave = m.group(1).upper(), m.group(2), m.group(3)
    return f"{letter}{accidental}{octave}"

def pitch_to_midi(pitch: str) -> int:
    pitch = normalize_pitch_str(pitch)
    return int(librosa.note_to_midi(pitch))

def midi_to_pitch(midi: int) -> str:
    return librosa.midi_to_note(midi, octave=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_audio_mono(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """Load audio, convert to mono float32, resample to target_sr."""
    y, sr = librosa.load(path, sr=None, mono=True)  # preserves original sr
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    y = y.astype(np.float32)
    return y, sr

def detect_onset_sample(y: np.ndarray, sr: int, search_limit_ms: Optional[float] = None) -> int:
    """
    Detect the first onset sample index using librosa onset detection.
    If search_limit_ms is provided, only search within that initial window.
    """
    if len(y) < int(0.01 * sr):  # too short
        return 0

    if search_limit_ms is not None:
        max_samp = min(len(y), int((search_limit_ms / 1000.0) * sr))
        y_search = y[:max_samp]
    else:
        y_search = y

    # onset envelope
    hop = 128  # small hop for better timing resolution
    oenv = librosa.onset.onset_strength(y=y_search, sr=sr, hop_length=hop)

    # onset frames
    frames = librosa.onset.onset_detect(
        onset_envelope=oenv,
        sr=sr,
        hop_length=hop,
        units="frames",
        backtrack=True,
        pre_max=3, post_max=3, pre_avg=3, post_avg=3,
        delta=0.2,
        wait=0
    )

    if frames is None or len(frames) == 0:
        return 0

    onset_frame = int(frames[0])
    onset_sample = int(librosa.frames_to_samples(onset_frame, hop_length=hop))

    # Clamp
    onset_sample = int(np.clip(onset_sample, 0, max(0, len(y_search) - 1)))
    return onset_sample

def infer_take_id(path: str) -> str:
    """
    Infer take_id from folders like single_1, string_2, etc.
    If no suffix, return "0".
    """
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        m = re.match(r'.*_(\d+)$', p)
        if m:
            return m.group(1)
    return "0"

def relative_type_from_path(path: str) -> str:
    """
    Decide sample type from folder names.
    """
    norm = os.path.normpath(path).split(os.sep)
    if "single" in norm or any(x.startswith("single_") for x in norm):
        return "single"
    if "string" in norm or any(x.startswith("string_") for x in norm):
        return "scale"
    if "chords" in norm:
        return "chord"
    # fallback: guess by filename patterns
    return "unknown"


# ----------------------------
# Parsers
# ----------------------------
def parse_single_filename(stem: str) -> Tuple[str, int]:
    """
    Example: F4_5  => pitch='F4', string=5
    """
    m = re.match(r'^(.+?)_(\d+)$', stem)
    if not m:
        raise ValueError(f"Single note filename must be like F4_5.wav, got: {stem}")
    pitch = normalize_pitch_str(m.group(1))
    string_idx = int(m.group(2))
    return pitch, string_idx

def parse_scale_filename(stem: str) -> Tuple[str, str]:
    """
    Example: E2-E4 => returns ('E2','E4') but we only use the start pitch for 24 chromatic steps.
    """
    m = re.match(r'^(.+?)-(.+?)$', stem)
    if not m:
        raise ValueError(f"Scale filename must be like E2-E4.wav, got: {stem}")
    start_pitch = normalize_pitch_str(m.group(1))
    end_pitch = normalize_pitch_str(m.group(2))
    return start_pitch, end_pitch

def parse_chord_filename(stem: str) -> Dict[int, Optional[str]]:
    """
    Example:
      E2-A3_1-D3_2-G3_3-B4_4-E4_5
    Rules:
      - tokens separated by '-'
      - each token:
          "0" or "0_#" => no note on that string
          "Pitch_String" => that pitch on that string
          "Pitch" (no _) => assume string 0
    Returns:
      dict {string_idx: pitch_str or None}
    """
    tokens = stem.split('-')
    string_map: Dict[int, Optional[str]] = {}

    for t in tokens:
        t = t.strip()
        if t == "":
            continue

        if "_" in t:
            left, right = t.split("_", 1)
            if left == "0":
                # explicit "no note" on that string
                try:
                    sidx = int(right)
                except:
                    raise ValueError(f"Bad chord token: {t}")
                string_map[sidx] = None
            else:
                pitch = normalize_pitch_str(left)
                sidx = int(right)
                string_map[sidx] = pitch
        else:
            # no _, could be "0" (meaning no note on string 0) or "Pitch" meaning string 0
            if t == "0":
                string_map[0] = None
            else:
                pitch = normalize_pitch_str(t)
                string_map[0] = pitch

    # Ensure 0..5 present? (optional; we'll fill missing later)
    return string_map


# ----------------------------
# Dataset builder
# ----------------------------
@dataclass
class BuiltSample:
    audio_path: str
    label_path: str
    notes_midi: List[int]

def build_vocab(collected_midis: List[int]) -> Dict[str, Any]:
    uniq = sorted(set(collected_midis))
    midi_to_index = {m: i for i, m in enumerate(uniq)}
    return {
        "sr": DEFAULT_SR,
        "transient_ms": TRANSIENT_MS,
        "pre_roll_ms": PRE_ROLL_MS,
        "midi_vocab": uniq,
        "midi_to_index": {str(k): v for k, v in midi_to_index.items()},
        "index_to_midi": [int(m) for m in uniq],
        "index_to_pitch": [midi_to_pitch(m) for m in uniq],
    }

def encode_multi_hot(notes_midi: List[int], midi_to_index: Dict[int, int], vocab_size: int) -> List[int]:
    v = np.zeros((vocab_size,), dtype=np.int32)
    for m in notes_midi:
        if m in midi_to_index:
            v[midi_to_index[m]] = 1
    return v.tolist()

def write_wav(path: str, y: np.ndarray, sr: int):
    # Daisy-friendly: float32 PCM
    sf.write(path, y, sr, subtype="FLOAT")

def align_and_label_transient(
    y: np.ndarray,
    sr: int,
    onset_sample: int,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Trim so onset is ~PRE_ROLL_MS into the audio.
    Return:
      y_out, onset_out, transient_start, transient_end (all in samples relative to y_out)
    """
    pre = int((PRE_ROLL_MS / 1000.0) * sr)
    tlen = int((TRANSIENT_MS / 1000.0) * sr)

    trim_start = max(0, onset_sample - pre)
    y_out = y[trim_start:].copy()

    onset_out = onset_sample - trim_start
    transient_start = onset_out
    transient_end = min(len(y_out), transient_start + tlen)

    return y_out, onset_out, transient_start, transient_end

def iter_wavs(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".wav"):
                yield os.path.join(dirpath, fn)

def build_dataset(raw_root: str, out_root: str, sr: int = DEFAULT_SR):
    audio_out = os.path.join(out_root, "audio")
    labels_out = os.path.join(out_root, "labels")
    ensure_dir(audio_out)
    ensure_dir(labels_out)

    # Pass 1: discover files & collect all MIDI notes for a global vocab
    discovered = []
    all_midis: List[int] = []

    for wav_path in iter_wavs(raw_root):
        rel_type = relative_type_from_path(wav_path)
        stem = os.path.splitext(os.path.basename(wav_path))[0]

        try:
            if rel_type == "single":
                pitch, _sidx = parse_single_filename(stem)
                all_midis.append(pitch_to_midi(pitch))
                discovered.append((wav_path, rel_type, None))
            elif rel_type == "scale":
                start_pitch, _end_pitch = parse_scale_filename(stem)
                start_midi = pitch_to_midi(start_pitch)
                for i in range(24):
                    all_midis.append(start_midi + i)
                discovered.append((wav_path, rel_type, None))
            elif rel_type == "chord":
                smap = parse_chord_filename(stem)
                for _s, p in smap.items():
                    if p is not None:
                        all_midis.append(pitch_to_midi(p))
                discovered.append((wav_path, rel_type, None))
            else:
                # try best-effort classification by pattern
                if "_" in stem and re.search(r'_[0-5]$', stem):
                    pitch, _ = parse_single_filename(stem)
                    all_midis.append(pitch_to_midi(pitch))
                    discovered.append((wav_path, "single", None))
                elif "-" in stem and re.match(r'.+?-.+?', stem):
                    # could be scale or chord. chord has many '-' tokens and '_' tokens
                    if "_" in stem:
                        smap = parse_chord_filename(stem)
                        for _s, p in smap.items():
                            if p is not None:
                                all_midis.append(pitch_to_midi(p))
                        discovered.append((wav_path, "chord", None))
                    else:
                        start_pitch, _ = parse_scale_filename(stem)
                        start_midi = pitch_to_midi(start_pitch)
                        for i in range(24):
                            all_midis.append(start_midi + i)
                        discovered.append((wav_path, "scale", None))
                else:
                    print(f"[WARN] Skipping unrecognized file: {wav_path}")

        except Exception as e:
            print(f"[WARN] Failed to parse {wav_path}: {e}")

    if len(all_midis) == 0:
        raise RuntimeError("No parsable notes found. Check your folder structure/filenames.")

    meta = build_vocab(all_midis)
    vocab_midis = meta["midi_vocab"]
    midi_to_index = {int(k): int(v) for k, v in meta["midi_to_index"].items()}
    vocab_size = len(vocab_midis)

    # Write metadata
    ensure_dir(out_root)
    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Pass 2: build samples
    manifest_path = os.path.join(out_root, "manifest.jsonl")
    sample_idx = 0

    with open(manifest_path, "w") as mf:
        for wav_path, rel_type, _ in discovered:
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            take_id = infer_take_id(wav_path)

            if rel_type in ("single", "chord"):
                y, _sr = read_audio_mono(wav_path, sr)

                if rel_type == "single":
                    pitch, sidx = parse_single_filename(stem)
                    notes_midi = [pitch_to_midi(pitch)]
                    string_map = {sidx: notes_midi[0]}
                else:
                    smap = parse_chord_filename(stem)
                    notes_midi = []
                    string_map = {}
                    for sidx in range(6):
                        p = smap.get(sidx, None)
                        if p is None:
                            # if missing in filename, treat as "unknown" not "no note"
                            # but for training, it's usually safer to say "no note" only when explicit.
                            # We'll mark missing as -1 (unknown) and explicit None as -1 too.
                            string_map[sidx] = -1
                        else:
                            if p is None:
                                string_map[sidx] = -1
                            else:
                                m = pitch_to_midi(p)
                                notes_midi.append(m)
                                string_map[sidx] = m

                onset = detect_onset_sample(y, sr, search_limit_ms=None)
                y_out, onset_out, t_start, t_end = align_and_label_transient(y, sr, onset)

                # Write audio
                out_name = f"sample_{sample_idx:06d}.wav"
                out_wav = os.path.join(audio_out, out_name)
                write_wav(out_wav, y_out, sr)

                multi_hot = encode_multi_hot(notes_midi, midi_to_index, vocab_size)

                label = {
                    "id": f"{sample_idx:06d}",
                    "type": rel_type,
                    "source_path": os.path.relpath(wav_path, raw_root),
                    "take_id": take_id,
                    "sr": sr,

                    "active_notes_midi": notes_midi,
                    "active_notes_pitch": [midi_to_pitch(m) for m in notes_midi],
                    "multi_hot": multi_hot,

                    "string_map_midi": {str(k): int(v) for k, v in string_map.items()},

                    # transient/sustain regions in samples (relative to this saved clip)
                    "transient_window_start": int(t_start),
                    "transient_window_end": int(t_end),
                    "sustain_region": [int(t_end), int(len(y_out))],

                    # for debugging alignment
                    "onset_sample": int(onset_out),
                    "num_samples": int(len(y_out)),
                }

                out_json = os.path.join(labels_out, f"sample_{sample_idx:06d}.json")
                with open(out_json, "w") as f:
                    json.dump(label, f, indent=2)

                mf.write(json.dumps({
                    "id": label["id"],
                    "audio": os.path.relpath(out_wav, out_root),
                    "label": os.path.relpath(out_json, out_root),
                    "type": rel_type,
                }) + "\n")

                sample_idx += 1

            elif rel_type in ("scale",):
                # Timed scale recording: 25s, 60bpm, 1 note per second, first second silent, 24 notes
                y, _sr = read_audio_mono(wav_path, sr)
                start_pitch, _end_pitch = parse_scale_filename(stem)
                start_midi = pitch_to_midi(start_pitch)

                one_sec = sr
                expected_len = 25 * one_sec
                if len(y) < (24 + 1) * one_sec:
                    print(f"[WARN] Scale file shorter than expected (~25s). Using available length: {wav_path}")

                # segments: ignore first second (index 0), take next 24
                for i in range(24):
                    seg_start = (i + 1) * one_sec
                    seg_end = min((i + 2) * one_sec, len(y))
                    if seg_start >= len(y) or seg_end <= seg_start:
                        break

                    seg = y[seg_start:seg_end].copy()
                    midi_note = start_midi + i
                    notes_midi = [midi_note]

                    # For scale segments, onset should be near the segment start; detect within first ONSET_SEARCH_MAX_MS
                    onset_local = detect_onset_sample(seg, sr, search_limit_ms=ONSET_SEARCH_MAX_MS)
                    seg_out, onset_out, t_start, t_end = align_and_label_transient(seg, sr, onset_local)

                    out_name = f"sample_{sample_idx:06d}.wav"
                    out_wav = os.path.join(audio_out, out_name)
                    write_wav(out_wav, seg_out, sr)

                    multi_hot = encode_multi_hot(notes_midi, midi_to_index, vocab_size)

                    label = {
                        "id": f"{sample_idx:06d}",
                        "type": "scale_segment",
                        "source_path": os.path.relpath(wav_path, raw_root),
                        "take_id": take_id,
                        "sr": sr,

                        "scale_segment_index": i,
                        "segment_time_in_source_sec": float(i + 1),

                        "active_notes_midi": notes_midi,
                        "active_notes_pitch": [midi_to_pitch(midi_note)],
                        "multi_hot": multi_hot,

                        # string unknown for scale unless you encode it in filename; keep optional
                        "string_map_midi": {str(k): -1 for k in range(6)},

                        "transient_window_start": int(t_start),
                        "transient_window_end": int(t_end),
                        "sustain_region": [int(t_end), int(len(seg_out))],

                        "onset_sample": int(onset_out),
                        "num_samples": int(len(seg_out)),
                    }

                    out_json = os.path.join(labels_out, f"sample_{sample_idx:06d}.json")
                    with open(out_json, "w") as f:
                        json.dump(label, f, indent=2)

                    mf.write(json.dumps({
                        "id": label["id"],
                        "audio": os.path.relpath(out_wav, out_root),
                        "label": os.path.relpath(out_json, out_root),
                        "type": "scale_segment",
                    }) + "\n")

                    sample_idx += 1

            else:
                # unknown type already warned in pass1; ignore here
                continue

    print(f"Done. Wrote {sample_idx} samples to: {out_root}")
    print(f"- metadata: {os.path.join(out_root, 'metadata.json')}")
    print(f"- manifest: {manifest_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="raw", help="Path to raw/ directory")
    ap.add_argument("--out", type=str, default="dataset", help="Output dataset directory")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR, help="Target sample rate (Daisy Seed = 48000)")
    args = ap.parse_args()

    build_dataset(args.raw, args.out, sr=args.sr)

if __name__ == "__main__":
    main()