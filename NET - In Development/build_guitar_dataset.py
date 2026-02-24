"""
build_guitar_dataset.py

Creates a transient-triggered + sustain-labeled dataset for low-latency polyphonic guitar note detection.

Supports Windows paths + your CURRENT filename patterns:

SINGLE (pitch-only OR pitch_string):
  single/A#3.wav
  single/F4_5.wav
  single/E2_0.wav   <-- you said you will use this notation

CHORDS (comma-separated tokens, sometimes '.' typo, and '$' accidental typo):
  chord/E2_0,A3_1,D3_2,G3_3.B4_4,E4_5.wav
  chord/G#2_0,C#3_1,F#3_2,B4_3,D$4_4,G#4_5.wav   (# $ -> #)
Also supports the original dash-separated chord pattern.

SCALES (timed chromatic @ 60 BPM; 1 note per second; first second silence):
  string/A2-A4.wav   -> INCLUSIVE range => 25 notes (A2..A4)
  string/E2-E4.wav   -> INCLUSIVE range => 25 notes (E2..E4)

NOTE REPEAT (single note repeated @ 60 BPM; 1 note per second; first second silence):
  note/E2_0.wav
  note_1/E4_5.wav
  note2/A3_1.wav
Each file produces ~30 1-second clips (or fewer if file is shorter), all labeled with the same pitch.
Each 1-second slot searches around the expected beat to locate the pick transient (human timing drift).
Clips are shifted so the transient is near the front (pre-roll) and guarded against next pick transient.

CRITICAL: Scale note-count is derived from filename range:
  n_notes = midi(end) - midi(start) + 1

Outputs:
dataset/
  audio/sample_000000.wav
  labels/sample_000000.json
  metadata.json
  manifest.jsonl
"""

import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import librosa
import soundfile as sf


# ----------------------------
# Config / constants
# ----------------------------
DEFAULT_SR = 48000

TRANSIENT_MS = 5.0
PRE_ROLL_MS = 10.0
ONSET_SEARCH_MAX_MS = 80.0  # (used for some cases; not required for scale/note with drift search)

# Sustain end detection tuning
SUS_HOP = 128                    # RMS hop (~2.67ms @ 48k)
SUS_FRAME = 512                  # RMS window (~10.7ms @ 48k)
SUS_MIN_AFTER_TRANSIENT_MS = 40  # don't end sustain too early
SUS_MIN_SUSTAIN_MS = 120         # enforce minimum sustain length after transient end
SUS_HANGOVER_MS = 60             # must stay quiet for this long
SUS_NOISE_PAD_DB = 6.0           # threshold above noise floor
SUS_REL_DROP_DB = 35.0           # or threshold relative to peak drop

# Scale fixed-1s segmentation with drift compensation
SCALE_FIRST_SILENCE_SEC = 1.0     # first second is silence
SCALE_STEP_SEC = 1.0              # 60 BPM => 1 note/sec
SCALE_SEARCH_HALF_SEC = 0.30      # search +/- 300ms around expected pick time
SCALE_GUARD_MS = 50               # keep this much time before next pick to avoid leakage
SCALE_PAD_MODE = "zeros"          # "zeros" or "edge"

# Note repeat recordings (single note repeated at 60 BPM; 1s intro silence)
NOTE_FIRST_SILENCE_SEC = 1.0
NOTE_STEP_SEC = 1.0
NOTE_SEARCH_HALF_SEC = 0.30
NOTE_GUARD_MS = 50
NOTE_PAD_MODE = "zeros"
NOTE_N_REPS_DEFAULT = 30


# ----------------------------
# Pitch helpers
# ----------------------------
NOTE_RE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+)$')


def normalize_pitch_str(p: str) -> str:
    """
    Normalize pitch tokens to librosa-friendly note names: e.g., 'f#4' -> 'F#4'
    Also fixes '$' -> '#'.
    """
    p = p.strip().replace("$", "#")
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


def scale_note_count(start_pitch: str, end_pitch: str) -> int:
    s = pitch_to_midi(start_pitch)
    e = pitch_to_midi(end_pitch)
    if e < s:
        raise ValueError(f"Scale end pitch {end_pitch} is below start {start_pitch}")
    return (e - s) + 1  # inclusive


# ----------------------------
# FS / Audio helpers
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_wavs(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".wav"):
                yield os.path.join(dirpath, fn)


def read_audio_mono(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def write_wav(path: str, y: np.ndarray, sr: int):
    sf.write(path, y, sr, subtype="FLOAT")


def infer_take_id(path: str) -> str:
    """
    Take id from folder suffixes like single_1, string_2, note3, etc.
    """
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        # match ..._12
        m = re.match(r'.*_(\d+)$', p)
        if m:
            return m.group(1)
        # match ...12 (note12, string2, single3)
        m = re.match(r'.*?(\d+)$', p)
        if m and not re.match(r'^\d+$', p):
            return m.group(1)
    return "0"


def relative_type_from_path(path: str) -> str:
    """
    Determine sample type from folder names.

    Supports multiple takes like:
      single, single_1, single2
      string, string_1, string2
      note, note_1, note2
      chord, chords
    """
    norm = os.path.normpath(path).split(os.sep)
    folders = [p.lower() for p in norm]

    def has_folder_like(base: str) -> bool:
        # Matches: base, base_#, base#
        pat = re.compile(rf"^{re.escape(base)}(?:_\d+|\d+)?$")
        return any(pat.match(f) for f in folders)

    if has_folder_like("single"):
        return "single"
    if has_folder_like("note"):
        return "note"
    if has_folder_like("string"):
        return "scale"
    if has_folder_like("chord") or "chords" in folders:
        return "chord"

    return "unknown"


# ----------------------------
# Robust onset detection (prevents librosa crash) for single/chord
# ----------------------------
def detect_onset_sample(y: np.ndarray, sr: int, search_limit_ms: Optional[float] = None) -> int:
    """
    Detect onset sample. Tries librosa onset_detect with backtrack; if it errors,
    retries without backtrack; if it still fails, uses a simple amplitude threshold.
    """
    if len(y) < int(0.01 * sr):
        return 0

    if search_limit_ms is not None:
        max_samp = min(len(y), int((search_limit_ms / 1000.0) * sr))
        y_search = y[:max_samp]
    else:
        y_search = y

    hop = 128
    oenv = librosa.onset.onset_strength(y=y_search, sr=sr, hop_length=hop)
    if np.max(oenv) < 1e-6:
        return 0

    for backtrack in (True, False):
        try:
            frames = librosa.onset.onset_detect(
                onset_envelope=oenv,
                sr=sr,
                hop_length=hop,
                units="frames",
                backtrack=backtrack,
                pre_max=3, post_max=3, pre_avg=3, post_avg=3,
                delta=0.2,
                wait=0
            )
            if frames is not None and len(frames) > 0:
                onset_frame = int(frames[0])
                onset_sample = int(librosa.frames_to_samples(onset_frame, hop_length=hop))
                return int(np.clip(onset_sample, 0, max(0, len(y_search) - 1)))
        except Exception:
            pass

    # Fallback: amplitude threshold crossing
    a = np.abs(y_search)
    thr = np.median(a) * 8.0
    idx = int(np.argmax(a > thr))
    if a[idx] > thr:
        return idx
    return 0


# ----------------------------
# Transient alignment
# ----------------------------
def align_and_label_transient(y: np.ndarray, sr: int, onset_sample: int) -> Tuple[np.ndarray, int, int, int]:
    """
    Trim so onset is ~PRE_ROLL_MS into the audio.
    Returns:
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


# ----------------------------
# Sustain end estimation
# ----------------------------
def _rms_env(y: np.ndarray, frame: int = SUS_FRAME, hop: int = SUS_HOP) -> np.ndarray:
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    n = 1 + (len(y) - frame) // hop
    env = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        w = y[s:s + frame]
        env[i] = np.sqrt(np.mean(w * w) + 1e-12)
    return env


def _to_db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, 1e-12))


def estimate_sustain_end(y: np.ndarray, sr: int, transient_end: int) -> Tuple[int, Dict[str, float]]:
    """
    Estimate sustain end in samples, clamped to [transient_end+1, len(y)].

    Uses:
      - Noise floor from tail median
      - Relative drop from peak
      - Hangover: must stay below threshold continuously
    """
    env = _rms_env(y)
    env_db = _to_db(env)

    min_after = int((SUS_MIN_AFTER_TRANSIENT_MS / 1000.0) * sr)
    min_sus = int((SUS_MIN_SUSTAIN_MS / 1000.0) * sr)
    start_search_samp = transient_end + max(min_after, min_sus)
    start_search_frame = max(0, start_search_samp // SUS_HOP)

    hang_frames = max(1, int((SUS_HANGOVER_MS / 1000.0) * sr / SUS_HOP))

    post_frame = max(0, transient_end // SUS_HOP)
    peak_db = float(np.max(env_db[post_frame:])) if len(env_db) > post_frame else float(env_db[-1])

    tail_len_s = 0.4
    tail_frames = int((tail_len_s * sr) / SUS_HOP)
    tail_frames = min(tail_frames, max(10, int(0.15 * len(env_db))))
    tail_start = max(0, len(env_db) - tail_frames)
    noise_db = float(np.median(env_db[tail_start:])) if len(env_db) > 0 else -120.0

    thr_abs_db = noise_db + SUS_NOISE_PAD_DB
    thr_rel_db = peak_db - SUS_REL_DROP_DB
    thr_db = max(thr_abs_db, thr_rel_db)

    below = env_db < thr_db
    end_frame = None
    for f in range(start_search_frame, len(env_db) - hang_frames):
        if np.all(below[f:f + hang_frames]):
            end_frame = f
            break

    if end_frame is None:
        end_samp = len(y)
    else:
        end_samp = int(end_frame * SUS_HOP)

    end_samp = int(np.clip(end_samp, transient_end + 1, len(y)))

    dbg = {
        "peak_db": float(peak_db),
        "noise_db": float(noise_db),
        "thr_db": float(thr_db),
        "thr_abs_db": float(thr_abs_db),
        "thr_rel_db": float(thr_rel_db),
        "hangover_ms": float(SUS_HANGOVER_MS),
    }
    return end_samp, dbg


# ----------------------------
# Scale/note drift-compensated pick detection (labels stay on 1s slots)
# ----------------------------
def find_pick_near_time(y: np.ndarray, sr: int, center_samp: int, half_window_samp: int) -> int:
    """
    Robust pick detector inside a window: maximize short-time energy on a pre-emphasized signal.
    Returns absolute sample index in y.
    """
    start = max(0, center_samp - half_window_samp)
    end = min(len(y), center_samp + half_window_samp)
    if end <= start + 64:
        return int(np.clip(center_samp, 0, len(y) - 1))

    w = y[start:end]
    d = np.diff(w, prepend=w[:1])  # pre-emphasis

    frame = 128  # ~2.7ms
    hop = 16     # ~0.33ms resolution
    n = 1 + max(0, (len(d) - frame) // hop)
    if n <= 1:
        return start

    e = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        seg = d[s:s + frame]
        e[i] = float(np.sum(seg * seg))

    idx = int(np.argmax(e))
    pick_local = idx * hop

    refine = 64
    r0 = max(0, pick_local - refine)
    r1 = min(len(d), pick_local + refine)
    fine = int(np.argmax(np.abs(d[r0:r1]))) + r0

    return start + fine


def pad_to_length(x: np.ndarray, length: int, mode: str = "zeros") -> np.ndarray:
    if len(x) >= length:
        return x[:length]
    pad = length - len(x)
    if mode == "edge" and len(x) > 0:
        return np.pad(x, (0, pad), mode="edge")
    return np.pad(x, (0, pad), mode="constant", constant_values=0.0)


# ----------------------------
# Filename parsers
# ----------------------------
def parse_single_filename(stem: str) -> Tuple[str, int]:
    """
    Accept:
      - F4_5 -> (F4, 5)
      - E2_0 -> (E2, 0)   (your preferred notation)
      - A#3  -> (A#3, -1) (string unknown)
    """
    if "_" in stem:
        m = re.match(r'^(.+?)_(\d+)$', stem)
        if not m:
            raise ValueError(f"Single note filename must be like F4_5.wav or A#3.wav, got: {stem}")
        pitch = normalize_pitch_str(m.group(1))
        string_idx = int(m.group(2))
        return pitch, string_idx

    pitch = normalize_pitch_str(stem)
    return pitch, -1


def parse_scale_filename(stem: str) -> Tuple[str, str]:
    """
    Example: A2-A4.wav (inclusive chromatic range)
    """
    m = re.match(r'^(.+?)-(.+?)$', stem)
    if not m:
        raise ValueError(f"Scale filename must be like A2-A4.wav, got: {stem}")
    start_pitch = normalize_pitch_str(m.group(1))
    end_pitch = normalize_pitch_str(m.group(2))
    return start_pitch, end_pitch


def normalize_chord_stem(stem: str) -> str:
    """
    Your chord filenames use comma separators and sometimes '.' instead of ','.
    Also supports '-' as separator.
    Fixes '$' -> '#'.
    """
    s = stem.strip()
    s = s.replace("$", "#")
    s = s.replace('-', ',')
    s = s.replace('.', ',')
    s = s.replace(';', ',')
    while ',,' in s:
        s = s.replace(',,', ',')
    return s


def parse_chord_filename(stem: str) -> Dict[int, Optional[str]]:
    """
    Accepts tokens separated by comma/dash/dot.

    Each token is one of:
      - Pitch_StringNumber, e.g. A3_1
      - 0_StringNumber (no note on that string), e.g. 0_3
      - Pitch (no underscore) => assume string 0
      - 0 (no underscore) => no note on string 0
    """
    stem = normalize_chord_stem(stem)
    tokens = [t.strip() for t in stem.split(',') if t.strip()]

    string_map: Dict[int, Optional[str]] = {}
    for t in tokens:
        if "_" in t:
            left, right = t.split("_", 1)
            try:
                sidx = int(right)
            except:
                raise ValueError(f"Bad chord token (string idx): {t}")

            if left == "0":
                string_map[sidx] = None
            else:
                string_map[sidx] = normalize_pitch_str(left)
        else:
            if t == "0":
                string_map[0] = None
            else:
                string_map[0] = normalize_pitch_str(t)

    return string_map


# ----------------------------
# Dataset builder internals
# ----------------------------
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
        "sustain_detection": {
            "hop": SUS_HOP,
            "frame": SUS_FRAME,
            "min_after_transient_ms": SUS_MIN_AFTER_TRANSIENT_MS,
            "min_sustain_ms": SUS_MIN_SUSTAIN_MS,
            "hangover_ms": SUS_HANGOVER_MS,
            "noise_pad_db": SUS_NOISE_PAD_DB,
            "rel_drop_db": SUS_REL_DROP_DB,
        },
        "scale_segmentation": {
            "slot_sec": SCALE_STEP_SEC,
            "first_silence_sec": SCALE_FIRST_SILENCE_SEC,
            "search_half_sec": SCALE_SEARCH_HALF_SEC,
            "guard_ms": SCALE_GUARD_MS,
            "pad_mode": SCALE_PAD_MODE,
        },
        "note_repeat": {
            "slot_sec": NOTE_STEP_SEC,
            "first_silence_sec": NOTE_FIRST_SILENCE_SEC,
            "search_half_sec": NOTE_SEARCH_HALF_SEC,
            "guard_ms": NOTE_GUARD_MS,
            "pad_mode": NOTE_PAD_MODE,
            "n_reps_default": NOTE_N_REPS_DEFAULT,
        },
    }


def encode_multi_hot(notes_midi: List[int], midi_to_index: Dict[int, int], vocab_size: int) -> List[int]:
    v = np.zeros((vocab_size,), dtype=np.int32)
    for m in notes_midi:
        if m in midi_to_index:
            v[midi_to_index[m]] = 1
    return v.tolist()


def safe_string_map_from_single(sidx: int, midi_note: int) -> Dict[int, int]:
    smap = {i: -1 for i in range(6)}
    if 0 <= sidx <= 5:
        smap[sidx] = int(midi_note)
    return smap


def safe_string_map_from_chord(parsed: Dict[int, Optional[str]]) -> Tuple[Dict[int, int], List[int]]:
    """
    Returns (string_map_midi, notes_midi_list).
    Missing strings => -1 (unknown)
    Explicit None => -1
    """
    string_map = {i: -1 for i in range(6)}
    notes_midi: List[int] = []
    for sidx in range(6):
        if sidx not in parsed:
            continue
        p = parsed[sidx]
        if p is None:
            string_map[sidx] = -1
        else:
            m = pitch_to_midi(p)
            string_map[sidx] = int(m)
            notes_midi.append(int(m))
    notes_midi = sorted(set(notes_midi))
    return string_map, notes_midi


# ----------------------------
# Main build
# ----------------------------
def build_dataset(raw_root: str, out_root: str, sr: int = DEFAULT_SR):
    audio_out = os.path.join(out_root, "audio")
    labels_out = os.path.join(out_root, "labels")
    ensure_dir(audio_out)
    ensure_dir(labels_out)

    discovered: List[Tuple[str, str]] = []
    all_midis: List[int] = []

    # Pass 1: discover + build global vocab
    for wav_path in iter_wavs(raw_root):
        rel_type = relative_type_from_path(wav_path)
        stem = os.path.splitext(os.path.basename(wav_path))[0]

        try:
            if rel_type == "single":
                pitch, _sidx = parse_single_filename(stem)
                all_midis.append(pitch_to_midi(pitch))
                discovered.append((wav_path, "single"))

            elif rel_type == "note":
                pitch, _sidx = parse_single_filename(stem)
                all_midis.append(pitch_to_midi(pitch))
                discovered.append((wav_path, "note"))

            elif rel_type == "scale":
                start_pitch, end_pitch = parse_scale_filename(stem)
                start_midi = pitch_to_midi(start_pitch)
                n_notes = scale_note_count(start_pitch, end_pitch)
                for i in range(n_notes):
                    all_midis.append(start_midi + i)
                discovered.append((wav_path, "scale"))

            elif rel_type == "chord":
                smap = parse_chord_filename(stem)
                for _, p in smap.items():
                    if p is not None:
                        all_midis.append(pitch_to_midi(p))
                discovered.append((wav_path, "chord"))

            else:
                # best-effort fallback classification
                if NOTE_RE.match(stem) or re.search(r'_[0-5]$', stem):
                    pitch, _ = parse_single_filename(stem)
                    all_midis.append(pitch_to_midi(pitch))
                    discovered.append((wav_path, "single"))
                elif "-" in stem and re.match(r'.+?-.+?', stem) and ("_" not in stem) and ("," not in stem) and ("." not in stem):
                    start_pitch, end_pitch = parse_scale_filename(stem)
                    start_midi = pitch_to_midi(start_pitch)
                    n_notes = scale_note_count(start_pitch, end_pitch)
                    for i in range(n_notes):
                        all_midis.append(start_midi + i)
                    discovered.append((wav_path, "scale"))
                elif ("_" in stem) and ("," in stem or "-" in stem or "." in stem):
                    smap = parse_chord_filename(stem)
                    for _, p in smap.items():
                        if p is not None:
                            all_midis.append(pitch_to_midi(p))
                    discovered.append((wav_path, "chord"))
                else:
                    print(f"[WARN] Skipping unrecognized file: {wav_path}")

        except Exception as e:
            print(f"[WARN] Failed to parse {wav_path}: {e}")

    if not all_midis:
        raise RuntimeError("No parsable notes found. Check folder structure/filenames.")

    # Write metadata/vocab
    meta = build_vocab(all_midis)
    midi_to_index = {int(k): int(v) for k, v in meta["midi_to_index"].items()}
    vocab_size = len(meta["midi_vocab"])

    ensure_dir(out_root)
    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Pass 2: generate samples
    manifest_path = os.path.join(out_root, "manifest.jsonl")
    sample_idx = 0

    with open(manifest_path, "w") as mf:
        for wav_path, rel_type in discovered:
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            take_id = infer_take_id(wav_path)

            if rel_type == "single":
                y, _sr = read_audio_mono(wav_path, sr)
                pitch, sidx = parse_single_filename(stem)
                midi_note = pitch_to_midi(pitch)

                notes_midi = [int(midi_note)]
                string_map = safe_string_map_from_single(sidx, midi_note)

                onset = detect_onset_sample(y, sr)
                y_out, onset_out, t_start, t_end = align_and_label_transient(y, sr, onset)

                sus_end, sus_dbg = estimate_sustain_end(y_out, sr, t_end)

                out_name = f"sample_{sample_idx:06d}.wav"
                out_wav = os.path.join(audio_out, out_name)
                write_wav(out_wav, y_out, sr)

                label = {
                    "id": f"{sample_idx:06d}",
                    "type": "single",
                    "source_path": os.path.relpath(wav_path, raw_root),
                    "take_id": take_id,
                    "sr": sr,

                    "active_notes_midi": notes_midi,
                    "active_notes_pitch": [midi_to_pitch(midi_note)],
                    "multi_hot": encode_multi_hot(notes_midi, midi_to_index, vocab_size),

                    "string_map_midi": {str(k): int(v) for k, v in string_map.items()},

                    "transient_window_start": int(t_start),
                    "transient_window_end": int(t_end),
                    "sustain_region": [int(t_end), int(min(sus_end, len(y_out)))],
                    "sustain_end_debug": sus_dbg,

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
                    "type": "single",
                }) + "\n")

                sample_idx += 1

            elif rel_type == "chord":
                y, _sr = read_audio_mono(wav_path, sr)
                parsed = parse_chord_filename(stem)
                string_map, notes_midi = safe_string_map_from_chord(parsed)

                if len(notes_midi) == 0:
                    print(f"[WARN] Chord parsed with 0 notes, skipping: {wav_path}")
                    continue

                onset = detect_onset_sample(y, sr)
                y_out, onset_out, t_start, t_end = align_and_label_transient(y, sr, onset)

                sus_end, sus_dbg = estimate_sustain_end(y_out, sr, t_end)

                out_name = f"sample_{sample_idx:06d}.wav"
                out_wav = os.path.join(audio_out, out_name)
                write_wav(out_wav, y_out, sr)

                label = {
                    "id": f"{sample_idx:06d}",
                    "type": "chord",
                    "source_path": os.path.relpath(wav_path, raw_root),
                    "take_id": take_id,
                    "sr": sr,

                    "active_notes_midi": notes_midi,
                    "active_notes_pitch": [midi_to_pitch(m) for m in notes_midi],
                    "multi_hot": encode_multi_hot(notes_midi, midi_to_index, vocab_size),

                    "string_map_midi": {str(k): int(v) for k, v in string_map.items()},

                    "transient_window_start": int(t_start),
                    "transient_window_end": int(t_end),
                    "sustain_region": [int(t_end), int(min(sus_end, len(y_out)))],
                    "sustain_end_debug": sus_dbg,

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
                    "type": "chord",
                }) + "\n")

                sample_idx += 1

            elif rel_type == "scale":
                # --- Fixed 1-second segmentation, but each segment is shifted using detected pick near expected beat ---
                y, _sr = read_audio_mono(wav_path, sr)
                start_pitch, end_pitch = parse_scale_filename(stem)
                start_midi = pitch_to_midi(start_pitch)
                n_notes = scale_note_count(start_pitch, end_pitch)

                one_sec = sr
                half = int(SCALE_SEARCH_HALF_SEC * sr)
                guard = int((SCALE_GUARD_MS / 1000.0) * sr)
                pre = int((PRE_ROLL_MS / 1000.0) * sr)

                expected_centers = []
                for i in range(n_notes):
                    t_sec = SCALE_FIRST_SILENCE_SEC + i * SCALE_STEP_SEC
                    expected_centers.append(int(t_sec * sr))

                picks: List[int] = []
                for c in expected_centers:
                    if c >= len(y):
                        break
                    p = find_pick_near_time(y, sr, c, half)
                    picks.append(p)

                if len(picks) < n_notes:
                    print(f"[WARN] Scale file ended early or missing notes; found {len(picks)}/{n_notes}: {wav_path}")

                for i in range(1, len(picks)):
                    if picks[i] <= picks[i - 1] + int(0.05 * sr):
                        picks[i] = min(len(y) - 1, picks[i - 1] + int(0.05 * sr))

                for i, pick in enumerate(picks):
                    midi_note = int(start_midi + i)
                    notes_midi = [midi_note]

                    clip_start = pick - pre
                    clip_end = clip_start + one_sec

                    if i + 1 < len(picks):
                        latest_end = picks[i + 1] - guard
                        if clip_end > latest_end:
                            clip_end = latest_end
                            clip_start = clip_end - one_sec

                    clip_start = int(np.clip(clip_start, 0, max(0, len(y) - 1)))
                    clip_end = int(clip_start + one_sec)

                    seg = y[clip_start:min(len(y), clip_end)].copy()
                    seg = pad_to_length(seg, one_sec, mode=SCALE_PAD_MODE)

                    onset_local = int(np.clip(pick - clip_start, 0, one_sec - 1))

                    seg_out, onset_out, t_start, t_end = align_and_label_transient(seg, sr, onset_local)

                    sus_end, sus_dbg = estimate_sustain_end(seg_out, sr, t_end)
                    sus_end = min(sus_end, len(seg_out))

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
                        "expected_time_sec": float(SCALE_FIRST_SILENCE_SEC + i * SCALE_STEP_SEC),

                        "active_notes_midi": notes_midi,
                        "active_notes_pitch": [midi_to_pitch(midi_note)],
                        "multi_hot": multi_hot,

                        "string_map_midi": {str(k): -1 for k in range(6)},

                        "transient_window_start": int(t_start),
                        "transient_window_end": int(t_end),
                        "sustain_region": [int(t_end), int(sus_end)],
                        "sustain_end_debug": sus_dbg,

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

            elif rel_type == "note":
                # --- Repeated single note @ 60 BPM: 1 second intro silence, then 1 note per second ---
                y, _sr = read_audio_mono(wav_path, sr)

                pitch, sidx = parse_single_filename(stem)
                midi_note = int(pitch_to_midi(pitch))

                one_sec = sr
                half = int(NOTE_SEARCH_HALF_SEC * sr)
                guard = int((NOTE_GUARD_MS / 1000.0) * sr)
                pre = int((PRE_ROLL_MS / 1000.0) * sr)

                # Estimate how many reps fit; then clamp to default 30
                max_possible = int((len(y) / sr) - NOTE_FIRST_SILENCE_SEC)
                n_reps = min(NOTE_N_REPS_DEFAULT, max(0, max_possible))

                if n_reps <= 0:
                    print(f"[WARN] Note-repeat file too short (no reps found), skipping: {wav_path}")
                    continue

                expected_centers = []
                for i in range(n_reps):
                    t_sec = NOTE_FIRST_SILENCE_SEC + i * NOTE_STEP_SEC
                    expected_centers.append(int(t_sec * sr))

                picks: List[int] = []
                for c in expected_centers:
                    if c >= len(y):
                        break
                    p = find_pick_near_time(y, sr, c, half)
                    picks.append(p)

                if len(picks) < n_reps:
                    print(f"[WARN] Note-repeat file ended early; found {len(picks)}/{n_reps}: {wav_path}")
                    n_reps = len(picks)

                for i in range(1, len(picks)):
                    if picks[i] <= picks[i - 1] + int(0.05 * sr):
                        picks[i] = min(len(y) - 1, picks[i - 1] + int(0.05 * sr))

                string_map = safe_string_map_from_single(sidx, midi_note)

                for i, pick in enumerate(picks[:n_reps]):
                    notes_midi = [midi_note]

                    clip_start = pick - pre
                    clip_end = clip_start + one_sec

                    if i + 1 < len(picks):
                        latest_end = picks[i + 1] - guard
                        if clip_end > latest_end:
                            clip_end = latest_end
                            clip_start = clip_end - one_sec

                    clip_start = int(np.clip(clip_start, 0, max(0, len(y) - 1)))
                    clip_end = int(clip_start + one_sec)

                    seg = y[clip_start:min(len(y), clip_end)].copy()
                    seg = pad_to_length(seg, one_sec, mode=NOTE_PAD_MODE)

                    onset_local = int(np.clip(pick - clip_start, 0, one_sec - 1))

                    seg_out, onset_out, t_start, t_end = align_and_label_transient(seg, sr, onset_local)

                    sus_end, sus_dbg = estimate_sustain_end(seg_out, sr, t_end)
                    sus_end = min(sus_end, len(seg_out))

                    out_name = f"sample_{sample_idx:06d}.wav"
                    out_wav = os.path.join(audio_out, out_name)
                    write_wav(out_wav, seg_out, sr)

                    multi_hot = encode_multi_hot(notes_midi, midi_to_index, vocab_size)

                    label = {
                        "id": f"{sample_idx:06d}",
                        "type": "note_repeat_segment",
                        "source_path": os.path.relpath(wav_path, raw_root),
                        "take_id": take_id,
                        "sr": sr,

                        "note_repeat_index": i,
                        "expected_time_sec": float(NOTE_FIRST_SILENCE_SEC + i * NOTE_STEP_SEC),

                        "active_notes_midi": notes_midi,
                        "active_notes_pitch": [midi_to_pitch(midi_note)],
                        "multi_hot": multi_hot,

                        "string_map_midi": {str(k): int(v) for k, v in string_map.items()},

                        "transient_window_start": int(t_start),
                        "transient_window_end": int(t_end),
                        "sustain_region": [int(t_end), int(sus_end)],
                        "sustain_end_debug": sus_dbg,

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
                        "type": "note_repeat_segment",
                    }) + "\n")

                    sample_idx += 1

            else:
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