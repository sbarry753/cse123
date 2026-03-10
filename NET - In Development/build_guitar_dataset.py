"""
build_guitar_dataset.py

Creates a positive-only dataset for low-latency polyphonic guitar note detection.

Behavior:
- NO negative / "no note" samples are generated
- NO pick_region or sustain_region labels are generated
- Keeps useful onset + source-region metadata:
    - "segment_source_region": [abs_start, abs_end] in ORIGINAL source wav samples
    - "onset_source_sample": absolute onset sample in ORIGINAL source wav
    - "onset_sample": onset sample in saved-clip coordinates

Scale/note-repeat behavior:
- We DO NOT trim/realign the 1-second segment.
- We keep seg_out exactly 1.0s (padded if needed).
- Onset is labeled inside that 1.0s saved clip.

String classifier support:
- Adds "string_idx" to every label JSON:
    0 = low E, 1 = A, 2 = D, 3 = G, 4 = B, 5 = high E
- For filenames like "F4_5.wav", "_5" is used as string_idx.
- If NO "_#" suffix exists, assumes string_idx = 0 (low E).
- For scale files (e.g. "A2-A4.wav") where filename doesn't include "_#",
  infer string_idx from folder name if possible (e.g. ".../string_3/..."),
  otherwise default to 0.
- For chord files, string_idx = -1 (multi-string / unknown).

Output:
- out_root/audio/sample_XXXXXX.wav
- out_root/labels/sample_XXXXXX.json
- out_root/manifest.jsonl
- out_root/metadata.json
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
PRE_ROLL_MS = 2.0
ONSET_SEARCH_MAX_MS = 80.0

# Scale fixed-1s segmentation with drift compensation
SCALE_FIRST_SILENCE_SEC = 1.0
SCALE_STEP_SEC = 1.0
SCALE_SEARCH_HALF_SEC = 0.30
SCALE_GUARD_MS = 50
SCALE_PAD_MODE = "zeros"  # "zeros" or "edge"

# Note repeat recordings
NOTE_FIRST_SILENCE_SEC = 1.0
NOTE_STEP_SEC = 1.0
NOTE_SEARCH_HALF_SEC = 0.30
NOTE_GUARD_MS = 50
NOTE_PAD_MODE = "zeros"
NOTE_N_REPS_DEFAULT = 30

# Pick detection tuning
PICK_FRAME = 128
PICK_HOP = 16
PICK_REFINE = 96
PICK_MIN_SEARCH_MS = 20
PICK_MAX_SEARCH_MS = 280
PICK_PEAK_FRAC = 0.55
PICK_NOISE_MULT = 8.0


# ----------------------------
# Pitch helpers
# ----------------------------
NOTE_RE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+)$')


def normalize_pitch_str(p: str) -> str:
    p = p.strip().replace("$", "#")
    if p == "0":
        return p
    m = NOTE_RE.match(p)
    if not m:
        raise ValueError(f"Bad pitch token: {p}")
    letter, accidental, octave = m.group(1).upper(), m.group(2), m.group(3)
    return f"{letter}{accidental}{octave}"


def _parse_note_oct(p: str):
    p = p.strip()
    p = (p.replace("$", "#")
           .replace("♯", "#")
           .replace("𝄪", "#")
           .replace("＃", "#")
           .replace("♭", "b")
           .replace("𝄫", "b"))
    m = NOTE_RE.match(p)
    if not m:
        raise ValueError(f"Bad pitch token: {p}")
    note = m.group(1).upper() + m.group(2)
    octv = int(m.group(3))
    return note, octv


def pitch_to_midi(pitch: str) -> int:
    note, a_oct = _parse_note_oct(pitch)
    if note[0] in ("A", "B"):
        std_oct = a_oct - 1
    else:
        std_oct = a_oct
    return int(librosa.note_to_midi(f"{note}{std_oct}"))


def midi_to_pitch(midi: int) -> str:
    s = librosa.midi_to_note(int(midi), octave=True)
    note, std_oct = _parse_note_oct(s)
    if note[0] in ("A", "B"):
        a_oct = std_oct + 1
    else:
        a_oct = std_oct
    return f"{note}{a_oct}"


def scale_note_count(start_pitch: str, end_pitch: str) -> int:
    s = pitch_to_midi(start_pitch)
    e = pitch_to_midi(end_pitch)
    if e < s:
        raise ValueError(f"Scale end pitch {end_pitch} is below start {start_pitch}")
    return (e - s) + 1


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
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        m = re.match(r'.*_(\d+)$', p)
        if m:
            return m.group(1)
        m = re.match(r'.*?(\d+)$', p)
        if m and not re.match(r'^\d+$', p):
            return m.group(1)
    return "0"


def relative_type_from_path(path: str) -> str:
    norm = os.path.normpath(path).split(os.sep)
    folders = [p.lower() for p in norm]

    def has_folder_like(base: str) -> bool:
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


def pad_to_length(x: np.ndarray, length: int, mode: str = "zeros") -> np.ndarray:
    if len(x) >= length:
        return x[:length]
    pad = length - len(x)
    if mode == "edge" and len(x) > 0:
        return np.pad(x, (0, pad), mode="edge")
    return np.pad(x, (0, pad), mode="constant", constant_values=0.0)


# ----------------------------
# String helpers
# ----------------------------
def infer_string_idx_from_path(path: str) -> int:
    """
    Infer string idx from directory names like:
      .../string_0/... or .../string0/... or .../str_5/...
    Returns 0..5 if found, else 0 (default low E).
    """
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    for p in parts:
        m = re.match(r'^(string|str)[_\-]?([0-5])$', p)
        if m:
            return int(m.group(2))
    return 0


# ----------------------------
# Onset detection (single/chord)
# ----------------------------
def detect_onset_sample(y: np.ndarray, sr: int, search_limit_ms: Optional[float] = None) -> int:
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

    a = np.abs(y_search)
    thr = np.median(a) * 8.0
    idx = int(np.argmax(a > thr))
    if a[idx] > thr:
        return idx
    return 0


# ----------------------------
# Transient alignment (kept for single/chord)
# ----------------------------
def transient_len_samples(sr: int) -> int:
    return int(round((TRANSIENT_MS / 1000.0) * sr))


def align_and_label_transient(y: np.ndarray, sr: int, onset_sample: int) -> Tuple[np.ndarray, int, int]:
    """
    Trims y so the onset is near the start (with PRE_ROLL_MS),
    and returns:
      y_out, onset_out, trim_start_in_source
    """
    pre = int((PRE_ROLL_MS / 1000.0) * sr)

    trim_start = max(0, onset_sample - pre)
    y_out = y[trim_start:].copy()

    onset_out = onset_sample - trim_start
    return y_out, onset_out, trim_start


# ----------------------------
# Drift-compensated pick detection for scale/note repeat
# ----------------------------
def find_pick_near_time(y: np.ndarray, sr: int, center_samp: int, half_window_samp: int) -> int:
    start = max(0, center_samp - half_window_samp)
    end = min(len(y), center_samp + half_window_samp)
    if end <= start + 64:
        return int(np.clip(center_samp, 0, len(y) - 1))

    w = y[start:end]
    d = np.diff(w, prepend=w[:1]).astype(np.float32)
    dpos = np.maximum(d, 0.0)

    frame = PICK_FRAME
    hop = PICK_HOP
    n = 1 + max(0, (len(dpos) - frame) // hop)
    if n <= 1:
        return int(np.clip(center_samp, 0, len(y) - 1))

    e = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        seg = dpos[s:s + frame]
        e[i] = float(np.sum(seg * seg))

    peak = float(np.max(e))
    med = float(np.median(e))
    if peak <= 1e-12:
        return int(np.clip(center_samp, 0, len(y) - 1))

    thr = max(peak * PICK_PEAK_FRAC, med * PICK_NOISE_MULT)

    min_i = int((PICK_MIN_SEARCH_MS / 1000.0) * sr / hop)
    max_i = int((PICK_MAX_SEARCH_MS / 1000.0) * sr / hop)
    min_i = int(np.clip(min_i, 0, n - 1))
    max_i = int(np.clip(max_i, 0, n - 1))
    if max_i <= min_i:
        min_i = 0
        max_i = n - 1

    idx = None
    for i in range(min_i, max_i + 1):
        if e[i] >= thr:
            idx = i
            break
    if idx is None:
        idx = int(np.argmax(e[min_i:max_i + 1]) + min_i)

    pick_local = idx * hop

    refine = PICK_REFINE
    r0 = max(0, pick_local - refine)
    r1 = min(len(d), pick_local + refine)
    fine = int(np.argmax(np.abs(d[r0:r1]))) + r0

    return int(start + fine)


# ----------------------------
# Filename parsers
# ----------------------------
def parse_single_filename(stem: str) -> Tuple[str, int]:
    """
    Supports:
      F4_5.wav  -> (pitch="F4", string_idx=5)
      A#3.wav   -> (pitch="A#3", string_idx=0)
    """
    if "_" in stem:
        m = re.match(r'^(.+?)_(\d+)$', stem)
        if not m:
            raise ValueError(f"Single note filename must be like F4_5.wav or A#3.wav, got: {stem}")
        pitch = normalize_pitch_str(m.group(1))
        string_idx = int(m.group(2))
        return pitch, string_idx
    pitch = normalize_pitch_str(stem)
    return pitch, 0


def parse_scale_filename(stem: str) -> Tuple[str, str]:
    m = re.match(r'^(.+?)-(.+?)$', stem)
    if not m:
        raise ValueError(f"Scale filename must be like A2-A4.wav, got: {stem}")
    start_pitch = normalize_pitch_str(m.group(1))
    end_pitch = normalize_pitch_str(m.group(2))
    return start_pitch, end_pitch


def normalize_chord_stem(stem: str) -> str:
    s = stem.strip()
    s = s.replace("$", "#")
    s = s.replace('-', ',')
    s = s.replace('.', ',')
    s = s.replace(';', ',')
    while ',,' in s:
        s = s.replace(',,', ',')
    return s


def parse_chord_filename(stem: str) -> Dict[int, Optional[str]]:
    stem = normalize_chord_stem(stem)
    tokens = [t.strip() for t in stem.split(',') if t.strip()]

    string_map: Dict[int, Optional[str]] = {}
    for t in tokens:
        if "_" in t:
            left, right = t.split("_", 1)
            sidx = int(right)
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
        "string_labeling": {
            "string_idx_meaning": "0=lowE, 1=A, 2=D, 3=G, 4=B, 5=highE; -1=unknown/multi",
            "filename_suffix_rule": "If filename token ends with _<0..5>, use that. If absent, default 0.",
            "scale_path_rule": "If scale filename has no suffix, try infer from folder like string_3, else default 0.",
            "chord_rule": "string_idx = -1",
        },
        "region_labels": {
            "segment_source_region": "Where the saved clip came from in the original wav: [abs_start, abs_end)",
            "onset_source_sample": "Absolute onset sample in original wav",
            "onset_sample": "Onset sample in saved clip coordinates",
        }
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
def build_dataset(
    raw_root: str,
    out_root: str,
    sr: int = DEFAULT_SR,
):
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
                # Fallback guesses
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

    meta = build_vocab(all_midis)
    midi_to_index = {int(k): int(v) for k, v in meta["midi_to_index"].items()}
    vocab_size = len(meta["midi_vocab"])

    ensure_dir(out_root)
    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    manifest_path = os.path.join(out_root, "manifest.jsonl")
    sample_idx = 0

    with open(manifest_path, "w") as mf:
        for wav_path, rel_type in discovered:
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            take_id = infer_take_id(wav_path)

            # ----------------------------
            # SINGLE
            # ----------------------------
            if rel_type == "single":
                y, _sr = read_audio_mono(wav_path, sr)
                pitch, sidx = parse_single_filename(stem)
                midi_note = pitch_to_midi(pitch)

                notes_midi = [int(midi_note)]
                string_map = safe_string_map_from_single(sidx, midi_note)

                onset_abs = detect_onset_sample(y, sr, search_limit_ms=ONSET_SEARCH_MAX_MS)
                y_out, onset_out, trim_start = align_and_label_transient(y, sr, onset_abs)

                out_name = f"sample_{sample_idx:06d}.wav"
                out_wav = os.path.join(audio_out, out_name)
                write_wav(out_wav, y_out, sr)

                label = {
                    "id": f"{sample_idx:06d}",
                    "type": "single",
                    "source_path": os.path.relpath(wav_path, raw_root),
                    "take_id": take_id,
                    "sr": sr,
                    "string_idx": int(sidx),

                    "active_notes_midi": notes_midi,
                    "active_notes_pitch": [midi_to_pitch(midi_note)],
                    "multi_hot": encode_multi_hot(notes_midi, midi_to_index, vocab_size),
                    "string_map_midi": {str(k): int(v) for k, v in string_map.items()},

                    "segment_source_region": [int(trim_start), int(len(y))],
                    "onset_source_sample": int(onset_abs),
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

            # ----------------------------
            # CHORD
            # ----------------------------
            elif rel_type == "chord":
                y, _sr = read_audio_mono(wav_path, sr)
                parsed = parse_chord_filename(stem)
                string_map, notes_midi = safe_string_map_from_chord(parsed)

                if len(notes_midi) == 0:
                    print(f"[WARN] Chord parsed with 0 notes, skipping: {wav_path}")
                    continue

                onset_abs = detect_onset_sample(y, sr, search_limit_ms=ONSET_SEARCH_MAX_MS)
                y_out, onset_out, trim_start = align_and_label_transient(y, sr, onset_abs)

                out_name = f"sample_{sample_idx:06d}.wav"
                out_wav = os.path.join(audio_out, out_name)
                write_wav(out_wav, y_out, sr)

                label = {
                    "id": f"{sample_idx:06d}",
                    "type": "chord",
                    "source_path": os.path.relpath(wav_path, raw_root),
                    "take_id": take_id,
                    "sr": sr,
                    "string_idx": -1,

                    "active_notes_midi": notes_midi,
                    "active_notes_pitch": [midi_to_pitch(m) for m in notes_midi],
                    "multi_hot": encode_multi_hot(notes_midi, midi_to_index, vocab_size),
                    "string_map_midi": {str(k): int(v) for k, v in string_map.items()},

                    "segment_source_region": [int(trim_start), int(len(y))],
                    "onset_source_sample": int(onset_abs),
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

            # ----------------------------
            # SCALE (1-second segments, NOT trimmed)
            # ----------------------------
            elif rel_type == "scale":
                y, _sr = read_audio_mono(wav_path, sr)
                start_pitch, end_pitch = parse_scale_filename(stem)
                start_midi = pitch_to_midi(start_pitch)
                n_notes = scale_note_count(start_pitch, end_pitch)

                scale_string_idx = infer_string_idx_from_path(wav_path)

                one_sec = sr
                half = int(SCALE_SEARCH_HALF_SEC * sr)
                guard = int((SCALE_GUARD_MS / 1000.0) * sr)
                pre = int((PRE_ROLL_MS / 1000.0) * sr)

                expected_centers = [int((SCALE_FIRST_SILENCE_SEC + i * SCALE_STEP_SEC) * sr) for i in range(n_notes)]

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

                for i, pick_abs in enumerate(picks):
                    midi_note = int(start_midi + i)
                    notes_midi = [midi_note]

                    clip_start = int(pick_abs - pre)
                    clip_end = int(clip_start + one_sec)

                    if i + 1 < len(picks):
                        latest_end = int(picks[i + 1] - guard)
                        if clip_end > latest_end:
                            clip_end = latest_end
                            clip_start = clip_end - one_sec

                    clip_start = int(np.clip(clip_start, 0, max(0, len(y) - 1)))
                    clip_end = int(clip_start + one_sec)

                    seg = y[clip_start:min(len(y), clip_end)].copy()
                    seg = pad_to_length(seg, one_sec, mode=SCALE_PAD_MODE)

                    onset_local = int(np.clip(pick_abs - clip_start, 0, one_sec - 1))

                    seg_out = seg
                    onset_out = onset_local

                    pos_id = f"{sample_idx:06d}"
                    out_name = f"sample_{sample_idx:06d}.wav"
                    out_wav = os.path.join(audio_out, out_name)
                    write_wav(out_wav, seg_out, sr)

                    label = {
                        "id": pos_id,
                        "type": "scale_segment",
                        "source_path": os.path.relpath(wav_path, raw_root),
                        "take_id": take_id,
                        "sr": sr,
                        "string_idx": int(scale_string_idx),

                        "scale_segment_index": i,
                        "expected_time_sec": float(SCALE_FIRST_SILENCE_SEC + i * SCALE_STEP_SEC),

                        "active_notes_midi": notes_midi,
                        "active_notes_pitch": [midi_to_pitch(midi_note)],
                        "multi_hot": encode_multi_hot(notes_midi, midi_to_index, vocab_size),
                        "string_map_midi": {str(k): -1 for k in range(6)},

                        "segment_source_region": [int(clip_start), int(min(len(y), clip_start + one_sec))],
                        "onset_source_sample": int(pick_abs),
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

            # ----------------------------
            # NOTE-REPEAT (1-second segments, NOT trimmed)
            # ----------------------------
            elif rel_type == "note":
                y, _sr = read_audio_mono(wav_path, sr)
                pitch, sidx = parse_single_filename(stem)
                midi_note = int(pitch_to_midi(pitch))

                one_sec = sr
                half = int(NOTE_SEARCH_HALF_SEC * sr)
                guard = int((NOTE_GUARD_MS / 1000.0) * sr)
                pre = int((PRE_ROLL_MS / 1000.0) * sr)

                max_possible = int((len(y) / sr) - NOTE_FIRST_SILENCE_SEC)
                n_reps = min(NOTE_N_REPS_DEFAULT, max(0, max_possible))
                if n_reps <= 0:
                    print(f"[WARN] Note-repeat file too short (no reps found), skipping: {wav_path}")
                    continue

                expected_centers = [int((NOTE_FIRST_SILENCE_SEC + i * NOTE_STEP_SEC) * sr) for i in range(n_reps)]

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

                for i, pick_abs in enumerate(picks[:n_reps]):
                    notes_midi = [midi_note]

                    clip_start = int(pick_abs - pre)
                    clip_end = int(clip_start + one_sec)

                    if i + 1 < len(picks):
                        latest_end = int(picks[i + 1] - guard)
                        if clip_end > latest_end:
                            clip_end = latest_end
                            clip_start = clip_end - one_sec

                    clip_start = int(np.clip(clip_start, 0, max(0, len(y) - 1)))
                    clip_end = int(clip_start + one_sec)

                    seg = y[clip_start:min(len(y), clip_end)].copy()
                    seg = pad_to_length(seg, one_sec, mode=NOTE_PAD_MODE)

                    onset_local = int(np.clip(pick_abs - clip_start, 0, one_sec - 1))

                    seg_out = seg
                    onset_out = onset_local

                    pos_id = f"{sample_idx:06d}"
                    out_name = f"sample_{sample_idx:06d}.wav"
                    out_wav = os.path.join(audio_out, out_name)
                    write_wav(out_wav, seg_out, sr)

                    label = {
                        "id": pos_id,
                        "type": "note_repeat_segment",
                        "source_path": os.path.relpath(wav_path, raw_root),
                        "take_id": take_id,
                        "sr": sr,
                        "string_idx": int(sidx),

                        "note_repeat_index": i,
                        "expected_time_sec": float(NOTE_FIRST_SILENCE_SEC + i * NOTE_STEP_SEC),

                        "active_notes_midi": notes_midi,
                        "active_notes_pitch": [midi_to_pitch(midi_note)],
                        "multi_hot": encode_multi_hot(notes_midi, midi_to_index, vocab_size),
                        "string_map_midi": {str(k): int(v) for k, v in string_map.items()},

                        "segment_source_region": [int(clip_start), int(min(len(y), clip_start + one_sec))],
                        "onset_source_sample": int(pick_abs),
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

    build_dataset(
        args.raw,
        args.out,
        sr=args.sr,
    )


if __name__ == "__main__":
    main()