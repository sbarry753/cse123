"""
build_guitar_dataset.py

Creates a transient-triggered + sustain-labeled dataset for low-latency polyphonic guitar note detection.

Key improvement:
- For scale/note-repeat (60 BPM grid with human drift), we detect the PICK
  as the earliest strong positive-going transient (not max energy).

NEGATIVES (fixed + reliable):
- Your recording pattern: 60 BPM, one note per second, and the FIRST second is silent.
  So we generate negatives from the known silent lead-in (first 1.0s) of each
  scale/note recording. This is stable and avoids "sus_end is early" problems.
- Optionally also generate tail-silence negatives inside each 1-second segment, but
  only when there is enough room and we find a truly quiet region.
- Global cap on negatives to prevent imbalance (e.g. 100).
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

# Sustain end detection tuning
SUS_HOP = 128
SUS_FRAME = 512
SUS_MIN_AFTER_TRANSIENT_MS = 40
SUS_MIN_SUSTAIN_MS = 120
SUS_HANGOVER_MS = 60
SUS_NOISE_PAD_DB = 6.0
SUS_REL_DROP_DB = 35.0

# Neg silence detection (used only for OPTIONAL tail negatives)
NEG_SILENCE_NOISE_PAD_DB = 2.0
NEG_SILENCE_REL_DROP_DB = 45.0
NEG_SILENCE_HANGOVER_MS = 120.0

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
# Robust onset detection (single/chord)
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
# Transient alignment
# ----------------------------
def align_and_label_transient(y: np.ndarray, sr: int, onset_sample: int) -> Tuple[np.ndarray, int, int, int]:
    pre = int((PRE_ROLL_MS / 1000.0) * sr)
    tlen = int((TRANSIENT_MS / 1000.0) * sr)

    trim_start = max(0, onset_sample - pre)
    y_out = y[trim_start:].copy()

    onset_out = onset_sample - trim_start
    transient_start = onset_out
    transient_end = min(len(y_out), transient_start + tlen)
    return y_out, onset_out, transient_start, transient_end


# ----------------------------
# Sustain end estimation (used for labeling only)
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

    end_samp = len(y) if end_frame is None else int(end_frame * SUS_HOP)
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
# OPTIONAL: Find truly quiet start for tail negatives
# (Works best when you have >1s of audio available to the right, but we keep it optional.)
# ----------------------------
def find_quiet_start_for_negative(
    y: np.ndarray,
    sr: int,
    search_from_samp: int,
    min_quiet_ms: float,
    *,
    hop: int = SUS_HOP,
    frame: int = SUS_FRAME,
    noise_pad_db: float = NEG_SILENCE_NOISE_PAD_DB,
    rel_drop_db: float = NEG_SILENCE_REL_DROP_DB,
    hangover_ms: float = NEG_SILENCE_HANGOVER_MS,
) -> Optional[int]:
    if len(y) < frame + hop:
        return None

    env = _rms_env(y, frame=frame, hop=hop)
    env_db = _to_db(env)

    start_frame = int(np.clip(search_from_samp // hop, 0, len(env_db) - 1))

    hang_frames = max(1, int((hangover_ms / 1000.0) * sr / hop))
    quiet_frames = max(1, int((min_quiet_ms / 1000.0) * sr / hop))
    need = max(quiet_frames, hang_frames)

    peak_db = float(np.max(env_db[start_frame:])) if start_frame < len(env_db) else float(env_db[-1])

    tail_len_s = 0.4
    tail_frames = int((tail_len_s * sr) / hop)
    tail_frames = min(tail_frames, max(10, int(0.15 * len(env_db))))
    tail_start = max(0, len(env_db) - tail_frames)
    noise_db = float(np.median(env_db[tail_start:])) if len(env_db) else -120.0

    thr_abs_db = noise_db + float(noise_pad_db)
    thr_rel_db = peak_db - float(rel_drop_db)
    thr_db = min(thr_abs_db, thr_rel_db)

    below = env_db < thr_db

    last_ok = None
    for f in range(start_frame, len(env_db) - need):
        if np.all(below[f:f + need]):
            last_ok = f

    if last_ok is None:
        return None
    return int(last_ok * hop)


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
        "sustain_detection": {
            "hop": SUS_HOP,
            "frame": SUS_FRAME,
            "min_after_transient_ms": SUS_MIN_AFTER_TRANSIENT_MS,
            "min_sustain_ms": SUS_MIN_SUSTAIN_MS,
            "hangover_ms": SUS_HANGOVER_MS,
            "noise_pad_db": SUS_NOISE_PAD_DB,
            "rel_drop_db": SUS_REL_DROP_DB,
        },
        "negatives": {
            "enabled": False,
            "neg_per_pos": 0,
            "neg_min_silence_ms": 0.0,
            "neg_pad_mode": "zeros",
            "neg_max_total": 0,
            "neg_source": "lead_silence_first_sec",
            "neg_silence_noise_pad_db": NEG_SILENCE_NOISE_PAD_DB,
            "neg_silence_rel_drop_db": NEG_SILENCE_REL_DROP_DB,
            "neg_silence_hangover_ms": NEG_SILENCE_HANGOVER_MS,
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


def emit_negative_clip(
    *,
    mf,
    audio_out: str,
    labels_out: str,
    out_root: str,
    raw_root: str,
    sample_idx: int,
    source_wav_path: str,
    take_id: str,
    sr: int,
    vocab_size: int,
    audio_1s: np.ndarray,
    neg_subtype: str,
    ref_pos_id: str,
    extra_dbg: Dict[str, Any],
) -> int:
    out_name = f"sample_{sample_idx:06d}.wav"
    out_wav = os.path.join(audio_out, out_name)
    write_wav(out_wav, audio_1s, sr)

    label = {
        "id": f"{sample_idx:06d}",
        "type": "neg_segment",
        "neg_subtype": str(neg_subtype),
        "ref_positive_id": str(ref_pos_id),
        "source_path": os.path.relpath(source_wav_path, raw_root),
        "take_id": take_id,
        "sr": sr,
        "active_notes_midi": [],
        "active_notes_pitch": [],
        "multi_hot": [0] * int(vocab_size),
        "string_map_midi": {str(k): -1 for k in range(6)},
        "transient_window_start": 0,
        "transient_window_end": 0,
        "sustain_region": [0, 0],
        "sustain_end_debug": extra_dbg,
        "onset_sample": 0,
        "num_samples": int(len(audio_1s)),
    }

    out_json = os.path.join(labels_out, f"sample_{sample_idx:06d}.json")
    with open(out_json, "w") as f:
        json.dump(label, f, indent=2)

    mf.write(json.dumps({
        "id": label["id"],
        "audio": os.path.relpath(out_wav, out_root),
        "label": os.path.relpath(out_json, out_root),
        "type": "neg_segment",
    }) + "\n")

    return sample_idx + 1


def maybe_emit_negative_from_lead_silence(
    *,
    mf,
    audio_out: str,
    labels_out: str,
    out_root: str,
    raw_root: str,
    sample_idx: int,
    source_wav_path: str,
    take_id: str,
    sr: int,
    vocab_size: int,
    full_y: np.ndarray,
    lead_silence_sec: float,
    neg_pad_mode: str,
    ref_pos_id: str,
) -> Tuple[int, bool]:
    """
    Reliable negatives: slice a 1-second negative out of the FIRST lead_silence_sec seconds.
    For your recordings, lead_silence_sec should be 1.0.
    """
    one_sec = int(sr)
    lead = int(lead_silence_sec * sr)
    if lead < one_sec or len(full_y) < one_sec:
        return sample_idx, False

    # choose random 1s window inside the lead silence region
    max_start = lead - one_sec
    st = 0 if max_start <= 0 else int(np.random.randint(0, max_start + 1))
    neg = full_y[st:st + one_sec].copy()
    neg = pad_to_length(neg, one_sec, mode=neg_pad_mode)

    dbg = {
        "neg_source": "lead_silence",
        "lead_silence_sec": float(lead_silence_sec),
        "start_sample": int(st),
        "end_sample": int(st + one_sec),
    }

    sample_idx = emit_negative_clip(
        mf=mf,
        audio_out=audio_out,
        labels_out=labels_out,
        out_root=out_root,
        raw_root=raw_root,
        sample_idx=sample_idx,
        source_wav_path=source_wav_path,
        take_id=take_id,
        sr=sr,
        vocab_size=vocab_size,
        audio_1s=neg,
        neg_subtype="lead_silence",
        ref_pos_id=ref_pos_id,
        extra_dbg=dbg,
    )
    return sample_idx, True


def maybe_emit_negative_from_tail_silence(
    *,
    mf,
    audio_out: str,
    labels_out: str,
    out_root: str,
    raw_root: str,
    sample_idx: int,
    source_wav_path: str,
    take_id: str,
    sr: int,
    vocab_size: int,
    sus_end: int,
    seg_out: np.ndarray,
    neg_min_silence_ms: float,
    neg_pad_mode: str,
    ref_pos_id: str,
    neg_min_after_sus_ms: float = 250.0,
) -> Tuple[int, bool]:
    """
    Optional negatives: try to find a quiet region AFTER sus_end.
    IMPORTANT: because seg_out is ~1s long, we only emit a 1s negative if quiet_start is near 0.
    So instead we emit from quiet_start to END of segment, then pad to 1s.
    """
    one_sec = int(sr)
    sus_end = int(np.clip(sus_end, 0, len(seg_out)))
    guard = int((neg_min_after_sus_ms / 1000.0) * sr)
    search_from = int(np.clip(sus_end + guard, 0, len(seg_out)))

    min_sil = int((neg_min_silence_ms / 1000.0) * sr)
    if (len(seg_out) - search_from) < max(1, min_sil):
        return sample_idx, False

    quiet_start = find_quiet_start_for_negative(
        seg_out, sr, search_from_samp=search_from, min_quiet_ms=neg_min_silence_ms
    )
    if quiet_start is None:
        return sample_idx, False

    quiet_start = int(np.clip(quiet_start, 0, len(seg_out)))

    # take from quiet_start to end; pad to 1s
    neg = seg_out[quiet_start:].copy()
    neg = pad_to_length(neg, one_sec, mode=neg_pad_mode)

    dbg = {
        "neg_source": "tail_silence",
        "quiet_start": int(quiet_start),
        "sus_end_used": int(sus_end),
        "search_from": int(search_from),
        "neg_min_silence_ms": float(neg_min_silence_ms),
        "neg_min_after_sus_ms": float(neg_min_after_sus_ms),
        "neg_silence_noise_pad_db": float(NEG_SILENCE_NOISE_PAD_DB),
        "neg_silence_rel_drop_db": float(NEG_SILENCE_REL_DROP_DB),
        "neg_silence_hangover_ms": float(NEG_SILENCE_HANGOVER_MS),
    }

    sample_idx = emit_negative_clip(
        mf=mf,
        audio_out=audio_out,
        labels_out=labels_out,
        out_root=out_root,
        raw_root=raw_root,
        sample_idx=sample_idx,
        source_wav_path=source_wav_path,
        take_id=take_id,
        sr=sr,
        vocab_size=vocab_size,
        audio_1s=neg,
        neg_subtype="tail_silence",
        ref_pos_id=ref_pos_id,
        extra_dbg=dbg,
    )
    return sample_idx, True


# ----------------------------
# Main build
# ----------------------------
def build_dataset(
    raw_root: str,
    out_root: str,
    sr: int = DEFAULT_SR,
    *,
    add_negatives: bool = False,
    neg_per_pos: int = 1,
    neg_min_silence_ms: float = 150.0,
    neg_pad_mode: str = "zeros",
    neg_max_total: int = 100,
    # NEW: reliable neg source from lead silence
    neg_use_lead_silence: bool = True,
    neg_lead_silence_sec: float = 1.0,
    # Optional: also try tail silence
    neg_use_tail_silence: bool = False,
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

    meta["negatives"]["enabled"] = bool(add_negatives)
    meta["negatives"]["neg_per_pos"] = int(neg_per_pos)
    meta["negatives"]["neg_min_silence_ms"] = float(neg_min_silence_ms)
    meta["negatives"]["neg_pad_mode"] = str(neg_pad_mode)
    meta["negatives"]["neg_max_total"] = int(neg_max_total)
    meta["negatives"]["neg_use_lead_silence"] = bool(neg_use_lead_silence)
    meta["negatives"]["neg_lead_silence_sec"] = float(neg_lead_silence_sec)
    meta["negatives"]["neg_use_tail_silence"] = bool(neg_use_tail_silence)

    ensure_dir(out_root)
    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    manifest_path = os.path.join(out_root, "manifest.jsonl")
    sample_idx = 0

    neg_written = 0
    neg_max_total = int(max(0, neg_max_total))

    with open(manifest_path, "w") as mf:
        # Cache full_y per wav for lead-silence negatives (scale/note only)
        full_audio_cache: Dict[str, np.ndarray] = {}

        for wav_path, rel_type in discovered:
            stem = os.path.splitext(os.path.basename(wav_path))[0]
            take_id = infer_take_id(wav_path)

            # Preload full audio for scale/note (so we can pull lead-silence negatives)
            if rel_type in ("scale", "note") and add_negatives and neg_use_lead_silence and (neg_written < neg_max_total):
                if wav_path not in full_audio_cache:
                    full_y, _sr = read_audio_mono(wav_path, sr)
                    full_audio_cache[wav_path] = full_y

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
                y, _sr = read_audio_mono(wav_path, sr)
                start_pitch, end_pitch = parse_scale_filename(stem)
                start_midi = pitch_to_midi(start_pitch)
                n_notes = scale_note_count(start_pitch, end_pitch)

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

                    # POS
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
                        "scale_segment_index": i,
                        "expected_time_sec": float(SCALE_FIRST_SILENCE_SEC + i * SCALE_STEP_SEC),
                        "active_notes_midi": notes_midi,
                        "active_notes_pitch": [midi_to_pitch(midi_note)],
                        "multi_hot": encode_multi_hot(notes_midi, midi_to_index, vocab_size),
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

                    # NEG(s)
                    if add_negatives and (neg_written < neg_max_total):
                        for _ in range(max(1, int(neg_per_pos))):
                            if neg_written >= neg_max_total:
                                break

                            wrote = False
                            # 1) Reliable lead-silence negatives
                            if neg_use_lead_silence:
                                full_y = full_audio_cache.get(wav_path, None)
                                if full_y is not None:
                                    sample_idx, wrote = maybe_emit_negative_from_lead_silence(
                                        mf=mf,
                                        audio_out=audio_out,
                                        labels_out=labels_out,
                                        out_root=out_root,
                                        raw_root=raw_root,
                                        sample_idx=sample_idx,
                                        source_wav_path=wav_path,
                                        take_id=take_id,
                                        sr=sr,
                                        vocab_size=vocab_size,
                                        full_y=full_y,
                                        lead_silence_sec=neg_lead_silence_sec,
                                        neg_pad_mode=neg_pad_mode,
                                        ref_pos_id=pos_id,
                                    )

                            # 2) Optional tail-silence negatives
                            if (not wrote) and neg_use_tail_silence:
                                sample_idx, wrote = maybe_emit_negative_from_tail_silence(
                                    mf=mf,
                                    audio_out=audio_out,
                                    labels_out=labels_out,
                                    out_root=out_root,
                                    raw_root=raw_root,
                                    sample_idx=sample_idx,
                                    source_wav_path=wav_path,
                                    take_id=take_id,
                                    sr=sr,
                                    vocab_size=vocab_size,
                                    sus_end=sus_end,
                                    seg_out=seg_out,
                                    neg_min_silence_ms=neg_min_silence_ms,
                                    neg_pad_mode=neg_pad_mode,
                                    ref_pos_id=pos_id,
                                )

                            if wrote:
                                neg_written += 1

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

                    # POS
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
                        "note_repeat_index": i,
                        "expected_time_sec": float(NOTE_FIRST_SILENCE_SEC + i * NOTE_STEP_SEC),
                        "active_notes_midi": notes_midi,
                        "active_notes_pitch": [midi_to_pitch(midi_note)],
                        "multi_hot": encode_multi_hot(notes_midi, midi_to_index, vocab_size),
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

                    # NEG(s)
                    if add_negatives and (neg_written < neg_max_total):
                        for _ in range(max(1, int(neg_per_pos))):
                            if neg_written >= neg_max_total:
                                break

                            wrote = False
                            if neg_use_lead_silence:
                                full_y = full_audio_cache.get(wav_path, None)
                                if full_y is None:
                                    full_y, _ = read_audio_mono(wav_path, sr)
                                    full_audio_cache[wav_path] = full_y
                                sample_idx, wrote = maybe_emit_negative_from_lead_silence(
                                    mf=mf,
                                    audio_out=audio_out,
                                    labels_out=labels_out,
                                    out_root=out_root,
                                    raw_root=raw_root,
                                    sample_idx=sample_idx,
                                    source_wav_path=wav_path,
                                    take_id=take_id,
                                    sr=sr,
                                    vocab_size=vocab_size,
                                    full_y=full_y,
                                    lead_silence_sec=neg_lead_silence_sec,
                                    neg_pad_mode=neg_pad_mode,
                                    ref_pos_id=pos_id,
                                )

                            if (not wrote) and neg_use_tail_silence:
                                sample_idx, wrote = maybe_emit_negative_from_tail_silence(
                                    mf=mf,
                                    audio_out=audio_out,
                                    labels_out=labels_out,
                                    out_root=out_root,
                                    raw_root=raw_root,
                                    sample_idx=sample_idx,
                                    source_wav_path=wav_path,
                                    take_id=take_id,
                                    sr=sr,
                                    vocab_size=vocab_size,
                                    sus_end=sus_end,
                                    seg_out=seg_out,
                                    neg_min_silence_ms=neg_min_silence_ms,
                                    neg_pad_mode=neg_pad_mode,
                                    ref_pos_id=pos_id,
                                )

                            if wrote:
                                neg_written += 1

            else:
                continue

    print(f"Done. Wrote {sample_idx} samples to: {out_root}")
    print(f"- metadata: {os.path.join(out_root, 'metadata.json')}")
    print(f"- manifest: {manifest_path}")
    if add_negatives:
        print(f"- negatives_written: {neg_written}/{neg_max_total}")
    else:
        print(f"- negatives_written: 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="raw", help="Path to raw/ directory")
    ap.add_argument("--out", type=str, default="dataset", help="Output dataset directory")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR, help="Target sample rate (Daisy Seed = 48000)")

    # Negatives
    ap.add_argument("--add_negatives", action="store_true",
                    help="Generate negative (no-sound) samples.")
    ap.add_argument("--neg_per_pos", type=int, default=1,
                    help="How many negatives to try per positive segment (scale/note).")
    ap.add_argument("--neg_pad_mode", type=str, default="zeros", choices=["zeros", "edge"],
                    help="Padding mode for negative clips.")
    ap.add_argument("--neg_max_total", type=int, default=100,
                    help="Global cap on total negatives to generate (prevents imbalance).")

    # New: reliable lead-silence negatives
    ap.add_argument("--neg_use_lead_silence", type=int, default=1,
                    help="Use first silent second of scale/note recordings for negatives (recommended).")
    ap.add_argument("--neg_lead_silence_sec", type=float, default=1.0,
                    help="How many seconds at the start are guaranteed silent (your setup: 1.0).")

    # Optional tail silence
    ap.add_argument("--neg_use_tail_silence", type=int, default=0,
                    help="Also try generating negatives from tail silence inside each 1s segment (optional).")
    ap.add_argument("--neg_min_silence_ms", type=float, default=150.0,
                    help="(Tail silence only) require this much contiguous silence.")

    args = ap.parse_args()

    build_dataset(
        args.raw,
        args.out,
        sr=args.sr,
        add_negatives=args.add_negatives,
        neg_per_pos=args.neg_per_pos,
        neg_min_silence_ms=args.neg_min_silence_ms,
        neg_pad_mode=args.neg_pad_mode,
        neg_max_total=args.neg_max_total,
        neg_use_lead_silence=bool(args.neg_use_lead_silence),
        neg_lead_silence_sec=float(args.neg_lead_silence_sec),
        neg_use_tail_silence=bool(args.neg_use_tail_silence),
    )


if __name__ == "__main__":
    main()