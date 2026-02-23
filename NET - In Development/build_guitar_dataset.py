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
DEFAULT_SR = 48000

# --- NEW: sustain end detection tuning ---
SUS_HOP = 128                    # RMS hop in samples (~2.67ms @ 48k)
SUS_FRAME = 512                  # RMS window (~10.7ms @ 48k)
SUS_MIN_AFTER_TRANSIENT_MS = 40  # don't end sustain too early
SUS_MIN_SUSTAIN_MS = 120         # enforce minimum sustain length after transient end
SUS_HANGOVER_MS = 60             # must stay below threshold for this long
SUS_NOISE_PAD_DB = 6.0           # threshold above noise floor
SUS_REL_DROP_DB = 35.0           # threshold relative to peak


# ----------------------------
# Helpers
# ----------------------------
NOTE_RE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+)$')

def normalize_pitch_str(p: str) -> str:
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
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

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
    onset_sample = int(np.clip(onset_sample, 0, max(0, len(y_search) - 1)))
    return onset_sample

def infer_take_id(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        m = re.match(r'.*_(\d+)$', p)
        if m:
            return m.group(1)
    return "0"

def relative_type_from_path(path: str) -> str:
    norm = os.path.normpath(path).split(os.sep)
    if "single" in norm or any(x.startswith("single_") for x in norm):
        return "single"
    if "string" in norm or any(x.startswith("string_") for x in norm):
        return "scale"
    if "chords" in norm:
        return "chord"
    return "unknown"

def write_wav(path: str, y: np.ndarray, sr: int):
    sf.write(path, y, sr, subtype="FLOAT")


# ----------------------------
# NEW: Sustain end estimation
# ----------------------------
def _rms_env(y: np.ndarray, frame: int = SUS_FRAME, hop: int = SUS_HOP) -> np.ndarray:
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    n = 1 + (len(y) - frame) // hop
    env = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        w = y[s:s+frame]
        env[i] = np.sqrt(np.mean(w*w) + 1e-12)
    return env

def _to_db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, 1e-12))

def estimate_sustain_end(y: np.ndarray, sr: int, transient_end: int) -> Tuple[int, Dict[str, float]]:
    """
    Returns (sustain_end_sample, debug_dict).
    sustain_end_sample is clamped to [transient_end+1, len(y)].
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

    # noise floor estimate from tail
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
# Parsers
# ----------------------------
def parse_single_filename(stem: str) -> Tuple[str, int]:
    m = re.match(r'^(.+?)_(\d+)$', stem)
    if not m:
        raise ValueError(f"Single note filename must be like F4_5.wav, got: {stem}")
    pitch = normalize_pitch_str(m.group(1))
    string_idx = int(m.group(2))
    return pitch, string_idx

def parse_scale_filename(stem: str) -> Tuple[str, str]:
    m = re.match(r'^(.+?)-(.+?)$', stem)
    if not m:
        raise ValueError(f"Scale filename must be like E2-E4.wav, got: {stem}")
    start_pitch = normalize_pitch_str(m.group(1))
    end_pitch = normalize_pitch_str(m.group(2))
    return start_pitch, end_pitch

def parse_chord_filename(stem: str) -> Dict[int, Optional[str]]:
    tokens = stem.split('-')
    string_map: Dict[int, Optional[str]] = {}
    for t in tokens:
        t = t.strip()
        if not t:
            continue
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
        "sustain_detection": {
            "hop": SUS_HOP,
            "frame": SUS_FRAME,
            "min_after_transient_ms": SUS_MIN_AFTER_TRANSIENT_MS,
            "min_sustain_ms": SUS_MIN_SUSTAIN_MS,
            "hangover_ms": SUS_HANGOVER_MS,
            "noise_pad_db": SUS_NOISE_PAD_DB,
            "rel_drop_db": SUS_REL_DROP_DB,
        }
    }

def encode_multi_hot(notes_midi: List[int], midi_to_index: Dict[int, int], vocab_size: int) -> List[int]:
    v = np.zeros((vocab_size,), dtype=np.int32)
    for m in notes_midi:
        if m in midi_to_index:
            v[midi_to_index[m]] = 1
    return v.tolist()

def align_and_label_transient(y: np.ndarray, sr: int, onset_sample: int) -> Tuple[np.ndarray, int, int, int]:
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

    discovered = []
    all_midis: List[int] = []

    # Pass 1: discover + build vocab
    for wav_path in iter_wavs(raw_root):
        rel_type = relative_type_from_path(wav_path)
        stem = os.path.splitext(os.path.basename(wav_path))[0]

        try:
            if rel_type == "single":
                pitch, _ = parse_single_filename(stem)
                all_midis.append(pitch_to_midi(pitch))
                discovered.append((wav_path, rel_type))
            elif rel_type == "scale":
                start_pitch, _ = parse_scale_filename(stem)
                start_midi = pitch_to_midi(start_pitch)
                for i in range(24):
                    all_midis.append(start_midi + i)
                discovered.append((wav_path, rel_type))
            elif rel_type == "chord":
                smap = parse_chord_filename(stem)
                for _, p in smap.items():
                    if p is not None:
                        all_midis.append(pitch_to_midi(p))
                discovered.append((wav_path, rel_type))
            else:
                # best-effort fallback
                if "_" in stem and re.search(r'_[0-5]$', stem):
                    pitch, _ = parse_single_filename(stem)
                    all_midis.append(pitch_to_midi(pitch))
                    discovered.append((wav_path, "single"))
                elif "-" in stem and re.match(r'.+?-.+?', stem):
                    if "_" in stem:
                        smap = parse_chord_filename(stem)
                        for _, p in smap.items():
                            if p is not None:
                                all_midis.append(pitch_to_midi(p))
                        discovered.append((wav_path, "chord"))
                    else:
                        start_pitch, _ = parse_scale_filename(stem)
                        start_midi = pitch_to_midi(start_pitch)
                        for i in range(24):
                            all_midis.append(start_midi + i)
                        discovered.append((wav_path, "scale"))
                else:
                    print(f"[WARN] Skipping unrecognized file: {wav_path}")
        except Exception as e:
            print(f"[WARN] Failed to parse {wav_path}: {e}")

    if not all_midis:
        raise RuntimeError("No parsable notes found. Check folder structure/filenames.")

    meta = build_vocab(all_midis)
    vocab_midis = meta["midi_vocab"]
    midi_to_index = {int(k): int(v) for k, v in meta["midi_to_index"].items()}
    vocab_size = len(vocab_midis)

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
                            string_map[sidx] = -1
                        else:
                            if p is None:
                                string_map[sidx] = -1
                            else:
                                m = pitch_to_midi(p)
                                notes_midi.append(m)
                                string_map[sidx] = m

                onset = detect_onset_sample(y, sr)
                y_out, onset_out, t_start, t_end = align_and_label_transient(y, sr, onset)

                # NEW: compute sustain end on the aligned clip
                sus_end, sus_dbg = estimate_sustain_end(y_out, sr, t_end)

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

                    "transient_window_start": int(t_start),
                    "transient_window_end": int(t_end),

                    # NEW: sustain ends at detected decay-to-noise
                    "sustain_region": [int(t_end), int(sus_end)],
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
                    "type": rel_type,
                }) + "\n")

                sample_idx += 1

            elif rel_type == "scale":
                # Timed scale: ignore 1st second, take next 24 seconds as segments
                y, _sr = read_audio_mono(wav_path, sr)
                start_pitch, _ = parse_scale_filename(stem)
                start_midi = pitch_to_midi(start_pitch)
                one_sec = sr

                for i in range(24):
                    seg_start = (i + 1) * one_sec
                    seg_end = min((i + 2) * one_sec, len(y))
                    if seg_start >= len(y) or seg_end <= seg_start:
                        break

                    seg = y[seg_start:seg_end].copy()
                    midi_note = start_midi + i
                    notes_midi = [midi_note]

                    onset_local = detect_onset_sample(seg, sr, search_limit_ms=ONSET_SEARCH_MAX_MS)
                    seg_out, onset_out, t_start, t_end = align_and_label_transient(seg, sr, onset_local)

                    # NEW: compute sustain end on aligned segment clip
                    sus_end, sus_dbg = estimate_sustain_end(seg_out, sr, t_end)

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

                        "string_map_midi": {str(k): -1 for k in range(6)},

                        "transient_window_start": int(t_start),
                        "transient_window_end": int(t_end),

                        # NEW
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

            else:
                continue

    print(f"Done. Wrote {sample_idx} samples to: {out_root}")
    print(f"- metadata: {os.path.join(out_root, 'metadata.json')}")
    print(f"- manifest: {os.path.join(out_root, 'manifest.jsonl')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="raw", help="Path to raw/ directory")
    ap.add_argument("--out", type=str, default="dataset", help="Output dataset directory")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR, help="Target sample rate (Daisy Seed = 48000)")
    args = ap.parse_args()

    build_dataset(args.raw, args.out, sr=args.sr)

if __name__ == "__main__":
    main()