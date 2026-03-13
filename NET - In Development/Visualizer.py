#!/usr/bin/env python3
"""
Improved guitar WAV -> MIDI synth front-end.

Main fixes for "off-key" notes:
- uses the trained STRING head to constrain pitch candidates to a real guitar string range
- estimates a global tuning offset (in cents) from the same short shared buffer
- shifts DSP candidate frequencies by that tuning offset before quantizing to MIDI
- defaults to a safer monophonic-first fusion path when the checkpoint is stage1/monophonic
- keeps silence gating so notes are not always on

This still shares the same 8-16 ms frame between the NN and the DSP detector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None

TARGET_SR = 48000
DEFAULT_FRAME_MS = 16.0
DEFAULT_HOP_MS = 16.0
N_STRINGS = 6
EPS = 1e-9
STANDARD_GUITAR_OPEN_MIDI = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4


def ms_to_samples(ms: float, sr: int) -> int:
    return int(round(ms * sr / 1000.0))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(x), dtype=np.float64) + EPS) + EPS)


def hz_to_midi(freq_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(freq_hz, 1e-8) / 440.0)


def midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def note_name_to_midi(name: str) -> Optional[int]:
    if not name or not isinstance(name, str):
        return None
    s = name.strip().replace("♯", "#").replace("♭", "b")
    pitch_map = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
        "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }
    if len(s) >= 3 and s[1] in ["#", "b"]:
        pc = s[:2]
        octv = s[2:]
    else:
        pc = s[:1]
        octv = s[1:]
    if pc not in pitch_map:
        return None
    try:
        octave = int(octv)
    except ValueError:
        return None
    return (octave + 1) * 12 + pitch_map[pc]


def simple_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    if resample_poly is not None:
        from math import gcd
        g = gcd(sr_in, sr_out)
        up = sr_out // g
        down = sr_in // g
        return resample_poly(x, up, down).astype(np.float32, copy=False)
    dur = len(x) / float(sr_in)
    n_out = int(round(dur * sr_out))
    t_in = np.linspace(0.0, dur, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, dur, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32, copy=False)


def rms_normalize(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x * x, dtype=np.float64) + 1e-12))
    return x / max(rms, 1e-4)


def preemphasis(x: np.ndarray, coef: float) -> np.ndarray:
    if coef <= 0.0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coef * x[:-1]
    return y


def hz_bin_interp(mag: np.ndarray, freqs: np.ndarray, target_hz: float) -> float:
    if target_hz <= freqs[0] or target_hz >= freqs[-1]:
        return 0.0
    idx = np.searchsorted(freqs, target_hz)
    idx = int(np.clip(idx, 1, len(freqs) - 1))
    f0, f1 = freqs[idx - 1], freqs[idx]
    m0, m1 = mag[idx - 1], mag[idx]
    alpha = (target_hz - f0) / max(f1 - f0, 1e-12)
    return float((1.0 - alpha) * m0 + alpha * m1)


def parabolic_peak(y_minus: float, y0: float, y_plus: float) -> float:
    denom = (y_minus - 2.0 * y0 + y_plus)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y_minus - y_plus) / denom


class OnsetNet(nn.Module):
    def __init__(self, note_vocab_size: int, width: int = 128, n_strings: int = 6):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=63, stride=2, padding=31, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=31, stride=2, padding=15, dilation=1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=15, stride=1, padding=14, dilation=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=15, stride=1, padding=28, dilation=4, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=15, stride=1, padding=56, dilation=8, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.note_head = nn.Linear(width, note_vocab_size)
        self.string_head = nn.Linear(width, n_strings)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.feat(x).squeeze(-1)
        return self.note_head(z), self.string_head(z)


@dataclass
class ModelMeta:
    index_to_pitch: List[str]
    index_to_midi: List[int]
    sr: int
    vocab_size: int
    width: int
    stage2: bool
    thresh: float
    preemph_coef: float


def build_model_meta(ckpt: Dict, metadata_path: Optional[str]) -> ModelMeta:
    meta = load_json(metadata_path) if metadata_path and os.path.exists(metadata_path) else {}
    index_to_pitch = meta.get("index_to_pitch", [])
    vocab_size = int(ckpt.get("vocab_size", len(index_to_pitch) or 0))
    width = int(ckpt.get("width", ckpt.get("args", {}).get("width", 96)))
    sr = int(ckpt.get("sr", meta.get("sr", TARGET_SR)))
    stage2 = bool(ckpt.get("stage2", False))
    thresh = float(ckpt.get("thresh", 0.2))
    preemph_coef = float(ckpt.get("args", {}).get("preemph_coef", 0.0))
    if not index_to_pitch:
        index_to_pitch = [str(i) for i in range(vocab_size)]
    if vocab_size <= 0:
        vocab_size = len(index_to_pitch)
    index_to_midi = []
    for i, p in enumerate(index_to_pitch[:vocab_size]):
        m = note_name_to_midi(p)
        index_to_midi.append(int(i if m is None else m))
    while len(index_to_midi) < vocab_size:
        index_to_midi.append(len(index_to_midi))
    while len(index_to_pitch) < vocab_size:
        index_to_pitch.append(str(len(index_to_pitch)))
    return ModelMeta(index_to_pitch, index_to_midi, sr, vocab_size, width, stage2, thresh, preemph_coef)


def load_model(ckpt_path: str, metadata_path: Optional[str], device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_meta = build_model_meta(ckpt, metadata_path)
    model = OnsetNet(model_meta.vocab_size, model_meta.width, N_STRINGS).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model_meta


class ActivityGate:
    def __init__(self, open_db=-48.0, close_db=-56.0, flux_open=0.06, flux_close=0.02, open_hold_frames=1, close_hold_frames=3):
        self.open_db = float(open_db)
        self.close_db = float(close_db)
        self.flux_open = float(flux_open)
        self.flux_close = float(flux_close)
        self.open_hold_frames = int(open_hold_frames)
        self.close_hold_frames = int(close_hold_frames)
        self.is_open = False
        self.prev_mag = None
        self.open_count = 0
        self.close_count = 0

    def step(self, frame: np.ndarray, mag: np.ndarray) -> Dict[str, float]:
        db = rms_db(frame)
        if self.prev_mag is None:
            flux = 0.0
        else:
            diff = mag - self.prev_mag
            flux = float(np.mean(np.maximum(diff, 0.0)) / (np.mean(self.prev_mag) + EPS))
        self.prev_mag = mag.copy()
        should_open = (db >= self.open_db) or (flux >= self.flux_open and db >= self.close_db)
        should_close = (db < self.close_db) and (flux < self.flux_close)
        if not self.is_open:
            self.open_count = self.open_count + 1 if should_open else 0
            if self.open_count >= self.open_hold_frames:
                self.is_open = True
                self.close_count = 0
        else:
            self.close_count = self.close_count + 1 if should_close else 0
            if self.close_count >= self.close_hold_frames:
                self.is_open = False
                self.open_count = 0
        return {"db": db, "flux": flux, "active": float(self.is_open)}


class SharedBufferPolyDetector:
    def __init__(self, midi_candidates: List[int], sr: int, frame_size: int, fft_size: int = 8192, max_harmonics: int = 6, max_polyphony: int = 4):
        self.midi_candidates = sorted(set(int(m) for m in midi_candidates))
        self.sr = int(sr)
        self.frame_size = int(frame_size)
        self.fft_size = int(max(fft_size, 2 ** int(math.ceil(math.log2(frame_size)))))
        self.max_harmonics = int(max_harmonics)
        self.max_polyphony = int(max_polyphony)
        self.window = np.hanning(self.frame_size).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sr)

    def frame_mag(self, frame: np.ndarray) -> np.ndarray:
        xw = frame * self.window
        spec = np.fft.rfft(xw, n=self.fft_size)
        mag = np.abs(spec).astype(np.float32)
        return mag / np.sqrt(self.freqs + 40.0, dtype=np.float32)

    def estimate_peak_hz(self, mag: np.ndarray, lo_hz: float = 70.0, hi_hz: float = 1400.0) -> Optional[float]:
        i0 = int(np.searchsorted(self.freqs, lo_hz))
        i1 = int(np.searchsorted(self.freqs, hi_hz))
        if i1 - i0 < 3:
            return None
        k = i0 + int(np.argmax(mag[i0:i1]))
        if k <= 0 or k >= len(mag) - 1:
            return float(self.freqs[k])
        delta = parabolic_peak(float(mag[k - 1]), float(mag[k]), float(mag[k + 1]))
        kf = float(k) + delta
        return float(kf * self.sr / self.fft_size)

    def candidate_salience(self, mag: np.ndarray, midi_note: int, tuning_cents: float = 0.0) -> float:
        f0 = midi_to_hz(midi_note + tuning_cents / 100.0)
        if f0 < 55.0 or f0 > self.sr * 0.45:
            return 0.0
        score = 0.0
        used = 0
        for h in range(1, self.max_harmonics + 1):
            fh = f0 * h
            if fh >= self.freqs[-1]:
                break
            score += hz_bin_interp(mag, self.freqs, fh) / (h ** 0.72)
            used += 1
        return float(score / max(used, 1)) if used >= 3 else 0.0

    def detect(self, frame: np.ndarray, allowed_midis: Optional[List[int]] = None, tuning_cents: float = 0.0) -> Tuple[np.ndarray, Dict[int, float], Optional[float]]:
        mag = self.frame_mag(frame)
        peak_hz = self.estimate_peak_hz(mag)
        mids = self.midi_candidates if allowed_midis is None else sorted(set(int(m) for m in allowed_midis))
        sal = {m: self.candidate_salience(mag, m, tuning_cents=tuning_cents) for m in mids}
        sal = {m: s for m, s in sal.items() if s > 0.0}
        if not sal:
            return mag, {}, peak_hz
        peak = max(sal.values())
        chosen: Dict[int, float] = {}
        remaining = sal.copy()
        while remaining and len(chosen) < self.max_polyphony:
            m_best = max(remaining, key=remaining.get)
            s_best = remaining[m_best]
            if s_best < 0.35 * peak:
                break
            chosen[m_best] = float(s_best)
            for m in list(remaining.keys()):
                if abs(m - m_best) <= 1 or abs(m - m_best) in [12, 19]:
                    remaining.pop(m, None)
        return mag, chosen, peak_hz


@dataclass
class ActiveNoteState:
    is_on: bool = False
    on_count: int = 0
    off_count: int = 0
    velocity: int = 100


@dataclass
class PitchState:
    midi: Optional[int] = None
    stable_count: int = 0
    hold_count: int = 0


@dataclass
class NoteEvent:
    type: str
    midi: int
    time_s: float
    velocity: int


class NoteTracker:
    def __init__(self, on_frames: int = 1, off_frames: int = 3):
        self.on_frames = int(on_frames)
        self.off_frames = int(off_frames)
        self.states: Dict[int, ActiveNoteState] = {}
        self.events: List[NoteEvent] = []

    def step(self, active_midis: Dict[int, int], t_s: float):
        seen = set(active_midis.keys())
        for midi, vel in active_midis.items():
            st = self.states.setdefault(midi, ActiveNoteState())
            st.velocity = int(vel)
            st.off_count = 0
            st.on_count += 1
            if not st.is_on and st.on_count >= self.on_frames:
                st.is_on = True
                self.events.append(NoteEvent("on", midi, t_s, st.velocity))
        for midi, st in list(self.states.items()):
            if midi in seen:
                continue
            st.on_count = 0
            if st.is_on:
                st.off_count += 1
                if st.off_count >= self.off_frames:
                    st.is_on = False
                    self.events.append(NoteEvent("off", midi, t_s, 0))

    def flush_all(self, t_s: float):
        for midi, st in list(self.states.items()):
            if st.is_on:
                self.events.append(NoteEvent("off", midi, t_s, 0))
                st.is_on = False


class StickyMonophonicDecoder:
    def __init__(self, semitone_stick: float = 0.22, semitone_switch_penalty: float = 0.18,
                 min_new_note_frames: int = 2, hold_frames: int = 3, octave_penalty: float = 0.10):
        self.semitone_stick = float(semitone_stick)
        self.semitone_switch_penalty = float(semitone_switch_penalty)
        self.min_new_note_frames = int(min_new_note_frames)
        self.hold_frames = int(hold_frames)
        self.octave_penalty = float(octave_penalty)
        self.state = PitchState()

    def _penalty(self, cand: int, prev: Optional[int]) -> float:
        if prev is None:
            return 0.0
        d = abs(int(cand) - int(prev))
        if d == 0:
            return -self.semitone_stick
        pen = self.semitone_switch_penalty * min(d, 12)
        if d in (12, 19):
            pen += self.octave_penalty
        return pen

    def step(self, scores: Dict[int, float], active: bool) -> Dict[int, float]:
        if not active or not scores:
            if self.state.hold_count < self.hold_frames and self.state.midi is not None:
                self.state.hold_count += 1
                return {int(self.state.midi): 0.95 * float(scores.get(self.state.midi, 0.75) if scores else 0.75)}
            self.state = PitchState()
            return {}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_midi = None
        best_score = -1e9
        for midi, sc in ranked[:8]:
            adj = float(sc) - self._penalty(int(midi), self.state.midi)
            if adj > best_score:
                best_score = adj
                best_midi = int(midi)

        assert best_midi is not None

        if self.state.midi is None:
            self.state.midi = best_midi
            self.state.stable_count = 1
            self.state.hold_count = 0
            return {best_midi: float(scores[best_midi])}

        if best_midi == self.state.midi:
            self.state.stable_count = min(self.state.stable_count + 1, 999999)
            self.state.hold_count = 0
            return {best_midi: float(scores[best_midi])}

        prev_score = float(scores.get(self.state.midi, 0.0))
        cand_score = float(scores.get(best_midi, 0.0))
        jump = abs(best_midi - self.state.midi)
        margin = 0.08 + 0.03 * min(jump, 4)
        if cand_score >= prev_score + margin:
            self.state.stable_count += 1
        else:
            self.state.stable_count = 0

        if self.state.stable_count >= self.min_new_note_frames:
            self.state.midi = best_midi
            self.state.stable_count = 1
            self.state.hold_count = 0
            return {best_midi: cand_score}

        self.state.hold_count = 0
        return {int(self.state.midi): max(prev_score, 0.82 * cand_score)}


def write_events_json(events: List[NoteEvent], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"type": e.type, "midi": e.midi, "time_s": e.time_s, "velocity": e.velocity} for e in events], f, indent=2)


def write_midi_if_possible(events: List[NoteEvent], path: str):
    try:
        import mido
    except Exception:
        alt = os.path.splitext(path)[0] + ".csv"
        with open(alt, "w", encoding="utf-8") as f:
            f.write("type,midi,time_s,velocity\n")
            for e in events:
                f.write(f"{e.type},{e.midi},{e.time_s:.6f},{e.velocity}\n")
        print(f"[WARN] mido not installed; wrote CSV instead: {alt}")
        return
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo = mido.bpm2tempo(120)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    prev_t = 0.0
    for e in sorted(events, key=lambda z: (z.time_s, 0 if z.type == "off" else 1)):
        dt = max(0.0, e.time_s - prev_t)
        prev_t = e.time_s
        ticks = int(round(mido.second2tick(dt, mid.ticks_per_beat, tempo)))
        if e.type == "on":
            track.append(mido.Message("note_on", note=e.midi, velocity=e.velocity, time=ticks))
        else:
            track.append(mido.Message("note_off", note=e.midi, velocity=0, time=ticks))
    mid.save(path)


def synth_from_events(events: List[NoteEvent], duration_s: float, sr: int, out_path: str):
    n = int(math.ceil(duration_s * sr))
    y = np.zeros(n, dtype=np.float32)
    active = {}

    def add_note(midi: int, start_s: int, end_s: int, velocity: int):
        if end_s <= start_s:
            return
        f0 = midi_to_hz(midi)
        t = np.arange(end_s - start_s, dtype=np.float32) / sr
        sig = (np.sin(2*np.pi*f0*t) + 0.25*np.sin(2*np.pi*2*f0*t) + 0.12*np.sin(2*np.pi*3*f0*t)).astype(np.float32)
        env = np.ones_like(sig)
        a = min(int(0.005*sr), len(env))
        r = min(int(0.02*sr), len(env))
        if a > 0:
            env[:a] *= np.linspace(0.0, 1.0, num=a, endpoint=False, dtype=np.float32)
        if r > 0:
            env[-r:] *= np.linspace(1.0, 0.0, num=r, endpoint=True, dtype=np.float32)
        y[start_s:end_s] += 0.18 * (velocity / 127.0) * env * sig

    for e in sorted(events, key=lambda z: (z.time_s, 0 if z.type == "off" else 1)):
        s = int(round(e.time_s * sr))
        if e.type == "on":
            active[e.midi] = (s, e.velocity)
        elif e.midi in active:
            st, vel = active.pop(e.midi)
            add_note(e.midi, st, s, vel)
    for midi, (st, vel) in active.items():
        add_note(midi, st, n, vel)
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 0.99:
        y = y / peak * 0.99
    sf.write(out_path, y, sr)


def nn_note_probs(model: OnsetNet, frame: np.ndarray, device: str, preemph_coef: float, stage2: bool) -> Tuple[np.ndarray, np.ndarray]:
    x = preemphasis(rms_normalize(frame.astype(np.float32, copy=True)), preemph_coef)
    xt = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        note_logits, string_logits = model(xt)
        note_probs = (torch.sigmoid(note_logits[0]) if stage2 else torch.softmax(note_logits[0], dim=0)).cpu().numpy().astype(np.float32)
        string_probs = torch.softmax(string_logits[0], dim=0).cpu().numpy().astype(np.float32)
    return note_probs, string_probs


def allowed_midis_from_string_probs(index_to_midi: List[int], string_probs: np.ndarray, topk_strings: int = 2, max_fret: int = 24) -> List[int]:
    top_strings = np.argsort(-string_probs)[:max(1, topk_strings)]
    allowed = set()
    for s in top_strings:
        lo = STANDARD_GUITAR_OPEN_MIDI[int(s)]
        hi = lo + max_fret
        for m in index_to_midi:
            if lo <= int(m) <= hi:
                allowed.add(int(m))
    return sorted(allowed)


def estimate_global_tuning_cents(x: np.ndarray, frame_size: int, hop_size: int, detector: SharedBufferPolyDetector, sample_frames: int = 160) -> float:
    cents = []
    total = 1 + max(0, (len(x) - frame_size) // hop_size)
    stride = max(1, total // max(1, sample_frames))
    for fi in range(0, total, stride):
        st = fi * hop_size
        frame = x[st:st + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        if rms_db(frame) < -45.0:
            continue
        mag = detector.frame_mag(frame)
        hz = detector.estimate_peak_hz(mag)
        if hz is None or hz < 70.0 or hz > 1400.0:
            continue
        midi_cont = hz_to_midi(hz)
        cents_off = 100.0 * (midi_cont - round(midi_cont))
        if abs(cents_off) <= 40.0:
            cents.append(cents_off)
    if not cents:
        return 0.0
    return float(np.median(np.array(cents, dtype=np.float32)))


def scores_to_active_midis(scores: Dict[int, float], score_thresh: float, dynamic_ratio: float) -> Dict[int, int]:
    if not scores:
        return {}
    peak = max(scores.values())
    out = {}
    for midi, sc in scores.items():
        if sc >= score_thresh and sc >= dynamic_ratio * peak:
            out[int(midi)] = int(clamp(28 + 99 * sc, 1, 127))
    return out


def fuse_monophonic_locked(poly_notes: Dict[int, float], note_probs: np.ndarray, index_to_midi: List[int], allowed_midis: List[int]) -> Dict[int, float]:
    i_best = int(np.argmax(note_probs))
    nn_top_midi = int(index_to_midi[i_best])
    best_p = float(note_probs[i_best])

    allowed_set = set(allowed_midis) if allowed_midis else set(index_to_midi)
    top_nn = []
    for i in np.argsort(-note_probs)[:6]:
        m = int(index_to_midi[int(i)])
        if m in allowed_set:
            top_nn.append((m, float(note_probs[int(i)])))

    scores: Dict[int, float] = {}
    for midi, sal in poly_notes.items():
        if midi not in allowed_set:
            continue
        dist = abs(midi - nn_top_midi)
        lock = 1.0 / (1.0 + 0.9 * dist)
        scores[midi] = 0.95 * float(sal) + 0.90 * best_p * lock

    for midi, p in top_nn:
        scores[midi] = max(scores.get(midi, 0.0), 1.15 * p)

    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2])


def fuse_poly_stage2(poly_notes: Dict[int, float], note_probs: np.ndarray, index_to_midi: List[int], nn_thresh: float, max_polyphony: int) -> Dict[int, float]:
    out = {}
    peak = max(poly_notes.values()) + EPS if poly_notes else 1.0
    for midi, sal in poly_notes.items():
        out[midi] = sal / peak
    for i, p in enumerate(note_probs):
        if p >= nn_thresh:
            midi = int(index_to_midi[i])
            out[midi] = max(out.get(midi, 0.0), 0.85 * float(p))
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:max_polyphony])


def process_wav(wav_path: str, ckpt_path: str, metadata_path: Optional[str], frame_ms: float, hop_ms: float,
                out_events: str, out_midi: Optional[str], out_synth: Optional[str], fft_size: int,
                max_polyphony: int, device: str, gate_open_db: float, gate_close_db: float,
                gate_flux_open: float, gate_flux_close: float, score_thresh: float, dynamic_ratio: float,
                max_fret: int, topk_strings: int, mono_sticky: bool, min_note_frames: int,
                pitch_hold_ms: float, semitone_stick: float, semitone_switch_penalty: float):
    model, model_meta = load_model(ckpt_path, metadata_path, device)
    x, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = simple_resample(x, sr, TARGET_SR)
    sr = TARGET_SR

    frame_size = ms_to_samples(frame_ms, sr)
    hop_size = ms_to_samples(hop_ms, sr)
    candidate_midis = sorted(set(int(m) for m in model_meta.index_to_midi if 20 <= int(m) <= 108)) or list(range(40, 89))
    poly = SharedBufferPolyDetector(candidate_midis, sr, frame_size, fft_size=fft_size, max_polyphony=max_polyphony)
    tuning_cents = estimate_global_tuning_cents(x, frame_size, hop_size, poly)
    print(f"[INFO] estimated tuning offset: {tuning_cents:+.2f} cents")

    gate = ActivityGate(gate_open_db, gate_close_db, gate_flux_open, gate_flux_close, 1, max(2, int(round(24.0 / max(hop_ms, 1e-6)))))
    tracker = NoteTracker(1, max(2, int(round(24.0 / max(hop_ms, 1e-6)))))
    sticky = None
    if mono_sticky and not model_meta.stage2:
        sticky = StickyMonophonicDecoder(
            semitone_stick=semitone_stick,
            semitone_switch_penalty=semitone_switch_penalty,
            min_new_note_frames=max(1, min_note_frames),
            hold_frames=max(1, int(round(pitch_hold_ms / max(hop_ms, 1e-6)))),
        )
    debug_rows = []
    num_frames = 1 + max(0, (len(x) - frame_size) // hop_size)

    for fi in range(num_frames):
        st = fi * hop_size
        t_s = st / float(sr)
        frame = x[st:st + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        mag = poly.frame_mag(frame)
        gate_info = gate.step(frame, mag)
        if gate_info["active"] < 0.5:
            if sticky is not None:
                sticky.step({}, active=False)
            tracker.step({}, t_s)
            debug_rows.append({"frame": fi, "time_s": t_s, "active": 0, "db": gate_info["db"], "notes": []})
            continue

        note_probs, string_probs = nn_note_probs(model, frame, device, model_meta.preemph_coef, model_meta.stage2)
        allowed_midis = allowed_midis_from_string_probs(model_meta.index_to_midi, string_probs, topk_strings=topk_strings, max_fret=max_fret)
        _, poly_notes, peak_hz = poly.detect(frame, allowed_midis=allowed_midis, tuning_cents=tuning_cents)

        if model_meta.stage2:
            fused = fuse_poly_stage2(poly_notes, note_probs, model_meta.index_to_midi, model_meta.thresh, max_polyphony)
        else:
            fused = fuse_monophonic_locked(poly_notes, note_probs, model_meta.index_to_midi, allowed_midis)

        if sticky is not None:
            fused = sticky.step(fused, active=True)
        active = scores_to_active_midis(fused, score_thresh, dynamic_ratio)
        tracker.step(active, t_s)
        debug_rows.append({
            "frame": fi,
            "time_s": t_s,
            "active": 1,
            "db": gate_info["db"],
            "peak_hz": None if peak_hz is None else float(peak_hz),
            "string_probs": [float(v) for v in string_probs.tolist()],
            "allowed_midis": allowed_midis,
            "notes": sorted(int(m) for m in active.keys()),
        })

    end_time = len(x) / float(sr)
    tracker.flush_all(end_time)
    write_events_json(tracker.events, out_events)
    print(f"[OK] wrote events: {out_events}")
    dbg = os.path.splitext(out_events)[0] + "_frames.json"
    with open(dbg, "w", encoding="utf-8") as f:
        json.dump(debug_rows, f, indent=2)
    print(f"[OK] wrote frame debug: {dbg}")
    if out_midi:
        write_midi_if_possible(tracker.events, out_midi)
        if os.path.exists(out_midi):
            print(f"[OK] wrote midi: {out_midi}")
    if out_synth:
        synth_from_events(tracker.events, end_time, sr, out_synth)
        print(f"[OK] wrote synth preview: {out_synth}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--metadata", default="")
    ap.add_argument("--frame_ms", type=float, default=DEFAULT_FRAME_MS)
    ap.add_argument("--hop_ms", type=float, default=DEFAULT_HOP_MS)
    ap.add_argument("--fft_size", type=int, default=8192)
    ap.add_argument("--max_polyphony", type=int, default=4)
    ap.add_argument("--gate_open_db", type=float, default=-48.0)
    ap.add_argument("--gate_close_db", type=float, default=-56.0)
    ap.add_argument("--gate_flux_open", type=float, default=0.06)
    ap.add_argument("--gate_flux_close", type=float, default=0.02)
    ap.add_argument("--score_thresh", type=float, default=0.52)
    ap.add_argument("--dynamic_ratio", type=float, default=0.72)
    ap.add_argument("--max_fret", type=int, default=24)
    ap.add_argument("--topk_strings", type=int, default=2)
    ap.add_argument("--mono_sticky", type=int, default=1)
    ap.add_argument("--min_note_frames", type=int, default=2)
    ap.add_argument("--pitch_hold_ms", type=float, default=28.0)
    ap.add_argument("--semitone_stick", type=float, default=0.22)
    ap.add_argument("--semitone_switch_penalty", type=float, default=0.18)
    ap.add_argument("--out_events", default="events.json")
    ap.add_argument("--out_midi", default="")
    ap.add_argument("--out_synth", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    process_wav(
        wav_path=args.wav,
        ckpt_path=args.ckpt,
        metadata_path=args.metadata or None,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        out_events=args.out_events,
        out_midi=args.out_midi or None,
        out_synth=args.out_synth or None,
        fft_size=args.fft_size,
        max_polyphony=args.max_polyphony,
        device=args.device,
        gate_open_db=args.gate_open_db,
        gate_close_db=args.gate_close_db,
        gate_flux_open=args.gate_flux_open,
        gate_flux_close=args.gate_flux_close,
        score_thresh=args.score_thresh,
        dynamic_ratio=args.dynamic_ratio,
        max_fret=args.max_fret,
        topk_strings=args.topk_strings,
        mono_sticky=bool(args.mono_sticky),
        min_note_frames=args.min_note_frames,
        pitch_hold_ms=args.pitch_hold_ms,
        semitone_stick=args.semitone_stick,
        semitone_switch_penalty=args.semitone_switch_penalty,
    )


if __name__ == "__main__":
    main()
