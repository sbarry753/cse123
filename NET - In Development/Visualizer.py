#!/usr/bin/env python3
"""
onsetnet_visualizer_duplex_synth_grace.py

Adds WAV sample synth support with looping:
- Use --synth-wave wav
- Use --wav-file path/to/file.wav
- Use --wav-root-midi to set the source sample pitch
- Use --wav-loop to loop the sample

Fixes your “E4 ↔ none” rapid toggling + synth noise when NN drops out:
- Adds a *voicing gate* (based on RMS dBFS) so we only consider notes while DI is actually sounding.
- Adds *note grace / hangover* time: if NN becomes uncertain but audio is still voiced, we keep the last note.
- Adds *pitch lock tracking* during grace: uses FFT search around the last MIDI (not top-K) to keep frequency stable.
- Adds *hysteresis* (enter vs exit thresholds) for note confidence + voicing so it doesn’t chatter.
- Synth now only plays when a note is “active” (after onset) and stays stable through the grace window.

Install:
  pip install numpy sounddevice pyqtgraph PyQt5 torch

Run:
  python onsetnet_visualizer_duplex_synth_grace.py --list-devices
  python onsetnet_visualizer_duplex_synth_grace.py --device "WASAPI" --wasapi-exclusive --synth \
    --checkpoint onset_best.pt --metadata metadata.json

WAV sample synth example:
  python onsetnet_visualizer_duplex_synth_grace.py --device "WASAPI" --wasapi-exclusive --synth \
    --synth-wave wav --wav-file mysample.wav --wav-root-midi 64 --wav-loop \
    --checkpoint onset_best.pt --metadata metadata.json

Notes:
- This assumes your checkpoint has picked_head and sustain_head.
"""

import argparse
import json
import math
import re
import sys
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import sounddevice as sd

import torch
import torch.nn as nn

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets


# =============================================================================
# WAV helpers
# =============================================================================
def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """
    Load a PCM/float WAV file as mono float32 in [-1, 1].
    Supports 8/16/24/32-bit PCM and 32-bit float WAVs.
    """
    with wave.open(path, "rb") as wf:
        n_channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        sr = int(wf.getframerate())
        n_frames = int(wf.getnframes())
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        # 8-bit PCM unsigned
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0

    elif sampwidth == 2:
        # 16-bit PCM
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    elif sampwidth == 3:
        # 24-bit PCM
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        y = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = (y & 0x800000) != 0
        y = y.astype(np.int32)
        y[sign] -= 1 << 24
        x = y.astype(np.float32) / 8388608.0

    elif sampwidth == 4:
        # Try float32 first, then int32 PCM fallback
        try:
            x = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 8.0:
                raise ValueError("Not float32 WAV payload")
        except Exception:
            x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
    return x, sr


# =============================================================================
# Synth
# =============================================================================
class MonoSynth:
    def __init__(
        self,
        sr: int,
        waveform: str = "sine",
        attack_ms: float = 2.0,
        release_ms: float = 35.0,
        gain: float = 0.14,
        wav_data: Optional[np.ndarray] = None,
        wav_sr: Optional[int] = None,
        wav_root_hz: float = 440.0,
        wav_loop: bool = True,
    ):
        self.sr = int(sr)
        self.waveform = str(waveform)
        self.phase = 0.0
        self.freq = 440.0
        self.gain = float(gain)

        self.env = 0.0
        self.target = 0.0
        self.attack_samps = max(1, int(round(attack_ms * 1e-3 * self.sr)))
        self.release_samps = max(1, int(round(release_ms * 1e-3 * self.sr)))

        # WAV mode
        self.wav_data = None if wav_data is None else np.asarray(wav_data, dtype=np.float32).reshape(-1)
        self.wav_sr = int(wav_sr) if wav_sr is not None else self.sr
        self.wav_root_hz = float(wav_root_hz)
        self.wav_loop = bool(wav_loop)
        self.wav_pos = 0.0

        self._lock = threading.Lock()

    def set_freq(self, f_hz: float):
        if np.isfinite(f_hz) and f_hz > 5.0:
            with self._lock:
                self.freq = float(f_hz)

    def set_active(self, on: bool):
        with self._lock:
            self.target = 1.0 if on else 0.0

    def set_params(
        self,
        *,
        waveform: Optional[str] = None,
        gain: Optional[float] = None,
        attack_ms: Optional[float] = None,
        release_ms: Optional[float] = None,
    ):
        with self._lock:
            if waveform is not None:
                self.waveform = str(waveform)
            if gain is not None:
                self.gain = float(gain)
            if attack_ms is not None:
                self.attack_samps = max(1, int(round(float(attack_ms) * 1e-3 * self.sr)))
            if release_ms is not None:
                self.release_samps = max(1, int(round(float(release_ms) * 1e-3 * self.sr)))

    def set_wav(
        self,
        wav_data: np.ndarray,
        wav_sr: int,
        *,
        root_hz: float = 440.0,
        loop: bool = True,
    ):
        with self._lock:
            self.wav_data = np.asarray(wav_data, dtype=np.float32).reshape(-1)
            self.wav_sr = int(wav_sr)
            self.wav_root_hz = float(root_hz)
            self.wav_loop = bool(loop)
            self.wav_pos = 0.0

    def _render_wav(
        self,
        n: int,
        freq: float,
        gain: float,
        target: float,
        attack_samps: int,
        release_samps: int,
    ) -> np.ndarray:
        out = np.zeros((n,), dtype=np.float32)
        if self.wav_data is None or self.wav_data.size == 0:
            return out

        src = self.wav_data
        src_len = int(src.size)
        pos = float(self.wav_pos)
        env = float(self.env)

        step_up = 1.0 / attack_samps
        step_dn = 1.0 / release_samps

        # Playback increment in source-samples per output-sample.
        # freq == wav_root_hz -> original sample pitch/speed.
        incr = (float(self.wav_sr) / float(self.sr)) * (float(freq) / max(1e-6, float(self.wav_root_hz)))

        for i in range(n):
            if env < target:
                env = min(target, env + step_up)
            elif env > target:
                env = max(target, env - step_dn)

            if self.wav_loop:
                while pos >= src_len:
                    pos -= src_len
                while pos < 0.0:
                    pos += src_len

                i0 = int(pos) % src_len
                i1 = (i0 + 1) % src_len
                frac = pos - float(int(pos))
                s = (1.0 - frac) * float(src[i0]) + frac * float(src[i1])
            else:
                if pos >= src_len - 1:
                    s = 0.0
                else:
                    i0 = int(pos)
                    i1 = min(i0 + 1, src_len - 1)
                    frac = pos - float(i0)
                    s = (1.0 - frac) * float(src[i0]) + frac * float(src[i1])

            out[i] = float(s) * (gain * env)
            pos += incr

        self.wav_pos = pos
        self.env = env
        return out

    def render(self, n: int) -> np.ndarray:
        n = int(n)
        out = np.zeros((n,), dtype=np.float32)
        if n <= 0:
            return out

        with self._lock:
            freq = float(self.freq)
            target = float(self.target)
            gain = float(self.gain)
            waveform = str(self.waveform)
            attack_samps = int(self.attack_samps)
            release_samps = int(self.release_samps)

        if waveform == "wav":
            return self._render_wav(n, freq, gain, target, attack_samps, release_samps)

        step_up = (1.0 / attack_samps)
        step_dn = (1.0 / release_samps)

        ph = float(self.phase)
        env = float(self.env)

        for i in range(n):
            if env < target:
                env = min(target, env + step_up)
            elif env > target:
                env = max(target, env - step_dn)

            if waveform == "square":
                s = 1.0 if ph < np.pi else -1.0
            elif waveform == "saw":
                s = (ph / np.pi) - 1.0
            else:
                s = math.sin(ph)

            out[i] = float(s) * (gain * env)

            ph += (2.0 * np.pi) * (freq / self.sr)
            if ph >= 2.0 * np.pi:
                ph -= 2.0 * np.pi

        self.phase = ph
        self.env = env
        return out


# =============================================================================
# Model (matches your training)
# =============================================================================
class OnsetNet(nn.Module):
    def __init__(self, note_vocab_size: int, width: int = 64, n_strings: int = 6, has_pick_sustain: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, width // 2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width // 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(width // 2, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.note_head = nn.Linear(width, note_vocab_size)
        self.string_head = nn.Linear(width, n_strings)

        self.has_pick_sustain = bool(has_pick_sustain)
        if self.has_pick_sustain:
            self.picked_head = nn.Linear(width, 1)
            self.sustain_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor):
        z = self.conv(x).squeeze(-1)
        note_logits = self.note_head(z)
        string_logits = self.string_head(z)
        if self.has_pick_sustain:
            picked_logit = self.picked_head(z).squeeze(-1)
            sustain_logit = self.sustain_head(z).squeeze(-1)
            return note_logits, string_logits, picked_logit, sustain_logit
        return note_logits, string_logits


def build_model_from_checkpoint(ckpt: dict, device: str = "cpu"):
    sd_ = ckpt["model_state"]
    note_vocab_size = int(sd_["note_head.weight"].shape[0])
    width = int(sd_["note_head.weight"].shape[1])
    n_strings = int(sd_["string_head.weight"].shape[0])

    has_pick = ("picked_head.weight" in sd_) and ("picked_head.bias" in sd_)
    has_sus = ("sustain_head.weight" in sd_) and ("sustain_head.bias" in sd_)
    has_pick_sustain = bool(has_pick and has_sus)

    model = OnsetNet(note_vocab_size=note_vocab_size, width=width, n_strings=n_strings, has_pick_sustain=has_pick_sustain).to(device)
    model.load_state_dict(sd_, strict=has_pick_sustain)
    model.eval()
    return model, note_vocab_size, width, n_strings, has_pick_sustain


# =============================================================================
# Music utils
# =============================================================================
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))

def midi_to_name(m: int) -> str:
    o = (m // 12) - 1
    n = NOTE_NAMES[m % 12]
    return f"{n}{o}"

def hz_to_midi(f_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(1e-12, f_hz) / 440.0)


# =============================================================================
# Device selection helpers
# =============================================================================
def list_audio_devices() -> str:
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    lines = []
    for i, d in enumerate(devs):
        ha = hostapis[d["hostapi"]]["name"]
        lines.append(f"[{i:2d}] {d['name']}  (hostapi={ha})  in={d['max_input_channels']}  out={d['max_output_channels']}")
    return "\n".join(lines)

def _find_device(kind: str, prefer: Optional[str]) -> int:
    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    want_in = (kind == "input")
    want_out = (kind == "output")

    candidates = []
    for i, d in enumerate(devs):
        if want_in and d.get("max_input_channels", 0) <= 0:
            continue
        if want_out and d.get("max_output_channels", 0) <= 0:
            continue
        name = str(d.get("name", ""))
        ha = hostapis[d["hostapi"]]["name"]
        candidates.append((i, name, ha))

    if prefer is not None:
        s = str(prefer).strip()
        if re.fullmatch(r"\d+", s):
            idx = int(s)
            if 0 <= idx < len(devs):
                d = devs[idx]
                if want_in and d.get("max_input_channels", 0) <= 0:
                    raise RuntimeError(f"Device {idx} has no input channels.\n\n{list_audio_devices()}")
                if want_out and d.get("max_output_channels", 0) <= 0:
                    raise RuntimeError(f"Device {idx} has no output channels.\n\n{list_audio_devices()}")
                return idx
            raise RuntimeError(f"Device index {idx} invalid.\n\n{list_audio_devices()}")
        p = s.lower()
        for i, name, ha in candidates:
            if p in name.lower() or p in ha.lower():
                return i
        raise RuntimeError(f"No {kind} device matched '{prefer}'.\n\n{list_audio_devices()}")

    for i, name, ha in candidates:
        if "asio" in name.lower() or "asio" in ha.lower():
            return i
    for i, name, ha in candidates:
        if "wasapi" in ha.lower():
            return i

    return sd.default.device[0 if want_in else 1]

def find_input_device(prefer: Optional[str]) -> int:
    return _find_device("input", prefer)

def find_output_device(prefer: Optional[str]) -> int:
    return _find_device("output", prefer)


# =============================================================================
# Ring buffer
# =============================================================================
class RingBuffer1D:
    def __init__(self, size: int):
        self.size = int(size)
        self.buf = np.zeros((self.size,), dtype=np.float32)
        self.w = 0
        self.filled = 0
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = int(x.size)
        if n <= 0:
            return
        if n >= self.size:
            x = x[-self.size :]
            n = self.size
        with self.lock:
            end = self.w + n
            if end <= self.size:
                self.buf[self.w:end] = x
            else:
                k = self.size - self.w
                self.buf[self.w:] = x[:k]
                self.buf[: end - self.size] = x[k:]
            self.w = (self.w + n) % self.size
            self.filled = min(self.size, self.filled + n)

    def get_last(self, n: int) -> np.ndarray:
        n = int(n)
        n = min(n, self.size)
        with self.lock:
            if self.filled <= 0:
                return np.zeros((n,), dtype=np.float32)
            start = (self.w - n) % self.size
            if start < self.w:
                out = self.buf[start:self.w].copy()
            else:
                out = np.concatenate([self.buf[start:], self.buf[: self.w]]).copy()

        if out.size < n:
            pad = np.zeros((n,), dtype=np.float32)
            pad[-out.size :] = out
            return pad
        return out


# =============================================================================
# DSP helpers
# =============================================================================
def preemph(x: np.ndarray, a: float) -> np.ndarray:
    a = float(a)
    if a <= 0.0 or x.size < 2:
        return x.astype(np.float32, copy=False)
    y = x.astype(np.float32, copy=True)
    y[1:] = y[1:] - a * y[:-1]
    return y

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def dbfs_from_rms(r: float) -> float:
    return 20.0 * math.log10(max(float(r), 1e-12))

def rms_norm_floor(x: np.ndarray, floor_rms: float = 1e-3) -> np.ndarray:
    r = rms(x)
    if r < float(floor_rms):
        return np.zeros_like(x, dtype=np.float32)
    return (x / r).astype(np.float32, copy=False)

def parabolic_refine(mag: np.ndarray, k: int) -> float:
    if k <= 0 or k >= mag.size - 1:
        return float(k)
    a, b, c = float(mag[k - 1]), float(mag[k]), float(mag[k + 1])
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return float(k)
    delta = 0.5 * (a - c) / denom
    return float(k) + float(delta)


# =============================================================================
# Guided FFT
# =============================================================================
@dataclass
class GuidedResult:
    midi: int
    prob: float
    refined_hz: float
    harm_score: float
    peak_mag: float
    lo_hz: float
    hi_hz: float

def guided_fft_for_targets(
    fft_mag: np.ndarray,
    sr: int,
    fft_size: int,
    targets: List[Tuple[int, float]],
    cents: float,
) -> List[GuidedResult]:
    bin_hz = sr / float(fft_size)
    out: List[GuidedResult] = []
    cents = float(cents)
    ratio = 2.0 ** (cents / 1200.0)

    for midi, prob in targets:
        c_hz = midi_to_hz(midi)
        lo = c_hz / ratio
        hi = c_hz * ratio

        lo_bin = max(2, int(math.floor(lo / bin_hz)))
        hi_bin = min(int(fft_mag.size - 2), int(math.ceil(hi / bin_hz)))
        if hi_bin <= lo_bin:
            continue

        band = fft_mag[lo_bin : hi_bin + 1]
        k_rel = int(np.argmax(band))
        k = lo_bin + k_rel

        k_ref = parabolic_refine(fft_mag, k)
        f_ref = float(k_ref * bin_hz)

        peak_mag = float(fft_mag[k])
        harm = peak_mag
        for h in (2, 3, 4):
            hb = int(round(f_ref * h / bin_hz))
            if 0 <= hb < fft_mag.size:
                harm += float(fft_mag[hb]) * (0.7 / h)

        out.append(GuidedResult(
            midi=int(midi),
            prob=float(prob),
            refined_hz=float(f_ref),
            harm_score=float(harm),
            peak_mag=float(peak_mag),
            lo_hz=float(lo),
            hi_hz=float(hi),
        ))

    out.sort(key=lambda r: r.harm_score, reverse=True)
    return out


# =============================================================================
# Pick detector (peak-based EMA + hysteresis + cooldown)
# =============================================================================
class PeakPickDetector:
    def __init__(self, hop_ms: float, tau_ms: float = 12.0, reset_hold_frames: int = 2):
        self.hop_ms = float(hop_ms)
        self.a = float(math.exp(-(self.hop_ms) / max(1e-6, float(tau_ms))))
        self.pick_ema = 0.0
        self.prev = 0.0

        self.armed = True
        self.rising = False
        self.peak_val = 0.0
        self.min_since_reset = 1.0
        self.reset_hold = 0
        self.reset_hold_frames = int(max(1, reset_hold_frames))

        self.cooldown = 0

    def step(
        self,
        pick_raw: float,
        *,
        pick_min_p: float,
        pick_rise_thresh: float,
        pick_fall_reset: float,
        cooldown_frames: int,
        sustain_blocked: bool,
    ) -> Tuple[bool, float, float]:
        if self.cooldown > 0:
            self.cooldown -= 1

        self.prev = self.pick_ema
        self.pick_ema = self.a * self.pick_ema + (1.0 - self.a) * float(pick_raw)
        dp = self.pick_ema - self.prev

        if not self.armed:
            if self.pick_ema < float(pick_fall_reset):
                self.reset_hold += 1
                if self.reset_hold >= self.reset_hold_frames:
                    self.armed = True
                    self.rising = False
                    self.peak_val = 0.0
                    self.min_since_reset = self.pick_ema
                    self.reset_hold = 0
            else:
                self.reset_hold = 0

        trigger = False
        if self.armed and self.cooldown == 0 and (not sustain_blocked):
            self.min_since_reset = min(self.min_since_reset, self.pick_ema)

            if dp > 0:
                self.rising = True
                self.peak_val = max(self.peak_val, self.pick_ema)
            elif self.rising and dp <= 0:
                peak = self.peak_val
                rise_amt = peak - self.min_since_reset
                trigger = (peak >= float(pick_min_p)) and (rise_amt >= float(pick_rise_thresh))
                if trigger:
                    self.cooldown = int(max(1, cooldown_frames))
                    self.armed = False
                    self.rising = False
                    self.peak_val = 0.0
                    self.min_since_reset = 1.0
                    self.reset_hold = 0
                else:
                    self.rising = False
                    self.peak_val = 0.0

        return trigger, float(self.pick_ema), float(dp)


# =============================================================================
# UI bits
# =============================================================================
class LabeledSlider(QtWidgets.QWidget):
    def __init__(self, label: str, mn: float, mx: float, step: float, init: float, fmt: str):
        super().__init__()
        self.fmt = fmt
        self.mn, self.mx, self.step = float(mn), float(mx), float(step)
        self.scale = 1.0 / self.step

        self.lab = QtWidgets.QLabel(label)
        self.lab.setStyleSheet("color:#3a4050; font-family:monospace; font-size:11px;")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(int(round(self.mn * self.scale)))
        self.slider.setMaximum(int(round(self.mx * self.scale)))
        self.slider.setValue(int(round(float(init) * self.scale)))

        self.val = QtWidgets.QLabel(self.fmt_value(self.value()))
        self.val.setFixedWidth(72)
        self.val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.val.setStyleSheet("color:#00ff88; font-family:monospace; font-size:11px;")

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.lab)
        row.addWidget(self.slider, 1)
        row.addWidget(self.val)
        row.setContentsMargins(0, 0, 0, 0)
        self.setLayout(row)
        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, _):
        self.val.setText(self.fmt_value(self.value()))

    def value(self) -> float:
        return float(self.slider.value()) / self.scale

    def fmt_value(self, v: float) -> str:
        return self.fmt.format(v)


@dataclass
class NNOut:
    note_probs: np.ndarray
    pick_score: float
    sus_score: float
    string_probs: Optional[np.ndarray] = None


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.setWindowTitle("ONSETNET SYNTH · Visualizer (Grace/Hysteresis)")
        self.resize(1220, 800)

        pg.setConfigOptions(antialias=True)

        self._bg = "#0a0c0f"
        self._surface = "#111418"
        self._border = "#1e2530"
        self._accent = "#00ff88"
        self._accent2 = "#ff4466"
        self._accent3 = "#4488ff"
        self._dim = "#3a4050"
        self._text = "#c8d0dc"

        self.setStyleSheet(
            f"""
            QMainWindow {{ background:{self._bg}; }}
            QWidget {{ background:{self._bg}; color:{self._text}; }}
            QPushButton {{
              border: 1px solid {self._border};
              padding: 10px;
              font-family: Arial;
              letter-spacing: 1px;
            }}
            QPushButton#startBtn {{
              border-color: {self._accent};
              color: {self._accent};
              font-weight: 600;
            }}
            QPushButton#stopBtn {{
              border-color: {self._accent2};
              color: {self._accent2};
              font-weight: 600;
            }}
            QLabel#title {{
              color: {self._accent};
              font-size: 26px;
              font-weight: 700;
              letter-spacing: 3px;
            }}
            """
        )

        # audio
        self.stream: Optional[sd.Stream] = None
        self.sr = 48000
        self.running = False

        # nn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[nn.Module] = None
        self.midi_vocab: Optional[List[float]] = None
        self.index_to_pitch: Optional[List[str]] = None
        self.vocab_size: int = 0

        # Core timing: locked 8ms window, hop configurable
        self.win_ms = 8.0
        self.hop_ms = float(args.hop_ms)
        self.topk = int(args.topk)

        # thresholds
        self.nn_on = float(args.nn_on)
        self.nn_off = float(args.nn_off)

        # FFT guidance
        self.fft_cents = float(args.fft_cents)
        self.fft_weight = float(args.fft_weight)
        self.fft_min_hz = float(args.fft_min_hz)

        # pick
        self.pick_min_p = float(args.pick_min_p)
        self.pick_rise_thresh = float(args.pick_rise)
        self.pick_fall_reset = float(args.pick_reset)
        self.sus_hold_ms = float(args.sus_hold_ms)
        self.sustain_block = float(args.sustain_block)
        self.preemph_coef = float(args.preemph)

        # voicing gate + grace/hold
        self.voiced_db_on = float(args.voiced_db_on)
        self.voiced_db_off = float(args.voiced_db_off)
        self.note_grace_ms = float(args.note_grace_ms)
        self.min_note_ms = float(args.min_note_ms)
        self.freq_smooth = float(args.freq_smooth)

        # internal state
        self.pickdet: Optional[PeakPickDetector] = None
        self.ring: Optional[RingBuffer1D] = None

        # synth
        self.synth_enabled = bool(args.synth)
        self.synth: Optional[MonoSynth] = None

        # history
        self.pick_hist_len = 220
        self.pick_hist = np.zeros((self.pick_hist_len,), dtype=np.float32)
        self.delta_hist = np.zeros((self.pick_hist_len,), dtype=np.float32)
        self.onset_hist = np.zeros((self.pick_hist_len,), dtype=np.float32)
        self.pick_hist_i = 0

        # note state machine
        self.note_active = False
        self.note_midi: Optional[int] = None
        self.note_name = "—"
        self.note_hz = float("nan")

        self.voiced = False
        self.grace_frames_left = 0
        self.min_frames_left = 0

        # UI
        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.tick)

        if args.checkpoint and args.metadata:
            self.load_nn(args.checkpoint, args.metadata)

    # ------------------------
    # NN
    # ------------------------
    def load_nn(self, checkpoint_path: str, metadata_path: str):
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        midi_vocab = meta.get("midi_vocab", None)
        index_to_pitch = meta.get("index_to_pitch", None)
        if not isinstance(midi_vocab, list) or len(midi_vocab) == 0:
            raise RuntimeError("metadata.json must contain a non-empty 'midi_vocab' list")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model, note_vocab_size, _width, _n_strings, has_ps = build_model_from_checkpoint(ckpt, device=self.device)
        if not has_ps:
            raise RuntimeError("Checkpoint missing picked_head/sustain_head (this UI expects them).")

        self.model = model
        self.midi_vocab = [float(x) for x in midi_vocab]
        self.index_to_pitch = index_to_pitch if isinstance(index_to_pitch, list) else None
        self.vocab_size = int(note_vocab_size)

        self.model_status.setText(f"NN loaded · vocab={note_vocab_size} · device={self.device}")
        self.model_status.setStyleSheet(f"color:{self._accent}; font-family:monospace; font-size:11px;")

    def nn_forward(self, x_norm: np.ndarray) -> NNOut:
        if self.model is None:
            raise RuntimeError("NN not loaded")

        x_t = torch.from_numpy(x_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            note_logits, string_logits, picked_logit, sustain_logit = self.model(x_t)

        note_probs = torch.softmax(note_logits[0].detach().cpu(), dim=0).numpy().astype(np.float32, copy=False)
        string_probs = torch.softmax(string_logits[0].detach().cpu(), dim=0).numpy().astype(np.float32, copy=False)
        pick_score = float(torch.sigmoid(picked_logit[0]).detach().cpu().item())
        sus_score = float(torch.sigmoid(sustain_logit[0]).detach().cpu().item())
        return NNOut(note_probs=note_probs, pick_score=pick_score, sus_score=sus_score, string_probs=string_probs)

    # ------------------------
    # UI helpers
    # ------------------------
    def _section_label(self, txt: str) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel(txt.upper())
        lab.setStyleSheet(f"color:{self._dim}; font-family:monospace; font-size:10px; letter-spacing:2px;")
        return lab

    def _pill(self, name: str, val: str, color: str | None = None) -> QtWidgets.QFrame:
        f = QtWidgets.QFrame()
        f.setStyleSheet(f"background:{self._bg}; border:1px solid {self._border};")
        l = QtWidgets.QVBoxLayout(f)
        l.setContentsMargins(8, 8, 8, 8)
        l.setSpacing(2)
        lab = QtWidgets.QLabel(name.upper())
        lab.setStyleSheet(f"color:{self._dim}; font-family:monospace; font-size:9px; letter-spacing:1px;")
        v = QtWidgets.QLabel(val)
        v.setStyleSheet(f"color:{color or self._accent}; font-family:monospace; font-size:18px;")
        v.setObjectName(f"pill_{name}")
        l.addWidget(lab)
        l.addWidget(v)
        return f

    def _pill_set(self, pill: QtWidgets.QFrame, name: str, txt: str):
        v = pill.findChild(QtWidgets.QLabel, f"pill_{name}")
        if v:
            v.setText(txt)

    def _plot_widget(self, title: str, height: int) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground(self._bg)
        pw.setMinimumHeight(height)
        pw.showGrid(x=True, y=True, alpha=0.2)
        pw.getPlotItem().setTitle(title, color=self._dim, size="10pt")
        pw.getPlotItem().getAxis("bottom").setPen(pg.mkPen(self._border))
        pw.getPlotItem().getAxis("left").setPen(pg.mkPen(self._border))
        pw.getPlotItem().getAxis("bottom").setTextPen(pg.mkPen(self._dim))
        pw.getPlotItem().getAxis("left").setTextPen(pg.mkPen(self._dim))
        return pw

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("ONSETNET SYNTH")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch(1)

        self.model_status = QtWidgets.QLabel("NN: not loaded (use --checkpoint + --metadata)")
        self.model_status.setStyleSheet(f"color:{self._dim}; font-family:monospace; font-size:11px;")
        hdr.addWidget(self.model_status)
        layout.addLayout(hdr, 0, 0, 1, 2)

        # sidebar
        sidebar = QtWidgets.QFrame()
        sidebar.setStyleSheet(f"background:{self._surface}; border:1px solid {self._border};")
        sidebar_l = QtWidgets.QVBoxLayout(sidebar)
        sidebar_l.setContentsMargins(12, 12, 12, 12)
        sidebar_l.setSpacing(10)

        self.btn_start = QtWidgets.QPushButton("START LISTENING")
        self.btn_start.setObjectName("startBtn")
        self.btn_start.clicked.connect(self.start_audio)

        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.clicked.connect(self.stop_audio)
        self.btn_stop.setEnabled(False)

        sidebar_l.addWidget(self.btn_start)
        sidebar_l.addWidget(self.btn_stop)

        # status pills
        self.stat_latency = self._pill("latency", "—")
        self.stat_fps = self._pill("fps", "—", color=self._accent3)
        self.stat_db = self._pill("db", "—", color=self._accent3)
        self.stat_voiced = self._pill("voiced", "—", color=self._accent3)
        self.stat_pick = self._pill("pick", "—", color=self._accent2)
        self.stat_sus = self._pill("sus", "—", color=self._accent3)
        self.stat_conf = self._pill("conf", "—", color=self._accent)
        self.stat_fft = self._pill("fft hz", "—", color=self._accent3)
        self.stat_grace = self._pill("grace", "—", color=self._accent)

        stat_grid = QtWidgets.QGridLayout()
        stat_grid.setSpacing(8)
        stat_grid.addWidget(self.stat_latency, 0, 0)
        stat_grid.addWidget(self.stat_fps, 0, 1)
        stat_grid.addWidget(self.stat_db, 1, 0)
        stat_grid.addWidget(self.stat_voiced, 1, 1)
        stat_grid.addWidget(self.stat_pick, 2, 0)
        stat_grid.addWidget(self.stat_sus, 2, 1)
        stat_grid.addWidget(self.stat_conf, 3, 0)
        stat_grid.addWidget(self.stat_fft, 3, 1)
        stat_grid.addWidget(self.stat_grace, 4, 0, 1, 2)
        sidebar_l.addLayout(stat_grid)

        sidebar_l.addSpacing(6)
        sidebar_l.addWidget(self._section_label("core"))

        self.sl_hop = LabeledSlider("hop ms", 1, 16, 0.5, self.hop_ms, "{:.1f}ms")
        self.sl_topk = LabeledSlider("top-K", 1, 8, 1, self.topk, "{:.0f}")

        self.sl_hop.slider.valueChanged.connect(lambda _: self._sync_params())
        self.sl_topk.slider.valueChanged.connect(lambda _: self._sync_params())

        sidebar_l.addWidget(self.sl_hop)
        sidebar_l.addWidget(self.sl_topk)

        sidebar_l.addSpacing(6)
        sidebar_l.addWidget(self._section_label("confidence hysteresis"))

        self.sl_nn_on = LabeledSlider("nn ON", 0.05, 0.95, 0.01, self.nn_on, "{:.2f}")
        self.sl_nn_off = LabeledSlider("nn OFF", 0.01, 0.90, 0.01, self.nn_off, "{:.2f}")

        self.sl_nn_on.slider.valueChanged.connect(lambda _: self._sync_params())
        self.sl_nn_off.slider.valueChanged.connect(lambda _: self._sync_params())

        sidebar_l.addWidget(self.sl_nn_on)
        sidebar_l.addWidget(self.sl_nn_off)

        sidebar_l.addSpacing(6)
        sidebar_l.addWidget(self._section_label("voicing + grace"))

        self.sl_v_on = LabeledSlider("voiced ON dB", -80, -10, 1, self.voiced_db_on, "{:.0f}dB")
        self.sl_v_off = LabeledSlider("voiced OFF dB", -90, -20, 1, self.voiced_db_off, "{:.0f}dB")
        self.sl_grace = LabeledSlider("note grace ms", 0, 600, 10, self.note_grace_ms, "{:.0f}ms")
        self.sl_min_note = LabeledSlider("min note ms", 0, 300, 10, self.min_note_ms, "{:.0f}ms")
        self.sl_smooth = LabeledSlider("freq smooth", 0.0, 0.95, 0.05, self.freq_smooth, "{:.2f}")

        for sl in (self.sl_v_on, self.sl_v_off, self.sl_grace, self.sl_min_note, self.sl_smooth):
            sl.slider.valueChanged.connect(lambda _: self._sync_params())

        sidebar_l.addWidget(self.sl_v_on)
        sidebar_l.addWidget(self.sl_v_off)
        sidebar_l.addWidget(self.sl_grace)
        sidebar_l.addWidget(self.sl_min_note)
        sidebar_l.addWidget(self.sl_smooth)

        sidebar_l.addSpacing(6)
        sidebar_l.addWidget(self._section_label("fft guidance"))

        self.sl_cents = LabeledSlider("search cents ±", 10, 250, 5, self.fft_cents, "{:.0f}¢")
        self.sl_fft_w = LabeledSlider("fft weight", 0, 1, 0.05, self.fft_weight, "{:.2f}")
        self.sl_fft_min = LabeledSlider("fft min hz", 40, 400, 10, self.fft_min_hz, "{:.0f}")

        for sl in (self.sl_cents, self.sl_fft_w, self.sl_fft_min):
            sl.slider.valueChanged.connect(lambda _: self._sync_params())

        sidebar_l.addWidget(self.sl_cents)
        sidebar_l.addWidget(self.sl_fft_w)
        sidebar_l.addWidget(self.sl_fft_min)

        sidebar_l.addSpacing(6)
        sidebar_l.addWidget(self._section_label("pick detection"))

        self.sl_pick_min = LabeledSlider("pick min", 0.01, 0.90, 0.01, self.pick_min_p, "{:.2f}")
        self.sl_pick_rise = LabeledSlider("pick rise", 0.05, 0.80, 0.01, self.pick_rise_thresh, "{:.2f}")
        self.sl_pick_reset = LabeledSlider("pick reset", 0.01, 0.50, 0.01, self.pick_fall_reset, "{:.2f}")
        self.sl_sus_hold = LabeledSlider("sus hold ms", 50, 2000, 50, self.sus_hold_ms, "{:.0f}ms")
        self.sl_preemph = LabeledSlider("preemph", 0.0, 0.99, 0.01, self.preemph_coef, "{:.2f}")

        for sl in (self.sl_pick_min, self.sl_pick_rise, self.sl_pick_reset, self.sl_sus_hold, self.sl_preemph):
            sl.slider.valueChanged.connect(lambda _: self._sync_params())

        sidebar_l.addWidget(self.sl_pick_min)
        sidebar_l.addWidget(self.sl_pick_rise)
        sidebar_l.addWidget(self.sl_pick_reset)
        sidebar_l.addWidget(self.sl_sus_hold)
        sidebar_l.addWidget(self.sl_preemph)

        sidebar_l.addStretch(1)
        layout.addWidget(sidebar, 1, 0, 2, 1)

        # right side plots
        right = QtWidgets.QWidget()
        right_l = QtWidgets.QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(10)

        self.p_wave = self._plot_widget("waveform · 8ms window", height=140)
        self.curve_wave = self.p_wave.plot(pen=pg.mkPen(self._accent, width=2))

        self.p_fft = self._plot_widget("fft spectrum · guided bands", height=180)
        self.curve_fft = self.p_fft.plot(pen=pg.mkPen(self._accent3, width=2))
        self.band_items = []

        self.p_pick = self._plot_widget("pick score · delta · onsets", height=120)
        self.curve_pick = self.p_pick.plot(pen=pg.mkPen(self._accent2, width=2))
        self.curve_delta = self.p_pick.plot(pen=pg.mkPen("#ff9933", width=2))

        right_l.addWidget(self.p_wave)
        right_l.addWidget(self.p_fft)
        right_l.addWidget(self.p_pick)

        # note panel
        note_panel = QtWidgets.QFrame()
        note_panel.setStyleSheet(f"background:{self._surface}; border:1px solid {self._border};")
        note_l = QtWidgets.QHBoxLayout(note_panel)
        note_l.setContentsMargins(12, 10, 12, 10)
        note_l.setSpacing(12)

        big = QtWidgets.QFrame()
        big.setStyleSheet(f"background:{self._bg}; border:1px solid {self._border};")
        big_l = QtWidgets.QVBoxLayout(big)
        big_l.setContentsMargins(12, 8, 12, 8)
        big_l.setSpacing(4)

        self.lbl_note = QtWidgets.QLabel("—")
        self.lbl_note.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_note.setStyleSheet(f"color:{self._accent}; font-size:64px; font-weight:800;")
        self.lbl_mode = QtWidgets.QLabel("waiting")
        self.lbl_mode.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_mode.setStyleSheet(f"color:{self._dim}; font-family:monospace; font-size:11px; letter-spacing:2px;")
        self.lbl_hz = QtWidgets.QLabel("— Hz")
        self.lbl_hz.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_hz.setStyleSheet(f"color:{self._accent3}; font-family:monospace; font-size:12px;")

        big_l.addWidget(self.lbl_note)
        big_l.addWidget(self.lbl_mode)
        big_l.addWidget(self.lbl_hz)
        big.setFixedWidth(240)
        note_l.addWidget(big)

        self.hist = QtWidgets.QTableWidget(8, 3)
        self.hist.setHorizontalHeaderLabels(["note", "prob", "bar"])
        self.hist.verticalHeader().setVisible(False)
        self.hist.horizontalHeader().setStretchLastSection(True)
        self.hist.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.hist.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.hist.setShowGrid(False)
        self.hist.setStyleSheet(
            f"""
            QTableWidget {{
                background:{self._surface};
                color:{self._text};
                border: none;
                font-family: monospace;
                font-size: 11px;
            }}
            QHeaderView::section {{
                background:{self._bg};
                color:{self._dim};
                border: 1px solid {self._border};
                padding: 4px;
                font-family: monospace;
                font-size: 10px;
            }}
            """
        )
        self.hist.setColumnWidth(0, 70)
        self.hist.setColumnWidth(1, 70)
        note_l.addWidget(self.hist, 1)

        pm = QtWidgets.QFrame()
        pm.setStyleSheet(f"background:{self._bg}; border:1px solid {self._border};")
        pm_l = QtWidgets.QVBoxLayout(pm)
        pm_l.setContentsMargins(10, 8, 10, 8)
        pm_l.setSpacing(6)

        self.pick_label = QtWidgets.QLabel("pick")
        self.pick_label.setAlignment(QtCore.Qt.AlignCenter)
        self.pick_label.setStyleSheet(f"color:{self._dim}; font-family:monospace; font-size:11px; letter-spacing:1px;")

        self.pick_bar = QtWidgets.QProgressBar()
        self.pick_bar.setRange(0, 1000)
        self.pick_bar.setValue(0)
        self.pick_bar.setTextVisible(False)
        self.pick_bar.setOrientation(QtCore.Qt.Vertical)
        self.pick_bar.setFixedSize(32, 120)
        self.pick_bar.setStyleSheet(
            f"""
            QProgressBar {{
              background:{self._border};
              border: none;
            }}
            QProgressBar::chunk {{
              background:{self._accent2};
            }}
            """
        )

        self.pick_event = QtWidgets.QLabel("—")
        self.pick_event.setAlignment(QtCore.Qt.AlignCenter)
        self.pick_event.setStyleSheet(f"color:{self._accent2}; font-family:monospace; font-size:11px; letter-spacing:1px;")

        pm_l.addWidget(self.pick_label)
        pm_l.addWidget(self.pick_bar, 0, QtCore.Qt.AlignCenter)
        pm_l.addWidget(self.pick_event)
        pm.setFixedWidth(90)
        note_l.addWidget(pm)

        right_l.addWidget(note_panel)

        self.status = QtWidgets.QLabel("state: idle")
        self.status.setStyleSheet(f"color:{self._dim}; font-family:monospace; font-size:11px;")
        right_l.addWidget(self.status)

        layout.addWidget(right, 1, 1, 2, 1)
        self._sync_params()

    def _sync_params(self):
        self.hop_ms = float(self.sl_hop.value())
        self.topk = int(round(self.sl_topk.value()))

        self.nn_on = float(self.sl_nn_on.value())
        self.nn_off = float(self.sl_nn_off.value())

        self.voiced_db_on = float(self.sl_v_on.value())
        self.voiced_db_off = float(self.sl_v_off.value())
        self.note_grace_ms = float(self.sl_grace.value())
        self.min_note_ms = float(self.sl_min_note.value())
        self.freq_smooth = float(self.sl_smooth.value())

        self.fft_cents = float(self.sl_cents.value())
        self.fft_weight = float(self.sl_fft_w.value())
        self.fft_min_hz = float(self.sl_fft_min.value())

        self.pick_min_p = float(self.sl_pick_min.value())
        self.pick_rise_thresh = float(self.sl_pick_rise.value())
        self.pick_fall_reset = float(self.sl_pick_reset.value())
        self.sus_hold_ms = float(self.sl_sus_hold.value())
        self.preemph_coef = float(self.sl_preemph.value())

        if self.running:
            self.pickdet = PeakPickDetector(self.hop_ms, tau_ms=12.0, reset_hold_frames=2)

    # ------------------------
    # name mapping
    # ------------------------
    def idx_to_display_name(self, idx: int, midi: int) -> str:
        if self.index_to_pitch and 0 <= idx < len(self.index_to_pitch):
            return str(self.index_to_pitch[idx])
        return midi_to_name(midi)

    # ------------------------
    # audio start/stop (duplex)
    # ------------------------
    def start_audio(self):
        if self.running:
            return

        try:
            in_idx = find_input_device(self.args.device)
            out_idx = find_output_device(self.args.out_device if self.args.out_device is not None else self.args.device)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Device error", str(e))
            return

        try:
            in_info = sd.query_devices(in_idx, "input")
            self.sr = int(in_info["default_samplerate"])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Audio error", f"Could not query devices.\n{e}")
            return

        hop_samp = max(64, int(round(self.hop_ms * 1e-3 * self.sr)))
        win_samp = max(hop_samp, int(round(self.win_ms * 1e-3 * self.sr)))

        self.ring = RingBuffer1D(size=max(win_samp, int(self.sr * 0.25)))
        self.pickdet = PeakPickDetector(self.hop_ms, tau_ms=12.0, reset_hold_frames=2)

        # synth
        if self.synth_enabled:
            wav_data = None
            wav_sr = None

            if self.args.synth_wave == "wav":
                if not self.args.wav_file:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "WAV error",
                        "You selected --synth-wave wav but did not provide --wav-file"
                    )
                    return
                try:
                    wav_data, wav_sr = load_wav_mono(self.args.wav_file)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self,
                        "WAV error",
                        f"Failed to load WAV file:\n\n{e}"
                    )
                    return

            self.synth = MonoSynth(
                sr=self.sr,
                waveform=self.args.synth_wave,
                attack_ms=self.args.synth_attack_ms,
                release_ms=self.args.synth_release_ms,
                gain=self.args.synth_gain,
                wav_data=wav_data,
                wav_sr=wav_sr,
                wav_root_hz=midi_to_hz(float(self.args.wav_root_midi)),
                wav_loop=bool(self.args.wav_loop),
            )
            self.synth.set_active(False)
        else:
            self.synth = None

        self.note_active = False
        self.note_midi = None
        self.note_name = "—"
        self.note_hz = float("nan")
        self.voiced = False
        self.grace_frames_left = 0
        self.min_frames_left = 0

        # FPS
        self._fps_frames = 0
        self._fps_last_t = time.perf_counter()
        self._fps_value = 0

        extra_in = None
        extra_out = None

        if self.args.wasapi_exclusive:
            try:
                extra_in = sd.WasapiSettings(exclusive=True)
                extra_out = sd.WasapiSettings(exclusive=True)
            except Exception:
                extra_in = None
                extra_out = None

        if self.args.asio_in_ch is not None or self.args.asio_out_ch is not None:
            try:
                in_sel = None
                out_sel = None
                if self.args.asio_in_ch is not None:
                    in_sel = [int(self.args.asio_in_ch)]
                if self.args.asio_out_ch is not None:
                    out_sel = [int(self.args.asio_out_ch)]
                extra_in = sd.AsioSettings(channel_selectors=in_sel) if in_sel is not None else extra_in
                extra_out = sd.AsioSettings(channel_selectors=out_sel) if out_sel is not None else extra_out
            except Exception:
                pass

        def cb(indata, outdata, frames, time_info, status):
            x = indata[:, 0].astype(np.float32, copy=False)
            if self.ring is not None:
                self.ring.push(x)

            if self.synth is not None:
                y = self.synth.render(frames)
                outdata[:, 0] = y
            else:
                outdata[:, 0] = 0.0

        try:
            self.stream = sd.Stream(
                samplerate=self.sr,
                blocksize=hop_samp,
                dtype="float32",
                device=(in_idx, out_idx),
                channels=(1, 1),
                callback=cb,
                latency="low",
                extra_settings=(extra_in, extra_out),
            )
            self.stream.start()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Stream error", f"Failed to open duplex stream.\n\n{e}")
            self.stream = None
            return

        self.running = True
        self.timer.start()

        self.status.setText(
            f"state: running · IN[{in_idx}] {sd.query_devices(in_idx)['name']} · OUT[{out_idx}] {sd.query_devices(out_idx)['name']} · sr={self.sr}"
        )
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_audio(self):
        self.running = False
        self.timer.stop()
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None

        if self.synth is not None:
            self.synth.set_active(False)
        self.synth = None

        self.note_active = False
        self.note_midi = None
        self.note_name = "—"
        self.note_hz = float("nan")
        self.lbl_note.setText("—")
        self.lbl_mode.setText("waiting")
        self.lbl_hz.setText("— Hz")
        self.pick_event.setText("—")
        self.pick_bar.setValue(0)

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("state: idle")

    # ------------------------
    # voicing hysteresis
    # ------------------------
    def update_voiced(self, db: float):
        if not self.voiced:
            if db >= self.voiced_db_on:
                self.voiced = True
        else:
            if db <= self.voiced_db_off:
                self.voiced = False

    # ------------------------
    # tick
    # ------------------------
    def tick(self):
        if not self.running or self.ring is None or self.pickdet is None:
            return
        if self.model is None or self.midi_vocab is None:
            self.status.setText("state: running (no NN loaded) · use --checkpoint + --metadata")
            return

        t0 = time.perf_counter()

        win_samp = int(round(self.win_ms * 1e-3 * self.sr))
        x = self.ring.get_last(win_samp)

        # measure RMS for voicing gate
        r = rms(x)
        db = dbfs_from_rms(r)
        self.update_voiced(db)

        # preprocess for NN/FFT
        x_p = preemph(x, self.preemph_coef)
        x_n = rms_norm_floor(x_p, floor_rms=1e-4)

        # FFT size
        n_fft = 4096
        if win_samp > n_fft:
            n_fft = 8192
        if win_samp > n_fft:
            n_fft = 16384

        w = np.hanning(x_n.size).astype(np.float32)
        xw = x_n * w
        if xw.size < n_fft:
            xw = np.pad(xw, (0, n_fft - xw.size))
        else:
            xw = xw[:n_fft]

        X = np.fft.rfft(xw)
        mag = np.abs(X).astype(np.float32)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr).astype(np.float32)
        mag[freqs < float(self.fft_min_hz)] = 0.0

        # NN forward
        nn_out = self.nn_forward(x_n)

        # top-k notes
        V = nn_out.note_probs.size
        k = max(1, int(self.topk))
        idxs = np.argsort(-nn_out.note_probs)[: min(k, V)]

        top_targets: List[Tuple[int, float]] = []
        for ii in idxs.tolist():
            if ii >= len(self.midi_vocab):
                continue
            midi = float(self.midi_vocab[ii])
            if not np.isfinite(midi):
                continue
            top_targets.append((int(round(midi)), float(nn_out.note_probs[ii])))

        # Guided FFT on top targets
        guided = guided_fft_for_targets(mag, self.sr, n_fft, top_targets, cents=self.fft_cents)
        best = guided[0] if guided else None

        nn_conf = float(top_targets[0][1]) if top_targets else 0.0
        refined_hz = float(best.refined_hz) if best else float("nan")

        # Fuse NN + FFT
        fused = 0.0
        if best is not None:
            nz = mag[mag > 0]
            med = float(np.median(nz)) if nz.size else 1e-9
            fft_conf = float(best.harm_score / (med + 1e-9))
            fft_conf = float(np.clip(fft_conf / 10.0, 0.0, 1.0))
            fused = (1.0 - float(self.fft_weight)) * nn_conf + float(self.fft_weight) * fft_conf

        # Pick detector
        sustain_blocked = (float(nn_out.sus_score) >= float(self.sustain_block))
        cooldown_frames = max(4, int(round((self.sus_hold_ms / max(1e-6, self.hop_ms)) * 0.25)))
        trig, pick_ema, dp = self.pickdet.step(
            nn_out.pick_score,
            pick_min_p=self.pick_min_p,
            pick_rise_thresh=self.pick_rise_thresh,
            pick_fall_reset=self.pick_fall_reset,
            cooldown_frames=cooldown_frames,
            sustain_blocked=sustain_blocked,
        )

        # Grace frame bookkeeping
        grace_frames = int(round(self.note_grace_ms / max(1e-6, self.hop_ms)))
        min_frames = int(round(self.min_note_ms / max(1e-6, self.hop_ms)))

        # ------------------------
        # NOTE STATE MACHINE
        # ------------------------
        started = False
        if trig and self.voiced and (best is not None) and (fused >= self.nn_on) and np.isfinite(refined_hz):
            idx0 = int(idxs[0])
            midi0 = int(round(float(self.midi_vocab[idx0]))) if idx0 < len(self.midi_vocab) else int(round(hz_to_midi(refined_hz)))
            name0 = self.idx_to_display_name(idx0, midi0) if idx0 < len(self.midi_vocab) else midi_to_name(midi0)

            self.note_active = True
            self.note_midi = midi0
            self.note_name = name0
            self.note_hz = float(refined_hz)

            self.grace_frames_left = grace_frames
            self.min_frames_left = min_frames

            started = True
            self.pick_event.setText("PICK")

            if self.synth is not None:
                self.synth.set_freq(self.note_hz)
                self.synth.set_active(True)

        # If we have an active note, keep it stable
        if self.note_active and not started:
            if self.min_frames_left > 0:
                self.min_frames_left -= 1

            if self.voiced:
                self.grace_frames_left = grace_frames

                targets = []
                if self.note_midi is not None:
                    targets.append((int(self.note_midi), 1.0))
                if top_targets:
                    targets.append((int(top_targets[0][0]), float(top_targets[0][1])))

                g2 = guided_fft_for_targets(mag, self.sr, n_fft, targets, cents=self.fft_cents)
                if g2:
                    hz2 = float(g2[0].refined_hz)
                    if np.isfinite(hz2) and hz2 > 20:
                        a = float(np.clip(self.freq_smooth, 0.0, 0.99))
                        if np.isfinite(self.note_hz):
                            self.note_hz = a * float(self.note_hz) + (1.0 - a) * hz2
                        else:
                            self.note_hz = hz2

                        if top_targets:
                            midi_nn = int(top_targets[0][0])
                            if float(top_targets[0][1]) >= self.nn_on and abs(midi_nn - int(self.note_midi or midi_nn)) <= 1:
                                self.note_midi = midi_nn
                                self.note_name = midi_to_name(midi_nn)

                        if self.synth is not None:
                            self.synth.set_freq(self.note_hz)

                if (fused < self.nn_off) and (self.min_frames_left <= 0):
                    self.grace_frames_left -= 1
                else:
                    self.grace_frames_left = grace_frames

            else:
                if self.min_frames_left <= 0:
                    self.grace_frames_left -= 1

            if self.grace_frames_left <= 0 and self.min_frames_left <= 0:
                self.note_active = False
                self.note_midi = None
                self.note_name = "—"
                self.note_hz = float("nan")
                self.pick_event.setText("—")
                if self.synth is not None:
                    self.synth.set_active(False)

        # UI: update labels
        self.lbl_note.setText(self.note_name if self.note_active else "—")
        self.lbl_mode.setText("ACTIVE" if self.note_active else "waiting")
        self.lbl_hz.setText(f"{self.note_hz:.1f} Hz" if (self.note_active and np.isfinite(self.note_hz)) else "— Hz")
        self.pick_bar.setValue(int(np.clip(pick_ema, 0.0, 1.0) * 1000.0))

        # FPS
        self._fps_frames += 1
        now = time.perf_counter()
        if (now - self._fps_last_t) >= 1.0:
            self._fps_value = self._fps_frames
            self._fps_frames = 0
            self._fps_last_t = now

        # pills
        self._pill_set(self.stat_pick, "pick", f"{nn_out.pick_score:.3f}")
        self._pill_set(self.stat_sus, "sus", f"{nn_out.sus_score:.3f}")
        self._pill_set(self.stat_conf, "conf", f"{nn_conf:.3f}")
        self._pill_set(self.stat_fft, "fft hz", f"{refined_hz:.1f}" if np.isfinite(refined_hz) else "—")
        self._pill_set(self.stat_fps, "fps", f"{self._fps_value:d}")
        self._pill_set(self.stat_db, "db", f"{db:.1f}")
        self._pill_set(self.stat_voiced, "voiced", "yes" if self.voiced else "no")
        self._pill_set(self.stat_grace, "grace", f"{max(0, self.grace_frames_left)}f / {int(round(self.note_grace_ms/self.hop_ms))}f")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._pill_set(self.stat_latency, "latency", f"{latency_ms:.1f}ms")

        # plot buffers
        i = self.pick_hist_i % self.pick_hist_len
        self.pick_hist[i] = float(pick_ema)
        self.delta_hist[i] = float(max(0.0, dp))
        self.onset_hist[i] = 1.0 if trig else 0.0
        self.pick_hist_i += 1

        # waveform plot
        xs = np.arange(x_n.size, dtype=np.float32)
        self.curve_wave.setData(xs, x_n)

        # fft plot
        max_hz = float(self.args.fft_view_max_hz)
        mask = freqs <= max_hz
        self.curve_fft.setData(freqs[mask], mag[mask])

        # bands
        for it in self.band_items:
            self.p_fft.removeItem(it)
        self.band_items.clear()

        ratio = 2.0 ** (self.fft_cents / 1200.0)
        for j, (midi, prob) in enumerate(top_targets):
            c = midi_to_hz(midi)
            lo = c / ratio
            hi = c * ratio
            color = (0, 255, 136, 55) if j == 0 else (68, 136, 255, 35)
            rgn = pg.LinearRegionItem([lo, hi], brush=color, pen=pg.mkPen((0, 0, 0, 0)))
            rgn.setZValue(-10)
            self.p_fft.addItem(rgn)
            self.band_items.append(rgn)

        # pick plot
        N = min(self.pick_hist_i, self.pick_hist_len)
        idxs_plot = (np.arange(N) + max(0, self.pick_hist_i - N)) % self.pick_hist_len
        p = self.pick_hist[idxs_plot]
        d = self.delta_hist[idxs_plot] * 3.0
        self.curve_pick.setData(np.arange(N), p)
        self.curve_delta.setData(np.arange(N), d)

        # histogram
        top8 = np.argsort(-nn_out.note_probs)[:8]
        for r_i, ii in enumerate(top8.tolist()):
            midi = int(round(float(self.midi_vocab[ii]))) if ii < len(self.midi_vocab) else 0
            name = self.idx_to_display_name(ii, midi)
            pr = float(nn_out.note_probs[ii])

            self.hist.setItem(r_i, 0, QtWidgets.QTableWidgetItem(name))
            self.hist.setItem(r_i, 1, QtWidgets.QTableWidgetItem(f"{pr*100:5.1f}%"))

            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(np.clip(pr, 0.0, 1.0) * 1000.0))
            bar.setTextVisible(False)
            bar.setStyleSheet(
                f"""
                QProgressBar {{ background:{self._bg}; border:1px solid {self._border}; }}
                QProgressBar::chunk {{ background:{self._accent if pr >= self.nn_on else self._accent3}; }}
                """
            )
            self.hist.setCellWidget(r_i, 2, bar)

        self.status.setText(
            f"state: {'ACTIVE' if self.note_active else 'idle'} · db={db:.1f} · voiced={self.voiced} · fused={fused:.2f} · pickEMA={pick_ema:.2f}"
        )

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.stop_audio()
        except Exception:
            pass
        event.accept()


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", type=str, default=None, help="Input device index or substring")
    ap.add_argument("--out-device", type=str, default=None, help="Output device index or substring (defaults to --device)")

    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--metadata", type=str, default=None)

    ap.add_argument("--hop-ms", type=float, default=4.0)
    ap.add_argument("--topk", type=int, default=3)

    # confidence hysteresis
    ap.add_argument("--nn-on", type=float, default=0.38, help="Enter note if fused >= this")
    ap.add_argument("--nn-off", type=float, default=0.25, help="Allow note to decay only if fused < this")

    ap.add_argument("--fft-cents", type=float, default=60.0)
    ap.add_argument("--fft-weight", type=float, default=0.40)
    ap.add_argument("--fft-min-hz", type=float, default=80.0)
    ap.add_argument("--fft-view-max-hz", type=float, default=2400.0)

    ap.add_argument("--pick-min-p", type=float, default=0.10)
    ap.add_argument("--pick-rise", type=float, default=0.35)
    ap.add_argument("--pick-reset", type=float, default=0.12)
    ap.add_argument("--sus-hold-ms", type=float, default=300.0)
    ap.add_argument("--sustain-block", type=float, default=0.75)
    ap.add_argument("--preemph", type=float, default=0.10)

    # voicing + grace hold
    ap.add_argument("--voiced-db-on", type=float, default=-45.0, help="DI considered voiced when dB >= this")
    ap.add_argument("--voiced-db-off", type=float, default=-55.0, help="DI considered unvoiced when dB <= this")
    ap.add_argument("--note-grace-ms", type=float, default=160.0, help="Hold last note this long while still voiced")
    ap.add_argument("--min-note-ms", type=float, default=60.0, help="Minimum note on-time after onset")
    ap.add_argument("--freq-smooth", type=float, default=0.85, help="0..0.99 smoothing for tracked FFT pitch")

    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--synth-wave", type=str, default="sine", choices=["sine", "square", "saw", "wav"])
    ap.add_argument("--synth-gain", type=float, default=0.14)
    ap.add_argument("--synth-attack-ms", type=float, default=2.0)
    ap.add_argument("--synth-release-ms", type=float, default=35.0)

    ap.add_argument("--wav-file", type=str, default=None, help="Path to WAV file to use as synth source")
    ap.add_argument("--wav-root-midi", type=float, default=69.0, help="MIDI note the WAV is tuned to")
    ap.add_argument("--wav-loop", action="store_true", help="Loop the WAV file continuously")

    ap.add_argument("--wasapi-exclusive", action="store_true")
    ap.add_argument("--asio-in-ch", type=int, default=None)
    ap.add_argument("--asio-out-ch", type=int, default=None)

    return ap.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        print(list_audio_devices())
        return

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()