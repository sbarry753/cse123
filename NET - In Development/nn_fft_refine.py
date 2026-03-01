#!/usr/bin/env python3
"""
nn_fft_refine_live_synth_ring_to_wav.py

Same idea as your script (parallel NN + always-on FFT on the SAME ring buffer),
but instead of playing audio live with sounddevice, it renders the synth output
and writes it to a WAV file incrementally.

Also includes a much less jittery “pick” trigger:
- Smooth picked_p (EMA)
- Trigger on a PEAK (slope goes + then -), with hysteresis + retrigger gate
- Optional sustain gating to prevent spam during sustained notes

Example:
python nn_fft_refine_live_synth_ring_to_wav.py \
  --checkpoint stage1_5ms_cos_sus_v2/onset_best.pt \
  --metadata labels/metadata.json \
  --wav ../Dataset/note/A5_5.wav \
  --out_wav out_synth.wav \
  --nn_window_ms 8 --fft_window_ms 8 --hop_ms 2 \
  --note_topk_refine 3 --search_cents 80 --n_fft 16384 \
  --pick_rise_thresh 0.35 --pick_min_p 0.05 \
  --onset_validate_ms 6 --onset_min_note_frames 2 --onset_max_cents_drift 10 \
  --retrigger_ms 80 --pick_fall_reset 0.10 \
  --note_gate_p 0.20 --energy_gate_db -100 --note_off_grace_ms 8 \
  --ref_hold_ms 120 --ref_fallback_search_cents 300
"""

import argparse
import json
from collections import deque

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


# ----------------------------
# Model
# ----------------------------
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

        self.width = width
        self.n_strings = n_strings

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

    model = OnsetNet(
        note_vocab_size=note_vocab_size,
        width=width,
        n_strings=n_strings,
        has_pick_sustain=has_pick_sustain,
    ).to(device)

    model.load_state_dict(sd_, strict=has_pick_sustain)
    model.eval()
    return model, note_vocab_size, width, n_strings, has_pick_sustain


# ----------------------------
# Ring buffer
# ----------------------------
class RingBuffer1D:
    def __init__(self, size: int):
        self.size = int(size)
        assert self.size > 0
        self.buf = np.zeros((self.size,), dtype=np.float32)
        self.w = 0
        self.filled = 0

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = int(x.size)
        if n <= 0:
            return
        if n >= self.size:
            x = x[-self.size:]
            n = self.size
        end = self.w + n
        if end <= self.size:
            self.buf[self.w:end] = x
        else:
            k = self.size - self.w
            self.buf[self.w:] = x[:k]
            self.buf[:end - self.size] = x[k:]
        self.w = (self.w + n) % self.size
        self.filled = min(self.size, self.filled + n)

    def _get_last(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        if n > self.size:
            n = self.size
        start = (self.w - n) % self.size
        if start < self.w:
            return self.buf[start:self.w].copy()
        return np.concatenate([self.buf[start:], self.buf[:self.w]]).copy()

    def get(self) -> np.ndarray:
        if self.filled < self.size:
            out = np.zeros((self.size,), dtype=np.float32)
            tmp = self._get_last(self.filled)
            out[-self.filled:] = tmp
            return out
        if self.w == 0:
            return self.buf.copy()
        return np.concatenate([self.buf[self.w:], self.buf[:self.w]]).astype(np.float32, copy=False)


# ----------------------------
# Audio utils
# ----------------------------
def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float32), int(sr)

def ms_to_samples(ms: float, sr: int) -> int:
    return int(round((ms / 1000.0) * float(sr)))

def rms_norm(x: np.ndarray, floor_rms: float = 1e-3) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms < floor_rms:
        return np.zeros_like(x, dtype=np.float32)
    return (x / rms).astype(np.float32, copy=False)

def apply_preemph(x: np.ndarray, coef: float) -> np.ndarray:
    if coef <= 0.0 or x.size < 2:
        return x
    y = x.copy()
    y[1:] = y[1:] - float(coef) * y[:-1]
    return y

def frame_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * np.log10(max(rms, 1e-12))

def smooth_freq(prev_f, new_f, alpha=0.85, max_cents_step=15.0):
    if prev_f is None or (not np.isfinite(prev_f)) or prev_f <= 1.0:
        return float(new_f)
    if (not np.isfinite(new_f)) or new_f <= 1.0:
        return float(prev_f)

    # clamp how far we can move per hop
    cents = 1200.0 * np.log2(new_f / prev_f)
    cents = float(np.clip(cents, -max_cents_step, +max_cents_step))
    target = prev_f * (2.0 ** (cents / 1200.0))

    # EMA in Hz after clamping
    return float(alpha * prev_f + (1.0 - alpha) * target)
# ----------------------------
# Pitch mapping + string lock
# ----------------------------
def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]
OPEN_STRING_HZ = [midi_to_hz(m) for m in OPEN_STRING_MIDI]

def string_freq_range(string_idx: int, max_fret: int = 24) -> tuple[float, float]:
    if not (0 <= int(string_idx) < 6):
        return (0.0, float("inf"))
    f0 = float(OPEN_STRING_HZ[int(string_idx)])
    fmax = f0 * (2.0 ** (max_fret / 12.0))
    return f0, fmax

def apply_soft_string_lock(lo: float, hi: float, s_lo: float, s_hi: float, lock_w: float) -> tuple[float, float]:
    lock_w = float(np.clip(lock_w, 0.0, 1.0))
    if not (np.isfinite(s_lo) and np.isfinite(s_hi)) or s_hi <= s_lo:
        return lo, hi
    lo0, hi0 = lo, hi
    if lo < s_lo:
        lo = lo + lock_w * (s_lo - lo)
    if hi > s_hi:
        hi = hi - lock_w * (hi - s_hi)
    if hi <= lo:
        return lo0, hi0
    return lo, hi


# ----------------------------
# FFT refine (top-K band)
# ----------------------------
def _parabolic_delta(mag: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(mag) - 1:
        return 0.0
    a, b, c = mag[i - 1], mag[i], mag[i + 1]
    denom = (a - 2.0 * b + c)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (a - c) / denom

def refine_pitch_fft_topk(
    x_1d: np.ndarray,
    sr: int,
    f_centers: list[float],
    center_weights: list[float],
    search_cents: float,
    n_fft: int,
    min_freq: float,
    max_freq: float,
    *,
    string_idx: int | None = None,
    string_conf: float = 0.0,
    string_soft_lock_max: float = 0.55,
    max_fret: int = 24,
    min_bins_in_band: int = 5,
    max_search_cents: float = 700.0,
    widen_factor: float = 1.6,
    use_harmonic_score: bool = True,
    max_harmonics: int = 5,
) -> tuple[float, float, str]:
    fc = [(float(f), float(w)) for f, w in zip(f_centers, center_weights) if np.isfinite(f) and f > 0 and w > 0]
    if not fc:
        return float("nan"), 0.0, "no_centers"

    ws = np.array([w for _, w in fc], dtype=np.float32)
    ws = ws / max(float(ws.sum()), 1e-12)
    fc = [(f, float(w)) for (f, _), w in zip(fc, ws.tolist())]

    x = x_1d.astype(np.float32)
    if x.size < 16:
        return float(fc[0][0]), 0.0, "too_short"

    wwin = np.hanning(len(x)).astype(np.float32)
    xw = x * wwin
    if len(xw) < n_fft:
        xw = np.pad(xw, (0, n_fft - len(xw)))
    else:
        xw = xw[:n_fft]

    X = np.fft.rfft(xw)
    mag = np.abs(X).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    lock_w = float(np.clip(string_conf, 0.0, 1.0)) * float(string_soft_lock_max)
    s_lo, s_hi = (0.0, float("inf"))
    if string_idx is not None and 0 <= int(string_idx) < 6:
        s_lo, s_hi = string_freq_range(int(string_idx), max_fret=max_fret)

    used_cents = float(search_cents)
    while True:
        ratio = 2.0 ** (used_cents / 1200.0)

        lo = float("inf")
        hi = 0.0
        for f, _w in fc:
            lo = min(lo, f / ratio)
            hi = max(hi, f * ratio)

        lo = max(float(min_freq), lo)
        hi = min(float(max_freq), hi)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return float(fc[0][0]), 0.0, "bad_band"

        lo, hi = apply_soft_string_lock(lo, hi, s_lo, s_hi, lock_w)

        band = np.where((freqs >= lo) & (freqs <= hi))[0]
        if band.size >= int(min_bins_in_band):
            break
        if used_cents >= float(max_search_cents):
            return float(fc[0][0]), 0.0, "band_too_small_fallback"
        used_cents = min(float(max_search_cents), used_cents * float(widen_factor))

    band_mag = mag[band]
    M = int(min(30, band.size))
    top_local = np.argpartition(-band_mag, M - 1)[:M]
    cand_bins = band[top_local]

    sigma_cents = max(15.0, used_cents * 0.35)

    def center_prior(f_hz: float) -> float:
        score = 0.0
        for f0, w0 in fc:
            cents = 1200.0 * np.log2(f_hz / f0)
            score += w0 * np.exp(-0.5 * (cents / sigma_cents) ** 2)
        return float(score)

    def harmonic_score(f0: float) -> float:
        score = 0.0
        for h in range(1, int(max_harmonics) + 1):
            fh = f0 * h
            if fh > max_freq:
                break
            kh = int(np.round(fh * n_fft / sr))
            if 0 <= kh < mag.size:
                score += float(mag[kh]) / float(h)
        return float(score)

    best_bin = int(cand_bins[0])
    best_score = -1.0
    for k in cand_bins:
        f0 = float(freqs[k])
        if f0 <= 0:
            continue
        prior = center_prior(f0)
        if prior <= 0:
            continue
        hs = harmonic_score(f0) if use_harmonic_score else float(mag[k])
        score = hs * prior
        if score > best_score:
            best_score = score
            best_bin = int(k)

    k = best_bin
    delta = _parabolic_delta(mag, k)
    k_ref = float(k) + float(delta)
    f_ref = k_ref * (sr / float(n_fft))
    if not np.isfinite(f_ref) or f_ref <= 0.0:
        return float(fc[0][0]), 0.0, "fallback_badf"

    f_ref = float(np.clip(f_ref, lo, hi))
    med = float(np.median(mag[band]) + 1e-12)
    conf = float(best_score / med) if med > 0 else 0.0
    return float(f_ref), float(conf), "topk"


# ----------------------------
# Note name
# ----------------------------
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def midi_to_note_name(m: int) -> str:
    o = (m // 12) - 1
    n = NOTE_NAMES[m % 12]
    return f"{n}{o}"


# ----------------------------
# Synth
# ----------------------------
class MonoSynth:
    def __init__(self, sr: int, waveform: str = "sine", attack_ms: float = 2.0, release_ms: float = 10.0, gain: float = 0.15):
        self.sr = int(sr)
        self.waveform = str(waveform)
        self.phase = 0.0
        self.freq = 440.0
        self.gain = float(gain)
        self.env = 0.0
        self.target_env = 0.0
        self.attack_samps = max(1, int(round(attack_ms * 1e-3 * self.sr)))
        self.release_samps = max(1, int(round(release_ms * 1e-3 * self.sr)))

    def set_freq(self, f_hz: float):
        if np.isfinite(f_hz) and f_hz > 1.0:
            self.freq = float(f_hz)

    def set_active(self, active: bool):
        self.target_env = 1.0 if active else 0.0

    def render(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        step = (1.0 / self.attack_samps) if self.target_env > self.env else (1.0 / self.release_samps)
        out = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            if self.env < self.target_env:
                self.env = min(self.target_env, self.env + step)
            elif self.env > self.target_env:
                self.env = max(self.target_env, self.env - step)

            ph = self.phase
            if self.waveform == "square":
                s = 1.0 if ph < np.pi else -1.0
            elif self.waveform == "saw":
                s = (ph / np.pi) - 1.0
            else:
                s = np.sin(ph)
            out[i] = float(s) * (self.gain * self.env)

            ph += (2.0 * np.pi) * (self.freq / self.sr)
            if ph >= 2.0 * np.pi:
                ph -= 2.0 * np.pi
            self.phase = ph
        return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out_wav", required=True, help="output synthesized WAV path")

    ap.add_argument("--nn_window_ms", type=float, default=8.0)
    ap.add_argument("--fft_window_ms", type=float, default=8.0)
    ap.add_argument("--share_buffer", action="store_true", help="use one shared ring for NN+FFT (default true)")
    ap.set_defaults(share_buffer=True)

    ap.add_argument("--hop_ms", type=float, default=2.0)
    ap.add_argument("--preemph_coef", type=float, default=0.10)

    ap.add_argument("--start_ms", type=float, default=0.0)
    ap.add_argument("--dur_ms", type=float, default=-1.0)

    ap.add_argument("--energy_gate_db", type=float, default=-45.0)
    ap.add_argument("--note_gate_p", type=float, default=0.35)
    ap.add_argument("--print_every", type=int, default=1)

    ap.add_argument("--no_fft", action="store_true")
    ap.add_argument("--search_cents", type=float, default=80.0)
    ap.add_argument("--n_fft", type=int, default=16384)
    ap.add_argument("--min_freq", type=float, default=70.0)
    ap.add_argument("--max_freq", type=float, default=2000.0)
    ap.add_argument("--note_topk_refine", type=int, default=3)
    ap.add_argument("--string_soft_lock_max", type=float, default=0.55)
    ap.add_argument("--max_fret", type=int, default=24)
    ap.add_argument("--min_bins_in_band", type=int, default=5)
    ap.add_argument("--max_search_cents", type=float, default=700.0)
    ap.add_argument("--widen_factor", type=float, default=1.6)
    ap.add_argument("--no_harmonic", action="store_true")
    ap.add_argument("--max_harmonics", type=int, default=5)

    # Synth
    ap.add_argument("--synth_wave", type=str, default="sine", choices=["sine", "square", "saw"])
    ap.add_argument("--synth_gain", type=float, default=0.15)
    ap.add_argument("--synth_attack_ms", type=float, default=2.0)
    ap.add_argument("--synth_release_ms", type=float, default=10.0)

    # Pick trigger (we keep your args but the trigger algorithm is improved)
    ap.add_argument("--pick_rise_window_ms", type=float, default=8.0)  # kept for compatibility; not used directly
    ap.add_argument("--pick_rise_thresh", type=float, default=0.45)
    ap.add_argument("--pick_min_p", type=float, default=0.30)
    ap.add_argument("--retrigger_ms", type=float, default=30.0)
    ap.add_argument("--pick_fall_reset", type=float, default=0.10)

    # NEW: smoothing + sustain gating for triggers
    ap.add_argument("--pick_ema_tau_ms", type=float, default=12.0, help="EMA smoothing time constant for picked/sustain")
    ap.add_argument("--sustain_block", type=float, default=0.75, help="block triggers when sustain_ema >= this (set >1 to disable)")
    ap.add_argument("--reset_hold_frames", type=int, default=2, help="how many frames below pick_fall_reset to re-arm")

    # Onset validation (jitter rejection)
    ap.add_argument("--onset_validate_ms", type=float, default=6.0)
    ap.add_argument("--onset_min_note_frames", type=int, default=2)
    ap.add_argument("--onset_max_cents_drift", type=float, default=40.0)

    # Hold / grace
    ap.add_argument("--note_off_grace_ms", type=float, default=4.0)

    # keep reference centers even when NN says none
    ap.add_argument("--ref_hold_ms", type=float, default=80.0, help="how long to keep last NN centers for FFT tracking during NN dropouts")
    ap.add_argument("--ref_fallback_search_cents", type=float, default=220.0, help="wider search when using held centers (dropouts)")
    ap.add_argument("--ref_min_topk_p", type=float, default=0.05, help="ignore centers whose prob < this when updating centers")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.metadata, "r") as f:
        meta = json.load(f)
    midi_vocab = meta.get("midi_vocab", None)
    index_to_pitch = meta.get("index_to_pitch", None)
    if not isinstance(midi_vocab, list) or len(midi_vocab) == 0:
        raise RuntimeError("metadata.json must contain a non-empty 'midi_vocab' list.")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model, note_vocab_size, width, n_strings, has_pick_sustain = build_model_from_checkpoint(ckpt, device=device)
    if not has_pick_sustain:
        raise RuntimeError("This script expects picked_head/sustain_head in the checkpoint.")

    y, sr = load_wav_mono(args.wav)

    nn_W = ms_to_samples(args.nn_window_ms, sr)
    fft_W = ms_to_samples(args.fft_window_ms, sr)
    hop = max(1, ms_to_samples(args.hop_ms, sr))

    # enforce shared buffer sizes if requested
    if args.share_buffer:
        if nn_W != fft_W:
            fft_W = nn_W

    start_s = ms_to_samples(args.start_ms, sr)
    end_s = len(y) if float(args.dur_ms) <= 0 else min(len(y), start_s + ms_to_samples(args.dur_ms, sr))

    if args.share_buffer:
        ring = RingBuffer1D(nn_W)
        nn_ring = ring
        fft_ring = ring
    else:
        nn_ring = RingBuffer1D(nn_W)
        fft_ring = RingBuffer1D(fft_W)

    # grace and retrigger
    grace_hops = int(np.ceil(ms_to_samples(args.note_off_grace_ms, sr) / hop)) if args.note_off_grace_ms > 0 else 0
    no_note_ctr = 0
    retrigger_hops = int(np.ceil(ms_to_samples(args.retrigger_ms, sr) / hop)) if args.retrigger_ms > 0 else 0
    retrigger_ctr = 0

    # onset validate
    validate_hops = max(1, int(np.ceil(ms_to_samples(args.onset_validate_ms, sr) / hop)))
    validating = False
    val_ctr = 0
    val_freqs: list[float] = []
    val_have_note = 0

    # synth
    synth = MonoSynth(sr=sr, waveform=args.synth_wave, attack_ms=args.synth_attack_ms, release_ms=args.synth_release_ms, gain=args.synth_gain)
    note_active = False
    active_freq: float | None = None

    # held NN centers for FFT tracking
    ref_hold_hops = max(1, int(np.ceil(ms_to_samples(args.ref_hold_ms, sr) / hop)))
    ref_age = ref_hold_hops + 1
    last_centers: list[float] = []
    last_weights: list[float] = []

    def update_centers_from_probs(note_probs: torch.Tensor, topk: int) -> tuple[list[float], list[float]]:
        k = max(1, int(topk))
        idxs = torch.topk(note_probs, k=min(k, note_probs.numel())).indices.tolist()
        fcs, ws = [], []
        for ci in idxs:
            if ci >= len(midi_vocab):
                continue
            p = float(note_probs[ci].item())
            if p < float(args.ref_min_topk_p):
                continue
            midi_c = float(midi_vocab[ci])
            fcs.append(midi_to_hz(midi_c))
            ws.append(p)
        return fcs, ws

    # NEW: EMA smoothing + peak trigger state
    tau = max(1e-6, float(args.pick_ema_tau_ms))
    ema_a = float(np.exp(-float(args.hop_ms) / tau))
    picked_ema = 0.0
    picked_ema_prev = 0.0
    sustain_ema = 0.0

    armed = True
    rising = False
    peak_val = 0.0
    min_since_reset = 1.0
    reset_hold = 0
    reset_hold_hops = max(1, int(args.reset_hold_frames))

    print("\n=== SYNTH TO WAV: PARALLEL NN + ALWAYS-ON FFT (shared ring) ===")
    print(f"SR={sr} ringW={nn_W} ({args.nn_window_ms:.2f}ms) hop={hop} ({args.hop_ms:.2f}ms) share={args.share_buffer}")
    print("Format: t  db  nn_note(p) pickEMA dPick  TRIG  STATE  f_track  ref_age")
    print(f"Writing: {args.out_wav}\n")

    ptr = int(start_s)
    samples_consumed = 0
    frame_idx = 0

    # Write incrementally to avoid huge RAM usage
    with sf.SoundFile(args.out_wav, mode="w", samplerate=sr, channels=1, subtype="PCM_16") as out_f:
        while ptr < end_s:
            block = y[ptr: ptr + hop]
            if block.size < hop:
                block = np.pad(block, (0, hop - block.size)).astype(np.float32)

            nn_ring.push(block)
            if not args.share_buffer:
                fft_ring.push(block)

            ptr += hop
            samples_consumed += hop

            x_nn = nn_ring.get()
            x_fft = fft_ring.get()

            db = frame_dbfs(x_nn)
            is_silence = (db < float(args.energy_gate_db))

            x_nn_p = apply_preemph(rms_norm(x_nn), args.preemph_coef)
            x_fft_p = apply_preemph(rms_norm(x_fft), args.preemph_coef)

            # NN forward
            x_t = torch.from_numpy(x_nn_p).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                note_logits, string_logits, picked_logit, sustain_logit = model(x_t)

            picked_p = float(torch.sigmoid(picked_logit[0]).detach().cpu().item())
            sustain_p = float(torch.sigmoid(sustain_logit[0]).detach().cpu().item())

            note_logits = note_logits[0].detach().cpu()
            string_logits = string_logits[0].detach().cpu()
            note_probs = torch.softmax(note_logits, dim=0)
            string_probs = torch.softmax(string_logits, dim=0)

            pred_idx = int(torch.argmax(note_probs).item())
            pred_p = float(note_probs[pred_idx].item())
            pred_string = int(torch.argmax(string_probs).item())
            pred_string_p = float(string_probs[pred_string].item())

            is_neg_class = (pred_idx >= len(midi_vocab))
            have_note = (not is_silence) and (not is_neg_class) and (pred_p >= float(args.note_gate_p))

            name = "(none)"
            midi = None
            if have_note:
                midi = int(round(float(midi_vocab[pred_idx])))
                name = str(index_to_pitch[pred_idx]) if isinstance(index_to_pitch, list) and pred_idx < len(index_to_pitch) else midi_to_note_name(midi)

            # update reference centers
            if have_note:
                fcs, ws = update_centers_from_probs(note_probs, args.note_topk_refine)
                if fcs:
                    last_centers, last_weights = fcs, ws
                    ref_age = 0
            else:
                ref_age += 1

            # ALWAYS-ON FFT tracking when we have refs and they're not too old
            f_track = float("nan")
            if (not args.no_fft) and last_centers and (ref_age <= ref_hold_hops):
                use_cents = float(args.search_cents) if have_note else float(args.ref_fallback_search_cents)
                f_track, fft_conf, _mode = refine_pitch_fft_topk(
                    x_1d=x_fft_p,
                    sr=sr,
                    f_centers=last_centers,
                    center_weights=last_weights,
                    search_cents=use_cents,
                    n_fft=int(args.n_fft),
                    min_freq=float(args.min_freq),
                    max_freq=float(args.max_freq),
                    string_idx=pred_string,
                    string_conf=pred_string_p,
                    string_soft_lock_max=float(args.string_soft_lock_max),
                    max_fret=int(args.max_fret),
                    min_bins_in_band=int(args.min_bins_in_band),
                    max_search_cents=float(args.max_search_cents),
                    widen_factor=float(args.widen_factor),
                    use_harmonic_score=(not args.no_harmonic),
                    max_harmonics=int(args.max_harmonics),
                )

            # NEW: EMA smoothing on pick/sustain
            picked_ema_prev = picked_ema
            picked_ema = ema_a * picked_ema + (1.0 - ema_a) * picked_p
            sustain_ema = ema_a * sustain_ema + (1.0 - ema_a) * sustain_p
            dp = picked_ema - picked_ema_prev

            # retrigger counter
            if retrigger_ctr > 0:
                retrigger_ctr -= 1

            # Hysteresis re-arm: must fall below reset for N frames
            if not armed:
                if picked_ema < float(args.pick_fall_reset):
                    reset_hold += 1
                    if reset_hold >= reset_hold_hops:
                        armed = True
                        rising = False
                        peak_val = 0.0
                        min_since_reset = picked_ema
                        reset_hold = 0
                else:
                    reset_hold = 0

            # PEAK trigger
            trigger = False
            if armed:
                min_since_reset = min(min_since_reset, picked_ema)

                if dp > 0:
                    rising = True
                    peak_val = max(peak_val, picked_ema)
                elif rising and dp <= 0:
                    # slope flipped -> peak
                    peak = peak_val
                    rise_amt = peak - min_since_reset

                    sustain_blocked = (sustain_ema >= float(args.sustain_block)) if float(args.sustain_block) <= 1.0 else False

                    trigger = (
                        (retrigger_ctr == 0)
                        and (not sustain_blocked)
                        and (peak >= float(args.pick_min_p))
                        and (rise_amt >= float(args.pick_rise_thresh))
                    )

                    if trigger:
                        validating = True
                        val_ctr = 0
                        val_freqs = []
                        val_have_note = 0
                        retrigger_ctr = retrigger_hops

                        armed = False  # disarm until we fall low again
                        rising = False
                        peak_val = 0.0
                        min_since_reset = 1.0
                        reset_hold = 0
                    else:
                        rising = False
                        peak_val = 0.0

            # onset validation window
            val_status = ""
            if validating:
                val_ctr += 1
                if have_note:
                    val_have_note += 1
                if np.isfinite(f_track) and f_track > 1.0:
                    val_freqs.append(float(f_track))

                if val_ctr >= validate_hops:
                    ok_note = (val_have_note >= int(args.onset_min_note_frames))
                    ok_pitch = False
                    chosen = None
                    if val_freqs:
                        f_min = float(np.min(val_freqs))
                        f_max = float(np.max(val_freqs))
                        drift_c = abs(1200.0 * np.log2(f_max / f_min)) if (f_min > 0 and f_max > 0) else 1e9
                        ok_pitch = (drift_c <= float(args.onset_max_cents_drift))
                        chosen = float(np.median(val_freqs))

                    if ok_note and ok_pitch and chosen is not None and np.isfinite(chosen) and chosen > 1.0:
                        note_active = True
                        active_freq = chosen
                        synth.set_freq(active_freq)
                        synth.set_active(True)
                        no_note_ctr = 0
                        val_status = "VAL_OK"
                    else:
                        val_status = "VAL_NO"
                        if not note_active:
                            synth.set_active(False)
                            active_freq = None
                    validating = False

            # sustain / off logic
            if note_active:
                if have_note or (ref_age <= ref_hold_hops and np.isfinite(f_track)):
                    no_note_ctr = 0
                    if (not validating) and np.isfinite(f_track) and f_track > 1.0:
                        active_freq = float(f_track)
                        synth.set_freq(active_freq)
                else:
                    no_note_ctr += 1
                    if no_note_ctr > grace_hops:
                        note_active = False
                        active_freq = None
                        synth.set_active(False)

            # render + write
            audio_out = synth.render(hop)
            out_f.write(audio_out.reshape(-1, 1))

            if (frame_idx % max(1, int(args.print_every))) == 0:
                t_ms = (samples_consumed / sr) * 1000.0 + float(args.start_ms)
                trig_mark = "ON" if trigger else "  "
                state = "PLAY" if note_active else "----"
                fshow = active_freq if active_freq is not None else float("nan")
                if validating:
                    val_status = f"VAL{val_ctr}/{validate_hops}"
                print(
                    f"{t_ms:8.2f}ms  db={db:6.1f}  nn={name:6s} p={pred_p:.2f}  "
                    f"pickE={picked_ema:.2f} dP={dp:+.3f}  {trig_mark} {state}  f={fshow:8.2f}  ref_age={ref_age:3d} {val_status}"
                )

            frame_idx += 1

    print("\nDone. Wrote:", args.out_wav)


if __name__ == "__main__":
    main()