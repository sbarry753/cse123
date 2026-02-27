#!/usr/bin/env python3
"""
nn_fft_refine_live_ring_synth.py

Ring-buffer "live" sim + simple synth output.

Trigger rule:
- Compute rolling average of picked_p over last pick_avg_ms (default 5ms) in hop frames
- If avg_pick >= pick_trigger AND have_note == True -> NOTE ON (if not already active)

Hold / release rule (gap-tolerant):
- While NOTE is active, do NOT release immediately on brief dropouts.
- If have_note becomes false:
    - If --fft_keepalive and FFT band energy is still present, keep holding.
    - Otherwise increment dropout counter.
- NOTE OFF only if dropout counter reaches hang_frames = ceil(note_off_hang_ms / hop_ms)

Synth:
- Sine oscillator with simple attack/release envelope
- Frequency: uses f_ref from FFT refine if enabled and valid, else MIDI->Hz
- While held: retunes only when have_note is true; otherwise holds last freq.

Playback / output:
- --play uses sounddevice to play in realtime
- --synth_out_wav writes synthesized audio to a WAV file

Example:
python nn_fft_refine_live_ring_synth.py --checkpoint stage1_5ms_cos/onset_best.pt --metadata labels/metadata.json \
  --wav ../Dataset/note/B5_5.wav --nn_window_ms 8 --fft_window_ms 8 --hop_ms 2 \
  --note_topk_refine 3 --search_cents 80 --n_fft 16384 \
  --note_off_hang_ms 12 --fft_keepalive --fft_keepalive_db -55 \
  --play --synth_out_wav synth.wav
"""

import argparse
import json
import math
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
    sd = ckpt["model_state"]
    note_vocab_size = int(sd["note_head.weight"].shape[0])
    width = int(sd["note_head.weight"].shape[1])
    n_strings = int(sd["string_head.weight"].shape[0])

    has_pick = ("picked_head.weight" in sd) and ("picked_head.bias" in sd)
    has_sus = ("sustain_head.weight" in sd) and ("sustain_head.bias" in sd)
    has_pick_sustain = bool(has_pick and has_sus)

    model = OnsetNet(
        note_vocab_size=note_vocab_size,
        width=width,
        n_strings=n_strings,
        has_pick_sustain=has_pick_sustain,
    ).to(device)

    # strict only if heads exist in checkpoint
    model.load_state_dict(sd, strict=has_pick_sustain)
    model.eval()
    return model, note_vocab_size, width, n_strings, has_pick_sustain


# ----------------------------
# Ring buffer
# ----------------------------
class RingBuffer1D:
    """
    Fixed-size circular buffer for float32 audio.
    - push(block): write sequential samples
    - get(): returns a contiguous array (oldest->newest) of length size
    """
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

    def get(self) -> np.ndarray:
        if self.filled < self.size:
            out = np.zeros((self.size,), dtype=np.float32)
            tmp = self._get_last(self.filled)
            out[-self.filled:] = tmp
            return out

        if self.w == 0:
            return self.buf.copy()
        return np.concatenate([self.buf[self.w:], self.buf[:self.w]]).astype(np.float32, copy=False)

    def _get_last(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        if n > self.size:
            n = self.size
        start = (self.w - n) % self.size
        if start < self.w:
            return self.buf[start:self.w].copy()
        else:
            return np.concatenate([self.buf[start:], self.buf[:self.w]]).copy()


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


def rms_norm(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return x / max(rms, 1e-4)


def apply_preemph(x: np.ndarray, coef: float) -> np.ndarray:
    if coef <= 0.0 or x.size < 2:
        return x
    y = x.copy()
    y[1:] = y[1:] - float(coef) * y[:-1]
    return y


def frame_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return 20.0 * np.log10(max(rms, 1e-12))


def band_rms_dbfs_fft(x: np.ndarray, sr: int, n_fft: int, lo: float, hi: float) -> float:
    """
    Computes a rough "signal present" metric from FFT magnitude:
    - windowed FFT
    - take magnitudes in [lo, hi]
    - return 20*log10(RMS(mag_band))
    This is not calibrated to true dBFS; it's just a consistent thresholdable metric.
    """
    x = x.astype(np.float32)
    if x.size < 16:
        return -120.0

    wwin = np.hanning(len(x)).astype(np.float32)
    xw = x * wwin
    if len(xw) < n_fft:
        xw = np.pad(xw, (0, n_fft - len(xw)))
    else:
        xw = xw[:n_fft]

    X = np.fft.rfft(xw)
    mag = np.abs(X).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    band = (freqs >= float(lo)) & (freqs <= float(hi))
    if not np.any(band):
        return -120.0

    rms = float(np.sqrt(np.mean(mag[band] * mag[band]) + 1e-12))
    return 20.0 * np.log10(max(rms, 1e-12))


# ----------------------------
# Pitch mapping
# ----------------------------
def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(m: int) -> str:
    o = (m // 12) - 1
    n = NOTE_NAMES[m % 12]
    return f"{n}{o}"


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
    mode = "topk_harmonic" if use_harmonic_score else "topk_peak"
    return float(f_ref), float(conf), mode


# ----------------------------
# Simple synth (sine + attack/release env)
# ----------------------------
class SimpleSineSynth:
    def __init__(self, sr: int, volume: float = 0.15, attack_ms: float = 5.0, release_ms: float = 30.0):
        self.sr = int(sr)
        self.volume = float(volume)

        self.attack_samps = max(1, int(round((attack_ms / 1000.0) * self.sr)))
        self.release_samps = max(1, int(round((release_ms / 1000.0) * self.sr)))

        self.active = False
        self.releasing = False

        self.freq = 440.0
        self.phase = 0.0

        self.env = 0.0
        self.env_step_up = 1.0 / self.attack_samps
        self.env_step_down = 1.0 / self.release_samps

    def note_on(self, freq_hz: float):
        self.freq = float(max(1.0, freq_hz))
        self.active = True
        self.releasing = False
        self.env_step_up = 1.0 / self.attack_samps
        self.env_step_down = 1.0 / self.release_samps

    def note_off(self):
        if self.active:
            self.releasing = True

    def set_freq(self, freq_hz: float):
        self.freq = float(max(1.0, freq_hz))

    def render(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)

        if not self.active and self.env <= 0.0:
            return np.zeros((n,), dtype=np.float32)

        out = np.zeros((n,), dtype=np.float32)
        w = 2.0 * math.pi * (self.freq / self.sr)

        for i in range(n):
            if self.active and not self.releasing:
                self.env = min(1.0, self.env + self.env_step_up)
            else:
                self.env = max(0.0, self.env - self.env_step_down)
                if self.env <= 0.0:
                    self.active = False
                    self.releasing = False

            out[i] = math.sin(self.phase) * (self.env * self.volume)

            self.phase += w
            if self.phase > 2.0 * math.pi:
                self.phase -= 2.0 * math.pi

        return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--metadata", required=True)

    ap.add_argument("--nn_window_ms", type=float, default=8.0)
    ap.add_argument("--fft_window_ms", type=float, default=8.0)
    ap.add_argument("--hop_ms", type=float, default=2.0)
    ap.add_argument("--preemph_coef", type=float, default=0.10)

    ap.add_argument("--start_ms", type=float, default=0.0)
    ap.add_argument("--dur_ms", type=float, default=-1.0, help="<=0 means full file")

    ap.add_argument("--energy_gate_db", type=float, default=-45.0)
    ap.add_argument("--note_gate_p", type=float, default=0.35)

    ap.add_argument("--no_fft", action="store_true")
    ap.add_argument("--search_cents", type=float, default=80.0)
    ap.add_argument("--n_fft", type=int, default=16384)
    ap.add_argument("--min_freq", type=float, default=70.0)
    ap.add_argument("--max_freq", type=float, default=2000.0)
    ap.add_argument("--note_topk_refine", type=int, default=3)
    ap.add_argument("--min_bins_in_band", type=int, default=5)
    ap.add_argument("--max_search_cents", type=float, default=700.0)
    ap.add_argument("--widen_factor", type=float, default=1.6)
    ap.add_argument("--no_harmonic", action="store_true")
    ap.add_argument("--max_harmonics", type=int, default=5)

    # --- synth controls ---
    ap.add_argument("--pick_trigger", type=float, default=0.80, help="avg picked_p over pick_avg_ms must exceed this")
    ap.add_argument("--pick_avg_ms", type=float, default=5.0, help="rolling average window for picked_p")
    ap.add_argument("--volume", type=float, default=0.15)
    ap.add_argument("--attack_ms", type=float, default=5.0)
    ap.add_argument("--release_ms", type=float, default=30.0)

    # --- dropout / keepalive controls ---
    ap.add_argument("--note_off_hang_ms", type=float, default=12.0,
                    help="allow this much no-note time before releasing (gap tolerance)")
    ap.add_argument("--fft_keepalive", action="store_true",
                    help="use FFT band energy to keep holding during short NN dropouts")
    ap.add_argument("--fft_keepalive_db", type=float, default=-55.0,
                    help="band RMS dB threshold for keepalive (higher = stricter)")
    ap.add_argument("--fft_keepalive_lo", type=float, default=70.0)
    ap.add_argument("--fft_keepalive_hi", type=float, default=2000.0)

    # --- playback/output ---
    ap.add_argument("--play", action="store_true", help="play synth audio live (requires sounddevice)")
    ap.add_argument("--synth_out_wav", type=str, default="", help="optional path to write synth output wav")

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
        raise RuntimeError("This synth trigger needs picked_head/sustain_head in the checkpoint (has_pick_sustain=False).")

    y, sr = load_wav_mono(args.wav)

    nn_W = ms_to_samples(args.nn_window_ms, sr)
    fft_W = ms_to_samples(args.fft_window_ms, sr)
    hop = max(1, ms_to_samples(args.hop_ms, sr))

    start_s = ms_to_samples(args.start_ms, sr)
    if float(args.dur_ms) > 0:
        end_s = min(len(y), start_s + ms_to_samples(args.dur_ms, sr))
    else:
        end_s = len(y)

    nn_ring = RingBuffer1D(nn_W)
    fft_ring = RingBuffer1D(fft_W)

    # pick average window in frames
    pick_frames = max(1, int(math.ceil(args.pick_avg_ms / max(1e-9, args.hop_ms))))
    pick_hist = deque(maxlen=pick_frames)

    # note-off hang frames
    hang_frames = max(1, int(math.ceil(args.note_off_hang_ms / max(1e-9, args.hop_ms))))
    no_note_run = 0

    synth = SimpleSineSynth(sr=sr, volume=args.volume, attack_ms=args.attack_ms, release_ms=args.release_ms)
    synth_chunks: list[np.ndarray] = []

    # optional live playback
    stream = None
    if args.play:
        try:
            import sounddevice as sd
            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", blocksize=hop)
            stream.start()
        except Exception as e:
            print(f"[WARN] --play requested but sounddevice failed: {e}")
            stream = None

    ptr = int(start_s)
    samples_consumed = 0

    note_active = False
    active_midi = None

    print("\n=== LIVE PEDAL TRACE + SYNTH (GAP-TOLERANT) ===")
    print(f"WAV: {args.wav}")
    print(f"SR={sr} nnW={nn_W} ({args.nn_window_ms:.2f}ms) fftW={fft_W} ({args.fft_window_ms:.2f}ms) hop={hop} ({args.hop_ms:.2f}ms)")
    print(f"Model: note_vocab={note_vocab_size}, width={width}, n_strings={n_strings}, pick/sus={has_pick_sustain}")
    print(f"Pick trigger: avg over {args.pick_avg_ms:.1f}ms (~{pick_frames} frames) >= {args.pick_trigger:.2f}")
    print(f"Note-off hang: {args.note_off_hang_ms:.1f}ms (~{hang_frames} frames)")
    if args.fft_keepalive and (not args.no_fft):
        print(f"FFT keepalive: ON  band=[{args.fft_keepalive_lo:.1f},{args.fft_keepalive_hi:.1f}]  thr={args.fft_keepalive_db:.1f} dB")
    else:
        print("FFT keepalive: OFF")
    print("")

    while ptr < end_s:
        block = y[ptr: ptr + hop]
        if block.size < hop:
            block = np.pad(block, (0, hop - block.size)).astype(np.float32)

        nn_ring.push(block)
        fft_ring.push(block)

        ptr += hop
        samples_consumed += hop

        x_nn = nn_ring.get()
        x_fft = fft_ring.get()

        db = frame_dbfs(x_nn)
        is_silence = (db < float(args.energy_gate_db))

        x_nn_p = apply_preemph(rms_norm(x_nn), args.preemph_coef)
        x_fft_p = apply_preemph(rms_norm(x_fft), args.preemph_coef)

        x_t = torch.from_numpy(x_nn_p).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            note_logits, string_logits, picked_logit, sustain_logit = model(x_t)

        picked_p = float(torch.sigmoid(picked_logit[0]).detach().cpu().item())
        sustain_p = float(torch.sigmoid(sustain_logit[0]).detach().cpu().item())

        note_logits = note_logits[0].detach().cpu()
        note_probs = torch.softmax(note_logits, dim=0)

        pred_idx = int(torch.argmax(note_probs).item())
        pred_p = float(note_probs[pred_idx].item())

        is_neg_class = (pred_idx >= len(midi_vocab))
        have_note = (not is_silence) and (not is_neg_class) and (pred_p >= float(args.note_gate_p))

        midi = None
        name = "(none)"
        if have_note:
            midi = int(round(float(midi_vocab[pred_idx])))
            if isinstance(index_to_pitch, list) and pred_idx < len(index_to_pitch):
                name = str(index_to_pitch[pred_idx])
            else:
                name = midi_to_note_name(midi)

        # optional FFT refine frequency
        f_ref = float("nan")
        if have_note and (not args.no_fft):
            K = max(1, int(args.note_topk_refine))
            top_idxs = torch.topk(note_probs, k=min(K, note_probs.numel())).indices.tolist()

            f_centers = []
            center_weights = []
            for ci in top_idxs:
                if ci >= len(midi_vocab):
                    continue
                midi_c = float(midi_vocab[ci])
                f_centers.append(midi_to_hz(midi_c))
                center_weights.append(float(note_probs[ci].item()))

            f_ref, _fft_conf, _mode = refine_pitch_fft_topk(
                x_1d=x_fft_p,
                sr=sr,
                f_centers=f_centers,
                center_weights=center_weights,
                search_cents=float(args.search_cents),
                n_fft=int(args.n_fft),
                min_freq=float(args.min_freq),
                max_freq=float(args.max_freq),
                min_bins_in_band=int(args.min_bins_in_band),
                max_search_cents=float(args.max_search_cents),
                widen_factor=float(args.widen_factor),
                use_harmonic_score=(not args.no_harmonic),
                max_harmonics=int(args.max_harmonics),
            )

        # pick trigger logic (rolling avg over pick_avg_ms)
        pick_hist.append(picked_p)
        avg_pick = float(np.mean(pick_hist)) if len(pick_hist) else 0.0
        pick_triggered = (avg_pick >= float(args.pick_trigger))

        # keepalive detection (only relevant when have_note is false)
        fft_alive = False
        fft_alive_db = -120.0
        if args.fft_keepalive and (not args.no_fft):
            fft_alive_db = band_rms_dbfs_fft(
                x_fft_p, sr=sr, n_fft=int(args.n_fft),
                lo=float(args.fft_keepalive_lo),
                hi=float(args.fft_keepalive_hi),
            )
            fft_alive = (fft_alive_db >= float(args.fft_keepalive_db))

        # choose synth frequency when we decide to play
        def choose_freq() -> float:
            if np.isfinite(f_ref) and f_ref > 0:
                return float(f_ref)
            if midi is not None:
                return float(midi_to_hz(midi))
            return 440.0

        # NOTE ON
        if (not note_active) and pick_triggered and have_note:
            freq = choose_freq()
            synth.note_on(freq)
            note_active = True
            active_midi = midi
            no_note_run = 0
            t_ms = (samples_consumed / sr) * 1000.0 + float(args.start_ms)
            print(f"[NOTE ON ] t={t_ms:8.2f}ms  {name:8s} midi={midi:3d}  freq={freq:9.2f}Hz  avg_pick={avg_pick:.2f}")

        # While holding: retune only when NN says we have a note (otherwise hold last freq)
        if note_active and have_note:
            synth.set_freq(choose_freq())
            active_midi = midi

        # Gap-tolerant NOTE OFF:
        # - If have_note: reset dropout counter
        # - Else if fft_alive: treat as "still sounding" and reset counter
        # - Else increment dropout counter and only release when it exceeds hang_frames
        if note_active:
            if have_note:
                no_note_run = 0
            else:
                if fft_alive:
                    no_note_run = 0
                else:
                    no_note_run += 1

            if no_note_run >= hang_frames:
                synth.note_off()
                note_active = False
                no_note_run = 0
                t_ms = (samples_consumed / sr) * 1000.0 + float(args.start_ms)
                if args.fft_keepalive and (not args.no_fft):
                    print(f"[NOTE OFF] t={t_ms:8.2f}ms  fft_alive_db={fft_alive_db:6.1f}  hang_frames={hang_frames}")
                else:
                    print(f"[NOTE OFF] t={t_ms:8.2f}ms  hang_frames={hang_frames}")

        # render one hop of synth audio
        synth_block = synth.render(hop)
        synth_chunks.append(synth_block.copy())

        if stream is not None:
            stream.write(synth_block.reshape(-1, 1))

    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    if args.synth_out_wav.strip():
        out = np.concatenate(synth_chunks, axis=0) if synth_chunks else np.zeros((0,), dtype=np.float32)
        sf.write(args.synth_out_wav, out, sr)
        print(f"\nWrote synth wav: {args.synth_out_wav}")

    print("\nDone.")


if __name__ == "__main__":
    main()
