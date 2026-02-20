#!/usr/bin/env python3
# lut_match.py
"""
Note matcher for LUTs produced by lut_build.py
(type: harmonic_fingerprint_lut_v3_multiframe_onset_stringaware)

Key improvements vs older matcher:
- Pulls analysis defaults from LUT meta (band template params, window, logmag, etc.)
- Optional onset-aligned analysis segment to match builder behavior:
    detect onset -> start = onset + post_onset -> dur seconds
- Optional per-string discriminative band weighting using LUT's string_band_importance
  (raw or residual weights if ideals exist)
- Optional std weighting and harmonic off-band penalties

Dependencies:
  pip install numpy scipy sounddevice

Examples:
  # list devices
  python lut_match_live_v2.py --lut lut.json --list_devices

  # live match (uses LUT defaults unless overridden)
  python lut_match_live_v2.py --lut lut.json --device 3 --sr 48000 --use_onset

  # test on wav
  python lut_match_live_v2.py --lut lut.json --wav test.wav --use_onset
"""

import argparse
import json
import queue
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, get_window

try:
    import sounddevice as sd
except Exception:
    sd = None


# -----------------------------
# Basic utilities
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def normalize_audio(x: np.ndarray, mode: str = "p95", target_rms: float = 0.1) -> np.ndarray:
    mode = (mode or "p95").lower()
    if mode == "none":
        return x
    if mode == "peak":
        d = float(np.max(np.abs(x)) + 1e-12)
        return x / d
    if mode in ("p95", "pct", "percentile"):
        d = float(np.percentile(np.abs(x), 95) + 1e-12)
        return x / d
    if mode == "rms":
        rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
        return x * (float(target_rms) / rms)
    raise ValueError("normalize must be one of: peak|p95|rms|none")


def highpass_filter(x: np.ndarray, fs: int, cutoff_hz: float = 80.0, order: int = 4) -> np.ndarray:
    if cutoff_hz <= 0:
        return x
    sos = butter(order, cutoff_hz / (fs * 0.5), btype="highpass", output="sos")
    return sosfilt(sos, x)


def make_log_band_edges(fmin_hz: float, fmax_hz: float, n_bands: int) -> np.ndarray:
    fmin_hz = max(1e-3, float(fmin_hz))
    fmax_hz = max(fmin_hz * 1.001, float(fmax_hz))
    n_bands = max(8, int(n_bands))
    return np.geomspace(fmin_hz, fmax_hz, n_bands + 1).astype(np.float64)


def spectrum_mag(seg: np.ndarray, fs: int, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    N = len(seg)
    w = get_window(window, N, fftbins=True).astype(np.float64)
    X = np.fft.rfft(seg * w)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    mag = np.abs(X).astype(np.float64)
    return freqs, mag


def pool_to_log_bands(freqs: np.ndarray, mag: np.ndarray, edges: np.ndarray, agg: str = "max") -> np.ndarray:
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    agg = (agg or "max").lower()
    for i in range(len(out)):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            out[i] = 0.0
        else:
            band = mag[mask]
            out[i] = float(np.max(band)) if agg == "max" else float(np.mean(band))
    return out


def frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if frame <= 0 or hop <= 0:
        raise ValueError("frame and hop must be > 0")
    n = len(x)
    if n <= frame:
        return np.zeros((1, frame), dtype=np.float64)
    count = 1 + (n - frame) // hop
    out = np.zeros((count, frame), dtype=np.float64)
    for i in range(count):
        s = i * hop
        out[i] = x[s:s + frame]
    return out


def aggregate_feature(frames_feat: np.ndarray, agg: str = "median") -> np.ndarray:
    agg = (agg or "median").lower()
    if frames_feat.ndim != 2:
        raise ValueError("frames_feat must be (T, D)")
    if agg == "mean":
        return np.mean(frames_feat, axis=0)
    return np.median(frames_feat, axis=0)


# -----------------------------
# Onset detection (same style as builder)
# -----------------------------
def detect_onset_sec_energy(
    x: np.ndarray,
    fs: int,
    frame: int,
    hop: int,
    thresh_ratio: float = 6.0,
    hold_frames: int = 3,
    search_start_sec: float = 0.0,
    search_end_sec: Optional[float] = None,
) -> float:
    """
    Finds first frame where RMS exceeds (noise_floor * thresh_ratio) for hold_frames.
    noise_floor estimated from first ~0.25s of search region.
    """
    n = len(x)
    if search_end_sec is None:
        search_end_sec = n / fs

    start_i = int(max(0, round(search_start_sec * fs)))
    end_i = int(min(n, round(search_end_sec * fs)))
    if end_i <= start_i + frame:
        return float(search_start_sec)

    xs = x[start_i:end_i]
    frames = frame_signal(xs, frame=frame, hop=hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)

    nf_frames = max(10, int(round(0.25 * fs / hop)))
    nf_frames = min(nf_frames, len(rms))
    noise_floor = float(np.median(rms[:nf_frames]) + 1e-12)
    thresh = noise_floor * float(thresh_ratio)

    hold_frames = max(1, int(hold_frames))
    for i in range(0, len(rms) - hold_frames + 1):
        if np.all(rms[i:i + hold_frames] >= thresh):
            onset_sample = start_i + i * hop
            return float(onset_sample / fs)

    return float(search_start_sec)


def pick_segment(x: np.ndarray, fs: int, start_sec: float, dur_sec: float) -> np.ndarray:
    start = int(round(start_sec * fs))
    length = int(round(dur_sec * fs))
    seg = x[start:start + length]
    if len(seg) < length:
        seg = np.pad(seg, (0, length - len(seg)))
    return seg


# -----------------------------
# Feature extraction (band template only, builder-compatible)
# -----------------------------
def compute_band_feature_multiframe(
    seg: np.ndarray,
    fs: int,
    *,
    edges: np.ndarray,
    window: str,
    band_pool_agg: str,
    band_logmag: bool,
    feat_agg: str,
    frame_size: int,
    hop_size: int,
) -> np.ndarray:
    if len(seg) < frame_size:
        seg = np.pad(seg, (0, frame_size - len(seg)))

    frames = frame_signal(seg, frame=frame_size, hop=hop_size)
    w = get_window(window, frame_size, fftbins=True).astype(np.float64)
    frames_w = frames * w[None, :]

    feats = []
    for i in range(frames_w.shape[0]):
        # already windowed; use boxcar to match builder’s “already windowed” behavior
        freqs, mag = spectrum_mag(frames_w[i], fs, window="boxcar")
        y = mag.astype(np.float64)
        if band_logmag:
            y = np.log1p(y)
        band = pool_to_log_bands(freqs, y, edges, agg=band_pool_agg)
        feats.append(band)

    mat = np.stack(feats, axis=0) if feats else np.zeros((1, len(edges) - 1), dtype=np.float64)
    v = aggregate_feature(mat, agg=feat_agg)
    return l2_normalize(v)


# -----------------------------
# LUT model
# -----------------------------
@dataclass
class LutClass:
    label: str
    base_note: str
    string_idx: int
    midi: int
    f0_hz: float
    band_med: np.ndarray
    band_std: np.ndarray
    harm_mask: Optional[np.ndarray]


def load_lut_classes(lut_path: str) -> Tuple[Dict[str, Any], List[LutClass]]:
    lut = load_json(lut_path)
    meta = lut.get("meta", {})
    entries = lut.get("entries", [])

    classes: List[LutClass] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        label = str(e.get("note", "")).strip()
        if not label:
            continue

        band_med = np.array(e.get("band_template_median", []), dtype=np.float64)
        band_std = np.array(e.get("band_template_std", []), dtype=np.float64)

        if band_med.size == 0:
            takes = e.get("takes", [])
            if takes:
                band_med = np.array(takes[0].get("band_template", []), dtype=np.float64)
                band_std = np.zeros_like(band_med)

        if band_std.size != band_med.size:
            band_std = np.zeros_like(band_med)

        harm_mask = None
        takes = e.get("takes", [])
        if takes and isinstance(takes, list) and isinstance(takes[0], dict) and "harm_band_mask" in takes[0]:
            hm = np.array(takes[0].get("harm_band_mask", []), dtype=np.float64)
            if hm.size == band_med.size:
                harm_mask = hm

        classes.append(
            LutClass(
                label=label,
                base_note=str(e.get("base_note", "")),
                string_idx=int(e.get("string_idx", 0)),
                midi=int(e.get("midi", -1)),
                f0_hz=float(e.get("f0_hz", 0.0)),
                band_med=l2_normalize(band_med),
                band_std=band_std,
                harm_mask=harm_mask,
            )
        )

    if not classes:
        raise RuntimeError("No usable entries found in LUT.")

    return lut, classes


# -----------------------------
# Scoring
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def compute_weight_from_std(std: np.ndarray, eps: float = 1e-3, power: float = 1.0) -> np.ndarray:
    w = 1.0 / (std + float(eps))
    if power != 1.0:
        w = w ** float(power)
    w /= (np.mean(w) + 1e-12)
    return w


def load_string_importance_weights(
    lut: Dict[str, Any],
    *,
    use_importance: str,  # "none"|"raw"|"residual"
) -> Dict[int, np.ndarray]:
    """
    Returns dict: string_idx -> weights (len = n_bands)
    """
    use_importance = (use_importance or "none").lower()
    out: Dict[int, np.ndarray] = {}

    if use_importance == "none":
        return out

    sbi = lut.get("string_band_importance", {})
    strings = sbi.get("strings", {})
    if not isinstance(strings, dict):
        return out

    for k, v in strings.items():
        try:
            si = int(k)
        except Exception:
            continue
        if not isinstance(v, dict):
            continue

        if use_importance == "residual" and v.get("importance_residual") is not None:
            w = np.array(v.get("importance_residual", []), dtype=np.float64)
        else:
            w = np.array(v.get("importance_raw", []), dtype=np.float64)

        if w.size > 0:
            w[w < 0.0] = 0.0
            w = w / (np.mean(w) + 1e-12)  # keep scale stable
            out[si] = w

    return out


def score_against_lut(
    feat: np.ndarray,
    classes: List[LutClass],
    *,
    use_std_weight: bool,
    std_eps: float,
    std_power: float,
    offharm_penalty_alpha: float,
    string_importance: Dict[int, np.ndarray],
) -> List[Tuple[float, LutClass]]:
    scored: List[Tuple[float, LutClass]] = []

    for c in classes:
        if feat.size != c.band_med.size:
            continue

        w_total = None

        # per-string importance weights
        if c.string_idx in string_importance:
            w_total = string_importance[c.string_idx].copy()
            if w_total.size != feat.size:
                w_total = None

        # std-based weights
        if use_std_weight:
            w_std = compute_weight_from_std(c.band_std, eps=std_eps, power=std_power)
            if w_total is None:
                w_total = w_std
            else:
                w_total = w_total * w_std

        if w_total is None:
            sim = cosine_sim(feat, c.band_med)
        else:
            # weighted cosine via scaling vectors
            aw = feat * w_total
            bw = c.band_med * w_total
            sim = cosine_sim(aw, bw)

        penalty = 0.0
        if offharm_penalty_alpha > 0 and c.harm_mask is not None and c.harm_mask.size == feat.size:
            penalty = float(np.sum(feat * (1.0 - c.harm_mask)))

        score = float(sim - offharm_penalty_alpha * penalty)
        scored.append((score, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def should_accept(best_score: float, second_score: float, *, abs_thresh: float, margin_thresh: float) -> bool:
    if best_score < abs_thresh:
        return False
    if (best_score - second_score) < margin_thresh:
        return False
    return True


# -----------------------------
# Live runtime config
# -----------------------------
@dataclass
class RuntimeConfig:
    sr: int

    # feature settings
    band_fmin: float
    band_fmax: float
    band_n: int
    band_pool_agg: str
    band_logmag: bool
    window: str
    feat_agg: str
    frame_ms: float
    hop_ms: float

    # onset-aligned segment
    use_onset: bool
    post_onset: float
    dur: float
    onset_thresh_ratio: float
    onset_hold_frames: int
    onset_search_start: float
    onset_search_end: float  # 0 => end
    onset_frame_ms: float
    onset_hop_ms: float

    # conditioning
    highpass: bool
    highpass_hz: float
    normalize_mode: str
    target_rms: float

    # activity gate (to avoid doing onset detect in silence)
    gate_frame_ms: float
    gate_open_ratio: float
    gate_close_ratio: float
    gate_hold_frames: int

    # scoring
    use_std_weight: bool
    std_eps: float
    std_power: float
    offharm_alpha: float
    abs_score_thresh: float
    margin_thresh: float
    use_importance: str  # none|raw|residual

    # output smoothing
    stable_count: int
    min_interval_ms: float


class ActivityGate:
    def __init__(self, open_ratio: float, close_ratio: float, hold_frames: int):
        self.open_ratio = float(open_ratio)
        self.close_ratio = float(close_ratio)
        self.hold_frames = max(1, int(hold_frames))
        self.is_open = False
        self._above = 0
        self._below = 0
        self._noise_hist: List[float] = []

    def update(self, rms_val: float) -> bool:
        self._noise_hist.append(float(rms_val))
        if len(self._noise_hist) > 200:
            self._noise_hist = self._noise_hist[-200:]

        arr = np.array(self._noise_hist, dtype=np.float64)
        if arr.size < 10:
            floor = float(np.median(arr) + 1e-12)
        else:
            arr_sorted = np.sort(arr)
            k = max(5, int(0.3 * arr_sorted.size))
            floor = float(np.median(arr_sorted[:k]) + 1e-12)

        open_th = floor * self.open_ratio
        close_th = floor * self.close_ratio

        if not self.is_open:
            self._above = self._above + 1 if rms_val >= open_th else 0
            if self._above >= self.hold_frames:
                self.is_open = True
                self._above = 0
        else:
            self._below = self._below + 1 if rms_val <= close_th else 0
            if self._below >= self.hold_frames:
                self.is_open = False
                self._below = 0

        return self.is_open


def run_match_loop_from_stream(lut: Dict[str, Any], classes: List[LutClass], cfg: RuntimeConfig, device: Optional[int]) -> None:
    if sd is None:
        raise RuntimeError("sounddevice not installed. pip install sounddevice")

    edges = make_log_band_edges(cfg.band_fmin, cfg.band_fmax, cfg.band_n)
    string_importance = load_string_importance_weights(lut, use_importance=cfg.use_importance)

    feat_frame = max(256, int(round(cfg.frame_ms / 1000.0 * cfg.sr)))
    feat_hop = max(64, int(round(cfg.hop_ms / 1000.0 * cfg.sr)))

    onset_frame = max(256, int(round(cfg.onset_frame_ms / 1000.0 * cfg.sr)))
    onset_hop = max(64, int(round(cfg.onset_hop_ms / 1000.0 * cfg.sr)))

    gate_frame = max(256, int(round(cfg.gate_frame_ms / 1000.0 * cfg.sr)))

    # for onset mode we need enough buffer to include pre-onset and post-onset+dur
    need_sec = 2.0 if cfg.use_onset else 1.0
    max_keep = int(round(need_sec * cfg.sr))

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)

    def callback(indata, frames, time_info, status):
        if status:
            pass
        x = indata[:, 0].astype(np.float64, copy=False)
        try:
            audio_q.put_nowait(x)
        except queue.Full:
            pass

    gate = ActivityGate(cfg.gate_open_ratio, cfg.gate_close_ratio, cfg.gate_hold_frames)

    last_emit_t = 0.0
    last_label = None
    stable_hits = 0

    buf = np.zeros(0, dtype=np.float64)

    print("Listening... (Ctrl+C to stop)")
    with sd.InputStream(
        samplerate=cfg.sr,
        channels=1,
        dtype="float32",
        callback=callback,
        device=device,
        blocksize=int(round(0.02 * cfg.sr)),  # ~20ms
    ):
        try:
            while True:
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                buf = np.concatenate([buf, chunk])
                if buf.size > max_keep:
                    buf = buf[-max_keep:]

                # activity gate on last gate_frame samples
                if buf.size >= gate_frame:
                    tail = buf[-gate_frame:]
                    rms_val = float(np.sqrt(np.mean(tail * tail) + 1e-12))
                    active = gate.update(rms_val)
                else:
                    active = False

                if not active:
                    last_label = None
                    stable_hits = 0
                    continue

                x = buf.copy()

                # conditioning (match builder order: normalize -> optional highpass -> normalize)
                x = normalize_audio(x, mode=cfg.normalize_mode, target_rms=cfg.target_rms)
                if cfg.highpass:
                    x = highpass_filter(x, cfg.sr, cutoff_hz=cfg.highpass_hz)
                    x = normalize_audio(x, mode=cfg.normalize_mode, target_rms=cfg.target_rms)

                if cfg.use_onset:
                    onset_sec = detect_onset_sec_energy(
                        x, cfg.sr,
                        frame=onset_frame,
                        hop=onset_hop,
                        thresh_ratio=cfg.onset_thresh_ratio,
                        hold_frames=cfg.onset_hold_frames,
                        search_start_sec=cfg.onset_search_start,
                        search_end_sec=None if cfg.onset_search_end <= 0 else cfg.onset_search_end,
                    )
                    seg_start = onset_sec + cfg.post_onset
                    seg = pick_segment(x, cfg.sr, seg_start, cfg.dur)
                else:
                    # fallback: take last dur seconds from buffer
                    seg = pick_segment(x, cfg.sr, max(0.0, (len(x) / cfg.sr) - cfg.dur), cfg.dur)

                seg = normalize_audio(seg, mode=cfg.normalize_mode, target_rms=cfg.target_rms)

                feat = compute_band_feature_multiframe(
                    seg, cfg.sr,
                    edges=edges,
                    window=cfg.window,
                    band_pool_agg=cfg.band_pool_agg,
                    band_logmag=cfg.band_logmag,
                    feat_agg=cfg.feat_agg,
                    frame_size=feat_frame,
                    hop_size=feat_hop,
                )

                scored = score_against_lut(
                    feat, classes,
                    use_std_weight=cfg.use_std_weight,
                    std_eps=cfg.std_eps,
                    std_power=cfg.std_power,
                    offharm_penalty_alpha=cfg.offharm_alpha,
                    string_importance=string_importance,
                )
                if len(scored) < 2:
                    continue

                (s1, c1), (s2, _c2) = scored[0], scored[1]
                if not should_accept(s1, s2, abs_thresh=cfg.abs_score_thresh, margin_thresh=cfg.margin_thresh):
                    last_label = None
                    stable_hits = 0
                    continue

                if last_label == c1.label:
                    stable_hits += 1
                else:
                    last_label = c1.label
                    stable_hits = 1

                now = time.time()
                if stable_hits >= cfg.stable_count and (now - last_emit_t) >= (cfg.min_interval_ms / 1000.0):
                    last_emit_t = now
                    print(f"{c1.label:8s}  score={s1:.3f}  margin={(s1 - s2):.3f}  midi={c1.midi}")

        except KeyboardInterrupt:
            print("\nStopped.")


# -----------------------------
# WAV-file mode
# -----------------------------
def read_wav_mono_float(path: str) -> Tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype.kind in "iu":
        x = x.astype(np.float64) / (float(np.iinfo(x.dtype).max) + 1e-12)
    else:
        x = x.astype(np.float64)
    return int(fs), x


def run_match_on_wav(lut: Dict[str, Any], classes: List[LutClass], cfg: RuntimeConfig, wav_path: str) -> None:
    fs, x = read_wav_mono_float(wav_path)
    if fs != cfg.sr:
        print(f"[warn] wav sr={fs} but --sr={cfg.sr}. Using wav sr.", file=sys.stderr)
        cfg = RuntimeConfig(**{**cfg.__dict__, "sr": fs})

    edges = make_log_band_edges(cfg.band_fmin, cfg.band_fmax, cfg.band_n)
    string_importance = load_string_importance_weights(lut, use_importance=cfg.use_importance)

    feat_frame = max(256, int(round(cfg.frame_ms / 1000.0 * cfg.sr)))
    feat_hop = max(64, int(round(cfg.hop_ms / 1000.0 * cfg.sr)))
    onset_frame = max(256, int(round(cfg.onset_frame_ms / 1000.0 * cfg.sr)))
    onset_hop = max(64, int(round(cfg.onset_hop_ms / 1000.0 * cfg.sr)))

    hop_len = int(round(0.05 * cfg.sr))  # scan step 50ms
    win_len = int(round(1.5 * cfg.sr))   # scan window to search onset in

    print(f"Scanning {wav_path} ...")
    t = 0
    while t < x.size:
        win = x[t:min(x.size, t + win_len)].copy()
        win = normalize_audio(win, mode=cfg.normalize_mode, target_rms=cfg.target_rms)
        if cfg.highpass:
            win = highpass_filter(win, cfg.sr, cutoff_hz=cfg.highpass_hz)
            win = normalize_audio(win, mode=cfg.normalize_mode, target_rms=cfg.target_rms)

        if cfg.use_onset:
            onset_sec = detect_onset_sec_energy(
                win, cfg.sr,
                frame=onset_frame,
                hop=onset_hop,
                thresh_ratio=cfg.onset_thresh_ratio,
                hold_frames=cfg.onset_hold_frames,
                search_start_sec=0.0,
                search_end_sec=None if cfg.onset_search_end <= 0 else cfg.onset_search_end,
            )
            seg_start = onset_sec + cfg.post_onset
            seg = pick_segment(win, cfg.sr, seg_start, cfg.dur)
        else:
            seg = pick_segment(win, cfg.sr, max(0.0, (len(win) / cfg.sr) - cfg.dur), cfg.dur)

        seg = normalize_audio(seg, mode=cfg.normalize_mode, target_rms=cfg.target_rms)

        feat = compute_band_feature_multiframe(
            seg, cfg.sr,
            edges=edges,
            window=cfg.window,
            band_pool_agg=cfg.band_pool_agg,
            band_logmag=cfg.band_logmag,
            feat_agg=cfg.feat_agg,
            frame_size=feat_frame,
            hop_size=feat_hop,
        )

        scored = score_against_lut(
            feat, classes,
            use_std_weight=cfg.use_std_weight,
            std_eps=cfg.std_eps,
            std_power=cfg.std_power,
            offharm_penalty_alpha=cfg.offharm_alpha,
            string_importance=string_importance,
        )
        if len(scored) >= 2:
            (s1, c1), (s2, _c2) = scored[0], scored[1]
            if should_accept(s1, s2, abs_thresh=cfg.abs_score_thresh, margin_thresh=cfg.margin_thresh):
                sec = t / cfg.sr
                print(f"{sec:8.3f}s  {c1.label:8s}  score={s1:.3f}  margin={(s1-s2):.3f}  midi={c1.midi}")

        t += hop_len


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut", required=True)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list_devices", action="store_true")
    ap.add_argument("--wav", type=str, default=None)

    # overrides (otherwise LUT meta used)
    ap.add_argument("--frame_ms", type=float, default=None)
    ap.add_argument("--hop_ms", type=float, default=None)
    ap.add_argument("--feat_agg", type=str, default=None, choices=["median", "mean"])
    ap.add_argument("--window", type=str, default=None)

    ap.add_argument("--band_fmin", type=float, default=None)
    ap.add_argument("--band_fmax", type=float, default=None)
    ap.add_argument("--band_n", type=int, default=None)
    ap.add_argument("--band_pool_agg", type=str, default=None, choices=["max", "mean"])
    ap.add_argument("--band_logmag", action="store_true", help="Force logmag on")

    # conditioning
    ap.add_argument("--highpass", action="store_true")
    ap.add_argument("--highpass_hz", type=float, default=80.0)
    ap.add_argument("--normalize", type=str, default=None, choices=["peak", "p95", "rms", "none"])
    ap.add_argument("--target_rms", type=float, default=None)

    # onset-aligned segment
    ap.add_argument("--use_onset", action="store_true", help="Use onset detection + post_onset + dur")
    ap.add_argument("--dur", type=float, default=None, help="analysis duration seconds")
    ap.add_argument("--post_onset", type=float, default=None)
    ap.add_argument("--onset_thresh_ratio", type=float, default=None)
    ap.add_argument("--onset_hold_frames", type=int, default=None)
    ap.add_argument("--onset_search_start", type=float, default=None)
    ap.add_argument("--onset_search_end", type=float, default=None, help="0 => end")
    ap.add_argument("--onset_frame_ms", type=float, default=None)
    ap.add_argument("--onset_hop_ms", type=float, default=None)

    # gate
    ap.add_argument("--gate_frame_ms", type=float, default=40.0)
    ap.add_argument("--gate_open_ratio", type=float, default=6.0)
    ap.add_argument("--gate_close_ratio", type=float, default=3.0)
    ap.add_argument("--gate_hold_frames", type=int, default=3)

    # scoring
    ap.add_argument("--abs_score", type=float, default=0.55)
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--use_std_weight", action="store_true")
    ap.add_argument("--std_eps", type=float, default=1e-3)
    ap.add_argument("--std_power", type=float, default=1.0)
    ap.add_argument("--offharm_alpha", type=float, default=0.0)

    ap.add_argument("--use_importance", type=str, default="none", choices=["none", "raw", "residual"])

    # output smoothing
    ap.add_argument("--stable_count", type=int, default=2)
    ap.add_argument("--min_interval_ms", type=float, default=120.0)

    args = ap.parse_args()

    if args.list_devices:
        if sd is None:
            print("sounddevice not installed.")
            return
        print(sd.query_devices())
        return

    lut, classes = load_lut_classes(args.lut)
    meta = lut.get("meta", {})
    band_meta = meta.get("band_template", {}) or {}
    analysis_meta = meta.get("analysis", {}) or {}
    onset_meta = (analysis_meta.get("onset", {}) or {})
    norm_meta = (analysis_meta.get("normalize", {}) or {})
    hp_meta = (analysis_meta.get("highpass", {}) or {})

    # pull defaults from LUT meta unless overridden
    band_fmin = args.band_fmin if args.band_fmin is not None else float(band_meta.get("fmin_hz", 40.0))
    band_fmax = args.band_fmax if args.band_fmax is not None else float(band_meta.get("fmax_hz", 8000.0))
    band_n = args.band_n if args.band_n is not None else int(band_meta.get("n_bands", 480))
    band_pool_agg = args.band_pool_agg if args.band_pool_agg is not None else str(band_meta.get("pool_agg", "max"))

    # logmag: if flag set -> True, else use LUT meta
    band_logmag = True if args.band_logmag else bool(band_meta.get("logmag", False))

    window = args.window if args.window is not None else str(analysis_meta.get("window", "hann"))
    feat_agg = args.feat_agg if args.feat_agg is not None else str((analysis_meta.get("multiframe", {}) or {}).get("feature_agg", "median"))

    frame_ms = args.frame_ms if args.frame_ms is not None else float(onset_meta.get("frame_ms", analysis_meta.get("frame_ms", 46.0)))
    hop_ms = args.hop_ms if args.hop_ms is not None else float(onset_meta.get("hop_ms", analysis_meta.get("hop_ms", 12.0)))

    dur = args.dur if args.dur is not None else float(analysis_meta.get("dur_sec", 0.20))
    post_onset = args.post_onset if args.post_onset is not None else float(analysis_meta.get("post_onset_sec", 0.06))

    onset_thresh_ratio = args.onset_thresh_ratio if args.onset_thresh_ratio is not None else float(onset_meta.get("thresh_ratio", 6.0))
    onset_hold_frames = args.onset_hold_frames if args.onset_hold_frames is not None else int(onset_meta.get("hold_frames", 3))
    onset_search_start = args.onset_search_start if args.onset_search_start is not None else float(onset_meta.get("search_start_sec", 0.0))
    onset_search_end = args.onset_search_end if args.onset_search_end is not None else float(onset_meta.get("search_end_sec", 0.0))

    # allow separate onset frame/hop, default to feature frame/hop
    onset_frame_ms = args.onset_frame_ms if args.onset_frame_ms is not None else float(onset_meta.get("frame_ms", frame_ms))
    onset_hop_ms = args.onset_hop_ms if args.onset_hop_ms is not None else float(onset_meta.get("hop_ms", hop_ms))

    normalize_mode = args.normalize if args.normalize is not None else str(norm_meta.get("mode", "p95"))
    target_rms = args.target_rms if args.target_rms is not None else float(norm_meta.get("target_rms", 0.10))

    # highpass: if flag set -> True, else use LUT meta default
    highpass = True if args.highpass else bool(hp_meta.get("enabled", False))

    cfg = RuntimeConfig(
        sr=int(args.sr),
        band_fmin=float(band_fmin),
        band_fmax=float(band_fmax),
        band_n=int(band_n),
        band_pool_agg=str(band_pool_agg),
        band_logmag=bool(band_logmag),
        window=str(window),
        feat_agg=str(feat_agg),
        frame_ms=float(frame_ms),
        hop_ms=float(hop_ms),

        use_onset=bool(args.use_onset),
        post_onset=float(post_onset),
        dur=float(dur),
        onset_thresh_ratio=float(onset_thresh_ratio),
        onset_hold_frames=int(onset_hold_frames),
        onset_search_start=float(onset_search_start),
        onset_search_end=float(onset_search_end),
        onset_frame_ms=float(onset_frame_ms),
        onset_hop_ms=float(onset_hop_ms),

        highpass=bool(highpass),
        highpass_hz=float(args.highpass_hz),
        normalize_mode=str(normalize_mode),
        target_rms=float(target_rms),

        gate_frame_ms=float(args.gate_frame_ms),
        gate_open_ratio=float(args.gate_open_ratio),
        gate_close_ratio=float(args.gate_close_ratio),
        gate_hold_frames=int(args.gate_hold_frames),

        use_std_weight=bool(args.use_std_weight),
        std_eps=float(args.std_eps),
        std_power=float(args.std_power),
        offharm_alpha=float(args.offharm_alpha),
        abs_score_thresh=float(args.abs_score),
        margin_thresh=float(args.margin),
        use_importance=str(args.use_importance),

        stable_count=int(args.stable_count),
        min_interval_ms=float(args.min_interval_ms),
    )

    # sanity check
    d0 = classes[0].band_med.size
    for c in classes:
        if c.band_med.size != d0:
            raise RuntimeError("LUT entries have inconsistent band sizes.")
    if d0 != cfg.band_n:
        print(f"[warn] LUT band dim={d0} but cfg.band_n={cfg.band_n}. Using LUT dim.", file=sys.stderr)
        cfg = RuntimeConfig(**{**cfg.__dict__, "band_n": d0})

    if args.wav:
        run_match_on_wav(lut, classes, cfg, args.wav)
    else:
        run_match_loop_from_stream(lut, classes, cfg, device=args.device)


if __name__ == "__main__":
    main()