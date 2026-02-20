#!/usr/bin/env python3
# lut_match_live.py
"""
Live (or file-based) note matcher for LUTs produced by lut_build.py (v3_multiframe_onset_stringaware).

What it does:
- Continuously grabs audio from your input device (mic / interface)
- Computes the SAME band-template feature style as the LUT builder (log-frequency pooled spectrum)
- Scores against LUT entries (uses entry-level median template + optional std weighting)
- Applies a rejection rule so it doesn't hallucinate notes in noise/chords/silence
- Prints detected note label like "E4_5" with a confidence score

Optional:
- Can run on a WAV file instead of mic (for testing)

Dependencies:
  pip install numpy scipy sounddevice

Examples:
  # Live, list devices:
  python lut_match_live.py --list_devices

  # Live match:
  python lut_match_live.py --lut lut.json --device 3 --sr 48000

  # Test on a wav file:
  python lut_match_live.py --lut lut.json --wav test.wav

Notes:
- This is monophonic-oriented. Chords may be rejected or misclassified (by design).
- For best results: record/build LUT with the same pickup/amp/room chain you use live.
"""

import argparse
import json
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, get_window
from scipy.io import wavfile

try:
    import sounddevice as sd
except Exception:
    sd = None


# -----------------------------
# Utility
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
    raise ValueError("normalize_mode must be one of: peak|p95|rms|none")


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


def spectrum_mag(seg: np.ndarray, fs: int, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    N = len(seg)
    w = get_window(window, N, fftbins=True)
    X = np.fft.rfft(seg * w)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    mag = np.abs(X)
    return freqs, mag


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
    """
    Produces the same kind of band_template feature as the builder:
      - per-frame: FFT mag (optionally log1p), pooled into log bands
      - aggregate frames (median/mean)
      - L2 normalize
    """
    if len(seg) < frame_size:
        seg = np.pad(seg, (0, frame_size - len(seg)))

    frames = frame_signal(seg, frame=frame_size, hop=hop_size)
    w = get_window(window, frame_size, fftbins=True).astype(np.float64)
    frames_w = frames * w[None, :]

    feats = []
    for i in range(frames_w.shape[0]):
        # already windowed, so use boxcar to avoid double-windowing
        freqs, mag = spectrum_mag(frames_w[i], fs, window="boxcar")
        y = mag.astype(np.float64)
        if band_logmag:
            y = np.log1p(y)
        band = pool_to_log_bands(freqs, y, edges, agg=band_pool_agg)
        feats.append(band)

    mat = np.stack(feats, axis=0) if feats else np.zeros((1, len(edges) - 1), dtype=np.float64)
    v = aggregate_feature(mat, agg=feat_agg)
    return l2_normalize(v)


def short_time_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    frames = frame_signal(x, frame=frame, hop=hop)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


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
    harm_mask: Optional[np.ndarray]  # from first take if present


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
            # fallback: if older LUT, try per-take
            takes = e.get("takes", [])
            if takes:
                band_med = np.array(takes[0].get("band_template", []), dtype=np.float64)
                band_std = np.zeros_like(band_med)
        if band_std.size != band_med.size:
            band_std = np.zeros_like(band_med)

        harm_mask = None
        takes = e.get("takes", [])
        if takes and isinstance(takes, list) and isinstance(takes[0], dict) and "harm_band_mask" in takes[0]:
            harm_mask = np.array(takes[0].get("harm_band_mask", []), dtype=np.float64)
            if harm_mask.size != band_med.size:
                harm_mask = None

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

    return meta, classes


# -----------------------------
# Scoring + rejection
# -----------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def weighted_cosine_sim(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted cosine: sim(a*w, b*w)
    w should be >=0.
    """
    aw = a * w
    bw = b * w
    return cosine_sim(aw, bw)


def compute_weight_from_std(std: np.ndarray, eps: float = 1e-3, power: float = 1.0) -> np.ndarray:
    """
    Lower std => higher weight.
    """
    w = 1.0 / (std + float(eps))
    if power != 1.0:
        w = w ** float(power)
    # normalize weights (optional)
    w /= (np.mean(w) + 1e-12)
    return w


def score_against_lut(
    feat: np.ndarray,
    classes: List[LutClass],
    *,
    use_std_weight: bool,
    std_eps: float,
    std_power: float,
    offharm_penalty_alpha: float,
) -> List[Tuple[float, LutClass]]:
    scored: List[Tuple[float, LutClass]] = []
    for c in classes:
        if feat.size != c.band_med.size:
            continue

        if use_std_weight:
            w = compute_weight_from_std(c.band_std, eps=std_eps, power=std_power)
            sim = weighted_cosine_sim(feat, c.band_med, w)
        else:
            sim = cosine_sim(feat, c.band_med)

        penalty = 0.0
        if offharm_penalty_alpha > 0 and c.harm_mask is not None and c.harm_mask.size == feat.size:
            # penalty is energy outside harmonic mask
            penalty = float(np.sum(feat * (1.0 - c.harm_mask)))
        score = float(sim - offharm_penalty_alpha * penalty)
        scored.append((score, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


def should_accept(
    best_score: float,
    second_score: float,
    *,
    abs_thresh: float,
    margin_thresh: float,
) -> bool:
    if best_score < abs_thresh:
        return False
    if (best_score - second_score) < margin_thresh:
        return False
    return True


# -----------------------------
# Live processing
# -----------------------------
@dataclass
class RuntimeConfig:
    sr: int
    frame_ms: float
    hop_ms: float
    seg_ms: float
    # activity detection (gate)
    gate_frame_ms: float
    gate_hop_ms: float
    gate_open_ratio: float
    gate_close_ratio: float
    gate_hold_frames: int
    # signal conditioning
    highpass: bool
    highpass_hz: float
    normalize_mode: str
    target_rms: float
    # feature
    band_fmin: float
    band_fmax: float
    band_n: int
    band_pool_agg: str
    band_logmag: bool
    feat_agg: str
    # scoring
    use_std_weight: bool
    std_eps: float
    std_power: float
    offharm_alpha: float
    abs_score_thresh: float
    margin_thresh: float
    # output smoothing
    stable_count: int
    min_interval_ms: float


class ActivityGate:
    """
    Simple RMS-based gate with hysteresis:
    - noise floor estimated on the fly from recent low-energy frames (median)
    - open when rms > floor*open_ratio for hold_frames
    - close when rms < floor*close_ratio for hold_frames
    """
    def __init__(self, open_ratio: float, close_ratio: float, hold_frames: int):
        self.open_ratio = float(open_ratio)
        self.close_ratio = float(close_ratio)
        self.hold_frames = max(1, int(hold_frames))

        self.is_open = False
        self._above = 0
        self._below = 0
        self._noise_hist: List[float] = []

    def update(self, rms_val: float) -> bool:
        # update noise floor estimate using lower half of observed RMS values
        self._noise_hist.append(float(rms_val))
        if len(self._noise_hist) > 200:
            self._noise_hist = self._noise_hist[-200:]

        # robust floor: median of lowest 30% of recent samples
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
            if rms_val >= open_th:
                self._above += 1
            else:
                self._above = 0
            if self._above >= self.hold_frames:
                self.is_open = True
                self._above = 0
        else:
            if rms_val <= close_th:
                self._below += 1
            else:
                self._below = 0
            if self._below >= self.hold_frames:
                self.is_open = False
                self._below = 0

        return self.is_open


def pick_latest_segment(buf: np.ndarray, seg_len: int) -> np.ndarray:
    if buf.size < seg_len:
        x = np.zeros(seg_len, dtype=np.float64)
        x[-buf.size:] = buf
        return x
    return buf[-seg_len:]


def run_match_loop_from_stream(meta: Dict[str, Any], classes: List[LutClass], cfg: RuntimeConfig, device: Optional[int]) -> None:
    if sd is None:
        raise RuntimeError("sounddevice is not installed. pip install sounddevice")

    edges = make_log_band_edges(cfg.band_fmin, cfg.band_fmax, cfg.band_n)

    seg_len = int(round(cfg.seg_ms / 1000.0 * cfg.sr))
    gate_frame = int(round(cfg.gate_frame_ms / 1000.0 * cfg.sr))
    gate_hop = int(round(cfg.gate_hop_ms / 1000.0 * cfg.sr))

    feat_frame = int(round(cfg.frame_ms / 1000.0 * cfg.sr))
    feat_hop = int(round(cfg.hop_ms / 1000.0 * cfg.sr))

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)

    def callback(indata, frames, time_info, status):
        if status:
            # don't spam; uncomment if needed
            # print(status, file=sys.stderr)
            pass
        x = indata[:, 0].astype(np.float64, copy=False)
        try:
            audio_q.put_nowait(x)
        except queue.Full:
            pass

    gate = ActivityGate(cfg.gate_open_ratio, cfg.gate_close_ratio, cfg.gate_hold_frames)

    # simple stability filter
    last_emit_t = 0.0
    last_label = None
    stable_hits = 0

    # ring buffer for recent audio
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
                # keep ~2 seconds max
                max_keep = int(2.0 * cfg.sr)
                if buf.size > max_keep:
                    buf = buf[-max_keep:]

                # update activity gate on the newest buffer region
                if buf.size >= gate_frame:
                    # compute RMS on last gate_frame window
                    tail = buf[-gate_frame:]
                    rms_val = float(np.sqrt(np.mean(tail * tail) + 1e-12))
                    active = gate.update(rms_val)
                else:
                    active = False

                if not active:
                    # reset stability when inactive
                    last_label = None
                    stable_hits = 0
                    continue

                seg = pick_latest_segment(buf, seg_len)

                # conditioning
                x = seg.copy()
                x = normalize_audio(x, mode=cfg.normalize_mode, target_rms=cfg.target_rms)
                if cfg.highpass:
                    x = highpass_filter(x, cfg.sr, cutoff_hz=cfg.highpass_hz)
                    x = normalize_audio(x, mode=cfg.normalize_mode, target_rms=cfg.target_rms)

                feat = compute_band_feature_multiframe(
                    x, cfg.sr,
                    edges=edges,
                    window="hann",
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
                )
                if len(scored) < 2:
                    continue
                (s1, c1), (s2, _c2) = scored[0], scored[1]

                accept = should_accept(
                    s1, s2,
                    abs_thresh=cfg.abs_score_thresh,
                    margin_thresh=cfg.margin_thresh,
                )
                if not accept:
                    last_label = None
                    stable_hits = 0
                    continue

                # stability filter (require same label stable_count times)
                if last_label == c1.label:
                    stable_hits += 1
                else:
                    last_label = c1.label
                    stable_hits = 1

                now = time.time()
                min_dt = cfg.min_interval_ms / 1000.0
                if stable_hits >= cfg.stable_count and (now - last_emit_t) >= min_dt:
                    last_emit_t = now
                    print(f"{c1.label:8s}  score={s1:.3f}  margin={(s1 - s2):.3f}  midi={c1.midi}")
        except KeyboardInterrupt:
            print("\nStopped.")


# -----------------------------
# WAV-file mode (for testing)
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


def run_match_on_wav(meta: Dict[str, Any], classes: List[LutClass], cfg: RuntimeConfig, wav_path: str) -> None:
    fs, x = read_wav_mono_float(wav_path)
    if fs != cfg.sr:
        print(f"[warn] wav sr={fs} but --sr={cfg.sr}. Using wav sr.", file=sys.stderr)
        cfg = RuntimeConfig(**{**cfg.__dict__, "sr": fs})

    edges = make_log_band_edges(cfg.band_fmin, cfg.band_fmax, cfg.band_n)
    seg_len = int(round(cfg.seg_ms / 1000.0 * cfg.sr))
    hop_len = int(round(0.05 * cfg.sr))  # 50ms step through file
    feat_frame = int(round(cfg.frame_ms / 1000.0 * cfg.sr))
    feat_hop = int(round(cfg.hop_ms / 1000.0 * cfg.sr))

    print(f"Scanning {wav_path} ...")
    t = 0
    while t + seg_len <= x.size:
        seg = x[t:t + seg_len]

        y = normalize_audio(seg, mode=cfg.normalize_mode, target_rms=cfg.target_rms)
        if cfg.highpass:
            y = highpass_filter(y, cfg.sr, cutoff_hz=cfg.highpass_hz)
            y = normalize_audio(y, mode=cfg.normalize_mode, target_rms=cfg.target_rms)

        feat = compute_band_feature_multiframe(
            y, cfg.sr,
            edges=edges,
            window="hann",
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
        )
        if len(scored) >= 2:
            (s1, c1), (s2, _c2) = scored[0], scored[1]
            if should_accept(s1, s2, abs_thresh=cfg.abs_score_thresh, margin_thresh=cfg.margin_thresh):
                sec = t / cfg.sr
                print(f"{sec:8.3f}s  {c1.label:8s} score={s1:.3f} margin={(s1-s2):.3f} midi={c1.midi}")

        t += hop_len


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut", required=True, help="Path to lut.json built by lut_build.py")
    ap.add_argument("--sr", type=int, default=48000, help="Sample rate to use for live input")
    ap.add_argument("--device", type=int, default=None, help="Input device index (use --list_devices)")
    ap.add_argument("--list_devices", action="store_true", help="List audio devices and exit")
    ap.add_argument("--wav", type=str, default=None, help="If provided, run matcher on this WAV instead of mic")

    # analysis window sizes
    ap.add_argument("--seg_ms", type=float, default=220.0, help="Segment length (ms) used for classification")
    ap.add_argument("--frame_ms", type=float, default=46.0, help="Feature frame length (ms)")
    ap.add_argument("--hop_ms", type=float, default=12.0, help="Feature hop length (ms)")
    ap.add_argument("--feat_agg", type=str, default="median", choices=["median", "mean"])

    # band template settings (should match LUT meta; defaults pulled from LUT if present)
    ap.add_argument("--band_fmin", type=float, default=None)
    ap.add_argument("--band_fmax", type=float, default=None)
    ap.add_argument("--band_n", type=int, default=None)
    ap.add_argument("--band_pool_agg", type=str, default=None, choices=["max", "mean"])
    ap.add_argument("--band_logmag", action="store_true", help="Force logmag on (else use LUT meta if available)")

    # conditioning
    ap.add_argument("--highpass", action="store_true")
    ap.add_argument("--highpass_hz", type=float, default=80.0)
    ap.add_argument("--normalize", type=str, default="p95", choices=["peak", "p95", "rms", "none"])
    ap.add_argument("--target_rms", type=float, default=0.10)

    # activity gate
    ap.add_argument("--gate_open_ratio", type=float, default=6.0)
    ap.add_argument("--gate_close_ratio", type=float, default=3.0)
    ap.add_argument("--gate_hold_frames", type=int, default=3)
    ap.add_argument("--gate_frame_ms", type=float, default=40.0)
    ap.add_argument("--gate_hop_ms", type=float, default=10.0)

    # scoring & rejection
    ap.add_argument("--abs_score", type=float, default=0.55, help="Absolute score threshold")
    ap.add_argument("--margin", type=float, default=0.06, help="Best-vs-2nd margin threshold")
    ap.add_argument("--use_std_weight", action="store_true", help="Use LUT std to weight stable bands more")
    ap.add_argument("--std_eps", type=float, default=1e-3)
    ap.add_argument("--std_power", type=float, default=1.0)
    ap.add_argument("--offharm_alpha", type=float, default=0.0, help="Penalty for off-harmonic energy (0 disables)")

    # output smoothing
    ap.add_argument("--stable_count", type=int, default=2, help="Require same label this many times before printing")
    ap.add_argument("--min_interval_ms", type=float, default=120.0, help="Min time between prints")

    args = ap.parse_args()

    if args.list_devices:
        if sd is None:
            print("sounddevice not installed.")
            return
        print(sd.query_devices())
        return

    meta, classes = load_lut_classes(args.lut)

    # pull band params from LUT meta if user didn't override
    band_meta = (meta.get("band_template") or {})
    band_fmin = args.band_fmin if args.band_fmin is not None else float(band_meta.get("fmin_hz", 40.0))
    band_fmax = args.band_fmax if args.band_fmax is not None else float(band_meta.get("fmax_hz", 8000.0))
    band_n = args.band_n if args.band_n is not None else int(band_meta.get("n_bands", 480))
    band_pool_agg = args.band_pool_agg if args.band_pool_agg is not None else str(band_meta.get("pool_agg", band_meta.get("agg", "max")))
    # if flag not set, respect LUT meta; if set, force True
    band_logmag = bool(args.band_logmag) if args.band_logmag else bool(band_meta.get("logmag", False))

    cfg = RuntimeConfig(
        sr=int(args.sr),
        frame_ms=float(args.frame_ms),
        hop_ms=float(args.hop_ms),
        seg_ms=float(args.seg_ms),
        gate_frame_ms=float(args.gate_frame_ms),
        gate_hop_ms=float(args.gate_hop_ms),
        gate_open_ratio=float(args.gate_open_ratio),
        gate_close_ratio=float(args.gate_close_ratio),
        gate_hold_frames=int(args.gate_hold_frames),
        highpass=bool(args.highpass),
        highpass_hz=float(args.highpass_hz),
        normalize_mode=str(args.normalize),
        target_rms=float(args.target_rms),
        band_fmin=float(band_fmin),
        band_fmax=float(band_fmax),
        band_n=int(band_n),
        band_pool_agg=str(band_pool_agg),
        band_logmag=bool(band_logmag),
        feat_agg=str(args.feat_agg),
        use_std_weight=bool(args.use_std_weight),
        std_eps=float(args.std_eps),
        std_power=float(args.std_power),
        offharm_alpha=float(args.offharm_alpha),
        abs_score_thresh=float(args.abs_score),
        margin_thresh=float(args.margin),
        stable_count=int(args.stable_count),
        min_interval_ms=float(args.min_interval_ms),
    )

    # quick sanity check for feature dim mismatch
    d = classes[0].band_med.size
    if d != cfg.band_n:
        # cfg.band_n should equal number of bands; band feature length is band_n
        # if LUT had different n_bands, we already pulled it, so this should match.
        pass
    for c in classes:
        if c.band_med.size != classes[0].band_med.size:
            raise RuntimeError("LUT entries have inconsistent band sizes.")

    if args.wav:
        run_match_on_wav(meta, classes, cfg, args.wav)
    else:
        run_match_loop_from_stream(meta, classes, cfg, device=args.device)


if __name__ == "__main__":
    main()