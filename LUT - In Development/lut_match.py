#!/usr/bin/env python3
# lut_match.py
"""
Matcher (mono + poly) for the LUT built by lut_build.py (string-aware).

Monophonic:
  python lut_match.py match --lut lut.json --wav unknown.wav --start 0.12 --dur 0.18 --plot

Polyphonic:
  python lut_match.py match_poly --lut lut.json --wav chord.wav --start 0.10 --dur 0.25 \
    --algo omp --prune 40 --thresh 0.25 --collapsed --plot
"""

import argparse
import json
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, butter, sosfilt
from scipy.optimize import nnls
import matplotlib.pyplot as plt


# -----------------------------
# Shared small utilities
# -----------------------------
def split_label(label: str) -> Tuple[str, int]:
    s = label.strip().replace("♯", "#").replace("♭", "b")
    m = re.match(r"^([A-Ga-g])([#bB]?)(-?\d+)(?:_([0-5]))?$", s)
    if not m:
        return label, 0
    base = f"{m.group(1).upper()}{(m.group(2) or '').replace('B','b')}{m.group(3)}"
    idx = int(m.group(4)) if m.group(4) is not None else 0
    # normalize flats/sharps
    base = base.replace("♯", "#").replace("♭", "b")
    return base, idx


def load_lut(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("entries", [])
    data.setdefault("meta", {})
    return data


# -----------------------------
# Audio + spectrum
# -----------------------------
def read_wav_mono_float(wav_path: str) -> Tuple[int, np.ndarray]:
    fs, x = wavfile.read(wav_path)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype.kind in "iu":
        maxv = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / maxv
    else:
        x = x.astype(np.float64)
    return int(fs), x


def normalize_peak(x: np.ndarray) -> np.ndarray:
    return x / (float(np.max(np.abs(x)) + 1e-12))


def pick_segment(x: np.ndarray, fs: int, start_sec: float, dur_sec: float) -> np.ndarray:
    start = int(round(start_sec * fs))
    length = int(round(dur_sec * fs))
    seg = x[start:start + length]
    if len(seg) < length:
        seg = np.pad(seg, (0, length - len(seg)))
    return seg


def highpass_filter(x: np.ndarray, fs: int, cutoff_hz: float = 80.0, order: int = 4) -> np.ndarray:
    if cutoff_hz <= 0:
        return x
    sos = butter(order, cutoff_hz / (fs * 0.5), btype="highpass", output="sos")
    return sosfilt(sos, x)


def spectrum_mag(seg: np.ndarray, fs: int, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    N = len(seg)
    w = get_window(window, N, fftbins=True)
    X = np.fft.rfft(seg * w)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    mag = np.abs(X)
    return freqs, mag


# -----------------------------
# Band templates (must match LUT meta)
# -----------------------------
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def build_live_band(seg: np.ndarray, fs: int, lut_meta: Dict, window: str) -> np.ndarray:
    bt = lut_meta.get("band_template", {}) if isinstance(lut_meta, dict) else {}
    fmin = float(bt.get("fmin_hz", 40.0))
    fmax = float(bt.get("fmax_hz", 8000.0))
    nb = int(bt.get("n_bands", 480))
    agg = str(bt.get("agg", "max"))
    logmag = bool(bt.get("logmag", True))

    edges = make_log_band_edges(fmin, fmax, nb)
    freqs, mag = spectrum_mag(seg, fs, window=window)
    y = mag.astype(np.float64)
    if logmag:
        y = np.log1p(y)
    band = pool_to_log_bands(freqs, y, edges, agg=agg)
    band /= (np.linalg.norm(band) + 1e-12)
    return band


def score_take_band(live_band: np.ndarray, take: Dict, offharm_alpha: float) -> float:
    tb = take.get("band_template", None)
    if tb is None:
        return -1e9
    take_band = np.array(tb, dtype=np.float64)
    sim = cosine_similarity(live_band, take_band)

    if offharm_alpha <= 0.0:
        return float(sim)

    hm = take.get("harm_band_mask", None)
    if hm is None:
        return float(sim)

    harm_mask = (np.array(hm, dtype=np.float64) > 0.5).astype(np.float64)
    inv = 1.0 - harm_mask

    yin = float(np.sum(np.abs(live_band) * harm_mask)) + 1e-12
    yout = float(np.sum(np.abs(live_band) * inv))
    live_ratio = yout / yin

    tin = float(np.sum(np.abs(take_band) * harm_mask)) + 1e-12
    tout = float(np.sum(np.abs(take_band) * inv))
    take_ratio = tout / tin

    pen = abs(live_ratio - take_ratio)
    return float(sim - offharm_alpha * pen)


def collapse_take_scores(scores: List[float], mode: str, topk: int) -> float:
    if not scores:
        return 0.0
    mode = mode.lower()
    if mode == "max":
        return float(max(scores))
    if mode == "mean":
        return float(np.mean(scores))
    if mode == "topk":
        k = max(1, int(topk))
        ss = sorted(scores, reverse=True)[:k]
        return float(np.mean(ss))
    raise ValueError("score_mode must be max|mean|topk")


# -----------------------------
# Match (mono)
# -----------------------------
def match_mono(lut_path: str,
               wav_path: str,
               start: float,
               dur: float,
               window: Optional[str],
               use_highpass: bool,
               highpass_hz: float,
               top_n: int,
               plot: bool,
               score_mode: str,
               topk: int,
               offharm_alpha: float):
    lut = load_lut(lut_path)
    entries = lut.get("entries", [])
    if not entries:
        raise ValueError("LUT has no entries.")

    fs, x = read_wav_mono_float(wav_path)
    x = normalize_peak(x)
    seg = pick_segment(x, fs, start, dur)
    if use_highpass:
        seg = highpass_filter(seg, fs, cutoff_hz=highpass_hz)
    seg = normalize_peak(seg)

    win = window or str(entries[0].get("window", "hann"))
    live_band = build_live_band(seg, fs, lut.get("meta", {}), window=win)

    results: List[Tuple[str, float, int]] = []
    for e in entries:
        label = str(e.get("note", "?"))
        takes = e.get("takes", [])
        take_scores = [score_take_band(live_band, t, offharm_alpha=offharm_alpha) for t in takes if isinstance(t, dict)]
        score = collapse_take_scores(take_scores, mode=score_mode, topk=topk)
        best_take = int(np.argmax(take_scores)) if take_scores else -1
        results.append((label, float(score), best_take))

    results.sort(key=lambda t: t[1], reverse=True)
    best_label, best_score, best_take = results[0]
    base, sidx = split_label(best_label)

    print(f"\nBest match: {best_label}  score={best_score:.4f}  base={base}  string={sidx}  best_take={best_take}")
    print(f"\nTop {top_n} matches:")
    for lab, sc, bt in results[:top_n]:
        b, si = split_label(lab)
        print(f"  {lab:8s} score={sc:.4f}  base={b:4s} string={si}  best_take={bt}")

    if plot:
        freqs, mag = spectrum_mag(seg, fs, window=win)
        plt.figure(figsize=(12, 4))
        plt.plot(freqs, mag)
        plt.title("Live Segment Magnitude Spectrum")
        plt.xlabel("Hz")
        plt.ylabel("|X(f)|")
        plt.xlim(0, min(8000, freqs[-1]))
        plt.tight_layout()
        plt.show()

        labels = [r[0] for r in results[:top_n]]
        vals = [r[1] for r in results[:top_n]]
        plt.figure(figsize=(12, 4))
        plt.bar(labels, vals)
        plt.title(f"Top {top_n} Class Scores (mono)")
        plt.xlabel("Class label (note_string)")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.show()


# -----------------------------
# Match (poly)
# -----------------------------
def quick_prune(entries: List[Dict], live_band: np.ndarray, prune: int) -> List[str]:
    scored: List[Tuple[str, float]] = []
    for e in entries:
        label = str(e.get("note", "?"))
        best = -1e9
        for t in e.get("takes", []):
            if not isinstance(t, dict) or "band_template" not in t:
                continue
            tb = np.array(t["band_template"], dtype=np.float64)
            best = max(best, cosine_similarity(live_band, tb))
        scored.append((label, float(best)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [lab for lab, _ in scored[:max(1, int(prune))]]


def solve_nonneg_omp(A: np.ndarray,
                     y: np.ndarray,
                     max_notes: int,
                     min_corr: float = 1e-3,
                     min_improve: float = 1e-4) -> np.ndarray:
    _m, n = A.shape
    selected: List[int] = []
    r = y.copy()
    prev_norm = float(np.linalg.norm(r))
    x_full = np.zeros(n, dtype=np.float64)

    for _ in range(int(max_notes)):
        scores = A.T @ r
        j = int(np.argmax(scores))
        best = float(scores[j])
        if best < float(min_corr):
            break
        if j not in selected:
            selected.append(j)

        As = A[:, selected]
        x_s, _ = nnls(As, y)
        r_new = y - As @ x_s
        new_norm = float(np.linalg.norm(r_new))
        if (prev_norm - new_norm) < float(min_improve):
            if selected:
                selected.pop()
            break
        r = r_new
        prev_norm = new_norm

    if selected:
        As = A[:, selected]
        x_s, _ = nnls(As, y)
        for idx, val in zip(selected, x_s):
            x_full[idx] = float(val)
    return x_full


def match_poly(lut_path: str,
               wav_path: str,
               start: float,
               dur: float,
               algo: str,
               window: Optional[str],
               use_highpass: bool,
               highpass_hz: float,
               prune: int,
               max_notes: int,
               thresh: float,
               collapsed: bool,
               plot: bool):
    lut = load_lut(lut_path)
    entries = lut.get("entries", [])
    if not entries:
        raise ValueError("LUT has no entries.")

    fs, x = read_wav_mono_float(wav_path)
    x = normalize_peak(x)
    seg = pick_segment(x, fs, start, dur)
    if use_highpass:
        seg = highpass_filter(seg, fs, cutoff_hz=highpass_hz)
    seg = normalize_peak(seg)

    win = window or str(entries[0].get("window", "hann"))
    live_band = build_live_band(seg, fs, lut.get("meta", {}), window=win)
    y = live_band / (np.linalg.norm(live_band) + 1e-12)

    keep = set(quick_prune(entries, live_band, prune=prune))

    cols: List[np.ndarray] = []
    col_label: List[str] = []
    for e in entries:
        label = str(e.get("note", "?"))
        if label not in keep:
            continue
        for t in e.get("takes", []):
            if not isinstance(t, dict) or "band_template" not in t:
                continue
            templ = np.array(t["band_template"], dtype=np.float64)
            nrm = float(np.linalg.norm(templ))
            if nrm < 1e-9:
                continue
            cols.append(templ / (nrm + 1e-12))
            col_label.append(label)

    if not cols:
        print("No templates built. Rebuild LUT with lut_build.py.")
        return

    A = np.column_stack(cols)

    algo_l = (algo or "omp").lower()
    if algo_l == "nnls":
        xw, _ = nnls(A, y)
    elif algo_l == "omp":
        xw = solve_nonneg_omp(A, y, max_notes=max_notes)
    else:
        raise ValueError("algo must be nnls|omp")

    label_strength: Dict[str, float] = {}
    for w, lab in zip(xw, col_label):
        label_strength[lab] = max(label_strength.get(lab, 0.0), float(w))

    if not label_strength:
        print("No note strengths found.")
        return

    m = max(label_strength.values()) + 1e-12
    items = sorted(((lab, s / m) for lab, s in label_strength.items()), key=lambda p: p[1], reverse=True)

    out: List[Tuple[str, float]] = []
    for lab, s in items:
        if len(out) >= int(max_notes):
            break
        if float(s) < float(thresh):
            break
        out.append((lab, float(s)))

    print("\nDetected classes (poly):")
    if not out:
        print("  (none above threshold)")
    else:
        for lab, s in out:
            b, si = split_label(lab)
            print(f"  {lab:8s} strength={s:.3f}  base={b:4s} string={si}")

    if collapsed:
        base_strength: Dict[str, float] = {}
        for lab, s in label_strength.items():
            b, _si = split_label(lab)
            base_strength[b] = max(base_strength.get(b, 0.0), float(s))
        m2 = max(base_strength.values()) + 1e-12
        base_items = sorted(((b, v / m2) for b, v in base_strength.items()), key=lambda p: p[1], reverse=True)
        print("\nCollapsed base-note strengths:")
        for b, v in base_items[:12]:
            print(f"  {b:4s} strength={v:.3f}")

    if plot:
        freqs, mag = spectrum_mag(seg, fs, window=win)
        plt.figure(figsize=(12, 4))
        plt.plot(freqs, mag)
        plt.title("Live Segment Magnitude Spectrum")
        plt.xlabel("Hz")
        plt.ylabel("|X(f)|")
        plt.xlim(0, min(8000, freqs[-1]))
        plt.tight_layout()
        plt.show()

        top_show = min(12, len(items))
        labels = [lab for lab, _ in items[:top_show]]
        vals = [s for _lab, s in items[:top_show]]
        plt.figure(figsize=(12, 4))
        plt.bar(labels, vals)
        plt.title(f"Top Class Strengths (poly) algo={algo_l} prune={prune}")
        plt.xlabel("Class label (note_string)")
        plt.ylabel("Strength (normalized)")
        plt.tight_layout()
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_m = sub.add_parser("match", help="Monophonic match")
    ap_m.add_argument("--lut", required=True)
    ap_m.add_argument("--wav", required=True)
    ap_m.add_argument("--start", type=float, default=0.10)
    ap_m.add_argument("--dur", type=float, default=0.20)
    ap_m.add_argument("--window", type=str, default=None)
    ap_m.add_argument("--highpass", action="store_true")
    ap_m.add_argument("--highpass_hz", type=float, default=80.0)
    ap_m.add_argument("--top", type=int, default=8)
    ap_m.add_argument("--plot", action="store_true")
    ap_m.add_argument("--score_mode", type=str, default="max", choices=["max", "mean", "topk"])
    ap_m.add_argument("--topk", type=int, default=3)
    ap_m.add_argument("--offharm_alpha", type=float, default=1.0)

    ap_p = sub.add_parser("match_poly", help="Polyphonic match")
    ap_p.add_argument("--lut", required=True)
    ap_p.add_argument("--wav", required=True)
    ap_p.add_argument("--start", type=float, default=0.10)
    ap_p.add_argument("--dur", type=float, default=0.25)
    ap_p.add_argument("--algo", type=str, default="omp", choices=["nnls", "omp"])
    ap_p.add_argument("--window", type=str, default=None)
    ap_p.add_argument("--highpass", action="store_true")
    ap_p.add_argument("--highpass_hz", type=float, default=80.0)
    ap_p.add_argument("--prune", type=int, default=40)
    ap_p.add_argument("--max_notes", type=int, default=6)
    ap_p.add_argument("--thresh", type=float, default=0.25)
    ap_p.add_argument("--collapsed", action="store_true")
    ap_p.add_argument("--plot", action="store_true")

    args = ap.parse_args()

    if args.cmd == "match":
        match_mono(
            lut_path=args.lut,
            wav_path=args.wav,
            start=args.start,
            dur=args.dur,
            window=args.window,
            use_highpass=args.highpass,
            highpass_hz=args.highpass_hz,
            top_n=args.top,
            plot=args.plot,
            score_mode=args.score_mode,
            topk=args.topk,
            offharm_alpha=args.offharm_alpha,
        )
    elif args.cmd == "match_poly":
        match_poly(
            lut_path=args.lut,
            wav_path=args.wav,
            start=args.start,
            dur=args.dur,
            algo=args.algo,
            window=args.window,
            use_highpass=args.highpass,
            highpass_hz=args.highpass_hz,
            prune=args.prune,
            max_notes=args.max_notes,
            thresh=args.thresh,
            collapsed=args.collapsed,
            plot=args.plot,
        )


if __name__ == "__main__":
    main()