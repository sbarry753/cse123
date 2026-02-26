#!/usr/bin/env python3
"""
nn_fft_refine.py

NN + local FFT pitch refinement, robust for very short windows (e.g. 5 ms),
with:
- STRING "soft lock" (nudges FFT band but NOTE guess stays primary)
- TOP-K NOTE guidance (FFT band covers top-K NN notes, not just top-1)
- Low-string special-case: if detected string == 0, allow searching down to E2 (~82.41 Hz)
- Auto-widening cents band if too few FFT bins (prevents band_too_small)

Usage (live pedal / 5ms):
  python nn_fft_refine.py --checkpoint stage1_5ms_cos/onset_best.pt --metadata labels/metadata.json \
    --wav ../Dataset/note/C#3_1.wav --nn_window_ms 5 --fft_window_ms 5 --preemph_coef 0.95 \
    --search_cents 80 --n_fft 16384 --note_topk_refine 3
"""

import argparse
import json
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


# ----------------------------
# Model (same as training)
# ----------------------------
class OnsetNet(nn.Module):
    def __init__(self, note_vocab_size: int, width: int = 64, n_strings: int = 6):
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
        self.width = width
        self.n_strings = n_strings

    def forward(self, x: torch.Tensor):
        z = self.conv(x).squeeze(-1)
        return self.note_head(z), self.string_head(z)


def build_model_from_checkpoint(ckpt: dict, device: str = "cpu"):
    sd = ckpt["model_state"]
    note_vocab_size = int(sd["note_head.weight"].shape[0])
    width = int(sd["note_head.weight"].shape[1])
    n_strings = int(sd["string_head.weight"].shape[0])

    model = OnsetNet(note_vocab_size=note_vocab_size, width=width, n_strings=n_strings).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, note_vocab_size, width, n_strings


# ----------------------------
# Audio utils (match training)
# ----------------------------
def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float32), int(sr)


def rms_norm(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    return x / max(rms, 1e-4)


def apply_preemph(x: np.ndarray, coef: float) -> np.ndarray:
    if coef <= 0.0 or x.size < 2:
        return x
    y = x.copy()
    y[1:] = y[1:] - float(coef) * y[:-1]
    return y


def ms_to_samples(ms: float, sr: int) -> int:
    return int(round((ms / 1000.0) * float(sr)))


def safe_slice(y: np.ndarray, start: int, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    if start < 0:
        start = 0
    end = start + length
    if end > len(y):
        y = np.pad(y, (0, end - len(y)))
    return y[start:end].astype(np.float32)


# ----------------------------
# Onset/energy picker
# ----------------------------
def pick_loudest_frame_start(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 5.0,
    hop_ms: float = 1.0,
    avoid_first_ms: float = 0.0,
) -> int:
    frame = max(8, ms_to_samples(frame_ms, sr))
    hop = max(1, ms_to_samples(hop_ms, sr))
    start0 = ms_to_samples(avoid_first_ms, sr)

    if len(y) <= frame:
        return 0

    best_i = start0
    best_e = -1.0
    for i in range(start0, len(y) - frame + 1, hop):
        seg = y[i:i + frame]
        e = float(np.mean(seg * seg))
        if e > best_e:
            best_e = e
            best_i = i
    return int(best_i)


# ----------------------------
# Pitch mapping
# ----------------------------
def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_midi(freq: float) -> float:
    if not np.isfinite(freq) or freq <= 0.0:
        return float("nan")
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def cents_offset(f_refined: float, midi_equal: float) -> float:
    f_eq = midi_to_hz(midi_equal)
    if f_refined <= 0.0 or not np.isfinite(f_refined):
        return float("nan")
    return 1200.0 * np.log2(f_refined / f_eq)


# ----------------------------
# STRING soft-lock helpers
# ----------------------------
OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
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
        return lo0, hi0  # ignore lock if it collapses band
    return lo, hi


# ----------------------------
# FFT refinement (top-K guided band)
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
    f_centers: list[float],          # from top-K NN notes (in Hz), ordered best->worse
    center_weights: list[float],     # corresponding probabilities (same length)
    search_cents: float,
    n_fft: int,
    min_freq: float,
    max_freq: float,
    *,
    # String soft lock inputs
    string_idx: int | None = None,
    string_conf: float = 0.0,
    string_soft_lock_max: float = 0.55,
    max_fret: int = 24,
    # Ensure enough FFT bins:
    min_bins_in_band: int = 5,
    max_search_cents: float = 700.0,
    widen_factor: float = 1.6,
    # Harmonic scoring:
    use_harmonic_score: bool = True,
    max_harmonics: int = 5,
) -> tuple[float, float, str, float, tuple[float, float]]:
    """
    Returns (f_refined_hz, confidence, mode, used_search_cents, (lo,hi))

    Band creation:
    - For each center f_i, build [f_i/ratio, f_i*ratio]
    - Union them: lo = min(...), hi = max(...)
    - This keeps note top-1 as "most important" in selection scoring (below),
      while still letting FFT recover if top-1 is off by a semitone or two.

    Selection scoring:
    - Candidate bins are drawn from the band.
    - Each candidate frequency gets a score:
        harmonic_score(f)  *  (sum_i w_i * gaussian_in_cents(f around center_i))
      This makes the top-1 center influence strongest, but allows others.
    """
    # Filter valid centers
    fc = [(float(f), float(w)) for f, w in zip(f_centers, center_weights) if np.isfinite(f) and f > 0 and w > 0]
    if not fc:
        return float("nan"), 0.0, "no_centers", float(search_cents), (min_freq, max_freq)

    # Normalize weights
    ws = np.array([w for _, w in fc], dtype=np.float32)
    ws = ws / max(float(ws.sum()), 1e-12)
    fc = [(f, float(w)) for (f, _), w in zip(fc, ws.tolist())]

    x = x_1d.astype(np.float32)
    if x.size < 16:
        # fallback to best center
        return float(fc[0][0]), 0.0, "too_short", float(search_cents), (min_freq, max_freq)

    wwin = np.hanning(len(x)).astype(np.float32)
    xw = x * wwin
    if len(xw) < n_fft:
        xw = np.pad(xw, (0, n_fft - len(xw)))
    else:
        xw = xw[:n_fft]

    X = np.fft.rfft(xw)
    mag = np.abs(X).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    # Allow lower min freq for string 0 (low E): down to ~E2
    if string_idx is not None and int(string_idx) == 0:
        min_freq = min(float(min_freq), float(OPEN_STRING_HZ[0]) * 0.97)

    lock_w = float(np.clip(string_conf, 0.0, 1.0)) * float(string_soft_lock_max)
    s_lo, s_hi = (0.0, float("inf"))
    if string_idx is not None and 0 <= int(string_idx) < 6:
        s_lo, s_hi = string_freq_range(int(string_idx), max_fret=max_fret)

    used_cents = float(search_cents)
    while True:
        ratio = 2.0 ** (used_cents / 1200.0)

        # Union of top-K bands
        lo = float("inf")
        hi = 0.0
        for f, _w in fc:
            lo = min(lo, f / ratio)
            hi = max(hi, f * ratio)

        lo = max(float(min_freq), lo)
        hi = min(float(max_freq), hi)

        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            # fallback to best center
            return float(fc[0][0]), 0.0, "bad_band", used_cents, (lo, hi)

        lo, hi = apply_soft_string_lock(lo, hi, s_lo, s_hi, lock_w)

        band = np.where((freqs >= lo) & (freqs <= hi))[0]
        if band.size >= int(min_bins_in_band):
            break

        if used_cents >= float(max_search_cents):
            # fallback to best center
            return float(fc[0][0]), 0.0, "band_too_small_fallback", used_cents, (lo, hi)

        used_cents = min(float(max_search_cents), used_cents * float(widen_factor))

    # Candidate bins: top magnitudes within band
    band_mag = mag[band]
    M = int(min(30, band.size))
    top_local = np.argpartition(-band_mag, M - 1)[:M]
    cand_bins = band[top_local]

    # Gaussian width (in cents) for "closeness to centers"
    # Smaller => stronger lock to centers. Make this proportional to used_cents.
    sigma_cents = max(15.0, used_cents * 0.35)

    def center_prior(f_hz: float) -> float:
        # Weighted sum of gaussians in cents distance
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

        if use_harmonic_score:
            hs = harmonic_score(f0)
        else:
            hs = float(mag[k])

        score = hs * prior
        if score > best_score:
            best_score = score
            best_bin = int(k)

    k = best_bin
    delta = _parabolic_delta(mag, k)
    k_ref = float(k) + float(delta)
    f_ref = k_ref * (sr / float(n_fft))

    if not np.isfinite(f_ref) or f_ref <= 0.0:
        return float(fc[0][0]), 0.0, "fallback_badf", used_cents, (lo, hi)

    f_ref = float(np.clip(f_ref, lo, hi))

    med = float(np.median(mag[band]) + 1e-12)
    conf = float(best_score / med) if med > 0 else 0.0

    mode = "topk_harmonic" if use_harmonic_score else "topk_peak"
    return float(f_ref), float(conf), mode, used_cents, (lo, hi)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--metadata", required=True)

    ap.add_argument("--nn_window_ms", type=float, default=5.0)
    ap.add_argument("--fft_window_ms", type=float, default=5.0)
    ap.add_argument("--preemph_coef", type=float, default=0.95)

    ap.add_argument("--auto_onset", action="store_true")
    ap.add_argument("--no_auto_onset", action="store_true")
    ap.add_argument("--start_ms", type=float, default=0.0)
    ap.add_argument("--onset_frame_ms", type=float, default=5.0)
    ap.add_argument("--onset_hop_ms", type=float, default=1.0)
    ap.add_argument("--avoid_first_ms", type=float, default=0.0)

    ap.add_argument("--search_cents", type=float, default=80.0)
    ap.add_argument("--n_fft", type=int, default=16384)
    ap.add_argument("--min_freq", type=float, default=70.0)
    ap.add_argument("--max_freq", type=float, default=2000.0)

    # String soft lock controls
    ap.add_argument("--string_soft_lock_max", type=float, default=0.55)
    ap.add_argument("--max_fret", type=int, default=24)

    # Low-note band widening controls
    ap.add_argument("--min_bins_in_band", type=int, default=5)
    ap.add_argument("--max_search_cents", type=float, default=700.0)
    ap.add_argument("--widen_factor", type=float, default=1.6)

    # Top-K note guidance
    ap.add_argument("--note_topk_refine", type=int, default=3,
                    help="Use top-K NN notes to form the FFT band (K>=1).")

    ap.add_argument("--no_harmonic", action="store_true")
    ap.add_argument("--max_harmonics", type=int, default=5)

    ap.add_argument("--print_topk", type=int, default=5, help="Print top-k NN classes")
    args = ap.parse_args()

    use_auto = True
    if args.no_auto_onset:
        use_auto = False
    if args.auto_onset:
        use_auto = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Metadata
    with open(args.metadata, "r") as f:
        meta = json.load(f)
    midi_vocab = meta.get("midi_vocab", None)
    index_to_pitch = meta.get("index_to_pitch", None)
    if not isinstance(midi_vocab, list) or len(midi_vocab) == 0:
        raise RuntimeError("metadata.json must contain a non-empty 'midi_vocab' list.")

    midi_to_name = {}
    if isinstance(index_to_pitch, list):
        for i, m in enumerate(midi_vocab):
            if i < len(index_to_pitch):
                midi_to_name[int(m)] = str(index_to_pitch[i])

    ckpt = torch.load(args.checkpoint, map_location=device)
    model, note_vocab_size, width, n_strings = build_model_from_checkpoint(ckpt, device=device)

    y, sr = load_wav_mono(args.wav)

    if use_auto:
        st_energy = pick_loudest_frame_start(
            y, sr,
            frame_ms=args.onset_frame_ms,
            hop_ms=args.onset_hop_ms,
            avoid_first_ms=args.avoid_first_ms,
        )
        center = st_energy + ms_to_samples(args.onset_frame_ms, sr) // 2
    else:
        center = ms_to_samples(args.start_ms, sr)

    nn_W = ms_to_samples(args.nn_window_ms, sr)
    fft_W = ms_to_samples(args.fft_window_ms, sr)

    nn_start = int(center - nn_W // 2)
    fft_start = int(center - fft_W // 2)

    x_nn = safe_slice(y, nn_start, nn_W)
    x_fft = safe_slice(y, fft_start, fft_W)

    x_nn = apply_preemph(rms_norm(x_nn), args.preemph_coef)
    x_fft = apply_preemph(rms_norm(x_fft), args.preemph_coef)

    x_t = torch.from_numpy(x_nn).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        note_logits, string_logits = model(x_t)

    note_logits = note_logits[0].detach().cpu()
    string_logits = string_logits[0].detach().cpu()

    note_probs = torch.softmax(note_logits, dim=0)
    string_probs = torch.softmax(string_logits, dim=0)

    pred_idx = int(torch.argmax(note_probs).item())
    pred_p = float(note_probs[pred_idx].item())

    pred_string = int(torch.argmax(string_probs).item())
    pred_string_p = float(string_probs[pred_string].item())

    # Print top NN classes (for debug)
    print_k = min(int(args.print_topk), note_probs.numel())
    top_print_idxs = torch.topk(note_probs, k=print_k).indices.tolist()

    # Handle neg-class
    pred_note_name = f"class_{pred_idx}"
    midi_guess = None
    f_center = None

    if pred_idx >= len(midi_vocab):
        pred_note_name = "(none/neg_class)"
    else:
        midi_guess = float(midi_vocab[pred_idx])
        f_center = midi_to_hz(midi_guess)
        if isinstance(index_to_pitch, list) and pred_idx < len(index_to_pitch):
            pred_note_name = str(index_to_pitch[pred_idx])
        else:
            pred_note_name = f"MIDI {midi_guess:.2f}"

    print("\n=== NN + FFT (top-K guided) ===")
    print(f"WAV              : {args.wav}")
    print(f"SR               : {sr}")
    print(f"Checkpoint model : note_vocab={note_vocab_size}, width={width}, n_strings={n_strings}")
    print(f"Auto onset       : {use_auto} (center_sample={center}, center_ms={center/sr*1000.0:.2f})")
    print(f"NN window        : {args.nn_window_ms:.2f} ms ({nn_W} samples) start_sample={nn_start}")
    print(f"FFT window       : {args.fft_window_ms:.2f} ms ({fft_W} samples) start_sample={fft_start}")
    print(f"Preemph coef     : {args.preemph_coef:.2f}")
    print(f"Top-K refine     : K={int(args.note_topk_refine)}  (NOTE top-1 still dominates scoring)")

    print("\n--- NN prediction ---")
    print(f"NOTE   : idx={pred_idx}  name={pred_note_name}  p={pred_p:.3f}")
    print(f"STRING : {pred_string}  p={pred_string_p:.3f}  (soft_lock_max={args.string_soft_lock_max:.2f})")

    print("\nTop-k NOTE classes:")
    for i in top_print_idxs:
        if i >= len(midi_vocab):
            name = "(none/neg_class)"
        else:
            if isinstance(index_to_pitch, list) and i < len(index_to_pitch):
                name = str(index_to_pitch[i])
            else:
                name = f"MIDI {midi_vocab[i]}"
        print(f"  idx={i:3d}  p={float(note_probs[i]):.3f}  {name}")

    print("\n--- FFT refinement ---")
    if f_center is None:
        print("Skipped: predicted neg-class.")
        print()
        return

    # Build top-K centers (ignore neg-class entries)
    K = max(1, int(args.note_topk_refine))
    top_idxs = torch.topk(note_probs, k=min(K, note_probs.numel())).indices.tolist()

    f_centers = []
    center_weights = []
    center_names = []

    for ci in top_idxs:
        if ci >= len(midi_vocab):
            continue
        midi_c = float(midi_vocab[ci])
        f_c = midi_to_hz(midi_c)
        p_c = float(note_probs[ci].item())
        name_c = midi_to_name.get(int(round(midi_c)), f"MIDI {midi_c:.2f}")
        f_centers.append(float(f_c))
        center_weights.append(float(p_c))
        center_names.append((ci, name_c, p_c, midi_c))

    print("Top-K centers used for FFT band:")
    for ci, name_c, p_c, midi_c in center_names:
        print(f"  idx={ci:3d}  p={p_c:.3f}  {name_c}  (midi={midi_c:.2f})")

    f_refined, fft_conf, mode, used_cents, (lo, hi) = refine_pitch_fft_topk(
        x_1d=x_fft,
        sr=sr,
        f_centers=f_centers,
        center_weights=center_weights,
        search_cents=float(args.search_cents),
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

    if (not np.isfinite(f_refined)) or f_refined <= 0.0:
        f_refined = float(f_center)
        mode = mode + "_hardfallback"

    midi_ref = hz_to_midi(f_refined)
    midi_ref_rounded = int(np.round(midi_ref)) if np.isfinite(midi_ref) else -999
    cents = cents_offset(f_refined, midi_guess)

    refined_name = midi_to_name.get(midi_ref_rounded, f"MIDI {midi_ref_rounded}")

    print(f"Mode             : {mode} (used_search_cents={used_cents:.1f})")
    print(f"Band Hz          : [{lo:.2f}, {hi:.2f}]  (string_soft_lock applied)")
    print(f"NN top-1 center   : MIDI {midi_guess:.2f} -> {f_center:.2f} Hz")
    print(f"Refined frequency : {f_refined:.2f} Hz")
    if midi_ref_rounded != -999:
        print(f"Refined note      : {refined_name} (midi~{midi_ref:.2f}, rounded {midi_ref_rounded})")
    else:
        print("Refined note      : (midi unavailable)")
    if np.isfinite(cents):
        print(f"Cents offset      : {cents:+.2f} cents (refined vs NN top-1 equal-tempered center)")
    else:
        print("Cents offset      : (nan)")
    print(f"FFT confidence    : {fft_conf:.2f} (relative score/median band energy)")
    print()


if __name__ == "__main__":
    main()