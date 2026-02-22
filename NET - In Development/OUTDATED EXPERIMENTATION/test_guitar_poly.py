#!/usr/bin/env python3
# test_guitar_poly.py
#
# Test the polyphonic guitar note detector on:
#   - A WAV file:   python test_guitar_poly.py --ckpt best.pt --wav my_guitar.wav
#   - Live input:   python test_guitar_poly.py --ckpt best.pt --live
#
# Install extras for live mode:
#   pip install sounddevice
#
# Output columns (live / per-frame):
#   [frame] cnt=N  E2────E6  notes: E2 A2 D3   ON:(E2)

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from scipy.io import wavfile

# ─────────────────────────────────────────────────
# MIDI / note name helpers
# ─────────────────────────────────────────────────

MIDI_MIN = 40   # E2
MIDI_MAX = 88   # E6
N_NOTES  = MIDI_MAX - MIDI_MIN + 1
MAX_POLY = 6

_MIDI_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(m: int) -> str:
    return f"{_MIDI_NAMES[m % 12]}{(m // 12) - 1}"


# ─────────────────────────────────────────────────
# Model  (must mirror train_guitar_poly.py exactly)
# ─────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=int(kernel_size),
                              dilation=int(dilation), padding=0, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(F.pad(x, (self.pad, 0)))))


class GuitarPolyTCN(nn.Module):
    def __init__(self, bands, notes=49, channels=64, n_count_classes=MAX_POLY + 1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(bands,    channels, 3, 1),
            CausalConv1d(channels, channels, 3, 2),
            CausalConv1d(channels, channels, 3, 4),
            CausalConv1d(channels, channels, 3, 8),
            CausalConv1d(channels, channels, 3, 16),
        )
        self.active_head = nn.Linear(channels, notes)
        self.onset_head  = nn.Linear(channels, notes)
        self.count_head  = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, n_count_classes),
        )

    def forward(self, x):
        h = self.net(x)[:, :, -1]
        return self.active_head(h), self.onset_head(h), self.count_head(h)


# ─────────────────────────────────────────────────
# Feature extraction  (mirrors dataset builder)
# ─────────────────────────────────────────────────

class FeatureExtractor:
    """
    Stateful: call push_samples() with each new audio chunk.
    Maintains a rolling context buffer of ctx_frames.
    """

    def __init__(self, meta: dict):
        self.sr         = int(meta["sr"])
        self.hop        = int(meta["hop_length"])
        self.n_fft      = int(meta["n_fft"])
        self.band_agg   = meta.get("band_agg", "mean")
        self.ctx_frames = int(meta["ctx_frames"])
        bands           = int(meta["bands"])

        fmin = max(10.0, float(meta["fmin"]))
        fmax = max(fmin * 1.01, float(meta["fmax"]))
        self.edges = np.geomspace(fmin, fmax, bands + 1).astype(np.float64)
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft).astype(np.float64)

        # pre-compute band masks once
        self.masks = [
            (self.freqs >= self.edges[b]) & (self.freqs < self.edges[b + 1])
            for b in range(bands)
        ]

        self.mu    = np.array(meta["norm"]["band_mu"],    dtype=np.float32)
        self.sigma = np.array(meta["norm"]["band_sigma"], dtype=np.float32)

        # rolling context buffer  (bands, ctx_frames)
        self.buffer = np.zeros((bands, self.ctx_frames), dtype=np.float32)

    # ------------------------------------------------------------------
    def push_samples(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """
        Accept exactly n_fft mono float32 samples (one STFT frame worth).
        Computes one feature vector, appends it to the rolling context buffer,
        and returns the current window (bands, ctx_frames).
        Returns None only if samples is too short to produce any STFT output.
        """
        if samples.size < self.n_fft:
            return None

        # Use hop_length=samples.size so we always get exactly 1 output frame
        # regardless of whether caller passes n_fft or n_fft+hop samples.
        S    = librosa.stft(samples.astype(np.float32),
                            n_fft=self.n_fft,
                            hop_length=max(1, samples.size - self.n_fft + 1),
                            window="hann",
                            center=False)
        P    = (np.abs(S) ** 2).astype(np.float32)   # (F, T)  T will be 1
        band = self._pool(P)                          # (bands, T)
        xlog = np.log1p(band)
        feat = np.clip((xlog - self.mu[:, None]) / (self.sigma[:, None] + 1e-6),
                       -6.0, 6.0)

        # Shift context buffer left by T, append new frame(s) on the right
        T = feat.shape[1]
        if T >= self.ctx_frames:
            self.buffer = feat[:, -self.ctx_frames:].copy()
        else:
            self.buffer = np.roll(self.buffer, -T, axis=1)
            self.buffer[:, -T:] = feat
        return self.buffer.copy()

    def _pool(self, P: np.ndarray) -> np.ndarray:
        out = np.zeros((len(self.masks), P.shape[1]), dtype=np.float32)
        for b, mask in enumerate(self.masks):
            if not np.any(mask):
                continue
            band = P[mask, :]
            out[b] = np.max(band, axis=0) if self.band_agg == "max" else np.mean(band, axis=0)
        return out

    def reset(self):
        self.buffer[:] = 0.0


# ─────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────

@torch.no_grad()
def infer(model, feat_window: np.ndarray, device,
          active_thresh: float = 0.5, onset_thresh: float = 0.5) -> dict:
    x = torch.from_numpy(feat_window[None]).to(device)      # (1, B, ctx)
    a_logits, o_logits, c_logits = model(x)
    a_probs = torch.sigmoid(a_logits)[0].cpu().numpy()      # (49,)
    o_probs = torch.sigmoid(o_logits)[0].cpu().numpy()      # (49,)
    count   = int(c_logits[0].argmax().item())               # 0-6
    return {
        "active":  [MIDI_MIN + i for i, p in enumerate(a_probs) if p >= active_thresh],
        "onset":   [MIDI_MIN + i for i, p in enumerate(o_probs) if p >= onset_thresh],
        "count":   count,
        "a_probs": a_probs,
        "o_probs": o_probs,
    }


# ─────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────

_ROLL_WIDTH = 49   # one char per note

def piano_roll_line(active_midis: List[int], onset_midis: set) -> str:
    """49-char ASCII piano roll, one cell per semitone E2..E6."""
    row = ["·"] * N_NOTES
    for m in active_midis:
        k = m - MIDI_MIN
        if 0 <= k < N_NOTES:
            row[k] = "█" if m in onset_midis else "▒"
    return "".join(row)


def format_line(frame_idx: int, t_sec: Optional[float], result: dict,
                no_roll: bool = False) -> str:
    active_names = [midi_to_name(m) for m in result["active"]]
    onset_set    = set(result["onset"])
    onset_names  = [midi_to_name(m) for m in result["onset"]]

    ts_str = f"t={t_sec:6.2f}s" if t_sec is not None else f"f={frame_idx:6d}"
    count  = result["count"]
    roll   = "" if no_roll else "  " + piano_roll_line(result["active"], onset_set)
    notes  = "  notes: " + (" ".join(active_names) if active_names else "—")
    ons    = ("  ON:(" + " ".join(onset_names) + ")") if onset_names else ""
    return f"  {ts_str}  cnt={count}{roll}{notes}{ons}"


# ─────────────────────────────────────────────────
# WAV mode
# ─────────────────────────────────────────────────

def run_wav(wav_path: Path, model, extractor: FeatureExtractor,
            device, args) -> List[dict]:

    print(f"\n{'─'*72}")
    print(f"  File   : {wav_path}")
    print(f"  E2{'·'*45}E6   (▒=active  █=onset)")
    print(f"{'─'*72}")

    # ── Load & resample ───────────────────────
    sr_file, audio = wavfile.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype.kind in "iu":
        audio = audio.astype(np.float32) / (float(np.iinfo(audio.dtype).max) + 1e-12)
    else:
        audio = audio.astype(np.float32)
    if int(sr_file) != extractor.sr:
        print(f"  Resampling {sr_file} → {extractor.sr} Hz ...")
        audio = librosa.resample(audio, orig_sr=int(sr_file), target_sr=extractor.sr)

    hop       = extractor.hop
    n_fft     = extractor.n_fft
    results   = []
    frame_idx = 0

    # Reset so no stale context from a previous file bleeds in
    extractor.reset()

    # Process exactly one hop at a time so the rolling context buffer
    # advances one frame per inference — this matches how the dataset was
    # built (center=False, each example = ctx_frames of history ending at t).
    #
    # Pre-pad with (n_fft - hop) zeros for causal alignment: the first
    # real hop of audio lands at position n_fft in the padded signal,
    # which is exactly what librosa center=False produces during training.
    pad   = np.zeros(n_fft - hop, dtype=np.float32)
    audio = np.concatenate([pad, audio])

    for start in range(0, len(audio) - n_fft + 1, hop):
        frame_audio = audio[start : start + n_fft]   # exactly n_fft samples
        feat_win    = extractor.push_samples(frame_audio)
        if feat_win is None:
            continue
        result = infer(model, feat_win, device,
                       active_thresh=args.active_thresh,
                       onset_thresh=args.onset_thresh)
        results.append(result)

        if not args.quiet:
            # t_sec relative to the original (unpadded) audio
            t_sec = max(0.0, (start - (n_fft - hop)) / extractor.sr)
            print(format_line(frame_idx, t_sec, result, no_roll=args.no_roll))
        frame_idx += 1

    # ── Summary ───────────────────────────────
    print(f"\n{'─'*72}")
    if results:
        counts    = [r["count"] for r in results]
        all_notes = set()
        for r in results:
            all_notes.update(r["active"])

        print(f"  Frames processed : {len(results)}")
        print(f"  Max polyphony    : {max(counts)}")
        print(f"  Mean polyphony   : {sum(counts)/len(counts):.2f}")
        print(f"  Non-silent pct   : {sum(1 for c in counts if c > 0)/len(counts)*100:.1f}%")
        if all_notes:
            print(f"  MIDI range seen  : {midi_to_name(min(all_notes))} – {midi_to_name(max(all_notes))}")
            print(f"  Notes heard      : {' '.join(midi_to_name(m) for m in sorted(all_notes))}")
        else:
            print("  No notes detected above threshold.")

        # polyphony histogram
        hist = [0] * (MAX_POLY + 1)
        for c in counts:
            hist[min(c, MAX_POLY)] += 1
        total = len(counts)
        print(f"\n  Polyphony histogram:")
        for k, v in enumerate(hist):
            bar = "█" * int(round(v / total * 40))
            print(f"    {k} notes : {bar} {v} frames ({v/total*100:.1f}%)")
    print(f"{'─'*72}")
    return results


# ─────────────────────────────────────────────────
# Live mode
# ─────────────────────────────────────────────────

def run_live(model, extractor: FeatureExtractor, device, args):
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not found.  Install it:  pip install sounddevice")
        sys.exit(1)

    sr        = extractor.sr
    hop       = extractor.hop
    blocksize = hop   # one hop per callback → one feature frame per callback

    print(f"\n{'─'*72}")
    print(f"  LIVE  sr={sr}  hop={hop}  blocksize={blocksize}")
    print(f"  device={args.device_id or 'default'}")
    print(f"  active_thresh={args.active_thresh}  onset_thresh={args.onset_thresh}")
    print(f"  E2{'·'*45}E6   (▒=active  █=onset)")
    print(f"  Ctrl+C to stop")
    print(f"{'─'*72}")

    frame_counter = [0]
    last_active   = [set()]

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n  [sd] {status}", file=sys.stderr)

        chunk    = indata[:, 0].astype(np.float32)
        feat_win = extractor.push_samples(chunk)
        if feat_win is None:
            return

        result = infer(model, feat_win, device,
                       active_thresh=args.active_thresh,
                       onset_thresh=args.onset_thresh)

        fc           = frame_counter[0]
        active_now   = set(result["active"])
        has_onset    = bool(result["onset"])
        changed      = active_now != last_active[0]
        should_print = changed or has_onset or (fc % args.idle_print == 0)

        if should_print:
            line = format_line(fc, None, result, no_roll=args.no_roll)
            # Overwrite line in terminal; newline on onset for readability
            end = "\n" if (has_onset or changed) else "\r"
            print(line + "   ", end=end, flush=True)

        last_active[0] = active_now
        frame_counter[0] += 1

    device_id = int(args.device_id) if (args.device_id and args.device_id.isdigit()) else args.device_id
    with sd.InputStream(
        samplerate=sr,
        blocksize=blocksize,
        channels=1,
        dtype="float32",
        device=device_id,
        callback=callback,
        latency="low",
    ):
        try:
            while True:
                time.sleep(0.01)
        except KeyboardInterrupt:
            print(f"\n  Stopped after {frame_counter[0]} frames.")


# ─────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────

def load_model_and_meta(ckpt_path: Path, meta_override: Optional[Path],
                         device: torch.device):
    ck       = torch.load(str(ckpt_path), map_location=device)
    meta_raw = ck.get("meta", {})

    # If norm stats are missing from the checkpoint, look for metadata.json nearby
    if "norm" not in meta_raw or meta_override:
        candidates = []
        if meta_override:
            candidates.append(Path(meta_override))
        candidates += [
            ckpt_path.parent / "train_meta.json",
            ckpt_path.parent / "metadata.json",
            ckpt_path.parent.parent / "metadata.json",
        ]
        for cand in candidates:
            if cand.exists():
                meta_raw = json.loads(cand.read_text(encoding="utf-8"))
                print(f"  Loaded norm stats from: {cand}")
                break
        else:
            raise FileNotFoundError(
                "Cannot find metadata.json with norm stats. "
                "Pass --meta_json /path/to/metadata.json explicitly."
            )

    bands    = int(ck.get("bands",    meta_raw.get("bands",    48)))
    channels = int(ck.get("channels", meta_raw.get("channels", 64)))

    model = GuitarPolyTCN(bands=bands, channels=channels).to(device)
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    epoch    = ck.get("epoch",    "?")
    best_val = ck.get("best_val", "?")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Epoch      : {epoch}   best_val={best_val}")
    print(f"  bands={bands}  channels={channels}  ctx={meta_raw.get('ctx_frames','?')}")
    return model, meta_raw


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Test polyphonic guitar detector on a WAV or live audio.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--ckpt", required=True,
                    help="Path to best.pt or last.pt")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wav",  type=str,
                      help="Path to a .wav file for offline testing.")
    mode.add_argument("--live", action="store_true",
                      help="Stream from a live audio input device.")

    ap.add_argument("--meta_json",     type=str,   default=None,
                    help="Override path to metadata.json (norm stats).")
    ap.add_argument("--active_thresh", type=float, default=0.5,
                    help="Sigmoid threshold for active notes  (default 0.5).")
    ap.add_argument("--onset_thresh",  type=float, default=0.4,
                    help="Sigmoid threshold for onset events  (default 0.4).")
    ap.add_argument("--device_id",     type=str,   default=None,
                    help="sounddevice input device index or name.\n"
                         "Run --list_devices to see options.")
    ap.add_argument("--idle_print",    type=int,   default=50,
                    help="Live: print heartbeat every N silent frames (default 50).")
    ap.add_argument("--no_roll",       action="store_true",
                    help="Disable the ASCII piano-roll column.")
    ap.add_argument("--quiet",         action="store_true",
                    help="WAV mode: suppress per-frame output, show summary only.")
    ap.add_argument("--cpu",           action="store_true",
                    help="Force CPU inference.")
    ap.add_argument("--list_devices",  action="store_true",
                    help="Print available audio input devices and exit.")
    args = ap.parse_args()

    # ── List audio devices ────────────────────
    if args.list_devices:
        try:
            import sounddevice as sd
            print(sd.query_devices())
        except ImportError:
            print("sounddevice not installed.  Run:  pip install sounddevice")
        sys.exit(0)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"\nTorch device : {device}")

    ckpt_path     = Path(args.ckpt)
    meta_override = Path(args.meta_json) if args.meta_json else None
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, meta = load_model_and_meta(ckpt_path, meta_override, device)
    extractor   = FeatureExtractor(meta)

    if args.live:
        run_live(model, extractor, device, args)
    else:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            print(f"WAV not found: {wav_path}")
            sys.exit(1)
        run_wav(wav_path, model, extractor, device, args)


if __name__ == "__main__":
    main()