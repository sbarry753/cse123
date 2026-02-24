"""
Onset-only training for <10ms guitar note detection on Daisy Seed.

Architecture:
  - OnsetNet: tiny conv, 480 samples (10ms @ 48kHz) input
  - Single head: 55-note multi-hot (but stage1 is single notes only)

Key design decisions:
  - Train ONLY on transient windows (first 10ms after pick attack)
  - Heavy gain augmentation to handle real-world dynamics
  - pos_weight=54.0 to counteract 1/55 class imbalance
  - No hold head (separate concern for sustain tracking)

Export:
  - Weights exported as raw C float arrays for bare metal Daisy
"""

import os
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


# ----------------------------
# Constants
# ----------------------------
WINDOW_MS = 10.0          # inference window on Daisy
PRE_ROLL_MS = 2.0         # how much before onset to include


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def read_manifest(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def f1_scores(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5):
    eps = 1e-9
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2.0 * prec * rec / (prec + rec + eps)
    return float(prec), float(rec), float(f1)


# ----------------------------
# Data
# ----------------------------
@dataclass
class ClipInfo:
    audio_abs: str
    multi_hot: List[int]
    onset_sample: int       # sample index of pick attack in the clip
    transient_end: int      # end of labeled transient region
    num_samples: int
    sr: int
    clip_type: str          # "single", "scale_segment", "chord"


def load_clips(dataset_root: str) -> Tuple[List[ClipInfo], int, int]:
    meta = load_json(os.path.join(dataset_root, "metadata.json"))
    sr = int(meta["sr"])
    vocab_size = len(meta["midi_vocab"])

    manifest = read_manifest(os.path.join(dataset_root, "manifest.jsonl"))
    clips = []

    for it in manifest:
        audio_abs = os.path.join(dataset_root, it["audio"])
        lab = load_json(os.path.join(dataset_root, it["label"]))

        mh = lab["multi_hot"]
        onset = int(lab.get("onset_sample", 0))
        t_end = int(lab.get("transient_window_end", onset + int(0.005 * sr)))
        n = int(lab.get("num_samples", 0))
        clip_type = lab.get("type", "unknown")

        clips.append(ClipInfo(
            audio_abs=audio_abs,
            multi_hot=mh,
            onset_sample=onset,
            transient_end=t_end,
            num_samples=n,
            sr=sr,
            clip_type=clip_type,
        ))

    return clips, sr, vocab_size


def train_val_split(clips: List[ClipInfo], val_ratio: float, seed: int):
    rng = random.Random(seed)
    idxs = list(range(len(clips)))
    rng.shuffle(idxs)
    n_val = int(round(len(idxs) * val_ratio))
    val_set = set(idxs[:n_val])
    train = [c for i, c in enumerate(clips) if i not in val_set]
    val   = [c for i, c in enumerate(clips) if i in val_set]
    return train, val


class OnsetDataset(Dataset):
    """
    Every sample is a W-sample window centered on a pick transient.
    
    Modes:
      "on"  (p_on):  window overlaps transient -> label = multi_hot
      "neg" (p_neg): window from silence before transient OR after sustain -> label = zeros
    
    Augmentations applied every sample:
      - Random gain (-18 to +6 dB)
      - Additive white noise (very low level)
      - Random polarity flip
      - Pre-emphasis (optional)
    """

    def __init__(
        self,
        clips: List[ClipInfo],
        window_samples: int,
        p_on: float,
        p_neg: float,
        virtual_len: int = 100000,
        audio_cache_max: int = 256,
        preemph_coef: float = 0.0,    # 0 = disabled
        noise_std: float = 0.001,
    ):
        self.clips = clips
        self.W = int(window_samples)
        self.virtual_len = int(virtual_len)
        self.preemph_coef = float(preemph_coef)
        self.noise_std = float(noise_std)

        s = p_on + p_neg
        self.p_on  = p_on / s
        self.p_neg = p_neg / s

        # Only clips with a clear transient window available
        self.on_candidates = [
            i for i, c in enumerate(clips)
            if c.transient_end > c.onset_sample
        ]

        self.audio_cache: Dict[str, np.ndarray] = {}
        self.cache_order: List[str] = []
        self.cache_max = int(audio_cache_max)

    def __len__(self):
        return self.virtual_len

    def _load(self, path: str, sr_expected: int) -> np.ndarray:
        if path in self.audio_cache:
            return self.audio_cache[path]
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        assert int(sr) == int(sr_expected), f"SR mismatch {path}"
        self.audio_cache[path] = y
        self.cache_order.append(path)
        if len(self.cache_order) > self.cache_max:
            old = self.cache_order.pop(0)
            self.audio_cache.pop(old, None)
        return y

    def _augment(self, x: np.ndarray) -> np.ndarray:
        # Gain
        gain_db = random.uniform(-18.0, 6.0)
        x = x * (10.0 ** (gain_db / 20.0))

        # Clip to [-1, 1] (simulate preamp saturation)
        x = np.clip(x, -1.0, 1.0)

        # White noise
        if self.noise_std > 0:
            x = x + np.random.randn(len(x)).astype(np.float32) * self.noise_std

        # Polarity flip
        if random.random() < 0.5:
            x = -x

        return x

    def _rms_norm(self, x: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        return x / max(rms, 1e-4)

    def __getitem__(self, _):
        mode = "on" if random.random() < self.p_on else "neg"

        if mode == "on" and self.on_candidates:
            ci = self.clips[random.choice(self.on_candidates)]
            y  = self._load(ci.audio_abs, ci.sr)

            # Window must overlap the transient
            # Place the onset somewhere inside the window with jitter
            max_start = max(0, ci.num_samples - self.W)
            pre = int(random.uniform(0, self.W * 0.7))  # onset lands at random position in window
            start = max(0, min(ci.onset_sample - pre, max_start))

            x = y[start: start + self.W]
            if len(x) < self.W:
                x = np.pad(x, (0, self.W - len(x)))

            label = np.array(ci.multi_hot, dtype=np.float32)

        else:
            # Negative: silence before transient or after sustain
            ci = self.clips[random.randrange(len(self.clips))]
            y  = self._load(ci.audio_abs, ci.sr)

            max_start = max(0, ci.num_samples - self.W)

            # Try to pull from pre-transient silence
            pre_end = max(0, ci.onset_sample - self.W)
            if pre_end > 0:
                start = random.randint(0, pre_end)
            else:
                # Pull from very end of file (after sustain dies)
                start = max(0, max_start - random.randint(0, min(max_start, int(0.1 * ci.sr))))

            start = int(np.clip(start, 0, max_start))
            x = y[start: start + self.W]
            if len(x) < self.W:
                x = np.pad(x, (0, self.W - len(x)))

            label = np.zeros(len(ci.multi_hot), dtype=np.float32)

        x = x.astype(np.float32)
        x = self._augment(x)
        x = self._rms_norm(x)

        # Pre-emphasis
        if self.preemph_coef > 0:
            x[1:] = x[1:] - self.preemph_coef * x[:-1]

        return (
            torch.from_numpy(x).unsqueeze(0),       # (1, W)
            torch.from_numpy(label),                 # (vocab_size,)
        )


# ----------------------------
# Model
# ----------------------------
class OnsetNet(nn.Module):
    """
    Tiny 1D conv sized for 480 samples (10ms @ 48kHz).
    
    MAC count estimate:
      Conv(1,16,9,s2):  9*1*16*240   =   34,560
      Conv(16,32,5,s2): 5*16*32*120  =  307,200
      Conv(32,32,5,s2): 5*32*32*60   =  307,200
      Linear(32,55):    32*55         =    1,760
      Total: ~650K MACs — fits in Daisy 10ms budget
    """

    def __init__(self, vocab_size: int, width: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            # (1, 480) -> (16, 240)
            nn.Conv1d(1, width // 2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width // 2),
            nn.ReLU(inplace=True),

            # (16, 240) -> (32, 120)
            nn.Conv1d(width // 2, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            # (32, 120) -> (32, 60)
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            # (32, 60) -> (32, 1)
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(width, vocab_size)
        self.vocab_size = vocab_size
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, W) -> logits: (B, vocab_size)"""
        z = self.conv(x).squeeze(-1)   # (B, width)
        return self.head(z)            # (B, vocab_size)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------
# Export to C header
# ----------------------------
def export_c_header(model: nn.Module, sr: int, vocab_size: int, path: str, meta: Dict):
    """
    Exports model weights as a C header for bare metal Daisy inference.
    Inference code will need a matching hand-written C conv forward pass.
    """
    model.eval()
    def ascii_pitch(p: str) -> str:
        # Replace unicode sharp/flat with ASCII equivalents for C strings
        return p.replace("\u266f", "#").replace("\u266d", "b")

    lines = [
        "// Auto-generated by train_onset.py - DO NOT EDIT",
        "// OnsetNet weights for Daisy Seed bare metal inference",
        f"// sr={sr} vocab_size={vocab_size} window_samples={int(round(WINDOW_MS/1000*sr))}",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define ONSET_SR           {sr}",
        f"#define ONSET_VOCAB_SIZE   {vocab_size}",
        f"#define ONSET_WINDOW       {int(round(WINDOW_MS/1000*sr))}",
        f"#define ONSET_WIDTH        {model.width}",
        "",
        "// Pitch names for each vocab index",
        "static const char* onset_pitch_names[] = {",
    ]
    for p in meta["index_to_pitch"]:
        lines.append(f'    "{ascii_pitch(p)}",')
    lines += ["};", ""]

    def arr(name: str, t: torch.Tensor):
        flat = t.detach().cpu().float().numpy().flatten()
        vals = ", ".join(f"{v:.8f}f" for v in flat)
        return [
            f"// shape: {list(t.shape)}",
            f"static const float {name}[{len(flat)}] = {{",
            f"    {vals}",
            "};",
            "",
        ]

    sd = model.state_dict()

    # Layer-by-layer export
    # Conv layers: weight (out, in, k), bias/bn params
    layer_map = [
        ("conv.0",  "l0_conv"),   # Conv1d
        ("conv.1",  "l0_bn"),     # BN
        ("conv.3",  "l1_conv"),
        ("conv.4",  "l1_bn"),
        ("conv.6",  "l2_conv"),
        ("conv.7",  "l2_bn"),
    ]

    for sd_prefix, c_prefix in layer_map:
        w_key = f"{sd_prefix}.weight"
        if w_key in sd:
            lines += arr(f"{c_prefix}_weight", sd[w_key])
        for suffix in ("bias", "running_mean", "running_var", "weight", "bias"):
            k = f"{sd_prefix}.{suffix}"
            # avoid double-exporting conv weight
            if k in sd and not (suffix == "weight" and "conv" in sd_prefix and k == w_key):
                lines += arr(f"{c_prefix}_{suffix}", sd[k])

    lines += arr("head_weight", sd["head.weight"])
    lines += arr("head_bias",   sd["head.bias"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[INFO] Exported C header -> {path}")


# ----------------------------
# Training
# ----------------------------
def train(args):
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    clips, sr, vocab_size = load_clips(args.dataset)
    meta = load_json(os.path.join(args.dataset, "metadata.json"))

    W = int(round(WINDOW_MS / 1000.0 * sr))
    print(f"[INFO] clips={len(clips)} sr={sr} vocab={vocab_size} window={W} samples ({WINDOW_MS}ms)")

    # Stage 1: no chords
    if not args.include_chords:
        clips = [c for c in clips if c.clip_type != "chord"]
        print(f"[INFO] Stage1 (no chords): {len(clips)} clips")

    train_clips, val_clips = train_val_split(clips, args.val_ratio, args.seed)
    print(f"[INFO] train={len(train_clips)} val={len(val_clips)}")

    train_ds = OnsetDataset(
        train_clips, W,
        p_on=args.p_on, p_neg=args.p_neg,
        virtual_len=args.virtual_len,
        audio_cache_max=args.audio_cache_max,
        preemph_coef=args.preemph_coef,
        noise_std=args.noise_std,
    )
    val_ds = OnsetDataset(
        val_clips, W,
        p_on=args.p_on, p_neg=args.p_neg,
        virtual_len=max(10000, args.virtual_len // 10),
        audio_cache_max=64,
        preemph_coef=args.preemph_coef,
        noise_std=0.0,   # no noise aug at val
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(1, args.workers // 2), pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    model = OnsetNet(vocab_size=vocab_size, width=args.width).to(device)
    print(f"[INFO] params={model.count_params():,}")

    # pos_weight: counteract 1/vocab_size class imbalance for single-note case
    pw = torch.full((vocab_size,), 8.0).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device == "cuda"))

    # Cosine LR with warm restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_score = -1.0
    best_path = os.path.join(args.out, "onset_best.pt")
    last_path = os.path.join(args.out, "onset_last.pt")

    def infinite(loader):
        while True:
            for b in loader:
                yield b

    train_it = infinite(train_loader)
    val_it   = infinite(val_loader)

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = 0.0

        for _ in range(args.steps_per_epoch):
            x, y = next(train_it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss   = loss_fn(logits, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            tr_loss += float(loss.item())

        tr_loss /= args.steps_per_epoch
        scheduler.step()

        # Validation
        model.eval()
        v_loss = 0.0
        all_logits, all_targets = [], []

        with torch.no_grad():
            for _ in range(args.val_steps):
                x, y = next(val_it)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                v_loss += float(loss_fn(logits, y).item())
                all_logits.append(logits.cpu())
                all_targets.append(y.cpu())

        v_loss /= args.val_steps
        all_logits  = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)

        prec, rec, f1 = f1_scores(all_logits, all_targets, thresh=args.thresh)
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0

        print(
            f"[EP {ep:03d}] train={tr_loss:.4f} val={v_loss:.4f} "
            f"F1={f1:.3f} P={prec:.3f} R={rec:.3f} "
            f"lr={lr_now:.2e} t={dt:.1f}s"
        )

        ckpt = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "best_score": best_score,
            "sr": sr,
            "vocab_size": vocab_size,
            "window_samples": W,
            "width": args.width,
            "thresh": args.thresh,
            "args": vars(args),
        }
        torch.save(ckpt, last_path)

        if f1 > best_score:
            best_score = f1
            torch.save(ckpt, best_path)
            print(f"[INFO] New best F1={best_score:.3f} -> {best_path}")

            # Export C header on every new best
            c_header_path = os.path.join(args.out, "onset_weights.h")
            export_c_header(model, sr, vocab_size, c_header_path, meta)

    print(f"\nDone. Best F1={best_score:.3f}")
    print(f"Weights: {best_path}")
    print(f"C header: {os.path.join(args.out, 'onset_weights.h')}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",         type=str,   default="dataset")
    ap.add_argument("--out",             type=str,   default="checkpoints")
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--epochs",          type=int,   default=80)
    ap.add_argument("--batch",           type=int,   default=2048)
    ap.add_argument("--width",           type=int,   default=32,    help="Model channel width (32=~650K MACs)")
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--workers",         type=int,   default=4)
    ap.add_argument("--audio_cache_max", type=int,   default=256)
    ap.add_argument("--virtual_len",     type=int,   default=100000)
    ap.add_argument("--val_ratio",       type=float, default=0.1)
    ap.add_argument("--steps_per_epoch", type=int,   default=300)
    ap.add_argument("--val_steps",       type=int,   default=60)
    ap.add_argument("--thresh",          type=float, default=0.5)
    ap.add_argument("--p_on",            type=float, default=0.6,   help="Fraction of transient windows")
    ap.add_argument("--p_neg",           type=float, default=0.4,   help="Fraction of silence windows")
    ap.add_argument("--preemph_coef",    type=float, default=0.97,  help="Pre-emphasis (0=off)")
    ap.add_argument("--noise_std",       type=float, default=0.001, help="Additive noise std")
    ap.add_argument("--include_chords",  action="store_true",       help="Include chord clips (stage 2)")
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()