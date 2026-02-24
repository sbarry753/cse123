import os
import json
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


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


def read_manifest(manifest_path: str) -> List[Dict[str, str]]:
    items = []
    with open(manifest_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def safe_int_pair(x):
    if isinstance(x, list) and len(x) == 2:
        return int(x[0]), int(x[1])
    return 0, 0


# ----------------------------
# Dataset
# ----------------------------
@dataclass
class ClipInfo:
    audio_abs: str
    label_abs: str
    multi_hot: List[int]
    transient_start: int
    transient_end: int
    sustain_start: int
    sustain_end: int
    num_samples: int
    sr: int


class TwoHeadWindowDataset(Dataset):
    """
    Randomly samples 5ms windows:
      - transient (on) windows -> y_on positive, y_hold = 0
      - sustain (hold) windows -> y_hold positive, y_on = 0
      - negative windows -> both zero

    Uses per-worker audio caching to avoid re-reading wav files constantly.
    """

    def __init__(
        self,
        clips: List[ClipInfo],
        window_samples: int,
        p_on: float = 0.34,
        p_hold: float = 0.33,
        p_neg: float = 0.33,
        audio_cache_max: int = 256,
        virtual_len: int = 200000,
    ):
        self.clips = clips
        self.W = int(window_samples)

        self.p_on = float(p_on)
        self.p_hold = float(p_hold)
        self.p_neg = float(p_neg)

        self.virtual_len = int(virtual_len)

        # per-worker cache
        self.audio_cache: Dict[str, np.ndarray] = {}
        self.audio_cache_order: List[str] = []
        self.audio_cache_max = int(audio_cache_max)

        self.on_candidates = [i for i, c in enumerate(self.clips) if (c.transient_end - c.transient_start) >= 2]
        self.hold_candidates = [i for i, c in enumerate(self.clips) if (c.sustain_end - c.sustain_start) >= self.W]

        if len(self.on_candidates) == 0:
            self.p_on = 0.0
        if len(self.hold_candidates) == 0:
            self.p_hold = 0.0

        s = self.p_on + self.p_hold + self.p_neg
        if s <= 0:
            self.p_neg = 1.0
            s = 1.0
        self.p_on /= s
        self.p_hold /= s
        self.p_neg /= s

    def __len__(self) -> int:
        # IMPORTANT: virtual length so DataLoader always yields batches even with huge batch sizes
        return self.virtual_len

    def _cache_put(self, path: str, y: np.ndarray):
        self.audio_cache[path] = y
        self.audio_cache_order.append(path)
        if len(self.audio_cache_order) > self.audio_cache_max:
            old = self.audio_cache_order.pop(0)
            self.audio_cache.pop(old, None)

    def _load_audio(self, audio_abs: str, sr_expected: int) -> np.ndarray:
        if audio_abs in self.audio_cache:
            return self.audio_cache[audio_abs]

        y, sr = sf.read(audio_abs, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if int(sr) != int(sr_expected):
            raise RuntimeError(f"SR mismatch in {audio_abs}: got {sr} expected {sr_expected}")

        y = y.astype(np.float32)
        self._cache_put(audio_abs, y)
        return y

    def _sample_mode(self) -> str:
        r = random.random()
        if r < self.p_on:
            return "on"
        r -= self.p_on
        if r < self.p_hold:
            return "hold"
        return "neg"

    def _sample_window_start_in_range(self, lo: int, hi: int, max_start: int) -> int:
        lo = int(max(lo, 0))
        hi = int(min(hi, max_start + 1))
        if hi <= lo:
            return lo
        return random.randrange(lo, hi)

    def _sample_negative_start(self, clip: ClipInfo) -> int:
        max_start = clip.num_samples - self.W
        if max_start <= 0:
            return 0

        # prefer after sustain_end
        a0 = clip.sustain_end
        if a0 + self.W <= clip.num_samples:
            return self._sample_window_start_in_range(a0, clip.num_samples - self.W + 1, max_start)

        # else before transient_start
        b1 = clip.transient_start
        if b1 - self.W > 0:
            return self._sample_window_start_in_range(0, b1 - self.W + 1, max_start)

        return random.randrange(0, max_start + 1)

    def __getitem__(self, _idx_ignored: int):
        mode = self._sample_mode()

        if mode == "on" and self.on_candidates:
            ci = self.clips[random.choice(self.on_candidates)]
            y = self._load_audio(ci.audio_abs, ci.sr)
            max_start = ci.num_samples - self.W

            t0, t1 = ci.transient_start, ci.transient_end
            jitter = self.W // 2
            lo = max(0, t0 - jitter)
            hi = min(max_start + 1, t1)
            start = self._sample_window_start_in_range(lo, hi, max_start)

            x = y[start:start + self.W]
            y_on = np.array(ci.multi_hot, dtype=np.float32)
            y_hold = np.array(ci.multi_hot, dtype=np.float32)

        elif mode == "hold" and self.hold_candidates:
            ci = self.clips[random.choice(self.hold_candidates)]
            y = self._load_audio(ci.audio_abs, ci.sr)
            max_start = ci.num_samples - self.W

            h0, h1 = ci.sustain_start, ci.sustain_end
            lo = h0
            hi = min(h1 - self.W + 1, max_start + 1)
            start = self._sample_window_start_in_range(lo, hi, max_start)

            x = y[start:start + self.W]
            y_hold = np.array(ci.multi_hot, dtype=np.float32)
            y_on = np.zeros_like(y_hold)

        else:
            ci = self.clips[random.randrange(0, len(self.clips))]
            y = self._load_audio(ci.audio_abs, ci.sr)
            start = self._sample_negative_start(ci)
            x = y[start:start + self.W]
            if len(x) < self.W:
                x = np.pad(x, (0, self.W - len(x)), mode="constant")
            y_on = np.zeros((len(ci.multi_hot),), dtype=np.float32)
            y_hold = np.zeros((len(ci.multi_hot),), dtype=np.float32)

        # cheap RMS normalize
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        x = x / max(rms, 1e-3)

        x_t = torch.from_numpy(x).float().unsqueeze(0)  # (1, W)
        y_on_t = torch.from_numpy(y_on).float()
        y_hold_t = torch.from_numpy(y_hold).float()
        return x_t, y_on_t, y_hold_t


# ----------------------------
# Model
# ----------------------------
class TinyConvBackbone(nn.Module):
    def __init__(self, in_ch: int = 1, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TwoHeadNet(nn.Module):
    def __init__(self, vocab_size: int, width: int = 64):
        super().__init__()
        self.backbone = TinyConvBackbone(1, width=width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.on_head = nn.Linear(width, vocab_size)
        self.hold_head = nn.Linear(width, vocab_size)

    def forward(self, x):
        z = self.backbone(x)
        z = self.pool(z).squeeze(-1)
        return self.on_head(z), self.hold_head(z)


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def f1_scores(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5, eps: float = 1e-9):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2.0 * prec * rec / (prec + rec + eps)
    return float(prec), float(rec), float(f1)


# ----------------------------
# Load clips
# ----------------------------
def load_clip_infos(dataset_root: str) -> Tuple[List[ClipInfo], int, int]:
    meta = load_json(os.path.join(dataset_root, "metadata.json"))
    sr = int(meta.get("sr", 48000))
    vocab_size = len(meta["midi_vocab"])

    manifest = read_manifest(os.path.join(dataset_root, "manifest.jsonl"))
    clips: List[ClipInfo] = []

    for it in manifest:
        audio_abs = os.path.join(dataset_root, it["audio"])
        label_abs = os.path.join(dataset_root, it["label"])
        lab = load_json(label_abs)

        mh = lab["multi_hot"]
        if len(mh) != vocab_size:
            raise RuntimeError(f"multi_hot size mismatch in {label_abs}: got {len(mh)} expected {vocab_size}")

        t0 = int(lab.get("transient_window_start", 0))
        t1 = int(lab.get("transient_window_end", 0))
        s0, s1 = safe_int_pair(lab.get("sustain_region", [t1, lab.get("num_samples", 0)]))

        n = int(lab.get("num_samples", 0))
        if n <= 0:
            y, _sr = sf.read(audio_abs, dtype="float32", always_2d=False)
            n = int(y.shape[0] if y.ndim == 1 else y.shape[0])

        clips.append(
            ClipInfo(
                audio_abs=audio_abs,
                label_abs=label_abs,
                multi_hot=mh,
                transient_start=t0,
                transient_end=t1,
                sustain_start=int(s0),
                sustain_end=int(s1),
                num_samples=n,
                sr=sr,
            )
        )

    return clips, sr, vocab_size


def train_val_split(clips: List[ClipInfo], val_ratio: float, seed: int):
    rng = random.Random(seed)
    idxs = list(range(len(clips)))
    rng.shuffle(idxs)
    n_val = int(round(len(idxs) * val_ratio))
    val_idx = set(idxs[:n_val])
    train = [c for i, c in enumerate(clips) if i not in val_idx]
    val = [c for i, c in enumerate(clips) if i in val_idx]
    return train, val


# ----------------------------
# Train
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--val_steps", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--audio_cache_max", type=int, default=256)
    ap.add_argument("--virtual_len", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--p_on", type=float, default=0.34)
    ap.add_argument("--p_hold", type=float, default=0.33)
    ap.add_argument("--p_neg", type=float, default=0.33)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    clips, sr, vocab_size = load_clip_infos(args.dataset)
    W = int(round(0.005 * sr))
    print(f"[INFO] clips={len(clips)} sr={sr} vocab={vocab_size} W={W}")

    train_clips, val_clips = train_val_split(clips, args.val_ratio, args.seed)
    print(f"[INFO] train_clips={len(train_clips)} val_clips={len(val_clips)}")

    train_ds = TwoHeadWindowDataset(
        train_clips, W, args.p_on, args.p_hold, args.p_neg,
        audio_cache_max=args.audio_cache_max,
        virtual_len=args.virtual_len,
    )
    val_ds = TwoHeadWindowDataset(
        val_clips, W, args.p_on, args.p_hold, args.p_neg,
        audio_cache_max=max(64, args.audio_cache_max // 2),
        virtual_len=max(20000, args.virtual_len // 10),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=False,  # IMPORTANT: prevents 0-batch deadlock
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        drop_last=False,
    )

    model = TwoHeadNet(vocab_size=vocab_size, width=args.width).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=(device == "cuda"))

    start_epoch = 1
    best_score = -1.0
    best_path = os.path.join(args.dataset, "twohead_best.pt")
    last_path = os.path.join(args.dataset, "twohead_last.pt")

    if args.resume is not None:
        print(f"[INFO] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt and device == "cuda":
            scaler.load_state_dict(ckpt["scaler_state"])
        best_score = float(ckpt.get("best_score", -1.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[INFO] start_epoch={start_epoch} best_score={best_score:.3f}")

    def infinite(loader):
        while True:
            for b in loader:
                yield b

    train_it = infinite(train_loader)
    val_it = infinite(val_loader)

    for ep in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for _ in range(args.steps_per_epoch):
            x, y_on, y_hold = next(train_it)
            x = x.to(device, non_blocking=True)
            y_on = y_on.to(device, non_blocking=True)
            y_hold = y_hold.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device == "cuda")):
                on_logits, hold_logits = model(x)
                loss_on = bce(on_logits, y_on)
                loss_hold = bce(hold_logits, y_hold)
                loss = 1.2 * loss_on + 1.0 * loss_hold

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            train_loss += float(loss.detach().item())

        scheduler.step()
        train_loss /= max(1, args.steps_per_epoch)

        # ---- Validation ----
        model.eval()
        v_loss = 0.0
        on_logits_all, on_targs_all = [], []
        hold_logits_all, hold_targs_all = [], []

        with torch.no_grad():
            for _ in range(args.val_steps):
                x, y_on, y_hold = next(val_it)
                x = x.to(device, non_blocking=True)
                y_on = y_on.to(device, non_blocking=True)
                y_hold = y_hold.to(device, non_blocking=True)

                on_logits, hold_logits = model(x)
                loss = 1.2 * bce(on_logits, y_on) + 1.0 * bce(hold_logits, y_hold)
                v_loss += float(loss.item())

                on_logits_all.append(on_logits.detach().cpu())
                on_targs_all.append(y_on.detach().cpu())
                hold_logits_all.append(hold_logits.detach().cpu())
                hold_targs_all.append(y_hold.detach().cpu())

        v_loss /= max(1, args.val_steps)
        on_logits_all = torch.cat(on_logits_all, dim=0)
        on_targs_all = torch.cat(on_targs_all, dim=0)
        hold_logits_all = torch.cat(hold_logits_all, dim=0)
        hold_targs_all = torch.cat(hold_targs_all, dim=0)

        p_on, r_on, f1_on = f1_scores(on_logits_all, on_targs_all, thresh=args.thresh)
        p_h, r_h, f1_h = f1_scores(hold_logits_all, hold_targs_all, thresh=args.thresh)
        score = 0.6 * f1_on + 0.4 * f1_h

        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[EP {ep:03d}] train_loss={train_loss:.4f} val_loss={v_loss:.4f} "
            f"F1_on={f1_on:.3f} (P={p_on:.3f} R={r_on:.3f}) "
            f"F1_hold={f1_h:.3f} (P={p_h:.3f} R={r_h:.3f}) "
            f"score={score:.3f} lr={lr_now:.2e} time={dt:.1f}s"
        )

        torch.save(
            {
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_score": best_score,
                "sr": sr,
                "window_samples": W,
                "vocab_size": vocab_size,
                "thresh": args.thresh,
                "args": vars(args),
            },
            last_path,
        )

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_score": best_score,
                    "sr": sr,
                    "window_samples": W,
                    "vocab_size": vocab_size,
                    "thresh": args.thresh,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"[INFO] Saved best -> {best_path} (best_score={best_score:.3f})")

    print("[DONE]")


if __name__ == "__main__":
    main()