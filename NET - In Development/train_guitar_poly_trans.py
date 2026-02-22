#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler


# -----------------------------
# Utils
# -----------------------------

def list_npz_files(root: Path) -> List[Path]:
    return sorted(root.glob("shard_*.npz"))

def load_meta(data_dir: Path) -> dict:
    return json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

def shard_has_key(npz_path: Path, key: str) -> bool:
    with np.load(npz_path) as z:
        return key in z

def compute_count_from_YA(YA: np.ndarray, max_poly: int = 6) -> np.ndarray:
    # YA: [N,49] {0,1}
    c = YA.sum(axis=1).astype(np.int64)
    c = np.clip(c, 0, max_poly)
    return c

@torch.no_grad()
def frame_f1(pred_prob: torch.Tensor, targ01: torch.Tensor, thr: float) -> float:
    """
    pred_prob, targ01: [N,49]
    """
    p = (pred_prob >= thr).int()
    t = targ01.int()
    tp = (p & t).sum().item()
    fp = (p & (1 - t)).sum().item()
    fn = ((1 - p) & t).sum().item()
    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0


# -----------------------------
# Models
# -----------------------------

class CRNN(nn.Module):
    """
    Good for longer context windows where temporal evolution matters.
    Input: X [N,B,C]  (bands, frames)
    """
    def __init__(self, bands: int, hidden: int = 128, n_notes: int = 49, n_count: int = 7, use_onset: bool = True):
        super().__init__()
        self.use_onset = bool(use_onset)

        self.conv = nn.Sequential(
            nn.Conv1d(bands, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.gru = nn.GRU(128, hidden, batch_first=True, bidirectional=True)
        emb = hidden * 2

        self.head_active = nn.Linear(emb, n_notes)
        self.head_onset  = nn.Linear(emb, n_notes) if self.use_onset else None
        self.head_count  = nn.Linear(emb, n_count)

    def forward(self, x):
        h = self.conv(x)            # [N,128,C]
        h = h.transpose(1, 2)       # [N,C,128]
        out, _ = self.gru(h)        # [N,C,emb]
        last = out[:, -1, :]        # [N,emb]
        a = self.head_active(last)
        o = self.head_onset(last) if self.use_onset else None
        c = self.head_count(last)
        return a, o, c


class AttackCNN(nn.Module):
    """
    Recommended for attack-transient datasets (very short windows).
    No GRU; just conv + global pooling over time.
    Input: X [N,B,C]
    """
    def __init__(self, bands: int, n_notes: int = 49, n_count: int = 7, use_onset: bool = False):
        super().__init__()
        self.use_onset = bool(use_onset)

        self.net = nn.Sequential(
            nn.Conv1d(bands, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # collapse time
        )
        self.head_active = nn.Linear(256, n_notes)
        self.head_onset  = nn.Linear(256, n_notes) if self.use_onset else None
        self.head_count  = nn.Linear(256, n_count)

    def forward(self, x):
        h = self.net(x).squeeze(-1)   # [N,256]
        a = self.head_active(h)
        o = self.head_onset(h) if self.use_onset else None
        c = self.head_count(h)
        return a, o, c


# -----------------------------
# Stats from shards (pos_weight + count weights)
# -----------------------------

@torch.no_grad()
def compute_stats_from_shards(train_paths: List[Path],
                             n_notes: int = 49,
                             max_poly: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes:
      pos_weight[n_notes] = neg/pos, clipped
      count_weight[max_poly+1] inverse-freq, normalized, clipped
    Works even if shards don't contain YC (we compute it from YA).
    """
    pos = np.zeros((n_notes,), dtype=np.float64)
    total = 0
    count_hist = np.zeros((max_poly + 1,), dtype=np.int64)  # 0..max_poly

    for p in train_paths:
        with np.load(p) as z:
            YA = z["YA"].astype(np.int64)  # [N,n_notes]
            pos += YA.sum(axis=0)
            total += YA.shape[0]

            if "YC" in z:
                yc = z["YC"].astype(np.int64)
                yc = np.clip(yc, 0, max_poly)
            else:
                yc = compute_count_from_YA(YA, max_poly=max_poly)

        count_hist += np.bincount(yc, minlength=max_poly + 1)

    neg = total - pos
    pos_weight = (neg / (pos + 1.0)).astype(np.float32)
    pos_weight = np.clip(pos_weight, 1.0, 50.0)

    w = 1.0 / (count_hist.astype(np.float64) + 1.0)
    w = w / np.mean(w)
    w = np.clip(w, 0.2, 5.0).astype(np.float32)

    return pos_weight, w


# -----------------------------
# Training
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)

    ap.add_argument("--model", choices=["crnn", "attackcnn"], default="attackcnn",
                    help="attackcnn recommended for attack-transient dataset")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--hidden", type=int, default=128)

    ap.add_argument("--w_active", type=float, default=1.0)
    ap.add_argument("--w_onset",  type=float, default=0.0)
    ap.add_argument("--w_count",  type=float, default=0.3)

    ap.add_argument("--thr", type=float, default=0.4)

    ap.add_argument("--save", type=str, default="note_model.pt")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--shuffle_within_shard", action="store_true")

    ap.add_argument("--disable_onset_head", action="store_true",
                    help="Recommended for onset-anchored attack datasets (YO often useless).")
    ap.add_argument("--disable_count_head", action="store_true")
    ap.add_argument("--max_poly", type=int, default=6)

    ap.add_argument("--log_batches", type=int, default=200,
                    help="Print a training step log every N minibatches (0 to disable).")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta = load_meta(data_dir)

    # label sizes
    n_notes = int(meta.get("label", {}).get("n_notes", 49))
    max_poly = int(meta.get("label", {}).get("count_max", args.max_poly))
    n_count = max_poly + 1

    train_paths = list_npz_files(data_dir / "train")
    val_paths   = list_npz_files(data_dir / "val")
    assert train_paths and val_paths, "Missing shard_*.npz in train/ or val/"

    # infer bands from shard X shape
    with np.load(train_paths[0]) as z0:
        bands = int(z0["X"].shape[1])
        frames = int(z0["X"].shape[2])
    print(f"[info] X example shape: bands={bands} frames={frames}")

    # shard keys
    has_YO = shard_has_key(train_paths[0], "YO")
    has_YC = shard_has_key(train_paths[0], "YC")

    use_onset = (not args.disable_onset_head)
    use_count = (not args.disable_count_head)

    if use_onset and (not has_YO):
        print("[info] YO not found in shards -> using YO := YA (onset=active).")
    if use_count and (not has_YC):
        print("[info] YC not found in shards -> using YC := sum(YA) clamped 0..max_poly.")

    # weights (cached)
    pw_path = data_dir / "pos_weight.npy"
    cw_path = data_dir / "count_weight.npy"

    if pw_path.exists() and cw_path.exists():
        pos_weight_np = np.load(pw_path).astype(np.float32)
        count_w_np    = np.load(cw_path).astype(np.float32)
        if pos_weight_np.shape[0] != n_notes or count_w_np.shape[0] != n_count:
            print("[warn] Cached weights shape mismatch; recomputing...")
            pos_weight_np, count_w_np = compute_stats_from_shards(train_paths, n_notes=n_notes, max_poly=max_poly)
            np.save(pw_path, pos_weight_np)
            np.save(cw_path, count_w_np)
    else:
        print("Computing pos_weight + count weights from shards (one-time)...")
        pos_weight_np, count_w_np = compute_stats_from_shards(train_paths, n_notes=n_notes, max_poly=max_poly)
        np.save(pw_path, pos_weight_np)
        np.save(cw_path, count_w_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    pos_weight = torch.from_numpy(pos_weight_np).float().to(device)
    count_w    = torch.from_numpy(count_w_np).float().to(device)

    print("pos_weight range:", float(pos_weight.min()), float(pos_weight.max()))
    print("count class weights:", count_w.detach().cpu().numpy())

    # model selection
    if args.model == "crnn":
        model = CRNN(bands=bands, hidden=args.hidden, n_notes=n_notes, n_count=n_count, use_onset=use_onset).to(device)
    else:
        # attackcnn default: onset off
        model = AttackCNN(bands=bands, n_notes=n_notes, n_count=n_count, use_onset=use_onset).to(device)

    bce_active = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_onset  = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if use_onset else None
    ce_count   = nn.CrossEntropyLoss(weight=count_w) if use_count else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_f1 = -1.0

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt and device.type == "cuda":
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", ckpt.get("best_val", -1.0)))

    def run_split(paths: List[Path], train: bool):
        model.train(train)
        total_loss = 0.0
        f1a = 0.0
        f1o = 0.0
        nb = 0
        step = 0

        for sp in paths:
            with np.load(sp) as z:
                X  = z["X"].astype(np.float32)          # [N,B,C]
                YA = z["YA"].astype(np.float32)         # [N,n_notes]

                # YO optional
                if use_onset:
                    YO = z["YO"].astype(np.float32) if ("YO" in z) else YA
                else:
                    YO = None

                # YC optional
                if use_count:
                    if "YC" in z:
                        YC = z["YC"].astype(np.int64)
                        YC = np.clip(YC, 0, max_poly)
                    else:
                        YC = compute_count_from_YA(YA.astype(np.int64), max_poly=max_poly)
                else:
                    YC = None

            N = X.shape[0]
            idx = np.arange(N)
            if train and args.shuffle_within_shard:
                np.random.shuffle(idx)

            for start in range(0, N - args.batch + 1, args.batch):
                sel = idx[start:start + args.batch]

                xb  = torch.from_numpy(X[sel]).to(device, non_blocking=True)
                yab = torch.from_numpy(YA[sel]).to(device, non_blocking=True)
                yob = torch.from_numpy(YO[sel]).to(device, non_blocking=True) if YO is not None else None
                ycb = torch.from_numpy(YC[sel]).to(device, non_blocking=True) if YC is not None else None

                if train:
                    opt.zero_grad(set_to_none=True)
                    with autocast("cuda", enabled=(device.type == "cuda")):
                        a_log, o_log, c_log = model(xb)
                        loss = args.w_active * bce_active(a_log, yab)

                        if use_onset:
                            loss = loss + args.w_onset * bce_onset(o_log, yob)

                        if use_count:
                            loss = loss + args.w_count * ce_count(c_log, ycb)

                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    with torch.no_grad(), autocast("cuda", enabled=(device.type == "cuda")):
                        a_log, o_log, c_log = model(xb)
                        loss = args.w_active * bce_active(a_log, yab)
                        if use_onset:
                            loss = loss + args.w_onset * bce_onset(o_log, yob)
                        if use_count:
                            loss = loss + args.w_count * ce_count(c_log, ycb)

                total_loss += float(loss.item())
                nb += 1

                if train and args.log_batches > 0 and (step % args.log_batches == 0):
                    print(f"  train step {step} loss={float(loss.item()):.4f}")
                step += 1

                if not train:
                    a = torch.sigmoid(a_log)
                    f1a += frame_f1(a, yab, thr=args.thr)
                    if use_onset:
                        o = torch.sigmoid(o_log)
                        f1o += frame_f1(o, yob, thr=args.thr)

        if train:
            return total_loss / max(1, nb), None, None
        else:
            f1o_out = (f1o / max(1, nb)) if use_onset else None
            return total_loss / max(1, nb), f1a / max(1, nb), f1o_out

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, _, _ = run_split(train_paths, train=True)
        va_loss, f1a, f1o = run_split(val_paths, train=False)

        if f1o is None:
            print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | F1 active {f1a:.3f}")
        else:
            print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | F1 active {f1a:.3f} | F1 onset {f1o:.3f}")

        if f1a > best_f1:
            best_f1 = f1a
            torch.save({
                "model": model.state_dict(),
                "meta": meta,
                "args": vars(args),
                "epoch": epoch,
                "best_f1": best_f1,
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if device.type == "cuda" else None,
            }, args.save)
            print(f"  [saved] {args.save} (best active F1={best_f1:.3f})")


if __name__ == "__main__":
    main()