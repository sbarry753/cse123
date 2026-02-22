#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler


# -----------------------------
# Utils
# -----------------------------

def list_npz_files(root: Path) -> List[Path]:
    return sorted(root.glob("shard_*.npz"))

def shard_has_key(npz_path: Path, key: str) -> bool:
    with np.load(npz_path) as z:
        return key in z

def load_meta(data_dir: Path) -> dict:
    return json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

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
# Model
# -----------------------------

class CRNN(nn.Module):
    """
    Input: X [N,B,C]  (bands, frames)
    """
    def __init__(self, bands: int, hidden: int = 128, n_notes: int = 49, n_count: int = 7):
        super().__init__()
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
        self.head_onset  = nn.Linear(emb, n_notes)
        self.head_count  = nn.Linear(emb, n_count)

    def forward(self, x):           # x [N,B,C]
        h = self.conv(x)            # [N,128,C]
        h = h.transpose(1, 2)       # [N,C,128]
        out, _ = self.gru(h)        # [N,C,emb]
        last = out[:, -1, :]
        return self.head_active(last), self.head_onset(last), self.head_count(last)


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
      count_weight[7] inverse-freq, normalized, clipped
    Works even if shards don't contain YC (we compute it from YA).
    """
    pos = np.zeros((n_notes,), dtype=np.float64)
    total = 0
    count_hist = np.zeros((max_poly + 1,), dtype=np.int64)  # 0..6

    for p in train_paths:
        z = np.load(p)
        YA = z["YA"].astype(np.int64)  # [N,49]
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

    # inverse freq, normalize
    w = 1.0 / (count_hist.astype(np.float64) + 1.0)
    w = w / np.mean(w)
    w = np.clip(w, 0.2, 5.0).astype(np.float32)

    return pos_weight, w


# -----------------------------
# Training loop over shards
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--w_active", type=float, default=1.0)
    ap.add_argument("--w_onset", type=float, default=0.3)
    ap.add_argument("--w_count", type=float, default=0.3)
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--save", type=str, default="note_crnn.pt")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--shuffle_within_shard", action="store_true")
    ap.add_argument("--disable_onset_head", action="store_true",
                    help="If set, don't train onset head (useful if YO doesn't exist / onset-anchored dataset).")
    ap.add_argument("--disable_count_head", action="store_true",
                    help="If set, don't train polyphony count head.")
    ap.add_argument("--max_poly", type=int, default=6)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta = load_meta(data_dir)

    # metadata fallbacks
    bands = int(meta.get("bands", 48))
    n_notes = int(meta.get("label", {}).get("n_notes", 49))
    max_poly = int(meta.get("label", {}).get("count_max", args.max_poly))
    n_count = max_poly + 1

    train_paths = list_npz_files(data_dir / "train")
    val_paths   = list_npz_files(data_dir / "val")
    assert train_paths and val_paths, "Missing shard_*.npz in train/ or val/"

    with np.load(train_paths[0]) as z0:
        bands = int(z0["X"].shape[1])
    print("[info] bands inferred from shard:", bands)
    val_paths   = list_npz_files(data_dir / "val")
    assert train_paths and val_paths, "Missing shard_*.npz in train/ or val/"

    # Detect label availability from first shard
    has_YO = shard_has_key(train_paths[0], "YO")
    has_YC = shard_has_key(train_paths[0], "YC")

    # If onset labels aren't present, default to "onset = active"
    if (not has_YO) and (not args.disable_onset_head):
        print("[info] YO not found in shards -> using YO := YA (onset=active).")

    # If count labels aren't present, we'll compute from YA
    if (not has_YC) and (not args.disable_count_head):
        print("[info] YC not found in shards -> using YC := sum(YA) clamped 0..6.")

    # Load / compute cached weights
    pw_path = data_dir / "pos_weight.npy"
    cw_path = data_dir / "count_weight.npy"

    if pw_path.exists() and cw_path.exists():
        pos_weight_np = np.load(pw_path).astype(np.float32)
        count_w_np    = np.load(cw_path).astype(np.float32)
        # safety: if count classes changed, recompute
        if count_w_np.shape[0] != n_count or pos_weight_np.shape[0] != n_notes:
            print("[warn] Cached weights shape mismatch; recomputing weights...")
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

    model = CRNN(bands=bands, hidden=args.hidden, n_notes=n_notes, n_count=n_count).to(device)

    pos_weight = torch.from_numpy(pos_weight_np).float().to(device)
    count_w    = torch.from_numpy(count_w_np).float().to(device)

    print("pos_weight range:", float(pos_weight.min()), float(pos_weight.max()))
    print("count class weights:", count_w.detach().cpu().numpy())

    bce_active = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_onset  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce_count   = nn.CrossEntropyLoss(weight=count_w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_f1 = -1.0

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "best_f1" in ckpt:
            best_f1 = float(ckpt["best_f1"])
        elif "best_val" in ckpt:
            best_f1 = float(ckpt["best_val"])  # backward compat

    def run_split(paths: List[Path], train: bool):
        model.train(train)
        total_loss = 0.0
        f1a = 0.0
        f1o = 0.0
        nb = 0

        for sp in paths:
            z = np.load(sp)

            X  = z["X"].astype(np.float32)     # [N,B,C]
            YA = z["YA"].astype(np.float32)    # [N,n_notes]

            # YO optional
            if (not args.disable_onset_head):
                if "YO" in z:
                    YO = z["YO"].astype(np.float32)
                else:
                    YO = YA  # onset-anchored dataset: treat onset=active
            else:
                YO = None

            # YC optional
            if (not args.disable_count_head):
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

            # batch over shard
            for start in range(0, N - args.batch + 1, args.batch):
                sel = idx[start:start + args.batch]

                xb  = torch.from_numpy(X[sel]).to(device, non_blocking=True)
                yab = torch.from_numpy(YA[sel]).to(device, non_blocking=True)

                if YO is not None:
                    yob = torch.from_numpy(YO[sel]).to(device, non_blocking=True)
                else:
                    yob = None

                if YC is not None:
                    ycb = torch.from_numpy(YC[sel]).to(device, non_blocking=True)
                else:
                    ycb = None

                if train:
                    opt.zero_grad(set_to_none=True)
                    with autocast("cuda", enabled=(device.type == "cuda")):
                        a_log, o_log, c_log = model(xb)

                        loss = args.w_active * bce_active(a_log, yab)

                        if (not args.disable_onset_head):
                            loss = loss + args.w_onset * bce_onset(o_log, yob)

                        if (not args.disable_count_head):
                            loss = loss + args.w_count * ce_count(c_log, ycb)

                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()

                else:
                    with torch.no_grad():
                        with autocast("cuda", enabled=(device.type == "cuda")):
                            a_log, o_log, c_log = model(xb)
                            loss = args.w_active * bce_active(a_log, yab)
                            if (not args.disable_onset_head):
                                loss = loss + args.w_onset * bce_onset(o_log, yob)
                            if (not args.disable_count_head):
                                loss = loss + args.w_count * ce_count(c_log, ycb)

                total_loss += float(loss.item())
                nb += 1

                if not train:
                    a = torch.sigmoid(a_log)
                    f1a += frame_f1(a, yab, thr=args.thr)

                    if (not args.disable_onset_head):
                        o = torch.sigmoid(o_log)
                        f1o += frame_f1(o, yob, thr=args.thr)

        if train:
            return total_loss / max(1, nb), None, None
        else:
            f1o_out = (f1o / max(1, nb)) if (not args.disable_onset_head) else None
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
            }, args.save)
            print(f"  [saved] {args.save} (best active F1={best_f1:.3f})")


if __name__ == "__main__":
    main()