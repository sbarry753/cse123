#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler


def list_npz_files(root: Path) -> List[Path]:
    return sorted(root.glob("shard_*.npz"))


class CRNN(nn.Module):
    def __init__(self, bands: int, hidden: int = 128):
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
        self.head_active = nn.Linear(emb, 49)
        self.head_onset  = nn.Linear(emb, 49)
        self.head_count  = nn.Linear(emb, 7)

    def forward(self, x):           # x [N,B,C]
        h = self.conv(x)            # [N,128,C]
        h = h.transpose(1, 2)       # [N,C,128]
        out, _ = self.gru(h)        # [N,C,emb]
        last = out[:, -1, :]
        return self.head_active(last), self.head_onset(last), self.head_count(last)


@torch.no_grad()
def frame_f1(pred: torch.Tensor, targ: torch.Tensor, thr: float) -> float:
    p = (pred >= thr).int()
    t = targ.int()
    tp = (p & t).sum().item()
    fp = (p & (1 - t)).sum().item()
    fn = ((1 - p) & t).sum().item()
    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0


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
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    ap.add_argument("--shuffle_within_shard", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    bands = int(meta["bands"])

    train_paths = list_npz_files(data_dir / "train")
    val_paths   = list_npz_files(data_dir / "val")
    assert train_paths and val_paths

    # cached weights
    pos_weight = torch.from_numpy(np.load(data_dir / "pos_weight.npy")).float()
    count_w    = torch.from_numpy(np.load(data_dir / "count_weight.npy")).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = CRNN(bands=bands, hidden=args.hidden).to(device)
    start_epoch = 1
    best_val = -1.0

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "best_val" in ckpt:
            best_val = ckpt["best_val"]
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
    bce_active = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    bce_onset  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    ce_count   = nn.CrossEntropyLoss(weight=count_w.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best = -1.0

    def run_split(paths: List[Path], train: bool):
        model.train(train)
        total_loss = 0.0
        f1a = 0.0
        f1o = 0.0
        nb = 0

        for sp in paths:
            z = np.load(sp)
            X  = z["X"].astype(np.float32)   # [N,B,C]
            YA = z["YA"].astype(np.float32)  # [N,49]
            YO = z["YO"].astype(np.float32)
            YC = z["YC"].astype(np.int64)    # [N]

            N = X.shape[0]
            idx = np.arange(N)
            if train and args.shuffle_within_shard:
                np.random.shuffle(idx)

            for start in range(0, N - args.batch + 1, args.batch):
                sel = idx[start:start+args.batch]
                xb  = torch.from_numpy(X[sel]).to(device, non_blocking=True)
                yab = torch.from_numpy(YA[sel]).to(device, non_blocking=True)
                yob = torch.from_numpy(YO[sel]).to(device, non_blocking=True)
                ycb = torch.from_numpy(YC[sel]).to(device, non_blocking=True)

                if train:
                    opt.zero_grad(set_to_none=True)
                    with autocast("cuda", enabled=(device.type == "cuda")):
                        a_log, o_log, c_log = model(xb)
                        loss = (args.w_active * bce_active(a_log, yab) +
                                args.w_onset  * bce_onset(o_log, yob) +
                                args.w_count  * ce_count(c_log, ycb))
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    with torch.no_grad():
                        with autocast("cuda", enabled=(device.type == "cuda")):
                            a_log, o_log, c_log = model(xb)
                            loss = (args.w_active * bce_active(a_log, yab) +
                                    args.w_onset  * bce_onset(o_log, yob) +
                                    args.w_count  * ce_count(c_log, ycb))

                total_loss += float(loss.item())
                nb += 1

                if not train:
                    a = torch.sigmoid(a_log)
                    o = torch.sigmoid(o_log)
                    f1a += frame_f1(a, yab, thr=args.thr)
                    f1o += frame_f1(o, yob, thr=args.thr)

        if train:
            return total_loss / max(1, nb), None, None
        else:
            return total_loss / max(1, nb), f1a / max(1, nb), f1o / max(1, nb)

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, _, _ = run_split(train_paths, train=True)
        va_loss, f1a, f1o = run_split(val_paths, train=False)

        print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | F1 active {f1a:.3f} | F1 onset {f1o:.3f}")

        if f1a > best:
            best = f1a
            torch.save({
                "model": model.state_dict(),
                "meta": meta,
                "args": vars(args),
                "epoch": epoch,
                "best_val": best_val,
            }, args.save)
            print(f"  [saved] {args.save} (best active F1={best:.3f})")


if __name__ == "__main__":
    main()