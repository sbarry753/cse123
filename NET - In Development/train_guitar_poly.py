#!/usr/bin/env python3
# train_guitar_poly.py  (v2 - with polyphony count head)
#
# Three output heads:
#   active_head  -> sigmoid BCEWithLogits, shape (49,)
#   onset_head   -> sigmoid BCEWithLogits, shape (49,)
#   count_head   -> softmax CrossEntropy,  shape (7,)  classes 0..6
#
# Resume: --resume fb_model_torch/last.pt

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

MAX_POLY = 6   # must match dataset builder


def list_shards(split_dir: Path) -> List[Path]:
    return sorted([p for p in split_dir.glob("shard_*.npz") if p.is_file()])


class ShardDataset(Dataset):
    def __init__(self, shard_paths: List[Path]):
        xs, ya, yo, yc = [], [], [], []
        for p in shard_paths:
            d = np.load(p)
            xs.append(d["X"])   # (N, B, ctx)   float16
            ya.append(d["YA"])  # (N, 49)        uint8
            yo.append(d["YO"])  # (N, 49)        uint8
            # YC may be absent in old shards — default to zeros
            yc.append(d["YC"] if "YC" in d else np.zeros(d["X"].shape[0], dtype=np.int8))
        self.X  = np.concatenate(xs,  axis=0)
        self.YA = np.concatenate(ya,  axis=0)
        self.YO = np.concatenate(yo,  axis=0)
        self.YC = np.concatenate(yc,  axis=0)
        assert self.X.ndim == 3

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x  = self.X[idx].astype(np.float32)    # (B, ctx)
        ya = self.YA[idx].astype(np.float32)   # (49,)
        yo = self.YO[idx].astype(np.float32)   # (49,)
        yc = int(self.YC[idx])                 # scalar 0-6
        return (
            torch.from_numpy(x),
            torch.from_numpy(ya),
            torch.from_numpy(yo),
            torch.tensor(yc, dtype=torch.long),
        )


# ─────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad  = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=0,
            bias=False,
        )
        self.bn  = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.act(self.bn(self.conv(x)))


class GuitarPolyTCN(nn.Module):
    """
    Causal TCN with three heads:
      active_head  - per-note presence (49 logits)
      onset_head   - per-note onset    (49 logits)
      count_head   - polyphony count   (MAX_POLY+1 logits, i.e. 7 classes)
    """
    def __init__(self, bands: int, notes: int = 49, channels: int = 64, n_count_classes: int = MAX_POLY + 1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(bands,    channels, kernel_size=3, dilation=1),
            CausalConv1d(channels, channels, kernel_size=3, dilation=2),
            CausalConv1d(channels, channels, kernel_size=3, dilation=4),
            CausalConv1d(channels, channels, kernel_size=3, dilation=8),
            CausalConv1d(channels, channels, kernel_size=3, dilation=16),
        )
        self.active_head = nn.Linear(channels, notes)
        self.onset_head  = nn.Linear(channels, notes)
        # Count head — slightly wider to capture full polyphony range
        self.count_head  = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, n_count_classes),
        )

    def forward(self, x: torch.Tensor):
        """x: (N, B, T) -> active_logits(N,49), onset_logits(N,49), count_logits(N,7)"""
        h      = self.net(x)          # (N, C, T)
        h_last = h[:, :, -1]          # (N, C)
        return self.active_head(h_last), self.onset_head(h_last), self.count_head(h_last)


# ─────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────

@torch.no_grad()
def compute_pos_weight_active(ds: ShardDataset) -> torch.Tensor:
    YA  = ds.YA.astype(np.float64)
    pos = YA.sum(axis=0) + 1e-6
    neg = YA.shape[0] - pos + 1e-6
    return torch.tensor(neg / pos, dtype=torch.float32)


@torch.no_grad()
def compute_count_class_weights(ds: ShardDataset, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weighting for polyphony count classes (handles skew toward 0)."""
    counts = np.bincount(ds.YC.astype(np.int64).clip(0, n_classes - 1), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w      = 1.0 / counts
    w      = w / w.mean()          # normalise so average weight ≈ 1
    return torch.tensor(w, dtype=torch.float32)


# ─────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optim,
    loss_active, loss_onset, loss_count,
    device, onset_weight: float, count_weight: float,
) -> float:
    model.train()
    total, n = 0.0, 0
    for x, ya, yo, yc in loader:
        x  = x.to(device)
        ya = ya.to(device)
        yo = yo.to(device)
        yc = yc.to(device)
        optim.zero_grad(set_to_none=True)
        a_logits, o_logits, c_logits = model(x)
        la   = loss_active(a_logits, ya)
        lo   = loss_onset(o_logits,  yo)
        lc   = loss_count(c_logits,  yc)
        loss = la + onset_weight * lo + count_weight * lc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        bs     = x.shape[0]
        total += float(loss.item()) * bs
        n     += bs
    return total / max(1, n)


@torch.no_grad()
def eval_one_epoch(
    model, loader,
    loss_active, loss_onset, loss_count,
    device, onset_weight: float, count_weight: float,
) -> Tuple[float, float]:
    """Returns (total_loss, count_accuracy)."""
    model.eval()
    total, n         = 0.0, 0
    correct, n_count = 0,   0
    for x, ya, yo, yc in loader:
        x  = x.to(device)
        ya = ya.to(device)
        yo = yo.to(device)
        yc = yc.to(device)
        a_logits, o_logits, c_logits = model(x)
        la    = loss_active(a_logits, ya)
        lo    = loss_onset(o_logits,  yo)
        lc    = loss_count(c_logits,  yc)
        loss  = la + onset_weight * lo + count_weight * lc
        bs    = x.shape[0]
        total += float(loss.item()) * bs
        n     += bs
        preds    = c_logits.argmax(dim=1)
        correct  += int((preds == yc).sum().item())
        n_count  += bs
    return total / max(1, n), correct / max(1, n_count)


# ─────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────

def save_ckpt(path: Path, *, model, optim, epoch, best_val, bands, ctx, channels, meta):
    torch.save(
        {
            "epoch":       int(epoch),
            "best_val":    float(best_val),
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "bands":       int(bands),
            "ctx":         int(ctx),
            "channels":    int(channels),
            "meta":        meta,
        },
        path,
    )


# ─────────────────────────────────────────────────
# Inference helpers (for use in your real-time code)
# ─────────────────────────────────────────────────

@torch.no_grad()
def predict_frame(
    model:        nn.Module,
    feat_window:  np.ndarray,   # (bands, ctx_frames) float32, already z-scored
    device:       torch.device,
    active_thresh: float = 0.5,
    onset_thresh:  float = 0.5,
) -> dict:
    """
    Run one inference frame.
    Returns:
      active_notes : list of MIDI numbers that are sounding
      onset_notes  : list of MIDI numbers with detected onset this frame
      count        : predicted polyphony count (int 0-6)
      active_probs : np.ndarray (49,) sigmoid probabilities
    """
    x = torch.from_numpy(feat_window[None]).to(device)   # (1, B, T)
    a_logits, o_logits, c_logits = model(x)
    a_probs = torch.sigmoid(a_logits)[0].cpu().numpy()   # (49,)
    o_probs = torch.sigmoid(o_logits)[0].cpu().numpy()   # (49,)
    count   = int(c_logits[0].argmax().item())            # 0-6
    return {
        "active_notes": [i + 40 for i, p in enumerate(a_probs) if p >= active_thresh],
        "onset_notes":  [i + 40 for i, p in enumerate(o_probs) if p >= onset_thresh],
        "count":        count,
        "active_probs": a_probs,
    }


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir",   required=True)
    ap.add_argument("--out_dir",       default="fb_model_torch")
    ap.add_argument("--epochs",        type=int,   default=25)
    ap.add_argument("--batch",         type=int,   default=512)
    ap.add_argument("--lr",            type=float, default=2e-3)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--channels",      type=int,   default=64)
    ap.add_argument("--onset_weight",  type=float, default=0.5,
                    help="Loss weight for onset BCE head.")
    ap.add_argument("--count_weight",  type=float, default=0.4,
                    help="Loss weight for polyphony count CE head.")
    ap.add_argument("--num_workers",   type=int,   default=2)
    ap.add_argument("--export_onnx",   action="store_true")
    ap.add_argument("--resume",        type=str,   default=None)
    args = ap.parse_args()

    ds_dir      = Path(args.dataset_dir)
    meta        = json.loads((ds_dir / "metadata.json").read_text(encoding="utf-8"))
    train_shards = list_shards(ds_dir / "train")
    val_shards   = list_shards(ds_dir / "val")
    if not train_shards or not val_shards:
        raise RuntimeError("No shards found. Run guitarset_make_fb_dataset.py first.")

    train_ds = ShardDataset(train_shards)
    val_ds   = ShardDataset(val_shards)

    bands  = int(train_ds.X.shape[1])
    ctx    = int(train_ds.X.shape[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  bands={bands}  ctx={ctx}")

    # Count class distribution (useful sanity check)
    count_dist = np.bincount(train_ds.YC.astype(np.int64).clip(0, MAX_POLY), minlength=MAX_POLY + 1)
    print(f"Train count distribution (0-{MAX_POLY}): {count_dist.tolist()}")

    n_count_classes = MAX_POLY + 1   # 7
    model  = GuitarPolyTCN(bands=bands, notes=49, channels=args.channels,
                            n_count_classes=n_count_classes).to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Losses ──────────────────────────────────
    pos_w        = compute_pos_weight_active(train_ds).to(device)
    loss_active  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    loss_onset   = nn.BCEWithLogitsLoss(pos_weight=torch.sqrt(pos_w))

    # Weighted CE for count: high-poly frames are rare — up-weight them
    count_w      = compute_count_class_weights(train_ds, n_classes=n_count_classes).to(device)
    loss_count   = nn.CrossEntropyLoss(weight=count_w)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    start_epoch = 1
    best_val    = float("inf")

    # ── Resume ──────────────────────────────────
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state"], strict=False)  # strict=False handles new count head
        if "optim_state" in ck:
            try:
                optim.load_state_dict(ck["optim_state"])
            except Exception:
                print("  [warn] optimizer state mismatch (new head added?) — starting fresh optimizer")
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_val    = float(ck.get("best_val", best_val))
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val={best_val:.4f})")

    # ── Optional ONNX export ────────────────────
    if args.export_onnx:
        print("Exporting ONNX...")
        model.eval()
        dummy     = torch.zeros((1, bands, ctx), dtype=torch.float32, device=device)
        onnx_path = out_dir / "model.onnx"
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["fb_feat"],
            output_names=["active_logits", "onset_logits", "count_logits"],
            opset_version=17,
            dynamic_axes={
                "fb_feat":       {0: "batch"},
                "active_logits": {0: "batch"},
                "onset_logits":  {0: "batch"},
                "count_logits":  {0: "batch"},
            },
        )
        print(f"ONNX exported -> {onnx_path}")

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    for epoch in range(start_epoch, args.epochs + 1):
        tr       = train_one_epoch(
            model, train_loader, optim,
            loss_active, loss_onset, loss_count,
            device, args.onset_weight, args.count_weight,
        )
        va, cnt_acc = eval_one_epoch(
            model, val_loader,
            loss_active, loss_onset, loss_count,
            device, args.onset_weight, args.count_weight,
        )
        print(f"epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | count_acc {cnt_acc:.3f}")

        save_ckpt(last_path, model=model, optim=optim, epoch=epoch,
                  best_val=best_val, bands=bands, ctx=ctx, channels=args.channels, meta=meta)

        if va < best_val:
            best_val = va
            save_ckpt(best_path, model=model, optim=optim, epoch=epoch,
                      best_val=best_val, bands=bands, ctx=ctx, channels=args.channels, meta=meta)
            print(f"  ✓ saved best (val={best_val:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()