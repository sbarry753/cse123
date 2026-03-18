"""
train.py — Train polyphonic Guitar -> Piano timbre transfer

This version:
- trains with CombinedLoss
- selects best_model.pt using a separate piano similarity score
- piano similarity is computed against validation piano targets

Usage:
  python train.py --data_dir ./data --epochs 100 --batch_size 16
"""

import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE
from dataset import make_dataloaders
from losses import CombinedLoss


def parse_args():
    p = argparse.ArgumentParser(description="Train polyphonic Guitar -> Piano model")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def smooth_rms(audio: torch.Tensor, window: int = 128) -> torch.Tensor:
    audio_sq = audio ** 2
    rms = F.avg_pool1d(
        audio_sq.unsqueeze(1),
        kernel_size=window,
        stride=max(1, window // 2),
        padding=window // 4,
    ).squeeze(1)
    return torch.sqrt(rms + 1e-8)


def compute_piano_similarity(pred: torch.Tensor, target: torch.Tensor, criterion: CombinedLoss) -> torch.Tensor:
    """
    Lower = closer to piano target.

    Uses piano-oriented features for checkpoint selection:
    - spectral timbre similarity
    - onset similarity
    - brightness similarity
    - loudness envelope similarity
    """
    # overall piano timbre similarity
    spec_dist = criterion.spectral_loss(pred, target)

    # onset / hammer-like attack similarity
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    onset_dist = F.l1_loss(pred_diff, target_diff)

    # bright attack similarity
    pred_hp = pred[:, 1:] - 0.95 * pred[:, :-1]
    target_hp = target[:, 1:] - 0.95 * target[:, :-1]
    bright_dist = F.l1_loss(pred_hp, target_hp)

    # envelope / decay similarity
    pred_rms = smooth_rms(pred)
    target_rms = smooth_rms(target)
    env_dist = F.l1_loss(pred_rms, target_rms)

    piano_score = (
        1.00 * spec_dist
        + 0.60 * onset_dist
        + 0.25 * bright_dist
        + 0.35 * env_dist
    )
    return piano_score


def train_epoch(model, loader, optimizer, criterion, device, epoch=0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Train", leave=False):
        guitar_frames = guitar_frames.to(device)
        piano_frames = piano_frames.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred, _, params = model(guitar_frames)
        loss = criterion(pred, piano_frames)

        # Mild regularization on large spectral residuals
        if isinstance(params, dict) and "residual" in params:
            loss = loss + 1e-4 * params["residual"].abs().mean()

        if torch.isnan(loss) or torch.isinf(loss):
            print("  ⚠ NaN/Inf loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    if loader is None:
        return None, None

    model.eval()
    total_loss = 0.0
    total_piano_score = 0.0
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Val  ", leave=False):
        guitar_frames = guitar_frames.to(device)
        piano_frames = piano_frames.to(device)

        pred, _, params = model(guitar_frames)

        loss = criterion(pred, piano_frames)

        if isinstance(params, dict) and "residual" in params:
            loss = loss + 1e-4 * params["residual"].abs().mean()

        piano_score = compute_piano_similarity(pred, piano_frames, criterion)

        total_loss += float(loss.item())
        total_piano_score += float(piano_score.item())
        n_batches += 1

    return (
        total_loss / max(1, n_batches),
        total_piano_score / max(1, n_batches),
    )


def save_checkpoint(model, optimizer, epoch, score_value, path, score_name="val_loss"):
    torch.save(
        {
            "epoch": epoch,
            score_name: score_value,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def plot_curves(train_losses, val_losses, piano_scores, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss", color="steelblue")
    plt.plot(val_losses, label="Val Loss", color="coral")
    plt.plot(piano_scores, label="Piano Score", color="seagreen")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Polyphonic Guitar->Piano Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close()
    print(f"  Curves saved to {output_dir}/loss_curves.png")


def export_torchscript(model, output_dir):
    model.eval()
    model.cpu()
    dummy = torch.randn(1, FRAME_SIZE)
    try:
        scripted = torch.jit.trace(model, dummy, strict=False)
        path = os.path.join(output_dir, "model_scripted.pt")
        scripted.save(path)
        print(f"  TorchScript model saved -> {path}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")
        torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pt"))


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    train_loader, val_loader = make_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
    )

    model = DDSPGuitarToPiano(
        hidden_size=args.hidden_size,
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = CombinedLoss().to(device)

    warmup_epochs = 5
    start_epoch = 0
    best_val = float("inf")
    best_piano_score = float("inf")

    train_losses = []
    val_losses = []
    piano_scores = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

        # resume gracefully whether old or new checkpoint format
        if "piano_score" in ckpt:
            best_piano_score = ckpt["piano_score"]
        elif "val_loss" in ckpt:
            best_piano_score = ckpt["val_loss"]

        if "val_loss" in ckpt:
            best_val = ckpt["val_loss"]

        print(
            f"Resumed from epoch {start_epoch}, "
            f"best val loss: {best_val:.4f}, "
            f"best piano score: {best_piano_score:.4f}"
        )

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            warmup_lr = args.lr * float(epoch + 1) / float(warmup_epochs)
            set_lr(optimizer, warmup_lr)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch)
        val_loss, piano_score = val_epoch(model, val_loader, criterion, device)

        if epoch >= warmup_epochs:
            scheduler.step()

        if val_loss is None:
            val_loss = train_loss
            piano_score = train_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        piano_scores.append(piano_score)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  "
            f"val={val_loss:.4f}  "
            f"piano_score={piano_score:.4f}  "
            f"lr={lr_now:.2e}"
        )

        # save every epoch
        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pt")
        save_checkpoint(model, optimizer, epoch, piano_score, ckpt_path, score_name="piano_score")

        # optional: best by raw validation loss too
        if val_loss < best_val:
            best_val = val_loss
            best_val_path = os.path.join(args.output_dir, "best_val_model.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, best_val_path, score_name="val_loss")
            print(f"  ✓ New best validation-loss model saved -> {best_val_path}")

        # primary best checkpoint: piano similarity
        if piano_score < best_piano_score:
            best_piano_score = piano_score
            best_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, piano_score, best_path, score_name="piano_score")
            print(f"  ✓ New best piano-like model saved -> {best_path}")

    print("\nExporting best piano-like model for real-time inference...")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(best_ckpt["model"])
    export_torchscript(model, args.output_dir)

    plot_curves(train_losses, val_losses, piano_scores, args.output_dir)

    print(f"\nDone.")
    print(f"Best validation loss : {best_val:.4f}")
    print(f"Best piano score     : {best_piano_score:.4f}")
    print(f"Checkpoints saved to : {args.output_dir}/")


if __name__ == "__main__":
    main()