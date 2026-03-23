"""Train temporal-context Guitar -> Piano timbre transfer."""

import os
import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, CONTEXT_FRAMES
from dataset import make_dataloaders
from losses import CombinedLoss


def parse_args():
    p = argparse.ArgumentParser(description="Train temporal-context Guitar -> Piano model")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./checkpoints_temporal")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--noise_std", type=float, default=1e-4)
    p.add_argument("--silence_threshold", type=float, default=0.01)
    p.add_argument("--silence_penalty_weight", type=float, default=0.05)
    p.add_argument("--residual_reg_weight", type=float, default=3e-4)
    p.add_argument("--context_frames", type=int, default=CONTEXT_FRAMES)
    p.add_argument("--no_cache", action="store_true")
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


def compute_silence_penalty(pred: torch.Tensor, target: torch.Tensor, silence_threshold: float) -> torch.Tensor:
    target_rms = torch.sqrt((target ** 2).mean(dim=1) + 1e-8)
    silent_mask = (target_rms < silence_threshold).float()
    if silent_mask.sum() <= 0:
        return pred.new_tensor(0.0)
    pred_energy = (pred ** 2).mean(dim=1)
    return (pred_energy * silent_mask).sum() / (silent_mask.sum() + 1e-8)


def train_epoch(model, loader, optimizer, criterion, device, silence_threshold=0.01, silence_penalty_weight=0.05, residual_reg_weight=3e-4):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for guitar_ctx, piano_frames in tqdm(loader, desc="Train", leave=False):
        guitar_ctx = guitar_ctx.to(device)
        piano_frames = piano_frames.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred, _, params = model(guitar_ctx)
        loss = criterion(pred, piano_frames)
        if "residual" in params:
            loss = loss + residual_reg_weight * params["residual"].abs().mean()
        silence_penalty = compute_silence_penalty(pred, piano_frames, silence_threshold)
        loss = loss + silence_penalty_weight * silence_penalty
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
def val_epoch(model, loader, criterion, device, silence_threshold=0.01, silence_penalty_weight=0.05, residual_reg_weight=3e-4):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for guitar_ctx, piano_frames in tqdm(loader, desc="Val  ", leave=False):
        guitar_ctx = guitar_ctx.to(device)
        piano_frames = piano_frames.to(device)
        pred, _, params = model(guitar_ctx)
        loss = criterion(pred, piano_frames)
        if "residual" in params:
            loss = loss + residual_reg_weight * params["residual"].abs().mean()
        silence_penalty = compute_silence_penalty(pred, piano_frames, silence_threshold)
        loss = loss + silence_penalty_weight * silence_penalty
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({"epoch": epoch, "val_loss": val_loss, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def plot_loss_curves(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train", color="steelblue")
    plt.plot(val_losses, label="Val", color="coral")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Temporal Guitar->Piano Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close()
    print(f"  Loss curve saved to {output_dir}/loss_curves.png")


def export_torchscript(model, output_dir, context_frames: int):
    model.eval()
    model.cpu()
    dummy = torch.randn(1, context_frames, FRAME_SIZE)
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
        noise_std=args.noise_std,
        context_frames=args.context_frames,
        cache=not args.no_cache,
    )

    model = DDSPGuitarToPiano(
        hidden_size=args.hidden_size,
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        n_fft=FRAME_SIZE,
        context_frames=args.context_frames,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = CombinedLoss().to(device)

    warmup_epochs = 5
    start_epoch = 0
    best_val = float("inf")
    train_losses, val_losses = [], []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["val_loss"]
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val:.4f}")

    print(f"\nTraining for {args.epochs} epochs.")
    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            warmup_lr = args.lr * float(epoch + 1) / float(warmup_epochs)
            set_lr(optimizer, warmup_lr)

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            silence_threshold=args.silence_threshold,
            silence_penalty_weight=args.silence_penalty_weight,
            residual_reg_weight=args.residual_reg_weight,
        )
        val_loss = val_epoch(
            model, val_loader, criterion, device,
            silence_threshold=args.silence_threshold,
            silence_penalty_weight=args.silence_penalty_weight,
            residual_reg_weight=args.residual_reg_weight,
        )

        if epoch >= warmup_epochs:
            scheduler.step()
        if val_loss is None:
            val_loss = train_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pt")
        save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"  ✓ New best model saved -> {best_path}")

    print("\nExporting for real-time inference.")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(best_ckpt["model"])
    export_torchscript(model, args.output_dir, args.context_frames)
    plot_loss_curves(train_losses, val_losses, args.output_dir)
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
