"""
train.py — DDSP Guitar-to-Piano Training Script

Usage:
  python train.py --data_dir ./data --epochs 100 --batch_size 64

Your data/ folder should look like:
  data/
    guitar/  ← raw guitar WAVs
    piano/   ← matched piano WAVs (same filenames)
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from model   import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE
from dataset import make_dataloaders
from losses  import CombinedLoss


def parse_args():
    p = argparse.ArgumentParser(description='Train DDSP Guitar → Piano')
    p.add_argument('--data_dir',    type=str,   default='./data',      help='Path to data/ folder')
    p.add_argument('--output_dir',  type=str,   default='./checkpoints')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--hidden_size', type=int,   default=512)
    p.add_argument('--n_harmonics', type=int,   default=64)
    p.add_argument('--resume',      type=str,   default=None,          help='Resume from checkpoint path')
    p.add_argument('--device',      type=str,   default='auto')
    return p.parse_args()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_device(preference: str) -> torch.device:
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')        # Apple Silicon
        else:
            return torch.device('cpu')
    return torch.device(preference)


def train_epoch(model, loader, optimizer, criterion, device, epoch=0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc='Train', leave=False):
        guitar_frames = guitar_frames.to(device)
        piano_frames  = piano_frames.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred, _, params = model(guitar_frames)

        spec_loss = criterion(pred, piano_frames)
        wave_loss = torch.nn.functional.l1_loss(pred, piano_frames)

        # Encourage quieter noise branch early in training
        noise_penalty = params['noise_mags'].mean()

        # Stronger penalty early, weaker later
        noise_weight = max(0.002, 0.02 * (0.98 ** epoch))

        loss = spec_loss + 0.25 * wave_loss + noise_weight * noise_penalty

        if torch.isnan(loss) or torch.isinf(loss):
            print("  ⚠ NaN/Inf loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    if loader is None:
        return None

    model.eval()
    total_loss = 0.0
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc='Val  ', leave=False):
        guitar_frames = guitar_frames.to(device)
        piano_frames  = piano_frames.to(device)

        pred, _, params = model(guitar_frames)

        spec_loss = criterion(pred, piano_frames)
        wave_loss = torch.nn.functional.l1_loss(pred, piano_frames)
        noise_penalty = params['noise_mags'].mean()

        loss = spec_loss + 0.25 * wave_loss + 0.002 * noise_penalty

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch':      epoch,
        'val_loss':   val_loss,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
    }, path)


def plot_loss_curves(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train', color='steelblue')
    plt.plot(val_losses,   label='Val',   color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDSP Guitar→Piano Training')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    print(f"  Loss curve saved to {output_dir}/loss_curves.png")


def export_torchscript(model, output_dir, device):
    """Export model to TorchScript for fast real-time inference."""
    model.eval()
    model.cpu()
    dummy = torch.randn(1, FRAME_SIZE)
    try:
        scripted = torch.jit.trace(model, dummy, strict=False)
        path = os.path.join(output_dir, 'model_scripted.pt')
        scripted.save(path)
        print(f"  TorchScript model saved → {path}")
    except Exception as e:
        print(f"  TorchScript export failed (will use state dict): {e}")
        # Fallback: just save weights
        torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pt'))


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────
    print("Loading dataset...")
    train_loader, val_loader = make_dataloaders(args.data_dir, batch_size=args.batch_size)

    # ── Model ─────────────────────────────────
    model = DDSPGuitarToPiano(
        hidden_size = args.hidden_size,
        n_harmonics = args.n_harmonics,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optim / Loss ──────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = CombinedLoss().to(device)
    warmup_epochs = 5

    start_epoch = 0
    best_val    = float('inf')
    train_losses, val_losses = [], []

    # ── Resume ────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt['val_loss']
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val:.4f}")

    # ── Training loop ─────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            warmup_lr = args.lr * float(epoch + 1) / float(warmup_epochs)
            set_lr(optimizer, warmup_lr)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch)
        val_loss   = val_epoch(model, val_loader, criterion, device)

        if epoch >= warmup_epochs:
            scheduler.step()

        if val_loss is None:
            val_loss = train_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
        )

        ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch+1:04d}.pt')
        save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"  ✓ New best model saved → {best_path}")

    # ── Export for real-time use ───────────────
    print("\nExporting for real-time inference...")
    best_ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'), map_location='cpu')
    model.load_state_dict(best_ckpt['model'])
    export_torchscript(model, args.output_dir, device)

    plot_loss_curves(train_losses, val_losses, args.output_dir)
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()