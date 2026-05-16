"""
Train polyphonic Guitar -> Piano timbre transfer

Usage:
  python train.py --data_dir ./data --epochs 100 --batch_size 16
"""

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE, N_FFT
from dataset import GuitarPianoDataset, load_split_manifest
from losses import CombinedLoss


def parse_args():
    p = argparse.ArgumentParser(description="Train polyphonic Guitar -> Piano model")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--split_manifest", type=str, default=None)
    p.add_argument("--eval_split", choices=["val", "test"], default="val")
    p.add_argument("--n_fft", type=int, default=N_FFT)
    p.add_argument("--win_length", type=int, default=FRAME_SIZE)
    p.add_argument("--hop_size", type=int, default=HOP_SIZE)
    p.add_argument("--frame_size", type=int, default=FRAME_SIZE)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--phase_tcn_ch", type=int, default=16)
    p.add_argument("--phase_tcn_layers", type=int, default=3)
    p.add_argument("--phase_max_delta", type=float, default=0.5)
    p.add_argument("--spectral_weight", type=float, default=1.0)
    p.add_argument("--waveform_weight", type=float, default=0.25)
    p.add_argument("--envelope_weight", type=float, default=0.2)
    p.add_argument("--onset_weight", type=float, default=0.75)
    p.add_argument("--spectral_convergence_weight", type=float, default=0.25)
    p.add_argument("--log_stft_weight", type=float, default=0.25)
    p.add_argument("--plain_log_stft_weight", type=float, default=0.1)
    p.add_argument("--hf_artifact_weight", type=float, default=0.05)
    p.add_argument("--hf_artifact_start_hz", type=float, default=8000.0)
    p.add_argument("--hf_artifact_margin", type=float, default=0.0)
    p.add_argument("--hf_artifact_topk_frac", type=float, default=0.25)
    p.add_argument("--energy_weight_floor", type=float, default=0.1)
    p.add_argument("--energy_weight_ceiling", type=float, default=5.0)
    p.add_argument("--intended_log_mag_weight", type=float, default=0.1)
    p.add_argument("--mask_reg_weight", type=float, default=0.001)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()
    args.win_length = int(args.win_length or args.n_fft)
    return args


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


def new_loss_totals():
    keys = [
        "total",
        "spectral",
        "waveform",
        "envelope",
        "onset",
        "spectral_mel",
        "spectral_convergence",
        "spectral_log_stft",
        "spectral_plain_log_stft",
        "spectral_hf_artifact",
        "weighted_spectral_mel",
        "weighted_spectral_convergence",
        "weighted_spectral_log_stft",
        "weighted_spectral_plain_log_stft",
        "weighted_spectral_hf_artifact",
        "weighted_spectral",
        "weighted_waveform",
        "weighted_envelope",
        "weighted_onset",
        "intended_log_mag",
        "weighted_intended_log_mag",
        "mask_reg",
        "weighted_mask_reg",
        "residual_reg",
    ]
    return {key: 0.0 for key in keys}


def average_loss_totals(totals, n_batches):
    denom = max(1, n_batches)
    return {key: value / denom for key, value in totals.items()}


def format_loss_components(prefix, metrics):
    return (
        f"{prefix}: total={metrics['total']:.4f} "
        f"spec={metrics['weighted_spectral']:.4f}({metrics['spectral']:.4f}) "
        f"wave={metrics['weighted_waveform']:.4f}({metrics['waveform']:.4f}) "
        f"env={metrics['weighted_envelope']:.4f}({metrics['envelope']:.4f}) "
        f"onset={metrics['weighted_onset']:.4f}({metrics['onset']:.4f}) "
        f"mel={metrics['weighted_spectral_mel']:.4f}({metrics['spectral_mel']:.4f}) "
        f"sc={metrics['weighted_spectral_convergence']:.4f}({metrics['spectral_convergence']:.4f}) "
        f"log_stft={metrics['weighted_spectral_log_stft']:.4f}({metrics['spectral_log_stft']:.4f}) "
        f"plain_log={metrics['weighted_spectral_plain_log_stft']:.4f}({metrics['spectral_plain_log_stft']:.4f}) "
        f"hf_art={metrics['weighted_spectral_hf_artifact']:.4f}({metrics['spectral_hf_artifact']:.4f}) "
        f"intended_log={metrics['weighted_intended_log_mag']:.4f}({metrics['intended_log_mag']:.4f}) "
        f"mask_reg={metrics['weighted_mask_reg']:.6f}({metrics['mask_reg']:.4f}) "
        f"reg={metrics['residual_reg']:.6f}"
    )


def intended_log_mag_loss(model, features, params, piano_frames):
    input_log_mag = features["input_log_mag"]
    intended_log_mag = input_log_mag * params["mask"] + params["residual"]
    piano_spec = model._stft(piano_frames)
    piano_mag = torch.abs(piano_spec)
    piano_log_mag = torch.log(torch.clamp(piano_mag, min=1.0e-5))
    return F.l1_loss(intended_log_mag, piano_log_mag)


def train_epoch(model, loader, optimizer, criterion, device, args, epoch=0):
    model.train()
    totals = new_loss_totals()
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Train", leave=False):
        guitar_frames = guitar_frames.to(device)
        piano_frames = piano_frames.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred, features, params = model(guitar_frames)
        loss_components = criterion.components(pred, piano_frames)
        
        loss = loss_components["total"]
        log_mag_loss = intended_log_mag_loss(model, features, params, piano_frames)
        weighted_log_mag_loss = args.intended_log_mag_weight * log_mag_loss

        loss = loss + weighted_log_mag_loss

        mask_reg = (params["mask"] - 1.0).abs().mean()
        weighted_mask_reg = args.mask_reg_weight * mask_reg
        loss = loss + weighted_mask_reg

        residual_reg = loss.new_tensor(0.0)

        # Mild regularization on very large residual outputs
        if "residual" in params:
            residual_reg = 1e-4 * params["residual"].abs().mean()
            loss = loss + residual_reg

        if torch.isnan(loss) or torch.isinf(loss):
            print("  ⚠ NaN/Inf loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        totals["total"] += float(loss.item())
        totals["intended_log_mag"] += float(log_mag_loss.item())
        totals["weighted_intended_log_mag"] += float(weighted_log_mag_loss.item())
        totals["mask_reg"] += float(mask_reg.item())
        totals["weighted_mask_reg"] += float(weighted_mask_reg.item())
        totals["residual_reg"] += float(residual_reg.item())
        for key, value in loss_components.items():
            if key != "total":
                totals[key] += float(value.item())
        n_batches += 1

    return average_loss_totals(totals, n_batches)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, args):
    if loader is None:
        return None

    model.eval()
    totals = new_loss_totals()
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Val  ", leave=False):
        guitar_frames = guitar_frames.to(device)
        piano_frames = piano_frames.to(device)

        pred, features, params = model(guitar_frames)
        loss_components = criterion.components(pred, piano_frames)
        loss = loss_components["total"]
        log_mag_loss = intended_log_mag_loss(model, features, params, piano_frames)
        weighted_log_mag_loss = args.intended_log_mag_weight * log_mag_loss
        loss = loss + weighted_log_mag_loss
        mask_reg = (params["mask"] - 1.0).abs().mean()
        weighted_mask_reg = args.mask_reg_weight * mask_reg
        loss = loss + weighted_mask_reg
        residual_reg = loss.new_tensor(0.0)

        if "residual" in params:
            residual_reg = 1e-4 * params["residual"].abs().mean()
            loss = loss + residual_reg

        totals["total"] += float(loss.item())
        totals["intended_log_mag"] += float(log_mag_loss.item())
        totals["weighted_intended_log_mag"] += float(weighted_log_mag_loss.item())
        totals["mask_reg"] += float(mask_reg.item())
        totals["weighted_mask_reg"] += float(weighted_mask_reg.item())
        totals["residual_reg"] += float(residual_reg.item())
        for key, value in loss_components.items():
            if key != "total":
                totals[key] += float(value.item())
        n_batches += 1

    return average_loss_totals(totals, n_batches)


def checkpoint_state(payload, model=None):
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    state = {key: value for key, value in state.items() if key != "window"}
    if model is None:
        return state

    current = model.state_dict()
    compatible = {}
    for key, value in state.items():
        if key in current and current[key].shape != value.shape:
            print(
                f"Ignoring checkpoint tensor with incompatible shape: "
                f"{key} checkpoint={tuple(value.shape)} model={tuple(current[key].shape)}"
            )
            continue
        compatible[key] = value
    return compatible


def save_checkpoint(model, optimizer, epoch, val_loss, path, args):
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "frame_size": args.frame_size,
            "hop_size": args.hop_size,
            "n_fft": args.n_fft,
            "win_length": args.win_length,
            "base_ch": args.base_ch,
            "phase_tcn_ch": args.phase_tcn_ch,
            "phase_tcn_layers": args.phase_tcn_layers,
            "phase_max_delta": args.phase_max_delta,
            "spectral_weight": args.spectral_weight,
            "waveform_weight": args.waveform_weight,
            "envelope_weight": args.envelope_weight,
            "onset_weight": args.onset_weight,
            "spectral_convergence_weight": args.spectral_convergence_weight,
            "log_stft_weight": args.log_stft_weight,
            "plain_log_stft_weight": args.plain_log_stft_weight,
            "hf_artifact_weight": args.hf_artifact_weight,
            "intended_log_mag_weight": args.intended_log_mag_weight,
            "mask_reg_weight": args.mask_reg_weight,
        },
        path,
    )


def plot_loss_curves(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train", color="steelblue")
    plt.plot(val_losses, label="Val", color="coral")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Polyphonic Guitar->Piano Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close()
    print(f"  Loss curve saved to {output_dir}/loss_curves.png")


def export_torchscript(model, output_dir, frame_size):
    model.eval()
    model.cpu()
    dummy = torch.randn(1, frame_size)
    try:
        scripted = torch.jit.trace(model, dummy, strict=False)
        path = os.path.join(output_dir, "model_scripted.pt")
        scripted.save(path)
        print(f"  TorchScript model saved -> {path}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")
        torch.save(model.state_dict(), os.path.join(output_dir, "model_weights.pt"))

def make_loader(
    data_dir: str,
    stems: list[str],
    batch_size: int,
    frame_size: int,
    hop_size: int,
    augment: bool,
    shuffle: bool,
):
    dataset = GuitarPianoDataset(
        data_dir=data_dir,
        stems=stems,
        sample_rate=SAMPLE_RATE,
        frame_size=frame_size,
        hop_size=hop_size,
        augment=augment,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=shuffle and len(dataset) >= batch_size,
    )

def make_dataloaders(args):
    splits = load_split_manifest(args.data_dir, args.split_manifest)
    if splits is None:
        raise ValueError(
            "Create a split manifest first with data_splits.py, or pass --split_manifest."
        )
    eval_stems = splits[args.eval_split]
    if not eval_stems:
        raise ValueError(f"Split manifest has no stems for eval split: {args.eval_split}")

    train_loader = make_loader(
        args.data_dir,
        splits["train"],
        args.batch_size,
        args.frame_size,
        args.hop_size,
        augment=False,
        shuffle=True,
    )
    eval_loader = make_loader(
        args.data_dir,
        eval_stems,
        args.batch_size * 2,
        args.frame_size,
        args.hop_size,
        augment=False,
        shuffle=False,
    )
    return train_loader, eval_loader


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    train_loader, val_loader = make_dataloaders(
        args
    )

    model = DDSPGuitarToPiano(
        hidden_size=args.hidden_size,
        sample_rate=SAMPLE_RATE,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        win_length=args.win_length,
        base_ch=args.base_ch,
        phase_tcn_ch=args.phase_tcn_ch,
        phase_tcn_layers=args.phase_tcn_layers,
        phase_max_delta=args.phase_max_delta,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = CombinedLoss(
        spectral_weight=args.spectral_weight,
        waveform_weight=args.waveform_weight,
        envelope_weight=args.envelope_weight,
        onset_weight=args.onset_weight,
        spectral_convergence_weight=args.spectral_convergence_weight,
        log_stft_weight=args.log_stft_weight,
        plain_log_stft_weight=args.plain_log_stft_weight,
        hf_artifact_weight=args.hf_artifact_weight,
        hf_artifact_start_hz=args.hf_artifact_start_hz,
        hf_artifact_margin=args.hf_artifact_margin,
        hf_artifact_topk_frac=args.hf_artifact_topk_frac,
        energy_weight_floor=args.energy_weight_floor,
        energy_weight_ceiling=args.energy_weight_ceiling,
    ).to(device)

    warmup_epochs = 5
    start_epoch = 0
    best_val = float("inf")
    train_losses, val_losses = [], []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        load_result = model.load_state_dict(checkpoint_state(ckpt, model), strict=False)
        if load_result.missing_keys:
            print(f"Resume missing model keys initialized from defaults: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Resume ignored unexpected model keys: {load_result.unexpected_keys}")
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["val_loss"]
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val:.4f}")

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            warmup_lr = args.lr * float(epoch + 1) / float(warmup_epochs)
            set_lr(optimizer, warmup_lr)

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, args, epoch=epoch)
        val_metrics = val_epoch(model, val_loader, criterion, device, args)

        if epoch >= warmup_epochs:
            scheduler.step()

        if val_metrics is None:
            val_metrics = train_metrics

        train_loss = train_metrics["total"]
        val_loss = val_metrics["total"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
        )
        print("  " + format_loss_components("train", train_metrics))
        print()
        print("  " + format_loss_components("val", val_metrics))

        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pt")
        save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path, args)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_loss, best_path, args)
            print(f"   New best model saved -> {best_path}")
        print()

    print("\nExporting for real-time inference...")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location="cpu")
    model.load_state_dict(checkpoint_state(best_ckpt, model), strict=False)
    export_torchscript(model, args.output_dir, args.frame_size)

    plot_loss_curves(train_losses, val_losses, args.output_dir)
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
