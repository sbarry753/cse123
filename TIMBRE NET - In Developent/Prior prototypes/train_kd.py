"""
Teacher + student training for polyphonic Guitar -> Piano timbre transfer.

Two modes:

1) Train a 4096-sample teacher
   python train_kd.py --mode teacher --data_dir ./data --output_dir ./teacher_4096

2) Distill a 512-sample student from that teacher
   python train_kd.py --mode distill --data_dir ./data --output_dir ./student_512_kd \
       --teacher_ckpt ./teacher_4096/best_model.pt
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import DDSPGuitarToPiano, SAMPLE_RATE, HOP_SIZE
from dataset import make_dataloaders, make_distill_dataloaders
from losses import CombinedLoss


DEFAULT_TEACHER_FRAME_SIZE = 4096
DEFAULT_STUDENT_FRAME_SIZE = 512


def parse_args():
    p = argparse.ArgumentParser(description="Train teacher/student Guitar -> Piano models")
    p.add_argument("--mode", type=str, choices=["teacher", "distill"], required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./checkpoints_kd")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--teacher_ckpt", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--teacher_frame_size", type=int, default=DEFAULT_TEACHER_FRAME_SIZE)
    p.add_argument("--student_frame_size", type=int, default=DEFAULT_STUDENT_FRAME_SIZE)
    p.add_argument("--hop_size", type=int, default=HOP_SIZE)

    p.add_argument("--noise_std", type=float, default=1e-4)
    p.add_argument("--silence_threshold", type=float, default=0.01)
    p.add_argument("--silence_penalty_weight", type=float, default=0.05)
    p.add_argument("--residual_reg_weight", type=float, default=3e-4)

    p.add_argument("--distill_alpha", type=float, default=0.7, help="ground-truth loss weight")
    p.add_argument("--distill_beta", type=float, default=0.3, help="teacher loss weight")
    p.add_argument("--seed", type=int, default=22)
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


def compute_silence_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    silence_threshold: float,
) -> torch.Tensor:
    target_rms = torch.sqrt((target ** 2).mean(dim=1) + 1e-8)
    silent_mask = (target_rms < silence_threshold).float()

    if silent_mask.sum() <= 0:
        return pred.new_tensor(0.0)

    pred_energy = (pred ** 2).mean(dim=1)
    penalty = (pred_energy * silent_mask).sum() / (silent_mask.sum() + 1e-8)
    return penalty


def save_checkpoint(model, optimizer, epoch, val_loss, path, extra: dict | None = None):
    payload = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def plot_loss_curves(train_losses, val_losses, output_dir, title: str):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train", color="steelblue")
    plt.plot(val_losses, label="Val", color="coral")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close()
    print(f"  Loss curve saved to {output_dir}/loss_curves.png")


def export_torchscript(model, output_dir, frame_size: int, filename: str = "model_scripted.pt"):
    model.eval()
    model.cpu()
    dummy = torch.randn(1, frame_size)
    try:
        scripted = torch.jit.trace(model, dummy, strict=False)
        path = os.path.join(output_dir, filename)
        scripted.save(path)
        print(f"  TorchScript model saved -> {path}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")
        torch.save(model.state_dict(), os.path.join(output_dir, filename.replace(".pt", "_weights.pt")))


def build_model(frame_size: int, hop_size: int, hidden_size: int) -> DDSPGuitarToPiano:
    return DDSPGuitarToPiano(
        hidden_size=hidden_size,
        sample_rate=SAMPLE_RATE,
        frame_size=frame_size,
        hop_size=hop_size,
        n_fft=frame_size,
    )


def load_state_dict_flexible(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        model.load_state_dict(payload["model"])
        return payload
    model.load_state_dict(payload)
    return payload


def train_teacher_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    silence_threshold=0.01,
    silence_penalty_weight=0.05,
    residual_reg_weight=3e-4,
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Train", leave=False):
        guitar_frames = guitar_frames.to(device, non_blocking=True)
        piano_frames = piano_frames.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred, _, params = model(guitar_frames)
        loss = criterion(pred, piano_frames)

        if "residual" in params:
            loss = loss + residual_reg_weight * params["residual"].abs().mean()

        silence_penalty = compute_silence_penalty(
            pred,
            piano_frames,
            silence_threshold=silence_threshold,
        )
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
def val_teacher_epoch(
    model,
    loader,
    criterion,
    device,
    silence_threshold=0.01,
    silence_penalty_weight=0.05,
    residual_reg_weight=3e-4,
):
    if loader is None:
        return None

    model.eval()
    total_loss = 0.0
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Val  ", leave=False):
        guitar_frames = guitar_frames.to(device, non_blocking=True)
        piano_frames = piano_frames.to(device, non_blocking=True)

        pred, _, params = model(guitar_frames)
        loss = criterion(pred, piano_frames)

        if "residual" in params:
            loss = loss + residual_reg_weight * params["residual"].abs().mean()

        silence_penalty = compute_silence_penalty(
            pred,
            piano_frames,
            silence_threshold=silence_threshold,
        )
        loss = loss + silence_penalty_weight * silence_penalty

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def center_crop_batch(x: torch.Tensor, crop_size: int) -> torch.Tensor:
    if x.shape[-1] < crop_size:
        raise ValueError(f"Cannot crop size {crop_size} from tensor with length {x.shape[-1]}")
    start = (x.shape[-1] - crop_size) // 2
    end = start + crop_size
    return x[..., start:end]


def train_distill_epoch(
    teacher,
    student,
    loader,
    optimizer,
    criterion,
    device,
    alpha=0.7,
    beta=0.3,
    silence_threshold=0.01,
    silence_penalty_weight=0.05,
    residual_reg_weight=3e-4,
):
    teacher.eval()
    student.train()

    total_loss = 0.0
    total_gt = 0.0
    total_kd = 0.0
    n_batches = 0

    for guitar_long, guitar_short, piano_short in tqdm(loader, desc="Train", leave=False):
        guitar_long = guitar_long.to(device, non_blocking=True)
        guitar_short = guitar_short.to(device, non_blocking=True)
        piano_short = piano_short.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_pred_long, _, _ = teacher(guitar_long)
            teacher_short = center_crop_batch(teacher_pred_long, guitar_short.shape[-1])

        student_pred, _, student_params = student(guitar_short)

        loss_gt = criterion(student_pred, piano_short)
        loss_kd = criterion(student_pred, teacher_short)
        loss = alpha * loss_gt + beta * loss_kd

        if "residual" in student_params:
            loss = loss + residual_reg_weight * student_params["residual"].abs().mean()

        silence_penalty = compute_silence_penalty(
            student_pred,
            piano_short,
            silence_threshold=silence_threshold,
        )
        loss = loss + silence_penalty_weight * silence_penalty

        if torch.isnan(loss) or torch.isinf(loss):
            print("  ⚠ NaN/Inf loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_gt += float(loss_gt.item())
        total_kd += float(loss_kd.item())
        n_batches += 1

    mean_loss = total_loss / max(1, n_batches)
    mean_gt = total_gt / max(1, n_batches)
    mean_kd = total_kd / max(1, n_batches)
    return mean_loss, mean_gt, mean_kd


@torch.no_grad()
def val_distill_epoch(
    teacher,
    student,
    loader,
    criterion,
    device,
    alpha=0.7,
    beta=0.3,
    silence_threshold=0.01,
    silence_penalty_weight=0.05,
    residual_reg_weight=3e-4,
):
    if loader is None:
        return None, None, None

    teacher.eval()
    student.eval()

    total_loss = 0.0
    total_gt = 0.0
    total_kd = 0.0
    n_batches = 0

    for guitar_long, guitar_short, piano_short in tqdm(loader, desc="Val  ", leave=False):
        guitar_long = guitar_long.to(device, non_blocking=True)
        guitar_short = guitar_short.to(device, non_blocking=True)
        piano_short = piano_short.to(device, non_blocking=True)

        teacher_pred_long, _, _ = teacher(guitar_long)
        teacher_short = center_crop_batch(teacher_pred_long, guitar_short.shape[-1])

        student_pred, _, student_params = student(guitar_short)

        loss_gt = criterion(student_pred, piano_short)
        loss_kd = criterion(student_pred, teacher_short)
        loss = alpha * loss_gt + beta * loss_kd

        if "residual" in student_params:
            loss = loss + residual_reg_weight * student_params["residual"].abs().mean()

        silence_penalty = compute_silence_penalty(
            student_pred,
            piano_short,
            silence_threshold=silence_threshold,
        )
        loss = loss + silence_penalty_weight * silence_penalty

        total_loss += float(loss.item())
        total_gt += float(loss_gt.item())
        total_kd += float(loss_kd.item())
        n_batches += 1

    mean_loss = total_loss / max(1, n_batches)
    mean_gt = total_gt / max(1, n_batches)
    mean_kd = total_kd / max(1, n_batches)
    return mean_loss, mean_gt, mean_kd


def run_teacher_training(args, device):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading teacher dataset...")
    train_loader, val_loader = make_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        sample_rate=SAMPLE_RATE,
        frame_size=args.teacher_frame_size,
        hop_size=args.hop_size,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    model = build_model(
        frame_size=args.teacher_frame_size,
        hop_size=args.hop_size,
        hidden_size=args.hidden_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Teacher parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = CombinedLoss().to(device)

    warmup_epochs = 5
    start_epoch = 0
    best_val = float("inf")
    train_losses, val_losses = [], []

    if args.resume:
        ckpt = load_state_dict_flexible(model, args.resume, device)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", -1) + 1
            best_val = ckpt.get("val_loss", best_val)
        print(f"Resumed teacher from epoch {start_epoch}, best val loss: {best_val:.4f}")

    print(f"\nTraining teacher for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            warmup_lr = args.lr * float(epoch + 1) / float(warmup_epochs)
            set_lr(optimizer, warmup_lr)

        train_loss = train_teacher_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            silence_threshold=args.silence_threshold,
            silence_penalty_weight=args.silence_penalty_weight,
            residual_reg_weight=args.residual_reg_weight,
        )
        val_loss = val_teacher_epoch(
            model,
            val_loader,
            criterion,
            device,
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
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}"
        )

        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch + 1:04d}.pt")
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            ckpt_path,
            extra={
                "mode": "teacher",
                "teacher_frame_size": args.teacher_frame_size,
                "hop_size": args.hop_size,
            },
        )

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                best_path,
                extra={
                    "mode": "teacher",
                    "teacher_frame_size": args.teacher_frame_size,
                    "hop_size": args.hop_size,
                },
            )
            print(f"  ✓ New best teacher saved -> {best_path}")

    print("\nExporting teacher TorchScript...")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    export_torchscript(model, args.output_dir, frame_size=args.teacher_frame_size, filename="teacher_scripted.pt")

    plot_loss_curves(train_losses, val_losses, args.output_dir, title="Teacher Training Loss")
    print(f"\nDone. Best teacher val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}/")


def run_distillation(args, device):
    if not args.teacher_ckpt:
        raise ValueError("--teacher_ckpt is required in distill mode")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading distillation dataset...")
    train_loader, val_loader = make_distill_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        sample_rate=SAMPLE_RATE,
        teacher_frame_size=args.teacher_frame_size,
        student_frame_size=args.student_frame_size,
        hop_size=args.hop_size,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    teacher = build_model(
        frame_size=args.teacher_frame_size,
        hop_size=args.hop_size,
        hidden_size=args.hidden_size,
    ).to(device)
    load_state_dict_flexible(teacher, args.teacher_ckpt, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_model(
        frame_size=args.student_frame_size,
        hop_size=args.hop_size,
        hidden_size=args.hidden_size,
    ).to(device)

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student parameters: {n_params:,}")

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = CombinedLoss().to(device)

    warmup_epochs = 5
    start_epoch = 0
    best_val = float("inf")
    train_losses, val_losses = [], []

    if args.resume:
        ckpt = load_state_dict_flexible(student, args.resume, device)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", -1) + 1
            best_val = ckpt.get("val_loss", best_val)
        print(f"Resumed student from epoch {start_epoch}, best val loss: {best_val:.4f}")

    print(f"\nDistilling student for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            warmup_lr = args.lr * float(epoch + 1) / float(warmup_epochs)
            set_lr(optimizer, warmup_lr)

        train_loss, train_gt, train_kd = train_distill_epoch(
            teacher,
            student,
            train_loader,
            optimizer,
            criterion,
            device,
            alpha=args.distill_alpha,
            beta=args.distill_beta,
            silence_threshold=args.silence_threshold,
            silence_penalty_weight=args.silence_penalty_weight,
            residual_reg_weight=args.residual_reg_weight,
        )
        val_loss, val_gt, val_kd = val_distill_epoch(
            teacher,
            student,
            val_loader,
            criterion,
            device,
            alpha=args.distill_alpha,
            beta=args.distill_beta,
            silence_threshold=args.silence_threshold,
            silence_penalty_weight=args.silence_penalty_weight,
            residual_reg_weight=args.residual_reg_weight,
        )

        if epoch >= warmup_epochs:
            scheduler.step()

        if val_loss is None:
            val_loss = train_loss
            val_gt = train_gt
            val_kd = train_kd

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}  "
            f"train={train_loss:.4f} (gt={train_gt:.4f}, kd={train_kd:.4f})  "
            f"val={val_loss:.4f} (gt={val_gt:.4f}, kd={val_kd:.4f})  "
            f"lr={lr_now:.2e}"
        )

        ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch + 1:04d}.pt")
        save_checkpoint(
            student,
            optimizer,
            epoch,
            val_loss,
            ckpt_path,
            extra={
                "mode": "distill",
                "teacher_ckpt": args.teacher_ckpt,
                "teacher_frame_size": args.teacher_frame_size,
                "student_frame_size": args.student_frame_size,
                "hop_size": args.hop_size,
                "distill_alpha": args.distill_alpha,
                "distill_beta": args.distill_beta,
            },
        )

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(
                student,
                optimizer,
                epoch,
                val_loss,
                best_path,
                extra={
                    "mode": "distill",
                    "teacher_ckpt": args.teacher_ckpt,
                    "teacher_frame_size": args.teacher_frame_size,
                    "student_frame_size": args.student_frame_size,
                    "hop_size": args.hop_size,
                    "distill_alpha": args.distill_alpha,
                    "distill_beta": args.distill_beta,
                },
            )
            print(f"  ✓ New best student saved -> {best_path}")

    print("\nExporting student for real-time inference...")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location="cpu", weights_only=False)
    student.load_state_dict(best_ckpt["model"])
    export_torchscript(student, args.output_dir, frame_size=args.student_frame_size, filename="model_scripted.pt")

    plot_loss_curves(train_losses, val_losses, args.output_dir, title="Student Distillation Loss")
    print(f"\nDone. Best student val loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}/")


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    if args.mode == "teacher":
        run_teacher_training(args, device)
    else:
        run_distillation(args, device)


if __name__ == "__main__":
    main()
