"""
Custom KD pretraining for DDSPGuitarToPiano -> TimbreStudent.

This trains the MAX78000-compatible student before ai8x QAT. The student learns
the same mask target used by max_dataset.py:

    input:      normalized guitar log-magnitude spectrogram
    hard label: clipped piano/guitar magnitude mask in [0, 1]
    soft label: clipped teacher-output/guitar magnitude mask in [0, 1]
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GuitarPianoDataset, load_split_manifest
from model import DDSPGuitarToPiano, FRAME_SIZE, HOP_SIZE, N_FFT, SAMPLE_RATE
# from model_distilled import TimbreStudent
from unet_distilled import TimbreUNetStudent
import ai8x

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

def parse_args():
    p = argparse.ArgumentParser(description="Custom KD pretraining for TimbreStudent")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./checkpoints_distilled")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--teacher_ckpt", "--teacher-ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n_fft", type=int, default=N_FFT)
    p.add_argument("--hop_size", type=int, default=HOP_SIZE)
    p.add_argument("--frame_size", type=int, default=FRAME_SIZE)
    p.add_argument("--base_ch", type=int, default=8)
    p.add_argument("--log_scale", type=float, default=6.0)
    p.add_argument("--hard_weight", type=float, default=0.5)
    p.add_argument("--soft_weight", type=float, default=0.5)
    p.add_argument("--split_manifest", type=str, default=None)
    p.add_argument("--eval_split", choices=["val", "test"], default="val")
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--ai8x_device", type=int, default=85, help="ai8x hardware device code, 85 for MAX78000.")
    p.add_argument("--simulate", action="store_true", help="Use ai8x hardware-simulation quantization behavior.")
    p.add_argument("--avg_pool_rounding", action="store_true", help="Use ai8x average-pooling rounding mode.")
    return p.parse_args()

def load_teacher(args, device):
    """
    Loads teacher model state dict and sets to eval
    """
    teacher = DDSPGuitarToPiano(
        sample_rate=SAMPLE_RATE,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
    ).to(device)

    payload = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    teacher.load_state_dict(state)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher

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
        augment=True,
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

def spectrogram_mag(audio, window, n_fft, hop_size, frame_size):
    """
    Computes spectrogram magnitude of the audio frame
    (B, samples) -> (B, freq_bins, frames)
    """
    spec = torch.stft(
        audio.float(),
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=frame_size,
        window=window.to(audio.device),
        return_complex=True,
        center=True,
    )
    return torch.abs(spec)

def normalize_log_mag(mag, log_scale):
    """
    Normalizes the log magnitude for better training stability.
    """
    return torch.clamp(torch.log1p(mag) / log_scale, 0.0, 1.0)

def clipped_mask(target_mag, guitar_mag):
    """
    Removes extreme outliers from target to improve training stability.
    """
    return torch.clamp(target_mag / (guitar_mag + 1e-5), 0.0, 2.0) / 2.0

def pad_to_multiple_2d(x, multiple=4):
    """
    Pads spectrogram tensors on the bottom/right so U-Net pooling dimensions line up.
    """
    freq_bins, time_frames = x.shape[-2:]
    pad_freq = (multiple - freq_bins % multiple) % multiple
    pad_time = (multiple - time_frames % multiple) % multiple
    if pad_freq == 0 and pad_time == 0:
        return x
    return F.pad(x, (0, pad_time, 0, pad_freq))

def padded_spectrogram_dimensions(args, multiple=4):
    freq_bins = args.n_fft // 2 + 1
    time_frames = args.frame_size // args.hop_size + 1
    padded_freq = freq_bins + (multiple - freq_bins % multiple) % multiple
    padded_time = time_frames + (multiple - time_frames % multiple) % multiple
    return padded_freq, padded_time

def make_kd_batch(teacher, guitar_frames, piano_frames, window, args):
    """
    Computes the student input and targets for KD training.
    The spectrogram magnitude is computed for the guitar input and piano output.
    The guitar spec-mag is transformed raw audio the student is expecting as input.
    The piano spec-mag 
    """
    guitar_mag = spectrogram_mag(
        guitar_frames, window, args.n_fft, args.hop_size, args.frame_size
    )
    piano_mag = spectrogram_mag(
        piano_frames, window, args.n_fft, args.hop_size, args.frame_size
    )

    student_input = pad_to_multiple_2d(
        normalize_log_mag(guitar_mag, args.log_scale).unsqueeze(1)
    )
    hard_target = pad_to_multiple_2d(clipped_mask(piano_mag, guitar_mag).unsqueeze(1))

    with torch.no_grad():
        teacher_audio, _, _ = teacher(guitar_frames)
        teacher_mag = spectrogram_mag(
            teacher_audio, window, args.n_fft, args.hop_size, args.frame_size
        )
        soft_target = pad_to_multiple_2d(
            clipped_mask(teacher_mag, guitar_mag).unsqueeze(1)
        )

    return student_input, hard_target, soft_target

def kd_loss(student_pred, hard_target, soft_target, hard_weight, soft_weight):
    hard_loss = torch.sqrt(F.mse_loss(student_pred, hard_target))
    soft_loss = torch.sqrt(F.mse_loss(student_pred, soft_target))
    loss = hard_weight * hard_loss + soft_weight * soft_loss
    return loss, hard_loss, soft_loss

def train_epoch(teacher, student, loader, optimizer, window, args, device):
    teacher.eval()
    student.train()
    total_loss = 0.0
    total_hard = 0.0
    total_soft = 0.0
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Train", leave=False):
        guitar_frames = guitar_frames.to(device, non_blocking=True)
        piano_frames = piano_frames.to(device, non_blocking=True)
        student_input, hard_target, soft_target = make_kd_batch(
            teacher, guitar_frames, piano_frames, window, args
        )

        optimizer.zero_grad(set_to_none=True)
        student_pred = student(student_input)
        loss, hard_loss, soft_loss = kd_loss(
            student_pred, hard_target, soft_target, args.hard_weight, args.soft_weight
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print("  NaN/Inf loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_hard += float(hard_loss.item())
        total_soft += float(soft_loss.item())
        n_batches += 1

    denom = max(1, n_batches)
    return total_loss / denom, total_hard / denom, total_soft / denom

@torch.no_grad()
def eval_epoch(teacher, student, loader, window, args, device):
    teacher.eval()
    student.eval()
    total_loss = 0.0
    total_hard = 0.0
    total_soft = 0.0
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Eval ", leave=False):
        guitar_frames = guitar_frames.to(device, non_blocking=True)
        piano_frames = piano_frames.to(device, non_blocking=True)
        student_input, hard_target, soft_target = make_kd_batch(
            teacher, guitar_frames, piano_frames, window, args
        )

        student_pred = student(student_input)
        loss, hard_loss, soft_loss = kd_loss(
            student_pred, hard_target, soft_target, args.hard_weight, args.soft_weight
        )

        total_loss += float(loss.item())
        total_hard += float(hard_loss.item())
        total_soft += float(soft_loss.item())
        n_batches += 1

    denom = max(1, n_batches)
    return total_loss / denom, total_hard / denom, total_soft / denom

def save_checkpoint(student, optimizer, epoch, val_loss, path, args):
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hard_weight": args.hard_weight,
            "soft_weight": args.soft_weight,
            "teacher_ckpt": args.teacher_ckpt,
            "frame_size": args.frame_size,
            "hop_size": args.hop_size,
            "n_fft": args.n_fft,
            "base_ch": args.base_ch,
            "log_scale": args.log_scale,
        },
        path,
    )

def plot_loss_curves(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TimbreStudent KD Pretraining")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss_curves.png", dpi=150)
    plt.close()

def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, eval_loader = make_dataloaders(args)
    teacher = load_teacher(args, device)
    ai8x.set_device(
        device=args.ai8x_device,
        simulate=args.simulate,
        round_avg=args.avg_pool_rounding,
    )
    student = TimbreUNetStudent(
        num_classes=1,
        num_channels=1,
        dimensions=padded_spectrogram_dimensions(args),
        base_ch=args.base_ch,
    ).to(device)
    window = torch.hann_window(args.frame_size, device=device)

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 0
    best_val = float("inf")
    train_losses = []
    val_losses = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val = ckpt.get("val_loss", best_val)

    warmup_epochs = min(5, args.epochs)
    print(f"Device: {device}")
    print(f"Training TimbreStudent KD for {args.epochs} epochs")

    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            set_lr(optimizer, args.lr * float(epoch + 1) / float(warmup_epochs))

        train_loss, train_hard, train_soft = train_epoch(
            teacher, student, train_loader, optimizer, window, args, device
        )
        val_loss, val_hard, val_soft = eval_epoch(
            teacher, student, eval_loader, window, args, device
        )

        if epoch >= warmup_epochs:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}\n"
            f"  train={train_loss:.5f} (hard={train_hard:.5f}, soft={train_soft:.5f})\n"
            f"  val={val_loss:.5f} (hard={val_hard:.5f}, soft={val_soft:.5f})"
        )

        if val_loss < best_val:
            best_val = val_loss
            print(f"    Best val loss: {best_val}, saving checkpoint")
            save_checkpoint(
                student,
                optimizer,
                epoch,
                val_loss,
                Path(args.output_dir) / "best_model.pt",
                args,
            )
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                student,
                optimizer,
                epoch,
                val_loss,
                Path(args.output_dir) / f"epoch_{epoch + 1:04d}.pt",
                args,
            )

        if (epoch + 1) % 5 == 0:
            plot_loss_curves(train_losses, val_losses, args.output_dir)
        
        
        print()
    print(f"Done. Best val loss: {best_val:.5f}")

if __name__ == "__main__":
    main()
