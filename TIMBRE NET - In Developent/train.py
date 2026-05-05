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
import itertools
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from model   import DDSPGuitarToPiano, SAMPLE_RATE, FRAME_SIZE
from dataset import make_context_dataloaders
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
    p.add_argument('--augment',     action=argparse.BooleanOptionalAction, default=True,
                   help='Enable train-only online data augmentation')
    p.add_argument('--augment_copies', type=int, default=4,
                   help='Virtual augmented copies of each training frame per epoch')
    p.add_argument('--cache_dir',   type=str,   default=None,
                   help='Optional on-disk frame cache directory for faster low-RAM training')
    p.add_argument('--cache_dtype', type=str,   default='float16', choices=['float16', 'float32'],
                   help='Storage dtype for --cache_dir frame cache')
    p.add_argument('--f0_cache_dir', type=str, default=None,
                   help='Directory of precomputed <stem>.npy f0 labels')
    p.add_argument('--require_f0_cache', action='store_true',
                   help='Fail if any clip is missing an f0 cache file')
    p.add_argument('--audio_loss_weight', type=float, default=1.0,
                   help='Weight for detached-f0 audio spectral/envelope loss')
    p.add_argument('--f0_loss_weight', type=float, default=0.1,
                   help='Weight for direct supervised f0 loss')
    p.add_argument('--voicing_loss_weight', type=float, default=0.5,
                   help='Weight for supervised voiced/unvoiced BCE loss')
    p.add_argument("--amp_loss_weight", type=float, default=1.0,
                   help='Weight for supervised global amplitude loss')
    p.add_argument('--high_freq_excess_weight', type=float, default=0.0,
                   help='Weight for one-sided high-frequency output excess loss')
    p.add_argument('--high_freq_hz', type=float, default=8000.0,
                   help='Frequency threshold for high-frequency excess loss')
    p.add_argument('--use_z', action=argparse.BooleanOptionalAction, default=True,
                   help='Enable learned ZEncoder latent conditioning')
    p.add_argument('--z_latent_size', type=int, default=64,
                   help='Dimensionality of learned ZEncoder latent')
    p.add_argument('--z_loss_weight', type=float, default=1e-4,
                   help='Weight for L2 z latent regularization')
    p.add_argument('--context_size', type=int, default=2048,
                   help='Number of guitar samples the encoder sees for each prediction')
    p.add_argument('--hop_size', type=int, default=FRAME_SIZE,
                   help='Number of target/output samples predicted per step')
    p.add_argument('--steps_per_epoch', type=int, default=None,
                   help='Limit training to this many batches per epoch')
    p.add_argument('--val_steps', type=int, default=None,
                   help='Limit validation to this many batches per epoch')
    p.add_argument('--f0_plot_every', type=int, default=10,
                   help='Save f0 diagnostic plots every this many epochs (0 disables)')
    p.add_argument('--f0_plot_points', type=int, default=256,
                   help='Maximum points to include in f0 diagnostic plots')
    p.add_argument('--resume',      type=str,   default=None,          help='Resume from checkpoint path')
    p.add_argument('--device',      type=str,   default='auto')
    return p.parse_args()


def get_device(preference: str) -> torch.device:
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')        # Apple Silicon
        else:
            return torch.device('cpu')
    return torch.device(preference)


def _limited_loader(loader, max_steps):
    if max_steps is None:
        return loader, len(loader)
    steps = min(max_steps, len(loader))
    return itertools.islice(loader, steps), steps

def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def unpack_batch(batch):
    if len(batch) == 3:
        guitar_frames, piano_frames, f0_labels = batch
        return {
            "guitar": guitar_frames,
            "piano": piano_frames,
            "f0": f0_labels,
        }
    guitar_frames, piano_frames = batch
    return {
        "guitar": guitar_frames,
        "piano": piano_frames,
        "f0": None,
    }

def supervised_f0_loss(pred_f0, f0_labels):
    voiced = f0_labels > 0.0
    if voiced.any():
        return F.smooth_l1_loss(
            torch.log(pred_f0[voiced].clamp_min(1.0)),
            torch.log(f0_labels[voiced].clamp_min(1.0)),
        )
    return pred_f0.sum() * 0.0

### ADD TO losses.py
def supervised_amp_loss(pred_global_amp, target_audio, max_amp=0.6, rms_gain=2.0):
    target_rms = torch.sqrt(torch.mean(target_audio.float() ** 2, dim=-1) + 1e-8)
    target_amp = (target_rms * rms_gain).clamp(0.0, max_amp)
    return F.smooth_l1_loss(pred_global_amp, target_amp)

def supervised_voicing_loss(voicing_logit, f0_labels):
    voiced = (f0_labels > 0.0).to(voicing_logit.dtype)
    return F.binary_cross_entropy_with_logits(voicing_logit, voiced)

def z_regularization_loss(features):
    z = features.get("z")
    if z is None:
        return None
    return torch.mean(z ** 2)

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    audio_loss_weight=1.0,
    f0_loss_weight=1.0,
    voicing_loss_weight=1.0,
    amp_loss_weight = 1.0,
    z_loss_weight=1e-4,
    max_steps=None,
):
    model.train()
    total_loss = 0.0
    total_audio_loss = 0.0
    total_f0_loss = 0.0
    total_voicing_loss = 0.0
    total_amp_loss = 0.0
    total_z_loss = 0.0
    n_batches = 0

    ### PROFILING
    # totals = {"data": 0, "forward": 0, "audio_loss": 0, "f0_loss": 0, "loss": 0, "backward": 0, "optim": 0}
    # end = time.perf_counter()

    batches, total_batches = _limited_loader(loader, max_steps)
    for batch in tqdm(batches, desc='Train', leave=False, total=total_batches):
        batch = unpack_batch(batch)
        if batch["f0"] is None:
            raise ValueError("train.py requires f0 labels. Pass --f0_cache_dir and --require_f0_cache.")
        curr_guitar = batch["guitar"].to(device, non_blocking=True)
        curr_piano = batch["piano"].to(device, non_blocking=True)
        curr_f0_labels = batch["f0"].to(device, non_blocking=True)

        if curr_f0_labels is None:
            raise ValueError("train.py requires f0 labels. Pass --f0_cache_dir and --require_f0_cache.")

        ### DATA FETCHING TIME
        # sync_if_cuda(device)
        # totals["data"] += time.perf_counter() - end

        optimizer.zero_grad(set_to_none=True)

        # sync_if_cuda(device)
        # t = time.perf_counter()

        # NOTE: autocast (fp16) is intentionally disabled.
        # torch.stft and torch.fft.rfft do not support float16 and will
        # silently return NaN under autocast on CUDA. This model is small
        # enough that float32 training is fast without it.
        features, params = model.predict_params(curr_guitar)
        pred = model.render_params(
            params,
            f0_override=curr_f0_labels,
            voicing_override=curr_f0_labels > 0.0,
        )

        # sync_if_cuda(device)
        # totals["forward"] += time.perf_counter() - t

        ### AUDIO LOSS
        # t = time.perf_counter()
        audio_loss = criterion(pred, curr_piano)
        # sync_if_cuda(device)
        # totals["audio_loss"] += time.perf_counter() - t

        ### F0 LOSS
        # t = time.perf_counter()
        f0_loss = supervised_f0_loss(params["f0_corrected"], curr_f0_labels)
        voicing_loss = supervised_voicing_loss(params["voicing_logit"], curr_f0_labels)
        # sync_if_cuda(device)
        # totals["f0_loss"] += time.perf_counter() - t

        ### AMP LOSS
        amp_loss = supervised_amp_loss(
            params["global_amp"],
            curr_piano,
            max_amp=model.decoder.global_amp_scale,
            rms_gain=2.0,
        )
        z_loss = z_regularization_loss(features)
        if z_loss is None:
            z_loss = audio_loss * 0.0
        # t = time.perf_counter()
        loss = (
            audio_loss_weight * audio_loss
            + f0_loss_weight * f0_loss
            + voicing_loss_weight * voicing_loss
            + amp_loss_weight * amp_loss
            + z_loss_weight * z_loss
        )
        # sync_if_cuda(device)
        # totals["loss"] += time.perf_counter() - t

        # NaN guard — skip corrupt batches rather than poisoning weights
        if torch.isnan(loss) or torch.isinf(loss):
            print("  ⚠ NaN/Inf loss detected, skipping batch")
            # end = time.perf_counter()
            continue

        ### BACKWARD
        # t = time.perf_counter()
        loss.backward()
        # sync_if_cuda(device)
        # totals["backward"] += time.perf_counter() - t

        ### OPTIMIZER
        # t = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # sync_if_cuda(device)
        # totals["optim"] += time.perf_counter() - t

        total_loss += loss.item()
        total_audio_loss += audio_loss.item()
        total_f0_loss += f0_loss.item()
        total_voicing_loss += voicing_loss.item()
        total_amp_loss += amp_loss.item()
        total_z_loss += z_loss.item()
        n_batches += 1
        # end = time.perf_counter()
    # print("=====PROFILING=====")
    # print({key: f"{value:.2f}s" for key, value in totals.items()}, f"total: {sum(totals.values())}")
    # print()
    n = max(n_batches, 1)

    return total_loss / n, {
        "audio": total_audio_loss / n,
        "f0": total_f0_loss / n,
        "voicing": total_voicing_loss / n,
        "amp": total_amp_loss / n,
        "z": total_z_loss / n,
    }


@torch.no_grad()
def val_epoch(
    model,
    loader,
    criterion,
    device,
    max_steps=None,
):
    model.eval()
    total_loss = 0.0
    total_f0_loss = 0.0
    total_voicing_loss = 0.0
    total_z_loss = 0.0
    f0_batches = 0
    z_batches = 0
    n_batches = 0

    batches, total_batches = _limited_loader(loader, max_steps)
    for batch in tqdm(batches, desc='Val  ', leave=False, total=total_batches):
        batch = unpack_batch(batch)
        guitar_frames = batch["guitar"].to(device)
        piano_frames = batch["piano"].to(device)
        f0_labels = batch["f0"].to(device) if batch["f0"] is not None else None

        features, params = model.predict_params(guitar_frames)
        pred = model.render_params(params)
        loss = criterion(pred, piano_frames)
        total_loss += loss.item()
        z_loss = z_regularization_loss(features)
        if z_loss is not None:
            total_z_loss += z_loss.item()
            z_batches += 1
        if f0_labels is not None:
            total_f0_loss += supervised_f0_loss(params["f0_corrected"], f0_labels).item()
            total_voicing_loss += supervised_voicing_loss(params["voicing_logit"], f0_labels).item()
            f0_batches += 1
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_f0_loss = (total_f0_loss / f0_batches) if f0_batches else None
    avg_voicing_loss = (total_voicing_loss / f0_batches) if f0_batches else None
    avg_z_loss = (total_z_loss / z_batches) if z_batches else None
    return avg_loss, avg_f0_loss, avg_voicing_loss, avg_z_loss


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch':      epoch,
        'val_loss':   val_loss,
        'model':      model.state_dict(),
        'optimizer':  optimizer.state_dict(),
    }, path)


def plot_loss_curves(
    train_losses,
    val_losses,
    output_dir,
    val_steps,
    val_epochs=None,
    train_f0_losses=None,
    val_f0_losses=None,
):
    has_f0 = bool(train_f0_losses) or bool(val_f0_losses)
    n_plots = 1 + int(has_f0)
    if n_plots > 1:
        _, axes = plt.subplots(n_plots, 1, figsize=(10, 4 + 3 * (n_plots - 1)), sharex=True)
        axes = list(axes)
        loss_ax = axes[0]
        next_ax = 1
        f0_ax = axes[next_ax] if has_f0 else None
    else:
        _, loss_ax = plt.subplots(figsize=(10, 4))
        f0_ax = None

    train_epochs = range(1, len(train_losses) + 1)
    loss_ax.plot(train_epochs, train_losses, label='Train', color='steelblue')
    plot_val = val_losses and (val_steps is None or val_steps > 0)
    if plot_val:
        if val_epochs is None:
            val_epochs = range(1, len(val_losses) + 1)
        loss_ax.plot(val_epochs, val_losses, label='Val', color='coral', marker='o')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title('DDSP Guitar→Piano Training')
    loss_ax.legend()

    if f0_ax is not None:
        if train_f0_losses:
            f0_epochs = range(1, len(train_f0_losses) + 1)
            f0_ax.plot(f0_epochs, train_f0_losses, label='Train f0', color='seagreen')
        if val_f0_losses and (val_steps is None or val_steps > 0):
            if val_epochs is None:
                val_epochs = range(1, len(val_f0_losses) + 1)
            f0_ax.plot(val_epochs, val_f0_losses, label='Val f0', color='darkorange', marker='o')
        f0_ax.set_ylabel('F0 loss')
        f0_ax.legend()

    if n_plots > 1:
        axes[-1].set_xlabel('Epoch')
    else:
        loss_ax.set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    print(f"  Loss curve saved to {output_dir}/loss_curves.png")


@torch.no_grad()
def plot_f0_diagnostics(model, loader, device, output_dir, epoch, max_points=256):
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return

    batch = unpack_batch(batch)
    guitar_frames = batch["guitar"]
    f0_labels = batch["f0"]
    if f0_labels is None:
        return

    guitar_frames = guitar_frames.to(device)
    f0_labels = f0_labels.to(device)
    features, params = model.predict_params(guitar_frames)

    encoder_f0 = features["f0"].detach().cpu()
    label_f0 = f0_labels.detach().cpu()
    pred_f0 = params["f0_corrected"].detach().cpu()
    pred_voicing = params.get("voicing_prob")
    pred_voicing = pred_voicing.detach().cpu() if pred_voicing is not None else None

    n = min(int(max_points), label_f0.numel())
    if n <= 0:
        return

    encoder_f0 = encoder_f0[:n]
    label_f0 = label_f0[:n]
    pred_f0 = pred_f0[:n]
    if pred_voicing is not None:
        pred_voicing = pred_voicing[:n]
    x = range(n)
    max_hz = max(
        1.0,
        float(encoder_f0.max().item()),
        float(label_f0.max().item()),
        float(pred_f0.max().item()),
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(x, encoder_f0, label='Encoder f0', color='steelblue')
    axes[0].plot(x, label_f0, label='Label f0', color='coral')
    axes[0].plot(x, pred_f0, label='Pred corrected f0', color='seagreen')
    if pred_voicing is not None:
        axes[0].plot(x, pred_voicing * max_hz, label='Pred voicing x max Hz', color='purple', alpha=0.45)
    axes[0].set_ylabel('Hz')
    axes[0].set_title(f'F0 Diagnostics - Epoch {epoch}')
    axes[0].legend()

    voiced = label_f0 > 0.0
    if voiced.any():
        axes[1].scatter(label_f0[voiced], encoder_f0[voiced], s=12, alpha=0.65, label='Encoder vs label')
        axes[1].scatter(label_f0[voiced], pred_f0[voiced], s=12, alpha=0.65, label='Pred vs label')
    axes[1].plot([0.0, max_hz], [0.0, max_hz], color='black', linewidth=1, linestyle='--', label='Ideal')
    axes[1].set_xlabel('Label f0 (Hz)')
    axes[1].set_ylabel('Predicted / encoder f0 (Hz)')
    axes[1].set_xlim(0.0, max_hz)
    axes[1].set_ylim(0.0, max_hz)
    axes[1].legend()

    plt.tight_layout()
    latest_path = os.path.join(output_dir, 'f0_diagnostics.png')
    epoch_path = os.path.join(output_dir, f'f0_diagnostics_epoch_{epoch:04d}.png')
    plt.savefig(latest_path, dpi=150)
    plt.savefig(epoch_path, dpi=150)
    plt.close(fig)
    print(f"  F0 diagnostics saved to {latest_path}")


def export_torchscript(model, output_dir, context_size):
    """Export model to TorchScript for fast real-time inference."""
    model.eval()
    model.cpu()
    dummy = torch.randn(1, context_size)
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

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────
    print("Loading dataset...")
    if args.context_size != 1024 or args.hop_size != FRAME_SIZE:
        print(f"Using context_size={args.context_size}, hop_size={args.hop_size}")
    else:
        print("Using context_size=1024, hop_size=256")
    if args.f0_cache_dir:
        print(f"Using precomputed f0 labels: {args.f0_cache_dir}")
    if not args.f0_cache_dir or not args.require_f0_cache:
        raise ValueError("Detached-f0 training requires --f0_cache_dir and --require_f0_cache.")

    train_loader, val_loader = make_context_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        context_size=args.context_size,
        hop_size=args.hop_size,
        augment=args.augment,
        augment_copies=args.augment_copies,
        cache_dir=args.cache_dir,
        cache_dtype=args.cache_dtype,
        f0_cache_dir=args.f0_cache_dir,
        require_f0_cache=args.require_f0_cache,
    )

    # ── Model ─────────────────────────────────
    model = DDSPGuitarToPiano(
        context_size = args.context_size,
        hop_size     = args.hop_size,
        hidden_size = args.hidden_size,
        n_harmonics = args.n_harmonics,
        use_z=args.use_z,
        z_latent_size=args.z_latent_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optim / Loss ──────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = CombinedLoss(
        high_freq_excess_weight=args.high_freq_excess_weight,
        high_freq_hz=args.high_freq_hz,
        sample_rate=SAMPLE_RATE,
    ).to(device)

    start_epoch = 0
    best_val    = float('inf')
    best_metric_name = 'loss'
    train_losses, val_losses, val_epochs = [], [], []
    train_f0_losses, val_f0_losses = [], []

    # ── Resume ────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt['val_loss']
        if best_val:
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val:.4f}")
        else:
            print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ─────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    if args.steps_per_epoch is not None:
        print(f"Training batches per epoch: {min(args.steps_per_epoch, len(train_loader)):,}")
    if args.val_steps is not None:
        print(f"Validation batches per epoch: {min(args.val_steps, len(val_loader)):,}")

    val_loss = None

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_components = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            audio_loss_weight=args.audio_loss_weight,
            f0_loss_weight=args.f0_loss_weight,
            voicing_loss_weight=args.voicing_loss_weight,
            max_steps=args.steps_per_epoch,
            amp_loss_weight=args.amp_loss_weight,
            z_loss_weight=args.z_loss_weight,
        )

        current_val_loss = None
        do_validation = args.val_steps != 0 and (epoch + 1) % 5 == 0
        if do_validation:
            val_loss, val_f0_loss, val_voicing_loss, val_z_loss = val_epoch(
                model,
                val_loader,
                criterion,
                device,
                max_steps=args.val_steps,
            )
            current_val_loss = val_loss

            val_losses.append(val_loss)
            val_epochs.append(epoch + 1)
            if val_f0_loss is not None:
                val_f0_losses.append(val_f0_loss)

        scheduler.step()
        train_losses.append(train_loss)
        train_f0_losses.append(train_components['f0'])

        lr_now = optimizer.param_groups[0]['lr']

        if do_validation:
            val_f0_str = f"  val_f0={val_f0_loss:.6f}" if val_f0_loss is not None else ""
            val_voice_str = f"  val_voice={val_voicing_loss:.6f}" if val_voicing_loss is not None else ""
            val_z_str = f"  val_z={val_z_loss:.6f}" if val_z_loss is not None else ""
            print(f"Epoch {epoch+1:3d}/{args.epochs}  "
                f"train={train_loss:.4f}  "
                f"audio={train_components['audio']:.4f}  "
                f"f0={train_components['f0']:.6f}  "
                f"voice={train_components['voicing']:.6f}  "
                f"amp={train_components['amp']:.6f}  "
                f"z={train_components['z']:.6f}  "
                f"val={val_loss:.4f}{val_f0_str}{val_voice_str}{val_z_str}  lr={lr_now:.2e}")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  "
                f"train={train_loss:.4f}  "
                f"audio={train_components['audio']:.4f}  "
                f"f0={train_components['f0']:.6f}  "
                f"voice={train_components['voicing']:.6f}  "
                f"amp={train_components['amp']:.6f}  "
                f"z={train_components['z']:.6f} lr={lr_now:.2e}")

        # Save best
        if current_val_loss and best_val and current_val_loss < best_val:
            best_val = current_val_loss
            save_checkpoint(
                model, optimizer, epoch, current_val_loss,
                os.path.join(args.output_dir, 'best_model.pt')
            )
            metric_name = "val" if current_val_loss is not None else "train"
            best_metric_name = metric_name
            print(f"  ✓ New best {metric_name} loss: {current_val_loss:.4f}")

        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, f'epoch_{epoch+1:04d}.pt')
            )

        # Plot loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_loss_curves(
                train_losses,
                val_losses,
                args.output_dir,
                args.val_steps,
                val_epochs,
                train_f0_losses,
                val_f0_losses,
            )

        if args.f0_plot_every > 0 and (epoch + 1) % args.f0_plot_every == 0:
            diagnostic_loader = val_loader if args.val_steps != 0 else train_loader
            plot_f0_diagnostics(
                model,
                diagnostic_loader,
                device,
                args.output_dir,
                epoch + 1,
                max_points=args.f0_plot_points,
            )

    # ── Export for real-time use ───────────────
    print("\nExporting for real-time inference...")
    best_ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'), map_location='cpu')
    model.load_state_dict(best_ckpt['model'])
    export_torchscript(model, args.output_dir, args.context_size)

    plot_loss_curves(
        train_losses,
        val_losses,
        args.output_dir,
        args.val_steps,
        val_epochs,
        train_f0_losses,
        val_f0_losses,
    )
    diagnostic_loader = val_loader if args.val_steps != 0 else train_loader
    plot_f0_diagnostics(
        model,
        diagnostic_loader,
        device,
        args.output_dir,
        args.epochs,
        max_points=args.f0_plot_points,
    )
    print(f"\nDone. Best {best_metric_name} loss: {best_val:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
