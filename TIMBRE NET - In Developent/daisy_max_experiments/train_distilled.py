"""
Identity-mask residual final-teacher KD pretraining for DDSPGuitarToPiano -> TimbreUNetStudent.

The MAX78000-compatible student receives a raw guitar log-magnitude spectrogram
and predicts a bounded deviation from identity mask plus an additive residual:

    student_mask = 1.0 + 0.5 * tanh(mask_delta)
    student_log_mag = input_log_mag * student_mask + student_residual

The default objective distills the teacher's final rendered output, including
the teacher TransientShaper, into a magnitude-only student target.
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
from losses import CombinedLoss
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
    p.add_argument("--split_manifest", type=str, default=None)
    p.add_argument("--eval_split", choices=["val", "test"], default="val")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--teacher_ckpt", "--teacher-ckpt", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n_fft", type=int, default=N_FFT)
    p.add_argument("--win_length", type=int, default=None)
    p.add_argument("--hop_size", type=int, default=HOP_SIZE)
    p.add_argument("--frame_size", type=int, default=FRAME_SIZE)
    p.add_argument("--base_ch", type=int, default=8)
    p.add_argument("--log_scale", type=float, default=6.0)
    p.add_argument("--mask_loss_weight", type=float, default=0.0)
    p.add_argument("--residual_loss_weight", type=float, default=0.0)
    p.add_argument("--final_residual_loss_weight", type=float, default=0.35)
    p.add_argument("--log_mag_loss_weight", type=float, default=0.6)
    p.add_argument("--piano_log_mag_loss_weight", type=float, default=0.0)
    p.add_argument("--piano_residual_loss_weight", type=float, default=0.0)
    p.add_argument("--log_floor", type=float, default=1e-5)
    p.add_argument("--residual_clip", type=float, default=None)
    p.add_argument("--energy_weight_log_mag_loss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--energy_weight_floor", type=float, default=0.1)
    p.add_argument("--energy_weight_ceiling", type=float, default=5.0)
    p.add_argument("--hf_suppression_loss_weight", type=float, default=0.5)
    p.add_argument("--hf_suppression_start_hz", type=float, default=8000.0)
    p.add_argument("--hf_suppression_margin", type=float, default=0.0)
    p.add_argument("--hf_suppression_topk_frac", type=float, default=0.25)
    p.add_argument("--use_teacher_audio_loss", action="store_true")
    p.add_argument("--teacher_audio_loss_weight", type=float, default=0.15)
    p.add_argument("--spectral_weight", type=float, default=1.0)
    p.add_argument("--waveform_weight", type=float, default=0.25)
    p.add_argument("--envelope_weight", type=float, default=0.10)
    p.add_argument("--onset_weight", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--ai8x_device", type=int, default=85, help="ai8x hardware device code, 85 for MAX78000.")
    p.add_argument("--simulate", action="store_true", help="Use ai8x hardware-simulation quantization behavior.")
    p.add_argument("--avg_pool_rounding", action="store_true", help="Use ai8x average-pooling rounding mode.")
    args = p.parse_args()
    args.win_length = int(args.win_length or args.n_fft)
    return args


def checkpoint_state(payload):
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return {key: value for key, value in state.items() if key != "window"}

def load_teacher(args, device):
    """
    Loads teacher model state dict and sets to eval
    """
    teacher = DDSPGuitarToPiano(
        sample_rate=SAMPLE_RATE,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        win_length=args.win_length,
    ).to(device)

    payload = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint_state(payload))
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

def spectrogram(
        audio: torch.Tensor, 
        window: torch.Tensor, 
        n_fft: int, 
        hop_size: int, 
        win_length: int) -> torch.Tensor:
    """
    Computes spectrogram of the audio frame
    (B, samples) -> (B, freq_bins, frames)
    """
    spec = torch.stft(
        audio.float(),
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_length,
        window=window.to(audio.device),
        return_complex=True,
        center=True,
    )
    return spec

def normalize_log_mag(mag, log_scale):
    """
    Normalizes the log magnitude for better training stability.
    """
    return torch.clamp(torch.log1p(mag) / log_scale, 0.0, 1.0)

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

def teacher_targets(
        teacher, 
        window: torch.Tensor, 
        length: int, 
        guitar_log_mag: torch.Tensor, 
        guitar_frames: torch.Tensor,
        phase: torch.Tensor,
        args) -> tuple[torch.Tensor, ...]:
    """
    Computes teacher pretransient diagnostics and final rendered spectral targets.
    """
    with torch.no_grad():
        teacher_mask, teacher_residual = teacher.unet(guitar_log_mag)
        teacher_pre_log_mag = guitar_log_mag * teacher_mask + teacher_residual
        teacher_pre_residual = teacher_pre_log_mag - guitar_log_mag
        if args.residual_clip is not None:
            teacher_pre_residual = torch.clamp(
                teacher_pre_residual,
                min=-args.residual_clip,
                max=args.residual_clip,
            )
            teacher_pre_log_mag = guitar_log_mag + teacher_pre_residual
            teacher_residual = teacher_pre_log_mag - guitar_log_mag * teacher_mask
        teacher_pre_mag = torch.exp(teacher_pre_log_mag.squeeze(1))

        teacher_spec = torch.polar(teacher_pre_mag, phase)
        teacher_pt_audio = torch.istft(
            teacher_spec,
            n_fft = args.n_fft,
            hop_length = args.hop_size,
            win_length = args.win_length,
            window = window.to(guitar_log_mag.device),
            center = True,
            length = length
        )

        teacher_final_audio = teacher.transient(teacher_pt_audio)
        teacher_final_audio = torch.tanh(teacher_final_audio)
        teacher_final_spec = spectrogram(
            teacher_final_audio,
            window,
            args.n_fft,
            args.hop_size,
            args.win_length,
        )
        teacher_final_mag = torch.abs(teacher_final_spec)
        teacher_final_log_mag = torch.log(
            torch.clamp(teacher_final_mag, min=args.log_floor)
        ).unsqueeze(1)
        teacher_final_residual = teacher_final_log_mag - guitar_log_mag

    return (
        teacher_mask,
        teacher_residual,
        teacher_pre_residual,
        teacher_pre_log_mag,
        teacher_pt_audio,
        teacher_pre_mag,
        teacher_final_audio,
        teacher_final_log_mag,
        teacher_final_residual,
        teacher_final_mag,
    )
        

def make_kd_batch(
            teacher,
            guitar_frames: torch.Tensor, 
            piano_frames: torch.Tensor, 
            window: torch.Tensor, args
        ) -> dict[str, torch.Tensor | tuple[int, int]]:
    result = {}

    guitar_spec = spectrogram(
        guitar_frames, window, args.n_fft, args.hop_size, args.win_length
    )

    guitar_mag = torch.abs(guitar_spec)
    phase = torch.angle(guitar_spec)
    valid_shape = guitar_mag.shape[-2:]
    guitar_log_mag = torch.log(torch.clamp(guitar_mag, args.log_floor)).unsqueeze(1)

    piano_spec = spectrogram(
        piano_frames, window, args.n_fft, args.hop_size, args.win_length
    )

    piano_mag = torch.abs(piano_spec)
    piano_log_mag = torch.log(torch.clamp(piano_mag, min=args.log_floor)).unsqueeze(1)
    piano_residual = piano_log_mag - guitar_log_mag

    (
        teacher_mask,
        teacher_residual,
        teacher_pre_residual,
        teacher_pre_log_mag,
        teacher_pt_audio,
        teacher_pre_mag,
        teacher_final_audio,
        teacher_final_log_mag,
        teacher_final_residual,
        teacher_final_mag,
    ) = teacher_targets(
        teacher,
        window,
        guitar_frames.shape[-1],
        guitar_log_mag,
        guitar_frames,
        phase,
        args,
    )
    input_log = pad_to_multiple_2d(guitar_log_mag)
    result["input_log"] = input_log
    result["student_input"] = input_log
    result["teacher_mask"] = pad_to_multiple_2d(teacher_mask)
    result["teacher_residual"] = pad_to_multiple_2d(teacher_residual)
    result["teacher_pre_residual"] = pad_to_multiple_2d(teacher_pre_residual)
    result["teacher_pre_log_mag"] = pad_to_multiple_2d(teacher_pre_log_mag)
    result["teacher_pt_audio"] = teacher_pt_audio
    result["teacher_pre_mag"] = pad_to_multiple_2d(teacher_pre_mag.unsqueeze(1))
    result["teacher_final_audio"] = teacher_final_audio
    result["teacher_final_log_mag"] = pad_to_multiple_2d(teacher_final_log_mag)
    result["teacher_final_residual"] = pad_to_multiple_2d(teacher_final_residual)
    result["teacher_final_mag"] = pad_to_multiple_2d(teacher_final_mag.unsqueeze(1))
    result["piano_log_mag"] = pad_to_multiple_2d(piano_log_mag)
    result["piano_residual"] = pad_to_multiple_2d(piano_residual)
    result["phase"] = phase
    result["valid_shape"] = valid_shape
    
    return result

def decode_student(student_pred: torch.Tensor, 
                   input_log: torch.Tensor,
                   args
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_delta = student_pred[:, :1]
        student_mask = 1.0 + 0.5 * torch.tanh(mask_delta)
        student_residual = student_pred[:, 1:2]
        pred_log_mag = input_log * student_mask + student_residual
        pred_mag = torch.exp(pred_log_mag.squeeze(1))

        return student_mask, student_residual, pred_log_mag, pred_mag

def energy_weighted_l1(pred_log_mag, teacher_log_mag, teacher_mag, args):
      abs_err = torch.abs(pred_log_mag - teacher_log_mag)
      if not args.energy_weight_log_mag_loss:
          return abs_err.mean()
      denom = teacher_mag.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
      weight = teacher_mag / denom
      weight = torch.clamp(
          weight,
          min=args.energy_weight_floor,
          max=args.energy_weight_ceiling,
      )
      return (weight * abs_err).mean()

def crop_valid(x: torch.Tensor, valid_shape: tuple[int, int]) -> torch.Tensor:
      freq_bins, time_frames = valid_shape
      return x[..., :freq_bins, :time_frames]

def high_frequency_suppression_loss(pred_log_mag, input_log, teacher_log_mag, args):
      freq_bins = pred_log_mag.shape[-2]
      bin_hz = torch.arange(
          freq_bins,
          device=pred_log_mag.device,
          dtype=pred_log_mag.dtype,
      ) * (SAMPLE_RATE / float(args.n_fft))
      high_freq_mask = (bin_hz >= args.hf_suppression_start_hz).view(1, 1, freq_bins, 1)

      suppression = F.relu(input_log - teacher_log_mag - args.hf_suppression_margin)
      suppression = suppression * high_freq_mask
      active = suppression > 0
      if not torch.any(active):
          return pred_log_mag.new_tensor(0.0)

      under_suppression = F.relu(pred_log_mag - teacher_log_mag)
      scores = (suppression * under_suppression).masked_select(active)
      topk_frac = min(max(args.hf_suppression_topk_frac, 0.0), 1.0)
      if topk_frac <= 0.0:
          return pred_log_mag.new_tensor(0.0)
      k = max(1, int(torch.ceil(scores.new_tensor(scores.numel() * topk_frac)).item()))
      return torch.topk(scores, k).values.mean()

def spectral_kd_loss(
    student_pred,
    input_log,
    phase,
    teacher_mask,
    teacher_residual,
    teacher_pre_residual,
    teacher_pre_log_mag,
    teacher_final_residual,
    teacher_final_log_mag,
    piano_log_mag,
    piano_residual,
    teacher_final_audio,
    teacher_final_mag,
    valid_shape,
    criterion,
    window,
    args,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    student_mask, student_residual, pred_log_mag, pred_mag = decode_student(
        student_pred, input_log, args
    )

    pred_log_mag_valid = crop_valid(pred_log_mag, valid_shape)
    student_mask_valid = crop_valid(student_mask, valid_shape)
    student_residual_valid = crop_valid(student_residual, valid_shape)
    teacher_mask_valid = crop_valid(teacher_mask, valid_shape)
    teacher_residual_valid = crop_valid(teacher_residual, valid_shape)
    teacher_pre_residual_valid = crop_valid(teacher_pre_residual, valid_shape)
    teacher_pre_log_mag_valid = crop_valid(teacher_pre_log_mag, valid_shape)
    teacher_final_residual_valid = crop_valid(teacher_final_residual, valid_shape)
    teacher_final_log_mag_valid = crop_valid(teacher_final_log_mag, valid_shape)
    piano_log_mag_valid = crop_valid(piano_log_mag, valid_shape)
    piano_residual_valid = crop_valid(piano_residual, valid_shape)
    teacher_final_mag_valid = crop_valid(teacher_final_mag, valid_shape)
    input_log_valid = crop_valid(input_log, valid_shape)

    log_mag_loss = energy_weighted_l1(
        pred_log_mag_valid,
        teacher_final_log_mag_valid,
        teacher_final_mag_valid,
        args,
    )

    log_mag_unweighted = F.l1_loss(pred_log_mag_valid, teacher_final_log_mag_valid)
    mask_loss = F.l1_loss(student_mask_valid, teacher_mask_valid)
    residual_loss = F.l1_loss(student_residual_valid, teacher_residual_valid)
    hf_suppression_loss = high_frequency_suppression_loss(
        pred_log_mag_valid,
        input_log_valid,
        teacher_final_log_mag_valid,
        args,
    )

    teacher_audio_loss = pred_log_mag.new_tensor(0.0)
    if args.use_teacher_audio_loss:
        pred_mag = pred_mag[..., :valid_shape[0], :valid_shape[1]]
        pred_spec = torch.polar(pred_mag, phase)
        pred_pt_audio = torch.istft(
            pred_spec,
            n_fft=args.n_fft,
            hop_length=args.hop_size,
            win_length=args.win_length,
            window=window.to(pred_mag.device),
            center=True,
            length=teacher_final_audio.shape[-1],
        )
        teacher_audio_loss = criterion(pred_pt_audio, teacher_final_audio)

    piano_log_mag_loss = F.l1_loss(pred_log_mag_valid, piano_log_mag_valid)
    decoded_residual_valid = pred_log_mag_valid - input_log_valid
    final_residual_loss = F.l1_loss(decoded_residual_valid, teacher_final_residual_valid)
    pre_residual_loss = F.l1_loss(decoded_residual_valid, teacher_pre_residual_valid)
    piano_residual_loss = F.l1_loss(decoded_residual_valid, piano_residual_valid)

    loss = (
        args.log_mag_loss_weight * log_mag_loss
        + args.mask_loss_weight * mask_loss
        + args.residual_loss_weight * residual_loss
        + args.final_residual_loss_weight * final_residual_loss
        + args.hf_suppression_loss_weight * hf_suppression_loss
        + args.piano_log_mag_loss_weight * piano_log_mag_loss
        + args.piano_residual_loss_weight * piano_residual_loss
    )
    if args.use_teacher_audio_loss:
        loss = loss + args.teacher_audio_loss_weight * teacher_audio_loss

    return loss, {
        "log_mag": log_mag_loss,
        "log_mag_unweighted": log_mag_unweighted,
        "mask": mask_loss,
        "piano_log_mag": piano_log_mag_loss,
        "piano_residual": piano_residual_loss,
        "residual": residual_loss,
        "final_residual": final_residual_loss,
        "pre_residual": pre_residual_loss,
        "pre_log_mag_unweighted": F.l1_loss(pred_log_mag_valid, teacher_pre_log_mag_valid),
        "hf_suppression": hf_suppression_loss,
        "teacher_audio_loss": teacher_audio_loss,
    }

def train_epoch(teacher, student, loader, criterion, optimizer, window, args, device):
    teacher.eval()
    student.train()
    total_loss = 0.0
    total_log_mag_loss = 0.0
    total_log_mag_unweighted_loss = 0.0
    total_mask_loss = 0.0
    total_piano_log_mag_loss = 0.0
    total_piano_residual_loss = 0.0
    total_residual_loss  = 0.0
    total_final_residual_loss = 0.0
    total_hf_suppression_loss = 0.0
    total_tch_audio_loss = 0.0

    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Train", leave=False):
        guitar_frames = guitar_frames.to(device, non_blocking=True)
        piano_frames = piano_frames.to(device, non_blocking=True)

        result = make_kd_batch(
            teacher, guitar_frames, piano_frames, window, args
        )

        optimizer.zero_grad(set_to_none=True)
        student_pred = student(result["student_input"])
        loss, loss_components = spectral_kd_loss(
            student_pred, result["input_log"], result["phase"], 
            result["teacher_mask"], result["teacher_residual"], result["teacher_pre_residual"],
            result["teacher_pre_log_mag"], result["teacher_final_residual"],
            result["teacher_final_log_mag"], result["piano_log_mag"], result["piano_residual"], 
            result["teacher_final_audio"], result["teacher_final_mag"], result["valid_shape"], 
            criterion, window, args
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print("  NaN/Inf loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_log_mag_loss += float(loss_components["log_mag"].item())
        total_log_mag_unweighted_loss += float(loss_components["log_mag_unweighted"].item())
        total_mask_loss += float(loss_components["mask"].item())
        total_piano_log_mag_loss += float(loss_components["piano_log_mag"].item())
        total_piano_residual_loss += float(loss_components["piano_residual"].item())
        total_residual_loss += float(loss_components["residual"].item())
        total_final_residual_loss += float(loss_components["final_residual"].item())
        total_hf_suppression_loss += float(loss_components["hf_suppression"].item())
        total_tch_audio_loss += float(loss_components["teacher_audio_loss"].item())
        n_batches += 1

    denom = max(1, n_batches)
    return (
        total_loss / denom,
        total_log_mag_loss / denom,
        total_log_mag_unweighted_loss / denom,
        total_mask_loss / denom,
        total_piano_log_mag_loss / denom,
        total_piano_residual_loss / denom,
        total_residual_loss / denom,
        total_final_residual_loss / denom,
        total_hf_suppression_loss / denom,
        total_tch_audio_loss / denom,
    )

@torch.no_grad()
def eval_epoch(teacher, student, loader, criterion, window, args, device):
    teacher.eval()
    student.eval()
    total_loss = 0.0
    total_log_mag_loss = 0.0
    total_log_mag_unweighted_loss = 0.0
    total_mask_loss = 0.0
    total_piano_log_mag_loss = 0.0
    total_piano_residual_loss = 0.0
    total_residual_loss  = 0.0
    total_final_residual_loss = 0.0
    total_hf_suppression_loss = 0.0
    total_tch_audio_loss = 0.0
    n_batches = 0

    for guitar_frames, piano_frames in tqdm(loader, desc="Eval ", leave=False):
        guitar_frames = guitar_frames.to(device, non_blocking=True)
        piano_frames = piano_frames.to(device, non_blocking=True)
        result = make_kd_batch(
            teacher, guitar_frames, piano_frames, window, args
        )

        student_pred = student(result["student_input"])
        loss, loss_components = spectral_kd_loss(
            student_pred, result["input_log"], result["phase"],
            result["teacher_mask"], result["teacher_residual"], result["teacher_pre_residual"],
            result["teacher_pre_log_mag"], result["teacher_final_residual"],
            result["teacher_final_log_mag"], result["piano_log_mag"], result["piano_residual"], 
            result["teacher_final_audio"], result["teacher_final_mag"], result["valid_shape"], 
            criterion, window, args
        )

        total_loss += float(loss.item())
        total_log_mag_loss += float(loss_components["log_mag"].item())
        total_log_mag_unweighted_loss += float(loss_components["log_mag_unweighted"].item())
        total_mask_loss += float(loss_components["mask"].item())
        total_piano_log_mag_loss += float(loss_components["piano_log_mag"].item())
        total_piano_residual_loss += float(loss_components["piano_residual"].item())
        total_residual_loss += float(loss_components["residual"].item())
        total_final_residual_loss += float(loss_components["final_residual"].item())
        total_hf_suppression_loss += float(loss_components["hf_suppression"].item())
        total_tch_audio_loss += float(loss_components["teacher_audio_loss"].item())
        n_batches += 1

    denom = max(1, n_batches)
    return (
        total_loss / denom,
        total_log_mag_loss / denom,
        total_log_mag_unweighted_loss / denom,
        total_mask_loss / denom,
        total_piano_log_mag_loss / denom,
        total_piano_residual_loss / denom,
        total_residual_loss / denom,
        total_final_residual_loss / denom,
        total_hf_suppression_loss / denom,
        total_tch_audio_loss / denom,
    )

def save_checkpoint(student, optimizer, epoch, val_loss, path, args):
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "teacher_ckpt": args.teacher_ckpt,
            "frame_size": args.frame_size,
            "hop_size": args.hop_size,
            "n_fft": args.n_fft,
            "win_length": args.win_length,
            "base_ch": args.base_ch,
            "log_scale": args.log_scale,
            "num_classes": 2,
            "spectral_output": "identity_mask_residual",
            "distill_target": "teacher_final",
            "log_floor": args.log_floor,
            "residual_clip": args.residual_clip,
            "energy_weight_log_mag_loss": args.energy_weight_log_mag_loss,
            "energy_weight_floor": args.energy_weight_floor,
            "energy_weight_ceiling": args.energy_weight_ceiling,
            "hf_suppression_loss_weight": args.hf_suppression_loss_weight,
            "hf_suppression_start_hz": args.hf_suppression_start_hz,
            "hf_suppression_margin": args.hf_suppression_margin,
            "hf_suppression_topk_frac": args.hf_suppression_topk_frac,
            "mask_loss_weight": args.mask_loss_weight,
            "residual_loss_weight": args.residual_loss_weight,
            "final_residual_loss_weight": args.final_residual_loss_weight,
            "log_mag_loss_weight": args.log_mag_loss_weight,
            "piano_log_mag_loss_weight": args.piano_log_mag_loss_weight,
            "piano_residual_loss_weight": args.piano_residual_loss_weight,
            "use_teacher_audio_loss": args.use_teacher_audio_loss,
            "teacher_audio_loss_weight": args.teacher_audio_loss_weight,
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

def format_metrics(metrics):
    return (
        f"loss={metrics[0]:.5f}, log_mag={metrics[1]:.5f}, "
        f"log_mag_unweighted={metrics[2]:.5f}, mask={metrics[3]:.5f}, "
        f"piano_log_mag={metrics[4]:.5f}, piano_residual={metrics[5]:.5f}, "
        f"residual={metrics[6]:.5f}, final_residual={metrics[7]:.5f}, "
        f"hf_suppression={metrics[8]:.5f}, teacher_audio={metrics[9]:.5f}"
    )

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
        num_classes=2,
        num_channels=1,
        dimensions=padded_spectrogram_dimensions(args),
        base_ch=args.base_ch,
    ).to(device)
    window = torch.hann_window(args.win_length, device=device)

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = CombinedLoss(
        spectral_weight=args.spectral_weight,
        waveform_weight=args.waveform_weight,
        envelope_weight=args.envelope_weight,
        onset_weight=args.onset_weight
    ).to(device)

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
    print(f"Training TimbreUNetStudent identity-mask residual spectral KD for {args.epochs} epochs")
    print(
        "Loss weights: "
        f"mask={args.mask_loss_weight}, "
        f"residual={args.residual_loss_weight}, "
        f"final_residual={args.final_residual_loss_weight}, "
        f"log_mag={args.log_mag_loss_weight}, "
        f"piano_log_mag={args.piano_log_mag_loss_weight}, "
        f"piano_residual={args.piano_residual_loss_weight}, "
        f"teacher_audio={args.teacher_audio_loss_weight}, "
        f"use_teacher_audio={args.use_teacher_audio_loss}, "
        f"log_floor={args.log_floor}, residual_clip={args.residual_clip}, "
        f"energy_weight_log_mag={args.energy_weight_log_mag_loss}, "
        f"energy_weight_floor={args.energy_weight_floor}, "
        f"energy_weight_ceiling={args.energy_weight_ceiling}, "
        f"hf_suppression={args.hf_suppression_loss_weight}, "
        f"hf_suppression_start_hz={args.hf_suppression_start_hz}, "
        f"hf_suppression_margin={args.hf_suppression_margin}, "
        f"hf_suppression_topk_frac={args.hf_suppression_topk_frac}"
    )

    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            set_lr(optimizer, args.lr * float(epoch + 1) / float(warmup_epochs))

        train_metrics = train_epoch(
            teacher, student, train_loader, criterion, optimizer, window, args, device
        )
        val_metrics = eval_epoch(
            teacher, student, eval_loader, criterion, window, args, device
        )
        train_loss = train_metrics[0]
        val_loss = val_metrics[0]

        if epoch >= warmup_epochs:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs}\n"
            f"  train: {format_metrics(train_metrics)}\n"
            f"  val:   {format_metrics(val_metrics)}"
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
