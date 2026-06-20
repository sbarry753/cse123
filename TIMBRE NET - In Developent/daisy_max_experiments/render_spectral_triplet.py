"""
Render guitar, final-teacher, student, and teacher-pretransient diagnostic WAVs.

This verifies whether a distilled spectral UNet checkpoint is close enough to
the teacher's final rendered output. It intentionally does not use
realtime_twostage.py because that runner may lag behind train_distilled.py's
raw-log mask/residual semantics.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

ai8x_dir = str(Path(__file__).resolve().parent.parent / "lib" / "ai8x-training")
sys.path.insert(0, ai8x_dir)
import ai8x
from losses import CombinedLoss
from model import DDSPGuitarToPiano, FRAME_SIZE, HOP_SIZE, N_FFT, SAMPLE_RATE
from unet_distilled import TimbreUNetStudent


def parse_args():
    p = argparse.ArgumentParser(description="Render spectral KD verification WAV triplet")
    p.add_argument("--guitar-wav", default="overfit/guitar/plaz.wav")
    p.add_argument("--teacher-ckpt", default="best_model.pt")
    p.add_argument("--student-ckpt", default="checkpoints_spectral/best_model.pt")
    p.add_argument("--output-dir", default="spectral_renders")
    p.add_argument("--device", default="auto")
    p.add_argument("--frame_size", type=int, default=None)
    p.add_argument("--hop_size", type=int, default=None)
    p.add_argument("--n_fft", type=int, default=None)
    p.add_argument("--base_ch", type=int, default=None)
    p.add_argument("--log_scale", type=float, default=None)
    p.add_argument("--log_floor", type=float, default=None)
    p.add_argument("--residual_clip", type=float, default=None)
    p.add_argument("--ai8x_device", type=int, default=85)
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--avg_pool_rounding", action="store_true")
    return p.parse_args()


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def checkpoint_state(payload):
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return {key: value for key, value in state.items() if key != "window"}


def metadata_get(payload: dict | None, name: str, default):
    if isinstance(payload, dict) and name in payload:
        return payload[name]
    return default


def padded_spectrogram_dimensions(n_fft: int, frame_size: int, hop_size: int, multiple: int = 4):
    freq_bins = n_fft // 2 + 1
    time_frames = frame_size // hop_size + 1
    padded_freq = freq_bins + (multiple - freq_bins % multiple) % multiple
    padded_time = time_frames + (multiple - time_frames % multiple) % multiple
    return padded_freq, padded_time


def pad_to_multiple_2d(x: torch.Tensor, multiple: int = 4) -> torch.Tensor:
    freq_bins, time_frames = x.shape[-2:]
    pad_freq = (multiple - freq_bins % multiple) % multiple
    pad_time = (multiple - time_frames % multiple) % multiple
    if pad_freq == 0 and pad_time == 0:
        return x
    return F.pad(x, (0, pad_time, 0, pad_freq))


def load_audio(path: str) -> np.ndarray:
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} -> {SAMPLE_RATE} Hz")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.squeeze(0).numpy().astype(np.float32)


def save_audio(path: Path, audio: np.ndarray):
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    torchaudio.save(str(path), torch.from_numpy(audio.astype(np.float32)).unsqueeze(0), SAMPLE_RATE)


def load_teacher(path: str, device: torch.device, frame_size: int, hop_size: int, n_fft: int, win_length: int):
    teacher = DDSPGuitarToPiano(
        sample_rate=SAMPLE_RATE,
        frame_size=frame_size,
        hop_size=hop_size,
        n_fft=n_fft,
        win_length=win_length,
    ).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    teacher.load_state_dict(checkpoint_state(payload))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def load_student(path: str, device: torch.device, args):
    payload = torch.load(path, map_location=device, weights_only=False)
    frame_size = int(args.frame_size or metadata_get(payload, "frame_size", FRAME_SIZE))
    hop_size = int(args.hop_size or metadata_get(payload, "hop_size", HOP_SIZE))
    n_fft = int(args.n_fft or metadata_get(payload, "n_fft", N_FFT))
    win_length = int(metadata_get(payload, "win_length", n_fft))
    base_ch = int(args.base_ch or metadata_get(payload, "base_ch", 8))
    log_scale = float(args.log_scale or metadata_get(payload, "log_scale", 6.0))
    spectral_output = str(metadata_get(payload, "spectral_output", "mask_residual"))
    distill_target = str(metadata_get(payload, "distill_target", "teacher_pretransient"))
    one_channel_outputs = {"log_residual", "log_mag"}
    default_num_classes = 1 if spectral_output in one_channel_outputs else 2
    num_classes = int(metadata_get(payload, "num_classes", default_num_classes))
    if spectral_output not in {"mask_residual", "identity_mask_residual", "log_residual", "log_mag"}:
        raise ValueError(f"Unsupported spectral_output={spectral_output!r}")
    expected_classes = 1 if spectral_output in one_channel_outputs else 2
    if num_classes != expected_classes:
        raise ValueError(
            f"Expected num_classes={expected_classes} for {spectral_output}, got {num_classes}."
        )
    log_floor = float(args.log_floor or metadata_get(payload, "log_floor", 1.0e-5))
    residual_clip = args.residual_clip
    if residual_clip is None:
        residual_clip = metadata_get(payload, "residual_clip", None)
    residual_clip = None if residual_clip is None else float(residual_clip)

    student = TimbreUNetStudent(
        num_classes=num_classes,
        num_channels=1,
        dimensions=padded_spectrogram_dimensions(n_fft, frame_size, hop_size),
        base_ch=base_ch,
    ).to(device)
    student.load_state_dict(checkpoint_state(payload))
    student.eval()
    for param in student.parameters():
        param.requires_grad_(False)
    return student, frame_size, hop_size, n_fft, win_length, base_ch, log_scale, log_floor, residual_clip, spectral_output, distill_target


class SpectralTripletRenderer(torch.nn.Module):
    def __init__(
        self,
        teacher: DDSPGuitarToPiano,
        student: TimbreUNetStudent,
        frame_size: int,
        hop_size: int,
        n_fft: int,
        win_length: int,
        log_scale: float,
        log_floor: float,
        residual_clip: float | None,
        spectral_output: str,
        distill_target: str,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_fft = n_fft
        self.win_length = win_length
        self.log_scale = log_scale
        self.log_floor = log_floor
        self.residual_clip = residual_clip
        self.spectral_output = spectral_output
        self.distill_target = distill_target
        self.register_buffer("window", torch.hann_window(win_length))

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            audio.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window.to(audio.device),
            return_complex=True,
            center=True,
        )

    def _istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window.to(spec.device),
            center=True,
            length=length,
        )

    @torch.no_grad()
    def forward(self, audio_frame: torch.Tensor):
        length = audio_frame.shape[-1]
        spec = self._stft(audio_frame)
        mag = torch.abs(spec)
        phase = torch.angle(spec)

        input_log = torch.log(torch.clamp(mag, min=self.log_floor)).unsqueeze(1)
        teacher_mask, teacher_residual = self.teacher.unet(input_log)
        teacher_log_mag = input_log * teacher_mask + teacher_residual
        teacher_mag = torch.exp(teacher_log_mag.squeeze(1))
        teacher_pre_audio = self._istft(torch.polar(teacher_mag, phase), length=length)
        teacher_final_audio = torch.tanh(self.teacher.transient(teacher_pre_audio))
        teacher_final_spec = self._stft(teacher_final_audio)
        teacher_final_mag = torch.abs(teacher_final_spec)
        teacher_final_log_mag = torch.log(
            torch.clamp(teacher_final_mag, min=self.log_floor)
        ).unsqueeze(1)

        input_log_padded = pad_to_multiple_2d(input_log)
        if self.spectral_output in {"log_mag", "identity_mask_residual"}:
            student_input = input_log_padded
        else:
            student_input = torch.clamp(torch.log1p(mag) / self.log_scale, 0.0, 1.0).unsqueeze(1)
            student_input = pad_to_multiple_2d(student_input)
        student_pred = self.student(pad_to_multiple_2d(student_input))

        metrics = {}
        if self.spectral_output == "log_mag":
            student_log_mag = student_pred[:, :1]
            target_residual = teacher_final_log_mag - input_log
            student_residual = student_log_mag[..., :mag.shape[-2], :mag.shape[-1]] - input_log
            metrics["residual_l1"] = F.l1_loss(student_residual, target_residual).detach()
        elif self.spectral_output == "log_residual":
            student_residual = student_pred[:, :1]
            if self.residual_clip is not None:
                student_residual = torch.clamp(
                    student_residual,
                    min=-self.residual_clip,
                    max=self.residual_clip,
                )
            target_residual = teacher_final_log_mag - input_log
            student_log_mag = input_log_padded + student_residual
            target_residual = target_residual[..., :mag.shape[-2], :mag.shape[-1]]
            student_residual = student_residual[..., :mag.shape[-2], :mag.shape[-1]]
            metrics["residual_l1"] = F.l1_loss(student_residual, target_residual).detach()
        elif self.spectral_output == "identity_mask_residual":
            student_mask = 1.0 + 0.5 * torch.tanh(student_pred[:, :1])
            student_residual = student_pred[:, 1:2]
            student_log_mag = input_log_padded * student_mask + student_residual
            target_residual = teacher_final_log_mag - input_log
            student_combined_residual = (
                student_log_mag[..., :mag.shape[-2], :mag.shape[-1]] - input_log
            )
            student_mask = student_mask[..., :mag.shape[-2], :mag.shape[-1]]
            student_residual = student_residual[..., :mag.shape[-2], :mag.shape[-1]]
            metrics["mask_mse"] = F.mse_loss(student_mask, teacher_mask).detach()
            metrics["residual_l1"] = F.l1_loss(student_residual, teacher_residual).detach()
            metrics["combined_residual_l1"] = F.l1_loss(
                student_combined_residual, target_residual
            ).detach()
        else:
            student_mask = torch.sigmoid(student_pred[:, :1]) * 2.0
            student_residual = student_pred[:, 1:2]
            student_log_mag = input_log_padded * student_mask + student_residual
            student_mask = student_mask[..., :mag.shape[-2], :mag.shape[-1]]
            student_residual = student_residual[..., :mag.shape[-2], :mag.shape[-1]]
            metrics["mask_mse"] = F.mse_loss(student_mask, teacher_mask).detach()
            metrics["residual_l1"] = F.l1_loss(student_residual, teacher_residual).detach()

        student_log_mag = student_log_mag[..., :mag.shape[-2], :mag.shape[-1]]
        student_mag = torch.exp(student_log_mag.squeeze(1))
        student_audio = self._istft(torch.polar(student_mag, phase), length=length)

        metrics["log_mag_l1"] = F.l1_loss(student_log_mag, teacher_final_log_mag).detach()
        metrics["teacher_pre_log_mag_l1"] = F.l1_loss(student_log_mag, teacher_log_mag).detach()
        metrics["teacher_pre_vs_final_log_mag_l1"] = F.l1_loss(
            teacher_log_mag, teacher_final_log_mag
        ).detach()
        return teacher_pre_audio, teacher_final_audio, student_audio, metrics


class RenderEngine:
    def __init__(self, renderer: SpectralTripletRenderer, device: torch.device):
        self.renderer = renderer
        self.device = device
        self.frame_size = renderer.frame_size
        self.hop_size = renderer.hop_size
        self.input_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.teacher_pre_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.teacher_final_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.student_ring = np.zeros(self.frame_size, dtype=np.float32)
        self.buf = torch.zeros(1, self.frame_size, device=device)

    def process_hop(self, in_hop: np.ndarray):
        self.input_ring[:-self.hop_size] = self.input_ring[self.hop_size:]
        self.input_ring[-self.hop_size:] = in_hop
        self.buf.copy_(torch.from_numpy(self.input_ring).to(self.device).unsqueeze(0))
        teacher_pre_frame, teacher_final_frame, student_frame, metrics = self.renderer(self.buf)

        self.teacher_pre_ring += teacher_pre_frame.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self.teacher_final_ring += teacher_final_frame.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self.student_ring += student_frame.squeeze(0).detach().cpu().numpy().astype(np.float32)

        teacher_pre_hop = self.teacher_pre_ring[:self.hop_size].copy()
        teacher_final_hop = self.teacher_final_ring[:self.hop_size].copy()
        student_hop = self.student_ring[:self.hop_size].copy()
        self.teacher_pre_ring[:-self.hop_size] = self.teacher_pre_ring[self.hop_size:]
        self.teacher_pre_ring[-self.hop_size:] = 0.0
        self.teacher_final_ring[:-self.hop_size] = self.teacher_final_ring[self.hop_size:]
        self.teacher_final_ring[-self.hop_size:] = 0.0
        self.student_ring[:-self.hop_size] = self.student_ring[self.hop_size:]
        self.student_ring[-self.hop_size:] = 0.0
        return teacher_pre_hop, teacher_final_hop, student_hop, metrics


def stats(name: str, audio: torch.Tensor) -> dict[str, float]:
    return {
        f"{name}_peak": float(audio.abs().max().item()),
        f"{name}_rms": float(torch.sqrt(torch.mean(audio * audio) + 1.0e-8).item()),
    }


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    ai8x.set_device(
        device=args.ai8x_device,
        simulate=args.simulate,
        round_avg=args.avg_pool_rounding,
    )

    (
        student,
        frame_size,
        hop_size,
        n_fft,
        win_length,
        base_ch,
        log_scale,
        log_floor,
        residual_clip,
        spectral_output,
        distill_target,
    ) = load_student(
        args.student_ckpt, device, args
    )
    teacher = load_teacher(args.teacher_ckpt, device, frame_size, hop_size, n_fft, win_length)
    renderer = SpectralTripletRenderer(
        teacher,
        student,
        frame_size,
        hop_size,
        n_fft,
        win_length,
        log_scale,
        log_floor,
        residual_clip,
        spectral_output,
        distill_target,
    ).to(device)
    criterion = CombinedLoss().to(device)

    audio = load_audio(args.guitar_wav)
    orig_len = len(audio)
    pad = (hop_size - (orig_len % hop_size)) % hop_size
    if pad:
        audio = np.concatenate([audio, np.zeros(pad, dtype=np.float32)])

    teacher_pre_out = np.zeros_like(audio)
    teacher_final_out = np.zeros_like(audio)
    student_out = np.zeros_like(audio)
    metric_sums = {"log_mag_l1": 0.0, "residual_l1": 0.0}
    if spectral_output in {"mask_residual", "identity_mask_residual"}:
        metric_sums["mask_mse"] = 0.0
    if spectral_output == "identity_mask_residual":
        metric_sums["combined_residual_l1"] = 0.0
    n_steps = len(audio) // hop_size
    engine = RenderEngine(renderer, device)

    print(f"Device: {device}")
    print(f"Input: {args.guitar_wav} ({orig_len / SAMPLE_RATE:.2f}s)")
    print(
        f"Student: {args.student_ckpt} | frame={frame_size}, hop={hop_size}, "
        f"n_fft={n_fft}, win={win_length}, base_ch={base_ch}, log_scale={log_scale}, "
        f"spectral_output={spectral_output}, distill_target={distill_target}"
    )

    for i in tqdm(range(n_steps), unit="hop", ncols=72):
        start = i * hop_size
        end = start + hop_size
        teacher_pre_hop, teacher_final_hop, student_hop, metrics = engine.process_hop(audio[start:end])
        teacher_pre_out[start:end] = teacher_pre_hop
        teacher_final_out[start:end] = teacher_final_hop
        student_out[start:end] = student_hop
        for key, value in metrics.items():
            metric_sums.setdefault(key, 0.0)
            metric_sums[key] += float(value.item())

    tail_hops = frame_size // hop_size
    if tail_hops:
        teacher_pre_tail = []
        teacher_final_tail = []
        student_tail = []
        zero_hop = np.zeros(hop_size, dtype=np.float32)
        for _ in range(tail_hops):
            teacher_pre_hop, teacher_final_hop, student_hop, metrics = engine.process_hop(zero_hop)
            teacher_pre_tail.append(teacher_pre_hop)
            teacher_final_tail.append(teacher_final_hop)
            student_tail.append(student_hop)
            for key, value in metrics.items():
                metric_sums.setdefault(key, 0.0)
                metric_sums[key] += float(value.item())
        teacher_pre_out = np.concatenate([teacher_pre_out, np.concatenate(teacher_pre_tail)])
        teacher_final_out = np.concatenate([teacher_final_out, np.concatenate(teacher_final_tail)])
        student_out = np.concatenate([student_out, np.concatenate(student_tail)])
        audio = np.concatenate([audio, np.zeros(frame_size, dtype=np.float32)])
        n_steps += tail_hops

    guitar_out = audio[:orig_len]
    teacher_pre_out = teacher_pre_out[:orig_len]
    teacher_final_out = teacher_final_out[:orig_len]
    student_out = student_out[:orig_len]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_audio(out_dir / "01_guitar_input.wav", guitar_out)
    save_audio(out_dir / "02_teacher_final.wav", teacher_final_out)
    save_audio(out_dir / "03_student.wav", student_out)
    save_audio(out_dir / "04_teacher_pretransient.wav", teacher_pre_out)

    teacher_t = torch.from_numpy(teacher_final_out).to(device).unsqueeze(0)
    teacher_pre_t = torch.from_numpy(teacher_pre_out).to(device).unsqueeze(0)
    student_t = torch.from_numpy(student_out).to(device).unsqueeze(0)
    guitar_t = torch.from_numpy(guitar_out).to(device).unsqueeze(0)
    combined = float(criterion(student_t, teacher_t).item())
    waveform_l1 = float(F.l1_loss(student_t, teacher_t).item())

    avg_metrics = {key: value / max(1, n_steps) for key, value in metric_sums.items()}
    report = {
        "teacher_student_combined_loss": combined,
        "teacher_student_waveform_l1": waveform_l1,
        **avg_metrics,
        **stats("guitar", guitar_t),
        **stats("teacher_final", teacher_t),
        **stats("teacher_pretransient", teacher_pre_t),
        **stats("student", student_t),
    }
    lines = [
        f"guitar_wav={args.guitar_wav}",
        f"teacher_ckpt={args.teacher_ckpt}",
        f"student_ckpt={args.student_ckpt}",
        f"frame_size={frame_size}",
        f"hop_size={hop_size}",
        f"n_fft={n_fft}",
        f"win_length={win_length}",
        f"base_ch={base_ch}",
        f"log_scale={log_scale}",
        f"log_floor={log_floor}",
        f"residual_clip={residual_clip}",
        f"spectral_output={spectral_output}",
        f"distill_target={distill_target}",
    ]
    lines.extend(f"{key}={value:.8f}" for key, value in report.items())
    (out_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved WAVs and metrics to {out_dir}")
    for key, value in report.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
