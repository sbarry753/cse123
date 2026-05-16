"""
Plot frame-level spectral KD debug maps for a distilled spectral student.

The plots compare guitar input, teacher pretransient target, teacher final
target, student output, residuals, errors, and training energy weights.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

ai8x_dir = str(Path(__file__).resolve().parent.parent / "lib" / "ai8x-training")
sys.path.insert(0, ai8x_dir)
import ai8x
from model import DDSPGuitarToPiano, FRAME_SIZE, HOP_SIZE, N_FFT, SAMPLE_RATE
from unet_distilled import TimbreUNetStudent


def parse_args():
    p = argparse.ArgumentParser(description="Plot spectral student debug maps by frame")
    p.add_argument("--guitar-wav", default="overfit/guitar/plaz.wav")
    p.add_argument("--piano-wav", default="overfit/piano/plaz.wav")
    p.add_argument("--teacher-ckpt", default="best_model.pt")
    p.add_argument("--student-ckpt", default="checkpoints_spectral/best_model.pt")
    p.add_argument("--output-dir", default="spectral_debug_plots")
    p.add_argument("--frame-indices", default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--frame_size", type=int, default=None)
    p.add_argument("--hop_size", type=int, default=None)
    p.add_argument("--n_fft", type=int, default=None)
    p.add_argument("--base_ch", type=int, default=None)
    p.add_argument("--log_scale", type=float, default=None)
    p.add_argument("--log_floor", type=float, default=None)
    p.add_argument("--residual_clip", type=float, default=None)
    p.add_argument("--energy_weight_floor", type=float, default=None)
    p.add_argument("--energy_weight_ceiling", type=float, default=None)
    p.add_argument("--debug_hf_start_hz", type=float, default=8000.0)
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


def crop_valid(x: torch.Tensor, valid_shape: tuple[int, int]) -> torch.Tensor:
    freq_bins, time_frames = valid_shape
    return x[..., :freq_bins, :time_frames]


def load_audio(path: str) -> np.ndarray:
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} -> {SAMPLE_RATE} Hz")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.squeeze(0).numpy().astype(np.float32)


def parse_frame_indices(args) -> list[int]:
    if args.frame_indices:
        indices = []
        for raw in args.frame_indices.split(","):
            raw = raw.strip()
            if raw:
                indices.append(int(raw))
        if not indices:
            raise ValueError("--frame-indices was provided but no valid indices were found")
        return indices
    if args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    return [args.start_frame + i * args.frame_stride for i in range(args.num_frames)]


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
    energy_floor = float(
        args.energy_weight_floor
        if args.energy_weight_floor is not None
        else metadata_get(payload, "energy_weight_floor", 0.1)
    )
    energy_ceiling = float(
        args.energy_weight_ceiling
        if args.energy_weight_ceiling is not None
        else metadata_get(payload, "energy_weight_ceiling", 5.0)
    )

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

    config = {
        "frame_size": frame_size,
        "hop_size": hop_size,
        "n_fft": n_fft,
        "win_length": win_length,
        "base_ch": base_ch,
        "log_scale": log_scale,
        "log_floor": log_floor,
        "residual_clip": residual_clip,
        "spectral_output": spectral_output,
        "distill_target": distill_target,
        "num_classes": num_classes,
        "energy_weight_floor": energy_floor,
        "energy_weight_ceiling": energy_ceiling,
    }
    return student, config


class SpectralDebugFrame(torch.nn.Module):
    def __init__(self, teacher, student, config: dict):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.config = config
        self.register_buffer("window", torch.hann_window(config["win_length"]))

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            audio.float(),
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_size"],
            win_length=self.config["win_length"],
            window=self.window.to(audio.device),
            return_complex=True,
            center=True,
        )

    @torch.no_grad()
    def forward(
        self,
        audio_frame: torch.Tensor,
        piano_frame: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[int, int]]:
        spec = self._stft(audio_frame)
        mag = torch.abs(spec)
        valid_shape = mag.shape[-2:]
        input_log = torch.log(torch.clamp(mag, min=self.config["log_floor"])).unsqueeze(1)
        piano_log_mag = None
        if piano_frame is not None:
            piano_spec = self._stft(piano_frame)
            piano_mag = torch.abs(piano_spec)
            piano_log_mag = torch.log(
                torch.clamp(piano_mag, min=self.config["log_floor"])
            ).unsqueeze(1)

        teacher_mask, teacher_residual = self.teacher.unet(input_log)
        teacher_log_mag = input_log * teacher_mask + teacher_residual
        target_residual = teacher_log_mag - input_log
        if self.config["residual_clip"] is not None:
            target_residual = torch.clamp(
                target_residual,
                min=-self.config["residual_clip"],
                max=self.config["residual_clip"],
            )
            teacher_log_mag = input_log + target_residual
        teacher_mag = torch.exp(teacher_log_mag)
        teacher_spec = torch.polar(teacher_mag.squeeze(1), torch.angle(spec))
        teacher_audio = torch.istft(
            teacher_spec,
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_size"],
            win_length=self.config["win_length"],
            window=self.window.to(audio_frame.device),
            center=True,
            length=audio_frame.shape[-1],
        )
        teacher_final_audio = torch.tanh(self.teacher.transient(teacher_audio))
        teacher_final_spec = self._stft(teacher_final_audio)
        teacher_final_mag = torch.abs(teacher_final_spec)
        teacher_final_log_mag = torch.log(
            torch.clamp(teacher_final_mag, min=self.config["log_floor"])
        ).unsqueeze(1)

        input_log_padded = pad_to_multiple_2d(input_log)
        if self.config["spectral_output"] in {"log_mag", "identity_mask_residual"}:
            student_input = input_log_padded
        else:
            scaled_log = torch.clamp(torch.log1p(mag) / self.config["log_scale"], 0.0, 1.0)
            student_input = pad_to_multiple_2d(scaled_log.unsqueeze(1))

        student_pred = self.student(student_input)
        if self.config["spectral_output"] == "log_mag":
            student_log_mag = student_pred[:, :1]
            predicted_residual = student_log_mag - input_log_padded
        elif self.config["spectral_output"] == "log_residual":
            predicted_residual = student_pred[:, :1]
            if self.config["residual_clip"] is not None:
                predicted_residual = torch.clamp(
                    predicted_residual,
                    min=-self.config["residual_clip"],
                    max=self.config["residual_clip"],
                )
            student_log_mag = input_log_padded + predicted_residual
        elif self.config["spectral_output"] == "identity_mask_residual":
            student_mask = 1.0 + 0.5 * torch.tanh(student_pred[:, :1])
            student_residual = student_pred[:, 1:2]
            student_log_mag = input_log_padded * student_mask + student_residual
            predicted_residual = student_log_mag - input_log_padded
        else:
            student_mask = torch.sigmoid(student_pred[:, :1]) * 2.0
            predicted_residual = student_pred[:, 1:2]
            student_log_mag = input_log_padded * student_mask + predicted_residual

        input_log = crop_valid(input_log, valid_shape)
        piano_residual = None
        if piano_log_mag is not None:
            piano_log_mag = crop_valid(piano_log_mag, valid_shape)
            piano_residual = piano_log_mag - input_log
        teacher_log_mag = crop_valid(teacher_log_mag, valid_shape)
        teacher_final_log_mag = crop_valid(teacher_final_log_mag, valid_shape)
        teacher_final_mag = crop_valid(teacher_final_mag.unsqueeze(1), valid_shape)
        teacher_final_residual = teacher_final_log_mag - input_log
        student_log_mag = crop_valid(student_log_mag, valid_shape)
        target_residual = crop_valid(target_residual, valid_shape)
        predicted_residual = crop_valid(predicted_residual, valid_shape)
        teacher_mag = crop_valid(teacher_mag, valid_shape)

        if self.config["distill_target"] == "teacher_final":
            primary_log_mag = teacher_final_log_mag
            primary_residual = teacher_final_residual
            primary_mag = teacher_final_mag
        else:
            primary_log_mag = teacher_log_mag
            primary_residual = target_residual
            primary_mag = teacher_mag

        abs_error = torch.abs(student_log_mag - primary_log_mag)
        signed_error = student_log_mag - primary_log_mag
        denom = primary_mag.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
        energy_weight = torch.clamp(
            primary_mag / denom,
            min=self.config["energy_weight_floor"],
            max=self.config["energy_weight_ceiling"],
        )
        weighted_error = energy_weight * abs_error

        result = {
            "input_log": input_log,
            "teacher_log_mag": teacher_log_mag,
            "teacher_final_log_mag": teacher_final_log_mag,
            "teacher_final_residual": teacher_final_residual,
            "student_log_mag": student_log_mag,
            "target_residual": target_residual,
            "primary_target_log_mag": primary_log_mag,
            "primary_target_residual": primary_residual,
            "predicted_residual": predicted_residual,
            "abs_error": abs_error,
            "signed_error": signed_error,
            "energy_weight": energy_weight,
            "weighted_error": weighted_error,
            "valid_shape": valid_shape,
        }
        if piano_log_mag is not None:
            result["piano_log_mag"] = piano_log_mag
            result["piano_residual"] = piano_residual
        return result


def tensor_map(x: torch.Tensor) -> np.ndarray:
    arr = x.squeeze(0).squeeze(0).detach().cpu().numpy()
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def finite_range(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([arr[np.isfinite(arr)].reshape(-1) for arr in arrays])
    if values.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if math.isclose(lo, hi):
        margin = max(1.0e-6, abs(lo) * 0.05)
        return lo - margin, hi + margin
    return lo, hi


def symmetric_range(*arrays: np.ndarray) -> tuple[float, float]:
    max_abs = max(float(np.max(np.abs(arr))) if arr.size else 0.0 for arr in arrays)
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 1.0
    return -max_abs, max_abs


def plot_frame(result: dict, out_png: Path, title: str):
    maps = {key: tensor_map(value) for key, value in result.items() if isinstance(value, torch.Tensor)}
    log_maps = [
        maps["input_log"],
        maps["teacher_log_mag"],
        maps["teacher_final_log_mag"],
        maps["student_log_mag"],
    ]
    if "piano_log_mag" in maps:
        log_maps.append(maps["piano_log_mag"])
    log_vmin, log_vmax = finite_range(*log_maps)
    residual_vmin, residual_vmax = symmetric_range(
        maps["target_residual"],
        maps["teacher_final_residual"],
        maps["primary_target_residual"],
        maps["predicted_residual"],
        maps["signed_error"],
        *( [maps["piano_residual"]] if "piano_residual" in maps else [] ),
    )
    error_vmin, error_vmax = finite_range(maps["abs_error"], maps["weighted_error"])
    weight_vmin, weight_vmax = finite_range(maps["energy_weight"])

    specs = [
        ("Guitar log-mag", "input_log", "magma", log_vmin, log_vmax),
    ]
    if "piano_log_mag" in maps:
        specs.append(("Target piano log-mag", "piano_log_mag", "magma", log_vmin, log_vmax))
    specs.extend([
        ("Teacher log-mag", "teacher_log_mag", "magma", log_vmin, log_vmax),
        ("Teacher final log-mag", "teacher_final_log_mag", "magma", log_vmin, log_vmax),
        ("Student log-mag", "student_log_mag", "magma", log_vmin, log_vmax),
        ("Target residual", "target_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Teacher final residual", "teacher_final_residual", "coolwarm", residual_vmin, residual_vmax),
    ])
    if "piano_residual" in maps:
        specs.append(("Piano residual", "piano_residual", "coolwarm", residual_vmin, residual_vmax))
    specs.extend([
        ("Predicted residual", "predicted_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Absolute error", "abs_error", "viridis", error_vmin, error_vmax),
        ("Energy weight", "energy_weight", "viridis", weight_vmin, weight_vmax),
        ("Weighted error", "weighted_error", "viridis", error_vmin, error_vmax),
        ("Signed error", "signed_error", "coolwarm", residual_vmin, residual_vmax),
    ])

    ncols = 4 if "piano_log_mag" in maps else 3
    nrows = math.ceil(len(specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), constrained_layout=True)
    fig.suptitle(title)
    axes_flat = list(np.atleast_1d(axes).flat)
    for ax, (label, key, cmap, vmin, vmax) in zip(axes_flat, specs):
        image = ax.imshow(
            maps[key],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(label)
        ax.set_xlabel("STFT frame")
        ax.set_ylabel("Frequency bin")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[len(specs):]:
        ax.axis("off")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_metrics(result: dict, out_txt: Path, args, config: dict, frame_index: int, start_sample: int):
    input_log = result["input_log"]
    teacher_log_mag = result["teacher_log_mag"]
    teacher_final_log_mag = result["teacher_final_log_mag"]
    teacher_final_residual = result["teacher_final_residual"]
    primary_target_log_mag = result["primary_target_log_mag"]
    primary_target_residual = result["primary_target_residual"]
    student_log_mag = result["student_log_mag"]
    piano_log_mag = result.get("piano_log_mag")
    piano_residual = result.get("piano_residual")
    target_residual = result["target_residual"]
    predicted_residual = result["predicted_residual"]
    abs_error = result["abs_error"]
    signed_error = result["signed_error"]
    weighted_error = result["weighted_error"]
    valid_shape = result["valid_shape"]

    def add_range(metrics: dict[str, float], name: str, value: torch.Tensor):
        metrics[f"{name}_min"] = float(value.min().item())
        metrics[f"{name}_max"] = float(value.max().item())

    def add_optional_stats(metrics: dict[str, float], prefix: str, value: torch.Tensor):
        if value.numel() == 0:
            metrics[f"{prefix}_mean"] = float("nan")
            metrics[f"{prefix}_min"] = float("nan")
            metrics[f"{prefix}_max"] = float("nan")
            return
        metrics[f"{prefix}_mean"] = float(value.mean().item())
        metrics[f"{prefix}_min"] = float(value.min().item())
        metrics[f"{prefix}_max"] = float(value.max().item())

    metrics = {
        "weighted_log_mag_l1": float(weighted_error.mean().item()),
        "unweighted_log_mag_l1": float(
            F.l1_loss(student_log_mag, primary_target_log_mag).item()
        ),
        "residual_l1": float(
            F.l1_loss(predicted_residual, primary_target_residual).item()
        ),
        "mean_absolute_error": float(abs_error.mean().item()),
        "max_absolute_error": float(abs_error.max().item()),
    }
    add_range(metrics, "input_log", input_log)
    add_range(metrics, "teacher_log_mag", teacher_log_mag)
    add_range(metrics, "teacher_final_log_mag", teacher_final_log_mag)
    add_range(metrics, "teacher_final_residual", teacher_final_residual)
    metrics["teacher_pre_vs_teacher_final_log_mag_l1"] = float(
        F.l1_loss(teacher_log_mag, teacher_final_log_mag).item()
    )
    metrics["student_vs_teacher_final_log_mag_l1"] = float(
        F.l1_loss(student_log_mag, teacher_final_log_mag).item()
    )
    metrics["student_vs_teacher_final_residual_l1"] = float(
        F.l1_loss(predicted_residual, teacher_final_residual).item()
    )
    metrics["student_vs_primary_target_log_mag_l1"] = float(
        F.l1_loss(student_log_mag, primary_target_log_mag).item()
    )
    metrics["student_vs_primary_target_residual_l1"] = float(
        F.l1_loss(predicted_residual, primary_target_residual).item()
    )
    add_range(metrics, "student_log_mag", student_log_mag)
    if piano_log_mag is not None:
        add_range(metrics, "piano_log_mag", piano_log_mag)
        metrics["teacher_vs_piano_log_mag_l1"] = float(
            F.l1_loss(teacher_log_mag, piano_log_mag).item()
        )
        metrics["teacher_final_vs_piano_log_mag_l1"] = float(
            F.l1_loss(teacher_final_log_mag, piano_log_mag).item()
        )
        metrics["student_vs_piano_log_mag_l1"] = float(
            F.l1_loss(student_log_mag, piano_log_mag).item()
        )
    if piano_residual is not None:
        add_range(metrics, "piano_residual", piano_residual)
        metrics["teacher_vs_piano_residual_l1"] = float(
            F.l1_loss(target_residual, piano_residual).item()
        )
        metrics["teacher_final_vs_piano_residual_l1"] = float(
            F.l1_loss(teacher_final_residual, piano_residual).item()
        )
        metrics["student_vs_piano_residual_l1"] = float(
            F.l1_loss(predicted_residual, piano_residual).item()
        )
    add_range(metrics, "target_residual", target_residual)
    add_range(metrics, "predicted_residual", predicted_residual)

    freq_bins = valid_shape[0]
    bin_hz = torch.arange(
        freq_bins,
        device=teacher_log_mag.device,
        dtype=teacher_log_mag.dtype,
    ) * (SAMPLE_RATE / float(config["n_fft"]))
    hf_mask = (bin_hz >= args.debug_hf_start_hz).view(1, 1, freq_bins, 1)
    hf_teacher = teacher_log_mag.masked_select(hf_mask.expand_as(teacher_log_mag))
    hf_teacher_final = teacher_final_log_mag.masked_select(
        hf_mask.expand_as(teacher_final_log_mag)
    )
    hf_primary = primary_target_log_mag.masked_select(
        hf_mask.expand_as(primary_target_log_mag)
    )
    hf_student = student_log_mag.masked_select(hf_mask.expand_as(student_log_mag))
    hf_signed_error = signed_error.masked_select(hf_mask.expand_as(signed_error))
    hf_abs_error = abs_error.masked_select(hf_mask.expand_as(abs_error))
    add_optional_stats(metrics, "high_freq_teacher_log_mag", hf_teacher)
    add_optional_stats(metrics, "high_freq_teacher_final_log_mag", hf_teacher_final)
    add_optional_stats(metrics, "high_freq_primary_target_log_mag", hf_primary)
    add_optional_stats(metrics, "high_freq_student_log_mag", hf_student)
    add_optional_stats(metrics, "high_freq_signed_error", hf_signed_error)
    add_optional_stats(metrics, "high_freq_abs_error", hf_abs_error)

    lines = [
        f"guitar_wav={args.guitar_wav}",
        f"piano_wav={args.piano_wav}",
        f"teacher_ckpt={args.teacher_ckpt}",
        f"student_ckpt={args.student_ckpt}",
        f"frame_index={frame_index}",
        f"frame_start_sample={start_sample}",
        f"frame_start_time_sec={start_sample / SAMPLE_RATE:.8f}",
        f"valid_stft_shape={valid_shape[0]}x{valid_shape[1]}",
        f"debug_hf_start_hz={args.debug_hf_start_hz}",
        f"frame_size={config['frame_size']}",
        f"hop_size={config['hop_size']}",
        f"n_fft={config['n_fft']}",
        f"win_length={config['win_length']}",
        f"base_ch={config['base_ch']}",
        f"log_scale={config['log_scale']}",
        f"log_floor={config['log_floor']}",
        f"residual_clip={config['residual_clip']}",
        f"spectral_output={config['spectral_output']}",
        f"distill_target={config['distill_target']}",
        f"energy_weight_floor={config['energy_weight_floor']}",
        f"energy_weight_ceiling={config['energy_weight_ceiling']}",
    ]
    lines.extend(f"{key}={value:.8f}" for key, value in metrics.items())
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def frame_slice(audio: np.ndarray, start_sample: int, frame_size: int) -> np.ndarray:
    if start_sample < 0:
        raise ValueError("Frame indices must be non-negative")
    frame = audio[start_sample : start_sample + frame_size]
    if len(frame) < frame_size:
        frame = np.pad(frame, (0, frame_size - len(frame)))
    return frame.astype(np.float32, copy=False)


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    ai8x.set_device(
        device=args.ai8x_device,
        simulate=args.simulate,
        round_avg=args.avg_pool_rounding,
    )

    student, config = load_student(args.student_ckpt, device, args)
    teacher = load_teacher(
        args.teacher_ckpt,
        device,
        config["frame_size"],
        config["hop_size"],
        config["n_fft"],
        config["win_length"],
    )
    debugger = SpectralDebugFrame(teacher, student, config).to(device)
    audio = load_audio(args.guitar_wav)
    piano_audio = None
    if args.piano_wav:
        piano_path = Path(args.piano_wav)
        if piano_path.exists():
            piano_audio = load_audio(args.piano_wav)
        else:
            print(f"Piano WAV not found, omitting target piano plot: {args.piano_wav}")
    frame_indices = parse_frame_indices(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Student: {args.student_ckpt} | frame={config['frame_size']}, "
        f"hop={config['hop_size']}, n_fft={config['n_fft']}, win={config['win_length']}, "
        f"spectral_output={config['spectral_output']}"
    )

    for frame_index in frame_indices:
        start_sample = frame_index * config["hop_size"]
        frame = frame_slice(audio, start_sample, config["frame_size"])
        audio_t = torch.from_numpy(frame).to(device).unsqueeze(0)
        piano_t = None
        if piano_audio is not None:
            piano_frame = frame_slice(piano_audio, start_sample, config["frame_size"])
            piano_t = torch.from_numpy(piano_frame).to(device).unsqueeze(0)
        result = debugger(audio_t, piano_t)
        png = out_dir / f"frame_{frame_index:04d}_debug.png"
        txt = out_dir / f"frame_{frame_index:04d}_metrics.txt"
        plot_frame(
            result,
            png,
            title=f"Frame {frame_index} | start {start_sample} samples ({start_sample / SAMPLE_RATE:.4f}s)",
        )
        write_metrics(result, txt, args, config, frame_index, start_sample)
        print(f"Saved {png} and {txt}")


if __name__ == "__main__":
    main()
