"""
Plot frame-level debug maps for retraining the teacher model.

The plots compare guitar input, target piano, teacher pretransient output, and
teacher final output. This is the teacher-training counterpart to
plot_spectral_debug.py, which focuses on student distillation failure modes.

python plot_teacher_debug.py --data-dir overfit --output-dir teacher_plots_tcn1D/ \
--teacher-ckpt checkpoints_teach_tcn1D/best_model.pt --win_length 1024 --base_ch 64 \
--frame_size 2048 --start-frame 500 --num-frames 5

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

from dataset import GuitarPianoDataset
from losses import CombinedLoss
from model import DDSPGuitarToPiano, FRAME_SIZE, HOP_SIZE, N_FFT, SAMPLE_RATE


def explicit_cli_args(parser, argv):
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit = set()
    for token in argv:
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit.add(dest)
    return explicit


def checkpoint_training_args(payload):
    if not isinstance(payload, dict):
        return {}
    training_args = payload.get("training_args")
    if isinstance(training_args, dict):
        return training_args
    return payload


def checkpoint_value(payload, name, default=None):
    training_args = checkpoint_training_args(payload)
    if name in training_args and training_args[name] is not None:
        return training_args[name]
    if isinstance(payload, dict):
        return payload.get(name, default)
    return default


def apply_checkpoint_config(args, payload):
    explicit = getattr(args, "_explicit_args", set())
    names = (
        "frame_size",
        "hop_size",
        "n_fft",
        "win_length",
        "hidden_size",
        "base_ch",
        "phase_tcn_ch",
        "phase_tcn_layers",
        "phase_max_delta",
        "phase_saturation_threshold",
        "energy_weight_floor",
        "energy_weight_ceiling",
    )
    for name in names:
        if name in explicit:
            continue
        value = checkpoint_value(payload, name)
        if value is not None:
            setattr(args, name, value)


def parse_args():
    p = argparse.ArgumentParser(description="Plot teacher-vs-piano spectral debug maps by frame")
    p.add_argument("--guitar-wav", default="overfit/guitar/plaz.wav")
    p.add_argument("--piano-wav", default="overfit/piano/plaz.wav")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--raw-audio", action="store_true")
    p.add_argument("--max_shift_ms", type=float, default=120.0)
    p.add_argument("--min_rms", type=float, default=0.002)
    p.add_argument("--keep_silence_prob", type=float, default=1.0)
    p.add_argument("--teacher-ckpt", default="best_model.pt", required=True)
    p.add_argument("--output-dir", default="teacher_debug_plots")
    p.add_argument("--frame-indices", default=None)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--frame_size", type=int, default=FRAME_SIZE)
    p.add_argument("--hop_size", type=int, default=HOP_SIZE)
    p.add_argument("--n_fft", type=int, default=N_FFT)
    p.add_argument("--win_length", type=int, default=None)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--phase_tcn_ch", type=int, default=None)
    p.add_argument("--phase_tcn_layers", type=int, default=None)
    p.add_argument("--phase_max_delta", type=float, default=None)
    p.add_argument("--phase_saturation_threshold", type=float, default=0.49)
    p.add_argument("--log_floor", type=float, default=1.0e-5)
    p.add_argument("--energy_weight_floor", type=float, default=0.1)
    p.add_argument("--energy_weight_ceiling", type=float, default=5.0)
    p.add_argument("--debug_hf_start_hz", type=float, default=8000.0)
    p.add_argument("--debug_attack_ms", type=float, default=20.0)
    p.add_argument("--debug_attack_contrast_margin", type=float, default=0.0)
    p.add_argument("--debug_sustain_start_ms", type=float, default=30.0)
    p.add_argument("--debug-low-energy-quantile", dest="low_energy_spectral_quantile", type=float, default=0.25)
    p.add_argument("--debug-low-energy-margin", dest="low_energy_spectral_margin", type=float, default=0.05)
    p.add_argument("--debug-onset-flux-std", dest="low_energy_onset_flux_std", type=float, default=1.5)
    p.add_argument("--debug-onset-pre-ms", dest="low_energy_onset_pre_ms", type=float, default=5.0)
    p.add_argument("--debug-onset-post-ms", dest="low_energy_onset_post_ms", type=float, default=35.0)
    p.add_argument("--debug-band-low-weight", dest="low_energy_band_low_weight", type=float, default=0.0)
    p.add_argument("--debug-band-low-mid-weight", dest="low_energy_band_low_mid_weight", type=float, default=0.5)
    p.add_argument("--debug-band-mid-weight", dest="low_energy_band_mid_weight", type=float, default=1.25)
    p.add_argument("--debug-band-high-weight", dest="low_energy_band_high_weight", type=float, default=1.25)
    p.add_argument("--debug-low-note-threshold-hz", dest="low_energy_low_note_threshold_hz", type=float, default=500.0)
    p.add_argument("--debug-low-note-ratio-threshold", dest="low_energy_low_note_ratio_threshold", type=float, default=0.45)
    p.add_argument("--debug-harmonic-protect", dest="low_energy_harmonic_protect", action="store_true")
    p.add_argument("--debug-harmonic-peak-margin", dest="low_energy_harmonic_peak_margin", type=float, default=0.10)
    p.add_argument("--debug-harmonic-peak-prominence", dest="low_energy_harmonic_peak_prominence", type=float, default=0.20)
    p.add_argument("--debug-high-energy-quantile", dest="high_energy_interharmonic_quantile", type=float, default=0.75)
    p.add_argument("--debug-high-energy-margin", dest="high_energy_interharmonic_margin", type=float, default=0.05)
    p.add_argument(
        "--debug-high-energy-peak-prominence",
        dest="high_energy_interharmonic_peak_prominence",
        type=float,
        default=0.20,
    )
    p.add_argument(
        "--debug-high-energy-peak-radius-bins",
        dest="high_energy_interharmonic_peak_radius_bins",
        type=int,
        default=1,
    )
    p.add_argument("--debug-high-energy-low-weight", dest="high_energy_interharmonic_low_weight", type=float, default=0.0)
    p.add_argument(
        "--debug-high-energy-low-mid-weight",
        dest="high_energy_interharmonic_low_mid_weight",
        type=float,
        default=0.0,
    )
    p.add_argument("--debug-high-energy-mid-weight", dest="high_energy_interharmonic_mid_weight", type=float, default=1.0)
    p.add_argument("--debug-high-energy-high-weight", dest="high_energy_interharmonic_high_weight", type=float, default=1.0)
    p.add_argument("--low_energy_spectral_quantile", dest="low_energy_spectral_quantile", type=float, help=argparse.SUPPRESS)
    p.add_argument("--low_energy_spectral_margin", dest="low_energy_spectral_margin", type=float, help=argparse.SUPPRESS)
    p.add_argument("--low_energy_onset_flux_std", dest="low_energy_onset_flux_std", type=float, help=argparse.SUPPRESS)
    p.add_argument("--low_energy_onset_pre_ms", dest="low_energy_onset_pre_ms", type=float, help=argparse.SUPPRESS)
    p.add_argument("--low_energy_onset_post_ms", dest="low_energy_onset_post_ms", type=float, help=argparse.SUPPRESS)
    p.add_argument("--low_energy_band_low_weight", dest="low_energy_band_low_weight", type=float, help=argparse.SUPPRESS)
    p.add_argument("--low_energy_band_low_mid_weight", dest="low_energy_band_low_mid_weight", type=float, help=argparse.SUPPRESS)
    p.add_argument("--low_energy_band_mid_weight", dest="low_energy_band_mid_weight", type=float, help=argparse.SUPPRESS)
    p.add_argument("--low_energy_band_high_weight", dest="low_energy_band_high_weight", type=float, help=argparse.SUPPRESS)
    p.add_argument("--low_energy_low_note_threshold_hz", dest="low_energy_low_note_threshold_hz", type=float, help=argparse.SUPPRESS)
    p.add_argument(
        "--low_energy_low_note_ratio_threshold",
        dest="low_energy_low_note_ratio_threshold",
        type=float,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--low_energy_harmonic_protect", dest="low_energy_harmonic_protect", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--low_energy_harmonic_peak_margin", dest="low_energy_harmonic_peak_margin", type=float, help=argparse.SUPPRESS)
    p.add_argument(
        "--low_energy_harmonic_peak_prominence",
        dest="low_energy_harmonic_peak_prominence",
        type=float,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--high_energy_interharmonic_quantile", dest="high_energy_interharmonic_quantile", type=float, help=argparse.SUPPRESS)
    p.add_argument("--high_energy_interharmonic_margin", dest="high_energy_interharmonic_margin", type=float, help=argparse.SUPPRESS)
    p.add_argument(
        "--high_energy_interharmonic_peak_prominence",
        dest="high_energy_interharmonic_peak_prominence",
        type=float,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--high_energy_interharmonic_peak_radius_bins",
        dest="high_energy_interharmonic_peak_radius_bins",
        type=int,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--high_energy_interharmonic_low_weight", dest="high_energy_interharmonic_low_weight", type=float, help=argparse.SUPPRESS)
    p.add_argument(
        "--high_energy_interharmonic_low_mid_weight",
        dest="high_energy_interharmonic_low_mid_weight",
        type=float,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--high_energy_interharmonic_mid_weight", dest="high_energy_interharmonic_mid_weight", type=float, help=argparse.SUPPRESS)
    p.add_argument("--high_energy_interharmonic_high_weight", dest="high_energy_interharmonic_high_weight", type=float, help=argparse.SUPPRESS)
    p.add_argument("--artifact_peak_prominence", type=float, default=0.20)
    p.add_argument("--artifact_peak_radius_bins", type=int, default=1)
    p.add_argument("--artifact_shimmer_margin", type=float, default=0.05)
    p.add_argument("--write-wavs", action="store_true")
    args = p.parse_args()
    args._explicit_args = explicit_cli_args(p, sys.argv[1:])
    return args


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


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


def load_audio(path: str) -> np.ndarray:
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} -> {SAMPLE_RATE} Hz")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio.squeeze(0).numpy().astype(np.float32)


def infer_data_dir(guitar_wav: str, piano_wav: str) -> Path | None:
    guitar_path = Path(guitar_wav)
    piano_path = Path(piano_wav)
    if guitar_path.parent.name == "guitar" and piano_path.parent.name == "piano":
        if guitar_path.parent.parent == piano_path.parent.parent:
            return guitar_path.parent.parent
    return None


def load_training_pair(args) -> tuple[np.ndarray, np.ndarray, str]:
    guitar_path = Path(args.guitar_wav)
    piano_path = Path(args.piano_wav)
    if guitar_path.stem != piano_path.stem:
        raise ValueError(
            "Training-style preprocessing requires matching guitar/piano stems. "
            f"Got {guitar_path.stem!r} and {piano_path.stem!r}."
        )

    data_dir = Path(args.data_dir) if args.data_dir else infer_data_dir(args.guitar_wav, args.piano_wav)
    if data_dir is None:
        raise ValueError(
            "Could not infer --data-dir. Pass --data-dir or use --raw-audio for direct WAV slicing."
        )

    dataset = GuitarPianoDataset(
        data_dir=str(data_dir),
        stems=[guitar_path.stem],
        sample_rate=SAMPLE_RATE,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        augment=False,
        max_shift_ms=args.max_shift_ms,
        min_rms=args.min_rms,
        keep_silence_prob=args.keep_silence_prob,
    )

    if len(dataset) == 0:
        raise ValueError(f"No aligned frames found for stem {guitar_path.stem!r}")

    guitar_frames = []
    piano_frames = []
    for idx in range(len(dataset)):
        guitar_frame, piano_frame = dataset[idx]
        guitar_frames.append(guitar_frame.numpy())
        piano_frames.append(piano_frame.numpy())

    guitar_audio = overlap_add_frames(guitar_frames, args.frame_size, args.hop_size)
    piano_audio = overlap_add_frames(piano_frames, args.frame_size, args.hop_size)
    source_desc = f"training-preprocessed pair from {data_dir} stem={guitar_path.stem}"
    return guitar_audio, piano_audio, source_desc


def overlap_add_frames(frames: list[np.ndarray], frame_size: int, hop_size: int) -> np.ndarray:
    if not frames:
        return np.zeros(0, dtype=np.float32)
    total_len = (len(frames) - 1) * hop_size + frame_size
    audio = np.zeros(total_len, dtype=np.float32)
    weight = np.zeros(total_len, dtype=np.float32)
    for idx, frame in enumerate(frames):
        start = idx * hop_size
        end = start + frame_size
        audio[start:end] += frame.astype(np.float32, copy=False)
        weight[start:end] += 1.0
    return audio / np.maximum(weight, 1.0e-8)


def load_debug_audio(args) -> tuple[np.ndarray, np.ndarray, str]:
    if args.raw_audio:
        return load_audio(args.guitar_wav), load_audio(args.piano_wav), "raw WAV slices"
    return load_training_pair(args)


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


def load_teacher(path: str, device: torch.device, args):
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict):
        apply_checkpoint_config(args, payload)
        args.win_length = int(args.win_length or checkpoint_value(payload, "win_length", args.n_fft))
        args.phase_tcn_ch = int(args.phase_tcn_ch or checkpoint_value(payload, "phase_tcn_ch", 16))
        args.phase_tcn_layers = int(args.phase_tcn_layers or checkpoint_value(payload, "phase_tcn_layers", 3))
        args.phase_max_delta = float(
            args.phase_max_delta
            if args.phase_max_delta is not None
            else checkpoint_value(payload, "phase_max_delta", 0.5)
        )
    else:
        args.win_length = int(args.win_length or args.n_fft)
        args.phase_tcn_ch = int(args.phase_tcn_ch or 16)
        args.phase_tcn_layers = int(args.phase_tcn_layers or 3)
        args.phase_max_delta = float(args.phase_max_delta if args.phase_max_delta is not None else 0.5)
    teacher = DDSPGuitarToPiano(
        sample_rate=SAMPLE_RATE,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hidden_size=args.hidden_size,
        base_ch=args.base_ch,
        phase_tcn_ch=args.phase_tcn_ch,
        phase_tcn_layers=args.phase_tcn_layers,
        phase_max_delta=args.phase_max_delta,
    ).to(device)
    load_result = teacher.load_state_dict(checkpoint_state(payload, teacher), strict=False)
    if load_result.missing_keys:
        print(f"Teacher missing model keys initialized from defaults: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Teacher ignored unexpected model keys: {load_result.unexpected_keys}")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


class TeacherDebugFrame(torch.nn.Module):
    def __init__(self, teacher, args):
        super().__init__()
        self.teacher = teacher
        self.frame_size = args.frame_size
        self.hop_size = args.hop_size
        self.n_fft = args.n_fft
        self.win_length = args.win_length
        self.log_floor = args.log_floor
        self.energy_weight_floor = args.energy_weight_floor
        self.energy_weight_ceiling = args.energy_weight_ceiling
        self.register_buffer("window", torch.hann_window(args.win_length))

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

    def _log_mag(self, audio: torch.Tensor) -> torch.Tensor:
        spec = self._stft(audio)
        mag = torch.abs(spec)
        return torch.log(torch.clamp(mag, min=self.log_floor)).unsqueeze(1)

    @torch.no_grad()
    def forward(self, guitar_frame: torch.Tensor, piano_frame: torch.Tensor) -> dict:
        spec = self._stft(guitar_frame)
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        input_log = torch.log(torch.clamp(mag, min=self.log_floor)).unsqueeze(1)

        piano_spec = self._stft(piano_frame)
        piano_mag = torch.abs(piano_spec)
        piano_phase = torch.angle(piano_spec)
        guitar_recon_audio = self._istft(spec, length=guitar_frame.shape[-1])
        piano_recon_audio = self._istft(piano_spec, length=piano_frame.shape[-1])
        piano_mag_guitar_phase_spec = torch.polar(piano_mag, phase)
        piano_mag_guitar_phase_audio = self._istft(
            piano_mag_guitar_phase_spec,
            length=guitar_frame.shape[-1],
        )

        mask, residual = self.teacher.unet(input_log)
        teacher_log_mag = input_log * mask + residual
        teacher_mag = torch.exp(teacher_log_mag.squeeze(1))
        oracle_intended_target_phase_spec = torch.polar(teacher_mag, piano_phase)
        oracle_intended_target_phase_audio = self._istft(
            oracle_intended_target_phase_spec,
            length=guitar_frame.shape[-1],
        )
        teacher_before_phase_tcn_spec = torch.polar(teacher_mag, phase)
        teacher_before_phase_tcn_audio = self._istft(
            teacher_before_phase_tcn_spec,
            length=guitar_frame.shape[-1],
        )
        teacher_final_audio, teacher_features, teacher_params = self.teacher(guitar_frame)
        teacher_before_transient_audio = teacher_features["before_transient_audio"]
        phase_delta = teacher_params["phase_delta"].unsqueeze(1)
        phase_delta_raw = teacher_params["phase_delta_raw"].unsqueeze(1)
        transient_delta = teacher_params["transient_delta"]
        oracle_intended_phase_tcn_phase_spec = torch.polar(
            teacher_mag,
            phase + teacher_params["phase_delta"],
        )
        oracle_intended_phase_tcn_phase_audio = self._istft(
            oracle_intended_phase_tcn_phase_spec,
            length=guitar_frame.shape[-1],
        )
        phase_tcn_waveform_delta = teacher_before_transient_audio - teacher_before_phase_tcn_audio
        transient_correction_delta = teacher_final_audio - teacher_before_transient_audio
        after_vs_before_waveform_delta = teacher_final_audio - teacher_before_phase_tcn_audio

        piano_log_mag = self._log_mag(piano_frame)
        piano_mag_guitar_phase_log_mag = self._log_mag(piano_mag_guitar_phase_audio)
        oracle_intended_target_phase_log_mag = self._log_mag(oracle_intended_target_phase_audio)
        oracle_intended_guitar_phase_log_mag = self._log_mag(teacher_before_phase_tcn_audio)
        oracle_intended_phase_tcn_phase_log_mag = self._log_mag(oracle_intended_phase_tcn_phase_audio)
        teacher_before_phase_tcn_log_mag = self._log_mag(teacher_before_phase_tcn_audio)
        teacher_before_transient_log_mag = self._log_mag(teacher_before_transient_audio)
        teacher_final_log_mag = self._log_mag(teacher_final_audio)
        input_residual = piano_log_mag - input_log
        teacher_intended_residual = teacher_log_mag - input_log
        oracle_intended_target_phase_residual = oracle_intended_target_phase_log_mag - input_log
        oracle_intended_guitar_phase_residual = oracle_intended_guitar_phase_log_mag - input_log
        oracle_intended_phase_tcn_phase_residual = oracle_intended_phase_tcn_phase_log_mag - input_log
        teacher_residual = teacher_before_phase_tcn_log_mag - input_log
        teacher_before_transient_residual = teacher_before_transient_log_mag - input_log
        teacher_final_residual = teacher_final_log_mag - input_log
        phase_tcn_log_mag_delta = teacher_before_transient_log_mag - teacher_before_phase_tcn_log_mag
        transient_correction_log_mag_delta = teacher_final_log_mag - teacher_before_transient_log_mag
        output_delta_log_mag = teacher_final_log_mag - teacher_before_phase_tcn_log_mag

        piano_mag_guitar_phase_error = piano_mag_guitar_phase_log_mag - piano_log_mag
        oracle_intended_target_phase_error = oracle_intended_target_phase_log_mag - piano_log_mag
        oracle_intended_guitar_phase_error = oracle_intended_guitar_phase_log_mag - piano_log_mag
        oracle_intended_phase_tcn_phase_error = oracle_intended_phase_tcn_phase_log_mag - piano_log_mag
        teacher_pre_error = teacher_before_phase_tcn_log_mag - piano_log_mag
        teacher_before_transient_error = teacher_before_transient_log_mag - piano_log_mag
        teacher_final_error = teacher_final_log_mag - piano_log_mag
        attack_logmag_contrast_delta = (
            torch.abs(teacher_final_log_mag - piano_log_mag)
            - torch.abs(teacher_final_log_mag - input_log)
        )
        teacher_flux = torch.relu(
            teacher_final_log_mag[..., 1:] - teacher_final_log_mag[..., :-1]
        )
        piano_flux = torch.relu(piano_log_mag[..., 1:] - piano_log_mag[..., :-1])
        guitar_flux = torch.relu(input_log[..., 1:] - input_log[..., :-1])
        attack_flux_contrast_delta = torch.abs(teacher_flux - piano_flux) - torch.abs(teacher_flux - guitar_flux)
        denom = torch.exp(piano_log_mag).mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
        energy_weight = torch.clamp(
            torch.exp(piano_log_mag) / denom,
            min=self.energy_weight_floor,
            max=self.energy_weight_ceiling,
        )

        return {
            "guitar_audio": guitar_frame,
            "piano_audio": piano_frame,
            "guitar_recon_audio": guitar_recon_audio,
            "piano_recon_audio": piano_recon_audio,
            "piano_mag_guitar_phase_audio": piano_mag_guitar_phase_audio,
            "oracle_intended_target_phase_audio": oracle_intended_target_phase_audio,
            "oracle_intended_guitar_phase_audio": teacher_before_phase_tcn_audio,
            "oracle_intended_phase_tcn_phase_audio": oracle_intended_phase_tcn_phase_audio,
            "teacher_before_phase_tcn_audio": teacher_before_phase_tcn_audio,
            "teacher_before_transient_audio": teacher_before_transient_audio,
            "teacher_after_phase_tcn_audio": teacher_before_transient_audio,
            "teacher_after_transient_audio": teacher_final_audio,
            "teacher_pre_audio": teacher_before_phase_tcn_audio,
            "teacher_final_audio": teacher_final_audio,
            "phase_tcn_waveform_delta": phase_tcn_waveform_delta,
            "transient_correction_delta": transient_correction_delta,
            "after_vs_before_waveform_delta": after_vs_before_waveform_delta,
            "input_log": input_log,
            "piano_log_mag": piano_log_mag,
            "piano_mag_guitar_phase_log_mag": piano_mag_guitar_phase_log_mag,
            "oracle_intended_target_phase_log_mag": oracle_intended_target_phase_log_mag,
            "oracle_intended_guitar_phase_log_mag": oracle_intended_guitar_phase_log_mag,
            "oracle_intended_phase_tcn_phase_log_mag": oracle_intended_phase_tcn_phase_log_mag,
            "teacher_log_mag": teacher_before_phase_tcn_log_mag,
            "teacher_intended_log_mag": teacher_log_mag,
            "teacher_before_phase_tcn_log_mag": teacher_before_phase_tcn_log_mag,
            "teacher_before_transient_log_mag": teacher_before_transient_log_mag,
            "teacher_final_log_mag": teacher_final_log_mag,
            "teacher_after_phase_tcn_log_mag": teacher_before_transient_log_mag,
            "teacher_after_transient_log_mag": teacher_final_log_mag,
            "piano_residual": input_residual,
            "teacher_residual": teacher_residual,
            "teacher_intended_residual": teacher_intended_residual,
            "oracle_intended_target_phase_residual": oracle_intended_target_phase_residual,
            "oracle_intended_guitar_phase_residual": oracle_intended_guitar_phase_residual,
            "oracle_intended_phase_tcn_phase_residual": oracle_intended_phase_tcn_phase_residual,
            "teacher_before_phase_tcn_residual": teacher_residual,
            "teacher_before_transient_residual": teacher_before_transient_residual,
            "teacher_final_residual": teacher_final_residual,
            "teacher_after_phase_tcn_residual": teacher_before_transient_residual,
            "teacher_after_transient_residual": teacher_final_residual,
            "phase_tcn_log_mag_delta": phase_tcn_log_mag_delta,
            "transient_correction_log_mag_delta": transient_correction_log_mag_delta,
            "output_delta_log_mag": output_delta_log_mag,
            "piano_mag_guitar_phase_signed_error": piano_mag_guitar_phase_error,
            "oracle_intended_target_phase_signed_error": oracle_intended_target_phase_error,
            "oracle_intended_guitar_phase_signed_error": oracle_intended_guitar_phase_error,
            "oracle_intended_phase_tcn_phase_signed_error": oracle_intended_phase_tcn_phase_error,
            "teacher_pre_signed_error": teacher_pre_error,
            "teacher_before_transient_signed_error": teacher_before_transient_error,
            "teacher_final_signed_error": teacher_final_error,
            "teacher_after_phase_tcn_signed_error": teacher_before_transient_error,
            "teacher_after_transient_signed_error": teacher_final_error,
            "piano_mag_guitar_phase_abs_error": torch.abs(piano_mag_guitar_phase_error),
            "oracle_intended_target_phase_abs_error": torch.abs(oracle_intended_target_phase_error),
            "oracle_intended_guitar_phase_abs_error": torch.abs(oracle_intended_guitar_phase_error),
            "oracle_intended_phase_tcn_phase_abs_error": torch.abs(oracle_intended_phase_tcn_phase_error),
            "teacher_pre_abs_error": torch.abs(teacher_pre_error),
            "teacher_before_transient_abs_error": torch.abs(teacher_before_transient_error),
            "teacher_final_abs_error": torch.abs(teacher_final_error),
            "teacher_after_phase_tcn_abs_error": torch.abs(teacher_before_transient_error),
            "teacher_after_transient_abs_error": torch.abs(teacher_final_error),
            "attack_logmag_contrast_delta": attack_logmag_contrast_delta,
            "attack_flux_contrast_delta": attack_flux_contrast_delta,
            "energy_weight": energy_weight,
            "weighted_teacher_final_abs_error": energy_weight * torch.abs(teacher_final_error),
            "mask": mask,
            "raw_residual": residual,
            "phase_delta": phase_delta,
            "phase_delta_abs": torch.abs(phase_delta),
            "phase_delta_raw": phase_delta_raw,
            "transient_delta": transient_delta.unsqueeze(1),
            "transient_delta_abs": torch.abs(transient_delta).unsqueeze(1),
            "valid_shape": input_log.shape[-2:],
        }


def tensor_map(x: torch.Tensor) -> np.ndarray:
    arr = x.squeeze(0).squeeze(0).detach().cpu().numpy()
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def audio_vec(x: torch.Tensor) -> np.ndarray:
    return x.squeeze(0).detach().cpu().numpy()


def low_energy_onset_exclusion_mask(target_log_mag: torch.Tensor, args) -> torch.Tensor:
    time_bins = target_log_mag.shape[-1]
    if time_bins <= 1:
        return torch.zeros(
            target_log_mag.shape[0],
            1,
            1,
            time_bins,
            device=target_log_mag.device,
            dtype=torch.bool,
        )

    flux = F.relu(target_log_mag[..., 1:] - target_log_mag[..., :-1]).mean(dim=-2).squeeze(1)
    flux = F.pad(flux, (1, 0))
    threshold = flux.mean(dim=1, keepdim=True) + float(args.low_energy_onset_flux_std) * flux.std(
        dim=1,
        keepdim=True,
        unbiased=False,
    )
    onset = flux > threshold
    pre_cols = int(float(args.low_energy_onset_pre_ms) * SAMPLE_RATE / 1000.0 / max(1, args.hop_size))
    post_cols = int(float(args.low_energy_onset_post_ms) * SAMPLE_RATE / 1000.0 / max(1, args.hop_size))
    excluded = torch.zeros_like(onset)
    for shift in range(-pre_cols, post_cols + 1):
        if shift < 0:
            excluded[:, :shift] |= onset[:, -shift:]
        elif shift > 0:
            excluded[:, shift:] |= onset[:, :-shift]
        else:
            excluded |= onset
    return excluded.view(target_log_mag.shape[0], 1, 1, time_bins)


def low_energy_band_weight_map(bin_hz: torch.Tensor, args) -> torch.Tensor:
    weights = torch.zeros_like(bin_hz)
    weights = torch.where(bin_hz < 500.0, torch.full_like(weights, float(args.low_energy_band_low_weight)), weights)
    weights = torch.where(
        (bin_hz >= 500.0) & (bin_hz < 2000.0),
        torch.full_like(weights, float(args.low_energy_band_low_mid_weight)),
        weights,
    )
    weights = torch.where(
        (bin_hz >= 2000.0) & (bin_hz < 8000.0),
        torch.full_like(weights, float(args.low_energy_band_mid_weight)),
        weights,
    )
    weights = torch.where(bin_hz >= 8000.0, torch.full_like(weights, float(args.low_energy_band_high_weight)), weights)
    return weights.view(1, 1, -1, 1)


def low_energy_harmonic_protection(target_log_mag: torch.Tensor, args) -> torch.Tensor:
    if not args.low_energy_harmonic_protect or target_log_mag.shape[-2] < 3:
        return torch.zeros_like(target_log_mag, dtype=torch.bool)
    center = target_log_mag[..., 1:-1, :]
    left = target_log_mag[..., :-2, :]
    right = target_log_mag[..., 2:, :]
    prominence = float(args.low_energy_harmonic_peak_prominence)
    peak = (center > left + prominence) & (center > right + prominence)
    return F.pad(peak, (0, 0, 1, 1))


def artifact_band_masks(bin_hz: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "low": (bin_hz >= 0.0) & (bin_hz < 500.0),
        "low_mid": (bin_hz >= 500.0) & (bin_hz < 2000.0),
        "mid": (bin_hz >= 2000.0) & (bin_hz < 8000.0),
        "high": bin_hz >= 8000.0,
    }


def dilate_frequency_mask(mask: torch.Tensor, radius_bins: int) -> torch.Tensor:
    radius_bins = max(0, int(radius_bins))
    if radius_bins == 0 or mask.shape[-2] <= 1:
        return mask
    dilated = mask.clone()
    for shift in range(1, radius_bins + 1):
        dilated[..., shift:, :] |= mask[..., :-shift, :]
        dilated[..., :-shift, :] |= mask[..., shift:, :]
    return dilated


def artifact_harmonic_region(
    piano_log_mag: torch.Tensor,
    prominence: float,
    radius_bins: int,
) -> torch.Tensor:
    if piano_log_mag.shape[-2] < 3:
        return torch.ones_like(piano_log_mag, dtype=torch.bool)
    center = piano_log_mag[..., 1:-1, :]
    left = piano_log_mag[..., :-2, :]
    right = piano_log_mag[..., 2:, :]
    peaks = (center > left + float(prominence)) & (center > right + float(prominence))
    peaks = F.pad(peaks, (0, 0, 1, 1))
    return dilate_frequency_mask(peaks, radius_bins)


def safe_masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value.masked_select(mask.expand_as(value))
    if selected.numel() == 0:
        return float("nan")
    return float(selected.mean().item())


def add_stage_band_metrics(
    metrics: dict[str, float],
    metric_name: str,
    values_by_stage: dict[str, torch.Tensor],
    masks_by_band: dict[str, torch.Tensor],
):
    for stage, value in values_by_stage.items():
        for band, mask in masks_by_band.items():
            key = (
                metric_name.format(stage=stage, band=band)
                if "{stage}" in metric_name or "{band}" in metric_name
                else f"{metric_name}_{stage}_{band}"
            )
            metrics[key] = safe_masked_mean(value, mask)


def audio_stats(x: torch.Tensor) -> dict[str, float]:
    y = x.detach().float().reshape(-1)
    peak = y.abs().max().item()
    rms = torch.sqrt(torch.mean(y ** 2) + 1.0e-12).item()
    return {
        "peak": float(peak),
        "rms": float(rms),
        "crest": float(peak / max(rms, 1.0e-12)),
        "pct_abs_gt_0p95": float((y.abs() > 0.95).float().mean().item() * 100.0),
        "pct_abs_gt_1p0": float((y.abs() > 1.0).float().mean().item() * 100.0),
    }


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


def plot_frame(result: dict, out_png: Path, title: str, args):
    maps = {
        key: tensor_map(value)
        for key, value in result.items()
        if isinstance(value, torch.Tensor) and value.ndim == 4
    }
    freq_bins = result["valid_shape"][0]
    bin_hz = torch.arange(
        freq_bins,
        device=result["teacher_final_log_mag"].device,
        dtype=result["teacher_final_log_mag"].dtype,
    ) * (SAMPLE_RATE / float(args.n_fft))
    target_mag = torch.exp(result["piano_log_mag"])
    threshold = torch.quantile(target_mag.reshape(-1), max(0.0, min(1.0, float(args.low_energy_spectral_quantile))))
    low_energy_mask = target_mag <= threshold
    onset_excluded = low_energy_onset_exclusion_mask(result["piano_log_mag"], args)
    sustain_mask = ~onset_excluded
    harmonic_protected = low_energy_harmonic_protection(result["piano_log_mag"], args)
    margin = float(args.low_energy_spectral_margin) + harmonic_protected.to(result["piano_log_mag"].dtype) * float(
        args.low_energy_harmonic_peak_margin
    )
    over = torch.relu(result["teacher_final_log_mag"] - result["piano_log_mag"] - margin)
    band_weights = low_energy_band_weight_map(bin_hz, args)
    maps["low_energy_sustain_mask"] = tensor_map(sustain_mask.to(over.dtype).expand_as(over))
    maps["low_energy_onset_excluded"] = tensor_map(onset_excluded.to(over.dtype).expand_as(over))
    maps["low_energy_weighted_over"] = tensor_map(over * low_energy_mask.to(over.dtype) * sustain_mask.to(over.dtype) * band_weights)
    log_vmin, log_vmax = finite_range(
        maps["input_log"],
        maps["piano_log_mag"],
        maps["piano_mag_guitar_phase_log_mag"],
        maps["teacher_intended_log_mag"],
        maps["oracle_intended_target_phase_log_mag"],
        maps["oracle_intended_guitar_phase_log_mag"],
        maps["oracle_intended_phase_tcn_phase_log_mag"],
        maps["teacher_log_mag"],
        maps["teacher_before_transient_log_mag"],
        maps["teacher_final_log_mag"],
    )
    residual_vmin, residual_vmax = symmetric_range(
        maps["piano_residual"],
        maps["teacher_intended_residual"],
        maps["oracle_intended_target_phase_residual"],
        maps["oracle_intended_guitar_phase_residual"],
        maps["oracle_intended_phase_tcn_phase_residual"],
        maps["teacher_residual"],
        maps["teacher_before_transient_residual"],
        maps["teacher_final_residual"],
        maps["phase_tcn_log_mag_delta"],
        maps["transient_correction_log_mag_delta"],
        maps["output_delta_log_mag"],
        maps["piano_mag_guitar_phase_signed_error"],
        maps["oracle_intended_target_phase_signed_error"],
        maps["oracle_intended_guitar_phase_signed_error"],
        maps["oracle_intended_phase_tcn_phase_signed_error"],
        maps["teacher_before_transient_signed_error"],
        maps["teacher_final_signed_error"],
    )
    contrast_vmin, contrast_vmax = symmetric_range(
        maps["attack_logmag_contrast_delta"],
        maps["attack_flux_contrast_delta"],
    )
    phase_vmin, phase_vmax = symmetric_range(maps["phase_delta"])
    phase_abs_vmin, phase_abs_vmax = finite_range(maps["phase_delta_abs"])
    error_vmin, error_vmax = finite_range(
        maps["piano_mag_guitar_phase_abs_error"],
        maps["oracle_intended_target_phase_abs_error"],
        maps["oracle_intended_guitar_phase_abs_error"],
        maps["oracle_intended_phase_tcn_phase_abs_error"],
        maps["teacher_pre_abs_error"],
        maps["teacher_before_transient_abs_error"],
        maps["teacher_final_abs_error"],
        maps["weighted_teacher_final_abs_error"],
    )
    mask_vmin, mask_vmax = finite_range(maps["mask"])
    weight_vmin, weight_vmax = finite_range(maps["energy_weight"])

    specs = [
        ("Guitar log-mag", "input_log", "magma", log_vmin, log_vmax),
        ("Target piano log-mag", "piano_log_mag", "magma", log_vmin, log_vmax),
        ("Piano mag + guitar phase log-mag", "piano_mag_guitar_phase_log_mag", "magma", log_vmin, log_vmax),
        ("Teacher intended log-mag", "teacher_intended_log_mag", "magma", log_vmin, log_vmax),
        ("Oracle intended + piano phase log-mag", "oracle_intended_target_phase_log_mag", "magma", log_vmin, log_vmax),
        ("Oracle intended + guitar phase log-mag", "oracle_intended_guitar_phase_log_mag", "magma", log_vmin, log_vmax),
        ("Oracle intended + phase TCN phase log-mag", "oracle_intended_phase_tcn_phase_log_mag", "magma", log_vmin, log_vmax),
        ("Teacher before phase TCN log-mag", "teacher_log_mag", "magma", log_vmin, log_vmax),
        ("Teacher before transient log-mag", "teacher_before_transient_log_mag", "magma", log_vmin, log_vmax),
        ("Teacher after transient log-mag", "teacher_final_log_mag", "magma", log_vmin, log_vmax),
        ("Piano residual", "piano_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Teacher intended residual", "teacher_intended_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Oracle intended + piano phase residual", "oracle_intended_target_phase_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Oracle intended + guitar phase residual", "oracle_intended_guitar_phase_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Oracle intended + phase TCN residual", "oracle_intended_phase_tcn_phase_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Teacher before phase TCN residual", "teacher_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Teacher before transient residual", "teacher_before_transient_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Teacher after transient residual", "teacher_final_residual", "coolwarm", residual_vmin, residual_vmax),
        ("Phase TCN log-mag delta", "phase_tcn_log_mag_delta", "coolwarm", residual_vmin, residual_vmax),
        ("Transient correction log-mag delta", "transient_correction_log_mag_delta", "coolwarm", residual_vmin, residual_vmax),
        ("Phase residual", "phase_delta", "coolwarm", phase_vmin, phase_vmax),
        ("Phase residual abs", "phase_delta_abs", "viridis", phase_abs_vmin, phase_abs_vmax),
        ("Piano mag + guitar phase signed error", "piano_mag_guitar_phase_signed_error", "coolwarm", residual_vmin, residual_vmax),
        ("Oracle intended + piano phase signed error", "oracle_intended_target_phase_signed_error", "coolwarm", residual_vmin, residual_vmax),
        ("Oracle intended + guitar phase signed error", "oracle_intended_guitar_phase_signed_error", "coolwarm", residual_vmin, residual_vmax),
        ("Oracle intended + phase TCN signed error", "oracle_intended_phase_tcn_phase_signed_error", "coolwarm", residual_vmin, residual_vmax),
        ("Teacher before phase TCN signed error", "teacher_pre_signed_error", "coolwarm", residual_vmin, residual_vmax),
        ("Teacher before transient signed error", "teacher_before_transient_signed_error", "coolwarm", residual_vmin, residual_vmax),
        ("Teacher after transient signed error", "teacher_final_signed_error", "coolwarm", residual_vmin, residual_vmax),
        ("Piano mag + guitar phase abs error", "piano_mag_guitar_phase_abs_error", "viridis", error_vmin, error_vmax),
        ("Oracle intended + piano phase abs error", "oracle_intended_target_phase_abs_error", "viridis", error_vmin, error_vmax),
        ("Oracle intended + guitar phase abs error", "oracle_intended_guitar_phase_abs_error", "viridis", error_vmin, error_vmax),
        ("Oracle intended + phase TCN abs error", "oracle_intended_phase_tcn_phase_abs_error", "viridis", error_vmin, error_vmax),
        ("Teacher before phase TCN abs error", "teacher_pre_abs_error", "viridis", error_vmin, error_vmax),
        ("Teacher before transient abs error", "teacher_before_transient_abs_error", "viridis", error_vmin, error_vmax),
        ("Teacher after transient abs error", "teacher_final_abs_error", "viridis", error_vmin, error_vmax),
        ("Piano energy weight", "energy_weight", "viridis", weight_vmin, weight_vmax),
        ("Attack logmag contrast", "attack_logmag_contrast_delta", "coolwarm", contrast_vmin, contrast_vmax),
        ("Attack flux contrast", "attack_flux_contrast_delta", "coolwarm", contrast_vmin, contrast_vmax),
        ("Weighted final abs error", "weighted_teacher_final_abs_error", "viridis", error_vmin, error_vmax),
        ("Low-energy sustain mask", "low_energy_sustain_mask", "viridis", 0.0, 1.0),
        ("Low-energy onset excluded", "low_energy_onset_excluded", "viridis", 0.0, 1.0),
        ("Low-energy weighted over", "low_energy_weighted_over", "viridis", None, None),
        ("Teacher mask", "mask", "viridis", mask_vmin, mask_vmax),
        ("Teacher raw residual", "raw_residual", "coolwarm", residual_vmin, residual_vmax),
    ]

    ncols = 4
    nrows = math.ceil(len(specs) / ncols) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), constrained_layout=True)
    fig.suptitle(title)
    axes_flat = list(np.atleast_1d(axes).flat)

    wave_ax = axes_flat[0]
    t = np.arange(audio_vec(result["guitar_audio"]).shape[-1]) / SAMPLE_RATE
    wave_ax.plot(t, audio_vec(result["guitar_audio"]), label="Guitar", alpha=0.55)
    wave_ax.plot(t, audio_vec(result["piano_audio"]), label="Target piano", alpha=0.8)
    wave_ax.plot(t, audio_vec(result["piano_mag_guitar_phase_audio"]), label="Piano mag + guitar phase", alpha=0.8)
    wave_ax.plot(t, audio_vec(result["oracle_intended_target_phase_audio"]), label="Intended + piano phase", alpha=0.8)
    wave_ax.plot(t, audio_vec(result["oracle_intended_guitar_phase_audio"]), label="Intended + guitar phase", alpha=0.8)
    wave_ax.plot(t, audio_vec(result["oracle_intended_phase_tcn_phase_audio"]), label="Intended + phase TCN phase", alpha=0.8)
    wave_ax.plot(t, audio_vec(result["teacher_before_phase_tcn_audio"]), label="Teacher before phase TCN", alpha=0.8)
    wave_ax.plot(t, audio_vec(result["teacher_before_transient_audio"]), label="Teacher before transient", alpha=0.8)
    wave_ax.plot(t, audio_vec(result["teacher_final_audio"]), label="Teacher after transient", alpha=0.8)
    wave_ax.set_title("Waveform overlay")
    wave_ax.set_xlabel("Seconds in frame")
    wave_ax.set_ylabel("Amplitude")
    wave_ax.legend(loc="upper right", fontsize="small")

    delta_ax = axes_flat[1]
    delta_ax.plot(t, audio_vec(result["phase_tcn_waveform_delta"]), label="Phase TCN delta", color="purple")
    delta_ax.set_title("Phase TCN waveform delta")
    delta_ax.set_xlabel("Seconds in frame")
    delta_ax.set_ylabel("Amplitude")
    delta_ax.legend(loc="upper right", fontsize="small")

    transient_ax = axes_flat[2]
    transient_ax.plot(t, audio_vec(result["transient_correction_delta"]), label="Transient correction", color="darkred")
    transient_ax.set_title("Transient correction delta")
    transient_ax.set_xlabel("Seconds in frame")
    transient_ax.set_ylabel("Amplitude")
    transient_ax.legend(loc="upper right", fontsize="small")
    for ax in axes_flat[3:ncols]:
        ax.axis("off")

    for ax, (label, key, cmap, vmin, vmax) in zip(axes_flat[ncols:], specs):
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
    for ax in axes_flat[ncols + len(specs):]:
        ax.axis("off")

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_metrics(result: dict, out_txt: Path, args, frame_index: int, start_sample: int, criterion):
    def scalar(value: torch.Tensor) -> float:
        return float(value.item())

    def add_range(metrics: dict[str, float], name: str, value: torch.Tensor):
        metrics[f"{name}_min"] = float(value.min().item())
        metrics[f"{name}_max"] = float(value.max().item())
        metrics[f"{name}_mean"] = float(value.mean().item())

    def add_optional_stats(metrics: dict[str, float], prefix: str, value: torch.Tensor):
        if value.numel() == 0:
            metrics[f"{prefix}_mean"] = float("nan")
            metrics[f"{prefix}_min"] = float("nan")
            metrics[f"{prefix}_max"] = float("nan")
            return
        metrics[f"{prefix}_mean"] = float(value.mean().item())
        metrics[f"{prefix}_min"] = float(value.min().item())
        metrics[f"{prefix}_max"] = float(value.max().item())

    def safe_mean(value: torch.Tensor) -> float:
        if value.numel() == 0:
            return float("nan")
        return float(value.mean().item())

    def add_attack_flux_band_metrics(
        metrics: dict[str, float],
        name: str,
        low_hz: float,
        high_hz: float | None,
        bin_hz: torch.Tensor,
        attack_flux_time_mask: torch.Tensor,
        teacher_flux: torch.Tensor,
        piano_flux: torch.Tensor,
        guitar_flux: torch.Tensor,
    ):
        if high_hz is None:
            freq_mask = bin_hz >= low_hz
        else:
            freq_mask = (bin_hz >= low_hz) & (bin_hz < high_hz)
        band_mask = freq_mask.view(1, 1, -1, 1) & attack_flux_time_mask
        full_mask = band_mask.expand_as(teacher_flux)
        teacher_piano = torch.abs(teacher_flux - piano_flux).masked_select(full_mask)
        teacher_guitar = torch.abs(teacher_flux - guitar_flux).masked_select(full_mask)

        metrics[f"attack_flux_{name}_teacher_piano_l1"] = safe_mean(teacher_piano)
        metrics[f"attack_flux_{name}_teacher_guitar_l1"] = safe_mean(teacher_guitar)
        if teacher_piano.numel() and teacher_guitar.numel():
            d_piano = teacher_piano.mean()
            d_guitar = teacher_guitar.mean()
            metrics[f"attack_flux_{name}_contrast"] = scalar(
                torch.relu(d_piano - d_guitar + d_piano.new_tensor(args.debug_attack_contrast_margin))
            )
            metrics[f"attack_flux_{name}_closer_to_piano"] = float(bool(d_piano < d_guitar))
        else:
            metrics[f"attack_flux_{name}_contrast"] = float("nan")
            metrics[f"attack_flux_{name}_closer_to_piano"] = float("nan")

    def sample_window(audio: torch.Tensor, start: int, end: int) -> torch.Tensor:
        length = audio.shape[-1]
        start = max(0, min(int(start), length))
        end = max(start, min(int(end), length))
        return audio[..., start:end]

    def stft_time_mask(start: int, end: int, num_frames: int, device: torch.device) -> torch.Tensor:
        centers = torch.arange(num_frames, device=device) * int(args.hop_size)
        return (centers >= start) & (centers < end)

    input_log = result["input_log"]
    piano_log_mag = result["piano_log_mag"]
    teacher_log_mag = result["teacher_log_mag"]
    teacher_intended_log_mag = result["teacher_intended_log_mag"]
    piano_mag_guitar_phase_log_mag = result["piano_mag_guitar_phase_log_mag"]
    oracle_intended_target_phase_log_mag = result["oracle_intended_target_phase_log_mag"]
    oracle_intended_guitar_phase_log_mag = result["oracle_intended_guitar_phase_log_mag"]
    oracle_intended_phase_tcn_phase_log_mag = result["oracle_intended_phase_tcn_phase_log_mag"]
    teacher_final_log_mag = result["teacher_final_log_mag"]
    teacher_residual = result["teacher_residual"]
    teacher_final_residual = result["teacher_final_residual"]
    piano_residual = result["piano_residual"]
    guitar_recon_audio = result["guitar_recon_audio"]
    piano_recon_audio = result["piano_recon_audio"]
    piano_mag_guitar_phase_audio = result["piano_mag_guitar_phase_audio"]
    oracle_intended_target_phase_audio = result["oracle_intended_target_phase_audio"]
    oracle_intended_guitar_phase_audio = result["oracle_intended_guitar_phase_audio"]
    oracle_intended_phase_tcn_phase_audio = result["oracle_intended_phase_tcn_phase_audio"]
    teacher_pre_audio = result["teacher_before_phase_tcn_audio"]
    teacher_before_transient_audio = result["teacher_before_transient_audio"]
    teacher_final_audio = result["teacher_final_audio"]
    phase_tcn_waveform_delta = result["phase_tcn_waveform_delta"]
    transient_correction_delta = result["transient_correction_delta"]
    after_vs_before_waveform_delta = result["after_vs_before_waveform_delta"]
    guitar_audio = result["guitar_audio"]
    piano_audio = result["piano_audio"]
    hybrid_signed_error = result["piano_mag_guitar_phase_signed_error"]
    hybrid_abs_error = result["piano_mag_guitar_phase_abs_error"]
    final_signed_error = result["teacher_final_signed_error"]
    final_abs_error = result["teacher_final_abs_error"]
    weighted_final_abs_error = result["weighted_teacher_final_abs_error"]

    frame_len = int(teacher_final_audio.shape[-1])
    attack_end = int(round(args.debug_attack_ms * SAMPLE_RATE / 1000.0))
    sustain_start = int(round(args.debug_sustain_start_ms * SAMPLE_RATE / 1000.0))
    attack_teacher = sample_window(teacher_final_audio, 0, attack_end)
    attack_before_transient = sample_window(teacher_before_transient_audio, 0, attack_end)
    attack_piano = sample_window(piano_audio, 0, attack_end)
    attack_guitar = sample_window(guitar_audio, 0, attack_end)
    attack_teacher_vs_piano = F.l1_loss(attack_teacher, attack_piano) if attack_teacher.numel() else torch.tensor(float("nan"))
    attack_teacher_vs_guitar = F.l1_loss(attack_teacher, attack_guitar) if attack_teacher.numel() else torch.tensor(float("nan"))
    attack_before_transient_vs_piano = (
        F.l1_loss(attack_before_transient, attack_piano)
        if attack_before_transient.numel()
        else torch.tensor(float("nan"))
    )
    attack_before_transient_vs_guitar = (
        F.l1_loss(attack_before_transient, attack_guitar)
        if attack_before_transient.numel()
        else torch.tensor(float("nan"))
    )

    metrics = {
        "guitar_recon_vs_guitar_waveform_l1": scalar(F.l1_loss(guitar_recon_audio, guitar_audio)),
        "piano_recon_vs_piano_waveform_l1": scalar(F.l1_loss(piano_recon_audio, piano_audio)),
        "piano_mag_guitar_phase_vs_piano_log_mag_l1": scalar(
            F.l1_loss(piano_mag_guitar_phase_log_mag, piano_log_mag)
        ),
        "piano_mag_guitar_phase_vs_piano_waveform_l1": scalar(
            F.l1_loss(piano_mag_guitar_phase_audio, piano_audio)
        ),
        "piano_mag_guitar_phase_combined_loss": scalar(criterion(piano_mag_guitar_phase_audio, piano_audio)),
        "oracle_intended_target_phase_vs_piano_log_mag_l1": scalar(
            F.l1_loss(oracle_intended_target_phase_log_mag, piano_log_mag)
        ),
        "oracle_intended_target_phase_vs_piano_waveform_l1": scalar(
            F.l1_loss(oracle_intended_target_phase_audio, piano_audio)
        ),
        "oracle_intended_target_phase_combined_loss": scalar(
            criterion(oracle_intended_target_phase_audio, piano_audio)
        ),
        "oracle_intended_guitar_phase_vs_piano_log_mag_l1": scalar(
            F.l1_loss(oracle_intended_guitar_phase_log_mag, piano_log_mag)
        ),
        "oracle_intended_guitar_phase_vs_piano_waveform_l1": scalar(
            F.l1_loss(oracle_intended_guitar_phase_audio, piano_audio)
        ),
        "oracle_intended_guitar_phase_combined_loss": scalar(
            criterion(oracle_intended_guitar_phase_audio, piano_audio)
        ),
        "oracle_intended_phase_tcn_phase_vs_piano_log_mag_l1": scalar(
            F.l1_loss(oracle_intended_phase_tcn_phase_log_mag, piano_log_mag)
        ),
        "oracle_intended_phase_tcn_phase_vs_piano_waveform_l1": scalar(
            F.l1_loss(oracle_intended_phase_tcn_phase_audio, piano_audio)
        ),
        "oracle_intended_phase_tcn_phase_combined_loss": scalar(
            criterion(oracle_intended_phase_tcn_phase_audio, piano_audio)
        ),
        "oracle_target_vs_guitar_phase_log_mag_l1": scalar(
            F.l1_loss(oracle_intended_target_phase_log_mag, oracle_intended_guitar_phase_log_mag)
        ),
        "oracle_phase_tcn_vs_guitar_phase_log_mag_l1": scalar(
            F.l1_loss(oracle_intended_phase_tcn_phase_log_mag, oracle_intended_guitar_phase_log_mag)
        ),
        "oracle_phase_tcn_vs_target_phase_log_mag_l1": scalar(
            F.l1_loss(oracle_intended_phase_tcn_phase_log_mag, oracle_intended_target_phase_log_mag)
        ),
        "teacher_pre_vs_piano_log_mag_l1": scalar(F.l1_loss(teacher_log_mag, piano_log_mag)),
        "teacher_final_vs_piano_log_mag_l1": scalar(F.l1_loss(teacher_final_log_mag, piano_log_mag)),
        "teacher_pre_vs_piano_residual_l1": scalar(F.l1_loss(teacher_residual, piano_residual)),
        "teacher_final_vs_piano_residual_l1": scalar(F.l1_loss(teacher_final_residual, piano_residual)),
        "teacher_pre_vs_piano_waveform_l1": scalar(F.l1_loss(teacher_pre_audio, piano_audio)),
        "teacher_final_vs_piano_waveform_l1": scalar(F.l1_loss(teacher_final_audio, piano_audio)),
        "teacher_pre_combined_loss": scalar(criterion(teacher_pre_audio, piano_audio)),
        "teacher_final_combined_loss": scalar(criterion(teacher_final_audio, piano_audio)),
        "teacher_before_phase_tcn_vs_piano_waveform_l1": scalar(F.l1_loss(teacher_pre_audio, piano_audio)),
        "teacher_before_transient_vs_piano_waveform_l1": scalar(
            F.l1_loss(teacher_before_transient_audio, piano_audio)
        ),
        "teacher_after_transient_vs_piano_waveform_l1": scalar(F.l1_loss(teacher_final_audio, piano_audio)),
        "teacher_after_phase_tcn_vs_piano_waveform_l1": scalar(
            F.l1_loss(teacher_before_transient_audio, piano_audio)
        ),
        "teacher_before_phase_tcn_combined_loss": scalar(criterion(teacher_pre_audio, piano_audio)),
        "teacher_before_transient_combined_loss": scalar(criterion(teacher_before_transient_audio, piano_audio)),
        "teacher_after_transient_combined_loss": scalar(criterion(teacher_final_audio, piano_audio)),
        "teacher_after_phase_tcn_combined_loss": scalar(criterion(teacher_before_transient_audio, piano_audio)),
        "teacher_pre_vs_final_log_mag_l1": scalar(F.l1_loss(teacher_log_mag, teacher_final_log_mag)),
        "intended_vs_before_phase_tcn_rendered_log_mag_l1": scalar(
            F.l1_loss(teacher_intended_log_mag, teacher_log_mag)
        ),
        "phase_tcn_waveform_l1": scalar(F.l1_loss(teacher_before_transient_audio, teacher_pre_audio)),
        "transient_correction_waveform_l1": scalar(
            F.l1_loss(teacher_final_audio, teacher_before_transient_audio)
        ),
        "after_vs_before_waveform_l1": scalar(F.l1_loss(teacher_final_audio, teacher_pre_audio)),
        "after_vs_before_log_mag_l1": scalar(F.l1_loss(teacher_final_log_mag, teacher_log_mag)),
        "mean_absolute_error": float(final_abs_error.mean().item()),
        "max_absolute_error": float(final_abs_error.max().item()),
        "weighted_final_log_mag_l1": float(weighted_final_abs_error.mean().item()),
        "phase_delta_mean": float(result["phase_delta"].mean().item()),
        "phase_delta_abs_mean": float(result["phase_delta_abs"].mean().item()),
        "phase_delta_abs_max": float(result["phase_delta_abs"].max().item()),
        "phase_delta_saturation_frac": float(
            (result["phase_delta_abs"] >= args.phase_saturation_threshold).float().mean().item()
        ),
        "attack_waveform_l1": scalar(attack_teacher_vs_piano),
        "attack_envelope_l1": scalar(F.l1_loss(torch.abs(attack_teacher), torch.abs(attack_piano)))
        if attack_teacher.numel()
        else float("nan"),
        "attack_before_transient_waveform_l1": scalar(attack_before_transient_vs_piano),
        "attack_after_transient_waveform_l1": scalar(attack_teacher_vs_piano),
        "attack_before_transient_envelope_l1": scalar(
            F.l1_loss(torch.abs(attack_before_transient), torch.abs(attack_piano))
        )
        if attack_before_transient.numel()
        else float("nan"),
        "attack_after_transient_envelope_l1": scalar(F.l1_loss(torch.abs(attack_teacher), torch.abs(attack_piano)))
        if attack_teacher.numel()
        else float("nan"),
        "attack_teacher_vs_guitar_waveform_l1": scalar(attack_teacher_vs_guitar),
        "attack_teacher_vs_piano_waveform_l1": scalar(attack_teacher_vs_piano),
        "before_transient_attack_closer_to_guitar_than_piano": float(
            bool(attack_before_transient_vs_guitar < attack_before_transient_vs_piano)
        )
        if attack_before_transient.numel()
        else float("nan"),
        "after_transient_attack_closer_to_guitar_than_piano": float(
            bool(attack_teacher_vs_guitar < attack_teacher_vs_piano)
        )
        if attack_teacher.numel()
        else float("nan"),
        "teacher_attack_closer_to_guitar_than_piano": float(
            bool(attack_teacher_vs_guitar < attack_teacher_vs_piano)
        )
        if attack_teacher.numel()
        else float("nan"),
    }
    for name in (
        "input_log",
        "piano_log_mag",
        "piano_mag_guitar_phase_log_mag",
        "teacher_intended_log_mag",
        "oracle_intended_target_phase_log_mag",
        "oracle_intended_guitar_phase_log_mag",
        "oracle_intended_phase_tcn_phase_log_mag",
        "teacher_log_mag",
        "teacher_before_transient_log_mag",
        "teacher_final_log_mag",
        "piano_residual",
        "teacher_intended_residual",
        "oracle_intended_target_phase_residual",
        "oracle_intended_guitar_phase_residual",
        "oracle_intended_phase_tcn_phase_residual",
        "teacher_residual",
        "teacher_before_transient_residual",
        "teacher_final_residual",
        "phase_tcn_log_mag_delta",
        "transient_correction_log_mag_delta",
        "output_delta_log_mag",
        "piano_mag_guitar_phase_signed_error",
        "oracle_intended_target_phase_signed_error",
        "oracle_intended_guitar_phase_signed_error",
        "oracle_intended_phase_tcn_phase_signed_error",
        "mask",
        "raw_residual",
        "phase_delta",
        "phase_delta_abs",
        "phase_delta_raw",
        "attack_logmag_contrast_delta",
        "attack_flux_contrast_delta",
    ):
        add_range(metrics, name, result[name])

    for name in (
        "guitar_audio",
        "piano_audio",
        "guitar_recon_audio",
        "piano_recon_audio",
        "piano_mag_guitar_phase_audio",
        "oracle_intended_target_phase_audio",
        "oracle_intended_guitar_phase_audio",
        "oracle_intended_phase_tcn_phase_audio",
        "teacher_before_phase_tcn_audio",
        "teacher_before_transient_audio",
        "teacher_after_phase_tcn_audio",
        "teacher_after_transient_audio",
        "phase_tcn_waveform_delta",
        "transient_correction_delta",
        "after_vs_before_waveform_delta",
    ):
        for stat_name, stat_value in audio_stats(result[name]).items():
            metrics[f"{name}_{stat_name}"] = stat_value

    freq_bins = result["valid_shape"][0]
    bin_hz = torch.arange(
        freq_bins,
        device=teacher_final_log_mag.device,
        dtype=teacher_final_log_mag.dtype,
    ) * (SAMPLE_RATE / float(args.n_fft))
    hf_mask = (bin_hz >= args.debug_hf_start_hz).view(1, 1, freq_bins, 1)
    phase_delta = result["phase_delta"]
    if phase_delta.shape[-1] > 1:
        metrics["phase_delta_dt_l1"] = float(torch.abs(phase_delta[..., 1:] - phase_delta[..., :-1]).mean().item())
    else:
        metrics["phase_delta_dt_l1"] = float("nan")
    if phase_delta.shape[-2] > 1:
        metrics["phase_delta_df_l1"] = float(torch.abs(phase_delta[..., 1:, :] - phase_delta[..., :-1, :]).mean().item())
    else:
        metrics["phase_delta_df_l1"] = float("nan")
    phase_delta_hf = result["phase_delta_abs"].masked_select(hf_mask.expand_as(result["phase_delta_abs"]))
    metrics["phase_delta_high_band_abs_mean"] = safe_mean(phase_delta_hf)

    hf_hybrid_error = hybrid_signed_error.masked_select(hf_mask.expand_as(hybrid_signed_error))
    hf_hybrid_abs_error = hybrid_abs_error.masked_select(hf_mask.expand_as(hybrid_abs_error))
    hf_final_error = final_signed_error.masked_select(hf_mask.expand_as(final_signed_error))
    hf_final_abs_error = final_abs_error.masked_select(hf_mask.expand_as(final_abs_error))
    add_optional_stats(metrics, "high_freq_piano_mag_guitar_phase_signed_error", hf_hybrid_error)
    add_optional_stats(metrics, "high_freq_piano_mag_guitar_phase_abs_error", hf_hybrid_abs_error)
    add_optional_stats(metrics, "high_freq_teacher_final_signed_error", hf_final_error)
    add_optional_stats(metrics, "high_freq_teacher_final_abs_error", hf_final_abs_error)

    time_frames = teacher_final_log_mag.shape[-1]
    attack_time_mask = stft_time_mask(0, attack_end, time_frames, teacher_final_log_mag.device).view(1, 1, 1, time_frames)
    sustain_time_mask = stft_time_mask(sustain_start, frame_len, time_frames, teacher_final_log_mag.device).view(
        1, 1, 1, time_frames
    )
    hf_over = torch.relu(teacher_final_log_mag - piano_log_mag)
    attack_hf_mask = hf_mask & attack_time_mask
    sustain_hf_mask = hf_mask & sustain_time_mask
    metrics["attack_hf_over_mean"] = safe_mean(hf_over.masked_select(attack_hf_mask.expand_as(hf_over)))
    metrics["sustain_hf_over_mean"] = safe_mean(hf_over.masked_select(sustain_hf_mask.expand_as(hf_over)))

    before_phase_hf_mag = torch.exp(teacher_log_mag).masked_select(sustain_hf_mask.expand_as(teacher_log_mag))
    after_phase_hf_mag = torch.exp(result["teacher_before_transient_log_mag"]).masked_select(
        sustain_hf_mask.expand_as(result["teacher_before_transient_log_mag"])
    )
    piano_sustain_hf_mag = torch.exp(piano_log_mag).masked_select(sustain_hf_mask.expand_as(piano_log_mag))
    if piano_sustain_hf_mag.numel():
        denom = piano_sustain_hf_mag.mean().clamp_min(1.0e-8)
        metrics["before_phase_tcn_sustain_hf_noise_floor_ratio"] = float(
            (before_phase_hf_mag.mean() / denom).item()
        ) if before_phase_hf_mag.numel() else float("nan")
        metrics["after_phase_tcn_sustain_hf_noise_floor_ratio"] = float(
            (after_phase_hf_mag.mean() / denom).item()
        ) if after_phase_hf_mag.numel() else float("nan")
    else:
        metrics["before_phase_tcn_sustain_hf_noise_floor_ratio"] = float("nan")
        metrics["after_phase_tcn_sustain_hf_noise_floor_ratio"] = float("nan")

    full_attack_mask = attack_time_mask.expand_as(teacher_final_log_mag)
    teacher_piano_attack_logmag = torch.abs(teacher_final_log_mag - piano_log_mag).masked_select(full_attack_mask)
    teacher_guitar_attack_logmag = torch.abs(teacher_final_log_mag - input_log).masked_select(full_attack_mask)
    metrics["attack_teacher_piano_logmag_l1"] = safe_mean(teacher_piano_attack_logmag)
    metrics["attack_teacher_guitar_logmag_l1"] = safe_mean(teacher_guitar_attack_logmag)
    if teacher_piano_attack_logmag.numel() and teacher_guitar_attack_logmag.numel():
        d_piano_log = teacher_piano_attack_logmag.mean()
        d_guitar_log = teacher_guitar_attack_logmag.mean()
        metrics["attack_contrast_logmag"] = scalar(
            torch.relu(d_piano_log - d_guitar_log + d_piano_log.new_tensor(args.debug_attack_contrast_margin))
        )
        metrics["attack_closer_to_piano_logmag"] = float(bool(d_piano_log < d_guitar_log))
    else:
        metrics["attack_contrast_logmag"] = float("nan")
        metrics["attack_closer_to_piano_logmag"] = float("nan")

    teacher_flux = torch.relu(teacher_final_log_mag[..., 1:] - teacher_final_log_mag[..., :-1])
    teacher_before_transient_flux = torch.relu(
        result["teacher_before_transient_log_mag"][..., 1:] - result["teacher_before_transient_log_mag"][..., :-1]
    )
    piano_flux = torch.relu(piano_log_mag[..., 1:] - piano_log_mag[..., :-1])
    guitar_flux = torch.relu(input_log[..., 1:] - input_log[..., :-1])
    flux_frames = teacher_flux.shape[-1]
    attack_flux_time_mask = stft_time_mask(
        args.hop_size,
        attack_end,
        flux_frames,
        teacher_final_log_mag.device,
    ).view(1, 1, 1, flux_frames)
    hf_flux_mask = (bin_hz >= args.debug_hf_start_hz).view(1, 1, freq_bins, 1) & attack_flux_time_mask
    metrics["attack_hf_flux_l1"] = safe_mean(
        torch.abs(teacher_flux - piano_flux).masked_select(hf_flux_mask.expand_as(teacher_flux))
    )
    metrics["attack_before_transient_hf_flux_l1"] = safe_mean(
        torch.abs(teacher_before_transient_flux - piano_flux).masked_select(
            hf_flux_mask.expand_as(teacher_before_transient_flux)
        )
    )
    metrics["attack_after_transient_hf_flux_l1"] = metrics["attack_hf_flux_l1"]
    full_attack_flux_mask = attack_flux_time_mask.expand_as(teacher_flux)
    teacher_piano_attack_flux = torch.abs(teacher_flux - piano_flux).masked_select(full_attack_flux_mask)
    teacher_guitar_attack_flux = torch.abs(teacher_flux - guitar_flux).masked_select(full_attack_flux_mask)
    metrics["attack_teacher_piano_flux_l1"] = safe_mean(teacher_piano_attack_flux)
    metrics["attack_teacher_guitar_flux_l1"] = safe_mean(teacher_guitar_attack_flux)
    if teacher_piano_attack_flux.numel() and teacher_guitar_attack_flux.numel():
        d_piano_flux = teacher_piano_attack_flux.mean()
        d_guitar_flux = teacher_guitar_attack_flux.mean()
        metrics["attack_contrast_flux"] = scalar(
            torch.relu(d_piano_flux - d_guitar_flux + d_piano_flux.new_tensor(args.debug_attack_contrast_margin))
        )
        metrics["attack_closer_to_piano_flux"] = float(bool(d_piano_flux < d_guitar_flux))
    else:
        metrics["attack_contrast_flux"] = float("nan")
        metrics["attack_closer_to_piano_flux"] = float("nan")

    add_attack_flux_band_metrics(
        metrics,
        "low_0_2k",
        0.0,
        2000.0,
        bin_hz,
        attack_flux_time_mask,
        teacher_flux,
        piano_flux,
        guitar_flux,
    )
    add_attack_flux_band_metrics(
        metrics,
        "mid_2k_8k",
        2000.0,
        8000.0,
        bin_hz,
        attack_flux_time_mask,
        teacher_flux,
        piano_flux,
        guitar_flux,
    )
    add_attack_flux_band_metrics(
        metrics,
        "high_8k_up",
        8000.0,
        None,
        bin_hz,
        attack_flux_time_mask,
        teacher_flux,
        piano_flux,
        guitar_flux,
    )

    teacher_hf_mag = torch.exp(teacher_final_log_mag).masked_select(sustain_hf_mask.expand_as(teacher_final_log_mag))
    piano_hf_mag = torch.exp(piano_log_mag).masked_select(sustain_hf_mask.expand_as(piano_log_mag))
    if teacher_hf_mag.numel() and piano_hf_mag.numel():
        metrics["sustain_hf_noise_floor_ratio"] = float(
            (teacher_hf_mag.mean() / piano_hf_mag.mean().clamp_min(1.0e-8)).item()
        )
    else:
        metrics["sustain_hf_noise_floor_ratio"] = float("nan")

    piano_abs = piano_audio.abs()
    low_energy_sample_threshold = torch.quantile(piano_abs.reshape(-1), 0.25)
    low_energy_sample_mask = piano_abs <= low_energy_sample_threshold
    low_energy_output = teacher_final_audio.masked_select(low_energy_sample_mask)
    low_energy_piano = piano_audio.masked_select(low_energy_sample_mask)
    if low_energy_output.numel() and low_energy_piano.numel():
        output_rms = torch.sqrt(torch.mean(low_energy_output.square()) + 1.0e-12)
        piano_rms = torch.sqrt(torch.mean(low_energy_piano.square()) + 1.0e-12)
        metrics["low_energy_output_rms"] = float(output_rms.item())
        metrics["low_energy_output_vs_piano_rms_ratio"] = float((output_rms / piano_rms.clamp_min(1.0e-8)).item())
    else:
        metrics["low_energy_output_rms"] = float("nan")
        metrics["low_energy_output_vs_piano_rms_ratio"] = float("nan")

    piano_frame_energy = torch.exp(piano_log_mag).mean(dim=-2, keepdim=True)
    low_energy_stft_threshold = torch.quantile(piano_frame_energy.reshape(-1), 0.25)
    low_energy_time_mask = piano_frame_energy <= low_energy_stft_threshold
    low_energy_hf_mask = hf_mask & low_energy_time_mask
    low_energy_teacher_hf = torch.exp(teacher_final_log_mag).masked_select(
        low_energy_hf_mask.expand_as(teacher_final_log_mag)
    )
    low_energy_piano_hf = torch.exp(piano_log_mag).masked_select(low_energy_hf_mask.expand_as(piano_log_mag))
    if low_energy_teacher_hf.numel() and low_energy_piano_hf.numel():
        metrics["low_energy_hf_mag_ratio"] = float(
            (low_energy_teacher_hf.mean() / low_energy_piano_hf.mean().clamp_min(1.0e-8)).item()
        )
    else:
        metrics["low_energy_hf_mag_ratio"] = float("nan")

    low_energy_quantile = max(0.0, min(1.0, float(args.low_energy_spectral_quantile)))
    low_energy_bin_threshold = torch.quantile(torch.exp(piano_log_mag).reshape(-1), low_energy_quantile)
    low_energy_bin_mask = torch.exp(piano_log_mag) <= low_energy_bin_threshold
    low_energy_over = torch.relu(teacher_final_log_mag - piano_log_mag - args.low_energy_spectral_margin)
    if low_energy_bin_mask.any():
        metrics["low_energy_spectral_over_mean"] = float(
            low_energy_over.masked_select(low_energy_bin_mask).mean().item()
        )
        metrics["low_energy_spectral_active_frac"] = float(low_energy_bin_mask.float().mean().item())
    else:
        metrics["low_energy_spectral_over_mean"] = float("nan")
        metrics["low_energy_spectral_active_frac"] = float("nan")

    low_energy_bin_hf_mask = low_energy_bin_mask & hf_mask.expand_as(low_energy_bin_mask)
    if low_energy_bin_hf_mask.any():
        metrics["low_energy_spectral_hf_over_mean"] = float(
            low_energy_over.masked_select(low_energy_bin_hf_mask).mean().item()
        )
    else:
        metrics["low_energy_spectral_hf_over_mean"] = float("nan")

    onset_excluded = low_energy_onset_exclusion_mask(piano_log_mag, args)
    sustain_low_energy_mask = low_energy_bin_mask & (~onset_excluded).expand_as(low_energy_bin_mask)
    harmonic_protected = low_energy_harmonic_protection(piano_log_mag, args)
    protected_margin = harmonic_protected.to(piano_log_mag.dtype) * float(args.low_energy_harmonic_peak_margin)
    sustain_over = torch.relu(teacher_final_log_mag - piano_log_mag - args.low_energy_spectral_margin - protected_margin)
    band_weights = low_energy_band_weight_map(bin_hz, args)
    weighted_sustain_over = sustain_over * band_weights

    metrics["low_energy_sustain_active_frac"] = float(sustain_low_energy_mask.float().mean().item())
    metrics["low_energy_onset_excluded_frac"] = float(onset_excluded.float().mean().item())
    metrics["low_energy_harmonic_protected_frac"] = float(harmonic_protected.float().mean().item())
    if sustain_low_energy_mask.any():
        metrics["low_energy_sustain_spectral_over_mean"] = float(
            weighted_sustain_over.masked_select(sustain_low_energy_mask).mean().item()
        )
    else:
        metrics["low_energy_sustain_spectral_over_mean"] = float("nan")

    def add_sustain_band_metric(name: str, low_hz: float, high_hz: float | None):
        if high_hz is None:
            freq_mask = bin_hz >= low_hz
        else:
            freq_mask = (bin_hz >= low_hz) & (bin_hz < high_hz)
        mask = sustain_low_energy_mask & freq_mask.view(1, 1, -1, 1)
        metrics[f"low_energy_sustain_spectral_{name}_over_mean"] = (
            float(weighted_sustain_over.masked_select(mask.expand_as(weighted_sustain_over)).mean().item())
            if mask.any()
            else float("nan")
        )

    add_sustain_band_metric("low", 0.0, 500.0)
    add_sustain_band_metric("low_mid", 500.0, 2000.0)
    add_sustain_band_metric("mid", 2000.0, 8000.0)
    add_sustain_band_metric("high", 8000.0, None)

    stage_log_mags = {
        "intended": teacher_intended_log_mag,
        "oracle_target_phase": result["oracle_intended_target_phase_log_mag"],
        "oracle_guitar_phase": result["oracle_intended_guitar_phase_log_mag"],
        "oracle_phase_tcn_phase": result["oracle_intended_phase_tcn_phase_log_mag"],
        "before_phase": teacher_log_mag,
        "before_transient": result["teacher_before_transient_log_mag"],
        "final": teacher_final_log_mag,
    }
    artifact_sustain_mask = (~onset_excluded).expand_as(piano_log_mag)
    harmonic_region = artifact_harmonic_region(
        piano_log_mag,
        args.artifact_peak_prominence,
        args.artifact_peak_radius_bins,
    )
    interharmonic_mask = artifact_sustain_mask & (~harmonic_region)
    band_freq_masks = artifact_band_masks(bin_hz)
    artifact_band_masks_full = {
        name: artifact_sustain_mask & freq_mask.view(1, 1, -1, 1)
        for name, freq_mask in band_freq_masks.items()
    }
    interharmonic_band_masks = {
        name: interharmonic_mask & freq_mask.view(1, 1, -1, 1)
        for name, freq_mask in band_freq_masks.items()
    }
    interharmonic_over_by_stage = {
        stage: torch.relu(stage_log - piano_log_mag - float(args.low_energy_spectral_margin))
        for stage, stage_log in stage_log_mags.items()
    }
    add_stage_band_metrics(
        metrics,
        "interharmonic_sustain_{stage}_{band}_over_mean",
        interharmonic_over_by_stage,
        interharmonic_band_masks,
    )

    high_energy_quantile = max(0.0, min(1.0, float(args.high_energy_interharmonic_quantile)))
    piano_time_energy = torch.exp(piano_log_mag).mean(dim=-2, keepdim=True)
    high_energy_threshold = torch.quantile(
        piano_time_energy.detach().flatten(1),
        high_energy_quantile,
        dim=1,
    ).view(-1, 1, 1)
    high_energy_time_mask = piano_time_energy >= high_energy_threshold
    high_energy_harmonic_region = artifact_harmonic_region(
        piano_log_mag,
        args.high_energy_interharmonic_peak_prominence,
        args.high_energy_interharmonic_peak_radius_bins,
    )
    high_energy_sustain_mask = artifact_sustain_mask & high_energy_time_mask
    high_energy_interharmonic_mask = high_energy_sustain_mask & (~high_energy_harmonic_region)
    high_energy_band_masks = {
        name: high_energy_interharmonic_mask & freq_mask.view(1, 1, -1, 1)
        for name, freq_mask in band_freq_masks.items()
    }
    high_energy_over_by_stage = {
        stage: torch.relu(stage_log - piano_log_mag - float(args.high_energy_interharmonic_margin))
        for stage, stage_log in stage_log_mags.items()
    }
    metrics["high_energy_sustain_active_frac"] = float(high_energy_sustain_mask.float().mean().item())
    metrics["high_energy_interharmonic_sustain_active_frac"] = float(
        high_energy_interharmonic_mask.float().mean().item()
    )
    add_stage_band_metrics(
        metrics,
        "high_energy_interharmonic_sustain_{stage}_{band}_over_mean",
        high_energy_over_by_stage,
        high_energy_band_masks,
    )

    if piano_log_mag.shape[-1] > 1:
        shimmer_sustain_mask = artifact_sustain_mask[..., 1:]
        shimmer_band_masks = {
            name: shimmer_sustain_mask & freq_mask.view(1, 1, -1, 1)
            for name, freq_mask in band_freq_masks.items()
        }
        piano_dt = torch.abs(piano_log_mag[..., 1:] - piano_log_mag[..., :-1])
        shimmer_by_stage = {
            stage: torch.relu(
                torch.abs(stage_log[..., 1:] - stage_log[..., :-1])
                - piano_dt
                - float(args.artifact_shimmer_margin)
            )
            for stage, stage_log in stage_log_mags.items()
        }
        add_stage_band_metrics(
            metrics,
            "sustain_shimmer_{stage}_{band}_excess_mean",
            shimmer_by_stage,
            shimmer_band_masks,
        )
    else:
        for stage in stage_log_mags:
            for band in band_freq_masks:
                metrics[f"sustain_shimmer_{stage}_{band}_excess_mean"] = float("nan")

    def flatness_by_time(log_mag: torch.Tensor, freq_mask: torch.Tensor) -> torch.Tensor | None:
        if not freq_mask.any():
            return None
        mag = torch.exp(log_mag).clamp_min(1.0e-8)
        mag_band = mag[:, :, freq_mask, :]
        return torch.exp(torch.log(mag_band).mean(dim=-2, keepdim=True)) / mag_band.mean(
            dim=-2,
            keepdim=True,
        ).clamp_min(1.0e-8)

    sustain_time_only_mask = artifact_sustain_mask.any(dim=-2, keepdim=True)
    for band, freq_mask in band_freq_masks.items():
        piano_flatness = flatness_by_time(piano_log_mag, freq_mask)
        if piano_flatness is None:
            for stage in stage_log_mags:
                metrics[f"sustain_flatness_{stage}_{band}_ratio"] = float("nan")
            continue
        for stage, stage_log in stage_log_mags.items():
            stage_flatness = flatness_by_time(stage_log, freq_mask)
            ratio = stage_flatness / piano_flatness.clamp_min(1.0e-8)
            metrics[f"sustain_flatness_{stage}_{band}_ratio"] = safe_masked_mean(
                ratio,
                sustain_time_only_mask,
            )

    for stage, stage_log in stage_log_mags.items():
        residual_abs = torch.abs(stage_log - piano_log_mag)
        total_residual = residual_abs.masked_select(artifact_sustain_mask).mean() if artifact_sustain_mask.any() else None
        for band, mask in artifact_band_masks_full.items():
            if total_residual is None or not mask.any():
                metrics[f"residual_energy_{stage}_{band}_ratio"] = float("nan")
                continue
            band_residual = residual_abs.masked_select(mask).mean()
            metrics[f"residual_energy_{stage}_{band}_ratio"] = float(
                (band_residual / total_residual.clamp_min(1.0e-8)).item()
            )

    low_note_freq_mask = (bin_hz <= float(args.low_energy_low_note_threshold_hz)).view(1, 1, -1, 1)
    full_mag = torch.exp(piano_log_mag)
    low_note_energy = (full_mag * low_note_freq_mask.to(full_mag.dtype)).sum(dim=-2, keepdim=True)
    total_energy = full_mag.sum(dim=-2, keepdim=True).clamp_min(1.0e-8)
    low_note_ratio = low_note_energy / total_energy
    low_note_time_mask = low_note_ratio >= float(args.low_energy_low_note_ratio_threshold)
    metrics["low_energy_low_note_ratio"] = float(low_note_ratio.mean().item())
    metrics["low_energy_low_note_active_frac"] = float(low_note_time_mask.float().mean().item())
    low_note_sustain_mask = sustain_low_energy_mask & low_note_time_mask.expand_as(sustain_low_energy_mask)
    mid_mask = ((bin_hz >= 2000.0) & (bin_hz < 8000.0)).view(1, 1, -1, 1)
    high_mask = (bin_hz >= 8000.0).view(1, 1, -1, 1)
    low_note_mid = low_note_sustain_mask & mid_mask
    low_note_high = low_note_sustain_mask & high_mask
    metrics["low_energy_low_note_sustain_mid_over_mean"] = (
        float(weighted_sustain_over.masked_select(low_note_mid.expand_as(weighted_sustain_over)).mean().item())
        if low_note_mid.any()
        else float("nan")
    )
    metrics["low_energy_low_note_sustain_high_over_mean"] = (
        float(weighted_sustain_over.masked_select(low_note_high.expand_as(weighted_sustain_over)).mean().item())
        if low_note_high.any()
        else float("nan")
    )

    lines = [
        f"guitar_wav={args.guitar_wav}",
        f"piano_wav={args.piano_wav}",
        f"audio_source={'raw' if args.raw_audio else 'training_preprocessed'}",
        f"data_dir={args.data_dir}",
        f"max_shift_ms={args.max_shift_ms}",
        f"min_rms={args.min_rms}",
        f"keep_silence_prob={args.keep_silence_prob}",
        f"teacher_ckpt={args.teacher_ckpt}",
        f"frame_index={frame_index}",
        f"frame_start_sample={start_sample}",
        f"frame_start_time_sec={start_sample / SAMPLE_RATE:.8f}",
        f"valid_stft_shape={result['valid_shape'][0]}x{result['valid_shape'][1]}",
        f"debug_hf_start_hz={args.debug_hf_start_hz}",
        f"debug_attack_ms={args.debug_attack_ms}",
        f"debug_attack_contrast_margin={args.debug_attack_contrast_margin}",
        f"debug_sustain_start_ms={args.debug_sustain_start_ms}",
        "artifact_diagnostics_only=True",
        f"debug_low_energy_quantile={args.low_energy_spectral_quantile}",
        f"debug_low_energy_margin={args.low_energy_spectral_margin}",
        f"debug_onset_flux_std={args.low_energy_onset_flux_std}",
        f"debug_onset_pre_ms={args.low_energy_onset_pre_ms}",
        f"debug_onset_post_ms={args.low_energy_onset_post_ms}",
        f"debug_band_low_weight={args.low_energy_band_low_weight}",
        f"debug_band_low_mid_weight={args.low_energy_band_low_mid_weight}",
        f"debug_band_mid_weight={args.low_energy_band_mid_weight}",
        f"debug_band_high_weight={args.low_energy_band_high_weight}",
        f"debug_low_note_threshold_hz={args.low_energy_low_note_threshold_hz}",
        f"debug_low_note_ratio_threshold={args.low_energy_low_note_ratio_threshold}",
        f"debug_harmonic_protect={args.low_energy_harmonic_protect}",
        f"debug_harmonic_peak_margin={args.low_energy_harmonic_peak_margin}",
        f"debug_harmonic_peak_prominence={args.low_energy_harmonic_peak_prominence}",
        f"debug_high_energy_quantile={args.high_energy_interharmonic_quantile}",
        f"debug_high_energy_margin={args.high_energy_interharmonic_margin}",
        f"debug_high_energy_peak_prominence={args.high_energy_interharmonic_peak_prominence}",
        f"debug_high_energy_peak_radius_bins={args.high_energy_interharmonic_peak_radius_bins}",
        f"debug_high_energy_low_weight={args.high_energy_interharmonic_low_weight}",
        f"debug_high_energy_low_mid_weight={args.high_energy_interharmonic_low_mid_weight}",
        f"debug_high_energy_mid_weight={args.high_energy_interharmonic_mid_weight}",
        f"debug_high_energy_high_weight={args.high_energy_interharmonic_high_weight}",
        f"artifact_peak_prominence={args.artifact_peak_prominence}",
        f"artifact_peak_radius_bins={args.artifact_peak_radius_bins}",
        f"artifact_shimmer_margin={args.artifact_shimmer_margin}",
        f"frame_size={args.frame_size}",
        f"hop_size={args.hop_size}",
        f"n_fft={args.n_fft}",
        f"win_length={args.win_length}",
        f"hidden_size={args.hidden_size}",
        f"base_ch={args.base_ch}",
        f"phase_tcn_ch={args.phase_tcn_ch}",
        f"phase_tcn_layers={args.phase_tcn_layers}",
        f"phase_max_delta={args.phase_max_delta}",
        f"phase_saturation_threshold={args.phase_saturation_threshold}",
        f"log_floor={args.log_floor}",
        f"energy_weight_floor={args.energy_weight_floor}",
        f"energy_weight_ceiling={args.energy_weight_ceiling}",
        f"write_wavs={args.write_wavs}",
    ]
    lines.extend(f"{key}={value:.8f}" for key, value in metrics.items())
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_debug_wavs(result: dict, out_dir: Path, frame_index: int):
    wavs = {
        "guitar": result["guitar_audio"],
        "target_piano": result["piano_audio"],
        "guitar_recon": result["guitar_recon_audio"],
        "piano_recon": result["piano_recon_audio"],
        "piano_mag_guitar_phase": result["piano_mag_guitar_phase_audio"],
        "oracle_intended_target_phase": result["oracle_intended_target_phase_audio"],
        "oracle_intended_guitar_phase": result["oracle_intended_guitar_phase_audio"],
        "oracle_intended_phase_tcn_phase": result["oracle_intended_phase_tcn_phase_audio"],
        "teacher_before_phase_tcn": result["teacher_before_phase_tcn_audio"],
        "teacher_after_phase_tcn": result["teacher_after_phase_tcn_audio"],
        "teacher_before_transient": result["teacher_before_transient_audio"],
        "teacher_after_transient": result["teacher_final_audio"],
        "phase_tcn_delta": result["phase_tcn_waveform_delta"],
        "transient_correction_delta": result["transient_correction_delta"],
        "after_vs_before_phase_tcn_delta": result["phase_tcn_waveform_delta"],
        "total_model_delta": result["after_vs_before_waveform_delta"],
    }
    for name, audio in wavs.items():
        path = out_dir / f"frame_{frame_index:04d}_{name}.wav"
        torchaudio.save(path, audio.detach().cpu(), SAMPLE_RATE)


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
    teacher = load_teacher(args.teacher_ckpt, device, args)
    debugger = TeacherDebugFrame(teacher, args).to(device)
    criterion = CombinedLoss().to(device)

    guitar_audio, piano_audio, source_desc = load_debug_audio(args)
    frame_indices = parse_frame_indices(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Teacher: {args.teacher_ckpt} | frame={args.frame_size}, "
        f"hop={args.hop_size}, n_fft={args.n_fft}, win={args.win_length}, base_ch={args.base_ch}"
    )
    print(f"Audio source: {source_desc}")

    for frame_index in frame_indices:
        start_sample = frame_index * args.hop_size
        guitar_frame = frame_slice(guitar_audio, start_sample, args.frame_size)
        piano_frame = frame_slice(piano_audio, start_sample, args.frame_size)
        guitar_t = torch.from_numpy(guitar_frame).to(device).unsqueeze(0)
        piano_t = torch.from_numpy(piano_frame).to(device).unsqueeze(0)

        result = debugger(guitar_t, piano_t)
        png = out_dir / f"frame_{frame_index:04d}_teacher_debug.png"
        txt = out_dir / f"frame_{frame_index:04d}_teacher_metrics.txt"
        plot_frame(
            result,
            png,
            title=f"Teacher frame {frame_index} | start {start_sample} samples ({start_sample / SAMPLE_RATE:.4f}s)",
            args=args,
        )
        write_metrics(result, txt, args, frame_index, start_sample, criterion)
        if args.write_wavs:
            write_debug_wavs(result, out_dir, frame_index)
        print(f"Saved {png} and {txt}")


if __name__ == "__main__":
    main()
