"""
plot_spectrogram_comparison.py - compare target piano audio against model output.

Example:
  python3 plot_spectrogram_comparison.py \
    --model checkpoints/best_model.pt \
    --input data_small/input/Guitar.wav \
    --target data_small/target/Piano.wav \
    --output checkpoints/spectrogram_comparison.png \
    --context_size 2048 \
    --save_audio checkpoints/rendered_output.wav
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from model import DDSPGuitarToPiano, OverlapAddRenderer, SAMPLE_RATE, FRAME_SIZE, N_MELS


def checkpoint_state_dict(ckpt):
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    config = {}
    first_weight = state_dict.get("decoder.net.0.weight")
    if first_weight is not None:
        config["hidden_size"] = int(first_weight.shape[0])
        decoder_input_size = int(first_weight.shape[1])
        z_latent_size = decoder_input_size - (2 + N_MELS)
        config["use_z"] = z_latent_size > 0
        config["z_latent_size"] = max(0, z_latent_size)

    harm_weight = state_dict.get("decoder.head_harmonic_amps.weight")
    if harm_weight is not None:
        config["n_harmonics"] = int(harm_weight.shape[0])

    return config


def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def load_model(path: str, args, device: torch.device):
    try:
        model = torch.jit.load(path, map_location="cpu")
        print(f"Loaded TorchScript model from {path}")
    except Exception:
        print(f"Loading state dict from {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint_state_dict(ckpt)
        inferred = infer_model_config(state_dict)
        hidden_size = inferred.get("hidden_size")
        n_harmonics = inferred.get("n_harmonics")
        use_z = inferred.get("use_z", True)
        z_latent_size = inferred.get("z_latent_size", 64)
        if inferred:
            print(
                "Inferred checkpoint config: "
                f"hidden_size={hidden_size}, n_harmonics={n_harmonics}, "
                f"use_z={use_z}, z_latent_size={z_latent_size}"
            )
        model = DDSPGuitarToPiano(
            hidden_size=hidden_size,
            n_harmonics=n_harmonics,
            context_size=args.context_size,
            hop_size=args.hop_size,
            use_z=use_z,
            z_latent_size=z_latent_size,
        )
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_audio(path: str, max_seconds: float | None = None) -> np.ndarray:
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        print(f"Resampling {path}: {sr} -> {SAMPLE_RATE} Hz")
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio_np = audio.squeeze(0).numpy().astype(np.float32)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 1e-8:
        audio_np = audio_np / peak
    if max_seconds is not None:
        max_samples = int(max_seconds * SAMPLE_RATE)
        audio_np = audio_np[:max_samples]
    return audio_np


def render_model_output(
    model,
    audio_np: np.ndarray,
    context_size: int,
    hop_size: int,
    device: torch.device,
    overlap_add: bool = True,
    high_freq_hz: float = 8000.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    orig_len = len(audio_np)
    pad = (hop_size - orig_len % hop_size) % hop_size
    if pad:
        audio_np = np.concatenate([audio_np, np.zeros(pad, dtype=np.float32)])

    n_frames = len(audio_np) // hop_size
    output_np = np.zeros_like(audio_np)
    context_buf = torch.zeros(1, context_size, device=device)
    traces = {
        "global_amp": [],
        "pred_f0": [],
        "noise_gain": [],
        "noise_mag_mean": [],
        "high_harmonic_ratio": [],
    }

    def append_param_traces(params):
        traces["global_amp"].append(float(params["global_amp"][0].detach().cpu()))
        traces["pred_f0"].append(float(params["f0_corrected"][0].detach().cpu()))
        traces["noise_gain"].append(float(params.get("noise_gain", torch.tensor([float("nan")]))[0].detach().cpu()))
        traces["noise_mag_mean"].append(float(params["noise_mags"][0].detach().mean().cpu()))

        harm_amps = params["harm_amps"][0].detach()
        f0 = params["f0_corrected"][0].detach()
        harmonic_idx = torch.arange(1, harm_amps.numel() + 1, device=harm_amps.device, dtype=harm_amps.dtype)
        high_mask = (f0 * harmonic_idx) >= high_freq_hz
        total_energy = torch.sum(harm_amps ** 2)
        if bool(high_mask.any()) and float(total_energy.detach().cpu()) > 1e-12:
            high_energy = torch.sum(harm_amps[high_mask] ** 2)
            ratio = high_energy / total_energy.clamp_min(1e-12)
            traces["high_harmonic_ratio"].append(float(ratio.detach().cpu()))
        else:
            traces["high_harmonic_ratio"].append(0.0)

    if hasattr(model, "reset_phase"):
        model.reset_phase()

    with torch.no_grad():
        ola_renderer = (
            OverlapAddRenderer(model, context_size, hop_size, device)
            if overlap_add and hasattr(model, "predict_params") and hasattr(model, "render_params")
            else None
        )
        for i in tqdm(range(n_frames), unit="frame", ncols=72):
            s, e = i * hop_size, (i + 1) * hop_size
            frame_np = audio_np[s:e]

            if ola_renderer is not None:
                pred_hop = ola_renderer.process_frame(frame_np)
                params = ola_renderer.last_params
                append_param_traces(params)
                output_np[s:e] = pred_hop.detach().cpu().numpy().astype(np.float32)
            else:
                frame = torch.from_numpy(frame_np).to(device)
                context_buf[:, :-hop_size] = context_buf[:, hop_size:].clone()
                context_buf[:, -hop_size:] = frame

                if hasattr(model, "predict_params") and hasattr(model, "render_params"):
                    _, params = model.predict_params(context_buf)
                    pred = model.render_params(params)
                    append_param_traces(params)
                else:
                    pred = model.infer_frame(context_buf) if hasattr(model, "infer_frame") else model(context_buf)[0]
                    traces["global_amp"].append(float("nan"))
                    traces["pred_f0"].append(float("nan"))
                    traces["noise_gain"].append(float("nan"))
                    traces["noise_mag_mean"].append(float("nan"))
                    traces["high_harmonic_ratio"].append(float("nan"))
                output_np[s:e] = pred[0].detach().cpu().numpy().astype(np.float32)

    return output_np[:orig_len], {
        key: np.asarray(value, dtype=np.float32)
        for key, value in traces.items()
    }


def trim_to_common_length(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    min_len = min(len(sig) for sig in signals)
    return tuple(sig[:min_len] for sig in signals)


def spectrogram_db(audio_np: np.ndarray, n_fft: int, hop_length: int, db_floor: float) -> np.ndarray:
    audio = torch.from_numpy(audio_np.astype(np.float32))
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = spec.abs().clamp_min(1e-8)
    db = 20.0 * torch.log10(mag)
    return db.clamp_min(db_floor).numpy()


def signal_stats(name: str, audio_np: np.ndarray) -> dict[str, float]:
    peak = float(np.max(np.abs(audio_np))) if len(audio_np) else 0.0
    rms = float(np.sqrt(np.mean(audio_np ** 2) + 1e-12)) if len(audio_np) else 0.0
    print(f"{name:<12} peak={peak:.4f}  rms={rms:.4f}")
    return {"peak": peak, "rms": rms}


def rms_match_output(target_np: np.ndarray, output_np: np.ndarray) -> tuple[np.ndarray, float]:
    target_rms = float(np.sqrt(np.mean(target_np ** 2) + 1e-12)) if len(target_np) else 0.0
    output_rms = float(np.sqrt(np.mean(output_np ** 2) + 1e-12)) if len(output_np) else 0.0
    if output_rms <= 1e-8:
        return output_np.copy(), 1.0
    gain = target_rms / output_rms
    return (output_np * gain).astype(np.float32), gain


def print_spectral_summary(target_db: np.ndarray, output_db: np.ndarray):
    diff = output_db - target_db
    mean_abs_diff = float(np.mean(np.abs(diff)))
    mean_signed_diff = float(np.mean(diff))

    print(f"Mean |output-target| dB      : {mean_abs_diff:.2f}")
    print(f"Mean output-target dB        : {mean_signed_diff:+.2f}")


def print_trace_summary(traces: dict[str, np.ndarray]):
    global_amp = traces.get("global_amp")
    if global_amp is None or not len(global_amp) or np.isnan(global_amp).all():
        return

    print(
        "Global amp summary           : "
        f"mean={np.nanmean(global_amp):.4f}  "
        f"median={np.nanmedian(global_amp):.4f}  "
        f"p95={np.nanpercentile(global_amp, 95):.4f}  "
        f"max={np.nanmax(global_amp):.4f}"
    )
    noise_gain = traces.get("noise_gain")
    if noise_gain is not None and len(noise_gain) and not np.isnan(noise_gain).all():
        print(
            "Noise gain summary          : "
            f"mean={np.nanmean(noise_gain):.5f}  "
            f"median={np.nanmedian(noise_gain):.5f}  "
            f"p95={np.nanpercentile(noise_gain, 95):.5f}  "
            f"max={np.nanmax(noise_gain):.5f}"
        )
    high_harm = traces.get("high_harmonic_ratio")
    if high_harm is not None and len(high_harm) and not np.isnan(high_harm).all():
        print(
            "High harmonic energy ratio  : "
            f"mean={np.nanmean(high_harm):.4f}  "
            f"median={np.nanmedian(high_harm):.4f}  "
            f"p95={np.nanpercentile(high_harm, 95):.4f}  "
            f"max={np.nanmax(high_harm):.4f}"
        )


def high_frequency_diff_trace(
    target_db: np.ndarray,
    output_db: np.ndarray,
    n_fft: int,
    high_freq_hz: float,
) -> tuple[np.ndarray, float]:
    freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, target_db.shape[0], dtype=np.float32)
    mask = freqs >= high_freq_hz
    if not np.any(mask):
        return np.zeros(target_db.shape[1], dtype=np.float32), high_freq_hz
    diff_trace = np.mean(output_db[mask, :] - target_db[mask, :], axis=0)
    return diff_trace.astype(np.float32), high_freq_hz


def plot_comparison(
    input_np: np.ndarray,
    target_np: np.ndarray,
    output_np: np.ndarray,
    traces: dict[str, np.ndarray],
    input_db: np.ndarray,
    target_db: np.ndarray,
    output_db: np.ndarray,
    output_path: str,
    hop_length: int,
    db_floor: float,
    model_hop_size: int,
    high_freq_hz: float,
):
    duration = len(target_np) / SAMPLE_RATE
    extent = [0.0, duration, 0.0, SAMPLE_RATE / 2.0]
    wave_times = np.arange(len(target_np), dtype=np.float32) / SAMPLE_RATE

    target_output_max = max(float(target_db.max()), float(output_db.max()))
    spec_vmin = target_output_max + db_floor
    spec_vmax = target_output_max

    input_max = float(input_db.max())
    input_vmin = input_max + db_floor
    input_vmax = input_max

    diff_db = output_db - target_db
    hf_diff_trace, hf_start_hz = high_frequency_diff_trace(
        target_db,
        output_db,
        n_fft=0,
        high_freq_hz=high_freq_hz,
    )
    spec_times = np.linspace(0.0, duration, hf_diff_trace.shape[0], dtype=np.float32)

    fig, axes = plt.subplots(
        8,
        1,
        figsize=(16, 20),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.35, 1.35, 1.35, 1.0, 0.7, 0.7, 0.7]},
    )

    im0 = axes[0].imshow(
        input_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=input_vmin,
        vmax=input_vmax,
        cmap="magma",
    )
    axes[0].set_title("Guitar Input Spectrogram")
    axes[0].set_ylabel("Hz")
    fig.colorbar(im0, ax=axes[0], pad=0.01, label="dB")

    im1 = axes[1].imshow(
        target_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=spec_vmin,
        vmax=spec_vmax,
        cmap="magma",
    )
    axes[1].set_title("Target Piano Spectrogram")
    axes[1].set_ylabel("Hz")
    fig.colorbar(im1, ax=axes[1], pad=0.01, label="dB")

    im2 = axes[2].imshow(
        output_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=spec_vmin,
        vmax=spec_vmax,
        cmap="magma",
    )
    axes[2].set_title("Model Output Spectrogram")
    axes[2].set_ylabel("Hz")
    fig.colorbar(im2, ax=axes[2], pad=0.01, label="dB")

    im3 = axes[3].imshow(
        diff_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-30.0,
        vmax=30.0,
        cmap="coolwarm",
    )
    axes[3].set_title("Difference Spectrogram: Output dB - Target dB")
    axes[3].set_ylabel("Hz")
    fig.colorbar(im3, ax=axes[3], pad=0.01, label="dB")

    axes[4].plot(wave_times, target_np, label="Target piano", linewidth=0.8, alpha=0.8)
    axes[4].plot(wave_times, output_np, label="Model output", linewidth=0.8, alpha=0.8)
    axes[4].set_title("Waveform Overlay")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Amplitude")
    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.25)

    frame_times = np.arange(len(traces["global_amp"]), dtype=np.float32) * model_hop_size / SAMPLE_RATE
    axes[5].plot(frame_times, traces["global_amp"], label="global_amp", linewidth=0.9)
    axes[5].set_title("Decoder Global Amplitude")
    axes[5].set_xlabel("Time (s)")
    axes[5].set_ylabel("Amp")
    axes[5].set_ylim(-0.02, 1.02)
    axes[5].legend(loc="upper right")
    axes[5].grid(True, alpha=0.25)

    noise_gain = traces.get("noise_gain", np.array([], dtype=np.float32))
    noise_mag_mean = traces.get("noise_mag_mean", np.array([], dtype=np.float32))
    if len(noise_gain):
        axes[6].plot(frame_times, noise_gain, label="noise_gain", linewidth=0.9, color="darkorange")
    if len(noise_mag_mean):
        axes[6].plot(frame_times, noise_mag_mean, label="mean noise_mags", linewidth=0.9, color="seagreen", alpha=0.8)
    axes[6].set_title("Decoder Noise Controls")
    axes[6].set_xlabel("Time (s)")
    axes[6].set_ylabel("Gain / Mag")
    axes[6].legend(loc="upper right")
    axes[6].grid(True, alpha=0.25)

    high_harmonic_ratio = traces.get("high_harmonic_ratio", np.array([], dtype=np.float32))
    if len(high_harmonic_ratio):
        axes[7].plot(frame_times, high_harmonic_ratio, label=f"harmonic energy ratio >= {high_freq_hz / 1000.0:.1f} kHz", linewidth=0.9, color="purple")
    axes[7].set_title("High Harmonic Energy / Total Harmonic Energy")
    axes[7].set_xlabel("Time (s)")
    axes[7].set_ylabel("Ratio")
    axes[7].set_ylim(-0.02, 1.02)
    axes[7].legend(loc="upper right")
    axes[7].grid(True, alpha=0.25)

    extra_ax = axes[7].twinx()
    extra_ax.plot(spec_times, hf_diff_trace, label=f"output-target dB >= {hf_start_hz / 1000.0:.1f} kHz", linewidth=0.7, color="steelblue", alpha=0.45)
    extra_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    extra_ax.set_ylabel("dB excess")
    extra_ax.legend(loc="lower right")

    # The harmonic ratio and high-frequency dB excess share a panel so their
    # correlation is visually obvious without making the diagnostic figure taller.
    axes[7].set_title("High Harmonic Ratio and High-Frequency Excess")

    for ax in axes[:4]:
        ax.set_ylim(0.0, SAMPLE_RATE / 2.0)
        ax.set_xlabel("Time (s)")

    fig.suptitle(
        f"Target vs Output Spectrogram Comparison "
        f"(STFT hop={hop_length}, sr={SAMPLE_RATE})",
        y=0.995,
    )
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a DDSP checkpoint and plot target/output spectrogram diagnostics."
    )
    parser.add_argument("--model", required=True, help="Model checkpoint or TorchScript path")
    parser.add_argument("--input", required=True, help="Aligned guitar input WAV")
    parser.add_argument("--target", required=True, help="Aligned piano target WAV")
    parser.add_argument("--output", default="spectrogram_comparison.png", help="Output PNG path")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--n_harmonics", type=int, default=64)
    parser.add_argument("--context_size", type=int, default=2048)
    parser.add_argument("--hop_size", type=int, default=FRAME_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_seconds", type=float, default=None, help="Optional duration limit")
    parser.add_argument("--save_audio", default=None, help="Optional path for rendered output WAV")
    parser.add_argument(
        "--rms_match_plot",
        action="store_true",
        help="Scale a copy of the model output to target RMS before spectrogram/waveform plotting",
    )
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--spec_hop_length", type=int, default=256)
    parser.add_argument("--db_floor", type=float, default=-80.0)
    parser.add_argument(
        "--high_freq_hz",
        type=float,
        default=8000.0,
        help="Frequency threshold for high-frequency excess diagnostic trace",
    )
    parser.add_argument(
        "--overlap_add",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inference-time Hann overlap-add smoothing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    model = load_model(args.model, args, device)

    input_np = load_audio(args.input, args.max_seconds)
    target_np = load_audio(args.target, args.max_seconds)
    print(f"Input samples before trim : {len(input_np):,}")
    print(f"Target samples before trim: {len(target_np):,}")

    output_np, traces = render_model_output(
        model,
        input_np,
        args.context_size,
        args.hop_size,
        device,
        overlap_add=args.overlap_add,
        high_freq_hz=args.high_freq_hz,
    )
    input_np, target_np, output_np = trim_to_common_length(input_np, target_np, output_np)
    print(f"Common comparison length  : {len(target_np):,} samples ({len(target_np) / SAMPLE_RATE:.2f}s)")

    if args.save_audio:
        Path(args.save_audio).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(args.save_audio, torch.from_numpy(output_np).unsqueeze(0), SAMPLE_RATE)
        print(f"Saved rendered audio: {args.save_audio}")

    signal_stats("Target", target_np)
    signal_stats("Output raw", output_np)

    plot_output_np = output_np
    output_label = "raw output"
    if args.rms_match_plot:
        plot_output_np, rms_gain = rms_match_output(target_np, output_np)
        output_label = "RMS-matched output"
        print(f"Plot RMS-match gain applied to output copy: {rms_gain:.6f}")
        signal_stats("Output plot", plot_output_np)

    input_db = spectrogram_db(input_np, args.n_fft, args.spec_hop_length, args.db_floor)
    target_db = spectrogram_db(target_np, args.n_fft, args.spec_hop_length, args.db_floor)
    output_db = spectrogram_db(plot_output_np, args.n_fft, args.spec_hop_length, args.db_floor)

    print(f"Spectral comparison uses: {output_label}")
    print_spectral_summary(target_db, output_db)
    print_trace_summary(traces)

    plot_comparison(
        input_np=input_np,
        target_np=target_np,
        output_np=plot_output_np,
        traces=traces,
        input_db=input_db,
        target_db=target_db,
        output_db=output_db,
        output_path=args.output,
        hop_length=args.spec_hop_length,
        db_floor=args.db_floor,
        model_hop_size=args.hop_size,
        high_freq_hz=args.high_freq_hz,
    )
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
