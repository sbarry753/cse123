import numpy as np
import soundfile as sf
import torchaudio
import torch
from pathlib import Path

TARGET_SR = 48000
MAX_SHIFT_MS = 50
MAX_SHIFT = int(TARGET_SR * MAX_SHIFT_MS / 1000)

INPUT_DIR = Path("dataset/input")
TARGET_DIR = Path("dataset/target")
OUTPUT_DIR = Path("dataset_aligned/target")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_resample(path):
    audio, sr = sf.read(path, always_2d=True)

    # convert to mono
    audio = audio.mean(axis=1).astype(np.float32)
    audio = torch.from_numpy(audio).unsqueeze(0)

    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SR)

    return audio.squeeze(0).numpy()


def envelope(x, win=128):
    x = np.abs(x)
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def best_shift(x, y, max_shift):
    best_s = 0
    best_score = -1e18

    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            xa = x[:len(x) - s]
            ya = y[s:]
        else:
            xa = x[-s:]
            ya = y[:len(y) + s]

        n = min(len(xa), len(ya))
        if n < 500:
            continue

        score = np.dot(xa[:n], ya[:n])

        if score > best_score:
            best_score = score
            best_s = s

    return best_s


for input_path in INPUT_DIR.glob("*.wav"):

    target_path = TARGET_DIR / input_path.name
    if not target_path.exists():
        continue

    print(f"\nProcessing {input_path.name}")

    x = load_resample(input_path)
    y = load_resample(target_path)

    x_env = envelope(x)
    y_env = envelope(y)

    shift = best_shift(x_env, y_env, MAX_SHIFT)

    print(f"Detected delay: {shift} samples ({shift / TARGET_SR * 1000:.2f} ms)")

    if shift > 0:
        y_shift = y[shift:]
        pad = np.zeros(shift)
        y_shift = np.concatenate([y_shift, pad])
    elif shift < 0:
        pad = np.zeros(-shift)
        y_shift = np.concatenate([pad, y])[:len(y)]
    else:
        y_shift = y

    out_path = OUTPUT_DIR / input_path.name
    sf.write(out_path, y_shift, TARGET_SR)

    print(f"Saved aligned file -> {out_path}")