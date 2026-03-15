import soundfile as sf
import torchaudio
import torch
from pathlib import Path
import numpy as np

TARGET_SR = 48000

INPUT_DIR = Path("dataset/input")
TARGET_DIR = Path("dataset/target")

OUT_INPUT = Path("dataset_48k/input")
OUT_TARGET = Path("dataset_48k/target")

OUT_INPUT.mkdir(parents=True, exist_ok=True)
OUT_TARGET.mkdir(parents=True, exist_ok=True)


def convert_file(in_path, out_path):
    audio, sr = sf.read(in_path, always_2d=True)

    audio = audio.astype(np.float32)
    audio = torch.from_numpy(audio.T)  # [C, T]

    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SR)
        print(f"Resampled {in_path.name}: {sr} -> {TARGET_SR}")
    else:
        print(f"Copied {in_path.name} (already 48k)")

    sf.write(out_path, audio.T.numpy(), TARGET_SR)


print("Processing INPUT files")
for f in INPUT_DIR.glob("*.wav"):
    convert_file(f, OUT_INPUT / f.name)

print("\nProcessing TARGET files")
for f in TARGET_DIR.glob("*.wav"):
    convert_file(f, OUT_TARGET / f.name)

print("\nDone.")