#!/usr/bin/env python3
import argparse
import math
import os
import struct
import subprocess
import tempfile
import wave


def generate_sine_wav(path, freq=1000, rate=48000, duration=3.0, channels=2, volume=0.25):
    """
    Generate a stereo sine wave WAV file.
    Default: 1 kHz, 48 kHz sample rate, 3 seconds.
    """
    frames = int(rate * duration)
    amplitude = int(volume * 32767)

    with wave.open(path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(rate)

        for n in range(frames):
            sample = int(amplitude * math.sin(2.0 * math.pi * freq * n / rate))

            if channels == 1:
                wav.writeframes(struct.pack("<h", sample))
            elif channels == 2:
                wav.writeframes(struct.pack("<hh", sample, sample))
            else:
                raise ValueError("Only mono or stereo supported")


def play_wav(path, device):
    """
    Play WAV through ALSA device using aplay.
    Example device: hw:APE,0 or hw:0,0
    """
    cmd = [
        "aplay",
        "-D", device,
        path
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate and play a test tone through Jetson I2S/ALSA.")
    parser.add_argument("-D", "--device", default="hw:APE,0",
                        help="ALSA playback device, e.g. hw:APE,0 or hw:0,0")
    parser.add_argument("-f", "--freq", type=float, default=1000,
                        help="Tone frequency in Hz")
    parser.add_argument("-r", "--rate", type=int, default=48000,
                        help="Sample rate")
    parser.add_argument("-d", "--duration", type=float, default=3.0,
                        help="Duration in seconds")
    parser.add_argument("-c", "--channels", type=int, default=2,
                        help="Number of channels: 1 or 2")
    parser.add_argument("-v", "--volume", type=float, default=0.25,
                        help="Volume from 0.0 to 1.0")
    parser.add_argument("--keep", action="store_true",
                        help="Keep generated WAV file instead of deleting it")

    args = parser.parse_args()

    if not 0.0 <= args.volume <= 1.0:
        raise ValueError("Volume must be between 0.0 and 1.0")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        print(f"Generating {args.freq} Hz tone at {args.rate} Hz...")
        generate_sine_wav(
            wav_path,
            freq=args.freq,
            rate=args.rate,
            duration=args.duration,
            channels=args.channels,
            volume=args.volume
        )

        print(f"Playing through ALSA device: {args.device}")
        play_wav(wav_path, args.device)

        if args.keep:
            print(f"Kept WAV file at: {wav_path}")

    finally:
        if not args.keep and os.path.exists(wav_path):
            os.remove(wav_path)


if __name__ == "__main__":
    main()