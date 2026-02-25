import numpy as np
from scipy.io import wavfile

# Audio settings
sample_rate = 44100
duration = 1.0 # 1 second long
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate a perfect 440 Hz smooth sine wave
frequency = 440.0
audio = np.sin(2 * np.pi * frequency * t)

# Add a decay envelope so it fades out like a guitar string pluck
envelope = np.exp(-4 * t)
audio = audio * envelope

# Convert to a standard 16-bit WAV file format
audio_16bit = np.int16(audio * 32767)
wavfile.write('/app/special_effects/test_files/PerfectTone.wav', sample_rate, audio_16bit)

print("Created PerfectTone.wav!")
