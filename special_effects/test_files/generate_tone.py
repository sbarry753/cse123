import numpy as np
from scipy.io import wavfile

<<<<<<< HEAD
# Match Daisy's exact audio settings
sample_rate = 48000  # Daisy uses 48kHz, not 44100
block_size = 48      # Try 4, 8, 16, 48 to test different block sizes

# Generate exactly one block's worth of samples
t = np.linspace(0, block_size / sample_rate, block_size, endpoint=False)

# 440 Hz sine wave, no envelope — constant amplitude like a live input
frequency = 440.0
audio = np.sin(2 * np.pi * frequency * t)

# Convert to 16-bit
audio_16bit = np.int16(audio * 32767)

wavfile.write(f'/app/special_effects/test_files/Block_{block_size}_samples.wav', sample_rate, audio_16bit)
print(f"Created block of {block_size} samples at {sample_rate}Hz ({block_size / sample_rate * 1000:.3f}ms)")

#import numpy as np
#from scipy.io import wavfile
#
## Audio settings
#sample_rate = 44100
#duration = 1.0 # 1 second long
#t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
#
## Generate a perfect 440 Hz smooth sine wave
#frequency = 440.0
#audio = np.sin(2 * np.pi * frequency * t)
#
## Add a decay envelope so it fades out like a guitar string pluck
#envelope = np.exp(-4 * t)
#audio = audio * envelope
#
## Convert to a standard 16-bit WAV file format
#audio_16bit = np.int16(audio * 32767)
#wavfile.write('/app/special_effects/test_files/PerfectTone.wav', sample_rate, audio_16bit)
#
#print("Created PerfectTone.wav!")
=======
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
>>>>>>> b786ae4c5ff541eba33bb3089403792ad699f197
