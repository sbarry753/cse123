import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

# 1. Load the audio files
fs_dry, data_dry = wavfile.read('/app/special_effects/test_files/DryGuitar.wav')
fs_fuzz, data_fuzz = wavfile.read('/app/special_effects/product/output.wav')

# 2. Convert to mono if the files are stereo
if data_dry.ndim > 1:
    data_dry = data_dry[:, 0]
if data_fuzz.ndim > 1:
    data_fuzz = data_fuzz[:, 0]

# 3. Zoom in on a tiny slice of audio (e.g., 1000 samples) to see the actual waveform shape
# You can change the start_sample to look at different parts of the recording
start_sample = 10000 
slice_length = 800

y_dry = data_dry[start_sample : start_sample + slice_length]
y_fuzz = data_fuzz[start_sample : start_sample + slice_length]

# 4. Plot the waveforms side-by-side
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(y_dry, color='blue')
plt.title('Dry Guitar (Original)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(y_fuzz, color='red')
plt.title('Fuzzed Guitar (Hard Clipped)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.tight_layout()

# 5. Save the result as an image file
plt.savefig('/app/special_effects/test_files/waveform_comparison.png')
print("Saved visualization to /app/special_effects/test_files/waveform_comparison.png")
