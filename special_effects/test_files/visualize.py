import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

# 1. Load the audio files
fs_dry,  data_dry  = wavfile.read('/app/special_effects/test_files/PerfectTone.wav')
fs_fuzz, data_fuzz = wavfile.read('/app/special_effects/localtesting/output.wav')

# 2. Convert to mono if stereo
if data_dry.ndim > 1:
    data_dry = data_dry[:, 0]
if data_fuzz.ndim > 1:
    data_fuzz = data_fuzz[:, 0]

# 3. Cast to float for plotting — no normalization, show true amplitude
y_dry  = data_dry.astype(np.float32)
y_fuzz = data_fuzz.astype(np.float32)

# 4. Zoom in on a slice to see actual waveform shape
start_sample = 10000
slice_length  = 800

y_dry_slice  = y_dry [start_sample : start_sample + slice_length]
y_fuzz_slice = y_fuzz[start_sample : start_sample + slice_length]

# 5. Use the dry signal's peak to set a shared y-axis scale for fair comparison
peak = np.max(np.abs(y_dry_slice)) * 1.5

# 6. Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(y_dry_slice, color='blue')
axes[0].set_title('Waveform Before Effects')
axes[0].set_ylabel('Amplitude')
axes[0].set_ylim(-peak, peak)

axes[1].plot(y_fuzz_slice, color='red')
axes[1].set_title('Waveform After Effects')
axes[1].set_xlabel('Samples')
axes[1].set_ylabel('Amplitude')
axes[1].set_ylim(-peak, peak)

plt.tight_layout()

# 7. Save
output_path = '/app/special_effects/test_files/waveform_comparison.png'
plt.savefig(output_path)
print(f"Saved visualization to {output_path}")
