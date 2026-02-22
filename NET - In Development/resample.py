import soundfile as sf
import librosa
import numpy as np

inp = r"..\samples\A4_3.wav"
out = r"..\samples\A4_3_48k.wav"

y, sr = sf.read(inp, always_2d=False)
if y.ndim > 1:
    y = y.mean(axis=1)

y = y.astype(np.float32)
y48 = librosa.resample(y, orig_sr=sr, target_sr=48000)

sf.write(out, y48, 48000, subtype="PCM_16")
print("in_sr=", sr, "out_sr=48000", "written=", out, "len=", len(y48))
print("verify:", sf.info(out))