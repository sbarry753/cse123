# DDSP Guitar → Piano (Live Timbre Transfer)

Real-time guitar-to-piano timbre transfer using DDSP (Differentiable Digital Signal Processing).
Designed for live performance with a **<12ms algorithmic latency budget**.

---

## Architecture

```
Guitar audio (256-sample frame, ~5.8ms)
  │
  ▼
Feature Encoder  [~2ms]
  • f0 via YIN autocorrelation
  • RMS loudness in dB
  • MFCC (20 coefficients)
  │
  ▼
MLP Decoder  [~0.5ms]
  • 4-layer MLP, 512 hidden units
  • Trained on YOUR paired data
  • Maps guitar features → piano synth params
  │
  ▼
Additive Synthesizer  [~2.5ms]
  • 64 harmonics + shaped noise
  • Phase-continuous across frames
  │
  ▼
Output audio  [~5.8ms buffer]

Total algorithmic latency: ~11ms ✓
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. On Windows, install ASIO for lowest driver latency:
#    Use ASIO4ALL or your interface's native ASIO driver
```

---

## Prepare Your Data

Organize your paired audio files like this:

```
data/
  guitar/
    phrase_001.wav
    phrase_002.wav
    ...
  piano/
    phrase_001.wav   ← same filename, time-aligned, same duration
    phrase_002.wav
    ...
```

**Requirements:**
- Any sample rate (will be resampled to 44100 Hz)
- WAV or FLAC format
- Files must be time-aligned (same musical content, same duration)
- More data = better results. 10+ minutes of paired audio recommended.

---

## Training

```bash
# Basic training (100 epochs, auto-detects GPU/MPS/CPU)
python train.py --data_dir ./data --epochs 100

# With custom settings
python train.py \
  --data_dir    ./data \
  --epochs      150 \
  --batch_size  128 \       # increase if GPU has memory
  --hidden_size 512 \
  --n_harmonics 64 \
  --lr          3e-4

# Resume from checkpoint
python train.py --data_dir ./data --resume ./checkpoints/epoch_0050.pt
```

Training outputs:
- `checkpoints/best_model.pt` — best checkpoint (use this)
- `checkpoints/model_scripted.pt` — TorchScript export for real-time
- `checkpoints/loss_curves.png` — training curves

**Expected training time:**
| Hardware         | ~100 epochs |
|-----------------|-------------|
| NVIDIA GPU      | 20–60 min   |
| Apple M-series  | 45–90 min   |
| CPU only        | 4–8 hours   |

---

## Live Performance

```bash
# List available audio devices first
python realtime.py --model ./checkpoints/model_scripted.pt --list-devices

# Run live
python realtime.py --model ./checkpoints/model_scripted.pt

# With settings
python realtime.py \
  --model   ./checkpoints/model_scripted.pt \
  --wet     1.0 \     # 1.0 = full piano, 0.5 = 50/50 blend
  --volume  1.0 \
  --device  cuda      # or mps or cpu
```

**Keyboard controls (while running):**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset phase (fix phase artifacts between songs) |
| `+` | Increase volume |
| `-` | Decrease volume |
| `m` | Toggle dry/wet |

---

## Audio Interface Setup

For minimum total latency:

| OS      | Driver | Target buffer |
|---------|--------|---------------|
| Windows | ASIO   | 64–128 samples |
| macOS   | CoreAudio | 64–128 samples |
| Linux   | JACK   | 64–128 samples |

Set `DEVICE_IN` / `DEVICE_OUT` at the top of `realtime.py` to your interface.

```python
# realtime.py — set these:
DEVICE_IN  = "Focusrite USB"     # your interface name (partial match works)
DEVICE_OUT = "Focusrite USB"
```

---

## Troubleshooting

**Model sounds bad / robotic:**
- Train longer (200+ epochs)
- Add more paired data (target >20 min)
- Check that guitar/piano files are actually time-aligned

**Audio dropouts:**
- Increase `BLOCKSIZE` to 512 (adds ~5.8ms latency but more stable)
- Use a dedicated audio interface (not built-in soundcard)
- Close other audio applications

**Phase artifacts between notes:**
- Press `r` to reset phase accumulator
- These are normal for notes with very different f0s

**Latency higher than expected:**
- Run `python realtime.py --model ... --list-devices` and check your interface's native ASIO latency
- Algorithmic latency is fixed at ~11ms; total latency = algorithmic + driver round-trip

---

## File Structure

```
ddsp_guitar2piano/
  model.py       — DDSP architecture (encoder, MLP, synth)
  dataset.py     — Paired audio dataset loader
  losses.py      — Multi-scale spectral loss
  train.py       — Training script
  realtime.py    — Live inference
  requirements.txt
```
