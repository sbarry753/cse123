# GuSynth Neural Timbre Transfer

This directory contains the neural-network portion of GuSynth: a low-latency
guitar-to-piano timbre-transfer model and the scripts used to run it on a
Jetson Orin Nano.

The ML system processes the incoming guitar waveform directly and
outputs a transformed waveform that is intended to sound more like piano.

## Current State

- Target transfer: electric guitar audio to piano-like audio.
- Runtime platform: Jetson Orin Nano.
- Runtime format: ONNX Runtime or TensorRT engine.
- Audio rate: 48 kHz mono.
- Default realtime hop: 256 samples, about 5.33 ms at 48 kHz.
- Default model frame: 1024 samples, about 21.33 ms of model context.
- Active runtime script: `realtime_onnx_orin.py`.
- Active model definition: `model.py`.
- Active export script: `export_onnx.py`.

The repository also contains older prototypes and experiments. For the current
Orin path, use the root-level `model.py`, `train.py`, `export_onnx.py`, and
`realtime_onnx_orin.py` files.

## Model Architecture

The active model is a polyphonic waveform-to-waveform timbre-transfer network.

Runtime processing:

```text
guitar audio frame
  -> STFT
  -> log-magnitude spectrogram
  -> spectral U-Net predicts magnitude mask and residual
  -> phase TCN predicts bounded phase correction
  -> ISTFT
  -> transient correction branch
  -> piano-like audio frame
```

The model does not estimate pitch, notes, chords, or MIDI during live use.
Paired guitar/piano examples and alignment are used during training only.

## How the Neural Net and Overlap-Add Engine Work

The neural network is trained as a frame-based audio transformation model. Each
training example is a short guitar waveform frame paired with the matching piano
waveform frame. During live use, the model sees only raw audio samples, transforming it into the target output audio.

Inside `model.py`, each frame is converted into a short-time Fourier transform
(STFT). The model separates the problem into magnitude, phase, and transient
correction:

- The spectral U-Net reads the input log-magnitude spectrogram and predicts a
  multiplicative mask plus an additive residual. This reshapes the frequency
  content toward piano-like harmonics.
- The phase TCN reads the input phase context and the predicted magnitude
  context, then predicts a bounded phase residual. This helps the output remain
  coherent after the magnitude has been changed.
- The inverse STFT converts the predicted spectrum back into waveform audio.
- The transient correction branch applies a small attack-local waveform
  correction so pick attacks can be reshaped toward a more piano-like onset.

For TensorRT export, `export_onnx.py` wraps the PyTorch model with an
export-friendly version of the same signal path. PyTorch's complex STFT/ISTFT
operations are replaced with real-valued convolution and transpose-convolution
operations so the exported ONNX graph can be optimized by TensorRT.

The live Orin runtime uses an overlap-add engine because the model needs a
full context window, but the audio device must receive small fixed-size hops.
With the default settings, the model window is 1024 samples and the streaming
hop is 256 samples. That means each new audio callback contributes 256 fresh
samples while the model still receives the most recent 1024 samples.

The overlap-add engine in `realtime_onnx_orin.py` works like this:

```text
new 256-sample input hop arrives
  -> append hop to a 1024-sample input ring buffer
  -> run ONNX Runtime or TensorRT on the full 1024-sample frame
  -> multiply the 1024-sample prediction by a Hann synthesis window
  -> add the windowed prediction into an output accumulation ring
  -> add the same window into a normalization ring
  -> emit the oldest 256 normalized output samples
  -> shift the rings forward by one hop
```

The normalization ring prevents overlapping Hann-windowed predictions from
changing the output level. Each emitted sample is divided by the accumulated
window weight at that sample position.

The audio callback itself does not run neural inference. It only moves 256
sample hops into and out of queues. A separate worker thread owns the ONNX or
TensorRT inference calls. This keeps GPU synchronization out of the realtime
audio callback and makes missed worker deadlines visible through the runtime's
dropped-input and missed-output counters.

## Important Files

```text
model.py                Neural network architecture
dataset.py              Paired guitar/piano dataset and frame alignment
data_splits.py          Clip-level train/val/test split manifest generator
losses.py               Spectral, waveform, envelope, onset, and attack losses
train.py                Training entry point
export_onnx.py          PyTorch checkpoint -> fixed-shape ONNX export
realtime.py             Local PyTorch realtime/WAV prototype
realtime_onnx_orin.py   Jetson Orin ONNX/TensorRT realtime runtime
requirements.txt        Python dependencies for local training/prototyping
model_files/            Current model artifacts (.onnx, .engine, etc.)
checkpoints/            Training checkpoints
```

Current deployable artifacts are in `model_files/`:

```text
model_files/best_model.pt        PyTorch checkpoint
model_files/orin_model1024.onnx  1024-sample Orin ONNX model
model_files/model_768.onnx       768-sample ONNX experiment
model_files/model_512.onnx       512-sample ONNX experiment
```

The default model is `orin_model1024.onnx` because it provides high-quality output audio in a reasonable latency budget. 
The shorter frame-size experiments improve the latency floor of the model, but output audio quality is noticeably worse. 

## Data Layout

Training expects paired audio files with matching stems:

```text
data/
  guitar/
    phrase_001.wav
    phrase_002.wav
  piano/
    phrase_001.wav
    phrase_002.wav
```

Requirements:

- WAV or FLAC files.
- Matching filenames in `data/guitar/` and `data/piano/`.
- Same musical content in each pair.
- Files can use different sample rates; training resamples to 48 kHz.
- The dataset loader estimates small timing offsets and frames audio with
  overlapping windows.

Create a clip-level split manifest before training:

```bash
python data_splits.py --data_dir ./data --output ./data/splits.json
```

If `data/splits.json` already exists and should be replaced:

```bash
python data_splits.py --data_dir ./data --output ./data/splits.json --overwrite
```

## Local Training

Create a Python environment and install the local dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train from paired data:

```bash
python train.py \
  --data_dir ./data \
  --split_manifest ./data/splits.json \
  --output_dir ./checkpoints \
  --epochs 100 \
  --batch_size 16
```

The training script writes epoch checkpoints and updates
`checkpoints/best_model.pt` when validation loss improves.

Useful model-size/runtime knobs:

```bash
python train.py \
  --data_dir ./data \
  --split_manifest ./data/splits.json \
  --output_dir ./checkpoints \
  --frame_size 1024 \
  --hop_size 256 \
  --base_ch 64 \
  --phase_tcn_ch 16 \
  --phase_tcn_layers 3 \
  --phase_max_delta 0.10
```

Resume training:

```bash
python train.py \
  --data_dir ./data \
  --split_manifest ./data/splits.json \
  --output_dir ./checkpoints \
  --resume ./checkpoints/best_model.pt
```

## Export to ONNX

Export a trained PyTorch checkpoint to a fixed-shape ONNX model:

```bash
python export_onnx.py \
  --checkpoint ./checkpoints/best_model.pt \
  --output ./model_files/orin_model1024.onnx
```

The export wrapper replaces PyTorch STFT/ISTFT and complex operations with
real-valued convolutional equivalents so the graph can be converted for
TensorRT. It also writes a metadata JSON file next to the ONNX file.

For a non-default frame size, pass the matching frame size:

```bash
python export_onnx.py \
  --checkpoint ./checkpoints/best_model.pt \
  --output ./model_files/model_768.onnx \
  --frame_size 768
```

## Jetson Orin Nano Setup

Use NVIDIA JetPack on the Orin Nano so CUDA and TensorRT are available. A
virtual environment created with `--system-site-packages` is recommended so
Python can see JetPack-provided packages such as `tensorrt`.

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
```

The Orin runtime also needs:

- `onnxruntime` with Jetson-compatible CUDA/TensorRT providers, for ONNX mode.
- Python `tensorrt` and `cuda` bindings, for direct TensorRT engine mode.
- PortAudio-compatible audio through JACK or ALSA.

If TensorRT is available, build the engine on the same Jetson/JetPack version
that will run it:

```bash
trtexec \
  --onnx=model_files/orin_model1024.onnx \
  --saveEngine=model_files/orin_model1024_fp16.plan \
  --fp16
```

Serialized TensorRT engines are hardware/software specific, so rebuild the
`.plan` file after changing JetPack, TensorRT, CUDA, or the ONNX model.

## Run on Jetson Orin Nano

List available PortAudio devices:

```bash
python realtime_onnx_orin.py --list-devices --host-api JACK
```

Run live through a TensorRT engine:

```bash
python realtime_onnx_orin.py \
  --engine model_files/orin_model1024_fp16.plan \
  --host-api JACK \
  --input-device system:capture_1 \
  --output-device system:playback_1
```

Run live through ONNX Runtime:

```bash
python realtime_onnx_orin.py \
  --model model_files/orin_model1024.onnx \
  --provider auto \
  --host-api JACK \
  --input-device system:capture_1 \
  --output-device system:playback_1
```

Use a shorter-frame model by matching the ONNX/engine file and `--frame_size`:

```bash
python realtime_onnx_orin.py \
  --model model_files/model_768.onnx \
  --frame_size 768 \
  --provider auto \
  --host-api JACK
```

Process a WAV file instead of live input:

```bash
python realtime_onnx_orin.py \
  --engine model_files/orin_model1024_fp16.plan \
  --input data/guitar/plaz.wav \
  --output piano_out.wav
```

Runtime controls while streaming:

| Key | Action |
| --- | --- |
| `q` | Quit |
| `+` | Increase output volume |
| `-` | Decrease output volume |
| `m` | Toggle full wet/dry mix |
| `r` | Reset overlap-add rings |

## Latency Notes

The realtime callback uses a 256-sample hop, which is about 5.33 ms at 48 kHz.
The model frame provides additional context, and total perceived latency also
depends on:

- selected frame size,
- overlap-add buffering,
- `--queue-hops`,
- JACK/ALSA/PortAudio latency,
- audio interface buffering,
- TensorRT or ONNX Runtime worker time.

At startup and shutdown, `realtime_onnx_orin.py` reports worker inference times,
missed output hops, dropped input hops, and callback status events. These are
the main indicators for whether a model is fast enough for the current Orin
configuration.

## Troubleshooting

If ONNX Runtime falls back to CPU, check the printed provider list. The Orin
runtime should use TensorRT or CUDA providers for live use.

If audio drops out, increase `--queue-hops`, use `--fallback dry` or increase the
audio backend latency.

If the output shape warning appears, make sure `--frame_size` matches the model
or engine being loaded.

If a model sounds mostly like filtered guitar, improve the paired training data:
the guitar and piano files should contain the same phrase, have clean levels,
and be close enough in time for the dataset alignment step to correct them.