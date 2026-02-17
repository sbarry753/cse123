# LUT Guitar Synth Pedal (Polyphonic)

A research/prototype project for a **polyphonic guitar “synth” pedal** built around a **harmonic fingerprint lookup table (LUT)**.

Instead of doing full waveform resynthesis, this approach focuses on **template-based harmonic matching** in the frequency domain:

- **Monophonic:** classify a single played note by comparing it against a LUT of recorded single-note templates.
- **Polyphonic:** detect multiple notes (chords/strums) by fitting the live spectrum as a **nonnegative mixture** of single-note templates (NNLS).

> Current focus: nailing the math + workflow in Python first, then porting the core ideas into realtime prototypes (JUCE + Daisy Seed).

---

## Project status

**Python LUT + matcher prototype is active** (note detection + chord inference).  
**Daisy Seed + Audio Shield prototype is in progress** (hardware pedal pipeline).  
**JUCE project is a skeleton** (intended as a VST/desktop prototype harness).

---

## System diagrams

### Pedal-level concept

![Pedal concept](Documentation/images/Guitar%20Pedal.drawio.png)

### Signal flow (LUT + polyphonic inference)

![LUT polyphonic flow](Documentation/images/LUTPolyphonic.drawio%20(1).png)

> If these images don’t render in GitHub, check the filenames in your repo root and update the paths above.

---

## Repo layout (high level)

- `LUT - In Development/`
  - Python prototype that builds the LUT from labeled `.wav` samples and performs:
    - monophonic note classification
    - polyphonic chord note detection (NNLS mixture)
- `JUCE/`
  - Skeleton project (future: realtime VST / desktop prototyping harness)
- `DAISY/`
  - Skeleton/in-progress hardware prototype (Daisy Seed + Audio Shield)

---

## What the Python prototype does

### 1) Build: create a multi-take LUT per note

For each labeled single-note recording (`.wav`), the build script:

1. Loads WAV (stereo → mono)
2. Normalizes peak amplitude
3. Extracts a short analysis segment (`--start`, `--dur`)
4. (Optional) highpass filters (`--highpass`)
5. Computes FFT magnitude spectrum (Hann by default)
6. For the note’s known fundamental `f0` (from MIDI + A4 ref), measures **harmonic peaks**

For harmonic index `h = 1..K`:

- Search near `h * f0` within `± tol_hz`
- Take the max magnitude in that band

This yields a harmonic amplitude vector:

`v = [A1, A2, ..., AH]`

Normalize to remove loudness:

`fingerprint = v / sum(v)`

Each **take** stored in the LUT also includes extra “stability” features (used for better scoring):

- harmonic peak freqs / amps
- harmonic slope (linear fit of `log(amp)` vs harmonic index)
- inharmonicity (avg deviation from ideal harmonic locations)
- spectral centroid
- spectral rolloff (85%)
- spectral flatness

So the LUT is **multi-take per note**, not “one template per note”.

---

### 2) Monophonic matching (single note)

This system does **not** do “estimate f0 → map to note”.

Instead it does:

> “Assume every candidate note is correct, extract harmonic features under that assumption, and score against stored templates.”

For each candidate note in the LUT:

1. Compute candidate `f0` from its MIDI (and A4)
2. Compute live features **assuming that f0**
3. Compare against **every stored take** for that note
4. Collapse take scores into one note score (default: best-take wins)

**Scoring**
- base: cosine similarity between harmonic fingerprints
- penalties: feature deltas (inharmonicity, centroid, rolloff, slope, flatness)
- weights are CLI-tunable

---

### 3) Polyphonic matching (chords / multi-string)

Polyphonic mode **does not store chord templates**. It still uses the **single-note LUT**.

Workflow:

**A) Candidate prune (fast prepass)**  
Compute a quick fingerprint cosine score per note; keep the top `--prune N`.

**B) Build FFT-bin templates for each take**  
Convert each take into a template over FFT bins by “painting” Gaussian bumps at harmonic frequencies.
Optionally add detuned variants (± `--detune_cents`) to handle tuning drift.

**C) Fit mixture with NNLS**  
Solve:

`min ||A x - y||  subject to x >= 0`

Where:
- `y` is the live FFT magnitude (optionally `log1p`)
- `A` columns are take templates (including detunes)
- `x` are nonnegative activations

**D) Collapse take activations → note strengths**  
Take max activation per note, normalize, threshold, return up to `--max_notes`.

---

## LUT format (`lut.json`)

Each LUT note entry contains:

- `note`, `midi`, `f0_hz`
- analysis params used at build time (`k`, `tol_hz`, `window`, `analysis_start_sec`, `analysis_dur_sec`)
- `takes[]`: list of per-recording templates/features
- `source_wavs[]`: list of wavs used for that note

Example (simplified):

```json
{
  "note": "E3",
  "midi": 52,
  "f0_hz": 164.81,
  "k": 60,
  "tol_hz": 15.0,
  "window": "hann",
  "analysis_start_sec": 0.12,
  "analysis_dur_sec": 0.18,
  "takes": [
    {
      "fingerprint": [ ... ],
      "peak_freqs": [ ... ],
      "peak_amps": [ ... ],
      "harm_slope": -0.42,
      "inharm": 0.0031,
      "centroid_hz": 812.4,
      "rolloff_hz": 2490.0,
      "flatness": 0.018
    }
  ],
  "source_wavs": ["samples/E3.wav", "samples/E3_lowstring.wav"]
}
## Analog Overdrive Section (Planned)

The final pedal will include a **fully analog overdrive circuit** alongside the LUT-based note detection and synth engine.

We are using the basic overdrive topology outlined by Brian Wampler as a learning and reference design:

https://www.wamplerpedals.com/blog/latest-news/2020/05/how-to-design-a-basic-overdrive-pedal-circuit/

All required components for this circuit are already on hand. The overdrive will be implemented in hardware as part of the final pedal build.

### What this adds to the pedal

- True analog soft-clipping overdrive
- Gain (Drive), Tone, and Volume control
- Traditional op-amp + diode clipping topology
- Integrated into the same enclosure as the digital synth system

### Why analog?

The LUT synth system handles pitch detection and synthesis digitally, but distortion is often more natural, responsive, and low-latency in analog form. Including an analog overdrive stage allows:

- Classic guitar drive tones
- Synth-through-drive textures
- Flexible routing options in the final hardware design

The current repository does **not** yet contain the analog schematic implementation — this will be part of the hardware build phase (Daisy Seed prototype → final pedal PCB).

More detailed circuit documentation will live in a dedicated `analog/` folder once finalized.

## Status Reports
### Latest Status Report
[Status Report 3](Documentation/status_reports/status_report_3)

### All Status Reports
[Status Report 3](Documentation/status_reports/status_report_3)

[Status Report 2](Documentation/status_reports/status_report_2)

[Status Report 1](Documentation/status_reports/status_report_1)