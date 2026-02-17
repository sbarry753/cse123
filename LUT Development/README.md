````markdown
# Guitar → Harmonic Fingerprint LUT → Note / Chord Detection (Python prototype for a future JUCE VST)

This repo is a research-stage **template-based pitch / note identification** system for guitar audio.

It builds a **lookup table (LUT)** of *single-note templates* from labeled `.wav` recordings, then:

- **Monophonic:** classifies a single played note by scoring it against every note in the LUT (and every *take* of that note).
- **Polyphonic:** detects multiple notes (like a chord / strum) by fitting the spectrum as a **nonnegative mixture** of note templates (NNLS).

This is not waveform resynthesis. It’s **harmonic-structure matching** in the frequency domain.

---

## What the code actually does

### 1) Build step: multi-take templates per note

For each labeled note recording, the script:

1. Loads the WAV (stereo → mono)
2. Normalizes peak amplitude
3. Extracts a short analysis window (`--start`, `--dur`)
4. Optionally highpass filters (`--highpass`)
5. Computes an FFT magnitude spectrum (Hann by default)
6. For that note’s known fundamental `f0` (from MIDI + A4 reference), measures **harmonic peaks**

For harmonic index `h = 1..K`:

- Look near `h * f0` within `± tol_hz`
- Take the **max magnitude** in that band

That yields a harmonic amplitude vector:

`v = [A1, A2, ..., A_H]`

Then it normalizes it to remove loudness:

`fingerprint = v / sum(v)`

**But** the LUT doesn’t store just the fingerprint — each take stores extra spectral features too:

- harmonic peak frequencies (per harmonic)
- harmonic peak magnitudes (per harmonic)
- harmonic slope (linear fit of `log(amp)` vs harmonic index)
- inharmonicity (average relative deviation from ideal `k*f0`)
- spectral centroid
- spectral rolloff (85%)
- spectral flatness

So the LUT is **multi-take per note**, and each take is a richer “feature snapshot”.

---

## LUT format (`lut.json`)

Each note entry contains:

- `note`, `midi`, `f0_hz`
- analysis params used at build time (`k`, `tol_hz`, `window`, `analysis_start_sec`, `analysis_dur_sec`)
- `takes[]`: list of per-recording templates
- `source_wavs[]`: which wav files produced the takes

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
      "fingerprint": [...],
      "peak_freqs": [...],
      "peak_amps": [...],
      "harm_slope": -0.42,
      "inharm": 0.0031,
      "centroid_hz": 812.4,
      "rolloff_hz": 2490.0,
      "flatness": 0.018
    }
  ],
  "source_wavs": ["samples/E3.wav", "samples/E3_lowstring.wav"]
}
````

---

## 2) Monophonic matching (single note)

Monophonic classification is **not**:

> “extract one fingerprint from the unknown and compare to stored fingerprints”

Instead the code does this:

For **each candidate note** in the LUT:

1. Compute that candidate note’s `f0` from its MIDI (and A4 ref)
2. Compute **live features assuming that f0** (fingerprint + other features)
3. Compare the live features against **every stored take** for that note

### Scoring: cosine similarity + feature penalties

* base similarity: `cosine(fingerprint_live, fingerprint_take)`
* penalties for:

  * inharmonicity difference
  * centroid ratio difference (log)
  * rolloff ratio difference (log)
  * harmonic slope difference
  * flatness difference

Weights are CLI tunable (`--w_inharm`, `--w_centroid`, etc.)

### Multi-take collapse (important)

A note has multiple takes → multiple scores.

Those take scores get collapsed into a **single note score** using:

* `--score_mode max` (default): best take wins
* `--score_mode mean`
* `--score_mode topk --topk N`: average of top N takes

Then the note with the highest final score is your winner.

So the classifier is basically:

> “try every note as if it were correct, see which assumption produces the best harmonic alignment + feature match.”

---

## 3) Polyphonic matching (chords / multi-string)

Polyphonic mode does **not** store chord templates.

It still uses the same **single-note LUT** — but match time changes.

### Step A — prune candidate notes (fast pre-pass)

It runs a cheap fingerprint cosine score per note and keeps only the top `--prune N` notes to limit NNLS size.

### Step B — build FFT-bin templates for each take

For each candidate note and each take:

* Convert the take fingerprint into a **template vector over FFT bins**
* “Paint” a small Gaussian bump at each harmonic frequency
* Weight bumps by the take’s fingerprint values
* Normalize template to unit norm

Optionally it adds detuned template variants (`--detune_cents`, default ±20c) to handle tuning drift.

You end up with a matrix:

`A = [template_1 template_2 ... template_M]`

### Step C — fit the live spectrum as a nonnegative mixture (NNLS)

It computes the live FFT magnitude `y` (optionally `log1p` with `--logmag`), normalizes it, then solves:

`min ||A x - y||  subject to x >= 0`

(using `scipy.optimize.nnls`)

### Step D — collapse templates → notes

Multiple templates correspond to the same note (multiple takes + detunes).

The code collapses them by taking the **max weight per note**, normalizes note strengths by the max, then outputs up to `--max_notes` above `--thresh`.

So polyphonic output is:

> “which notes have strong activations in the NNLS mixture.”

---

## Commands

### Build LUT from explicit labels

```bash
python wav_to_lut.py build --out lut.json --k 60 --tol 15 --start 0.12 --dur 0.18 \
  E3=samples/E3.wav E3=samples/E3_lowstring.wav
```

### Build LUT from a directory (note parsed from filename)

Supports filenames like:

* `E6.wav`
* `F#4_take2.wav`
* `Bb3-clean.wav`
* `E3_lowstring.wav`

```bash
python wav_to_lut.py build_dir --out lut.json --dir samples --k 60 --tol 15 --start 0.12 --dur 0.18
```

### Match monophonic

```bash
python wav_to_lut.py match --lut lut.json --wav unknown.wav --start 0.12 --dur 0.18 --plot
```

Useful knobs:

* `--score_mode max|mean|topk`
* `--w_inharm ...` etc (feature penalty weights)

### Match polyphonic (chord)

```bash
python wav_to_lut.py match_poly --lut lut.json --wav chord.wav --start 0.12 --dur 0.18 \
  --max_notes 6 --thresh 0.25 --prune 60 --detune_cents 20 --logmag --plot
```

---

## Notes / assumptions baked into the approach

* The system **does not do explicit f0 estimation** first.

  * It instead evaluates candidates by “assuming” their f0 and measuring harmonic alignment.
* Short windows matter: `--start` and `--dur` heavily affect stability (attack vs sustain).
* Polyphonic mode is spectrum-mixture-based (NNLS), so it works best when:

  * notes are reasonably harmonic
  * not too much distortion / noise
  * pruning keeps the NNLS problem manageable

---

## Roadmap (VST direction)

This Python code is meant to prototype:

* a note/chord detector that can run on short windows
* multi-take robustness (different strings / pickups / picking styles)
* template-mixture chord inference

A JUCE plugin version would likely need:

* streaming / overlap-add analysis windows
* faster solvers or approximations than full NNLS
* onset (“strike”) detection and tracking across frames

```
```
