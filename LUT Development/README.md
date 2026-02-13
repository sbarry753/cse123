# Guitar → Harmonic Fingerprint LUT → Note Detection → Synth (JUCE VST)

A research-stage pipeline for converting recorded guitar notes into **normalized harmonic fingerprints**, stored in a lookup table (LUT), for real-time note detection inside a future JUCE VST.

---

## Project Goal

The long-term goal is to build a guitar-to-synth VST that:

1. Takes live instrument input  
2. Identifies which note is being played  
3. Uses that detected note to trigger a separate synthesized or sampled sound  

This repository contains the **Python modeling and detection stage**, which:

- Extracts harmonic fingerprints from recorded guitar notes  
- Stores them in a lookup table (LUT)  
- Matches unknown input audio against the LUT  
- Selects the most similar note using cosine similarity  

This system does **not** recreate the waveform directly.  
Instead, it performs **template matching in harmonic space**.

---

# Core Idea

Instead of fitting sine waves to reconstruct the signal, we:

1. Assume a candidate note with known fundamental frequency \( f_0 \)
2. Measure harmonic energy at integer multiples:
   
   \[
   f_k = k f_0
   \]

3. Normalize those harmonic amplitudes
4. Store that normalized vector as the note’s fingerprint

Later, live input is classified by comparing its harmonic fingerprint to the LUT.

---

# Harmonic Fingerprint Model

For a note with fundamental \( f_0 \), we compute:

\[
H_k = \text{Energy near } k f_0
\]

for \( k = 1, 2, ..., K \)

We then normalize:

\[
\tilde{H}_k = \frac{H_k}{\sum_{i=1}^{K} H_i}
\]

The resulting vector:

\[
\mathbf{h} = [\tilde{H}_1, \tilde{H}_2, ..., \tilde{H}_K]
\]

is the **harmonic fingerprint** of that note.

This removes loudness dependency and focuses only on harmonic structure.

---

# Why This Works

A guitar tone consists of:

- A fundamental frequency
- Harmonic overtones
- A transient attack

The harmonic amplitude ratios are relatively stable for:

- A given string
- A given fret
- A given pickup configuration

Even if:

- Volume changes
- Pick strength changes
- Slight EQ shifts occur

The **relative harmonic structure** remains similar.

This makes harmonic fingerprints effective for classification.

---

# FFT Analysis

We compute the short-time FFT of a Hann-windowed segment:

\[
X[k] = \sum_{n=0}^{N-1} x[n] w[n] e^{-j 2\pi kn/N}
\]

We then measure:

\[
H_k = \max |X(f)| \text{ near } k f_0
\]

within a small tolerance window (e.g., ±15 Hz).

This is repeated for each harmonic.

---

# Live Note Detection (Template Matching)

For an unknown input signal:

1. For each candidate note in the LUT:
   - Assume its known \( f_0 \)
   - Extract harmonic fingerprint from live audio
2. Compare live fingerprint \( \mathbf{h}_{live} \) with stored template \( \mathbf{h}_{lut} \)

We use cosine similarity:

\[
\text{similarity} =
\frac{\mathbf{h}_{live} \cdot \mathbf{h}_{lut}}
{\|\mathbf{h}_{live}\| \|\mathbf{h}_{lut}\|}
\]

The note with highest similarity is selected.

This avoids explicit fundamental detection and reduces octave errors.

---

# LUT Format

Example entry:

```json
{
  "note": "E6",
  "midi": 88,
  "f0_hz": 1318.51,
  "k": 60,
  "fingerprint": [
    0.28,
    0.21,
    0.16,
    0.09,
    ...
  ]
}
