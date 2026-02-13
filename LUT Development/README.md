# Guitar → Sine Model LUT → Synth (JUCE VST)

A research-stage pipeline for converting recorded guitar notes and chords into a **mathematical sine-wave representation**, stored in a lookup table (LUT), for later real-time synthesis inside a JUCE VST.

---

## Project Goal

The long-term goal is to build a guitar-to-synth VST that:

1. Takes live instrument input  
2. Matches the incoming waveform to a stored sine-wave model  
3. Uses the closest match to drive a synthesized output  

This repository contains the **Python modeling stage**, which:

- Loads recorded guitar notes or chords  
- Approximates them as a **sum of sine waves**  
- Plots original vs reconstructed waveform  
- Saves the mathematical parameters to a LUT (JSON)  

---

# The Math

We approximate a short segment of a guitar signal as:

$$
x[n] \approx \sum_{k=1}^{K} A_k \sin\left(2\pi f_k \frac{n}{f_s} + \phi_k\right)
$$

Where:

- $x[n]$ = discrete-time audio sample  
- $f_s$ = sample rate  
- $f_k$ = detected frequency component  
- $A_k$ = amplitude  
- $\phi_k$ = phase  
- $K$ = number of sine components  

---

## Why This Works

A guitar tone can be approximated as:

- A fundamental frequency  
- Harmonic overtones  
- A transient attack  

Single note model:

$$
x[n] \approx A \sin\left(2\pi f \frac{n}{f_s} + \phi\right)
$$

Chord model:

$$
x[n] \approx \sum_{k=1}^{K} A_k \sin\left(2\pi f_k \frac{n}{f_s} + \phi_k\right)
$$

---

# 🔍 Frequency Detection

We compute the FFT of a Hann-windowed segment:

$$
X[k] = \sum_{n=0}^{N-1} x[n] w[n] e^{-j 2\pi kn/N}
$$

Dominant peaks in $|X[k]|$ correspond to frequency components.

---

# 📐 Least Squares Fitting

Instead of fitting:

$$
A \sin(\omega n + \phi)
$$

We fit:

$$
a \cos(\omega n) + b \sin(\omega n)
$$

This makes it a linear least-squares problem:

$$
\mathbf{x} = \mathbf{M}\theta
$$

Solve using:

$$
\theta = (M^T M)^{-1} M^T x
$$

Convert to amplitude and phase:

$$
A = \sqrt{a^2 + b^2}
$$

$$
\phi = \tan^{-1}\left(\frac{a}{b}\right)
$$

---

# LUT Format

Example entry:

```json
{
  "label": "E2_pick",
  "sample_rate_hz": 44100,
  "model": {
    "type": "sum_of_sines",
    "components": [
      {
        "freq_hz": 82.41,
        "amp": 0.732,
        "phase_rad": 1.13
      }
    ]
  }
}
```

Inside JUCE:

```cpp
float sample = 0.0f;
for (auto& c : components)
{
    sample += c.amp * std::sin(2.0f * float_Pi * c.freq * n / sampleRate + c.phase);
}
```

---

# Requirements

```bash
pip install numpy scipy matplotlib
```

---

# Status

- IN TESTING 

---

# Mathematical Inspiration

- Sinusoidal modeling synthesis  
- Short-time Fourier analysis  
- Linear least squares estimation  
- Harmonic analysis of plucked strings  
