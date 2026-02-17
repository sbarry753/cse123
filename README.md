# LUT Guitar Synth Pedal (Polyphonic)

A research and hardware project building a **polyphonic guitar synth pedal** using a **harmonic fingerprint lookup table (LUT)** for note and chord detection.

Instead of traditional waveform resynthesis or naive pitch tracking, this system performs:

- **Monophonic note detection** via harmonic template matching  
- **Polyphonic chord detection** using the template matching to help optimize NNLS or FFT
- **Strike detection** to separate attacks from sustain for better tracking  

The long-term goal is a standalone hardware pedal combining:

- Real-time digital note/chord detection
- Synth/sample playback engine
- Analog preamp + overdrive section
- Flexible routing between clean, driven, and synthesized signals

---

## Project Status

- **Python LUT prototype active** - note + chord detection working offline
- **Strike detection system in development**
- **Daisy Seed hardware prototype in progress**
- **JUCE project scaffolded for desktop/VST testing**
- **Analog front-end planned and partially prototyped**

---

## System Overview

### Pedal Concept

![Pedal concept](Documentation/images/Guitar%20Pedal.drawio.png)

### Detection + Synthesis Flow

![LUT polyphonic flow](Documentation/images/LUTPolyphonic.drawio%20(1).png)

---

## Core Detection Approach

### Harmonic Fingerprint LUT

Single-note recordings are analyzed and stored as harmonic “fingerprints.”  
Each note in the LUT contains multiple takes for robustness.

During detection, the system:

1. Assumes each possible note
2. Extracts harmonic features under that assumption
3. Scores against stored templates
4. Selects the best match

This avoids fragile single-f0 estimation and improves stability across strings and playing styles.

---

### Polyphonic (Chord) Detection

Chord detection does **not** use chord templates.

Instead:

- Single-note templates are converted into spectral templates
- The live spectrum is modeled as a **nonnegative mixture**
- NNLS determines which notes are present
- Results are pruned and thresholded

This allows detection of multiple simultaneous strings without pre-defining chord shapes.

---

### Strike Detection

Accurate detection requires separating:

- **Attack transients (strike)**
- **Sustained harmonic content**

Strike detection helps:
- Trigger new synth events cleanly
- Avoid re-triggering during sustain
- Improve chord onset recognition

This subsystem is currently under active development for real-time hardware use.

---

## Analog Front-End (Planned)

The final pedal will include a dedicated **analog input and overdrive section**.

### Clean Preamp (Line-Level Conditioning)

The microcontroller expects a stable line-level signal.  
A clean preamp stage is required to:

- Properly buffer guitar pickups
- Provide adjustable gain
- Condition signal before ADC

A functional prototype based on a CMOY-style op-amp design already exists and will be adapted for guitar-level input:

https://tangentsoft.com/audio/cmoy/

This will likely be redesigned onto a custom PCB.

---

### Analog Overdrive

Two overdrive-style circuits are being explored:

- A classic op-amp soft-clipping design (Wampler reference topology)  
  https://www.wamplerpedals.com/blog/latest-news/2020/05/how-to-design-a-basic-overdrive-pedal-circuit/

- A preamp-style overdrive inspired by the Tascam 424  
  https://aionfx.com/app/files/docs/424_preamp_documentation.pdf

The final hardware pedal will include an analog drive stage for:

- Traditional guitar overdrive tones
- Synth-through-drive textures
- Flexible signal routing options

---

## Analog Preprocessing & Detection Accuracy

There is ongoing investigation into whether analog preprocessing can improve polyphonic detection accuracy. Possible areas:

- Adjustable input gain
- Mild high-pass filtering
- Dynamic range conditioning
- Controlled saturation before ADC

A clean, well-scaled input signal is critical for stable harmonic fingerprint matching.

---

## Repository Layout

- `LUT - In Development/`  
  Python prototype for LUT building and note/chord detection

- `JUCE - In Development/`  
  Desktop/VST prototyping environment (skeleton)

- `DAISY - In Development/`  
  Hardware prototype (Daisy Seed + Audio Shield)

- `Documentation/`  
  Diagrams, status reports, and supporting materials

---

## Status Reports

### Latest
[Status Report 3](Documentation/status_reports/status_report_3)

### Archive
- [Status Report 2](Documentation/status_reports/status_report_2)
- [Status Report 1](Documentation/status_reports/status_report_1)

---

## Long-Term Goal

A standalone polyphonic guitar synth pedal combining:

- Harmonic fingerprint note/chord detection
- Real-time synthesis
- Analog drive and tone shaping
- Hardware-friendly DSP
- Expandable routing architecture

