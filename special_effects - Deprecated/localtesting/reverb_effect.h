#ifndef REVERB_EFFECT_H
#define REVERB_EFFECT_H
#pragma once

#include <cmath>
#include <cstring>

// ----------------------------------------------------------------------------
// Schroeder Reverb
// Architecture: 4 parallel comb filters -> 2 series allpass filters
// This is a classic, CPU-efficient reverb suitable for embedded targets.
// No heap allocation occurs during processing — all buffers are statically sized.
//
// Sample rate assumption: 48000 Hz
// To retune for a different Fs, scale all DELAY_* constants proportionally.
// ----------------------------------------------------------------------------

// Comb filter delay lengths (in samples at 48kHz) — prime-ish to reduce resonance
static constexpr int COMB_DELAY_0 = 1687;
static constexpr int COMB_DELAY_1 = 1601;
static constexpr int COMB_DELAY_2 = 2053;
static constexpr int COMB_DELAY_3 = 2251;

// Allpass filter delay lengths
static constexpr int AP_DELAY_0 = 347;
static constexpr int AP_DELAY_1 = 113;

// Feedback and allpass coefficients
static constexpr float COMB_FEEDBACK = 0.85f;  // Controls reverb tail length
static constexpr float ALLPASS_COEFF = 0.7f;   // Standard Schroeder value
static constexpr float WET_MIX       = 0.40f;  // Wet/dry blend base (0=dry, 1=wet)

// ----------------------------------------------------------------------------
// Internal comb filter — feedback delay line with single-pole LP damping
// ----------------------------------------------------------------------------
struct CombFilter {
    float buf[2253] = {};  // Sized to max comb delay; zero-initialized
    int   size      = 0;
    int   idx       = 0;
    float feedback  = COMB_FEEDBACK;
    float lpState   = 0.0f;
    float damping   = 0.1f; // LP coefficient — higher = more high-freq damping

    void init(int delaySamples) {
        size = delaySamples;
        idx  = 0;
        memset(buf, 0, sizeof(buf));
    }

    inline float process(float input) {
        float output = buf[idx];
        // Single-pole lowpass on the feedback path (models air absorption)
        lpState      = output * (1.0f - damping) + lpState * damping;
        // Write feedback sum WITHOUT clamping — preserve unity gain fidelity
        buf[idx]     = input + lpState * feedback;
        idx          = (idx + 1 >= size) ? 0 : idx + 1; // branchless-friendly
        return output;
    }
};

// ----------------------------------------------------------------------------
// Internal allpass filter — adds diffusion without coloring the tone
// ----------------------------------------------------------------------------
struct AllpassFilter {
    float buf[348] = {};  // Sized to max allpass delay; zero-initialized
    int   size     = 0;
    int   idx      = 0;

    void init(int delaySamples) {
        size = delaySamples;
        idx  = 0;
        memset(buf, 0, sizeof(buf));
    }

    inline float process(float input) {
        float delayed = buf[idx];
        float output  = -input + delayed;
        buf[idx]      = input + delayed * ALLPASS_COEFF;
        idx           = (idx + 1 >= size) ? 0 : idx + 1;
        return output;
    }
};

// ----------------------------------------------------------------------------
// ReverbEffect — stateful processor; instantiate once per channel
// ----------------------------------------------------------------------------
class ReverbEffect {
public:
    ReverbEffect() {
        comb[0].init(COMB_DELAY_0);
        comb[1].init(COMB_DELAY_1);
        comb[2].init(COMB_DELAY_2);
        comb[3].init(COMB_DELAY_3);
        ap[0].init(AP_DELAY_0);
        ap[1].init(AP_DELAY_1);
    }

    // gain: scales the wet signal (0.0 = dry, 1.0+ = fully wet, clamped to 1.0)
    inline float process(float input, float gain) {
    // Remap gain to [0.0, 1.0] — treats gain=1 as default mix, gain=5 as fully wet
    float normalizedGain = std::fminf(gain / 5.0f, 1.0f);
    
    float wet = 0.0f;
    wet += comb[0].process(input);
    wet += comb[1].process(input);
    wet += comb[2].process(input);
    wet += comb[3].process(input);
    wet *= 0.25f;

    wet = ap[0].process(wet);
    wet = ap[1].process(wet);

    float wetLevel = WET_MIX * normalizedGain;
    float output   = input * (1.0f - wetLevel) + wet * wetLevel;

    return std::fmaxf(-1.0f, std::fminf(output, 1.0f));
    }

private:
    CombFilter    comb[4];
    AllpassFilter ap[2];
};

// ----------------------------------------------------------------------------
// Stateless-style wrapper — matches the processFuzz / processDistortion API.
// NOTE: reverb is inherently stateful. Callers must pass a persistent
//       ReverbEffect instance. Create one per channel, reuse across samples.
// ----------------------------------------------------------------------------
inline float processReverb(float input, float gain, ReverbEffect& reverb) {
    return reverb.process(input, gain);
}

#endif // REVERB_EFFECT_H
