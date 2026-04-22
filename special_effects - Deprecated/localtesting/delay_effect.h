#ifndef DELAY_EFFECT_H
#define DELAY_EFFECT_H
#pragma once

#include <cmath>
#include <cstring>

// ----------------------------------------------------------------------------
// Tape-style Delay Effect
// Architecture: circular buffer delay line with feedback
//
// All memory is statically allocated. No heap usage during processing.
// Sample rate assumption: 48kHz
//
// MAX_DELAY_SAMPLES controls the longest delay time supported.
// At 48kHz, 48000 samples = 1 second max delay.
// ----------------------------------------------------------------------------

static constexpr int   MAX_DELAY_SAMPLES    = 24000; // 1 second at 48kHz
static constexpr float DELAY_FEEDBACK       = 0.60f; // Repeat attenuation (0=single echo, <1=fading repeats)
static constexpr float DELAY_WET_MIX        = 0.50f; // Wet blend base (0=dry only, 1=wet only)

// Default delay time: ~300ms at 48kHz
static constexpr int   DEFAULT_DELAY_SAMPLES = 14400;

// ----------------------------------------------------------------------------
// DelayEffect — stateful circular buffer delay line.
// Instantiate once per channel, reuse across all samples.
// ----------------------------------------------------------------------------
class DelayEffect {
public:
    DelayEffect() {
        memset(buf, 0, sizeof(buf));
        writeIdx     = 0;
        delaySamples = DEFAULT_DELAY_SAMPLES;
    }

    // Set delay time in samples (must be < MAX_DELAY_SAMPLES)
    void setDelaySamples(int samples) {
        if (samples > 0 && samples < MAX_DELAY_SAMPLES)
            delaySamples = samples;
    }

    // gain: scales the wet/feedback level (acts as a depth control)
    // Clamped to [0, 1] so dry+wet always sums to 1.0 — prevents amplitude blowup
    inline float process(float input, float gain) {
        // Compute read index from write position
	float normalizedGain = std::fminf(gain / 5.0f, 1.0f);
        int readIdx = writeIdx - delaySamples;
        if (readIdx < 0) readIdx += MAX_DELAY_SAMPLES;

        float delayed = buf[readIdx];

        // Write feedback sum into buffer WITHOUT clamping — preserve unity gain fidelity
        buf[writeIdx] = input + delayed * DELAY_FEEDBACK;

        // Advance write pointer
        writeIdx = (writeIdx + 1 >= MAX_DELAY_SAMPLES) ? 0 : writeIdx + 1;

        // Clamp wetLevel to [0, 1] so dry+wet always sums to 1.0
        // This prevents amplitude blowup at high gain values (e.g. gain=5)
        float wetLevel = std::fminf(DELAY_WET_MIX * normalizedGain, 1.0f);
        float output   = input * (1.0f - wetLevel) + delayed * wetLevel;

        // Clamp final output — AGC handles this downstream, kept as safety net
	return std::fmaxf(-1.0f, std::fminf(output, 1.0f));
    }

private:
    float buf[MAX_DELAY_SAMPLES];
    int   writeIdx;
    int   delaySamples;
};

// ----------------------------------------------------------------------------
// Stateless-style wrapper — matches the processFuzz / processDistortion API.
// NOTE: delay is inherently stateful. Callers must pass a persistent
//       DelayEffect instance. Create one per channel, reuse across samples.
// ----------------------------------------------------------------------------
inline float processDelay(float input, float gain, DelayEffect& delay) {
    return delay.process(input, gain);
}

#endif // DELAY_EFFECT_H
