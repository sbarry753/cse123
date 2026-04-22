<<<<<<< HEAD
#pragma once
#include <cmath>

inline float processWavefolding(float inputSample, float amount) {
    float driven = inputSample * amount;
    
    // Branchless folding: no while loops, no unpredictable CPU stalling.
    // This perfectly calculates the triangle-wave folds in fixed time.
    return std::fabs(std::fmod(driven + 1.0f, 4.0f) - 2.0f) - 1.0f;
}

=======
#ifndef WAVEFOLDING_EFFECT_H
#define WAVEFOLDING_EFFECT_H

// Declare the wavefolding function
float processWavefolding(float inputSample, float gain);

#endif // WAVEFOLDING_EFFECT_H
>>>>>>> b786ae4c5ff541eba33bb3089403792ad699f197
