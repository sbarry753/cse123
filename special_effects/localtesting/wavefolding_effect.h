#pragma once
#include <cmath>

inline float processWavefolding(float inputSample, float amount) {
    float driven = inputSample * amount;
    
    // Branchless folding: no while loops, no unpredictable CPU stalling.
    // This perfectly calculates the triangle-wave folds in fixed time.
    return std::fabs(std::fmod(driven + 1.0f, 4.0f) - 2.0f) - 1.0f;
}

