#pragma once
#include <cmath>

// Declare the fuzz function so other files can see it
inline float processFuzz(float inputSample, float amount) {
    float driven = inputSample * amount;

    // Uses fminf/fmaxf which translate to single-cycle ARM hardware instructions
    return std::fmaxf(-1.0f, std::fminf(driven, 1.0f));
}

