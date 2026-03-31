<<<<<<< HEAD
#pragma once
#include <cmath>

// Declare the fuzz function so other files can see it
inline float processFuzz(float inputSample, float amount) {
    float driven = inputSample * amount;

    // Uses fminf/fmaxf which translate to single-cycle ARM hardware instructions
    return std::fmaxf(-1.0f, std::fminf(driven, 1.0f));
}

=======
#ifndef FUZZ_EFFECT_H
#define FUZZ_EFFECT_H

// Declare the fuzz function so other files can see it
float processFuzz(float inputSample, float gain);

#endif // FUZZ_EFFECT_H
>>>>>>> b786ae4c5ff541eba33bb3089403792ad699f197
