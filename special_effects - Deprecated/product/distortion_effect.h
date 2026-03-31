<<<<<<< HEAD
#pragma once

inline float processDistortion(float inputSample, float amount) {
    float driven = inputSample * amount;
    float driven2 = driven * driven;

    // Rational polynomial approximation (No std::tanh overhead)
    return driven * (27.0f + driven2) / (27.0f + 9.0f * driven2);
}

=======
#ifndef DISTORTION_EFFECT_H
#define DISTORTION_EFFECT_H

// Declare the basic distortion function
float processDistortion(float inputSample, float drive);

#endif // DISTORTION_EFFECT_H
>>>>>>> b786ae4c5ff541eba33bb3089403792ad699f197
