#ifndef DISTORTION_EFFECT_H
#define DISTORTION_EFFECT_H
#pragma once

// Declare the basic distortion function
float processDistortion(float inputSample, float drive);

inline float processDistortion(float inputSample, float amount) {
    float driven = inputSample * amount;
    float driven2 = driven * driven;

    // Rational polynomial approximation (No std::tanh overhead)
    return driven * (27.0f + driven2) / (27.0f + 9.0f * driven2);
}

#endif // DISTORTION_EFFECT_H

