#pragma once

inline float processDistortion(float inputSample, float amount) {
    float driven = inputSample * amount;
    float driven2 = driven * driven;

    // Rational polynomial approximation (No std::tanh overhead)
    return driven * (27.0f + driven2) / (27.0f + 9.0f * driven2);
}

