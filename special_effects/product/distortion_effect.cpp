#include "distortion_effect.h"
#include <cmath> // Required for std::tanh

float processDistortion(float inputSample, float gain) {
    // 1. Push the volume way up
    float driven = inputSample * gain;
    
    // 2. Soft clip the signal. 
    // std::tanh naturally rounds the peaks and traps the output between -1.0 and 1.0
    float distorted = std::tanh(driven);
    
    // 3. Drop the output volume back down to a safe level
    return distorted * 0.5f; 
}
