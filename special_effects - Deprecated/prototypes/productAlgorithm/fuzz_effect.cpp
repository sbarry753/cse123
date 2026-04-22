#include "fuzz_effect.h"
#include <algorithm> 

float processFuzz(float inputSample, float amount) {
    // 1. Push the signal to change the shape
    float driven = inputSample * amount; 
    
    // 2. Hard clip it. The output will never exceed 1.0 or -1.0.
    return std::clamp(driven, -1.0f, 1.0f);
}
