#include "fuzz_effect.h"
#include <algorithm> // For std::clamp

// Define the actual math for the fuzz effect
float processFuzz(float inputSample, float gain) {
    float boosted = inputSample * gain; 
    float clipped = std::clamp(boosted, -1.0f, 1.0f);
    return clipped * 0.5f; 
}
