#include "distortion_effect.h"
#include <cmath> 

float processDistortion(float inputSample, float amount) {
    // 1. Push the signal to change the shape
    float driven = inputSample * amount;
    
    // 2. Apply the soft clipping curve
    float distorted = std::tanh(driven);
    
    // 3. Auto-Normalize! 
    // Dividing by tanh(amount) guarantees the peak of the wave is always exactly 1.0
    return distorted; 
}
