#include "wavefolding_effect.h"

float processWavefolding(float inputSample, float amount) {
    // 1. Push the signal to force folds
    float folded = inputSample * amount;
    
    // 2. Bounce the signal off the 1.0 / -1.0 ceilings
    while (folded > 1.0f || folded < -1.0f) {
        if (folded > 1.0f) {
            folded = 2.0f - folded;
        } else if (folded < -1.0f) {
            folded = -2.0f - folded;
        }
    }
    
    // 3. Return the folded wave. The peak will never exceed 1.0.
    return folded; 
}
