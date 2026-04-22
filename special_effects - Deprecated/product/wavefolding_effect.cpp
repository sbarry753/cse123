#include "wavefolding_effect.h"

float processWavefolding(float inputSample, float gain) {
    // 1. Push the volume up
    float folded = inputSample * gain;
    
    // 2. Fold the signal back on itself if it crosses 1.0 or -1.0
    while (folded > 1.0f || folded < -1.0f) {
        if (folded > 1.0f) {
            // If it goes over 1.0, mirror it back down
            folded = 2.0f - folded;
        } else if (folded < -1.0f) {
            // If it goes under -1.0, mirror it back up
            folded = -2.0f - folded;
        }
    }
    
    // 3. Drop the volume safely for output
    return folded * 0.5f; 
}
