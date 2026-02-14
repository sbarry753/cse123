#include "DaisySP/daisysp.h"
#include "AudioFile.h"
#include <algorithm> // For std::clamp

AudioFile<double> audioFile;

float processFuzz(float inputSample, float gain) {
    float boosted = inputSample * gain; 
    float clipped = std::clamp(boosted, -1.0f, 1.0f);
    return clipped * 0.5f; 
}
