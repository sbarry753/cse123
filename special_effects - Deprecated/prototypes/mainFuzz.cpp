//#include "DaisySP/daisysp.h"
#include <iostream>
#include "AudioFile.h"
#include <algorithm> // For std::clamp

AudioFile<double> audioFile;

float processFuzz(float inputSample, float gain) {
    float boosted = inputSample * gain; 
    float clipped = std::clamp(boosted, -1.0f, 1.0f);
    return clipped * 0.5f; 
}

int main() {
    AudioFile<float> audioFile;
    // 1. Load your clean test track (make sure this file exists in your folder!)
    if (!audioFile.load("/app/special_effects/test_files/DryGuitar.wav")) {
        std::cerr << "Error: Could not load clean_guitar.wav\n";
        return 1;
    }
    // Set how hard we want to push the fuzz
    float gain = 50000.0f;
    // 2. Process the audio frame by frame
    int numChannels = audioFile.getNumChannels();
    int numSamples = audioFile.getNumSamplesPerChannel();
    for (int channel = 0; channel < numChannels; channel++) {
        for (int i = 0; i < numSamples; i++) {
            float currentSample = audioFile.samples[channel][i];

            // Apply your algorithm
            audioFile.samples[channel][i] = processFuzz(currentSample, gain);
        }
    }
    // 3. Save the manipulated audio to a new file
    audioFile.save("output.wav");
    std::cout << "Success! Fuzz applied and saved to fuzzed_output.wav\n";
    return 0;
}
