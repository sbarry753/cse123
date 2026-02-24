//#include "DaisySP/daisysp.h"
#include <iostream>
#include <string>
#include "AudioFile.h"
#include "fuzz_effect.h"

int main(int argc, char* argv[]) {
    // Check if the user provided an input file argument
    if (argc < 2) {
        std::cerr << "Usage: ./effect_processor <path_to_input_file.wav>\n";
        return 1;
    }

    // Grab the file path from the command line
    std::string inputFile = argv[1];
    std::string outputFile = "output.wav";

    AudioFile<float> audioFile;

    // 1. Load the dynamically provided test track
    if (!audioFile.load(inputFile)) {
        std::cerr << "Error: Could not load " << inputFile << "\n";
        return 1;
    }

    float gain = 50000.0f; // High gain to force clipping

    // 2. Process the audio frame by frame
    int numChannels = audioFile.getNumChannels();
    int numSamples = audioFile.getNumSamplesPerChannel();

    for (int channel = 0; channel < numChannels; channel++) {
        for (int i = 0; i < numSamples; i++) {
            float currentSample = audioFile.samples[channel][i];

            // Apply the algorithm from fuzz_effect.cpp
            audioFile.samples[channel][i] = processFuzz(currentSample, gain);
        }
    }

    // 3. Save the manipulated audio to a new file
    audioFile.save(outputFile);
    std::cout << "Success! Effect applied and saved to " << outputFile << "\n";

    return 0;
}
