//#include "DaisySP/daisysp.h"
#include <iostream>
#include <string>
#include "AudioFile.h"
#include "fuzz_effect.h"
#include "distortion_effect.h"
#include "wavefolding_effect.h" // Include your properly named wavefolder

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./effect_processor <effect_name> <path_to_input_file.wav>\n";
        std::cerr << "Available effects: fuzz_effect, distortion_effect, wavefolding_effect\n";
        return 1;
    }

    std::string effectName = argv[1];
    std::string inputFile = argv[2];
    std::string outputFile = "output.wav";

    AudioFile<float> audioFile;

    if (!audioFile.load(inputFile)) {
        std::cerr << "Error: Could not load " << inputFile << "\n";
        return 1;
    }

    int numChannels = audioFile.getNumChannels();
    int numSamples = audioFile.getNumSamplesPerChannel();

    // Route the audio to the correct effect
    if (effectName == "fuzz_effect") {
        std::cout << "Applying Fuzz Effect (Hard Clipping)...\n";
        float fuzzGain = 10.0f; 
        for (int channel = 0; channel < numChannels; channel++) {
            for (int i = 0; i < numSamples; i++) {
                audioFile.samples[channel][i] = processFuzz(audioFile.samples[channel][i], fuzzGain);
            }
        }
    } 
    else if (effectName == "distortion_effect") {
        std::cout << "Applying Distortion Effect (Soft Clipping)...\n";
        float distGain = 2.5f; 
        for (int channel = 0; channel < numChannels; channel++) {
            for (int i = 0; i < numSamples; i++) {
                audioFile.samples[channel][i] = processDistortion(audioFile.samples[channel][i], distGain);
            }
        }
    }
    else if (effectName == "wavefolding_effect") {
        std::cout << "Applying Wavefolding Effect...\n";
        // A gain of 3.0 to 5.0 will create clearly visible multiple folds
        float waveGain = 3.0f; 
        for (int channel = 0; channel < numChannels; channel++) {
            for (int i = 0; i < numSamples; i++) {
                audioFile.samples[channel][i] = processWavefolding(audioFile.samples[channel][i], waveGain);
            }
        }
    }
    else {
        std::cerr << "Error: Unknown effect '" << effectName << "'\n";
        return 1;
    }

    audioFile.save(outputFile);
    std::cout << "Success! Effect applied and saved to " << outputFile << "\n";

    return 0;
}
