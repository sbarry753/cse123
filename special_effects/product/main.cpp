//#include "DaisySP/daisysp.h"
#include <iostream>
#include <string>
#include "AudioFile.h"
#include "fuzz_effect.h"
#include "distortion_effect.h"
#include "wavefolding_effect.h"
#include "agc.h"

int main(int argc, char* argv[]) {
    // We now expect 4 arguments: the program, the gain, the effect, and the file
    if (argc < 4) {
        std::cerr << "Usage: ./effect_processor <gain> <effect_name> <path_to_input_file.wav>\n";
        std::cerr << "Example: ./effect_processor 10.5 fuzz_effect audio.wav\n";
        return 1;
    }

    // 1. Parse the arguments
    float userGain = 0.0f;
    try {
        userGain = std::stof(argv[1]); // Convert the text input into a decimal number
    } catch (const std::exception& e) {
        std::cerr << "Error: Gain must be a valid number.\n";
        return 1;
    }
    
    std::string effectName = argv[2];
    std::string inputFile = argv[3];
    std::string outputFile = "output.wav";

    // 2. Validate the effect name before doing any heavy lifting
    if (effectName != "fuzz_effect" && effectName != "distortion_effect" && effectName != "wavefolding_effect") {
        std::cerr << "Error: Unknown effect '" << effectName << "'\n";
        std::cerr << "Available effects: fuzz_effect, distortion_effect, wavefolding_effect\n";
        return 1;
    }

    AudioFile<float> audioFile;

    if (!audioFile.load(inputFile)) {
        std::cerr << "Error: Could not load " << inputFile << "\n";
        return 1;
    }

    int numChannels = audioFile.getNumChannels();
    int numSamples = audioFile.getNumSamplesPerChannel();

    // Create our real-time volume leveler
    AutoGainControl agc;

    std::cout << "Processing audio in real-time with a gain of " << userGain << "...\n";

    for (int channel = 0; channel < numChannels; channel++) {
        for (int i = 0; i < numSamples; i++) {
            float cleanSample = audioFile.samples[channel][i];
            float shapedSample = 0.0f;

            // 3. Apply the raw math using your custom userGain variable
            if (effectName == "fuzz_effect") {
                shapedSample = processFuzz(cleanSample, userGain);
            } 
            else if (effectName == "distortion_effect") {
                shapedSample = processDistortion(cleanSample, userGain);
            }
            else if (effectName == "wavefolding_effect") {
                shapedSample = processWavefolding(cleanSample, userGain);
            }
            
            // 4. Pass the clean and shaped samples to the AGC to fix the VOLUME
            audioFile.samples[channel][i] = agc.process(cleanSample, shapedSample);
        }
    }

    audioFile.save(outputFile);
    std::cout << "Success! Effect applied and saved to " << outputFile << "\n";

    return 0;
}
