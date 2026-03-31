//#include "DaisySP/daisysp.h"
#include <iostream>
#include <string>
#include "AudioFile.h"
#include "fuzz_effect.h"
#include "distortion_effect.h"
#include "wavefolding_effect.h"
#include "reverb_effect.h"
#include "delay_effect.h"
#include "agc.h"

#define strike_detection 0

// Stateful effects are allocated once per channel inside the loop.
// They are declared here as per-channel arrays to avoid repeated construction.
// MAX_CHANNELS = 1 — guitar signal is mono.
static constexpr int MAX_CHANNELS = 1;

static ReverbEffect reverbProcessors[MAX_CHANNELS];
static DelayEffect  delayProcessors[MAX_CHANNELS];

int main(int argc, char* argv[]) {
    // Expect 4 arguments: the program, the gain, the effect, and the file
    if (argc < 4) {
        std::cerr << "Usage: ./effect_processor <gain> <effect_name> <path_to_input_file.wav>\n";
        std::cerr << "Example: ./effect_processor 1.0 reverb_effect audio.wav\n";
        std::cerr << "Available effects: fuzz_effect, distortion_effect, wavefolding_effect, reverb_effect, delay_effect\n";
        return 1;
    }

    // 1. Parse gain argument
    float userGain = 0.0f;
    try {
        userGain = std::stof(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: Gain must be a valid number.\n";
        return 1;
    }

    std::string effectName = argv[2];
    std::string inputFile  = argv[3];
    std::string outputFile = "output.wav";

    // 2. Validate effect name
    if (effectName != "fuzz_effect"        &&
        effectName != "distortion_effect"  &&
        effectName != "wavefolding_effect" &&
        effectName != "reverb_effect"      &&
        effectName != "delay_effect") {
        std::cerr << "Error: Unknown effect '" << effectName << "'\n";
        std::cerr << "Available effects: fuzz_effect, distortion_effect, wavefolding_effect, reverb_effect, delay_effect\n";
        return 1;
    }

    // 3. Load audio file
    AudioFile<float> audioFile;
    if (!audioFile.load(inputFile)) {
        std::cerr << "Error: Could not load " << inputFile << "\n";
        return 1;
    }

    int numChannels = audioFile.getNumChannels();
    int numSamples  = audioFile.getNumSamplesPerChannel();

    if (numChannels > MAX_CHANNELS) {
        std::cerr << "Error: File has " << numChannels << " channels; max supported is " << MAX_CHANNELS << "\n";
        return 1;
    }

    const bool isStateful = (effectName == "reverb_effect" || effectName == "delay_effect");

    std::cout << "Processing '" << effectName << "' with gain " << userGain << "...\n";
	// Add this temporarily right after loading, before the processing loop
    std::cout << "Sample[0]: " << audioFile.samples[0][0] << "\n";
    std::cout << "Sample[1000]: " << audioFile.samples[0][1000] << "\n";

    for (int channel = 0; channel < numChannels; channel++) {
        // AGC is per-channel — instantiate here so each channel levels independently
        AutoGainControl agc;

        for (int i = 0; i < numSamples; i++) {
    float cleanSample = audioFile.samples[channel][i]; // already in [-1.0, 1.0]

    float processInput = isStateful ? agc.process(cleanSample, cleanSample) : cleanSample;

    float shapedSample = 0.0f;

    if (effectName == "fuzz_effect")
        shapedSample = processFuzz(processInput, userGain);
    else if (effectName == "distortion_effect")
        shapedSample = processDistortion(processInput, userGain);
    else if (effectName == "wavefolding_effect")
        shapedSample = processWavefolding(processInput, userGain);
    else if (effectName == "reverb_effect")
        shapedSample = processReverb(processInput, userGain, reverbProcessors[channel]);
    else if (effectName == "delay_effect")
        shapedSample = processDelay(processInput, userGain, delayProcessors[channel]);

    // No denormalization needed — AudioFile saves back to WAV correctly
    if (strike_detection) {
        audioFile.samples[channel][i] = shapedSample;
    } else {
        audioFile.samples[channel][i] = isStateful
            ? shapedSample
            : agc.process(cleanSample, shapedSample);
    	    }
	}
    }

    audioFile.save(outputFile);
    std::cout << "Success! Saved to " << outputFile << "\n";
    return 0;
}
