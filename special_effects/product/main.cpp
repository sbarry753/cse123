<<<<<<< HEAD
#include "daisy_seed.h"
#include "fuzz_effect.h"
#include "distortion_effect.h"
#include "wavefolding_effect.h"
#include "agc.h"

using namespace daisy;

DaisySeed hw;
AutoGainControl agc[2]; // one per channel

// This replaces your for loop the hardware calls it automatically
void AudioCallback(AudioHandle::InputBuffer in,
                   AudioHandle::OutputBuffer out,
                   size_t size) {

    // Read knob on ADC pin 0 for gain (maps 0.0-1.0 to 1.0-20.0)
    float userGain = 1.0f + hw.adc.GetFloat(0) * 19.0f;

    for (size_t i = 0; i < size; i++) {
        for (int ch = 0; ch < 2; ch++) {
            float cleanSample = in[ch][i];
            
            // For now hardcode an effect later a switch can read a physical switch pin
            float shapedSample = processDistortion(cleanSample, userGain);
            
            out[ch][i] = agc[ch].process(cleanSample, shapedSample);
        }
    }
}

int main() {
    hw.Init();
    hw.SetAudioBlockSize(4);

    // Set up one ADC pin for the gain knob
    AdcChannelConfig adcConfig;
    adcConfig.InitSingle(hw.GetPin(15)); // pin 15 is a common ADC pin on the Seed
    hw.adc.Init(&adcConfig, 1);
    hw.adc.Start();

    hw.StartAudio(AudioCallback);

    while (true) {} // everything happens in the callback
=======
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
>>>>>>> b786ae4c5ff541eba33bb3089403792ad699f197
}
