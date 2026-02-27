#include "daisy_seed.h"
#include "fuzz_effect.h"
#include "distortion_effect.h"
#include "wavefolding_effect.h"
#include "agc.h"

using namespace daisy;

DaisySeed hw;
AutoGainControl agc[2]; // one per channel

// This replaces your for loop — the hardware calls it automatically
void AudioCallback(AudioHandle::InputBuffer in,
                   AudioHandle::OutputBuffer out,
                   size_t size) {

    // Read knob on ADC pin 0 for gain (maps 0.0-1.0 to 1.0-20.0)
    float userGain = 1.0f + hw.adc.GetFloat(0) * 19.0f;

    for (size_t i = 0; i < size; i++) {
        for (int ch = 0; ch < 2; ch++) {
            float cleanSample = in[ch][i];
            
            // For now hardcode an effect — later a switch can read a physical switch pin
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
}
