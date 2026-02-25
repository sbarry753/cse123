#include <cstddef>
#include <cstring>

#include "daisy_seed.h"

#include "effects.h"

using namespace daisy;
using namespace daisy::seed;

#define POT1 A6

constexpr size_t BLOCK_SIZE = 16;
constexpr auto SAMPLE_RATE = SaiHandle::Config::SampleRate::SAI_48KHZ;

DaisySeed hw;
CpuLoadMeter cpu_load;
Passthrough effect;
daisysp::Oscillator osc;

static float in_buf[2][BLOCK_SIZE];
static const float* const in_ptrs[2] = {in_buf[0], in_buf[1]};
static AudioHandle::InputBuffer effect_in = in_ptrs;

static void initADC() {
	AdcChannelConfig adc;
	adc.InitSingle(POT1);
	hw.adc.Init(&adc, 1);
    hw.adc.Start();
}

static float processADC() {
	return daisysp::fmap(hw.adc.GetFloat(0), 0.0f, 1.0f, daisysp::Mapping::EXP);
}

static inline void generateInput(size_t size) {
	for (size_t i = 0; i < size; i++)  {
		const float s = osc.Process();
		in_buf[0][i] = s;
		in_buf[1][i] = s;
	}
}

static void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size) {
	cpu_load.OnBlockStart();
	float pot_val = processADC();
	generateInput(size);
	effect.process_audio(effect_in, out, size, pot_val);
	cpu_load.OnBlockEnd();
}

int main(void) {
	hw.Init();
	hw.StartLog();
	hw.SetAudioBlockSize(BLOCK_SIZE);
	hw.SetAudioSampleRate(SAMPLE_RATE);
	effect.init(hw.AudioSampleRate());
	cpu_load.Init(hw.AudioSampleRate(), hw.AudioBlockSize());
	initADC();

	osc.Init((float)SAMPLE_RATE);
	osc.SetWaveform(daisysp::Oscillator::WAVE_SIN);
	osc.SetFreq(220.0f);
	osc.SetAmp(0.4f);

	hw.StartAudio(AudioCallback);

	int counter = 0;

	while(1) {
		const float avgLoad = cpu_load.GetAvgCpuLoad();
		const float maxLoad = cpu_load.GetMaxCpuLoad();
		const float minLoad = cpu_load.GetMinCpuLoad();
		hw.PrintLine("Processing Load %:");
		hw.PrintLine("Max: " FLT_FMT3, FLT_VAR3(maxLoad * 100.0f));
		hw.PrintLine("Avg: " FLT_FMT3, FLT_VAR3(avgLoad * 100.0f));
		hw.PrintLine("Min: " FLT_FMT3, FLT_VAR3(minLoad * 100.0f));
		hw.PrintLine("");
		counter++;
		System::Delay(1000);

		if (counter == 10) {
			hw.PrintLine("Resetting CPU Load...");
			counter = 0;
			cpu_load.Reset();
		}
	}
}
