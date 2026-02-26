#include <cstddef>
#include <cstring>

#include "daisy_seed.h"

#include "effects.h"

using namespace daisy;
using namespace daisy::seed;

#define POT1 A6

constexpr size_t BLOCK_SIZE = 32;
constexpr auto SAMPLE_RATE = SaiHandle::Config::SampleRate::SAI_48KHZ;

DaisySeed hw;
CpuLoadMeter cpu_load;
Reverb effect;
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

	osc.Init(hw.AudioSampleRate());
	osc.SetWaveform(daisysp::Oscillator::WAVE_SIN);
	osc.SetFreq(220.0f);
	osc.SetAmp(0.4f);

	hw.StartAudio(AudioCallback);

	int counter = 0;
	float total_avg = 0.0f;
	float max_load = 0.0f;
	float min_load = 10000.0f;

	while(1) {
		const float avg_sample = cpu_load.GetAvgCpuLoad();
		if (!std::isnan(avg_sample)) {
			total_avg += avg_sample;
			counter++;
		}
		const float cur_max = cpu_load.GetMaxCpuLoad();
		const float cur_min = cpu_load.GetMinCpuLoad();
		if (!std::isnan(cur_max)) max_load = std::max(cur_max, max_load);
		if (!std::isnan(cur_min)) min_load = std::min(cur_min, min_load);

		System::Delay(500);

		if (counter == 10) {
			float avg_load = total_avg / (float)counter;
			hw.PrintLine("Processing Load %:");
			hw.PrintLine("Max: " FLT_FMT3, FLT_VAR3(max_load * 100.0f));
			hw.PrintLine("Avg: " FLT_FMT3, FLT_VAR3(avg_load * 100.0f));
			hw.PrintLine("Min: " FLT_FMT3, FLT_VAR3(min_load * 100.0f));
			hw.PrintLine("");
			counter = 0;
			total_avg = 0.0f;
			max_load = 0.0f;
			min_load = 10000.0f;
			cpu_load.Reset();
		}
	}
}
