#include "daisy_pod.h"
#include "effects.h"

using namespace daisy;

constexpr size_t BLOCK_SIZE = 4;
constexpr auto SAMPLE_RATE = SaiHandle::Config::SampleRate::SAI_48KHZ;

DaisyPod hw;
CpuLoadMeter cpu_load;
Passthrough effect;

void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size) {
	cpu_load.OnBlockStart();
	hw.ProcessAllControls();
	effect.process_audio(in, out, size);
	cpu_load.OnBlockEnd();
}

int main(void)
{
	hw.Init();
	hw.seed.StartLog();
	hw.SetAudioBlockSize(BLOCK_SIZE);
	hw.SetAudioSampleRate(SAMPLE_RATE);
	cpu_load.Init(hw.AudioSampleRate(), hw.AudioBlockSize());
	hw.StartAdc();
	hw.StartAudio(AudioCallback);

	while(1) {
		const float avgLoad = cpu_load.GetAvgCpuLoad();
		const float maxLoad = cpu_load.GetMaxCpuLoad();
		const float minLoad = cpu_load.GetMinCpuLoad();

		hw.seed.PrintLine("Processing Load %:");
		hw.seed.PrintLine("Max: " FLT_FMT3, FLT_VAR3(maxLoad * 100.0f));
		hw.seed.PrintLine("Avg: " FLT_FMT3, FLT_VAR3(avgLoad * 100.0f));
		hw.seed.PrintLine("Min: " FLT_FMT3, FLT_VAR3(minLoad * 100.0f));

		System::Delay(500);
	}
}
