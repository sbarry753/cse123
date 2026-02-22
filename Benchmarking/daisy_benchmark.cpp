#include "daisy_pod.h"
#include "effects.h"

using namespace daisy;

constexpr size_t BLOCK_SIZE = 16;
constexpr auto SAMPLE_RATE = SaiHandle::Config::SampleRate::SAI_48KHZ;

DaisyPod hw;
CpuLoadMeter cpu_load;
Passthrough effect;

/*
* TODO:
* - Find guitar sample (5s) and place in Flash for common benchmarking input
*
*
*/ 

void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size) {
	cpu_load.OnBlockStart();
	// Don't allow use of Pod knobs + rotary encoder to keep benchmarking fair
	// hw.ProcessAllControls(); 
	effect.process_audio(in, out, size);
	cpu_load.OnBlockEnd();
}

int main(void) {
	hw.Init();
	hw.seed.StartLog(true);
	hw.SetAudioBlockSize(BLOCK_SIZE);
	hw.SetAudioSampleRate(SAMPLE_RATE);
	cpu_load.Init(hw.AudioSampleRate(), hw.AudioBlockSize());
	hw.StartAdc();
	hw.StartAudio(AudioCallback);

	int counter = 0;

	while(1) {
		while (counter < 5) {
			const float avgLoad = cpu_load.GetAvgCpuLoad();
			const float maxLoad = cpu_load.GetMaxCpuLoad();
			const float minLoad = cpu_load.GetMinCpuLoad();
			
			hw.seed.PrintLine("Processing Load %:");
			hw.seed.PrintLine("Max: " FLT_FMT3, FLT_VAR3(maxLoad * 100.0f));
			hw.seed.PrintLine("Avg: " FLT_FMT3, FLT_VAR3(avgLoad * 100.0f));
			hw.seed.PrintLine("Min: " FLT_FMT3, FLT_VAR3(minLoad * 100.0f));
			hw.seed.PrintLine("");
			counter++;
			System::Delay(500);
			
		} 
		
	}
}
