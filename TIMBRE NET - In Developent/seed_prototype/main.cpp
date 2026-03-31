#include "daisysp.h"
#include "daisy_pod.h"

using namespace daisy;
using namespace daisysp;
DaisyPod hw;

constexpr int BLOCK_SIZE = 256;

void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size)
{
	// Allows processing of buttons/knobs on Pod
	hw.ProcessAllControls();

	for (size_t i = 0; i < size; i++)
	{	
		// See Pod pinout for which 3.5mm jacks these correspond to
		// Right now, just sends input directly to output
		out[0][i] = in[0][i]; // AUDIO_OUT_L and AUDIO_IN_L
		out[1][i] = in[1][i]; // AUDIO_OUT_R and AUDIO_IN_R
	}
}

int main(void)
{

	hw.Init();
	hw.SetAudioBlockSize(256); 
	hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
	hw.StartAdc();

	hw.StartAudio(AudioCallback);

	while(1) {
		hw.led1.Set(0,1,0); hw.led2.Set(0,1,0); hw.UpdateLeds();
        System::Delay(300);
        hw.led1.Set(0,0,0); hw.led2.Set(0,0,0); hw.UpdateLeds();
	}
}