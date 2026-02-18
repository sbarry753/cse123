#include "daisysp.h"

#if defined(POD)
#include "daisy_pod.h"
using namespace daisy;
using namespace daisysp;

DaisyPod hw;
#elif defined(SEED)
#include "daisy_seed.h"
using namespace daisy;
using namespace daisysp;

DaisySeed hw;
#else
	#error "Must specify POD or SEED"
#endif

// Put effect logic here
void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size)
{
	#if defined(POD)
	// Only needed for Pod
	// Allows processing of buttons/knobs on Pod
	hw.ProcessAllControls(); 
	#endif

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
	// Blink for seed
	#if defined(SEED)
	bool led_state;
    led_state = true;
	#endif

	hw.Init();
	hw.SetAudioBlockSize(4); // number of samples handled per callback
	hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

	#if defined(POD)
	hw.StartAdc(); // Only needed for Pod
	#endif

	hw.StartAudio(AudioCallback);

	while(1) {
		// Blink for seed
		#if defined(SEED)
		hw.SetLed(led_state);
        led_state = !led_state;
        System::Delay(500);
		#endif
	}
}
