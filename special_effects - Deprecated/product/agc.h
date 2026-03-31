#include <cmath>

class AutoGainControl {
private:
    float envIn = 0.0f;
    float envOut = 0.0f;
    
    // These dictate how fast the "volume knob" turns. 
    // Small numbers mean it tracks the average volume over time smoothly 
    // without warping the shape of the individual sound waves.
    float attack = 0.01f;
    float release = 0.001f;

public:
    float process(float cleanSample, float processedSample) {
        // 1. Track the smooth envelope of the clean input
        float absIn = std::abs(cleanSample);
        if (absIn > envIn) envIn = attack * absIn + (1.0f - attack) * envIn;
        else               envIn = release * absIn + (1.0f - release) * envIn;

        // 2. Track the smooth envelope of the fuzzed output
        float absOut = std::abs(processedSample);
        if (absOut > envOut) envOut = attack * absOut + (1.0f - attack) * envOut;
        else                 envOut = release * absOut + (1.0f - release) * envOut;

        // 3. Prevent division by zero during dead silence
        if (envOut < 0.0001f) return processedSample;

        // 4. Calculate the real-time volume adjustment needed
        float makeupGain = envIn / envOut;

        // 5. Apply the volume fix to the fuzzed wave
        return processedSample * makeupGain;
    }
};

