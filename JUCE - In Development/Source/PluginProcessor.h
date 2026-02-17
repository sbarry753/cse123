#pragma once
#include <JuceHeader.h>
#include <memory>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

class Demo3AudioProcessor : public juce::AudioProcessor
{
public:
    Demo3AudioProcessor();
    ~Demo3AudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
#endif

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    bool loadSampleFromFile(const juce::File& file);

    juce::AudioProcessorValueTreeState apvts;
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

private:
    //==============================================================================
    // Sample data (atomic-ish swap via shared_ptr snapshot per block)
    juce::SpinLock sampleLock;
    std::shared_ptr<juce::AudioBuffer<float>> samplePtr;

    double sr = 44100.0;

    //==============================================================================
    // Helpers
    float computeRmsDb(const float* x, int N) const;

    static float midiToHz(float midi)
    {
        return 440.0f * std::pow(2.0f, (midi - 69.0f) / 12.0f);
    }

    // Catmull-Rom cubic interpolation
    static float cubicInterp(float y0, float y1, float y2, float y3, float t)
    {
        const float t2 = t * t;
        const float t3 = t2 * t;
        return 0.5f * ((2.0f * y1)
            + (-y0 + y2) * t
            + (2.0f * y0 - 5.0f * y1 + 4.0f * y2 - y3) * t2
            + (-y0 + 3.0f * y1 - 3.0f * y2 + y3) * t3);
    }

    static float parabolicInterp(float mL, float mC, float mR)
    {
        const float denom = (mL - 2.0f * mC + mR);
        if (std::abs(denom) < 1.0e-12f) return 0.0f;
        return 0.5f * (mL - mR) / denom;
    }

    static int findNearestZeroCrossing(const float* x, int len, int start, int searchRadius)
    {
        if (!x || len <= 1) return 0;
        start = juce::jlimit(0, len - 1, start);

        const int lo = juce::jlimit(0, len - 1, start - searchRadius);
        const int hi = juce::jlimit(0, len - 1, start + searchRadius);

        auto sign = [](float v) { return (v >= 0.0f) ? 1 : -1; };

        int best = start;
        int bestDist = 1 << 30;

        for (int i = lo + 1; i <= hi; ++i)
        {
            const float a = x[i - 1];
            const float b = x[i];

            if (sign(a) != sign(b))
            {
                const int dist = std::abs(i - start);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    best = i;
                }
            }
        }
        return best;
    }

    //==============================================================================
    // Monophonic pitch follower (FFT autocorrelation)
    static constexpr int fftOrder = 12;             // 4096
    static constexpr int fftSize = 1 << fftOrder;
    static constexpr int hopSize = 256;

    juce::dsp::FFT fft{ fftOrder };
    juce::dsp::WindowingFunction<float> window{ fftSize, juce::dsp::WindowingFunction<float>::hann, true };

    juce::AudioBuffer<float> ring{ 1, fftSize };
    int ringWrite = 0;
    int hopCounter = 0;

    std::array<float, fftSize> timeBuf{};
    std::array<float, 2 * fftSize> freqBuf{};      // real-only FFT uses 2N
    std::array<float, 2 * fftSize> freqWork{};     // for power spectrum
    std::array<float, fftSize> autocorr{};         // time-domain autocorrelation (first N useful)

    // returns (hz, confidence[0..1]); hz=0 if no reliable pitch
    std::pair<float, float> estimatePitchHz(const float* x, int N, float minHz, float maxHz);

    //==============================================================================
    // Single sample voice (continuous pitch)
    struct MonoVoice
    {
        bool active = false;
        bool gateOpen = false;

        double playPos = 0.0;

        float targetHz = 0.0f;
        float smoothHz = 0.0f;

        // small de-click on gate open
        int deClickSamplesLeft = 0;
        int deClickLen = 0;

        // loop crossfade
        int loopFadeSamplesLeft = 0;
        int loopFadeLen = 0;

        juce::ADSR adsr;
        juce::ADSR::Parameters adsrParams{ 0.008f, 0.05f, 0.9f, 0.18f };

        void reset(double sampleRate)
        {
            active = false;
            gateOpen = false;
            playPos = 0.0;
            targetHz = 0.0f;
            smoothHz = 0.0f;
            deClickSamplesLeft = 0;
            deClickLen = 0;
            loopFadeSamplesLeft = 0;
            loopFadeLen = 0;

            adsr.setSampleRate(sampleRate);
            adsr.reset();
            adsr.setParameters(adsrParams);
        }
    };

    MonoVoice voice;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Demo3AudioProcessor)
};
