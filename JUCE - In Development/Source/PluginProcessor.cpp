#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cstring>

//==============================================================================
Demo3AudioProcessor::Demo3AudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor(BusesProperties()
#if ! JucePlugin_IsMidiEffect
#if ! JucePlugin_IsSynth
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
#endif
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)
#endif
    )
#endif
    , apvts(*this, nullptr, "PARAMS", createParameterLayout())
{
}

Demo3AudioProcessor::~Demo3AudioProcessor() = default;

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout Demo3AudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;

    p.push_back(std::make_unique<juce::AudioParameterFloat>(
        "gateDb", "Gate (dB)", juce::NormalisableRange<float>(-80.0f, 0.0f, 0.1f), -45.0f));

    p.push_back(std::make_unique<juce::AudioParameterFloat>(
        "hystDb", "Gate Hysteresis (dB)", juce::NormalisableRange<float>(0.0f, 20.0f, 0.1f), 8.0f));

    p.push_back(std::make_unique<juce::AudioParameterFloat>(
        "wet", "Wet", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 1.0f));

    p.push_back(std::make_unique<juce::AudioParameterFloat>(
        "dry", "Dry", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.0f));

    p.push_back(std::make_unique<juce::AudioParameterInt>(
        "rootMidi", "Root MIDI Note", 0, 127, 60));

    p.push_back(std::make_unique<juce::AudioParameterFloat>(
        "attackMs", "Attack (ms)", juce::NormalisableRange<float>(0.1f, 200.0f, 0.1f, 0.5f), 8.0f));

    p.push_back(std::make_unique<juce::AudioParameterFloat>(
        "releaseMs", "Release (ms)", juce::NormalisableRange<float>(5.0f, 3000.0f, 1.0f, 0.5f), 200.0f));

    p.push_back(std::make_unique<juce::AudioParameterFloat>(
        "glideMs", "Glide (ms)", juce::NormalisableRange<float>(0.0f, 200.0f, 0.1f, 0.6f), 25.0f));

    // Kept for compatibility with your existing editor, but unused in this mono version:
    p.push_back(std::make_unique<juce::AudioParameterInt>(
        "maxVoices", "Max Voices", 1, 8, 1));

    // Kept for compatibility, but unused:
    p.push_back(std::make_unique<juce::AudioParameterFloat>(
        "peakDb", "Peak Threshold (dB)", juce::NormalisableRange<float>(-90.0f, -10.0f, 0.1f), -45.0f));

    // Kept for compatibility, but unused:
    p.push_back(std::make_unique<juce::AudioParameterBool>(
        "harmonicSuppress", "Harmonic Suppress", true));

    return { p.begin(), p.end() };
}

//==============================================================================
const juce::String Demo3AudioProcessor::getName() const { return JucePlugin_Name; }
bool Demo3AudioProcessor::acceptsMidi() const { return false; }
bool Demo3AudioProcessor::producesMidi() const { return false; }
bool Demo3AudioProcessor::isMidiEffect() const { return false; }
double Demo3AudioProcessor::getTailLengthSeconds() const { return 0.0; }
int Demo3AudioProcessor::getNumPrograms() { return 1; }
int Demo3AudioProcessor::getCurrentProgram() { return 0; }
void Demo3AudioProcessor::setCurrentProgram(int) {}
const juce::String Demo3AudioProcessor::getProgramName(int) { return {}; }
void Demo3AudioProcessor::changeProgramName(int, const juce::String&) {}

//==============================================================================
void Demo3AudioProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    sr = sampleRate;

    ring.clear();
    ringWrite = 0;
    hopCounter = 0;

    voice.reset(sr);
}

void Demo3AudioProcessor::releaseResources() {}

#ifndef JucePlugin_PreferredChannelConfigurations
bool Demo3AudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

#if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
#endif

    return true;
}
#endif

//==============================================================================
float Demo3AudioProcessor::computeRmsDb(const float* x, int N) const
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        sum += (double)x[i] * (double)x[i];

    const double rms = std::sqrt(sum / (double)N);
    const double r = juce::jmax(rms, 1.0e-12);
    return (float)(20.0 * std::log10(r));
}

//==============================================================================
std::pair<float, float> Demo3AudioProcessor::estimatePitchHz(const float* x, int N, float minHz, float maxHz)
{
    // Copy and window
    for (int i = 0; i < fftSize; ++i)
        timeBuf[(size_t)i] = (i < N) ? x[i] : 0.0f;

    // DC remove
    double mean = 0.0;
    for (int i = 0; i < N; ++i) mean += timeBuf[(size_t)i];
    mean /= juce::jmax(1, N);
    for (int i = 0; i < N; ++i) timeBuf[(size_t)i] -= (float)mean;

    window.multiplyWithWindowingTable(timeBuf.data(), fftSize);

    // FFT
    std::fill(freqBuf.begin(), freqBuf.end(), 0.0f);
    std::memcpy(freqBuf.data(), timeBuf.data(), sizeof(float) * (size_t)fftSize);
    fft.performRealOnlyForwardTransform(freqBuf.data());

    // Power spectrum: freqWork = X * conj(X)
    // For real-only layout: bins are stored as interleaved re/imag for k>=1.
    std::fill(freqWork.begin(), freqWork.end(), 0.0f);
    freqWork[0] = freqBuf[0] * freqBuf[0]; // DC power
    freqWork[1] = 0.0f;

    const int binCount = fftSize / 2;
    for (int k = 1; k < binCount; ++k)
    {
        const float re = freqBuf[(size_t)(2 * k)];
        const float im = freqBuf[(size_t)(2 * k + 1)];
        const float p = re * re + im * im;
        freqWork[(size_t)(2 * k)] = p;      // real
        freqWork[(size_t)(2 * k + 1)] = 0.0f; // imag = 0
    }

    // IFFT to get autocorrelation
    fft.performRealOnlyInverseTransform(freqWork.data());

    // Normalize autocorr by r0
    const float r0 = juce::jmax(freqWork[0], 1.0e-12f);
    for (int i = 0; i < fftSize; ++i)
        autocorr[(size_t)i] = freqWork[(size_t)i] / r0;

    const int minLag = (int)std::floor(sr / juce::jmax(maxHz, 1.0f));
    const int maxLag = (int)std::ceil(sr / juce::jmax(minHz, 1.0f));

    if (minLag < 2 || maxLag >= fftSize - 2 || minLag >= maxLag)
        return { 0.0f, 0.0f };

    // Find best peak in autocorr within [minLag, maxLag]
    int bestLag = -1;
    float bestVal = 0.0f;

    // optional: skip small lags near 0 where autocorr is high
    for (int lag = minLag; lag <= maxLag; ++lag)
    {
        const float v = autocorr[(size_t)lag];

        // local maxima check
        if (lag > minLag && lag < maxLag)
        {
            if (v > autocorr[(size_t)(lag - 1)] && v > autocorr[(size_t)(lag + 1)])
            {
                if (v > bestVal)
                {
                    bestVal = v;
                    bestLag = lag;
                }
            }
        }
    }

    // Confidence heuristics:
    // - bestVal is normalized autocorr peak, typically 0..1
    // - require a minimum to avoid noise
    if (bestLag < 0 || bestVal < 0.25f)
        return { 0.0f, 0.0f };

    // Refine lag with parabolic interpolation
    const float mL = autocorr[(size_t)(bestLag - 1)];
    const float mC = autocorr[(size_t)(bestLag)];
    const float mR = autocorr[(size_t)(bestLag + 1)];
    const float d = parabolicInterp(mL, mC, mR);
    const float refinedLag = (float)bestLag + d;

    const float hz = (refinedLag > 1.0f) ? (float)(sr / (double)refinedLag) : 0.0f;
    if (hz < minHz || hz > maxHz)
        return { 0.0f, 0.0f };

    return { hz, juce::jlimit(0.0f, 1.0f, bestVal) };
}

//==============================================================================
bool Demo3AudioProcessor::loadSampleFromFile(const juce::File& file)
{
    juce::AudioFormatManager fm;
    fm.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader(fm.createReaderFor(file));
    if (!reader)
        return false;

    auto newBuf = std::make_shared<juce::AudioBuffer<float>>((int)reader->numChannels,
        (int)reader->lengthInSamples);

    reader->read(newBuf.get(), 0, (int)reader->lengthInSamples, 0, true, true);

    {
        const juce::SpinLock::ScopedLockType sl(sampleLock);
        samplePtr = std::move(newBuf);
    }

    voice.reset(sr);
    return true;
}

//==============================================================================
void Demo3AudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const int totalIn = getTotalNumInputChannels();
    const int totalOut = getTotalNumOutputChannels();
    const int numSamples = buffer.getNumSamples();

    for (int ch = totalIn; ch < totalOut; ++ch)
        buffer.clear(ch, 0, numSamples);

    // Snapshot sample once per block
    std::shared_ptr<juce::AudioBuffer<float>> localSample;
    {
        const juce::SpinLock::ScopedLockType sl(sampleLock);
        localSample = samplePtr;
    }
    if (!localSample || localSample->getNumSamples() < 16)
        return;

    // Parameters
    const float gateDb = apvts.getRawParameterValue("gateDb")->load();
    const float hystDb = apvts.getRawParameterValue("hystDb")->load();
    const float wet = apvts.getRawParameterValue("wet")->load();
    const float dry = apvts.getRawParameterValue("dry")->load();
    const int rootMidi = (int)apvts.getRawParameterValue("rootMidi")->load();
    const float attackMs = apvts.getRawParameterValue("attackMs")->load();
    const float relMs = apvts.getRawParameterValue("releaseMs")->load();
    float glideMs = apvts.getRawParameterValue("glideMs")->load();

    // Update ADSR
    voice.adsrParams.attack = attackMs / 1000.0f;
    voice.adsrParams.decay = 0.05f;
    voice.adsrParams.sustain = 0.9f;
    voice.adsrParams.release = relMs / 1000.0f;
    voice.adsr.setParameters(voice.adsrParams);

    // Copy dry input
    juce::AudioBuffer<float> dryBuf;
    dryBuf.makeCopyOf(buffer);

    // Clear output to render wet
    for (int ch = 0; ch < totalOut; ++ch)
        buffer.clear(ch, 0, numSamples);

    const float* inL = dryBuf.getReadPointer(0);
    const float* inR = (totalIn > 1) ? dryBuf.getReadPointer(1) : nullptr;

    // Root of the loaded sample
    const float rootHz = midiToHz((float)rootMidi);

    // Gate state (persistent)
    static bool globalGateOpen = false;

    // Pitch follower settings
    constexpr float minHz = 70.0f;
    constexpr float maxHz = 1200.0f;
    constexpr float minConfidence = 0.30f; // higher = more stable, less responsive

    // Feed ring + analyze every hop
    for (int i = 0; i < numSamples; ++i)
    {
        float mono = inL[i];
        if (inR) mono = 0.5f * (mono + inR[i]);

        ring.setSample(0, ringWrite, mono);
        ringWrite = (ringWrite + 1) % fftSize;

        hopCounter++;
        if (hopCounter >= hopSize)
        {
            hopCounter = 0;

            // grab latest fftSize samples from ring into timeBuf
            for (int n = 0; n < fftSize; ++n)
            {
                const int idx = (ringWrite + n) % fftSize;
                timeBuf[(size_t)n] = ring.getSample(0, idx);
            }

            const float rmsDb = computeRmsDb(timeBuf.data(), fftSize);

            const float gateOpenDb = gateDb;
            const float gateCloseDb = gateDb - hystDb;

            const bool wasOpen = globalGateOpen;

            if (!globalGateOpen)
            {
                if (rmsDb > gateOpenDb) globalGateOpen = true;
            }
            else
            {
                if (rmsDb < gateCloseDb) globalGateOpen = false;
            }

            if (wasOpen && !globalGateOpen)
            {
                voice.gateOpen = false;
                voice.adsr.noteOff();
            }
            else if (globalGateOpen)
            {
                auto [hz, conf] = estimatePitchHz(timeBuf.data(), fftSize, minHz, maxHz);

                // If confidence is low, keep previous target (prevents chatter)
                if (hz > 0.0f && conf >= minConfidence)
                    voice.targetHz = hz;

                if (!voice.gateOpen)
                {
                    // Gate just opened: start voice cleanly
                    voice.gateOpen = true;
                    voice.active = true;

                    const int sampleLen = localSample->getNumSamples();
                    const float* s0 = localSample->getReadPointer(0);
                    const int start = findNearestZeroCrossing(s0, sampleLen, 0, juce::jmin(2048, sampleLen - 1));
                    voice.playPos = (double)start;

                    voice.deClickLen = (int)(0.003 * sr);
                    voice.deClickSamplesLeft = voice.deClickLen;

                    voice.loopFadeLen = 256;
                    voice.loopFadeSamplesLeft = 0;

                    voice.smoothHz = (voice.targetHz > 0.0f) ? voice.targetHz : 0.0f;

                    voice.adsr.noteOn();
                }
            }
        }
    }

    //========================
    // RENDER: single voice
    const int sampleLen = localSample->getNumSamples();
    const int sampleCh = localSample->getNumChannels();
    const float* sampL = localSample->getReadPointer(0);
    const float* sampR = localSample->getReadPointer(sampleCh > 1 ? 1 : 0);

    glideMs = juce::jmax(glideMs, 8.0f);
    const double glideSamples = (glideMs / 1000.0) * sr;
    const double g = (glideSamples <= 1.0) ? 0.0 : std::exp(-1.0 / glideSamples);

    for (int i = 0; i < numSamples; ++i)
    {
        float wetL = 0.0f, wetR = 0.0f;

        if (voice.active)
        {
            // Smooth pitch (Hz domain)
            if (voice.smoothHz <= 0.0f) voice.smoothHz = voice.targetHz;
            voice.smoothHz = (float)(g * (double)voice.smoothHz + (1.0 - g) * (double)voice.targetHz);

            const double pitchRatio = (rootHz > 0.0f && voice.smoothHz > 0.0f)
                ? (double)(voice.smoothHz / rootHz)
                : 1.0;

            const float env = voice.adsr.getNextSample();

            float deClick = 1.0f;
            if (voice.deClickSamplesLeft > 0 && voice.deClickLen > 0)
            {
                deClick = 1.0f - (float)voice.deClickSamplesLeft / (float)voice.deClickLen;
                voice.deClickSamplesLeft--;
            }

            const float gain = env * deClick;

            if (gain <= 1.0e-6f && !voice.adsr.isActive())
            {
                voice.active = false;
                voice.targetHz = 0.0f;
            }
            else
            {
                // Looping with small crossfade at wrap to avoid clicks
                if (voice.playPos >= (double)(sampleLen - 3))
                {
                    voice.playPos = std::fmod(voice.playPos, (double)(sampleLen - 3));
                    voice.loopFadeSamplesLeft = voice.loopFadeLen; // crossfade in after wrap
                }

                int p1 = juce::jlimit(0, sampleLen - 1, (int)voice.playPos);
                int p0 = juce::jlimit(0, sampleLen - 1, p1 - 1);
                int p2 = juce::jlimit(0, sampleLen - 1, p1 + 1);
                int p3 = juce::jlimit(0, sampleLen - 1, p1 + 2);

                const float frac = (float)(voice.playPos - (double)p1);

                float l = cubicInterp(sampL[p0], sampL[p1], sampL[p2], sampL[p3], frac);
                float r = cubicInterp(sampR[p0], sampR[p1], sampR[p2], sampR[p3], frac);

                // crossfade ramp-in just after wrap (helps with non-loop-perfect samples)
                if (voice.loopFadeSamplesLeft > 0 && voice.loopFadeLen > 0)
                {
                    const float a = 1.0f - (float)voice.loopFadeSamplesLeft / (float)voice.loopFadeLen;
                    l *= a;
                    r *= a;
                    voice.loopFadeSamplesLeft--;
                }

                wetL += l * gain;
                wetR += r * gain;

                voice.playPos += pitchRatio;
            }
        }

        const float dL = dryBuf.getSample(0, i);
        const float dR = (dryBuf.getNumChannels() > 1) ? dryBuf.getSample(1, i) : dL;

        buffer.setSample(0, i, wet * wetL + dry * dL);
        if (totalOut > 1)
            buffer.setSample(1, i, wet * wetR + dry * dR);
    }
}

//==============================================================================
bool Demo3AudioProcessor::hasEditor() const { return true; }

juce::AudioProcessorEditor* Demo3AudioProcessor::createEditor()
{
    return new Demo3AudioProcessorEditor(*this);
}

//==============================================================================
void Demo3AudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    juce::MemoryOutputStream mos(destData, true);
    apvts.state.writeToStream(mos);
}

void Demo3AudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    auto vt = juce::ValueTree::readFromData(data, (size_t)sizeInBytes);
    if (vt.isValid())
        apvts.replaceState(vt);
}

//==============================================================================
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Demo3AudioProcessor();
}
