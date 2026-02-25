#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

class Demo3AudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit Demo3AudioProcessorEditor(Demo3AudioProcessor&);
    ~Demo3AudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    Demo3AudioProcessor& audioProcessor;

    juce::TextButton loadButton{ "Load Sample" };

    juce::Slider gateSlider, hystSlider, wetSlider, drySlider, rootMidiSlider, attackSlider, releaseSlider, glideSlider;
    juce::Slider maxVoicesSlider, peakDbSlider;
    juce::ToggleButton harmonicSuppressToggle{ "Harmonic Suppress" };

    using Attach = juce::AudioProcessorValueTreeState::SliderAttachment;
    using BtnAttach = juce::AudioProcessorValueTreeState::ButtonAttachment;

    std::unique_ptr<Attach> gateAtt, hystAtt, wetAtt, dryAtt, rootAtt, atkAtt, relAtt, glideAtt;
    std::unique_ptr<Attach> maxVAtt, peakAtt;
    std::unique_ptr<BtnAttach> harmAtt;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Demo3AudioProcessorEditor)
};
