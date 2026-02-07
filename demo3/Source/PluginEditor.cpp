#include "PluginEditor.h"
#include <memory>

static void styleKnob(juce::Slider& s, const juce::String& suffix = {})
{
    s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 80, 20);
    s.setTextValueSuffix(suffix);
}

static void styleNumber(juce::Slider& s)
{
    s.setSliderStyle(juce::Slider::LinearHorizontal);
    s.setTextBoxStyle(juce::Slider::TextBoxRight, false, 70, 20);
}

Demo3AudioProcessorEditor::Demo3AudioProcessorEditor(Demo3AudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p)
{
    setSize(820, 300);

    styleKnob(gateSlider, " dB");
    styleKnob(hystSlider, " dB");
    styleKnob(wetSlider);
    styleKnob(drySlider);
    styleKnob(rootMidiSlider);
    styleKnob(attackSlider, " ms");
    styleKnob(releaseSlider, " ms");
    styleKnob(glideSlider, " ms");

    styleNumber(maxVoicesSlider);
    styleNumber(peakDbSlider);
    peakDbSlider.setTextValueSuffix(" dB");
    maxVoicesSlider.setTextValueSuffix(" voices");

    addAndMakeVisible(gateSlider);
    addAndMakeVisible(hystSlider);
    addAndMakeVisible(wetSlider);
    addAndMakeVisible(drySlider);
    addAndMakeVisible(rootMidiSlider);
    addAndMakeVisible(attackSlider);
    addAndMakeVisible(releaseSlider);
    addAndMakeVisible(glideSlider);

    addAndMakeVisible(maxVoicesSlider);
    addAndMakeVisible(peakDbSlider);
    addAndMakeVisible(harmonicSuppressToggle);

    addAndMakeVisible(loadButton);

    gateAtt = std::make_unique<Attach>(audioProcessor.apvts, "gateDb", gateSlider);
    hystAtt = std::make_unique<Attach>(audioProcessor.apvts, "hystDb", hystSlider);
    wetAtt = std::make_unique<Attach>(audioProcessor.apvts, "wet", wetSlider);
    dryAtt = std::make_unique<Attach>(audioProcessor.apvts, "dry", drySlider);
    rootAtt = std::make_unique<Attach>(audioProcessor.apvts, "rootMidi", rootMidiSlider);
    atkAtt = std::make_unique<Attach>(audioProcessor.apvts, "attackMs", attackSlider);
    relAtt = std::make_unique<Attach>(audioProcessor.apvts, "releaseMs", releaseSlider);
    glideAtt = std::make_unique<Attach>(audioProcessor.apvts, "glideMs", glideSlider);

    maxVAtt = std::make_unique<Attach>(audioProcessor.apvts, "maxVoices", maxVoicesSlider);
    peakAtt = std::make_unique<Attach>(audioProcessor.apvts, "peakDb", peakDbSlider);

    harmAtt = std::make_unique<BtnAttach>(audioProcessor.apvts, "harmonicSuppress", harmonicSuppressToggle);

    // JUCE 7 safe FileChooser (keep alive)
    loadButton.onClick = [this]
        {
            auto chooser = std::make_shared<juce::FileChooser>(
                "Choose a sample (wav/aiff/...)",
                juce::File{},
                "*.wav;*.aiff;*.aif;*.flac");

            auto flags = juce::FileBrowserComponent::openMode
                | juce::FileBrowserComponent::canSelectFiles;

            chooser->launchAsync(flags, [this, chooser](const juce::FileChooser& fc)
                {
                    auto file = fc.getResult();
                    if (!file.existsAsFile())
                        return;

                    if (!audioProcessor.loadSampleFromFile(file))
                    {
                        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
                            "Load failed",
                            "Could not read that audio file.");
                    }
                });
        };
}

Demo3AudioProcessorEditor::~Demo3AudioProcessorEditor() = default;

void Demo3AudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);

    g.setFont(16.0f);
    g.drawText("Guitar -> Poly Talking Sample Synth (FFT peaks)", 14, 10, getWidth() - 28, 24, juce::Justification::left);

    g.setFont(12.0f);
    g.drawText("Gate", 20, 60, 80, 20, juce::Justification::centredLeft);
    g.drawText("Hyst", 120, 60, 80, 20, juce::Justification::centredLeft);
    g.drawText("Wet", 220, 60, 80, 20, juce::Justification::centredLeft);
    g.drawText("Dry", 320, 60, 80, 20, juce::Justification::centredLeft);
    g.drawText("Root", 420, 60, 80, 20, juce::Justification::centredLeft);

    g.drawText("Attack", 20, 165, 80, 20, juce::Justification::centredLeft);
    g.drawText("Release", 120, 165, 80, 20, juce::Justification::centredLeft);
    g.drawText("Glide", 220, 165, 80, 20, juce::Justification::centredLeft);

    g.drawText("Max voices", 420, 150, 120, 20, juce::Justification::centredLeft);
    g.drawText("Peak thresh", 420, 190, 120, 20, juce::Justification::centredLeft);
}

void Demo3AudioProcessorEditor::resized()
{
    const int row1Y = 80;
    const int row2Y = 185;

    gateSlider.setBounds(20, row1Y, 90, 80);
    hystSlider.setBounds(120, row1Y, 90, 80);
    wetSlider.setBounds(220, row1Y, 90, 80);
    drySlider.setBounds(320, row1Y, 90, 80);
    rootMidiSlider.setBounds(420, row1Y, 90, 80);

    attackSlider.setBounds(20, row2Y, 90, 80);
    releaseSlider.setBounds(120, row2Y, 90, 80);
    glideSlider.setBounds(220, row2Y, 90, 80);

    maxVoicesSlider.setBounds(540, 150, 220, 24);
    peakDbSlider.setBounds(540, 190, 220, 24);

    harmonicSuppressToggle.setBounds(540, 225, 180, 24);
    loadButton.setBounds(680, 255, 120, 32);
}
