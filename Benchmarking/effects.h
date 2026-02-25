#pragma once

#include <cmath>
#include <cstddef>

#include "daisysp.h"


using namespace daisy;

struct BaseEffect {
    BaseEffect() = default;
    virtual ~BaseEffect() = default;
    virtual void init(float sample_rate) = 0;
    virtual void process_audio(AudioHandle::InputBuffer& in,
                               AudioHandle::OutputBuffer& out,
                               size_t size,
                               float pot_val) = 0;
};

struct Passthrough : BaseEffect {
    void init(float sample_rate) override {};

    inline void process_audio(AudioHandle::InputBuffer& in,
                       AudioHandle::OutputBuffer& out,
                       size_t size,
                       float pot_val) override {
        (void)pot_val;
        for (size_t i = 0; i < size; i++) {
            out[0][i] = in[0][i];
            out[1][i] = in[1][i];
        }
    }

};

/*
Low-Complexity Effects
*/
struct Distortion : BaseEffect {
    void init(float sample_rate) override {
        overdrive_.Init();   
    }

    inline void process_audio(AudioHandle::InputBuffer& in,
                              AudioHandle::OutputBuffer& out,
                       size_t size,
                       float pot_val) override {
        overdrive_.SetDrive(pot_val);
        for (size_t i = 0; i < size; i++) {
            out[0][i] = overdrive_.Process(in[0][i]);
            out[1][i] = overdrive_.Process(in[1][i]);
        }
    }

  private:
    daisysp::Overdrive overdrive_;
};

struct Bitcrush : BaseEffect {
    void init(float sample_rate) override {
        bitcrush_.Init(sample_rate);
    }

    inline void process_audio(AudioHandle::InputBuffer& in,
                              AudioHandle::OutputBuffer& out,
                       size_t size,
                       float pot_val) override {
        (void)in;
        bitcrush_.SetBitDepth(pot_val + 2);
        bitcrush_.SetCrushRate((pot_val + 2) * 2000);

        for(size_t i = 0; i < size; i++) {
            const float input = in[0][i];
            float sig_out = bitcrush_.Process(input);

            out[0][i] = sig_out;
            out[1][i] = sig_out;
        }
    }

  private:
    daisysp::Bitcrush bitcrush_;
};

/*
Medium-Complexity Effects
*/
struct Delay : BaseEffect {
    static constexpr size_t kMaxDelaySamples = 48000;

    void init(float sample_rate) override {
        delay_.Init();
    }

    inline void process_audio(AudioHandle::InputBuffer& in,
                              AudioHandle::OutputBuffer& out,
                       size_t size,
                       float pot_val) override {
        for(size_t i = 0; i < size; i++) {
            float input = in[0][i];
            const float del_out = delay_.Read();

            float sig_out = input + del_out;
            float feedback = (del_out * pot_val) + input;

            delay_.Write(feedback);

            out[0][i] = sig_out;
            out[1][i] = sig_out;
        }
    }

  private:
    daisysp::DelayLine<float, kMaxDelaySamples> delay_;
};

struct Chorus : BaseEffect {
    void init(float sample_rate) override {
        chorus_.Init(sample_rate);
        chorus_.SetLfoFreq(0.33f, 0.2f);
        chorus_.SetLfoDepth(1.0f, 1.0f);
        chorus_.SetDelay(0.75f, 0.9f);
    }

    inline void process_audio(AudioHandle::InputBuffer& in,
                              AudioHandle::OutputBuffer& out,
                       size_t size,
                       float pot_val) override {

        for(size_t i = 0; i < size; i++) {
            chorus_.Process(in[0][i]);
            out[0][i] = chorus_.GetLeft();
            out[1][i] = chorus_.GetRight();
        }
    }

  private:
    daisysp::Chorus chorus_;
};

/*
High-Complexity Effects
*/
struct Reverb : BaseEffect {
    void init(float sample_rate) override {
        reverb_.Init(sample_rate);
        reverb_.SetFeedback(0.9f);
        reverb_.SetLpFreq(18000.0f);
    }

    inline void process_audio(AudioHandle::InputBuffer& in,
                              AudioHandle::OutputBuffer& out,
                       size_t size,
                       float pot_val) override {
        (void)pot_val;
        for(size_t i = 0; i < size; i++) {
            float out_l = 0.0f;
            float out_r = 0.0f;
            reverb_.Process(in[0][i], in[1][i], &out_l, &out_r);
            out[0][i] = out_l;
            out[1][i] = out_r;
        }
    }

  private:
    daisysp::ReverbSc reverb_;
};

struct PitchShifter : BaseEffect {
    void init(float sample_rate) override {
        shifter_l_.Init(sample_rate);
        shifter_r_.Init(sample_rate);
        shifter_l_.SetTransposition(12.0f);
        shifter_r_.SetTransposition(12.0f);
    }

    inline void process_audio(AudioHandle::InputBuffer& in,
                              AudioHandle::OutputBuffer& out,
                       size_t size,
                       float pot_val) override {
        (void)pot_val;
        for(size_t i = 0; i < size; i++) {
            float in_l = in[0][i];
            float in_r = in[1][i];
            out[0][i] = shifter_l_.Process(in_l);
            out[1][i] = shifter_r_.Process(in_r);
        }
    }

  private:
    daisysp::PitchShifter shifter_l_;
    daisysp::PitchShifter shifter_r_;
};
