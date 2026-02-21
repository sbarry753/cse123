#pragma once

#include "daisysp.h"

using namespace daisysp;
using namespace daisy;

struct BaseEffect {
    BaseEffect() = default;

    virtual ~BaseEffect() = default;
    virtual void process_audio(AudioHandle::InputBuffer &in,
                                AudioHandle::OutputBuffer &out,
                                size_t size) = 0;
};

struct Passthrough : BaseEffect {
    void process_audio(AudioHandle::InputBuffer &in,
                       AudioHandle::OutputBuffer &out, 
                       size_t size) override {

        for (size_t i = 0; i < size; i++) {
            out[0][i] = in[0][i];
            out[1][i] = in[1][i];
	    }
    }
};