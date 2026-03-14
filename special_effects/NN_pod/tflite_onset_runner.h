#pragma once

#include <cstddef>

#include "daisy_seed.h"
#include "onset_weights.h"

namespace onset_tflite {

constexpr int kNoteHeadOut = ONSET_NOTE_VOCAB_SIZE + 1;

struct Output {
    float string_probs[ONSET_N_STRINGS];
    float note_probs[kNoteHeadOut];
    float pick_score;
    float sus_score;
};

bool init();
const char* init_error();
void log_init_snapshot(daisy::DaisySeed& seed);
size_t arena_used_bytes();
void forward(const float* input, size_t input_size, Output& out);

}  // namespace onset_tflite
