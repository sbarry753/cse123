// Auto-generated - DO NOT EDIT
// OnsetNet weights for Daisy Seed bare metal inference
// sr=48000 note_vocab_size=49 window_samples=384

#pragma once
#include <stdint.h>

#define ONSET_SR                 48000
#define ONSET_NOTE_VOCAB_SIZE    49
#define ONSET_WINDOW             384
#define ONSET_WIDTH              96
#define ONSET_N_STRINGS          6
#define ONSET_HAS_PICK_SUSTAIN   1

static const char* onset_pitch_names[] = {
    "E2",
    "F2",
    "F#2",
    "G2",
    "G#2",
    "A3",
    "A#3",
    "B3",
    "C3",
    "C#3",
    "D3",
    "D#3",
    "E3",
    "F3",
    "F#3",
    "G3",
    "G#3",
    "A4",
    "A#4",
    "B4",
    "C4",
    "C#4",
    "D4",
    "D#4",
    "E4",
    "F4",
    "F#4",
    "G4",
    "G#4",
    "A5",
    "A#5",
    "B5",
    "C5",
    "C#5",
    "D5",
    "D#5",
    "E5",
    "F5",
    "F#5",
    "G5",
    "G#5",
    "A6",
    "A#6",
    "B6",
    "C6",
    "C#6",
    "D6",
    "D#6",
    "E6",
};

// shape: [48, 1, 9]
static const float l0_conv_weight[432] = {
};

// shape: [48]
static const float l0_bn_weight[48] = {
 
};

// shape: [48]
static const float l0_bn_bias[48] = {
 
};

// shape: [48]
static const float l0_bn_running_mean[48] = {
   
};

// shape: [48]
static const float l0_bn_running_var[48] = {
   
};


// shape: [96, 48, 5]
static const float l1_conv_weight[23040] = {
   
};

// shape: [96]
static const float l1_bn_weight[96] = {
   
};

// shape: [96]
static const float l1_bn_bias[96] = {
   
};

// shape: [96]
static const float l1_bn_running_mean[96] = {
   
};

// shape: [96]
static const float l1_bn_running_var[96] = {
   
};


// shape: [96, 96, 5]
static const float l2_conv_weight[46080] = {
   
};

// shape: [96]
static const float l2_bn_weight[96] = {
   
};

// shape: [96]
static const float l2_bn_bias[96] = {
   
};

// shape: [96]
static const float l2_bn_running_mean[96] = {
   
};

// shape: [96]
static const float l2_bn_running_var[96] = {
   
};

// shape: [50, 96]
static const float note_head_weight[4800] = {
   
};

// shape: [50]
static const float note_head_bias[50] = {
   
};

// shape: [6, 96]
static const float string_head_weight[576] = {
   
};

// shape: [6]
static const float string_head_bias[6] = {
   
};

// shape: [1, 96]
static const float picked_head_weight[96] = {
   
};

// shape: [1]
static const float picked_head_bias[1] = {
    -0.05130583f
};

// shape: [1, 96]
static const float sustain_head_weight[96] = {
   
};

// shape: [1]
static const float sustain_head_bias[1] = {
    0.52748460f
};

