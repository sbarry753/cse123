#include "tflite_onset_runner.h"

#include "tflite_onset.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace onset_tflite {

constexpr size_t kTensorArenaSize = 128 * 1024;
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

const tflite::Model* g_tflm_model = nullptr;
tflite::MicroInterpreter* g_tflm_interpreter = nullptr;
TfLiteTensor* g_tflm_input = nullptr;
TfLiteTensor* g_tflm_string = nullptr;
TfLiteTensor* g_tflm_sus = nullptr;
TfLiteTensor* g_tflm_note = nullptr;
TfLiteTensor* g_tflm_pick = nullptr;
bool g_tflm_ready = false;
char g_tflm_init_error[128]  = "not started";
int g_tflm_dbg_input_type   = -1;
unsigned long g_tflm_dbg_input_bytes  = 0;
unsigned long g_tflm_dbg_input_ptr    = 0;
unsigned long g_tflm_dbg_input_dims_ptr = 0;

bool set_tflm_error(const char* msg) {
    strncpy(g_tflm_init_error, msg, sizeof(g_tflm_init_error) - 1);
    g_tflm_init_error[sizeof(g_tflm_init_error) - 1] = '\0';
    return false;
}

int tensor_flat_size(const TfLiteTensor* tensor) {
    if(tensor == nullptr || tensor->dims == nullptr)
        return 0;

    int size = 1;
    for(int i = 0; i < tensor->dims->size; ++i)
        size *= tensor->dims->data[i];
    return size;
}

int8_t quantize_float(float x, const TfLiteTensor* tensor) {
    int32_t q = (int32_t)lrintf(x / tensor->params.scale) + tensor->params.zero_point;
    if(q < -128) q = -128;
    if(q > 127) q = 127;
    return (int8_t)q;
}

float dequantize_int8(int8_t x, const TfLiteTensor* tensor) {
    return (float)((int32_t)x - tensor->params.zero_point) * tensor->params.scale;
}

void softmax_ip(float* a, int n) {
    float mx = a[0];
    for(int i = 1; i < n; ++i)
        if(a[i] > mx)
            mx = a[i];

    float s = 0.0f;
    for(int i = 0; i < n; ++i) {
        a[i] = expf(a[i] - mx);
        s += a[i];
    }

    const float inv = 1.0f / (s + 1e-12f);
    for(int i = 0; i < n; ++i)
        a[i] *= inv;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

bool init() {
    g_tflm_ready = false;
    g_tflm_model = nullptr;
    g_tflm_interpreter = nullptr;
    g_tflm_input = nullptr;
    g_tflm_string = nullptr;
    g_tflm_sus = nullptr;
    g_tflm_note = nullptr;
    g_tflm_pick = nullptr;
    g_tflm_dbg_input_type  = -1;
    g_tflm_dbg_input_bytes = 0;
    g_tflm_dbg_input_ptr   = 0;
    g_tflm_dbg_input_dims_ptr = 0;
    set_tflm_error("init_tflm() started");

    g_tflm_model = tflite::GetModel(onset_best_tflite);
    if(g_tflm_model == nullptr)
        return set_tflm_error("GetModel returned null");

    if(g_tflm_model->version() != TFLITE_SCHEMA_VERSION) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "schema mismatch: model=%lu runtime=%d",
                 (unsigned long)g_tflm_model->version(),
                 TFLITE_SCHEMA_VERSION);
        return false;
    }

    static tflite::MicroMutableOpResolver<9> resolver;
    static bool resolver_ready = false;
    if(!resolver_ready) {
        if(resolver.AddReshape() != kTfLiteOk) return set_tflm_error("resolver.AddReshape failed");
        if(resolver.AddPad() != kTfLiteOk) return set_tflm_error("resolver.AddPad failed");
        if(resolver.AddDepthwiseConv2D() != kTfLiteOk)
            return set_tflm_error("resolver.AddDepthwiseConv2D failed");
        if(resolver.AddMul() != kTfLiteOk) return set_tflm_error("resolver.AddMul failed");
        if(resolver.AddAdd() != kTfLiteOk) return set_tflm_error("resolver.AddAdd failed");
        if(resolver.AddTranspose() != kTfLiteOk)
            return set_tflm_error("resolver.AddTranspose failed");
        if(resolver.AddConv2D() != kTfLiteOk) return set_tflm_error("resolver.AddConv2D failed");
        if(resolver.AddSum() != kTfLiteOk) return set_tflm_error("resolver.AddSum failed");
        if(resolver.AddFullyConnected() != kTfLiteOk)
            return set_tflm_error("resolver.AddFullyConnected failed");
        resolver_ready = true;
    }

    static tflite::MicroInterpreter interpreter(
        g_tflm_model, resolver, g_tensor_arena, sizeof(g_tensor_arena));
    g_tflm_interpreter = &interpreter;

    if(g_tflm_interpreter->initialization_status() != kTfLiteOk)
        return set_tflm_error("MicroInterpreter initialization_status failed");

    if(g_tflm_interpreter->AllocateTensors() != kTfLiteOk)
        return set_tflm_error("AllocateTensors failed");

    if(g_tflm_interpreter->inputs_size() != 1) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "input count mismatch: got=%lu expected=1",
                 (unsigned long)g_tflm_interpreter->inputs_size());
        return false;
    }

    if(g_tflm_interpreter->outputs_size() != 4) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "output count mismatch: got=%lu expected=4",
                 (unsigned long)g_tflm_interpreter->outputs_size());
        return false;
    }

    g_tflm_input  = g_tflm_interpreter->input(0);
    g_tflm_string = g_tflm_interpreter->output(0);
    g_tflm_sus    = g_tflm_interpreter->output(1);
    g_tflm_note   = g_tflm_interpreter->output(2);
    g_tflm_pick   = g_tflm_interpreter->output(3);

    if(g_tflm_input != nullptr) {
        g_tflm_dbg_input_type = (int)g_tflm_input->type;
        g_tflm_dbg_input_bytes = (unsigned long)g_tflm_input->bytes;
        g_tflm_dbg_input_ptr = (unsigned long)g_tflm_input;
        g_tflm_dbg_input_dims_ptr = (unsigned long)g_tflm_input->dims;
    }

    if(g_tflm_input == nullptr || g_tflm_string == nullptr || g_tflm_sus == nullptr
       || g_tflm_note == nullptr || g_tflm_pick == nullptr)
        return set_tflm_error("one or more tensors are null");

    if(g_tflm_input->type != kTfLiteInt8) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "input type mismatch: got=%d expected=%d",
                 (int)g_tflm_input->type,
                 (int)kTfLiteInt8);
        return false;
    }

    if(tensor_flat_size(g_tflm_input) != ONSET_WINDOW) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "input size mismatch: got=%d expected=%d",
                 tensor_flat_size(g_tflm_input),
                 ONSET_WINDOW);
        return false;
    }

    if(g_tflm_string->type != kTfLiteInt8) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "string type mismatch: got=%d expected=%d",
                 (int)g_tflm_string->type,
                 (int)kTfLiteInt8);
        return false;
    }

    if(tensor_flat_size(g_tflm_string) != ONSET_N_STRINGS) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "string size mismatch: got=%d expected=%d",
                 tensor_flat_size(g_tflm_string),
                 ONSET_N_STRINGS);
        return false;
    }

    if(g_tflm_note->type != kTfLiteInt8) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "note type mismatch: got=%d expected=%d",
                 (int)g_tflm_note->type,
                 (int)kTfLiteInt8);
        return false;
    }

    if(tensor_flat_size(g_tflm_note) != kNoteHeadOut) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "note size mismatch: got=%d expected=%d",
                 tensor_flat_size(g_tflm_note),
                 kNoteHeadOut);
        return false;
    }

    if(g_tflm_pick->type != kTfLiteInt8) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "pick type mismatch: got=%d expected=%d",
                 (int)g_tflm_pick->type,
                 (int)kTfLiteInt8);
        return false;
    }

    if(tensor_flat_size(g_tflm_pick) != 1)
        return set_tflm_error("pick size mismatch: expected 1");

    if(g_tflm_sus->type != kTfLiteInt8) {
        snprintf(g_tflm_init_error,
                 sizeof(g_tflm_init_error),
                 "sustain type mismatch: got=%d expected=%d",
                 (int)g_tflm_sus->type,
                 (int)kTfLiteInt8);
        return false;
    }

    if(tensor_flat_size(g_tflm_sus) != 1)
        return set_tflm_error("sustain size mismatch: expected 1");

    g_tflm_ready = true;
    set_tflm_error("ok");
    return true;
}

const char* init_error() {
    return g_tflm_init_error;
}

void log_init_snapshot(daisy::DaisySeed& seed) {
    seed.PrintLine("TFLM dbg: input type=%d input bytes=%lu input ptr=%lu input dims ptr=%lu",
                   g_tflm_dbg_input_type,
                   g_tflm_dbg_input_bytes,
                   g_tflm_dbg_input_ptr,
                   g_tflm_dbg_input_dims_ptr);
}

size_t arena_used_bytes() {
    if(g_tflm_interpreter == nullptr)
        return 0;
    return g_tflm_interpreter->arena_used_bytes();
}

void forward(const float* input, size_t input_size, Output& out) {
    memset(&out, 0, sizeof(out));
    if(!g_tflm_ready || g_tflm_input == nullptr || g_tflm_interpreter == nullptr)
        return;
    if(input == nullptr || input_size != ONSET_WINDOW)
        return;

    int8_t* in = g_tflm_input->data.int8;
    for(size_t i = 0; i < input_size; ++i)
        in[i] = quantize_float(input[i], g_tflm_input);

    if(g_tflm_interpreter->Invoke() != kTfLiteOk)
        return;

    for(int i = 0; i < ONSET_N_STRINGS; ++i)
        out.string_probs[i] = dequantize_int8(g_tflm_string->data.int8[i], g_tflm_string);
    softmax_ip(out.string_probs, ONSET_N_STRINGS);

    for(int i = 0; i < kNoteHeadOut; ++i)
        out.note_probs[i] = dequantize_int8(g_tflm_note->data.int8[i], g_tflm_note);
    softmax_ip(out.note_probs, kNoteHeadOut);

    out.pick_score = sigmoid(dequantize_int8(g_tflm_pick->data.int8[0], g_tflm_pick));
    out.sus_score = sigmoid(dequantize_int8(g_tflm_sus->data.int8[0], g_tflm_sus));
}

}  // namespace onset_tflite
