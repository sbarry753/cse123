#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "daisy.h"
#include "daisy_pod.h"

#include "generated_tflm_model_data.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef TF_LITE_STATIC_MEMORY
#error "tflm_benchmark.cpp must be compiled with TF_LITE_STATIC_MEMORY to match the prebuilt TFLM library."
#endif

using namespace daisy;

constexpr float kPi = 3.14159265358979323846f;
constexpr int kWarmupIters = 5;
constexpr int kBenchIters = 20;

constexpr std::size_t kUnetArenaBytes = 3U * 1024U * 1024U;
constexpr std::size_t kTransientArenaBytes = 256U * 1024U;

constexpr int kTfSize = kFreqBins * kStftFrames;
constexpr int kFeatureSize = kFeatureChannels * kTfSize;
constexpr int kPaddedFrameSize = kFrameSize + kFrameSize;

struct BenchmarkCase {
    const char* name;
    const float* input;
    const float* golden;
};

struct DebugCase {
    const char* name;
    const float* input;
    const float* current_log_mag;
    uint32_t current_log_mag_checksum;
    const float* current_phase;
    uint32_t current_phase_checksum;
    const float* features;
    uint32_t features_checksum;
    const int8_t* unet_input_int8;
    uint32_t unet_input_int8_checksum;
    const int8_t* mask_int8;
    uint32_t mask_int8_checksum;
    const int8_t* residual_int8;
    uint32_t residual_int8_checksum;
    const int8_t* phase_delta_int8;
    uint32_t phase_delta_int8_checksum;
    const float* mask_dequant;
    uint32_t mask_dequant_checksum;
    const float* residual_dequant;
    uint32_t residual_dequant_checksum;
    const float* phase_delta_dequant;
    uint32_t phase_delta_dequant_checksum;
    const float* pre_transient;
    uint32_t pre_transient_checksum;
    const int8_t* transient_input_int8;
    uint32_t transient_input_int8_checksum;
    const int8_t* transient_output_int8;
    uint32_t transient_output_int8_checksum;
    const float* transient_output_dequant;
    uint32_t transient_output_dequant_checksum;
    const float* final_output;
    uint32_t final_output_checksum;
};

struct StageCycles {
    uint32_t preproc = 0;
    uint32_t unet = 0;
    uint32_t recon = 0;
    uint32_t transient = 0;
    uint32_t total = 0;
};

struct QuantParams {
    float scale = 1.0f;
    int zero_point = 0;
};

DaisyPod hw;

static bool g_initialized = false;

static uint16_t g_fft_bitrev[kFftSize];
static float g_fft_cos[9][kFftSize / 2];
static float g_fft_sin[9][kFftSize / 2];
static float g_istft_window_sumsquare[kPaddedFrameSize];

static float DSY_SDRAM_BSS g_log_mag_all[kContextFrames * kTfSize];
static float DSY_SDRAM_BSS g_current_log_mag[kTfSize];
static float DSY_SDRAM_BSS g_current_phase[kTfSize];
static float DSY_SDRAM_BSS g_features[kFeatureSize];
static float DSY_SDRAM_BSS g_mask[kTfSize];
static float DSY_SDRAM_BSS g_residual[kTfSize];
static float DSY_SDRAM_BSS g_phase_delta[kTfSize];
static float DSY_SDRAM_BSS g_spec_re[kTfSize];
static float DSY_SDRAM_BSS g_spec_im[kTfSize];
static float DSY_SDRAM_BSS g_fft_re[kFftSize];
static float DSY_SDRAM_BSS g_fft_im[kFftSize];
static float DSY_SDRAM_BSS g_istft_accum[kPaddedFrameSize];
static float DSY_SDRAM_BSS g_pre_transient[kFrameSize];
static float DSY_SDRAM_BSS g_output_audio[kFrameSize];
static float DSY_SDRAM_BSS g_transient_output_dequant[kFrameSize];
static int8_t DSY_SDRAM_BSS g_unet_input_q[kFeatureSize];
static int8_t DSY_SDRAM_BSS g_mask_q[kTfSize];
static int8_t DSY_SDRAM_BSS g_residual_q[kTfSize];
static int8_t DSY_SDRAM_BSS g_phase_delta_q[kTfSize];
static int8_t DSY_SDRAM_BSS g_transient_input_q[kFrameSize];
static int8_t DSY_SDRAM_BSS g_transient_output_q[kFrameSize];

alignas(16) static uint8_t DSY_SDRAM_BSS g_unet_tensor_arena[kUnetArenaBytes];
alignas(16) static uint8_t DSY_SDRAM_BSS g_transient_tensor_arena[kTransientArenaBytes];

alignas(tflite::RecordingMicroInterpreter) static uint8_t
    g_unet_interpreter_storage[sizeof(tflite::RecordingMicroInterpreter)];
alignas(tflite::RecordingMicroInterpreter) static uint8_t
    g_transient_interpreter_storage[sizeof(tflite::RecordingMicroInterpreter)];

static tflite::RecordingMicroInterpreter* g_unet_interpreter = nullptr;
static tflite::RecordingMicroInterpreter* g_transient_interpreter = nullptr;

static const tflite::Model* g_unet_model = nullptr;
static const tflite::Model* g_transient_model = nullptr;
static tflite::MicroMutableOpResolver<14> g_op_resolver;

static QuantParams g_unet_input_quant;
static QuantParams g_unet_output_quant[3];
static QuantParams g_transient_input_quant;
static QuantParams g_transient_output_quant;

static std::size_t g_unet_arena_used = 0;
static std::size_t g_transient_arena_used = 0;

constexpr std::size_t kDspScratchBytes = sizeof(g_log_mag_all) + sizeof(g_current_log_mag)
                                       + sizeof(g_current_phase) + sizeof(g_features)
                                       + sizeof(g_mask) + sizeof(g_residual)
                                       + sizeof(g_phase_delta) + sizeof(g_spec_re)
                                       + sizeof(g_spec_im) + sizeof(g_fft_re)
                                       + sizeof(g_fft_im) + sizeof(g_istft_accum)
                                       + sizeof(g_pre_transient) + sizeof(g_output_audio)
                                       + sizeof(g_transient_output_dequant)
                                       + sizeof(g_unet_input_q) + sizeof(g_mask_q)
                                       + sizeof(g_residual_q) + sizeof(g_phase_delta_q)
                                       + sizeof(g_transient_input_q)
                                       + sizeof(g_transient_output_q)
                                       + sizeof(g_fft_bitrev) + sizeof(g_fft_cos)
                                       + sizeof(g_fft_sin)
                                       + sizeof(g_istft_window_sumsquare);
constexpr std::size_t kReservedArenaBytes = kUnetArenaBytes + kTransientArenaBytes;
constexpr std::size_t kModelBytes = kUnetModelBytes + kTransientModelBytes;
constexpr std::size_t kStaticReservedBytes = kDspScratchBytes + kReservedArenaBytes + kModelBytes;

inline int IdxTF(int freq, int time) {
    return freq * kStftFrames + time;
}

inline int IdxFeat(int channel, int freq, int time) {
    return (channel * kFreqBins + freq) * kStftFrames + time;
}

inline float Clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

inline int ReflectIndex(int idx, int size) {
    if(size <= 1) return 0;
    while(idx < 0 || idx >= size) {
        if(idx < 0) {
            idx = -idx;
        }
        if(idx >= size) {
            idx = 2 * size - 2 - idx;
        }
    }
    return idx;
}

inline int8_t QuantizeInt8(float value, const QuantParams& quant) {
    const float scaled = value / quant.scale + static_cast<float>(quant.zero_point);
    const int rounded = static_cast<int>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
    const int clamped = rounded < -128 ? -128 : (rounded > 127 ? 127 : rounded);
    return static_cast<int8_t>(clamped);
}

inline float DequantizeInt8(int8_t value, const QuantParams& quant) {
    return (static_cast<int>(value) - quant.zero_point) * quant.scale;
}

uint32_t Fnv1aBytes(const void* data, std::size_t byte_count) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint32_t hash = 2166136261u;
    for(std::size_t i = 0; i < byte_count; ++i) {
        hash ^= bytes[i];
        hash *= 16777619u;
    }
    return hash;
}

uint32_t ChecksumFloatArray(const float* data, int count) {
    return Fnv1aBytes(data, static_cast<std::size_t>(count) * sizeof(float));
}

uint32_t ChecksumInt8Array(const int8_t* data, int count) {
    return Fnv1aBytes(data, static_cast<std::size_t>(count) * sizeof(int8_t));
}

uint32_t ToScaledAbs(float value, float scale) {
    const float scaled = fabsf(value) * scale + 0.5f;
    return static_cast<uint32_t>(scaled);
}

void PrintQuantParamLine(const char* name,
                         float actual_scale,
                         int actual_zero_point,
                         float expected_scale,
                         int expected_zero_point) {
    const uint32_t actual_scaled = ToScaledAbs(actual_scale, 1000000000.0f);
    const uint32_t expected_scaled = ToScaledAbs(expected_scale, 1000000000.0f);
    hw.seed.PrintLine(
        "%s scale actual=%s%lu.%09lu expected=%s%lu.%09lu zp actual=%d expected=%d",
        name,
        actual_scale < 0.0f ? "-" : "",
        static_cast<unsigned long>(actual_scaled / 1000000000u),
        static_cast<unsigned long>(actual_scaled % 1000000000u),
        expected_scale < 0.0f ? "-" : "",
        static_cast<unsigned long>(expected_scaled / 1000000000u),
        static_cast<unsigned long>(expected_scaled % 1000000000u),
        actual_zero_point,
        expected_zero_point);
}

void PrintSignedScaledValue(const char* label, float value, float scale) {
    const uint32_t scaled = ToScaledAbs(value, scale);
    const char* sign = value < 0.0f ? "-" : "";
    hw.seed.PrintLine("%s%s%lu.%06lu",
                      label,
                      sign,
                      static_cast<unsigned long>(scaled / 1000000u),
                      static_cast<unsigned long>(scaled % 1000000u));
}

void PrintFloatPreview(const char* label,
                       const float* actual,
                       const float* expected,
                       int count,
                       int start_index) {
    const int preview = count < 4 ? count : 4;
    const int begin = start_index < 0 ? 0 : (start_index >= count ? count - 1 : start_index);
    for(int offset = 0; offset < preview && (begin + offset) < count; ++offset) {
        const int i = begin + offset;
        const uint32_t actual_scaled = ToScaledAbs(actual[i], 1000000.0f);
        const uint32_t expected_scaled = ToScaledAbs(expected[i], 1000000.0f);
        hw.seed.PrintLine("%s[%d]: actual=%s%lu.%06lu expected=%s%lu.%06lu",
                          label,
                          i,
                          actual[i] < 0.0f ? "-" : "",
                          static_cast<unsigned long>(actual_scaled / 1000000u),
                          static_cast<unsigned long>(actual_scaled % 1000000u),
                          expected[i] < 0.0f ? "-" : "",
                          static_cast<unsigned long>(expected_scaled / 1000000u),
                          static_cast<unsigned long>(expected_scaled % 1000000u));
    }
}

void PrintInt8Preview(const char* label,
                      const int8_t* actual,
                      const int8_t* expected,
                      int count,
                      int start_index) {
    const int preview = count < 8 ? count : 8;
    const int begin = start_index < 0 ? 0 : (start_index >= count ? count - 1 : start_index);
    for(int offset = 0; offset < preview && (begin + offset) < count; ++offset) {
        const int i = begin + offset;
        hw.seed.PrintLine("%s[%d]: actual=%d expected=%d",
                          label,
                          i,
                          static_cast<int>(actual[i]),
                          static_cast<int>(expected[i]));
    }
}

float PhaseWrappedAbsDiff(float actual, float expected) {
    const float diff = remainderf(actual - expected, 2.0f * kPi);
    return fabsf(diff);
}

bool CompareFloatStage(const char* stage_name,
                       const float* actual,
                       const float* expected,
                       int count,
                       float tolerance,
                       uint32_t expected_checksum,
                       bool wrap_phase = false) {
    const uint32_t actual_checksum = ChecksumFloatArray(actual, count);
    float max_abs = 0.0f;
    int first_bad = -1;
    for(int i = 0; i < count; ++i) {
        const float diff = wrap_phase ? PhaseWrappedAbsDiff(actual[i], expected[i])
                                      : fabsf(actual[i] - expected[i]);
        if(diff > max_abs) {
            max_abs = diff;
        }
        if(first_bad < 0 && diff > tolerance) {
            first_bad = i;
        }
    }

    const uint32_t max_abs_scaled = ToScaledAbs(max_abs, 1000000.0f);
    hw.seed.PrintLine("stage %s: checksum=0x%08lx expected=0x%08lx max_abs=%lu.%06lu first_bad=%d",
                      stage_name,
                      static_cast<unsigned long>(actual_checksum),
                      static_cast<unsigned long>(expected_checksum),
                      static_cast<unsigned long>(max_abs_scaled / 1000000u),
                      static_cast<unsigned long>(max_abs_scaled % 1000000u),
                      first_bad);
    if(first_bad >= 0) {
        hw.seed.PrintLine("FIRST FAILING STAGE: %s", stage_name);
        PrintFloatPreview(stage_name, actual, expected, count, first_bad);
        return false;
    }
    return true;
}

bool CompareInt8Stage(const char* stage_name,
                      const int8_t* actual,
                      const int8_t* expected,
                      int count,
                      uint32_t expected_checksum) {
    const uint32_t actual_checksum = ChecksumInt8Array(actual, count);
    int first_bad = -1;
    for(int i = 0; i < count; ++i) {
        if(actual[i] != expected[i]) {
            first_bad = i;
            break;
        }
    }

    hw.seed.PrintLine("stage %s: checksum=0x%08lx expected=0x%08lx first_bad=%d",
                      stage_name,
                      static_cast<unsigned long>(actual_checksum),
                      static_cast<unsigned long>(expected_checksum),
                      first_bad);
    if(first_bad >= 0) {
        hw.seed.PrintLine("FIRST FAILING STAGE: %s", stage_name);
        PrintInt8Preview(stage_name, actual, expected, count, first_bad);
        return false;
    }
    return true;
}

void ZeroBuffer(float* data, int count) {
    std::memset(data, 0, static_cast<std::size_t>(count) * sizeof(float));
}

void EnableCycleCounter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void PrecomputeFftTables() {
    for(int i = 0; i < kFftSize; ++i) {
        unsigned value = static_cast<unsigned>(i);
        unsigned rev = 0;

        for(int bit = 0; bit < 9; ++bit) {
            rev = (rev << 1U) | (value & 1U);
            value >>= 1U;
        }
        g_fft_bitrev[i] = static_cast<uint16_t>(rev);
    }

    for(int stage = 0; stage < 9; ++stage) {
        const int len = 1 << (stage + 1);
        const int half = len >> 1;
        const float base = -2.0f * kPi / static_cast<float>(len);
        for(int j = 0; j < half; ++j) {
            const float angle = base * static_cast<float>(j);
            g_fft_cos[stage][j] = cosf(angle);
            g_fft_sin[stage][j] = sinf(angle);
        }
    }
}

void PrecomputeIstftWindowNorm() {
    ZeroBuffer(g_istft_window_sumsquare, kPaddedFrameSize);

    for(int frame = 0; frame < kStftFrames; ++frame) {
        const int start = frame * kHopSize;
        for(int n = 0; n < kFrameSize; ++n) {
            const float w = g_window[n];
            g_istft_window_sumsquare[start + n] += w * w;
        }
    }
}

void FftInPlace(float* re, float* im, bool inverse) {
    for(int i = 0; i < kFftSize; ++i) {
        const int j = g_fft_bitrev[i];
        if(j > i) {
            const float tmp_re = re[i];
            const float tmp_im = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = tmp_re;
            im[j] = tmp_im;
        }
    }

    for(int stage = 0; stage < 9; ++stage) {
        const int len = 1 << (stage + 1);
        const int half = len >> 1;
        for(int start = 0; start < kFftSize; start += len) {
            for(int j = 0; j < half; ++j) {
                const float wre = g_fft_cos[stage][j];
                const float wim = inverse ? -g_fft_sin[stage][j] : g_fft_sin[stage][j];
                const int u = start + j;
                const int v = u + half;
                const float tre = re[v] * wre - im[v] * wim;
                const float tim = re[v] * wim + im[v] * wre;
                const float ure = re[u];
                const float uim = im[u];
                re[u] = ure + tre;
                im[u] = uim + tim;
                re[v] = ure - tre;
                im[v] = uim - tim;
            }
        }
    }

    if(inverse) {
        const float scale = 1.0f / static_cast<float>(kFftSize);
        for(int i = 0; i < kFftSize; ++i) {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
}

void BuildSpectralFeatures(const float* audio_ctx) {
    const int newest_offset = (kContextFrames - 1) * kFrameSize;
    for(int ctx = 0; ctx < kContextFrames; ++ctx) {
        const float* audio = audio_ctx + ctx * kFrameSize;

        for(int frame = 0; frame < kStftFrames; ++frame) {
            const int start = frame * kHopSize - (kFrameSize / 2);

            for(int n = 0; n < kFrameSize; ++n) {
                const int sample_idx = ReflectIndex(start + n, kFrameSize);
                g_fft_re[n] = audio[sample_idx] * g_window[n];
                g_fft_im[n] = 0.0f;
            }

            FftInPlace(g_fft_re, g_fft_im, false);

            for(int freq = 0; freq < kFreqBins; ++freq) {
                const int tf_idx = ctx * kTfSize + IdxTF(freq, frame);
                const float re = g_fft_re[freq];
                const float im = g_fft_im[freq];
                const float mag = sqrtf(re * re + im * im);
                const float logmag = logf(mag < 1.0e-5f ? 1.0e-5f : mag);
                g_log_mag_all[tf_idx] = logmag;

                if(ctx == kContextFrames - 1) {
                    const int idx = IdxTF(freq, frame);
                    g_current_log_mag[idx] = logmag;
                    g_current_phase[idx] = atan2f(im, re);
                }
            }
        }
    }

    float env_accum = 0.0f;
    float signed_mean_accum = 0.0f;
    const float* current_audio = audio_ctx + newest_offset;
    for(int i = 0; i < kFrameSize; ++i) {
        env_accum += current_audio[i] * current_audio[i];
        signed_mean_accum += current_audio[i];
    }

    const float env = sqrtf(env_accum / static_cast<float>(kFrameSize) + 1.0e-8f);
    const float signed_mean = signed_mean_accum / static_cast<float>(kFrameSize);

    for(int freq = 0; freq < kFreqBins; ++freq) {
        for(int time = 0; time < kStftFrames; ++time) {
            const int cur_idx = IdxTF(freq, time);

            for(int ctx = 0; ctx < kContextFrames; ++ctx) {
                g_features[IdxFeat(ctx, freq, time)] =
                    g_log_mag_all[ctx * kTfSize + cur_idx];
            }

            for(int ctx = 0; ctx < kContextFrames - 1; ++ctx) {
                g_features[IdxFeat(kContextFrames + ctx, freq, time)] =
                    g_log_mag_all[ctx * kTfSize + cur_idx] - g_current_log_mag[cur_idx];
            }

            g_features[IdxFeat(7, freq, time)] = env;
            g_features[IdxFeat(8, freq, time)] = signed_mean;
        }
    }
}

void ReconstructWaveform(float* audio_out) {
    for(int i = 0; i < kTfSize; ++i) {
        const float out_log_mag = g_current_log_mag[i] * g_mask[i] + g_residual[i];
        const float out_mag = expf(out_log_mag);
        const float out_phase = g_current_phase[i] + g_phase_delta[i];
        g_spec_re[i] = out_mag * cosf(out_phase);
        g_spec_im[i] = out_mag * sinf(out_phase);
    }

    ZeroBuffer(g_istft_accum, kPaddedFrameSize);

    for(int frame = 0; frame < kStftFrames; ++frame) {
        ZeroBuffer(g_fft_re, kFftSize);
        ZeroBuffer(g_fft_im, kFftSize);

        for(int freq = 0; freq < kFreqBins; ++freq) {
            const int idx = IdxTF(freq, frame);
            if(freq == 0 || freq == (kFftSize / 2)) {
                g_fft_re[freq] = g_spec_re[idx];
                g_fft_im[freq] = 0.0f;
            } else {
                g_fft_re[freq] = g_spec_re[idx];
                g_fft_im[freq] = g_spec_im[idx];
                g_fft_re[kFftSize - freq] = g_spec_re[idx];
                g_fft_im[kFftSize - freq] = -g_spec_im[idx];
            }
        }

        FftInPlace(g_fft_re, g_fft_im, true);

        const int start = frame * kHopSize;
        for(int n = 0; n < kFrameSize; ++n) {
            g_istft_accum[start + n] += g_fft_re[n] * g_window[n];
        }
    }

    for(int n = 0; n < kFrameSize; ++n) {
        const int padded_idx = n + (kFrameSize / 2);
        const float denom = g_istft_window_sumsquare[padded_idx];
        audio_out[n] = denom > 1.0e-11f ? g_istft_accum[padded_idx] / denom : 0.0f;
    }
}

TfLiteStatus RegisterRequiredOps() {
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddAbs());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddAdd());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddConcatenation());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddConv2D());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddExpandDims());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddLogistic());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddMean());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddMul());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddReshape());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddResizeBilinear());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddRsqrt());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddSquaredDifference());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddSub());
    TF_LITE_ENSURE_STATUS(g_op_resolver.AddTanh());
    return kTfLiteOk;
}

bool PopulateQuantParams() {
    TfLiteTensor* unet_input = g_unet_interpreter->input(0);
    if(unet_input == nullptr) return false;
    g_unet_input_quant.scale = unet_input->params.scale;
    g_unet_input_quant.zero_point = unet_input->params.zero_point;
    for(int i = 0; i < 3; ++i) {
        TfLiteTensor* tensor = g_unet_interpreter->output(i);
        if(tensor == nullptr) return false;
        g_unet_output_quant[i].scale = tensor->params.scale;
        g_unet_output_quant[i].zero_point = tensor->params.zero_point;
    }

    TfLiteTensor* transient_input = g_transient_interpreter->input(0);
    TfLiteTensor* transient_output = g_transient_interpreter->output(0);
    if(transient_input == nullptr || transient_output == nullptr) return false;
    g_transient_input_quant.scale = transient_input->params.scale;
    g_transient_input_quant.zero_point = transient_input->params.zero_point;
    g_transient_output_quant.scale = transient_output->params.scale;
    g_transient_output_quant.zero_point = transient_output->params.zero_point;
    return true;
}

bool InitTflm() {
    if(g_initialized) return true;

    g_unet_model = tflite::GetModel(g_unet_int8_model_data);
    g_transient_model = tflite::GetModel(g_transient_int8_model_data);
    if(g_unet_model == nullptr || g_transient_model == nullptr) {
        hw.seed.PrintLine("TFLM init failed: model parse");
        return false;
    }
    if(g_unet_model->version() != TFLITE_SCHEMA_VERSION
       || g_transient_model->version() != TFLITE_SCHEMA_VERSION) {
        hw.seed.PrintLine("TFLM init failed: schema version");
        return false;
    }

    if(RegisterRequiredOps() != kTfLiteOk) {
        hw.seed.PrintLine("TFLM init failed: resolver");
        return false;
    }

    g_unet_interpreter = new (g_unet_interpreter_storage) tflite::RecordingMicroInterpreter(
        g_unet_model,
        g_op_resolver,
        g_unet_tensor_arena,
        sizeof(g_unet_tensor_arena));
    g_transient_interpreter
        = new (g_transient_interpreter_storage) tflite::RecordingMicroInterpreter(
            g_transient_model,
            g_op_resolver,
            g_transient_tensor_arena,
            sizeof(g_transient_tensor_arena));

    if(g_unet_interpreter->AllocateTensors() != kTfLiteOk) {
        hw.seed.PrintLine("TFLM init failed: unet arena");
        return false;
    }
    if(g_transient_interpreter->AllocateTensors() != kTfLiteOk) {
        hw.seed.PrintLine("TFLM init failed: transient arena");
        return false;
    }

    g_unet_arena_used = g_unet_interpreter->arena_used_bytes();
    g_transient_arena_used = g_transient_interpreter->arena_used_bytes();

    if(!PopulateQuantParams()) {
        hw.seed.PrintLine("TFLM init failed: quant params");
        return false;
    }

    PrecomputeFftTables();
    PrecomputeIstftWindowNorm();
    g_initialized = true;
    return true;
}

bool RunUnetTflm() {
    int8_t* input = g_unet_interpreter->typed_input_tensor<int8_t>(0);
    for(int freq = 0; freq < kFreqBins; ++freq) {
        for(int time = 0; time < kStftFrames; ++time) {
            for(int ch = 0; ch < kFeatureChannels; ++ch) {
                const int src_idx = IdxFeat(ch, freq, time);
                const int dst_idx = ((freq * kStftFrames + time) * kFeatureChannels) + ch;
                const int8_t quantized = QuantizeInt8(g_features[src_idx], g_unet_input_quant);
                input[dst_idx] = quantized;
                g_unet_input_q[dst_idx] = quantized;
            }
        }
    }

    if(g_unet_interpreter->Invoke() != kTfLiteOk) {
        return false;
    }

    const int8_t* mask_q = g_unet_interpreter->typed_output_tensor<int8_t>(0);
    const int8_t* residual_q = g_unet_interpreter->typed_output_tensor<int8_t>(1);
    const int8_t* phase_q = g_unet_interpreter->typed_output_tensor<int8_t>(2);
    for(int i = 0; i < kTfSize; ++i) {
        g_mask_q[i] = mask_q[i];
        g_residual_q[i] = residual_q[i];
        g_phase_delta_q[i] = phase_q[i];
        g_mask[i] = DequantizeInt8(mask_q[i], g_unet_output_quant[0]);
        g_residual[i] = DequantizeInt8(residual_q[i], g_unet_output_quant[1]);
        g_phase_delta[i] = DequantizeInt8(phase_q[i], g_unet_output_quant[2]);
    }
    return true;
}

bool RunTransientTflm(const float* current_audio, float* audio_out) {
    int8_t* input = g_transient_interpreter->typed_input_tensor<int8_t>(0);
    for(int i = 0; i < kFrameSize; ++i) {
        const int8_t quantized = QuantizeInt8(audio_out[i], g_transient_input_quant);
        input[i] = quantized;
        g_transient_input_q[i] = quantized;
    }

    if(g_transient_interpreter->Invoke() != kTfLiteOk) {
        return false;
    }

    const int8_t* output_q = g_transient_interpreter->typed_output_tensor<int8_t>(0);
    for(int i = 0; i < kFrameSize; ++i) {
        g_transient_output_q[i] = output_q[i];
        const float shaped = DequantizeInt8(output_q[i], g_transient_output_quant);
        g_transient_output_dequant[i] = shaped;
        audio_out[i] = Clamp(0.985f * shaped + 0.015f * current_audio[i], -1.0f, 1.0f);
    }
    return true;
}

bool TflmInfer(const float audio_ctx[kContextFrames][kFrameSize],
               float audio_out[kFrameSize],
               StageCycles* cycles) {
    if(!InitTflm()) return false;

    const float* current_audio = audio_ctx[kContextFrames - 1];
    const uint32_t total_start = DWT->CYCCNT;

    const uint32_t preproc_start = DWT->CYCCNT;
    BuildSpectralFeatures(&audio_ctx[0][0]);
    const uint32_t preproc_end = DWT->CYCCNT;

    const uint32_t unet_start = DWT->CYCCNT;
    if(!RunUnetTflm()) return false;
    const uint32_t unet_end = DWT->CYCCNT;

    const uint32_t recon_start = DWT->CYCCNT;
    ReconstructWaveform(g_pre_transient);
    const uint32_t recon_end = DWT->CYCCNT;

    std::memcpy(audio_out, g_pre_transient, sizeof(g_pre_transient));
    const uint32_t transient_start = DWT->CYCCNT;
    if(!RunTransientTflm(current_audio, audio_out)) return false;
    const uint32_t transient_end = DWT->CYCCNT;

    if(cycles != nullptr) {
        cycles->preproc = preproc_end - preproc_start;
        cycles->unet = unet_end - unet_start;
        cycles->recon = recon_end - recon_start;
        cycles->transient = transient_end - transient_start;
        cycles->total = transient_end - total_start;
    }
    return true;
}

void ComputeError(const float* pred, const float* golden, float& max_abs, float& rmse) {
    max_abs = 0.0f;
    double accum = 0.0;
    for(int i = 0; i < kFrameSize; ++i) {
        const float diff = pred[i] - golden[i];
        const float ad = fabsf(diff);
        if(ad > max_abs) max_abs = ad;
        accum += static_cast<double>(diff) * static_cast<double>(diff);
    }
    rmse = sqrtf(static_cast<float>(accum / static_cast<double>(kFrameSize)));
}

uint32_t CyclesToUsX1000(uint32_t cycles, uint32_t cpu_hz) {
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(cycles) * 1000000000ULL) / static_cast<uint64_t>(cpu_hz));
}

uint32_t CyclesToHopPctX100(uint32_t cycles, uint32_t cpu_hz) {
    const uint64_t hop_budget_cycles
        = (static_cast<uint64_t>(cpu_hz) * static_cast<uint64_t>(kHopSize))
          / static_cast<uint64_t>(kSampleRate);
    if(hop_budget_cycles == 0) return 0;
    return static_cast<uint32_t>((static_cast<uint64_t>(cycles) * 10000ULL) / hop_budget_cycles);
}

void PrintMemorySummary() {
    hw.seed.PrintLine("UNet model bytes: %lu", static_cast<unsigned long>(kUnetModelBytes));
    hw.seed.PrintLine("Transient model bytes: %lu", static_cast<unsigned long>(kTransientModelBytes));
    hw.seed.PrintLine("DSP scratch bytes: %lu", static_cast<unsigned long>(kDspScratchBytes));
    hw.seed.PrintLine("Reserved arena bytes: %lu", static_cast<unsigned long>(kReservedArenaBytes));
    hw.seed.PrintLine("Static reserved total: %lu", static_cast<unsigned long>(kStaticReservedBytes));
    hw.seed.PrintLine("UNet arena used/reserved: %lu / %lu",
                      static_cast<unsigned long>(g_unet_arena_used),
                      static_cast<unsigned long>(kUnetArenaBytes));
    hw.seed.PrintLine("Transient arena used/reserved: %lu / %lu",
                      static_cast<unsigned long>(g_transient_arena_used),
                      static_cast<unsigned long>(kTransientArenaBytes));
}

void RunBenchmarkCase(const BenchmarkCase& bench_case, uint32_t cpu_hz) {
    static float output[kFrameSize];

    const auto* input = reinterpret_cast<const float (*)[kFrameSize]>(bench_case.input);
    hw.seed.PrintLine("----------------------------------------");
    hw.seed.PrintLine("Running case: %s", bench_case.name);
    hw.seed.PrintLine("Warmup/bench: %d / %d", kWarmupIters, kBenchIters);

    for(int i = 0; i < kWarmupIters; ++i) {
        StageCycles warmup_cycles;
        if(!TflmInfer(input, output, &warmup_cycles)) {
            hw.seed.PrintLine("Inference failed in warmup");
            return;
        }
    }

    uint32_t min_total = 0xffffffffu;
    uint32_t max_total = 0u;
    uint64_t sum_total = 0;
    uint64_t sum_preproc = 0;
    uint64_t sum_unet = 0;
    uint64_t sum_recon = 0;
    uint64_t sum_transient = 0;

    for(int i = 0; i < kBenchIters; ++i) {
        StageCycles stage;
        if(!TflmInfer(input, output, &stage)) {
            hw.seed.PrintLine("Inference failed in benchmark");
            return;
        }

        if(stage.total < min_total) min_total = stage.total;
        if(stage.total > max_total) max_total = stage.total;
        sum_total += stage.total;
        sum_preproc += stage.preproc;
        sum_unet += stage.unet;
        sum_recon += stage.recon;
        sum_transient += stage.transient;
    }

    const uint32_t avg_total
        = static_cast<uint32_t>((sum_total / static_cast<uint64_t>(kBenchIters)) + 0.5);
    const uint32_t avg_preproc
        = static_cast<uint32_t>((sum_preproc / static_cast<uint64_t>(kBenchIters)) + 0.5);
    const uint32_t avg_unet
        = static_cast<uint32_t>((sum_unet / static_cast<uint64_t>(kBenchIters)) + 0.5);
    const uint32_t avg_recon
        = static_cast<uint32_t>((sum_recon / static_cast<uint64_t>(kBenchIters)) + 0.5);
    const uint32_t avg_transient
        = static_cast<uint32_t>((sum_transient / static_cast<uint64_t>(kBenchIters)) + 0.5);

    float max_abs = 0.0f;
    float rmse = 0.0f;
    ComputeError(output, bench_case.golden, max_abs, rmse);

    const uint32_t min_us_x1000 = CyclesToUsX1000(min_total, cpu_hz);
    const uint32_t avg_us_x1000 = CyclesToUsX1000(avg_total, cpu_hz);
    const uint32_t max_us_x1000 = CyclesToUsX1000(max_total, cpu_hz);
    const uint32_t hop_pct_x100 = CyclesToHopPctX100(avg_total, cpu_hz);
    const uint32_t preproc_us_x1000 = CyclesToUsX1000(avg_preproc, cpu_hz);
    const uint32_t unet_us_x1000 = CyclesToUsX1000(avg_unet, cpu_hz);
    const uint32_t recon_us_x1000 = CyclesToUsX1000(avg_recon, cpu_hz);
    const uint32_t transient_us_x1000 = CyclesToUsX1000(avg_transient, cpu_hz);

    const uint32_t max_abs_x1000000
        = static_cast<uint32_t>(max_abs * 1000000.0f + 0.5f);
    const uint32_t rmse_x1000000 = static_cast<uint32_t>(rmse * 1000000.0f + 0.5f);

    hw.seed.PrintLine("Done case: %s", bench_case.name);
    hw.seed.PrintLine("min/avg/max cyc: %lu / %lu / %lu",
                      static_cast<unsigned long>(min_total),
                      static_cast<unsigned long>(avg_total),
                      static_cast<unsigned long>(max_total));
    hw.seed.PrintLine("min/avg/max us: %lu.%03lu / %lu.%03lu / %lu.%03lu",
                      static_cast<unsigned long>(min_us_x1000 / 1000U),
                      static_cast<unsigned long>(min_us_x1000 % 1000U),
                      static_cast<unsigned long>(avg_us_x1000 / 1000U),
                      static_cast<unsigned long>(avg_us_x1000 % 1000U),
                      static_cast<unsigned long>(max_us_x1000 / 1000U),
                      static_cast<unsigned long>(max_us_x1000 % 1000U));
    hw.seed.PrintLine("avg stage us: pre=%lu.%03lu unet=%lu.%03lu recon=%lu.%03lu transient=%lu.%03lu",
                      static_cast<unsigned long>(preproc_us_x1000 / 1000U),
                      static_cast<unsigned long>(preproc_us_x1000 % 1000U),
                      static_cast<unsigned long>(unet_us_x1000 / 1000U),
                      static_cast<unsigned long>(unet_us_x1000 % 1000U),
                      static_cast<unsigned long>(recon_us_x1000 / 1000U),
                      static_cast<unsigned long>(recon_us_x1000 % 1000U),
                      static_cast<unsigned long>(transient_us_x1000 / 1000U),
                      static_cast<unsigned long>(transient_us_x1000 % 1000U));
    hw.seed.PrintLine("avg hop budget: %lu.%02lu%%",
                      static_cast<unsigned long>(hop_pct_x100 / 100U),
                      static_cast<unsigned long>(hop_pct_x100 % 100U));
    hw.seed.PrintLine("max abs / rmse: %lu.%06lu / %lu.%06lu",
                      static_cast<unsigned long>(max_abs_x1000000 / 1000000U),
                      static_cast<unsigned long>(max_abs_x1000000 % 1000000U),
                      static_cast<unsigned long>(rmse_x1000000 / 1000000U),
                      static_cast<unsigned long>(rmse_x1000000 % 1000000U));
}

void TflmRunBenchmarks() {
    static const BenchmarkCase kCases[] = {
        {"zero", case_zero_input, case_zero_golden},
        {"impulse", case_impulse_input, case_impulse_golden},
        {"random", case_random_input, case_random_golden},
    };

    const uint32_t cpu_hz = System::GetSysClkFreq();

    hw.seed.PrintLine("");
    hw.seed.PrintLine("Daisy TFLM int8 benchmark");
    hw.seed.PrintLine("CPU Hz: %lu", static_cast<unsigned long>(cpu_hz));
    hw.seed.PrintLine("Frame/Hop: %d / %d", kFrameSize, kHopSize);
    hw.seed.PrintLine("STFT bins/frames: %d / %d", kFreqBins, kStftFrames);
    hw.seed.PrintLine("Feature channels: %d", kFeatureChannels);
    hw.seed.PrintLine("Activation note: quick-gelu approx in exported TFLite mirror");
    PrintMemorySummary();

    for(const auto& bench_case : kCases) {
        RunBenchmarkCase(bench_case, cpu_hz);
    }

    hw.seed.PrintLine("----------------------------------------");
}

int main(void) {
    hw.Init(true);
    hw.seed.sdram_handle.Init();
    hw.seed.StartLog(true);
    EnableCycleCounter();

    if(!InitTflm()) {
        hw.seed.PrintLine("TFLM benchmark init failed");
        while(true) {
            System::Delay(500);
        }
    }

    TflmRunBenchmarks();

    while(true) {
        hw.led1.Set(0.0f, 1.0f, 0.0f);
        hw.led2.Set(0.0f, 1.0f, 0.0f);
        hw.UpdateLeds();
        System::Delay(300);
        hw.led1.Set(0.0f, 0.0f, 0.0f);
        hw.led2.Set(0.0f, 0.0f, 0.0f);
        hw.UpdateLeds();
        System::Delay(300);
    }
}
