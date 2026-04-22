#include <cmath>
#include <cstdint>
#include <cstring>

#include "daisy.h"
#include "daisy_pod.h"

#include "generated_model_data.h"

using namespace daisy;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kSqrtHalf = 0.70710678118654752440f;
constexpr float kInstanceNormEps = 1.0e-5f;
constexpr int kWarmupIters = 5;
constexpr int kBenchIters = 20;

#if defined(BASELINE_USE_SDRAM_WEIGHTS)
constexpr bool kUseSdramWeights = true;
#else
constexpr bool kUseSdramWeights = false;
#endif

constexpr int kStftFrames = (kFrameSize + kFftSize) / kHopSize - (kFftSize / kHopSize) + 1;
constexpr int kInputSize = kContextFrames * kFrameSize;
constexpr int kTfSize = kFreqBins * kStftFrames;
constexpr int kFeatureSize = kFeatureChannels * kTfSize;
constexpr int kPaddedFrameSize = kFrameSize + kFrameSize;
constexpr int kEnc1Size = 32 * 257 * 5;
constexpr int kEnc2Size = 64 * 129 * 3;
constexpr int kEnc3Size = 128 * 65 * 2;
constexpr int kUpCatSize = 64 * 257 * 5;
constexpr int kTmp2dSize = kEnc1Size;
constexpr int kTmp1d32Size = 32 * kFrameSize;
constexpr int kTmp1d8Size = 8 * kFrameSize;

struct Conv2DLayer {
    int out_ch;
    int in_ch;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    const float* weight;
    const float* bias;
};

struct Norm2DLayer {
    int channels;
    const float* weight;
    const float* bias;
};

struct ConvBlock2D {
    Conv2DLayer conv0;
    Norm2DLayer norm0;
    Conv2DLayer conv1;
    Norm2DLayer norm1;
};

struct Conv1DLayer {
    int out_ch;
    int in_ch;
    int kernel;
    int stride;
    int pad;
    const float* weight;
    const float* bias;
};

struct BenchmarkCase {
    const char* name;
    const float* input;
    const float* golden;
};

#define CONV2D_DESC(name, out_ch_, in_ch_, kh_, kw_, sh_, sw_, ph_, pw_, w_, b_) \
    static Conv2DLayer name = {out_ch_, in_ch_, kh_, kw_, sh_, sw_, ph_, pw_, w_, b_}

#define NORM2D_DESC(name, ch_, w_, b_) \
    static Norm2DLayer name = {ch_, w_, b_}

#define CONV1D_DESC(name, out_ch_, in_ch_, k_, s_, p_, w_, b_) \
    static Conv1DLayer name = {out_ch_, in_ch_, k_, s_, p_, w_, b_}

CONV2D_DESC(kEnc1Conv0, 32, 9, 3, 3, 1, 1, 1, 1, g_unet_enc1_block_0_weight, g_unet_enc1_block_0_bias);
NORM2D_DESC(kEnc1Norm0, 32, g_unet_enc1_block_1_weight, g_unet_enc1_block_1_bias);
CONV2D_DESC(kEnc1Conv1, 32, 32, 3, 3, 1, 1, 1, 1, g_unet_enc1_block_3_weight, g_unet_enc1_block_3_bias);
NORM2D_DESC(kEnc1Norm1, 32, g_unet_enc1_block_4_weight, g_unet_enc1_block_4_bias);
static ConvBlock2D kEnc1Block = {kEnc1Conv0, kEnc1Norm0, kEnc1Conv1, kEnc1Norm1};

CONV2D_DESC(kEnc2Conv0, 64, 32, 3, 3, 2, 2, 1, 1, g_unet_enc2_block_0_weight, g_unet_enc2_block_0_bias);
NORM2D_DESC(kEnc2Norm0, 64, g_unet_enc2_block_1_weight, g_unet_enc2_block_1_bias);
CONV2D_DESC(kEnc2Conv1, 64, 64, 3, 3, 1, 1, 1, 1, g_unet_enc2_block_3_weight, g_unet_enc2_block_3_bias);
NORM2D_DESC(kEnc2Norm1, 64, g_unet_enc2_block_4_weight, g_unet_enc2_block_4_bias);
static ConvBlock2D kEnc2Block = {kEnc2Conv0, kEnc2Norm0, kEnc2Conv1, kEnc2Norm1};

CONV2D_DESC(kEnc3Conv0, 128, 64, 3, 3, 2, 2, 1, 1, g_unet_enc3_block_0_weight, g_unet_enc3_block_0_bias);
NORM2D_DESC(kEnc3Norm0, 128, g_unet_enc3_block_1_weight, g_unet_enc3_block_1_bias);
CONV2D_DESC(kEnc3Conv1, 128, 128, 3, 3, 1, 1, 1, 1, g_unet_enc3_block_3_weight, g_unet_enc3_block_3_bias);
NORM2D_DESC(kEnc3Norm1, 128, g_unet_enc3_block_4_weight, g_unet_enc3_block_4_bias);
static ConvBlock2D kEnc3Block = {kEnc3Conv0, kEnc3Norm0, kEnc3Conv1, kEnc3Norm1};

CONV2D_DESC(kBottleneckConv0, 128, 128, 3, 3, 1, 1, 1, 1, g_unet_bottleneck_block_0_weight, g_unet_bottleneck_block_0_bias);
NORM2D_DESC(kBottleneckNorm0, 128, g_unet_bottleneck_block_1_weight, g_unet_bottleneck_block_1_bias);
CONV2D_DESC(kBottleneckConv1, 128, 128, 3, 3, 1, 1, 1, 1, g_unet_bottleneck_block_3_weight, g_unet_bottleneck_block_3_bias);
NORM2D_DESC(kBottleneckNorm1, 128, g_unet_bottleneck_block_4_weight, g_unet_bottleneck_block_4_bias);
static ConvBlock2D kBottleneckBlock = {kBottleneckConv0, kBottleneckNorm0, kBottleneckConv1, kBottleneckNorm1};

CONV2D_DESC(kDec3Conv0, 64, 256, 3, 3, 1, 1, 1, 1, g_unet_dec3_block_0_weight, g_unet_dec3_block_0_bias);
NORM2D_DESC(kDec3Norm0, 64, g_unet_dec3_block_1_weight, g_unet_dec3_block_1_bias);
CONV2D_DESC(kDec3Conv1, 64, 64, 3, 3, 1, 1, 1, 1, g_unet_dec3_block_3_weight, g_unet_dec3_block_3_bias);
NORM2D_DESC(kDec3Norm1, 64, g_unet_dec3_block_4_weight, g_unet_dec3_block_4_bias);
static ConvBlock2D kDec3Block = {kDec3Conv0, kDec3Norm0, kDec3Conv1, kDec3Norm1};

CONV2D_DESC(kDec2Conv0, 32, 128, 3, 3, 1, 1, 1, 1, g_unet_dec2_block_0_weight, g_unet_dec2_block_0_bias);
NORM2D_DESC(kDec2Norm0, 32, g_unet_dec2_block_1_weight, g_unet_dec2_block_1_bias);
CONV2D_DESC(kDec2Conv1, 32, 32, 3, 3, 1, 1, 1, 1, g_unet_dec2_block_3_weight, g_unet_dec2_block_3_bias);
NORM2D_DESC(kDec2Norm1, 32, g_unet_dec2_block_4_weight, g_unet_dec2_block_4_bias);
static ConvBlock2D kDec2Block = {kDec2Conv0, kDec2Norm0, kDec2Conv1, kDec2Norm1};

CONV2D_DESC(kDec1Conv0, 32, 64, 3, 3, 1, 1, 1, 1, g_unet_dec1_block_0_weight, g_unet_dec1_block_0_bias);
NORM2D_DESC(kDec1Norm0, 32, g_unet_dec1_block_1_weight, g_unet_dec1_block_1_bias);
CONV2D_DESC(kDec1Conv1, 32, 32, 3, 3, 1, 1, 1, 1, g_unet_dec1_block_3_weight, g_unet_dec1_block_3_bias);
NORM2D_DESC(kDec1Norm1, 32, g_unet_dec1_block_4_weight, g_unet_dec1_block_4_bias);
static ConvBlock2D kDec1Block = {kDec1Conv0, kDec1Norm0, kDec1Conv1, kDec1Norm1};

CONV2D_DESC(kOutMask, 1, 32, 1, 1, 1, 1, 0, 0, g_unet_out_mask_weight, g_unet_out_mask_bias);
CONV2D_DESC(kOutRes, 1, 32, 1, 1, 1, 1, 0, 0, g_unet_out_res_weight, g_unet_out_res_bias);
CONV2D_DESC(kOutPhase, 1, 32, 1, 1, 1, 1, 0, 0, g_unet_out_phase_weight, g_unet_out_phase_bias);

CONV1D_DESC(kTransientDelta0, 32, 1, 9, 1, 4, g_transient_delta_net_0_weight, g_transient_delta_net_0_bias);
CONV1D_DESC(kTransientDelta2, 32, 32, 9, 1, 4, g_transient_delta_net_2_weight, g_transient_delta_net_2_bias);
CONV1D_DESC(kTransientDelta4, 32, 32, 5, 1, 2, g_transient_delta_net_4_weight, g_transient_delta_net_4_bias);
CONV1D_DESC(kTransientDelta6, 1, 32, 1, 1, 0, g_transient_delta_net_6_weight, g_transient_delta_net_6_bias);
CONV1D_DESC(kTransientGate0, 8, 1, 7, 1, 3, g_transient_gate_net_0_weight, g_transient_gate_net_0_bias);
CONV1D_DESC(kTransientGate2, 1, 8, 1, 1, 0, g_transient_gate_net_2_weight, g_transient_gate_net_2_bias);

DaisyPod hw;

static bool g_initialized = false;

static uint16_t g_fft_bitrev[kFftSize];
static float g_fft_cos[9][kFftSize / 2];
static float g_fft_sin[9][kFftSize / 2];
static float g_istft_window_sumsquare[kPaddedFrameSize];

constexpr std::size_t kWindowFloats = kFrameSize;
#if defined(BASELINE_USE_SDRAM_WEIGHTS)
constexpr std::size_t kWeightFloats = kWeightBytes / sizeof(float);
static float DSY_SDRAM_BSS g_weights_sdram[kWeightFloats];
static float DSY_SDRAM_BSS g_window_sdram[kWindowFloats];
#endif
static const float* g_win = g_window;

static float DSY_SDRAM_BSS g_log_mag_all[kContextFrames * kTfSize];
static float DSY_SDRAM_BSS g_current_mag[kTfSize];
static float DSY_SDRAM_BSS g_current_log_mag[kTfSize];
static float DSY_SDRAM_BSS g_current_phase[kTfSize];
static float DSY_SDRAM_BSS g_features[kFeatureSize];

static float DSY_SDRAM_BSS g_enc1[kEnc1Size];
static float DSY_SDRAM_BSS g_enc2[kEnc2Size];
static float DSY_SDRAM_BSS g_enc3[kEnc3Size];
static float DSY_SDRAM_BSS g_bottleneck[kEnc3Size];
static float DSY_SDRAM_BSS g_upcat[kUpCatSize];
static float DSY_SDRAM_BSS g_tmp2d[kTmp2dSize];
static float DSY_SDRAM_BSS g_dec3[64 * 65 * 2];
static float DSY_SDRAM_BSS g_dec2[32 * 129 * 3];
static float DSY_SDRAM_BSS g_dec1[kEnc1Size];
static float DSY_SDRAM_BSS g_mask[kTfSize];
static float DSY_SDRAM_BSS g_residual[kTfSize];
static float DSY_SDRAM_BSS g_phase_delta[kTfSize];
static float DSY_SDRAM_BSS g_spec_re[kTfSize];
static float DSY_SDRAM_BSS g_spec_im[kTfSize];

static float DSY_SDRAM_BSS g_fft_re[kFftSize];
static float DSY_SDRAM_BSS g_fft_im[kFftSize];
static float DSY_SDRAM_BSS g_istft_accum[kPaddedFrameSize];

static float DSY_SDRAM_BSS g_tmp1d_a[kTmp1d32Size];
static float DSY_SDRAM_BSS g_tmp1d_b[kTmp1d32Size];
static float DSY_SDRAM_BSS g_tmp1d_c[kTmp1d32Size];
static float DSY_SDRAM_BSS g_tmp1d_gate[kTmp1d8Size];
static float DSY_SDRAM_BSS g_delta[kFrameSize];
static float DSY_SDRAM_BSS g_gate[kFrameSize];
static float DSY_SDRAM_BSS g_abs_input[kFrameSize];

#if defined(BASELINE_USE_SDRAM_WEIGHTS)
constexpr std::size_t kWeightCopyBytes = sizeof(g_weights_sdram) + sizeof(g_window_sdram);
#else
constexpr std::size_t kWeightCopyBytes = 0;
#endif

constexpr std::size_t kScratchBytes = kWeightCopyBytes + sizeof(g_log_mag_all) + sizeof(g_current_mag)
                                    + sizeof(g_current_log_mag) + sizeof(g_current_phase)
                                    + sizeof(g_features) + sizeof(g_enc1) + sizeof(g_enc2)
                                    + sizeof(g_enc3) + sizeof(g_bottleneck) + sizeof(g_upcat)
                                    + sizeof(g_tmp2d) + sizeof(g_dec3) + sizeof(g_dec2)
                                    + sizeof(g_dec1) + sizeof(g_mask) + sizeof(g_residual)
                                    + sizeof(g_phase_delta) + sizeof(g_spec_re)
                                    + sizeof(g_spec_im) + sizeof(g_fft_re) + sizeof(g_fft_im)
                                    + sizeof(g_istft_accum) + sizeof(g_tmp1d_a)
                                    + sizeof(g_tmp1d_b) + sizeof(g_tmp1d_c)
                                    + sizeof(g_tmp1d_gate) + sizeof(g_delta)
                                    + sizeof(g_gate) + sizeof(g_abs_input);

constexpr std::size_t kTotalStaticBytes = kScratchBytes + kWeightBytes;

inline int Idx2D(int c, int h, int w, int height, int width) {
    return (c * height + h) * width + w;
}

inline int Idx1D(int c, int t, int length) {
    return c * length + t;
}

inline int IdxTF(int freq, int time) {
    return freq * kStftFrames + time;
}

inline int IdxFeat(int channel, int freq, int time) {
    return (channel * kFreqBins + freq) * kStftFrames + time;
}

inline float Clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

inline float Gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * kSqrtHalf));
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

inline int ConvOutSize(int in_size, int kernel, int stride, int pad) {
    return ((in_size + 2 * pad - kernel) / stride) + 1;
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
    for(int i = 0; i < kFftSize; i++) {
        unsigned value = static_cast<unsigned>(i);
        unsigned rev= 0;

        for(int bit = 0; bit < 9; bit++) {
            rev = (rev << 1U) | (value & 1U);
            value >>= 1U;
        }
        g_fft_bitrev[i] = static_cast<uint16_t>(rev);
    }

    for(int stage = 0; stage < 9; stage++) {
        const int len = 1 << (stage + 1);
        const int half = len >> 1;
        const float base = -2.0f * kPi / static_cast<float>(len);
        
        for(int j = 0; j < half; j++) {
            const float angle = base * static_cast<float>(j);
            g_fft_cos[stage][j] = cosf(angle);
            g_fft_sin[stage][j] = sinf(angle);
        }
    }
}

void PrecomputeIstftWindowNorm() {
    ZeroBuffer(g_istft_window_sumsquare, kPaddedFrameSize);

    for(int frame = 0; frame < kStftFrames; frame++) {
        const int start = frame * kHopSize;

        for(int n = 0; n < kFrameSize; n++) {
            const float w = g_win[n];
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

    for(int stage = 0; stage < 9; stage++) {
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

void Conv2DForward(const float* input,
                   int in_h,
                   int in_w,
                   const Conv2DLayer& layer,
                   float* output) {
    const int out_h = ConvOutSize(in_h, layer.kernel_h, layer.stride_h, layer.pad_h);
    const int out_w = ConvOutSize(in_w, layer.kernel_w, layer.stride_w, layer.pad_w);

    for(int oc = 0; oc < layer.out_ch; oc++) {
        for(int oh = 0; oh < out_h; oh++) {
            for(int ow = 0; ow < out_w; ow++) {
                float sum = layer.bias[oc];
                for(int ic = 0; ic < layer.in_ch; ic++) {
                    for(int kh = 0; kh < layer.kernel_h; kh++) {
                        const int ih = oh * layer.stride_h + kh - layer.pad_h;

                        if(ih < 0 || ih >= in_h) continue;

                        for(int kw = 0; kw < layer.kernel_w; kw++) {
                            const int iw = ow * layer.stride_w + kw - layer.pad_w;
                            if(iw < 0 || iw >= in_w) continue;

                            const int input_idx = Idx2D(ic, ih, iw, in_h, in_w);
                            const int weight_idx = (((oc * layer.in_ch + ic) * layer.kernel_h+ kh)
                                                     * layer.kernel_w)
                                                     + kw;
                            sum += input[input_idx] * layer.weight[weight_idx];
                        }
                    }
                }
                output[Idx2D(oc, oh, ow, out_h, out_w)] = sum;
            }
        }
    }
}

void InstanceNorm2DInPlace(float* data, int channels, int height, int width, const Norm2DLayer& norm) {
    const int spatial = height * width;
    for(int c = 0; c < channels; c++) {
        float mean = 0.0f;
        for(int i = 0; i < spatial; i++) {
            mean += data[c * spatial + i];
        }

        mean /= static_cast<float>(spatial);

        float var = 0.0f;
        for(int i = 0; i < spatial; i++) {
            const float diff = data[c * spatial + i] - mean;
            var += diff * diff;
        }

        var /= static_cast<float>(spatial);

        const float inv_std = 1.0f / sqrtf(var + kInstanceNormEps);
        const float gamma = norm.weight[c];
        const float beta = norm.bias[c];

        for(int i = 0; i < spatial; i++) {
            data[c * spatial + i] = ((data[c * spatial + i] - mean) * inv_std) * gamma + beta;
        }
    }
}

void GeluInPlace(float* data, int count) {
    for(int i = 0; i < count; i++) {
        data[i] = Gelu(data[i]);
    }
}

void ConvBlock2DForward(const float* input,
                        int in_h,
                        int in_w,
                        const ConvBlock2D& block,
                        float* output) {
    const int mid_h = ConvOutSize(in_h, block.conv0.kernel_h, block.conv0.stride_h, block.conv0.pad_h);
    const int mid_w = ConvOutSize(in_w, block.conv0.kernel_w, block.conv0.stride_w, block.conv0.pad_w);
    Conv2DForward(input, in_h, in_w, block.conv0, g_tmp2d);
    InstanceNorm2DInPlace(g_tmp2d, block.conv0.out_ch, mid_h, mid_w, block.norm0);
    GeluInPlace(g_tmp2d, block.conv0.out_ch * mid_h * mid_w);

    const int out_h = ConvOutSize(mid_h, block.conv1.kernel_h, block.conv1.stride_h, block.conv1.pad_h);
    const int out_w = ConvOutSize(mid_w, block.conv1.kernel_w, block.conv1.stride_w, block.conv1.pad_w);
    Conv2DForward(g_tmp2d, mid_h, mid_w, block.conv1, output);
    InstanceNorm2DInPlace(output, block.conv1.out_ch, out_h, out_w, block.norm1);
    GeluInPlace(output, block.conv1.out_ch * out_h * out_w);
}

void BilinearUpsample(const float* input,
                      int channels,
                      int in_h,
                      int in_w,
                      int out_h,
                      int out_w,
                      float* output) {
    const float scale_h = static_cast<float>(in_h) / static_cast<float>(out_h);
    const float scale_w = static_cast<float>(in_w) / static_cast<float>(out_w);

    for(int c = 0; c < channels; c++) {
        for(int oh = 0; oh < out_h; oh++) {
            const float src_h = ((static_cast<float>(oh) + 0.5f) * scale_h) - 0.5f;
            const float h0f = floorf(src_h);
            const int h0 = static_cast<int>(h0f < 0.0f ? 0.0f : h0f);
            const int h1 = h0 + 1 < in_h ? h0 + 1 : in_h - 1;
            const float lh = src_h <= 0.0f ? 0.0f : src_h - static_cast<float>(h0);
            const float hh = 1.0f - lh;

            for(int ow = 0; ow < out_w; ow++) {
                const float src_w = ((static_cast<float>(ow) + 0.5f) * scale_w) - 0.5f;
                const float w0f = floorf(src_w);
                const int w0 = static_cast<int>(w0f < 0.0f ? 0.0f : w0f);
                const int w1 = w0 + 1 < in_w ? w0 + 1 : in_w - 1;
                const float lw = src_w <= 0.0f ? 0.0f : src_w - static_cast<float>(w0);
                const float hw = 1.0f - lw;

                const float v00 = input[Idx2D(c, h0, w0, in_h, in_w)];
                const float v01 = input[Idx2D(c, h0, w1, in_h, in_w)];
                const float v10 = input[Idx2D(c, h1, w0, in_h, in_w)];
                const float v11 = input[Idx2D(c, h1, w1, in_h, in_w)];

                output[Idx2D(c, oh, ow, out_h, out_w)] = hh * (hw * v00 + lw * v01) + lh * (hw * v10 + lw * v11);
            }
        }
    }
}

void UpsampleConcat(const float* input,
                    int in_ch,
                    int in_h,
                    int in_w,
                    const float* skip,
                    int skip_ch,
                    int skip_h,
                    int skip_w,
                    float* output) {
    BilinearUpsample(input, in_ch, in_h, in_w, skip_h, skip_w, output);
    std::memcpy(output + in_ch * skip_h * skip_w,skip,
                static_cast<std::size_t>(skip_ch * skip_h * skip_w) * sizeof(float));
}

void Conv1DForward(const float* input, int input_len, const Conv1DLayer& layer, float* output) {
    const int out_len = ConvOutSize(input_len, layer.kernel, layer.stride, layer.pad);
    for(int oc = 0; oc < layer.out_ch; oc++) {
        for(int t = 0; t < out_len; t++) {
            float sum = layer.bias[oc];

            for(int ic = 0; ic < layer.in_ch; ic++) {
                for(int k = 0; k < layer.kernel; k++) {
                    const int in_t = t * layer.stride + k - layer.pad;
                    if(in_t < 0 || in_t >= input_len) continue;

                    const int input_idx = Idx1D(ic, in_t, input_len);
                    const int weight_idx = (oc * layer.in_ch + ic) * layer.kernel + k;
                    sum += input[input_idx] * layer.weight[weight_idx];
                }
            }
            output[Idx1D(oc, t, out_len)] = sum;
        }
    }
}

void BuildSpectralFeatures(const float* audio_ctx) {
    const int newest_offset = (kContextFrames - 1) * kFrameSize;

    for(int ctx = 0; ctx < kContextFrames; ctx++) {
        const float* audio = audio_ctx + ctx * kFrameSize;

        for(int frame = 0; frame < kStftFrames; frame++) {
            const int start = frame * kHopSize - (kFrameSize / 2);

            for(int n = 0; n < kFrameSize; n++) {
                const int sample_idx = ReflectIndex(start + n, kFrameSize);
                g_fft_re[n] = audio[sample_idx] * g_win[n];
                g_fft_im[n] = 0.0f;
            }

            FftInPlace(g_fft_re, g_fft_im, false);

            for(int freq = 0; freq < kFreqBins; freq++) {
                const int tf_idx = ctx * kTfSize + IdxTF(freq, frame);
                const float re = g_fft_re[freq];
                const float im = g_fft_im[freq];
                const float mag = sqrtf(re * re + im * im);
                const float logmag = logf(mag < 1.0e-5f ? 1.0e-5f : mag);
                g_log_mag_all[tf_idx] = logmag;

                if(ctx == kContextFrames - 1) {
                    const int idx = IdxTF(freq, frame);
                    g_current_mag[idx] = mag;
                    g_current_log_mag[idx] = logmag;
                    g_current_phase[idx] = atan2f(im, re);
                }
            }
        }
    }

    float env_accum = 0.0f;
    float signed_mean_accum = 0.0f;
    const float* current_audio = audio_ctx + newest_offset;

    for(int i = 0; i < kFrameSize; i++) {
        env_accum += current_audio[i] * current_audio[i];
        signed_mean_accum += current_audio[i];
    }

    const float env = sqrtf(env_accum / static_cast<float>(kFrameSize) + 1.0e-8f);
    const float signed_mean = signed_mean_accum / static_cast<float>(kFrameSize);

    for(int freq = 0; freq < kFreqBins; freq++) {
        for(int time = 0; time < kStftFrames; time++) {
            const int cur_idx = IdxTF(freq, time);

            for(int ctx = 0; ctx < kContextFrames; ctx++) {
                const int source = ctx * kTfSize + cur_idx;
                g_features[IdxFeat(ctx, freq, time)] = g_log_mag_all[source];
            }
            
            for(int ctx = 0; ctx < kContextFrames - 1; ctx++) {
                const int source = ctx * kTfSize + cur_idx;
                g_features[IdxFeat(kContextFrames + ctx, freq, time)] = g_log_mag_all[source] - g_current_log_mag[cur_idx];
            }

            g_features[IdxFeat(7, freq, time)] = env;
            g_features[IdxFeat(8, freq, time)] = signed_mean;
        }
    }
}

void SpectralUnetForward()
{
    ConvBlock2DForward(g_features, 257, 5, kEnc1Block, g_enc1);
    ConvBlock2DForward(g_enc1, 257, 5, kEnc2Block, g_enc2);
    ConvBlock2DForward(g_enc2, 129, 3, kEnc3Block, g_enc3);
    ConvBlock2DForward(g_enc3, 65, 2, kBottleneckBlock, g_bottleneck);

    UpsampleConcat(g_bottleneck, 128, 65, 2, g_enc3, 128, 65, 2, g_upcat);
    ConvBlock2DForward(g_upcat, 65, 2, kDec3Block, g_dec3);

    UpsampleConcat(g_dec3, 64, 65, 2, g_enc2, 64, 129, 3, g_upcat);
    ConvBlock2DForward(g_upcat, 129, 3, kDec2Block, g_dec2);

    UpsampleConcat(g_dec2, 32, 129, 3, g_enc1, 32, 257, 5, g_upcat);
    ConvBlock2DForward(g_upcat, 257, 5, kDec1Block, g_dec1);

    Conv2DForward(g_dec1, 257, 5, kOutMask, g_mask);
    Conv2DForward(g_dec1, 257, 5, kOutRes, g_residual);
    Conv2DForward(g_dec1, 257, 5, kOutPhase, g_phase_delta);

    for(int i = 0; i < kTfSize; i++) {
        g_mask[i] = 0.5f + 2.5f / (1.0f + expf(-g_mask[i]));
        g_phase_delta[i] = 0.45f * tanhf(g_phase_delta[i]);
    }
}

void ReconstructWaveform(float* audio_out) {
    for(int i = 0; i < kTfSize; i++) {
        const float out_log_mag = g_current_log_mag[i] * g_mask[i] + g_residual[i];
        const float out_mag = expf(out_log_mag);
        const float out_phase = g_current_phase[i] + g_phase_delta[i];
        g_spec_re[i] = out_mag * cosf(out_phase);
        g_spec_im[i] = out_mag * sinf(out_phase);
    }

    ZeroBuffer(g_istft_accum, kPaddedFrameSize);

    for(int frame = 0; frame < kStftFrames; frame++) {
        ZeroBuffer(g_fft_re, kFftSize);
        ZeroBuffer(g_fft_im, kFftSize);

        for(int freq = 0; freq < kFreqBins; freq++) {
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
        for(int n = 0; n < kFrameSize; n++) {
            g_istft_accum[start + n] += g_fft_re[n] * g_win[n];
        }
    }

    for(int n = 0; n < kFrameSize; n++) {
        const int padded_idx = n + (kFrameSize / 2);
        const float denom = g_istft_window_sumsquare[padded_idx];
        audio_out[n] = denom > 1.0e-11f ? g_istft_accum[padded_idx] / denom : 0.0f;
    }
}

void TransientShaperForward(float* audio_out) {
    Conv1DForward(audio_out, kFrameSize, kTransientDelta0, g_tmp1d_a);
    GeluInPlace(g_tmp1d_a, kTmp1d32Size);
    Conv1DForward(g_tmp1d_a, kFrameSize, kTransientDelta2, g_tmp1d_b);
    GeluInPlace(g_tmp1d_b, kTmp1d32Size);
    Conv1DForward(g_tmp1d_b, kFrameSize, kTransientDelta4, g_tmp1d_c);
    GeluInPlace(g_tmp1d_c, kTmp1d32Size);
    Conv1DForward(g_tmp1d_c, kFrameSize, kTransientDelta6, g_delta);

    for(int i = 0; i < kFrameSize; i++) {
        g_abs_input[i] = fabsf(audio_out[i]);
    }

    Conv1DForward(g_abs_input, kFrameSize, kTransientGate0, g_tmp1d_gate);
    GeluInPlace(g_tmp1d_gate, kTmp1d8Size);
    Conv1DForward(g_tmp1d_gate, kFrameSize, kTransientGate2, g_gate);

    for(int i = 0; i < kFrameSize; i++) {
        g_gate[i] = 1.0f / (1.0f + expf(-g_gate[i]));
        audio_out[i] += 0.25f * g_gate[i] * g_delta[i];
    }
}

#if defined(BASELINE_USE_SDRAM_WEIGHTS)
static float* g_sdram_cursor;

const float* CopyToSdram(const float* src, int count) {
    float* dst = g_sdram_cursor;
    std::memcpy(dst, src, static_cast<std::size_t>(count) * sizeof(float));
    g_sdram_cursor += count;
    return dst;
}

void CopyConv2D(Conv2DLayer& layer, int weight_count, int bias_count) {
    layer.weight = CopyToSdram(layer.weight, weight_count);
    layer.bias   = CopyToSdram(layer.bias, bias_count);
}

void CopyNorm2D(Norm2DLayer& layer) {
    layer.weight = CopyToSdram(layer.weight, layer.channels);
    layer.bias   = CopyToSdram(layer.bias, layer.channels);
}

void CopyConv1D(Conv1DLayer& layer) {
    layer.weight = CopyToSdram(layer.weight, layer.out_ch * layer.in_ch * layer.kernel);
    layer.bias = CopyToSdram(layer.bias, layer.out_ch);
}

void CopyBlock(ConvBlock2D& block) {
    CopyConv2D(block.conv0,
               block.conv0.out_ch * block.conv0.in_ch * block.conv0.kernel_h * block.conv0.kernel_w,
               block.conv0.out_ch);
    CopyNorm2D(block.norm0);
    CopyConv2D(block.conv1,
               block.conv1.out_ch * block.conv1.in_ch * block.conv1.kernel_h * block.conv1.kernel_w,
               block.conv1.out_ch);
    CopyNorm2D(block.norm1);
}

void CopyWeightsToSdram() {
    g_sdram_cursor = g_weights_sdram;

    CopyBlock(kEnc1Block);
    CopyBlock(kEnc2Block);
    CopyBlock(kEnc3Block);
    CopyBlock(kBottleneckBlock);
    CopyBlock(kDec3Block);
    CopyBlock(kDec2Block);
    CopyBlock(kDec1Block);

    CopyConv2D(kOutMask, kOutMask.out_ch * kOutMask.in_ch * kOutMask.kernel_h * kOutMask.kernel_w, kOutMask.out_ch);
    CopyConv2D(kOutRes, kOutRes.out_ch * kOutRes.in_ch * kOutRes.kernel_h * kOutRes.kernel_w, kOutRes.out_ch);
    CopyConv2D(kOutPhase, kOutPhase.out_ch * kOutPhase.in_ch * kOutPhase.kernel_h * kOutPhase.kernel_w, kOutPhase.out_ch);

    CopyConv1D(kTransientDelta0);
    CopyConv1D(kTransientDelta2);
    CopyConv1D(kTransientDelta4);
    CopyConv1D(kTransientDelta6);
    CopyConv1D(kTransientGate0);
    CopyConv1D(kTransientGate2);

    std::memcpy(g_window_sdram, g_window, sizeof(g_window_sdram));
    g_win = g_window_sdram;
}
#endif

void BaselineInit() {
    if(g_initialized) return;
#if defined(BASELINE_USE_SDRAM_WEIGHTS)
    CopyWeightsToSdram();
#endif
    PrecomputeFftTables();
    PrecomputeIstftWindowNorm();
    g_initialized = true;
}

void BaselineInfer(const float audio_ctx[kContextFrames][kFrameSize], float audio_out[kFrameSize]) {
    BaselineInit();
    BuildSpectralFeatures(&audio_ctx[0][0]);
    SpectralUnetForward();
    ReconstructWaveform(audio_out);
    TransientShaperForward(audio_out);

    const float* current_audio = audio_ctx[kContextFrames - 1];
    for(int i = 0; i < kFrameSize; i++) {
        audio_out[i] = Clamp(0.985f * audio_out[i] + 0.015f * current_audio[i], -1.0f, 1.0f);
    }
}

void ComputeError(const float* pred, const float* golden, float& max_abs, float& rmse) {
    max_abs = 0.0f;
    double accum = 0.0;

    for(int i = 0; i < kFrameSize; i++) {
        const float diff = pred[i] - golden[i];
        const float ad = fabsf(diff);
        if(ad > max_abs) max_abs = ad;
        accum += static_cast<double>(diff) * static_cast<double>(diff);
    }
    rmse = sqrtf(static_cast<float>(accum / static_cast<double>(kFrameSize)));
}

uint32_t CyclesToUsX1000(uint32_t cycles, uint32_t cpu_hz) {
    return static_cast<uint32_t>((static_cast<uint64_t>(cycles) * 1000000000ULL) / static_cast<uint64_t>(cpu_hz));
}

uint32_t CyclesToHopPctX100(uint32_t cycles, uint32_t cpu_hz) {
    const uint64_t hop_budget_cycles = (static_cast<uint64_t>(cpu_hz) * static_cast<uint64_t>(kHopSize)) / static_cast<uint64_t>(kSampleRate);

    if(hop_budget_cycles == 0) {
        return 0;
    }

    return static_cast<uint32_t>((static_cast<uint64_t>(cycles) * 10000ULL) / hop_budget_cycles);
}

void PrintMemorySummary() {
    hw.seed.PrintLine("Model params: %lu", static_cast<unsigned long>(kParameterCount));
    hw.seed.PrintLine("Weight bytes: %lu", static_cast<unsigned long>(kWeightBytes));
    hw.seed.PrintLine("Weight source: %s", kUseSdramWeights ? "SDRAM copy" : "QSPI");
    hw.seed.PrintLine("Scratch bytes: %lu", static_cast<unsigned long>(kScratchBytes));
    hw.seed.PrintLine("Static total bytes: %lu", static_cast<unsigned long>(kTotalStaticBytes));
}

void RunBenchmarkCase(const BenchmarkCase& bench_case, uint32_t cpu_hz) {
    alignas(16) static float output[kFrameSize];

    const auto* input = reinterpret_cast<const float (*)[kFrameSize]>(bench_case.input);
    hw.seed.PrintLine("----------------------------------------");
    hw.seed.PrintLine("Running case: %s", bench_case.name);
    hw.seed.PrintLine("Warmup/bench: %d / %d", kWarmupIters, kBenchIters);

    for(int i = 0; i < kWarmupIters; i++) {
        BaselineInfer(input, output);
    }

    uint32_t min_cycles = 0xffffffffu;
    uint32_t max_cycles = 0u;
    double sum_cycles = 0.0;

    for(int i = 0; i < kBenchIters; ++i) {
        const uint32_t start = DWT->CYCCNT;
        BaselineInfer(input, output);
        const uint32_t end = DWT->CYCCNT;
        const uint32_t cycles = end - start;

        if(cycles < min_cycles) min_cycles = cycles;
        if(cycles > max_cycles) max_cycles = cycles;
        sum_cycles += static_cast<double>(cycles);
    }

    float max_abs = 0.0f;
    float rmse = 0.0f;
    ComputeError(output, bench_case.golden, max_abs, rmse);

    const uint32_t avg_cycles = static_cast<uint32_t>((sum_cycles / static_cast<double>(kBenchIters)) + 0.5);
    const uint32_t min_us_x1000 = CyclesToUsX1000(min_cycles, cpu_hz);
    const uint32_t avg_us_x1000 = CyclesToUsX1000(avg_cycles, cpu_hz);
    const uint32_t max_us_x1000 = CyclesToUsX1000(max_cycles, cpu_hz);
    const uint32_t hop_pct_x100 = CyclesToHopPctX100(avg_cycles, cpu_hz);
    const uint32_t max_abs_x1000000 = static_cast<uint32_t>(max_abs * 1000000.0f + 0.5f);
    const uint32_t rmse_x1000000 = static_cast<uint32_t>(rmse * 1000000.0f + 0.5f);

    hw.seed.PrintLine("Done case: %s", bench_case.name);
    hw.seed.PrintLine("min/avg/max cyc: %lu / %lu / %lu",
                      static_cast<unsigned long>(min_cycles),
                      static_cast<unsigned long>(avg_cycles),
                      static_cast<unsigned long>(max_cycles));
    hw.seed.PrintLine("min/avg/max us: %lu.%03lu / %lu.%03lu / %lu.%03lu",
                      static_cast<unsigned long>(min_us_x1000 / 1000U),
                      static_cast<unsigned long>(min_us_x1000 % 1000U),
                      static_cast<unsigned long>(avg_us_x1000 / 1000U),
                      static_cast<unsigned long>(avg_us_x1000 % 1000U),
                      static_cast<unsigned long>(max_us_x1000 / 1000U),
                      static_cast<unsigned long>(max_us_x1000 % 1000U));
    hw.seed.PrintLine("avg hop budget: %lu.%02lu%%",
                      static_cast<unsigned long>(hop_pct_x100 / 100U),
                      static_cast<unsigned long>(hop_pct_x100 % 100U));
    hw.seed.PrintLine("max abs / rmse: %lu.%06lu / %lu.%06lu",
                      static_cast<unsigned long>(max_abs_x1000000 / 1000000U),
                      static_cast<unsigned long>(max_abs_x1000000 % 1000000U),
                      static_cast<unsigned long>(rmse_x1000000 / 1000000U),
                      static_cast<unsigned long>(rmse_x1000000 % 1000000U));
}

void BaselineRunBenchmarks() {
    static const BenchmarkCase kCases[] = {
        {"zero", case_zero_input, case_zero_golden},
        {"impulse", case_impulse_input, case_impulse_golden},
        {"random", case_random_input, case_random_golden},
    };

    const uint32_t cpu_hz = System::GetSysClkFreq();

    hw.seed.PrintLine("");
    hw.seed.PrintLine("Daisy raw-C baseline");
    hw.seed.PrintLine("CPU Hz: %lu", static_cast<unsigned long>(cpu_hz));
    hw.seed.PrintLine("Frame/Hop: %d / %d", kFrameSize, kHopSize);
    hw.seed.PrintLine("STFT bins/frames: %d / %d", kFreqBins, kStftFrames);
    hw.seed.PrintLine("Feature channels: %d", kFeatureChannels);
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
    BaselineInit();
    BaselineRunBenchmarks();

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
