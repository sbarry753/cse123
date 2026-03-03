#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

// Your generated weights header:
#include "onset_weights.h"

// -------------------------
// Tiny helpers
// -------------------------
static inline float sigmoidf_fast(float x)
{
    // good enough for gating
    return 1.0f / (1.0f + expf(-x));
}

static inline float clampf(float x, float lo, float hi)
{
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

static inline float rmsf(const float* x, int n)
{
    double acc = 0.0;
    for(int i = 0; i < n; i++)
        acc += (double)x[i] * (double)x[i];
    acc /= (double)(n > 0 ? n : 1);
    return (float)sqrt(acc + 1e-12);
}

static inline float dbfs_from_rms(float r)
{
    // dBFS where full-scale = 1.0
    if(r < 1e-12f)
        r = 1e-12f;
    return 20.0f * log10f(r);
}

// -------------------------
// OnsetNet forward pass
// matches the shapes implied by onset_weights.h:
//
// Input: [1, 384]
// Conv0: [48, 1, 9], stride2 pad4 -> T=192
// Conv1: [96, 48, 5], stride2 pad2 -> T=96
// Conv2: [96, 96, 5], stride2 pad2 -> T=48
// AvgPool over T -> [96]
// Heads:
// note:   [50, 96]
// string: [6,  96]
// picked: [1,  96]
// sus:    [1,  96]
// -------------------------
struct OnsetNetOut
{
    float note_logits[50];
    float string_logits[ONSET_N_STRINGS];
    float picked_logit;
    float sustain_logit;
};

// BatchNorm: y = (x - mean) * invstd * weight + bias
static inline float bn1(float x, float mean, float var, float w, float b)
{
    const float eps = 1e-5f;
    float invstd    = 1.0f / sqrtf(var + eps);
    return (x - mean) * invstd * w + b;
}

class OnsetNetInfer
{
  public:
    // x must be 384 samples, roughly normalized to ~RMS 1 (or close).
    void Forward(const float* x, OnsetNetOut& y) const
    {
        // Buffers sized by the known layer shapes.
        // (Static to avoid stack blowups if you prefer; but this is fine on H7.)
        float c0[48 * 192];
        float c1[96 * 96];
        float c2[96 * 48];

        // ---- Conv0 ----
        conv1d_1in(x,
                  384,
                  c0,
                  48,
                  192,
                  l0_conv_weight,
                  /*k=*/9,
                  /*stride=*/2,
                  /*pad=*/4);
        bn_relu_inplace(c0, 48, 192, l0_bn_running_mean, l0_bn_running_var, l0_bn_weight, l0_bn_bias);

        // ---- Conv1 ----
        conv1d_nin(c0,
                   48,
                   192,
                   c1,
                   96,
                   96,
                   l1_conv_weight,
                   /*k=*/5,
                   /*stride=*/2,
                   /*pad=*/2);
        bn_relu_inplace(c1, 96, 96, l1_bn_running_mean, l1_bn_running_var, l1_bn_weight, l1_bn_bias);

        // ---- Conv2 ----
        conv1d_nin(c1,
                   96,
                   96,
                   c2,
                   96,
                   48,
                   l2_conv_weight,
                   /*k=*/5,
                   /*stride=*/2,
                   /*pad=*/2);
        bn_relu_inplace(c2, 96, 48, l2_bn_running_mean, l2_bn_running_var, l2_bn_weight, l2_bn_bias);

        // ---- Global avg pool over time (T=48) -> feat[96] ----
        float feat[96];
        for(int oc = 0; oc < 96; oc++)
        {
            double acc = 0.0;
            const float* row = &c2[oc * 48];
            for(int t = 0; t < 48; t++)
                acc += row[t];
            feat[oc] = (float)(acc / 48.0);
        }

        // ---- Heads ----
        // Note logits (50)
        for(int i = 0; i < 50; i++)
        {
            double acc = (double)note_head_bias[i];
            const float* w = &note_head_weight[i * 96];
            for(int j = 0; j < 96; j++)
                acc += (double)w[j] * (double)feat[j];
            y.note_logits[i] = (float)acc;
        }

        // String logits (6)
        for(int i = 0; i < ONSET_N_STRINGS; i++)
        {
            double acc = (double)string_head_bias[i];
            const float* w = &string_head_weight[i * 96];
            for(int j = 0; j < 96; j++)
                acc += (double)w[j] * (double)feat[j];
            y.string_logits[i] = (float)acc;
        }

        // Pick / sustain
        {
            double accp = (double)picked_head_bias[0];
            double accs = (double)sustain_head_bias[0];
            for(int j = 0; j < 96; j++)
            {
                accp += (double)picked_head_weight[j] * (double)feat[j];
                accs += (double)sustain_head_weight[j] * (double)feat[j];
            }
            y.picked_logit  = (float)accp;
            y.sustain_logit = (float)accs;
        }
    }

    // Softmax for 50 logits -> probs
    static void Softmax50(const float* logits, float* probs)
    {
        float mx = logits[0];
        for(int i = 1; i < 50; i++)
            if(logits[i] > mx) mx = logits[i];

        double sum = 0.0;
        for(int i = 0; i < 50; i++)
        {
            double e = exp((double)(logits[i] - mx));
            probs[i] = (float)e;
            sum += e;
        }
        float inv = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
        for(int i = 0; i < 50; i++)
            probs[i] *= inv;
    }

  private:
    static void bn_relu_inplace(float* x,
                               int C,
                               int T,
                               const float* mean,
                               const float* var,
                               const float* w,
                               const float* b)
    {
        for(int c = 0; c < C; c++)
        {
            float m = mean[c];
            float v = var[c];
            float ww = w[c];
            float bb = b[c];
            float* row = &x[c * T];
            for(int t = 0; t < T; t++)
            {
                float y = bn1(row[t], m, v, ww, bb);
                row[t] = (y > 0.0f) ? y : 0.0f;
            }
        }
    }

    // Conv for in_ch = 1
    static void conv1d_1in(const float* in,
                          int inT,
                          float* out,
                          int outC,
                          int outT,
                          const float* w, // [outC,1,K]
                          int K,
                          int stride,
                          int pad)
    {
        for(int oc = 0; oc < outC; oc++)
        {
            const float* ww = &w[oc * K];
            float* o = &out[oc * outT];
            for(int ot = 0; ot < outT; ot++)
            {
                int base = ot * stride - pad;
                double acc = 0.0;
                for(int k = 0; k < K; k++)
                {
                    int it = base + k;
                    if((unsigned)it < (unsigned)inT)
                        acc += (double)ww[k] * (double)in[it];
                }
                o[ot] = (float)acc;
            }
        }
    }

    // Conv for general inC
    static void conv1d_nin(const float* in,
                           int inC,
                           int inT,
                           float* out,
                           int outC,
                           int outT,
                           const float* w, // [outC,inC,K]
                           int K,
                           int stride,
                           int pad)
    {
        for(int oc = 0; oc < outC; oc++)
        {
            float* o = &out[oc * outT];
            const float* w_oc = &w[oc * (inC * K)];
            for(int ot = 0; ot < outT; ot++)
            {
                int base = ot * stride - pad;
                double acc = 0.0;
                for(int ic = 0; ic < inC; ic++)
                {
                    const float* inrow = &in[ic * inT];
                    const float* ww = &w_oc[ic * K];
                    for(int k = 0; k < K; k++)
                    {
                        int it = base + k;
                        if((unsigned)it < (unsigned)inT)
                            acc += (double)ww[k] * (double)inrow[it];
                    }
                }
                o[ot] = (float)acc;
            }
        }
    }
};