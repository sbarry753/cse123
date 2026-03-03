#include "daisysp.h"

// Choose one:
// - For Pod:  include "daisy_pod.h"
// - For Seed: include "daisy_seed.h"
#if defined(USE_DAISY_POD)
#include "daisy_pod.h"
using HW = daisy::DaisyPod;
#else
#include "daisy_seed.h"
using HW = daisy::DaisySeed;
#endif

#include "onsetnet_infer.h"

using namespace daisy;
using namespace daisysp;

static HW hw;
static OnsetNetInfer nn;

static float ring[ONSET_WINDOW];
static int   ring_w = 0;
static bool  ring_filled = false;

// Preemph + RMS norm (lightweight version)
static inline void preprocess_window(const float* in, float* out, int n, float preemph)
{
    // preemph: y[t] = x[t] - a*x[t-1]
    out[0] = in[0];
    for(int i = 1; i < n; i++)
        out[i] = in[i] - preemph * in[i - 1];

    float r = rmsf(out, n);
    if(r < 1e-4f)
    {
        for(int i = 0; i < n; i++) out[i] = 0.0f;
        return;
    }
    float inv = 1.0f / r;
    for(int i = 0; i < n; i++) out[i] *= inv;
}

// Very simple “pick detector”: EMA + slope threshold
struct PickDet
{
    float ema = 0.0f;
    float prev = 0.0f;

    bool Step(float p, float alpha, float min_p, float rise_thresh)
    {
        prev = ema;
        ema = alpha * ema + (1.0f - alpha) * p;
        float dp = ema - prev;
        return (ema >= min_p) && (dp >= rise_thresh);
    }
};

static PickDet pickdet;

// Voicing hysteresis
static bool voiced = false;
static void update_voiced(float db, float on_db, float off_db)
{
    if(!voiced)
    {
        if(db >= on_db) voiced = true;
    }
    else
    {
        if(db <= off_db) voiced = false;
    }
}

// Simple synth
static float phase = 0.0f;
static float freq_hz = 220.0f;
static float env = 0.0f;
static float env_target = 0.0f;

static inline float midi_to_hz(float m)
{
    return 440.0f * powf(2.0f, (m - 69.0f) / 12.0f);
}

// Map note index -> display name (49 names), last idx=49 is NONE
static const char* note_name_from_idx(int idx)
{
    if(idx >= 0 && idx < ONSET_NOTE_VOCAB_SIZE)
        return onset_pitch_names[idx];
    return "—"; // none / unknown
}

// You’ll probably tune these:
static constexpr float PREEMPH = 0.10f;

static constexpr float VOICED_ON_DB  = -45.0f;
static constexpr float VOICED_OFF_DB = -55.0f;

static constexpr float PICK_EMA_ALPHA = 0.85f;
static constexpr float PICK_MIN_P     = 0.10f;
static constexpr float PICK_RISE      = 0.02f;

static constexpr float NOTE_ON_PROB   = 0.38f; // “confident enough”
static constexpr float NOTE_OFF_PROB  = 0.25f; // (optional) hysteresis for release

static int   active_note = -1; // 0..48, or -1
static float active_conf = 0.0f;

static void AudioCallback(AudioHandle::InputBuffer  in,
                          AudioHandle::OutputBuffer out,
                          size_t size)
{
    for(size_t i = 0; i < size; i++)
    {
        // mono in (use L)
        float x = in[0][i];

        // ring buffer (window=384)
        ring[ring_w++] = x;
        if(ring_w >= ONSET_WINDOW)
        {
            ring_w = 0;
            ring_filled = true;
        }

        // ---- run NN once per callback sample-block? ----
        // Instead: run when we wrap (every 384 samples ~8ms).
        // If you want a 2ms hop, run every 96 samples with a counter.
        if(ring_filled && ring_w == 0)
        {
            // grab contiguous window
            float win[ONSET_WINDOW];
            // ring_w == 0 => already contiguous in ring[]
            memcpy(win, ring, sizeof(win));

            float db = dbfs_from_rms(rmsf(win, ONSET_WINDOW));
            update_voiced(db, VOICED_ON_DB, VOICED_OFF_DB);

            float x_norm[ONSET_WINDOW];
            preprocess_window(win, x_norm, ONSET_WINDOW, PREEMPH);

            OnsetNetOut y;
            nn.Forward(x_norm, y);

            float probs[50];
            OnsetNetInfer::Softmax50(y.note_logits, probs);

            float pick_p = sigmoidf_fast(y.picked_logit);
            float sus_p  = sigmoidf_fast(y.sustain_logit);

            // pick trigger (block triggers while sustain is high if you want)
            bool pick = pickdet.Step(pick_p, PICK_EMA_ALPHA, PICK_MIN_P, PICK_RISE) && (sus_p < 0.75f);

            // best note (0..49)
            int best = 0;
            float bestp = probs[0];
            for(int k = 1; k < 50; k++)
            {
                if(probs[k] > bestp)
                {
                    bestp = probs[k];
                    best = k;
                }
            }

            // Start a note only on PICK + voiced + confident + not “none”
            if(pick && voiced && best != 49 && bestp >= NOTE_ON_PROB)
            {
                active_note = best; // 0..48
                active_conf = bestp;

                // If you trained these as actual pitches already, you can map index->midi here.
                // Since the header only includes pitch names, we’ll do a quick hard-map by parsing:
                // (For simplicity here: approximate via a tiny table is better; see note below.)
                // Quick: E2 is MIDI 40, then steps vary—so instead: make a midi table once.
                // For now, just “musical-ish”: treat idx as chromatic starting at E2:
                float midi = 40.0f + (float)active_note;
                freq_hz = midi_to_hz(midi);

                env_target = 1.0f;

                // Optional: print over USB serial (works if you enable logging)
                // hw.PrintLine("PICK %s (p=%.2f) voiced=%d", note_name_from_idx(best), bestp, (int)voiced);
            }

            // Release if unvoiced and confidence drops (simple hysteresis)
            if(active_note >= 0)
            {
                if(!voiced || (best == 49) || (bestp < NOTE_OFF_PROB))
                    env_target = 0.0f;
            }
        }

        // Synth render (simple sine + tiny AR envelope)
        // envelope time constants roughly ~2ms attack, ~35ms release at 48k
        float atk = 1.0f / (0.002f * 48000.0f);
        float rel = 1.0f / (0.035f * 48000.0f);
        float step = (env_target > env) ? atk : rel;
        env += (env_target - env) * clampf(step, 0.0f, 1.0f);

        phase += (2.0f * M_PI) * (freq_hz / 48000.0f);
        if(phase >= 2.0f * M_PI)
            phase -= 2.0f * M_PI;

        float s = sinf(phase) * env * 0.14f;

        out[0][i] = s;
        out[1][i] = s;
    }
}

int main(void)
{
#if defined(USE_DAISY_POD)
    hw.Init();
#else
    hw.Configure();
    hw.Init();
#endif

    hw.SetAudioBlockSize(48); // 1ms @ 48k; fine for low latency
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

    hw.StartAudio(AudioCallback);

    while(1)
    {
#if defined(USE_DAISY_POD)
        hw.ProcessAllControls();
        // You can show active note on LEDs here if you want
        System::Delay(5);
#else
        System::Delay(5);
#endif
    }
}