/**
 * onsetnet_pod.cpp  — v6
 *
 * Architecture:
 *   - 8ms rolling audio window, processed every 8ms (one audio block)
 *   - NN runs on the window → top-K note candidates
 *   - Goertzel runs on the SAME window, guided by NN top-K → picks best pitch
 *   - Voicing gate (RMS dBFS) determines if audio is present
 *   - Note is ON whenever: voiced AND NN confident (no pick detector required)
 *   - Pick detector only used to detect NEW note onset (pitch change)
 *   - Grace window holds note on after confidence drops
 *
 * Knob 1 = synth gain
 * Knob 2 = grace time (0–500ms)
 * Button 1 = cycle waveform (sine/square/saw)
 * Button 2 = mute
 *
 * LEDs:
 *   LED1 GREEN  = note active, NN confident
 *   LED1 RED    = note in grace (holding on after confidence dropped)
 *   LED2 BLUE   = voiced (signal present)
 *   LED2 WHITE  = note just changed (brief flash)
 */

#include "daisy_pod.h"
#include "daisy_seed.h"
#include "daisysp.h"
#include "onset_weights.h"
#include "tflite_onset_runner.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace daisy;
using namespace daisysp;

#define NOTE_HEAD_OUT 50

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
static constexpr int   SR              = ONSET_SR;       // 48000
static constexpr int   kWinSamp        = ONSET_WINDOW;   // 384 = 8ms
static constexpr float kHopMs          = 8.0f;           // must match block size
static constexpr int   kTopK           = 3;
static constexpr float kNnOn           = 0.35f;   // enter note if confidence >= this
static constexpr float kNnOff          = 0.25f;   // exit note if confidence < this
static constexpr float kVoicedDbOn     = -75.0f;  // voiced if dBFS >= this
static constexpr float kVoicedDbOff    = -85.0f;  // unvoiced if dBFS <= this
static constexpr float kDefaultGraceMs = 10.0f;
static constexpr float kMinNoteMs      = 0.0f;
static constexpr float kFreqSmooth     = 0.7f;
static constexpr float kPreemph        = 0.10f;
static constexpr float kFftCents       = 80.0f;   // search window ±80 cents
static constexpr float kFftMinHz       = 70.0f;
static constexpr float kSynthAttackMs  = 4.0f;
static constexpr float kSynthReleaseMs = 150.0f;

// ---------------------------------------------------------------------------
// Hardware
// ---------------------------------------------------------------------------
static DaisyPod hw;

// ---------------------------------------------------------------------------
// Synth
// ---------------------------------------------------------------------------
enum Wave { SINE=0, SQUARE, SAW, N_WAVES };

struct Synth {
    int sr; Wave wave;
    float ph, freq, gain, env, tgt, sup, sdn;

    void init(int _sr, float att_ms, float rel_ms) {
        sr=_sr; wave=SINE; ph=0; freq=440; gain=0.5f; env=0; tgt=0;
        int as = (int)(att_ms*1e-3f*sr); if(as<1)as=1;
        int rs = (int)(rel_ms*1e-3f*sr); if(rs<1)rs=1;
        sup=1.0f/as; sdn=1.0f/rs;
    }
    void set_freq(float f)   { if(f>5.0f) freq=f; }
    void set_active(bool on) { tgt = on ? 1.0f : 0.0f; }

    void render(float* out, int n) {
        const float tw   = 2.0f * (float)M_PI;
        const float step = tw * freq / (float)sr;
        for(int i=0;i<n;i++){
            if(env<tgt){ env+=sup; if(env>tgt)env=tgt; }
            else if(env>tgt){ env-=sdn; if(env<tgt)env=tgt; }
            float s;
            switch(wave){
                case SQUARE: s = ph<(float)M_PI ? 1.0f : -1.0f; break;
                case SAW:    s = ph/(float)M_PI - 1.0f;          break;
                default:     s = sinf(ph);                        break;
            }
            out[i] = s * gain * env;
            ph += step; if(ph >= tw) ph -= tw;
        }
    }
} g_synth;

// ---------------------------------------------------------------------------
// Ring buffer  — written from audio ISR, read from main loop
// ---------------------------------------------------------------------------
static constexpr int kRingSize = SR / 4;  // 250ms
static float g_ring[kRingSize];
static volatile int g_ring_w    = 0;
static volatile int g_ring_fill = 0;

static void ring_get_last(float* out, int n) {
    // safe to call from main loop while ISR writes — worst case one sample glitch
    int w = g_ring_w;
    int start = (w - n + kRingSize) % kRingSize;
    for(int i=0;i<n;i++) out[i] = g_ring[(start+i) % kRingSize];
}

// ---------------------------------------------------------------------------
// Audio callback
// ---------------------------------------------------------------------------
static bool  g_muted = false;

static void AudioCallback(AudioHandle::InputBuffer  in,
                          AudioHandle::OutputBuffer out,
                          size_t size)
{
    hw.ProcessAllControls();

    if(hw.button1.RisingEdge())
        g_synth.wave = (Wave)((g_synth.wave + 1) % N_WAVES);
    if(hw.button2.RisingEdge()){
        g_muted = !g_muted;
        if(g_muted) g_synth.set_active(false);
    }

    // Knob 1 → gain, floor at 0.3 so it can't accidentally be zero
    float k1 = hw.knob1.Process();
    g_synth.gain = k1 < 0.06f ? 0.3f : k1 * 0.7f;

    // Push mono input into ring
    int w = g_ring_w;
    for(size_t i=0;i<size;i++){
        g_ring[w] = (in[0][i] + in[1][i]) * 0.5f;
        if(++w >= kRingSize) w = 0;
    }
    g_ring_w    = w;
    g_ring_fill += (int)size;
    if(g_ring_fill > kRingSize) g_ring_fill = kRingSize;

    // Render synth to output
    float buf[kWinSamp];
    g_synth.render(buf, (int)size);
    for(size_t i=0;i<size;i++){
        float s = g_muted ? 0.0f : buf[i];
        out[0][i] = s;
        out[1][i] = s;
    }
}

// ---------------------------------------------------------------------------
// DSP helpers
// ---------------------------------------------------------------------------
static float calc_rms(const float* x, int n){
    float s=1e-12f;
    for(int i=0;i<n;i++) s+=x[i]*x[i];
    return sqrtf(s/(float)n);
}
static float calc_dbfs(float r){ return 20.0f*log10f(r<1e-12f?1e-12f:r); }

static void do_preemph(float* y, const float* x, int n, float a){
    y[0]=x[0];
    for(int i=1;i<n;i++) y[i]=x[i]-a*x[i-1];
}

static void rms_norm(float* y, const float* x, int n){
    float r=calc_rms(x,n);
    if(r<1e-4f){ memset(y,0,n*sizeof(float)); return; }
    float inv=1.0f/r;
    for(int i=0;i<n;i++) y[i]=x[i]*inv;
}

// ---------------------------------------------------------------------------
// Goertzel pitch search
// Evaluates power at f + first two harmonics
// ---------------------------------------------------------------------------
static float goertzel_power(const float* x, int n, float f){
    float w     = 2.0f*(float)M_PI*f/(float)SR;
    float coeff = 2.0f*cosf(w);
    float s0=0,s1=0,s2=0;
    for(int i=0;i<n;i++){ s0=x[i]+coeff*s1-s2; s2=s1; s1=s0; }
    return s1*s1 + s2*s2 - coeff*s1*s2;
}

static float harmonic_power(const float* x, int n, float f){
    float p = goertzel_power(x, n, f);
    if(f*2.0f < SR*0.48f) p += 0.55f * goertzel_power(x, n, f*2.0f);
    if(f*3.0f < SR*0.48f) p += 0.35f * goertzel_power(x, n, f*3.0f);
    if(f*4.0f < SR*0.48f) p += 0.20f * goertzel_power(x, n, f*4.0f);
    return p;
}

// Search ±cents around centre_hz, nsteps points
static float refine_hz(const float* x, int n, float centre_hz, float cents,
                        int nsteps=16){
    float ratio = powf(2.0f, cents/1200.0f);
    float lo = centre_hz / ratio;
    float hi = centre_hz * ratio;
    if(lo < kFftMinHz) lo = kFftMinHz;
    float bp=-1.0f, bh=centre_hz;
    for(int s=0;s<=nsteps;s++){
        float f = lo + (hi-lo)*(float)s/(float)nsteps;
        float p = harmonic_power(x, n, f);
        if(p > bp){ bp=p; bh=f; }
    }
    return bh;
}

// Given top-K candidates, run Goertzel on each and return the best
// Returns index into candidates array (not midi index)
static int best_candidate(const float* x, int n,
                           const float* cand_hz, const float* cand_conf,
                           int k, float* best_hz_out){
    float bp = -1.0f;
    int   bi = 0;
    float bh = cand_hz[0];
    for(int i=0;i<k;i++){
        if(cand_hz[i] < kFftMinHz) continue;
        float p = harmonic_power(x, n, cand_hz[i]);
        // weight by NN confidence too
        float score = p * (0.6f + 0.4f * cand_conf[i]);
        if(score > bp){ bp=score; bi=i; bh=cand_hz[i]; }
    }
    // refine around winner
    *best_hz_out = refine_hz(x, n, bh, kFftCents);
    return bi;
}

// ---------------------------------------------------------------------------
// MIDI vocab
// ---------------------------------------------------------------------------
static float g_midi_vocab[NOTE_HEAD_OUT];
static inline float midi_to_hz(float m){ return 440.0f*powf(2.0f,(m-69.0f)/12.0f); }
static inline float hz_to_midi(float f){ return 69.0f+12.0f*log2f(f>1e-9f?f/440.0f:1e-9f); }

static int pitch_name_to_midi(const char* s){
    int n;
    switch(s[0]){
        case 'C':n=0;break; case 'D':n=2;break; case 'E':n=4;break;
        case 'F':n=5;break; case 'G':n=7;break; case 'A':n=9;break;
        case 'B':n=11;break; default:n=0;
    }
    int i=1;
    if(s[i]=='#'){ n++; i++; }
    return ((s[i]-'0')+1)*12 + n;
}

static void build_midi_vocab(){
    for(int i=0;i<ONSET_NOTE_VOCAB_SIZE;i++)
        g_midi_vocab[i] = (float)pitch_name_to_midi(onset_pitch_names[i]);
    if(NOTE_HEAD_OUT > ONSET_NOTE_VOCAB_SIZE)
        g_midi_vocab[NOTE_HEAD_OUT-1] = 0.0f;  // silence class
}

// ---------------------------------------------------------------------------
// OnsetNet forward pass
// ---------------------------------------------------------------------------
static float g_win[kWinSamp], g_pe[kWinSamp], g_nn_in[kWinSamp];
using NNOut = onset_tflite::Output;

static void nn_forward(NNOut& o) {
    onset_tflite::forward(g_nn_in, kWinSamp, o);
}

static void top_k(const float* p, int n, int k, int* idx){
    bool used[NOTE_HEAD_OUT]={};
    for(int i=0;i<k;i++){
        float best=-1.0f; int bi=0;
        for(int j=0;j<n;j++) if(!used[j]&&p[j]>best){ best=p[j]; bi=j; }
        idx[i]=bi; used[bi]=true;
    }
}

// ---------------------------------------------------------------------------
// Note state machine
//
// Key difference from v5: the synth is ON whenever we are voiced AND confident.
// We do NOT require a pick event to start playing.
// Pick events only reset the grace window (new note onset).
// ---------------------------------------------------------------------------
static NNOut  g_nn_out;
static bool   g_voiced      = false;
static bool   g_note_active = false;
static int    g_note_midi   = -1;
static float  g_note_hz     = 0.0f;
static int    g_grace_left  = 0;
static int    g_min_left    = 0;
static int    g_change_flash= 0;

struct TickNnProfile {
    uint32_t ticks;
    uint32_t window_ticks;
    uint32_t voicing_ticks;
    uint32_t prep_ticks;
    uint32_t nn_ticks;
    uint32_t candidates_ticks;
    uint32_t goertzel_ticks;
    uint32_t state_ticks;
    uint32_t leds_ticks;
    uint32_t total_ticks;
    uint32_t max_total_ticks;
    bool ready;
};

static constexpr uint32_t kProfileEveryTicks = 32;
static TickNnProfile g_tick_nn_profile = {};

static inline uint32_t ticks_to_us(uint32_t ticks) {
    uint32_t ticks_per_us = System::GetTickFreq() / 1000000U;
    if(ticks_per_us == 0) ticks_per_us = 1;
    return ticks / ticks_per_us;
}

static inline void profile_tick_nn(uint32_t window_ticks,
                                   uint32_t voicing_ticks,
                                   uint32_t prep_ticks,
                                   uint32_t nn_ticks,
                                   uint32_t candidates_ticks,
                                   uint32_t goertzel_ticks,
                                   uint32_t state_ticks,
                                   uint32_t leds_ticks,
                                   uint32_t total_ticks) {
    g_tick_nn_profile.ticks += 1;
    g_tick_nn_profile.window_ticks += window_ticks;
    g_tick_nn_profile.voicing_ticks += voicing_ticks;
    g_tick_nn_profile.prep_ticks += prep_ticks;
    g_tick_nn_profile.nn_ticks += nn_ticks;
    g_tick_nn_profile.candidates_ticks += candidates_ticks;
    g_tick_nn_profile.goertzel_ticks += goertzel_ticks;
    g_tick_nn_profile.state_ticks += state_ticks;
    g_tick_nn_profile.leds_ticks += leds_ticks;
    g_tick_nn_profile.total_ticks += total_ticks;
    if(total_ticks > g_tick_nn_profile.max_total_ticks)
        g_tick_nn_profile.max_total_ticks = total_ticks;
    if(g_tick_nn_profile.ticks >= kProfileEveryTicks)
        g_tick_nn_profile.ready = true;
}

// To see console output, connect to the Daisy serial connection
// I used 'screen /dev/tty.usb*' to connect to the USB serial interface

static void print_tick_nn_profile(){
    if(!g_tick_nn_profile.ready || g_tick_nn_profile.ticks == 0)
        return;

    const uint32_t ticks = g_tick_nn_profile.ticks;
    hw.seed.PrintLine(
        "tick_nn avg_us[%lu]: total=%lu win=%lu voice=%lu prep=%lu nn=%lu cand=%lu goertzel=%lu state=%lu led=%lu max=%lu",
        (unsigned long)ticks,
        (unsigned long)ticks_to_us(g_tick_nn_profile.total_ticks / ticks),
        (unsigned long)ticks_to_us(g_tick_nn_profile.window_ticks / ticks),
        (unsigned long)ticks_to_us(g_tick_nn_profile.voicing_ticks / ticks),
        (unsigned long)ticks_to_us(g_tick_nn_profile.prep_ticks / ticks),
        (unsigned long)ticks_to_us(g_tick_nn_profile.nn_ticks / ticks),
        (unsigned long)ticks_to_us(g_tick_nn_profile.candidates_ticks / ticks),
        (unsigned long)ticks_to_us(g_tick_nn_profile.goertzel_ticks / ticks),
        (unsigned long)ticks_to_us(g_tick_nn_profile.state_ticks / ticks),
        (unsigned long)ticks_to_us(g_tick_nn_profile.leds_ticks / ticks),
        (unsigned long)ticks_to_us(g_tick_nn_profile.max_total_ticks));
    g_tick_nn_profile = {};
}

static bool tick_nn(){
    uint32_t t0 = System::GetTick();
    uint32_t t_prev = t0;

    // 1. Get window
    ring_get_last(g_win, kWinSamp);
    uint32_t t1 = System::GetTick();
    uint32_t window_ticks = t1 - t_prev;
    t_prev = t1;

    // 2. Voicing gate
    float r  = calc_rms(g_win, kWinSamp);
    float db = calc_dbfs(r);
    if(!g_voiced){ if(db >= kVoicedDbOn)  g_voiced=true; }
    else         { if(db <= kVoicedDbOff) g_voiced=false; }
    uint32_t t2 = System::GetTick();
    uint32_t voicing_ticks = t2 - t_prev;
    t_prev = t2;

    // 3. Pre-emphasis + RMS normalise for NN
    do_preemph(g_pe, g_win, kWinSamp, kPreemph);
    rms_norm(g_nn_in, g_pe, kWinSamp);
    uint32_t t3 = System::GetTick();
    uint32_t prep_ticks = t3 - t_prev;
    t_prev = t3;

    // 4. NN forward
    nn_forward(g_nn_out);
    uint32_t t4 = System::GetTick();
    uint32_t nn_ticks = t4 - t_prev;
    t_prev = t4;

    // 5. Top-K candidates — skip silence class (index 49)
    int tk[kTopK];
    top_k(g_nn_out.note_probs, NOTE_HEAD_OUT, kTopK, tk);

    // Build candidate hz + conf arrays for Goertzel
    float cand_hz[kTopK], cand_conf[kTopK];
    int   cand_midi_idx[kTopK];
    int   valid_k = 0;
    for(int i=0;i<kTopK;i++){
        int idx = tk[i];
        if(idx >= ONSET_NOTE_VOCAB_SIZE) continue;  // skip silence
        float m = g_midi_vocab[idx];
        if(m < 28.0f || m > 96.0f) continue;        // sanity range
        cand_hz[valid_k]      = midi_to_hz(m);
        cand_conf[valid_k]    = g_nn_out.note_probs[idx];
        cand_midi_idx[valid_k]= idx;
        valid_k++;
    }
    uint32_t t5 = System::GetTick();
    uint32_t candidates_ticks = t5 - t_prev;
    t_prev = t5;

    // NN top-1 confidence (ignoring silence)
    float nn_conf = (valid_k > 0) ? cand_conf[0] : 0.0f;

    // 6. Goertzel — find best pitch among NN candidates
    float refined_hz = 0.0f;
    int   winner     = 0;
    if(valid_k > 0 && g_voiced){
        winner = best_candidate(g_nn_in, kWinSamp,
                                cand_hz, cand_conf, valid_k,
                                &refined_hz);
    }
    uint32_t t6 = System::GetTick();
    uint32_t goertzel_ticks = t6 - t_prev;
    t_prev = t6;

    // 7. Grace frame counts from knob2
    float grace_ms = hw.knob2.Process() * 500.0f + 50.0f;  // 50–550ms
    int grace_f = (int)(grace_ms / kHopMs); if(grace_f<1) grace_f=1;
    int min_f   = (int)(kMinNoteMs / kHopMs); if(min_f<1) min_f=1;

    // 8. Note state machine
    //    Condition to be "playing": voiced AND nn_conf >= kNnOn AND valid pitch
    bool should_play = g_voiced && (nn_conf >= kNnOn) && (refined_hz > kFftMinHz);

    if(should_play){
        int new_midi = (valid_k>0) ? (int)(g_midi_vocab[cand_midi_idx[winner]] + 0.5f) : g_note_midi;

        if(!g_note_active){
            // Start new note
            g_note_active = true;
            g_note_midi   = new_midi;
            g_note_hz     = refined_hz;
            g_grace_left  = grace_f;
            g_min_left    = min_f;
            g_change_flash= 6;
            if(!g_muted){ g_synth.set_freq(g_note_hz); g_synth.set_active(true); }
        } else {
            // Already playing — update pitch smoothly
            // If note changed by more than 1 semitone, snap immediately
            float semitone_diff = fabsf(hz_to_midi(refined_hz) - (float)g_note_midi);
            if(semitone_diff > 1.2f && nn_conf > 0.35f){
                // Note changed
                g_note_midi    = new_midi;
                g_note_hz      = refined_hz;
                g_change_flash = 6;
            } else {
                // Same note — smooth frequency tracking
                g_note_hz = kFreqSmooth * g_note_hz + (1.0f-kFreqSmooth) * refined_hz;
            }
            g_grace_left = grace_f;  // refresh grace while confident
            if(!g_muted) g_synth.set_freq(g_note_hz);
        }
    } else if(g_note_active){
        // Losing confidence — count down grace
        if(g_min_left > 0) g_min_left--;

        if(g_voiced && nn_conf >= kNnOff){
            // Still somewhat confident while voiced — refresh grace
            // (hysteresis: kNnOff < kNnOn)
            g_grace_left = grace_f;
            // Keep tracking pitch even in grace
            if(valid_k > 0 && refined_hz > kFftMinHz)
                g_note_hz = kFreqSmooth * g_note_hz + (1.0f-kFreqSmooth) * refined_hz;
            if(!g_muted) g_synth.set_freq(g_note_hz);
        } else {
            if(g_min_left <= 0) g_grace_left--;
        }

        if(g_grace_left <= 0 && g_min_left <= 0){
            g_note_active = false;
            g_note_midi   = -1;
            g_note_hz     = 0.0f;
            g_synth.set_active(false);
        }
    }
    uint32_t t7 = System::GetTick();
    uint32_t state_ticks = t7 - t_prev;
    t_prev = t7;

    // 9. LEDs
    bool in_grace = g_note_active && !should_play;
    hw.led1.Set(
        in_grace ? 1.0f : 0.0f,
        (g_note_active && !in_grace) ? 1.0f : 0.0f,
        0.0f
    );
    if(g_change_flash > 0){ g_change_flash--; hw.led2.Set(1,1,1); }
    else hw.led2.Set(0.0f, 0.0f, g_voiced ? 1.0f : 0.0f);
    hw.UpdateLeds();
    uint32_t t8 = System::GetTick();
    uint32_t leds_ticks = t8 - t_prev;

    profile_tick_nn(window_ticks,
                    voicing_ticks,
                    prep_ticks,
                    nn_ticks,
                    candidates_ticks,
                    goertzel_ticks,
                    state_ticks,
                    leds_ticks,
                    t8 - t0);
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(){
    hw.Init(true);
    // Block size = window size so each callback delivers exactly one hop
    hw.SetAudioBlockSize(kWinSamp);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);

    build_midi_vocab();
    g_synth.init(SR, kSynthAttackMs, kSynthReleaseMs);
    hw.seed.StartLog();
    hw.seed.PrintLine("Pod booted");
    
    if(onset_tflite::init()){
        hw.seed.PrintLine("TFLM ready, arena=%lu bytes",
                          (unsigned long)onset_tflite::arena_used_bytes());
    } else {
        hw.seed.PrintLine("TFLM init failed: %s", onset_tflite::init_error());
        onset_tflite::log_init_snapshot(hw.seed);
    }

    hw.StartAudio(AudioCallback);

    // Brief LED startup flash so you know it booted
    hw.led1.Set(0,1,0); hw.led2.Set(0,1,0); hw.UpdateLeds();
    System::Delay(300);
    hw.led1.Set(0,0,0); hw.led2.Set(0,0,0); hw.UpdateLeds();

    uint32_t last = System::GetNow();
    while(true){
        uint32_t now = System::GetNow();
        // Run one tick per audio block (every 8ms)
        if((int32_t)(now-last) >= (int32_t)kHopMs){
            last = now;
            tick_nn();
            print_tick_nn_profile();
        }
    }
}
