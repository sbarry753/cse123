/**
 * onsetnet_desktop.cpp  — low-latency (~8ms window) NN + guided FFT pitch, with smoother playback.
 *
 *   1) FIXED data race: synth control params (freq/tgt) are atomic (audio cb reads, tick writes).
 *   2) Removed per-hop printf spam: status prints ~20 Hz (every 50ms), ON/OFF still immediate.
 *   3) Removed per-hop heap+median: replaced FFT "median" with a cheap running mag-floor estimator.
 *   4) Note-name switching now requires FFT evidence + persistence (candidate latch), NN is only permission.
 *   5) Tracking FFT targets: locked note always; NN top1 only if very confident (so NN is guidance, not steering).
 *
 * Build:
 *   g++ -std=c++14 -O2 -o onsetnet onsetnet_desktop.cpp -lportaudio -lm
   g++ -std=c++14 -O2 \
  -I/opt/homebrew/include \
  -L/opt/homebrew/lib \
  -o onsetnet desktop.cpp \
  -lportaudio \
  -framework CoreAudio \
  -framework AudioToolbox \
  -framework AudioUnit \
  -framework CoreFoundation \
  -framework CoreServices \
  -lm
 *
 *
 * Run:
 *   ./onsetnet --list
 *   ./onsetnet --in 1 --out 3
 *   ./onsetnet --synth-gain 0.3 --synth-wave square
 *   ./onsetnet --nn-on 0.38 --voiced-db -45
 */

#include "onset_weights.h"
#include <portaudio.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <atomic>
#include <csignal>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NOTE_HEAD_OUT 50

// ---------------------------------------------------------------------------
// CLI config  (defaults match Python defaults)
// ---------------------------------------------------------------------------
static float cfg_synth_gain    = 0.14f;
static float cfg_synth_atk     = 2.0f;
static float cfg_synth_rel     = 35.0f;
static float cfg_nn_on         = 0.38f;
static float cfg_nn_off        = 0.25f;
static float cfg_voiced_on     = -75.0f;
static float cfg_voiced_off    = -85.0f;
static float cfg_grace_ms      = 10.0f;
static float cfg_min_note_ms   = 4.0f;
static float cfg_freq_smooth   = 0.15f;
static float cfg_fft_cents     = 60.0f;
static float cfg_fft_weight    = 0.40f;
static float cfg_fft_min_hz    = 80.0f;
static float cfg_pick_min_p    = 0.05f;
static float cfg_pick_rise     = 0.15f;
static float cfg_pick_reset    = 0.10f;
static float cfg_sus_hold_ms   = 8.0f;
static float cfg_sus_block     = 0.75f;
static float cfg_preemph       = 0.10f;
static int   cfg_topk          = 3;
static int   cfg_in_dev        = -1;
static int   cfg_out_dev       = -1;

// New: note switch hysteresis (FFT-evidence latch)
static float cfg_switch_cents  = 35.0f;  // how close to the new semitone before we consider switching
static float cfg_switch_fused  = 0.50f;  // minimum fused confidence to allow switching
static int   cfg_switch_frames = 4;      // persistence in hops; 4*2ms=8ms

enum WaveType { WAVE_SINE=0, WAVE_SQUARE, WAVE_SAW };
static WaveType cfg_wave = WAVE_SINE;

static constexpr int   SR        = ONSET_SR;      // 48000
static constexpr int   kWinSamp  = ONSET_WINDOW;  // 384 = 8ms
static constexpr float kHopMs    = 2.0f;

// FFT size — use 4096, zero-pad the 384-sample window into it
static constexpr int   kFftSize  = 4096;

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
static std::atomic<bool> g_running{true};
static void on_sigint(int){ g_running=false; }

// ---------------------------------------------------------------------------
// Synth  (atomic control params to avoid data race with audio callback)
// ---------------------------------------------------------------------------
struct Synth {
    WaveType wave  = WAVE_SINE;
    float    ph    = 0.f;

    std::atomic<float> freq_atomic{440.f};
    std::atomic<float> tgt_atomic{0.f};

    float    gain  = 0.14f;
    float    env   = 0.f;
    float    sup   = 0.f;
    float    sdn   = 0.f;

    void init(float att_ms, float rel_ms) {
        int as = (int)(att_ms * 1e-3f * SR); if(as<1)as=1;
        int rs = (int)(rel_ms * 1e-3f * SR); if(rs<1)rs=1;
        sup = 1.f / as;
        sdn = 1.f / rs;
    }
    void set_freq(float f)   { if(f > 5.f) freq_atomic.store(f, std::memory_order_relaxed); }
    void set_active(bool on) { tgt_atomic.store(on ? 1.f : 0.f, std::memory_order_relaxed); }

    float next_sample() {
        float freq = freq_atomic.load(std::memory_order_relaxed);
        float tgt  = tgt_atomic.load(std::memory_order_relaxed);

        if(env < tgt){ env += sup; if(env > tgt) env = tgt; }
        else if(env > tgt){ env -= sdn; if(env < tgt) env = tgt; }

        float s;
        switch(wave){
            case WAVE_SQUARE: s = ph < (float)M_PI ? 1.f : -1.f; break;
            case WAVE_SAW:    s = ph / (float)M_PI - 1.f;         break;
            default:          s = sinf(ph);                        break;
        }
        ph += 2.f * (float)M_PI * freq / SR;
        if(ph >= 2.f * (float)M_PI) ph -= 2.f * (float)M_PI;
        return s * gain * env;
    }
} g_synth;

// ---------------------------------------------------------------------------
// Ring buffer
// ---------------------------------------------------------------------------
static constexpr int kRingSize = SR / 2;
static float         g_ring[kRingSize];
static std::atomic<int> g_ring_w{0};

static void ring_push(const float* x, int n) {
    int w = g_ring_w.load(std::memory_order_relaxed);
    for(int i=0;i<n;i++){
        g_ring[w] = x[i];
        if(++w >= kRingSize) w = 0;
    }
    g_ring_w.store(w, std::memory_order_release);
}

static void ring_get_last(float* out, int n) {
    int w     = g_ring_w.load(std::memory_order_acquire);
    int start = (w - n + kRingSize) % kRingSize;
    for(int i=0;i<n;i++) out[i] = g_ring[(start+i) % kRingSize];
}

static int g_in_channels = 1;

// ---------------------------------------------------------------------------
// PortAudio callbacks  (separate input + output streams)
// ---------------------------------------------------------------------------
static int pa_in_cb(const void* input, void*, unsigned long frames,
                    const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void*)
{
    if(!input) return paContinue;
    const float* in = (const float*)input;
    float mono[kWinSamp];
    if(g_in_channels == 1){
        for(unsigned long i=0;i<frames;i++) mono[i] = in[i];
    } else {
        for(unsigned long i=0;i<frames;i++) mono[i] = (in[i*2] + in[i*2+1]) * 0.5f;
    }
    ring_push(mono, (int)frames);
    return paContinue;
}

static int pa_out_cb(const void*, void* output, unsigned long frames,
                     const PaStreamCallbackTimeInfo*, PaStreamCallbackFlags, void*)
{
    float* out = (float*)output;
    for(unsigned long i=0;i<frames;i++){
        float s = g_synth.next_sample();
        out[i*2]   = s;
        out[i*2+1] = s;
    }
    return paContinue;
}

// ---------------------------------------------------------------------------
// DSP helpers
// ---------------------------------------------------------------------------
static float calc_rms(const float* x, int n){
    float s=1e-12f;
    for(int i=0;i<n;i++) s+=x[i]*x[i];
    return sqrtf(s/(float)n);
}
static float calc_dbfs(float r){ return 20.f*log10f(r<1e-12f?1e-12f:r); }

// Pre-emphasis filter
static void do_preemph(float* y, const float* x, int n, float a){
    y[0] = x[0];
    for(int i=1;i<n;i++) y[i] = x[i] - a*x[i-1];
}

// RMS normalise with floor
static void rms_norm_floor(float* y, const float* x, int n, float floor_rms=1e-4f){
    float r = calc_rms(x, n);
    if(r < floor_rms){ memset(y, 0, n*sizeof(float)); return; }
    float inv = 1.f/r;
    for(int i=0;i<n;i++) y[i] = x[i]*inv;
}

// ---------------------------------------------------------------------------
// FFT (iterative radix-2)
// ---------------------------------------------------------------------------
struct Complex { float r, i; };

static void fft_inplace(Complex* x, int n) {
    for(int i=1,j=0; i<n; i++){
        int bit = n>>1;
        for(; j&bit; bit>>=1) j^=bit;
        j^=bit;
        if(i<j){ Complex t=x[i]; x[i]=x[j]; x[j]=t; }
    }
    for(int len=2; len<=n; len<<=1){
        float ang = -2.f*(float)M_PI/len;
        float wr=cosf(ang), wi=sinf(ang);
        for(int i=0; i<n; i+=len){
            float cr=1.f, ci=0.f;
            for(int j=0; j<len/2; j++){
                Complex u=x[i+j], v={x[i+j+len/2].r*cr - x[i+j+len/2].i*ci,
                                      x[i+j+len/2].r*ci + x[i+j+len/2].i*cr};
                x[i+j]        = {u.r+v.r, u.i+v.i};
                x[i+j+len/2]  = {u.r-v.r, u.i-v.i};
                float ncr = cr*wr - ci*wi;
                ci = cr*wi + ci*wr;
                cr = ncr;
            }
        }
    }
}

static Complex g_fft_buf[kFftSize];
static float   g_fft_mag[kFftSize/2+1];
static float   g_fft_win[kWinSamp];   // Hann window precomputed

static void build_hann_window(){
    for(int i=0;i<kWinSamp;i++)
        g_fft_win[i] = 0.5f*(1.f - cosf(2.f*(float)M_PI*i/(kWinSamp-1)));
}

static void compute_fft_mag(const float* x){
    for(int i=0;i<kFftSize;i++){
        if(i < kWinSamp) g_fft_buf[i] = {x[i]*g_fft_win[i], 0.f};
        else            g_fft_buf[i] = {0.f, 0.f};
    }
    fft_inplace(g_fft_buf, kFftSize);
    int half = kFftSize/2+1;
    for(int i=0;i<half;i++)
        g_fft_mag[i] = sqrtf(g_fft_buf[i].r*g_fft_buf[i].r + g_fft_buf[i].i*g_fft_buf[i].i);
}

// ---------------------------------------------------------------------------
// Parabolic peak refinement
// ---------------------------------------------------------------------------
static float parabolic_refine(const float* mag, int n, int k){
    if(k<=0 || k>=n-1) return (float)k;
    float a=mag[k-1], b=mag[k], c=mag[k+1];
    float denom = a - 2.f*b + c;
    if(fabsf(denom) < 1e-12f) return (float)k;
    float delta = 0.5f*(a-c)/denom;
    return (float)k + delta;
}

// ---------------------------------------------------------------------------
// Guided FFT pitch finding
// ---------------------------------------------------------------------------
struct GuidedResult {
    int   midi;
    float prob;
    float refined_hz;
    float harm_score;
};

static constexpr float kBinHz = (float)SR / (float)kFftSize;

static GuidedResult guided_fft(const float* mag, int mag_len,
                               const int* midi_targets, const float* probs, int k,
                               float cents, float min_hz)
{
    float ratio = powf(2.f, cents/1200.f);
    GuidedResult best{-1, 0.f, 0.f, -1.f};

    for(int t=0;t<k;t++){
        int   midi  = midi_targets[t];
        float prob  = probs[t];
        float c_hz  = 440.f * powf(2.f, (midi-69.f)/12.f);
        float lo    = c_hz / ratio;
        float hi    = c_hz * ratio;
        if(lo < min_hz) lo = min_hz;

        int lo_bin = (int)floorf(lo / kBinHz);
        int hi_bin = (int)ceilf(hi / kBinHz);
        if(lo_bin < 2) lo_bin = 2;
        if(hi_bin >= mag_len) hi_bin = mag_len-2;
        if(hi_bin <= lo_bin) continue;

        int pk = lo_bin;
        float pkv = mag[lo_bin];
        for(int b=lo_bin+1;b<=hi_bin;b++){
            if(mag[b]>pkv){ pkv=mag[b]; pk=b; }
        }

        float pk_ref = parabolic_refine(mag, mag_len, pk);
        float f_ref  = pk_ref * kBinHz;

        float harm = pkv;
        for(int h=2;h<=4;h++){
            int hb = (int)roundf(f_ref * h / kBinHz);
            if(hb>=0 && hb<mag_len) harm += mag[hb] * (0.7f/h);
        }

        if(harm > best.harm_score){
            best.midi        = midi;
            best.prob        = prob;
            best.refined_hz  = f_ref;
            best.harm_score  = harm;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Fast FFT confidence: replace per-hop median with running magnitude floor
// ---------------------------------------------------------------------------
static float g_mag_floor = 1e-6f;

static void update_mag_floor(const float* mag, int mag_len){
    // robust cheap estimate: median-of-9 spaced bins (avoids heap / nth_element on huge vectors)
    float s[9];
    for(int i=0;i<9;i++){
        int b = 10 + (i * (mag_len-20))/8;
        if(b < 0) b = 0;
        if(b >= mag_len) b = mag_len-1;
        s[i] = mag[b];
    }
    std::nth_element(s, s+4, s+9);
    float med9 = s[4];

    float a = 0.95f;
    g_mag_floor = a*g_mag_floor + (1.f-a)*med9;
    if(g_mag_floor < 1e-9f) g_mag_floor = 1e-9f;
}

static float fuse_confidence_fast(float nn_conf, float harm_score, float fft_weight){
    float fft_conf = harm_score / (g_mag_floor + 1e-9f) / 10.f;
    if(fft_conf > 1.f) fft_conf = 1.f;
    if(fft_conf < 0.f) fft_conf = 0.f;
    return (1.f - fft_weight)*nn_conf + fft_weight*fft_conf;
}

// ---------------------------------------------------------------------------
// MIDI vocab
// ---------------------------------------------------------------------------
static float g_vocab[NOTE_HEAD_OUT];
static inline float midi_to_hz(float m){ return 440.f*powf(2.f,(m-69.f)/12.f); }
static inline float hz_to_midi(float f){ return 69.f+12.f*log2f(f>1e-9f?f/440.f:1e-9f); }

static float cents_between(float a_hz, float b_hz){
    return 1200.f * log2f((a_hz>1e-9f?a_hz:1e-9f) / (b_hz>1e-9f?b_hz:1e-9f));
}

static const char* NOTE_NAMES[12]={"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
static const char* mname(int m){
    static char b[8];
    snprintf(b,8,"%s%d",NOTE_NAMES[m%12],m/12-1);
    return b;
}
static int pname2midi(const char* s){
    int n;
    switch(s[0]){case 'C':n=0;break;case 'D':n=2;break;case 'E':n=4;break;
        case 'F':n=5;break;case 'G':n=7;break;case 'A':n=9;break;case 'B':n=11;break;default:n=0;}
    int i=1; if(s[i]=='#'){n++;i++;}
    return ((s[i]-'0')+1)*12+n;
}
static void build_vocab(){
    for(int i=0;i<ONSET_NOTE_VOCAB_SIZE;i++) g_vocab[i]=(float)pname2midi(onset_pitch_names[i]);
    if(NOTE_HEAD_OUT>ONSET_NOTE_VOCAB_SIZE) g_vocab[NOTE_HEAD_OUT-1]=0.f;
}

// ---------------------------------------------------------------------------
// OnsetNet forward pass  (unchanged)
// ---------------------------------------------------------------------------
static constexpr int kT0=192, kT1=96, kT2=48;
static float g_a0[48*kT0], g_a1[96*kT1], g_a2[96*kT2], g_z[ONSET_WIDTH];
static float g_win_buf[kWinSamp], g_pe[kWinSamp], g_nn_in[kWinSamp];

static void conv_bn_relu(
    const float* x, int ci, int Ti, float* o, int co, int To,
    const float* W, const float* bw, const float* bb,
    const float* bm, const float* bv, int K, int st, int pd)
{
    for(int c=0;c<co;c++){
        const float* Wc=W+c*ci*K; float* oc=o+c*To;
        for(int t=0;t<To;t++){
            float acc=0; int t0=t*st-pd;
            for(int j=0;j<ci;j++){
                const float* xj=x+j*Ti, *Wj=Wc+j*K;
                for(int k=0;k<K;k++){int ti=t0+k;if((unsigned)ti<(unsigned)Ti)acc+=xj[ti]*Wj[k];}
            }
            float bn=(acc-bm[c])/sqrtf(bv[c]+1e-5f)*bw[c]+bb[c];
            oc[t]=bn>0?bn:0;
        }
    }
}
static void softmax_ip(float* a, int n){
    float mx=a[0]; for(int i=1;i<n;i++) if(a[i]>mx)mx=a[i];
    float s=0; for(int i=0;i<n;i++){a[i]=expf(a[i]-mx);s+=a[i];}
    float inv=1/(s+1e-12f); for(int i=0;i<n;i++) a[i]*=inv;
}
static float sigmoid(float x){ return 1.f/(1.f+expf(-x)); }

struct NNOut{ float np[NOTE_HEAD_OUT]; float pk,su; };

static void nn_forward(NNOut& o){
    conv_bn_relu(g_nn_in,1,kWinSamp,g_a0,48,kT0,l0_conv_weight,l0_bn_weight,l0_bn_bias,l0_bn_running_mean,l0_bn_running_var,9,2,4);
    conv_bn_relu(g_a0,48,kT0,g_a1,96,kT1,l1_conv_weight,l1_bn_weight,l1_bn_bias,l1_bn_running_mean,l1_bn_running_var,5,2,2);
    conv_bn_relu(g_a1,96,kT1,g_a2,96,kT2,l2_conv_weight,l2_bn_weight,l2_bn_bias,l2_bn_running_mean,l2_bn_running_var,5,2,2);
    for(int c=0;c<ONSET_WIDTH;c++){
        float acc=0; const float* ch=g_a2+c*kT2;
        for(int t=0;t<kT2;t++) acc+=ch[t]; g_z[c]=acc/kT2;
    }
    for(int i=0;i<NOTE_HEAD_OUT;i++){
        float acc=note_head_bias[i]; const float* w=note_head_weight+i*ONSET_WIDTH;
        for(int j=0;j<ONSET_WIDTH;j++) acc+=w[j]*g_z[j]; o.np[i]=acc;
    }
    softmax_ip(o.np,NOTE_HEAD_OUT);
    {float a=picked_head_bias[0]; for(int j=0;j<ONSET_WIDTH;j++)a+=picked_head_weight[j]*g_z[j]; o.pk=sigmoid(a);}
    {float a=sustain_head_bias[0];for(int j=0;j<ONSET_WIDTH;j++)a+=sustain_head_weight[j]*g_z[j];o.su=sigmoid(a);}
}

static void topk(const float* p, int n, int k, int* idx){
    bool used[NOTE_HEAD_OUT]={};
    for(int i=0;i<k;i++){
        float b=-1; int bi=0;
        for(int j=0;j<n;j++) if(!used[j]&&p[j]>b){b=p[j];bi=j;}
        idx[i]=bi; used[bi]=true;
    }
}

// ---------------------------------------------------------------------------
// Pick detector
// ---------------------------------------------------------------------------
struct PickDet {
    float a, ema, prev, peak, msr;
    bool  armed, rising;
    int   rh, rhf, cd;

    void init(float hop_ms, float tau_ms=12.f, int reset_hold=2){
        a = expf(-hop_ms / (tau_ms>1e-6f?tau_ms:1e-6f));
        ema=prev=peak=0; msr=1; armed=true; rising=false;
        rh=0; rhf=reset_hold>1?reset_hold:1; cd=0;
    }

    bool step(float raw, float min_p, float rise_thresh,
              float fall_reset, int cdf, bool sus_blocked)
    {
        if(cd>0) cd--;
        prev=ema; ema=a*ema+(1-a)*raw; float dp=ema-prev;

        if(!armed){
            if(ema < fall_reset){ if(++rh>=rhf){armed=true;rising=false;peak=0;msr=ema;rh=0;}}
            else rh=0;
        }

        bool trig=false;
        if(armed && cd==0 && !sus_blocked){
            if(ema<msr) msr=ema;
            if(dp>0){ rising=true; if(ema>peak)peak=ema; }
            else if(rising && dp<=0){
                float rise_amt = peak - msr;
                trig = (peak>=min_p) && (rise_amt>=rise_thresh);
                if(trig){ cd=cdf>1?cdf:1; armed=false; rising=false; peak=0; msr=1; rh=0; }
                else     { rising=false; peak=0; }
            }
        }
        return trig;
    }
} g_pick;

// ---------------------------------------------------------------------------
// Note state
// ---------------------------------------------------------------------------
static NNOut  g_nn_out;
static bool   g_voiced      = false;
static bool   g_note_active = false;
static int    g_note_midi   = -1;
static float  g_note_hz     = 0.f;
static int    g_grace_left  = 0;
static int    g_min_left    = 0;

// Candidate latch for note-name switching
static int g_cand_midi  = -1;
static int g_cand_count = 0;

// Print throttle
static int g_print_div = 0;

static void update_voiced(float db){
    if(!g_voiced){ if(db >= cfg_voiced_on)  g_voiced=true; }
    else         { if(db <= cfg_voiced_off) g_voiced=false; }
}

static void tick(){
    // 1. Get window
    ring_get_last(g_win_buf, kWinSamp);

    // 2. Voicing gate
    float db = calc_dbfs(calc_rms(g_win_buf, kWinSamp));
    update_voiced(db);

    // 3. Preprocess for NN
    do_preemph(g_pe, g_win_buf, kWinSamp, cfg_preemph);
    rms_norm_floor(g_nn_in, g_pe, kWinSamp, 1e-4f);

    // 4. FFT on the same normalised window
    compute_fft_mag(g_nn_in);
    int mag_len = kFftSize/2 + 1;

    int min_bin = (int)floorf(cfg_fft_min_hz / kBinHz);
    for(int i=0;i<min_bin && i<mag_len;i++) g_fft_mag[i]=0.f;

    // Update running mag floor (for fast FFT confidence)
    update_mag_floor(g_fft_mag, mag_len);

    // 5. NN forward
    nn_forward(g_nn_out);

    // 6. Top-K candidates
    int tk[8];
    topk(g_nn_out.np, NOTE_HEAD_OUT, cfg_topk, tk);

    int   top_midi[8];
    float top_prob[8];
    int   valid_k = 0;
    for(int i=0;i<cfg_topk;i++){
        int idx=tk[i];
        if(idx >= ONSET_NOTE_VOCAB_SIZE) continue;
        float m=g_vocab[idx]; if(m<28||m>96) continue;
        top_midi[valid_k] = (int)(m+0.5f);
        top_prob[valid_k] = g_nn_out.np[idx];
        valid_k++;
    }

    float nn_conf = valid_k>0 ? top_prob[0] : 0.f;

    // 7. Guided FFT on top-K
    GuidedResult best{-1,0.f,0.f,-1.f};
    if(valid_k > 0){
        best = guided_fft(g_fft_mag, mag_len,
                          top_midi, top_prob, valid_k,
                          cfg_fft_cents, cfg_fft_min_hz);
    }

    // 8. Fused confidence (fast)
    float fused = 0.f;
    if(best.midi >= 0){
        fused = fuse_confidence_fast(nn_conf, best.harm_score, cfg_fft_weight);
    }

    // 9. Pick detector
    bool sus_blocked = (g_nn_out.su >= cfg_sus_block);
    int  cdf = (int)(cfg_sus_hold_ms / kHopMs * 0.25f); if(cdf<4)cdf=4;
    bool trig = g_pick.step(g_nn_out.pk, cfg_pick_min_p, cfg_pick_rise,
                            cfg_pick_reset, cdf, sus_blocked);

    int grace_f = (int)(cfg_grace_ms  / kHopMs); if(grace_f<1)grace_f=1;
    int min_f   = (int)(cfg_min_note_ms / kHopMs); if(min_f<1)min_f=1;

    // 10. Note state machine
    bool started = false;

    if(trig && g_voiced && best.midi>=0 && fused>=cfg_nn_on && best.refined_hz>0.f){
        g_note_active = true;
        g_note_midi   = best.midi;
        g_note_hz     = best.refined_hz;
        g_grace_left  = grace_f;
        g_min_left    = min_f;
        started       = true;

        g_cand_midi = -1;
        g_cand_count = 0;

        g_synth.set_freq(g_note_hz);
        g_synth.set_active(true);

        // printf("\nON  %-4s  %.1f Hz  nn=%.3f  fused=%.3f  db=%.1f\n",
        //        mname(g_note_midi), g_note_hz, nn_conf, fused, db);
    }

    if(g_note_active && !started){
        if(g_min_left>0) g_min_left--;

        if(g_voiced){
            g_grace_left = grace_f;

            // Track pitch: ALWAYS include locked note.
            // Include NN top-1 ONLY if very confident (so NN is guidance, not constant steering).
            int   track_midi[2]; float track_prob[2]; int tk2=0;
            track_midi[tk2]=g_note_midi; track_prob[tk2]=1.f; tk2++;

            if(valid_k>0 && top_midi[0]!=g_note_midi && top_prob[0] >= 0.60f){
                track_midi[tk2]=top_midi[0]; track_prob[tk2]=top_prob[0]; tk2++;
            }

            GuidedResult g2 = guided_fft(g_fft_mag, mag_len,
                                         track_midi, track_prob, tk2,
                                         cfg_fft_cents, cfg_fft_min_hz);

            if(g2.midi>=0 && g2.refined_hz>20.f){
                float a = cfg_freq_smooth;
                g_note_hz = a*g_note_hz + (1.f-a)*g2.refined_hz;
                g_synth.set_freq(g_note_hz);
            }

            // --- NOTE NAME SWITCHING (stable, FFT-driven) ---
            // Only switch note identity if:
            //   - tracked frequency is clearly nearer a new semitone (within cfg_switch_cents),
            //   - NN "permits" it (top1 near the proposed note and confident),
            //   - fused confidence is reasonable,
            //   - the condition persists cfg_switch_frames hops.
            {
                int cur = g_note_midi;

                float m_est   = hz_to_midi(g_note_hz);
                int   m_round = (int)lroundf(m_est);

                float cur_hz   = midi_to_hz((float)cur);
                float cur_errc = fabsf(cents_between(g_note_hz, cur_hz));

                int proposed = cur;
                if(m_round != cur){
                    float prop_hz   = midi_to_hz((float)m_round);
                    float prop_errc = fabsf(cents_between(g_note_hz, prop_hz));

                    // require meaningful improvement and small absolute error to the proposed note
                    if(prop_errc + 10.f < cur_errc && prop_errc <= cfg_switch_cents){
                        proposed = m_round;
                    }
                }

                bool nn_allows = false;
                if(valid_k>0){
                    if(fabsf((float)top_midi[0] - (float)proposed) <= 1.f &&
                       top_prob[0] >= cfg_nn_on){
                        nn_allows = true;
                    }
                }

                if(proposed != cur && nn_allows && fused >= cfg_switch_fused){
                    if(g_cand_midi != proposed){
                        g_cand_midi  = proposed;
                        g_cand_count = 1;
                    } else {
                        g_cand_count++;
                        if(g_cand_count >= cfg_switch_frames){
                            g_note_midi  = proposed;
                            g_cand_midi  = -1;
                            g_cand_count = 0;
                        }
                    }
                } else {
                    g_cand_midi  = -1;
                    g_cand_count = 0;
                }
            }

            // Allow exit only if fused < nn_off AND min time met
            if(fused < cfg_nn_off && g_min_left<=0)
                g_grace_left--;
            else
                g_grace_left = grace_f;

        } else {
            if(g_min_left<=0) g_grace_left--;
        }

        if(g_grace_left<=0 && g_min_left<=0){
            // printf("\nOFF %-4s\n", mname(g_note_midi));
            g_note_active = false;
            g_note_midi   = -1;
            g_note_hz     = 0.f;
            g_cand_midi   = -1;
            g_cand_count  = 0;
            g_synth.set_active(false);
        }
    }

    // 11. Terminal status (throttled to ~20 Hz)
    // if(++g_print_div >= 25){ // 25 hops * 2ms = 50ms
    //     g_print_div = 0;
    //     printf("\r  db=%6.1f  voiced=%d  nn=%.3f  fused=%.3f  note=%-4s  hz=%7.2f   ",
    //            db, (int)g_voiced, nn_conf, fused,
    //            g_note_active ? mname(g_note_midi) : "----", g_note_hz);
    //     fflush(stdout);
    // }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static void list_devices(){
    Pa_Initialize();
    int n=Pa_GetDeviceCount();
    printf("%-4s  %-6s  %-6s  %s\n","IDX","IN CH","OUT CH","NAME");
    for(int i=0;i<n;i++){
        const PaDeviceInfo* d=Pa_GetDeviceInfo(i);
        printf("[%2d]  %-6d  %-6d  %s\n",i,d->maxInputChannels,d->maxOutputChannels,d->name);
    }
    Pa_Terminate();
}

int main(int argc, char** argv){
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--list"||a=="-l"){ list_devices(); return 0; }
        if(a=="--in"         &&i+1<argc) cfg_in_dev      =atoi(argv[++i]);
        if(a=="--out"        &&i+1<argc) cfg_out_dev     =atoi(argv[++i]);
        if(a=="--synth-gain" &&i+1<argc) cfg_synth_gain  =atof(argv[++i]);
        if(a=="--synth-wave" &&i+1<argc){
            std::string w=argv[++i];
            cfg_wave=w=="square"?WAVE_SQUARE:w=="saw"?WAVE_SAW:WAVE_SINE;
        }
        if(a=="--nn-on"      &&i+1<argc) cfg_nn_on       =atof(argv[++i]);
        if(a=="--nn-off"     &&i+1<argc) cfg_nn_off      =atof(argv[++i]);
        if(a=="--voiced-db"  &&i+1<argc){ cfg_voiced_on  =atof(argv[++i]); cfg_voiced_off=cfg_voiced_on-10.f; }
        if(a=="--grace"      &&i+1<argc) cfg_grace_ms    =atof(argv[++i]);
        if(a=="--fft-weight" &&i+1<argc) cfg_fft_weight  =atof(argv[++i]);
        if(a=="--topk"       &&i+1<argc) cfg_topk        =atoi(argv[++i]);

        // New tuning knobs
        if(a=="--switch-cents"  &&i+1<argc) cfg_switch_cents  =atof(argv[++i]);
        if(a=="--switch-fused"  &&i+1<argc) cfg_switch_fused  =atof(argv[++i]);
        if(a=="--switch-frames" &&i+1<argc) cfg_switch_frames =atoi(argv[++i]);
    }

    build_vocab();
    build_hann_window();
    g_pick.init(kHopMs, 12.f, 2);

    g_synth.init(cfg_synth_atk, cfg_synth_rel);
    g_synth.wave = cfg_wave;
    g_synth.gain = cfg_synth_gain;
    g_synth.set_active(false);
    g_synth.set_freq(440.f);

    PaError err = Pa_Initialize();
    if(err!=paNoError){ fprintf(stderr,"Pa_Initialize: %s\n",Pa_GetErrorText(err)); return 1; }

    if(cfg_in_dev  < 0) cfg_in_dev  = Pa_GetDefaultInputDevice();
    if(cfg_out_dev < 0) cfg_out_dev = Pa_GetDefaultOutputDevice();

    const PaDeviceInfo* di  = Pa_GetDeviceInfo(cfg_in_dev);
    const PaDeviceInfo* doo = Pa_GetDeviceInfo(cfg_out_dev);
    if(!di||!doo){ fprintf(stderr,"Invalid device indices. Run --list\n"); Pa_Terminate(); return 1; }

    PaStream *s_in=nullptr, *s_out=nullptr;
    for(int inch : {2, 1}){
        PaStreamParameters ip{}, op{};
        ip.device=cfg_in_dev; ip.channelCount=inch; ip.sampleFormat=paFloat32;
        ip.suggestedLatency=di->defaultLowInputLatency;
        op.device=cfg_out_dev; op.channelCount=2; op.sampleFormat=paFloat32;
        op.suggestedLatency=doo->defaultLowOutputLatency;

        PaError e1=Pa_OpenStream(&s_in, &ip,nullptr, SR,kWinSamp,paClipOff,pa_in_cb, nullptr);
        PaError e2=Pa_OpenStream(&s_out,nullptr,&op, SR,kWinSamp,paClipOff,pa_out_cb,nullptr);
        if(e1==paNoError && e2==paNoError){
            g_in_channels=inch;
            printf("Input  [%d] %s (%d ch)\n", cfg_in_dev,  di->name,  inch);
            printf("Output [%d] %s\n",          cfg_out_dev, doo->name);
            break;
        }
        if(s_in) { Pa_CloseStream(s_in);  s_in=nullptr;  }
        if(s_out){ Pa_CloseStream(s_out); s_out=nullptr; }
    }
    if(!s_in || !s_out){
        fprintf(stderr,"Could not open audio. Try --list and --in / --out\n");
        Pa_Terminate(); return 1;
    }

    Pa_StartStream(s_in);
    Pa_StartStream(s_out);

    // printf("nn_on=%.2f  nn_off=%.2f  voiced_db=%.0f  grace=%.0fms  fft_weight=%.2f  wave=%s  gain=%.2f\n",
    //        cfg_nn_on, cfg_nn_off, cfg_voiced_on, cfg_grace_ms, cfg_fft_weight,
    //        cfg_wave==WAVE_SQUARE?"square":cfg_wave==WAVE_SAW?"saw":"sine", cfg_synth_gain);

    // printf("switch: cents=%.0f  fused=%.2f  frames=%d (%.1fms)\n",
    //        cfg_switch_cents, cfg_switch_fused, cfg_switch_frames, cfg_switch_frames*kHopMs);

    // printf("Ctrl+C to quit\n\n");

    signal(SIGINT, on_sigint);

    while(g_running){
        tick();
        Pa_Sleep((int)kHopMs);
    }

    printf("\nDone.\n");
    Pa_StopStream(s_in);  Pa_CloseStream(s_in);
    Pa_StopStream(s_out); Pa_CloseStream(s_out);
    Pa_Terminate();
    return 0;
}