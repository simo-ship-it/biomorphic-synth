/*
 * BIOMORPHIC // MEDIAPIPE & C++ HAND-CONTROLLED SYNTHESIZER
 *
 * Native C++20 / Objective-C++ Application:
 * - Section 01: 4/4 16-Step Microsound Sequencer with ONLY the first sound ("01 · Ikeda Sub Pulse")
 *   clocked sample-accurately inside the CoreAudio real-time thread.
 * - Section 02: Continuous 2D XY Synth Pad controlled via live 21-Landmark Hand Tracking
 *   (MediaPipe Hands 21-joint topology: X = Frequency 55–880 Hz, Y = Intensity 0–100%),
 *   with live camera feed + 21-point hand skeleton overlay (plus mouse fallback).
 * - Right Panel: TouchDesigner-style 3D Audio-Reactive Point Cloud & CHOP Spectrum Telemetry.
 */

#include <AudioToolbox/AudioToolbox.h>
#include <SDL.h>

#include "mediapipe_hand_tracker.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

static constexpr int SAMPLE_RATE = 44100;
static constexpr int NUM_AUDIO_BUFFERS = 3;
static constexpr int AUDIO_BUFFER_FRAMES = 512;
static constexpr double TWO_PI = 6.28318530717958647692;

// =====================================================================
// BUILT-IN 5x7 BITMAP FONT FOR CRISP ZERO-DEPENDENCY UI TEXT
// =====================================================================

static const uint8_t* getGlyph5x7(char c) {
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    static const uint8_t SPACE[7] = {0,0,0,0,0,0,0};
    static const uint8_t A[7] = {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11};
    static const uint8_t B[7] = {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E};
    static const uint8_t C[7] = {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E};
    static const uint8_t D[7] = {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C};
    static const uint8_t E[7] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F};
    static const uint8_t F[7] = {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10};
    static const uint8_t G[7] = {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F};
    static const uint8_t H[7] = {0x11,0x11,0x11,0x1F,0x11,0x11,0x11};
    static const uint8_t I[7] = {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E};
    static const uint8_t J[7] = {0x07,0x02,0x02,0x02,0x12,0x12,0x0C};
    static const uint8_t K[7] = {0x11,0x12,0x14,0x18,0x14,0x12,0x11};
    static const uint8_t L[7] = {0x10,0x10,0x10,0x10,0x10,0x10,0x1F};
    static const uint8_t M[7] = {0x11,0x1B,0x15,0x15,0x11,0x11,0x11};
    static const uint8_t N[7] = {0x11,0x19,0x15,0x13,0x11,0x11,0x11};
    static const uint8_t O[7] = {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E};
    static const uint8_t P[7] = {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10};
    static const uint8_t Q[7] = {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D};
    static const uint8_t R[7] = {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11};
    static const uint8_t S[7] = {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E};
    static const uint8_t T[7] = {0x1F,0x04,0x04,0x04,0x04,0x04,0x04};
    static const uint8_t U[7] = {0x11,0x11,0x11,0x11,0x11,0x11,0x0E};
    static const uint8_t V[7] = {0x11,0x11,0x11,0x11,0x11,0x0A,0x04};
    static const uint8_t W[7] = {0x11,0x11,0x11,0x15,0x15,0x1B,0x11};
    static const uint8_t X[7] = {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11};
    static const uint8_t Y[7] = {0x11,0x11,0x0A,0x04,0x04,0x04,0x04};
    static const uint8_t Z[7] = {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F};
    static const uint8_t D0[7] = {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E};
    static const uint8_t D1[7] = {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E};
    static const uint8_t D2[7] = {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F};
    static const uint8_t D3[7] = {0x1F,0x02,0x04,0x02,0x01,0x11,0x0E};
    static const uint8_t D4[7] = {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02};
    static const uint8_t D5[7] = {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E};
    static const uint8_t D6[7] = {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E};
    static const uint8_t D7[7] = {0x1F,0x01,0x02,0x04,0x08,0x08,0x08};
    static const uint8_t D8[7] = {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E};
    static const uint8_t D9[7] = {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C};
    static const uint8_t COLON[7] = {0x00,0x04,0x04,0x00,0x04,0x04,0x00};
    static const uint8_t DOT[7] = {0x00,0x00,0x00,0x00,0x00,0x06,0x06};
    static const uint8_t DASH[7] = {0x00,0x00,0x00,0x1F,0x00,0x00,0x00};
    static const uint8_t PLUS[7] = {0x00,0x04,0x04,0x1F,0x04,0x04,0x00};
    static const uint8_t SLASH[7] = {0x01,0x02,0x02,0x04,0x08,0x08,0x10};
    static const uint8_t PCT[7] = {0x19,0x19,0x02,0x04,0x08,0x13,0x13};
    static const uint8_t LBRK[7] = {0x0E,0x08,0x08,0x08,0x08,0x08,0x0E};
    static const uint8_t RBRK[7] = {0x0E,0x02,0x02,0x02,0x02,0x02,0x0E};
    static const uint8_t PIPE[7] = {0x04,0x04,0x04,0x04,0x04,0x04,0x04};

    switch (c) {
        case 'A': return A; case 'B': return B; case 'C': return C; case 'D': return D;
        case 'E': return E; case 'F': return F; case 'G': return G; case 'H': return H;
        case 'I': return I; case 'J': return J; case 'K': return K; case 'L': return L;
        case 'M': return M; case 'N': return N; case 'O': return O; case 'P': return P;
        case 'Q': return Q; case 'R': return R; case 'S': return S; case 'T': return T;
        case 'U': return U; case 'V': return V; case 'W': return W; case 'X': return X;
        case 'Y': return Y; case 'Z': return Z;
        case '0': return D0; case '1': return D1; case '2': return D2; case '3': return D3;
        case '4': return D4; case '5': return D5; case '6': return D6; case '7': return D7;
        case '8': return D8; case '9': return D9;
        case ':': return COLON; case '.': return DOT; case '-': return DASH;
        case '+': return PLUS; case '/': return SLASH; case '%': return PCT;
        case '[': case '(': return LBRK; case ']': case ')': return RBRK;
        case '|': return PIPE;
        default: return SPACE;
    }
}

static void drawText(SDL_Renderer *ren, int x, int y, const std::string &text,
                     SDL_Color col, int scale = 2) {
    SDL_SetRenderDrawColor(ren, col.r, col.g, col.b, col.a);
    int cx = x;
    for (char ch : text) {
        const uint8_t *g = getGlyph5x7(ch);
        for (int row = 0; row < 7; ++row) {
            uint8_t bits = g[row];
            for (int col_idx = 0; col_idx < 5; ++col_idx) {
                if (bits & (1 << (4 - col_idx))) {
                    SDL_Rect r = {cx + col_idx * scale, y + row * scale, scale, scale};
                    SDL_RenderFillRect(ren, &r);
                }
            }
        }
        cx += 6 * scale;
    }
}

// =====================================================================
// REAL-TIME COREAUDIO ENGINE (SAMPLE-ACCURATE 4/4 IKEDA SUB + XY SYNTH)
// =====================================================================

struct AudioSynthState {
    std::mutex mtx;

    // 1. Pre-synthesized "01 · Ikeda Sub Pulse" sample in RAM
    std::vector<float> sub_pulse_sample;
    double sub_voice_pos = -1.0; // < 0 means inactive
    float sub_voice_gain = 0.9f;

    // 2. Sample-accurate 4/4 16-Step Sequencer state
    bool is_playing = true;
    bool sub_muted = false;
    int bpm = 108;
    std::array<int, 16> grid = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    int current_step = 0;
    double step_sample_counter = 0.0;

    // 3. Continuous 2D XY Hand Synthesizer state
    float target_freq = 220.0f;
    float current_freq = 220.0f;
    float target_amp = 0.0f;
    float current_amp = 0.0f;
    int synth_mode = 0; // 0 = Buchla Wavefold, 1 = Harmonic Drone, 2 = Crystal FM
    double phase1 = 0.0;
    double phase2 = 0.0;
    double phase3 = 0.0;
    double lfo_phase = 0.0;

    // 4. Audio Analysis CHOP Envelopes for 3D Visualizer
    std::atomic<float> chop_sub{0.0f};
    std::atomic<float> chop_synth{0.0f};
    std::atomic<float> chop_micro{0.0f};

    AudioQueueRef queue = nullptr;
    AudioQueueBufferRef buffers[NUM_AUDIO_BUFFERS];
    bool running = false;

    void generateIkedaSubPulse() {
        const float duration = 0.28f;
        int n = (int)(SAMPLE_RATE * duration);
        sub_pulse_sample.resize(n);
        double ph = 0.0;
        float peak = 0.001f;

        for (int i = 0; i < n; ++i) {
            double t = (double)i / SAMPLE_RATE;
            double freq = 46.0 + 110.0 * std::exp(-t * 48.0);
            ph += TWO_PI * freq / SAMPLE_RATE;
            double body = std::sin(ph) + 0.18 * std::sin(2.0 * ph);
            double env = std::exp(-t * 11.5);
            // Razor-sharp 2.5ms microsound transient
            double click = 0.9 * std::sin(TWO_PI * 2400.0 * t) * std::exp(-t * 900.0);
            float s = (float)(body * env + click);
            sub_pulse_sample[i] = s;
            peak = std::max(peak, std::fabs(s));
        }
        for (float &s : sub_pulse_sample) {
            s = (s / peak) * 0.90f;
        }
    }

    void triggerSubPulse(float gain = 0.95f) {
        sub_voice_pos = 0.0;
        sub_voice_gain = gain;
        chop_sub.store(std::min(2.2f, chop_sub.load() + 1.15f));
        chop_micro.store(std::min(2.0f, chop_micro.load() + 0.45f));
    }
};

static void coreAudioCallback(void *user_data, AudioQueueRef queue, AudioQueueBufferRef buf) {
    AudioSynthState *st = static_cast<AudioSynthState *>(user_data);
    int16_t *out = static_cast<int16_t *>(buf->mAudioData);

    std::lock_guard<std::mutex> lock(st->mtx);
    double samples_per_step = (double)SAMPLE_RATE * 60.0 / (std::max(40, st->bpm) * 4.0);

    for (int i = 0; i < AUDIO_BUFFER_FRAMES; ++i) {
        // 1. Advance sample-accurate 4/4 16-step sequencer clock
        if (st->is_playing) {
            if (st->step_sample_counter <= 0.0) {
                int s = st->current_step;
                if (st->grid[s] && !st->sub_muted) {
                    bool accent = (s % 4 == 0);
                    st->triggerSubPulse(accent ? 0.96f : 0.80f);
                }
                st->current_step = (s + 1) % 16;
                st->step_sample_counter += samples_per_step;
            }
            st->step_sample_counter -= 1.0;
        }

        float mix_l = 0.0f;
        float mix_r = 0.0f;

        // 2. Render "01 · Ikeda Sub Pulse" voice
        if (st->sub_voice_pos >= 0.0) {
            int idx = (int)st->sub_voice_pos;
            if (idx < (int)st->sub_pulse_sample.size() - 1) {
                float v = st->sub_pulse_sample[idx] * st->sub_voice_gain;
                mix_l += v;
                mix_r += v;
                st->sub_voice_pos += 1.0;
            } else {
                st->sub_voice_pos = -1.0;
            }
        }

        // 3. Render Continuous 2D XY Hand Synthesizer (per-sample click-free smoothing)
        st->current_freq += 0.0035f * (st->target_freq - st->current_freq);
        st->current_amp  += 0.0028f * (st->target_amp  - st->current_amp);

        if (st->current_amp > 0.0002f) {
            float f = st->current_freq;
            float amp = st->current_amp;

            st->lfo_phase += TWO_PI * 0.45 / SAMPLE_RATE;
            if (st->lfo_phase >= TWO_PI) st->lfo_phase -= TWO_PI;
            float lfo = (float)std::sin(st->lfo_phase);

            float syn_l = 0.0f;
            float syn_r = 0.0f;

            if (st->synth_mode == 0) {
                // Mode 0: Biomorphic Buchla Wavefolder
                st->phase1 += TWO_PI * f / SAMPLE_RATE;
                st->phase2 += TWO_PI * (f * 0.5) / SAMPLE_RATE;
                st->phase3 += TWO_PI * (f * 1.502) / SAMPLE_RATE;
                if (st->phase1 >= TWO_PI) st->phase1 -= TWO_PI;
                if (st->phase2 >= TWO_PI) st->phase2 -= TWO_PI;
                if (st->phase3 >= TWO_PI) st->phase3 -= TWO_PI;

                float fold = 1.0f + amp * (2.8f + 0.4f * lfo);
                float raw = (float)std::sin(st->phase1) + 0.35f * amp * (float)std::sin(st->phase3);
                float folded = (float)std::sin(raw * fold);
                float sub = 0.45f * (float)std::sin(st->phase2);

                syn_l = (folded * 0.65f + sub) * amp * 0.55f;
                syn_r = (folded * 0.65f + sub * 0.95f) * amp * 0.55f;
            } else if (st->synth_mode == 1) {
                // Mode 1: Organic Harmonic Drone
                st->phase1 += TWO_PI * (f * (1.0 + 0.003 * lfo)) / SAMPLE_RATE;
                st->phase2 += TWO_PI * (f * (1.0 - 0.003 * lfo)) / SAMPLE_RATE;
                st->phase3 += TWO_PI * (f * 1.4983) / SAMPLE_RATE;
                if (st->phase1 >= TWO_PI) st->phase1 -= TWO_PI;
                if (st->phase2 >= TWO_PI) st->phase2 -= TWO_PI;
                if (st->phase3 >= TWO_PI) st->phase3 -= TWO_PI;

                float osc1 = (float)std::sin(st->phase1) + 0.3f * amp * (float)std::sin(st->phase1 * 2.0);
                float osc2 = (float)std::sin(st->phase2) + 0.3f * amp * (float)std::sin(st->phase2 * 2.0);
                float fifth = (0.15f + 0.35f * amp) * (float)std::sin(st->phase3);

                syn_l = (osc1 * 0.6f + fifth * 0.4f) * amp * 0.55f;
                syn_r = (osc2 * 0.6f + fifth * 0.4f) * amp * 0.55f;
            } else {
                // Mode 2: Post-Nature Crystal FM
                st->phase1 += TWO_PI * f / SAMPLE_RATE;
                st->phase2 += TWO_PI * (f * 1.41421356) / SAMPLE_RATE;
                st->phase3 += TWO_PI * (f * 2.002) / SAMPLE_RATE;
                if (st->phase1 >= TWO_PI) st->phase1 -= TWO_PI;
                if (st->phase2 >= TWO_PI) st->phase2 -= TWO_PI;
                if (st->phase3 >= TWO_PI) st->phase3 -= TWO_PI;

                float mod_idx = 0.2f + amp * amp * 3.8f;
                float mod = (float)std::sin(st->phase2);
                float car_l = (float)std::sin(st->phase1 + mod_idx * mod);
                float car_r = (float)std::sin(st->phase1 + mod_idx * mod * 0.92f + 0.25f * lfo);
                float oct = 0.28f * amp * (float)std::sin(st->phase3);

                syn_l = (car_l * 0.65f + oct) * amp * 0.50f;
                syn_r = (car_r * 0.65f + oct) * amp * 0.50f;
            }

            mix_l += syn_l;
            mix_r += syn_r;
        }

        // Soft clip stereo output
        auto softClip = [](float x) -> int16_t {
            if (x > 1.0f) x = 1.0f - 1.0f / (1.0f + x * 2.0f);
            else if (x < -1.0f) x = -1.0f + 1.0f / (1.0f - x * 2.0f);
            else x *= 0.80f;
            int32_t p = (int32_t)(x * 32767.0f);
            return (int16_t)std::clamp(p, -32768, 32767);
        };

        out[i * 2]     = softClip(mix_l);
        out[i * 2 + 1] = softClip(mix_r);
    }

    buf->mAudioDataByteSize = AUDIO_BUFFER_FRAMES * 4;
    if (st->running) {
        AudioQueueEnqueueBuffer(queue, buf, 0, nullptr);
    }
}

static bool initCoreAudio(AudioSynthState &st) {
    st.generateIkedaSubPulse();

    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate = SAMPLE_RATE;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    fmt.mBitsPerChannel = 16;
    fmt.mChannelsPerFrame = 2;
    fmt.mBytesPerFrame = 4;
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerPacket = 4;

    if (AudioQueueNewOutput(&fmt, coreAudioCallback, &st, nullptr, nullptr, 0, &st.queue) != noErr) {
        return false;
    }

    st.running = true;
    int buf_bytes = AUDIO_BUFFER_FRAMES * 4;
    for (int i = 0; i < NUM_AUDIO_BUFFERS; ++i) {
        AudioQueueAllocateBuffer(st.queue, buf_bytes, &st.buffers[i]);
        std::memset(st.buffers[i]->mAudioData, 0, buf_bytes);
        st.buffers[i]->mAudioDataByteSize = buf_bytes;
        AudioQueueEnqueueBuffer(st.queue, st.buffers[i], 0, nullptr);
    }
    return AudioQueueStart(st.queue, nullptr) == noErr;
}

static void shutdownCoreAudio(AudioSynthState &st) {
    if (!st.running) return;
    st.running = false;
    if (st.queue) {
        AudioQueueStop(st.queue, true);
        AudioQueueDispose(st.queue, true);
        st.queue = nullptr;
    }
}

// =====================================================================
// HELPER DRAWING PRIMITIVES
// =====================================================================

static void fillRect(SDL_Renderer *ren, SDL_Rect r, SDL_Color fill, SDL_Color border) {
    SDL_SetRenderDrawColor(ren, fill.r, fill.g, fill.b, fill.a);
    SDL_RenderFillRect(ren, &r);
    SDL_SetRenderDrawColor(ren, border.r, border.g, border.b, border.a);
    SDL_RenderDrawRect(ren, &r);
}

static void drawCircle(SDL_Renderer *ren, int cx, int cy, int radius, SDL_Color col, bool filled = false) {
    SDL_SetRenderDrawColor(ren, col.r, col.g, col.b, col.a);
    if (filled) {
        for (int dy = -radius; dy <= radius; ++dy) {
            int dx = (int)std::sqrt(std::max(0, radius * radius - dy * dy));
            SDL_RenderDrawLine(ren, cx - dx, cy + dy, cx + dx, cy + dy);
        }
    } else {
        const int segs = 36;
        for (int i = 0; i < segs; ++i) {
            double a0 = TWO_PI * i / segs;
            double a1 = TWO_PI * (i + 1) / segs;
            int x0 = cx + (int)(std::cos(a0) * radius);
            int y0 = cy + (int)(std::sin(a0) * radius);
            int x1 = cx + (int)(std::cos(a1) * radius);
            int y1 = cy + (int)(std::sin(a1) * radius);
            SDL_RenderDrawLine(ren, x0, y0, x1, y1);
        }
    }
}

static bool pointInRect(int x, int y, const SDL_Rect &r) {
    return x >= r.x && x < (r.x + r.w) && y >= r.y && y < (r.y + r.h);
}

// =====================================================================
// MAIN APPLICATION
// =====================================================================

struct VisNode3D {
    float bx, by, bz, seed;
};

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    std::srand((unsigned)std::time(nullptr));

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    const int WIN_W = 1220;
    const int WIN_H = 800;
    SDL_Window *window = SDL_CreateWindow(
        "BIOMORPHIC // MediaPipe & C++ Hand-Controlled Synth (4/4)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIN_W, WIN_H,
        SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI
    );
    if (!window) return 1;

    SDL_Renderer *ren = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (!ren) {
        ren = SDL_CreateRenderer(window, -1, 0);
    }
    SDL_RenderSetLogicalSize(ren, WIN_W, WIN_H);

    // Initialize Real-Time CoreAudio Engine
    AudioSynthState audio;
    initCoreAudio(audio);

    // Initialize MediaPipe 21-Landmark Hand Tracker (Webcam)
    mediapipe_hands::HandTracker handTracker;
    handTracker.start();

    SDL_Texture *camTex = SDL_CreateTexture(
        ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, 320, 240
    );

    // 3D Biomorphic Point Cloud Nodes
    std::vector<VisNode3D> visNodes(28);
    const float phi = 3.14159265f * (3.0f - std::sqrt(5.0f));
    for (int i = 0; i < 28; ++i) {
        float y = 1.0f - (i / 27.0f) * 2.0f;
        float rad = std::sqrt(std::max(0.0f, 1.0f - y * y));
        float th = phi * i;
        visNodes[i] = {std::cos(th) * rad, y, std::sin(th) * rad, (float)(i * 1.37f)};
    }
    float visAngle = 0.0f;

    // UI & Interaction State
    bool running = true;
    bool xyHold = false;
    bool bioSonify = false;
    bool mouseDraggingXY = false;
    float normX = 0.5f;
    float normY = 0.65f;
    float currentHz = 220.0f;
    float currentIntensity = 0.0f;
    std::vector<std::pair<float, float>> xyTrail;

    // Palette
    const SDL_Color COL_BG      = {11, 13, 18, 255};
    const SDL_Color COL_PANEL   = {18, 22, 31, 255};
    const SDL_Color COL_CELL    = {27, 33, 48, 255};
    const SDL_Color COL_BEAT    = {36, 44, 64, 255};
    const SDL_Color COL_BORDER  = {42, 51, 72, 255};
    const SDL_Color COL_TEXT    = {226, 232, 240, 255};
    const SDL_Color COL_MUTED   = {124, 139, 161, 255};
    const SDL_Color COL_CYAN    = {0, 245, 212, 255};
    const SDL_Color COL_MAGENTA = {241, 91, 181, 255};
    const SDL_Color COL_PURPLE  = {155, 93, 229, 255};
    const SDL_Color COL_GREEN   = {56, 176, 0, 255};
    const SDL_Color COL_YELLOW  = {254, 228, 64, 255};
    const SDL_Color COL_DARK    = {8, 10, 15, 255};

    // Layout Rectangles
    const SDL_Rect headerRect = {14, 12, WIN_W - 28, 58};
    const SDL_Rect playBtnRect = {WIN_W - 630, 23, 145, 36};
    const SDL_Rect holdBtnRect = {WIN_W - 475, 23, 145, 36};
    const SDL_Rect bioBtnRect  = {WIN_W - 320, 23, 150, 36};
    const SDL_Rect bpmMinusRect = {WIN_W - 155, 23, 34, 36};
    const SDL_Rect bpmPlusRect  = {WIN_W - 55,  23, 34, 36};

    const SDL_Rect seqPanelRect = {14, 80, 816, 142};
    const SDL_Rect presetRects[4] = {
        {430, 88, 86, 24}, // 4/4 Cooper
        {522, 88, 86, 24}, // Ikeda
        {614, 88, 86, 24}, // Mutate
        {706, 88, 72, 24}  // Clear
    };
    const SDL_Rect trackNameRect = {26, 148, 165, 52};
    const SDL_Rect muteBtnRect   = {195, 148, 28, 52};
    const SDL_Rect randBtnRect   = {227, 148, 28, 52};

    // Square/Rectangular 2D Hand XY Synth Pad (Section 02)
    const SDL_Rect xyPanelRect = {14, 232, 816, 554};
    const SDL_Rect xyPadRect   = {28, 276, 788, 496};
    const SDL_Rect modeRects[3] = {
        {470, 242, 112, 24}, // Buchla
        {588, 242, 112, 24}, // Drone
        {706, 242, 112, 24}  // Crystal FM
    };

    // Right Column: TouchDesigner 3D Visualizer
    const SDL_Rect visPanelRect = {840, 80, 366, 706};

    uint64_t startTicks = SDL_GetPerformanceCounter();

    while (running) {
        double nowSec = (double)(SDL_GetPerformanceCounter() - startTicks) /
                        (double)SDL_GetPerformanceFrequency();

        // -------------------------------------------------------------
        // 1. POLL INPUT EVENTS (Mouse + Keyboard)
        // -------------------------------------------------------------
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                running = false;
            } else if (ev.type == SDL_KEYDOWN) {
                if (ev.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                } else if (ev.key.keysym.sym == SDLK_SPACE) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    audio.is_playing = !audio.is_playing;
                    audio.current_step = 0;
                    audio.step_sample_counter = 0.0;
                }
            } else if (ev.type == SDL_MOUSEBUTTONDOWN) {
                int mx = ev.button.x;
                int my = ev.button.y;

                if (ev.button.button == SDL_BUTTON_RIGHT && pointInRect(mx, my, xyPadRect)) {
                    xyHold = !xyHold;
                } else if (pointInRect(mx, my, playBtnRect)) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    audio.is_playing = !audio.is_playing;
                    audio.current_step = 0;
                    audio.step_sample_counter = 0.0;
                } else if (pointInRect(mx, my, holdBtnRect)) {
                    xyHold = !xyHold;
                } else if (pointInRect(mx, my, bioBtnRect)) {
                    bioSonify = !bioSonify;
                } else if (pointInRect(mx, my, bpmMinusRect)) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    audio.bpm = std::max(60, audio.bpm - 4);
                } else if (pointInRect(mx, my, bpmPlusRect)) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    audio.bpm = std::min(165, audio.bpm + 4);
                } else if (pointInRect(mx, my, presetRects[0])) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    audio.grid = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
                } else if (pointInRect(mx, my, presetRects[1])) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    audio.grid = {1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1};
                } else if (pointInRect(mx, my, presetRects[2])) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    for (int s = 0; s < 16; ++s) audio.grid[s] = (std::rand() % 100 < 35) ? 1 : 0;
                    audio.grid[0] = 1;
                    audio.grid[8] = 1;
                } else if (pointInRect(mx, my, presetRects[3])) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    audio.grid.fill(0);
                } else if (pointInRect(mx, my, trackNameRect)) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    audio.triggerSubPulse(0.95f);
                } else if (pointInRect(mx, my, muteBtnRect)) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    audio.sub_muted = !audio.sub_muted;
                } else if (pointInRect(mx, my, randBtnRect)) {
                    std::lock_guard<std::mutex> lock(audio.mtx);
                    for (int s = 0; s < 16; ++s) audio.grid[s] = (std::rand() % 100 < 35) ? 1 : 0;
                    audio.grid[0] = 1;
                } else {
                    // Check 16-step cells of the single top drum track
                    int gridX0 = 262;
                    int gridW = 554;
                    float stepW = gridW / 16.0f;
                    if (mx >= gridX0 && mx < gridX0 + gridW && my >= 148 && my <= 200) {
                        int s = (int)((mx - gridX0) / stepW);
                        if (s >= 0 && s < 16) {
                            std::lock_guard<std::mutex> lock(audio.mtx);
                            audio.grid[s] = audio.grid[s] ? 0 : 1;
                            if (audio.grid[s]) audio.triggerSubPulse(0.90f);
                        }
                    }
                    // Check Synth Engine Mode buttons
                    for (int m = 0; m < 3; ++m) {
                        if (pointInRect(mx, my, modeRects[m])) {
                            std::lock_guard<std::mutex> lock(audio.mtx);
                            audio.synth_mode = m;
                        }
                    }
                    // Check XY Pad Mouse Drag fallback
                    if (pointInRect(mx, my, xyPadRect)) {
                        mouseDraggingXY = true;
                        normX = std::clamp((float)(mx - xyPadRect.x) / xyPadRect.w, 0.0f, 1.0f);
                        normY = std::clamp(1.0f - (float)(my - xyPadRect.y) / xyPadRect.h, 0.0f, 1.0f);
                    }
                }
            } else if (ev.type == SDL_MOUSEMOTION && mouseDraggingXY) {
                int mx = ev.motion.x;
                int my = ev.motion.y;
                normX = std::clamp((float)(mx - xyPadRect.x) / xyPadRect.w, 0.0f, 1.0f);
                normY = std::clamp(1.0f - (float)(my - xyPadRect.y) / xyPadRect.h, 0.0f, 1.0f);
            } else if (ev.type == SDL_MOUSEBUTTONUP) {
                mouseDraggingXY = false;
            }
        }

        // -------------------------------------------------------------
        // 2. READ MEDIAPIPE 21-LANDMARK HAND TRACKER & UPDATE SYNTH
        // -------------------------------------------------------------
        mediapipe_hands::HandTrackingFrame hf = handTracker.getLatestFrame();
        if (hf.camera_ready && !hf.cam_pixels.empty()) {
            SDL_UpdateTexture(camTex, nullptr, hf.cam_pixels.data(), hf.cam_w * sizeof(uint32_t));
        }

        bool synthActive = false;
        if (hf.hand_detected) {
            // Hand position inside the square pad controls X (Frequency) and Y (Intensity)
            normX = hf.norm_x;
            normY = hf.norm_y;
            synthActive = true;
        } else if (mouseDraggingXY) {
            synthActive = true;
        } else if (bioSonify) {
            normX = std::clamp(0.5f + 0.36f * (float)std::sin(nowSec * 0.55) + 0.12f * (float)std::cos(nowSec * 1.31), 0.02f, 0.98f);
            normY = std::clamp(0.55f + 0.30f * (float)std::cos(nowSec * 0.42) + 0.12f * (float)std::sin(nowSec * 1.73), 0.10f, 0.95f);
            synthActive = true;
        } else if (xyHold) {
            synthActive = true;
        }

        // Exponential 4-octave frequency curve: 55 Hz (left) -> 880 Hz (right)
        currentHz = 55.0f * std::pow(2.0f, 4.0f * normX);
        currentIntensity = synthActive ? normY : 0.0f;

        {
            std::lock_guard<std::mutex> lock(audio.mtx);
            audio.target_freq = currentHz;
            audio.target_amp = currentIntensity;
        }

        if (synthActive) {
            xyTrail.push_back({normX, normY});
            if (xyTrail.size() > 28) xyTrail.erase(xyTrail.begin());
        }

        // Update CHOP envelopes for the 3D visualizer
        float subE = audio.chop_sub.load() * 0.92f;
        float microE = audio.chop_micro.load() * 0.86f;
        float synthE = currentIntensity * 1.35f;
        audio.chop_sub.store(subE);
        audio.chop_micro.store(microE);
        audio.chop_synth.store(synthE);

        // -------------------------------------------------------------
        // 3. RENDER UI
        // -------------------------------------------------------------
        SDL_SetRenderDrawColor(ren, COL_BG.r, COL_BG.g, COL_BG.b, 255);
        SDL_RenderClear(ren);

        // --- HEADER BAR ---
        fillRect(ren, headerRect, COL_PANEL, COL_BORDER);
        drawText(ren, 28, 22, "BIOMORPHIC // MEDIAPIPE & C++ HAND SYNTH", COL_CYAN, 2);
        drawText(ren, 28, 44, "01: IKEDA 4/4 SUB PULSE  |  02: 21-LANDMARK HAND XY CONTINUOUS SYNTH", COL_MUTED, 1);

        bool isPlayingCopy = false;
        int bpmCopy = 108;
        int currentStepCopy = 0;
        bool subMutedCopy = false;
        int synthModeCopy = 0;
        std::array<int, 16> gridCopy;
        {
            std::lock_guard<std::mutex> lock(audio.mtx);
            isPlayingCopy = audio.is_playing;
            bpmCopy = audio.bpm;
            currentStepCopy = audio.current_step;
            subMutedCopy = audio.sub_muted;
            synthModeCopy = audio.synth_mode;
            gridCopy = audio.grid;
        }

        fillRect(ren, playBtnRect, isPlayingCopy ? COL_MAGENTA : COL_CYAN, COL_BORDER);
        drawText(ren, playBtnRect.x + 14, playBtnRect.y + 11,
                 isPlayingCopy ? "STOP 4/4 [SPC]" : "PLAY 4/4 [SPC]",
                 isPlayingCopy ? COL_TEXT : COL_DARK, 1);

        fillRect(ren, holdBtnRect, xyHold ? COL_PURPLE : COL_CELL, COL_BORDER);
        drawText(ren, holdBtnRect.x + 18, holdBtnRect.y + 11,
                 xyHold ? "XY HOLD: ON" : "XY HOLD: OFF", COL_TEXT, 1);

        fillRect(ren, bioBtnRect, bioSonify ? COL_GREEN : COL_CELL, COL_BORDER);
        drawText(ren, bioBtnRect.x + 14, bioBtnRect.y + 11,
                 bioSonify ? "BIO-SONIFY: ON" : "BIO-SONIFY: OFF",
                 bioSonify ? COL_DARK : COL_TEXT, 1);

        fillRect(ren, bpmMinusRect, COL_CELL, COL_BORDER);
        drawText(ren, bpmMinusRect.x + 11, bpmMinusRect.y + 11, "-", COL_CYAN, 2);
        char bpmBuf[32];
        std::snprintf(bpmBuf, sizeof(bpmBuf), "BPM %d", bpmCopy);
        drawText(ren, bpmMinusRect.x + 40, bpmMinusRect.y + 14, bpmBuf, COL_TEXT, 1);
        fillRect(ren, bpmPlusRect, COL_CELL, COL_BORDER);
        drawText(ren, bpmPlusRect.x + 11, bpmPlusRect.y + 11, "+", COL_CYAN, 2);

        // --- SECTION 01: SINGLE-TRACK 4/4 SEQUENCER (ONLY FIRST SOUND) ---
        fillRect(ren, seqPanelRect, COL_PANEL, COL_BORDER);
        drawText(ren, 26, 94, "01 // 4/4 DRUM MACHINE (SINGLE TOP VOICE)", COL_TEXT, 2);

        const char *presetLabels[4] = {"COOPER 4/4", "IKEDA SYNC", "MUTATE", "CLEAR"};
        for (int p = 0; p < 4; ++p) {
            fillRect(ren, presetRects[p], COL_CELL, COL_BORDER);
            drawText(ren, presetRects[p].x + 10, presetRects[p].y + 8, presetLabels[p], COL_TEXT, 1);
        }

        // Track Name Box ("01 · IKEDA SUB PULSE")
        fillRect(ren, trackNameRect, COL_CELL, subMutedCopy ? COL_BORDER : COL_CYAN);
        drawText(ren, trackNameRect.x + 10, trackNameRect.y + 19, "01 . IKEDA SUB PULSE",
                 subMutedCopy ? COL_MUTED : COL_CYAN, 1);

        fillRect(ren, muteBtnRect, subMutedCopy ? COL_MAGENTA : COL_CELL, COL_BORDER);
        drawText(ren, muteBtnRect.x + 9, muteBtnRect.y + 19, "M",
                 subMutedCopy ? COL_DARK : COL_MUTED, 1);

        fillRect(ren, randBtnRect, COL_CELL, COL_BORDER);
        drawText(ren, randBtnRect.x + 9, randBtnRect.y + 19, "R", COL_TEXT, 1);

        // 16 Steps (4/4 Grid)
        int gridX0 = 262;
        int gridW = 554;
        float stepW = gridW / 16.0f;
        for (int s = 0; s < 16; ++s) {
            int sx = gridX0 + (int)(s * stepW) + 2;
            int sw = (int)stepW - 4;
            bool isDownbeat = (s % 4 == 0);
            bool isPlayhead = isPlayingCopy && (s == currentStepCopy);
            bool active = (gridCopy[s] != 0);

            char beatLbl[8];
            std::snprintf(beatLbl, sizeof(beatLbl), "%d.%d", (s / 4) + 1, (s % 4) + 1);
            drawText(ren, sx + 6, 132, beatLbl,
                     isPlayhead ? COL_CYAN : (isDownbeat ? COL_TEXT : COL_MUTED), 1);

            SDL_Rect cellRect = {sx, 148, sw, 52};
            SDL_Color fillCol = active ? (subMutedCopy ? COL_MUTED : COL_CYAN)
                                       : (isDownbeat ? COL_BEAT : COL_CELL);
            SDL_Color bordCol = isPlayhead ? COL_TEXT : (active ? COL_CYAN : COL_BORDER);
            fillRect(ren, cellRect, fillCol, bordCol);

            if (active && !subMutedCopy) {
                drawCircle(ren, sx + sw / 2, 174, 4, COL_DARK, true);
            }
        }

        // --- SECTION 02: MEDIAPIPE 21-LANDMARK HAND XY CONTINUOUS SYNTH PAD ---
        fillRect(ren, xyPanelRect, COL_PANEL, COL_BORDER);
        drawText(ren, 26, 246, "02 // HAND XY SYNTH PAD (LEFT-RIGHT: FREQ  |  HEIGHT: INTENSITY)", COL_TEXT, 2);

        const char *modeNames[3] = {"BUCHLA FOLD", "GLASS DRONE", "CRYSTAL FM"};
        const SDL_Color modeColors[3] = {COL_CYAN, COL_PURPLE, COL_YELLOW};
        for (int m = 0; m < 3; ++m) {
            bool sel = (synthModeCopy == m);
            fillRect(ren, modeRects[m], sel ? modeColors[m] : COL_CELL, sel ? COL_TEXT : COL_BORDER);
            drawText(ren, modeRects[m].x + 12, modeRects[m].y + 8, modeNames[m],
                     sel ? COL_DARK : COL_TEXT, 1);
        }

        SDL_Color activeCol = modeColors[synthModeCopy];

        // Render Live Mirrored Camera Preview inside the XY Pad
        if (hf.camera_ready) {
            SDL_RenderCopy(ren, camTex, nullptr, &xyPadRect);
        } else {
            fillRect(ren, xyPadRect, COL_DARK, COL_BORDER);
        }
        SDL_SetRenderDrawColor(ren, COL_BORDER.r, COL_BORDER.g, COL_BORDER.b, 255);
        SDL_RenderDrawRect(ren, &xyPadRect);

        // Draw Octave / Frequency Reference Lines (55 Hz -> 880 Hz)
        const char *octLabels[5] = {"55 HZ", "110 HZ", "220 HZ", "440 HZ", "880 HZ"};
        for (int o = 0; o < 5; ++o) {
            float fnx = o * 0.25f;
            int gx = xyPadRect.x + (int)(fnx * xyPadRect.w);
            SDL_SetRenderDrawColor(ren, 28, 38, 54, 255);
            SDL_RenderDrawLine(ren, gx, xyPadRect.y, gx, xyPadRect.y + xyPadRect.h);
            int lx = std::clamp(gx - 18, xyPadRect.x + 6, xyPadRect.x + xyPadRect.w - 48);
            drawText(ren, lx, xyPadRect.y + xyPadRect.h - 14, octLabels[o], COL_MUTED, 1);
        }

        // Draw Horizontal Intensity Lines (25%, 50%, 75%)
        const char *intLabels[3] = {"INTENSITY 25%", "INTENSITY 50%", "INTENSITY 75%"};
        for (int il = 1; il <= 3; ++il) {
            float iny = il * 0.25f;
            int gy = xyPadRect.y + (int)((1.0f - iny) * xyPadRect.h);
            SDL_SetRenderDrawColor(ren, 24, 34, 48, 255);
            SDL_RenderDrawLine(ren, xyPadRect.x, gy, xyPadRect.x + xyPadRect.w, gy);
            drawText(ren, xyPadRect.x + 8, gy - 10, intLabels[il - 1], COL_MUTED, 1);
        }

        // Draw Motion Trail inside XY Pad
        if (xyTrail.size() >= 2) {
            SDL_SetRenderDrawColor(ren, activeCol.r, activeCol.g, activeCol.b, 255);
            for (size_t i = 1; i < xyTrail.size(); ++i) {
                int x0 = xyPadRect.x + (int)(xyTrail[i - 1].first * xyPadRect.w);
                int y0 = xyPadRect.y + (int)((1.0f - xyTrail[i - 1].second) * xyPadRect.h);
                int x1 = xyPadRect.x + (int)(xyTrail[i].first * xyPadRect.w);
                int y1 = xyPadRect.y + (int)((1.0f - xyTrail[i].second) * xyPadRect.h);
                SDL_RenderDrawLine(ren, x0, y0, x1, y1);
            }
        }

        // Draw 21-Landmark MediaPipe Hand Skeleton inside XY Pad when hand is visible
        if (hf.hand_detected) {
            // Draw 21 bone connections
            SDL_SetRenderDrawColor(ren, COL_CYAN.r, COL_CYAN.g, COL_CYAN.b, 255);
            for (const auto &edge : mediapipe_hands::HAND_CONNECTIONS) {
                const auto &p0 = hf.landmarks[edge.first];
                const auto &p1 = hf.landmarks[edge.second];
                if (p0.confidence > 0.2f && p1.confidence > 0.2f) {
                    int x0 = xyPadRect.x + (int)(p0.x * xyPadRect.w);
                    int y0 = xyPadRect.y + (int)(p0.y * xyPadRect.h);
                    int x1 = xyPadRect.x + (int)(p1.x * xyPadRect.w);
                    int y1 = xyPadRect.y + (int)(p1.y * xyPadRect.h);
                    SDL_RenderDrawLine(ren, x0, y0, x1, y1);
                }
            }
            // Draw 21 joints
            for (int j = 0; j < mediapipe_hands::NUM_LANDMARKS; ++j) {
                const auto &lm = hf.landmarks[j];
                if (lm.confidence > 0.2f) {
                    int jx = xyPadRect.x + (int)(lm.x * xyPadRect.w);
                    int jy = xyPadRect.y + (int)(lm.y * xyPadRect.h);
                    bool isTip = (j == 4 || j == 8 || j == 12 || j == 16 || j == 20);
                    drawCircle(ren, jx, jy, isTip ? 5 : 3, isTip ? COL_MAGENTA : COL_TEXT, true);
                }
            }
        }

        // Draw Active XY Control Point & Frequency/Intensity HUD
        int curX = xyPadRect.x + (int)(normX * xyPadRect.w);
        int curY = xyPadRect.y + (int)((1.0f - normY) * xyPadRect.h);

        if (synthActive && currentIntensity > 0.01f) {
            SDL_SetRenderDrawColor(ren, 35, 52, 74, 255);
            SDL_RenderDrawLine(ren, curX, xyPadRect.y, curX, xyPadRect.y + xyPadRect.h);
            SDL_RenderDrawLine(ren, xyPadRect.x, curY, xyPadRect.x + xyPadRect.w, curY);

            int pulseR = 16 + (int)(currentIntensity * 36.0f + 5.0f * std::sin(nowSec * 14.0));
            drawCircle(ren, curX, curY, pulseR, activeCol, false);
            drawCircle(ren, curX, curY, 8, activeCol, true);

            int hudX = (curX < xyPadRect.x + xyPadRect.w - 190) ? (curX + 18) : (curX - 165);
            char freqStr[64], intStr[64];
            std::snprintf(freqStr, sizeof(freqStr), "%.1f HZ", currentHz);
            std::snprintf(intStr, sizeof(intStr), "INTENSITY %d%%", (int)(currentIntensity * 100.0f));
            drawText(ren, hudX, curY - 16, freqStr, COL_TEXT, 2);
            drawText(ren, hudX, curY + 4, intStr, activeCol, 1);
        } else {
            drawCircle(ren, curX, curY, 7, COL_MUTED, false);
        }

        // Status banner at top of XY Pad
        const char *statusTxt = hf.hand_detected
            ? "HAND TRACKED (21 MEDIAPIPE LANDMARKS ACTIVE)"
            : (hf.camera_ready ? "SHOW HAND IN CAMERA (OR DRAG WITH MOUSE)"
                               : "CAMERA INITIALIZING - DRAG WITH MOUSE OR USE BIO-SONIFY");
        drawText(ren, xyPadRect.x + 12, xyPadRect.y + 12, statusTxt,
                 hf.hand_detected ? COL_GREEN : COL_MUTED, 1);

        // --- RIGHT PANEL: TOUCHDESIGNER 3D AUDIO-REACTIVE POINT CLOUD ---
        fillRect(ren, visPanelRect, COL_PANEL, COL_BORDER);
        drawText(ren, visPanelRect.x + 14, visPanelRect.y + 14, "TD // AUDIO ANALYSIS & BIOMESH", COL_CYAN, 2);
        drawText(ren, visPanelRect.x + 14, visPanelRect.y + 36, "REAL-TIME CHOP SPECTRUM + SOP MESH", COL_MUTED, 1);

        SDL_Rect visCanvas = {visPanelRect.x + 12, visPanelRect.y + 54, visPanelRect.w - 24, visPanelRect.h - 66};
        fillRect(ren, visCanvas, COL_DARK, COL_BORDER);

        int vcx = visCanvas.x + visCanvas.w / 2;
        int vcy = visCanvas.y + (int)(visCanvas.h * 0.42f);

        int haloR = 72 + (int)(subE * 34.0f);
        drawCircle(ren, vcx, vcy, haloR, COL_BEAT, false);

        visAngle += 0.015f + (subE + synthE + microE) * 0.018f;
        float ca = std::cos(visAngle), sa = std::sin(visAngle);
        float cb = std::cos(visAngle * 0.63f), sb = std::sin(visAngle * 0.63f);
        float baseScale = 96.0f;

        struct ProjPt { int x, y; float z, disp; };
        std::vector<ProjPt> proj;
        proj.reserve(visNodes.size());

        for (const auto &n : visNodes) {
            float disp = 1.0f
                + 0.24f * subE * std::sin(n.seed + (float)nowSec * 4.0f)
                + 0.28f * synthE * std::cos(n.seed * 1.7f + (float)nowSec * 6.0f)
                + 0.18f * microE * std::sin(n.seed * 3.1f + (float)nowSec * 18.0f);
            float nx = n.bx * disp, ny = n.by * disp, nz = n.bz * disp;
            float rx = nx * ca - nz * sa;
            float rz = nx * sa + nz * ca;
            float ry = ny * cb - rz * sb;
            float rz2 = ny * sb + rz * cb;
            float fov = 2.6f / (2.6f + rz2 * 0.7f);
            int px = vcx + (int)(rx * baseScale * fov);
            int py = vcy + (int)(ry * baseScale * fov);
            proj.push_back({px, py, rz2, disp});
        }

        int threshSq = (int)std::pow(50.0f + synthE * 24.0f, 2.0f);
        SDL_Color meshCol = (subE + synthE > 0.25f) ? activeCol : COL_BEAT;
        SDL_SetRenderDrawColor(ren, meshCol.r, meshCol.g, meshCol.b, 255);
        for (size_t i = 0; i < proj.size(); ++i) {
            for (size_t j = i + 1; j < proj.size(); ++j) {
                int dx = proj[i].x - proj[j].x;
                int dy = proj[i].y - proj[j].y;
                if (dx * dx + dy * dy < threshSq) {
                    SDL_RenderDrawLine(ren, proj[i].x, proj[i].y, proj[j].x, proj[j].y);
                }
            }
        }
        for (const auto &p : proj) {
            drawCircle(ren, p.x, p.y, 3, (p.disp > 1.06f) ? COL_CYAN : COL_TEXT, true);
        }

        // CHOP Telemetry Bars at bottom of right panel
        int chopY = visCanvas.y + visCanvas.h - 110;
        SDL_Rect chopBox = {visCanvas.x + 8, chopY, visCanvas.w - 16, 102};
        fillRect(ren, chopBox, COL_PANEL, COL_BORDER);

        struct ChopBar { const char *name; float val; SDL_Color col; };
        ChopBar bars[3] = {
            {"CHOP: SUB PULSE", subE, COL_CYAN},
            {"CHOP: HAND XY",   synthE, activeCol},
            {"CHOP: MICRO",     microE, COL_MAGENTA}
        };
        for (int b = 0; b < 3; ++b) {
            int by = chopY + 16 + b * 28;
            drawText(ren, chopBox.x + 10, by + 3, bars[b].name, COL_MUTED, 1);
            SDL_Rect barBg = {chopBox.x + 118, by, chopBox.w - 130, 14};
            fillRect(ren, barBg, COL_DARK, COL_BORDER);
            int fw = (int)(barBg.w * std::clamp(bars[b].val / 1.6f, 0.0f, 1.0f));
            if (fw > 2) {
                SDL_Rect barFg = {barBg.x, barBg.y, fw, barBg.h};
                fillRect(ren, barFg, bars[b].col, bars[b].col);
            }
        }

        SDL_RenderPresent(ren);
    }

    handTracker.stop();
    shutdownCoreAudio(audio);
    if (camTex) SDL_DestroyTexture(camTex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
