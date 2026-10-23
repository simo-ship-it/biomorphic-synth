/*
 * Ultra-Low-Latency CoreAudio Polyphonic RAM Mixer + Continuous Real-Time XY Synthesizer
 * Buffer size: 512 frames @ 44100 Hz (~11.6 ms latency)
 * - 32-voice sample-accurate polyphonic drum mixer
 * - Phase-continuous real-time XY Pad synthesizer with per-sample parameter smoothing
 */

#include <AudioToolbox/AudioToolbox.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAMPLE_RATE 44100
#define NUM_BUFFERS 3
#define BUFFER_FRAMES 512
#define MAX_SAMPLES 64
#define MAX_VOICES 32
#define TWO_PI 6.28318530717958647692

typedef struct {
    float *data;
    int length;
} SampleBuffer;

typedef struct {
    int active;
    int sample_id;
    double pos;
    double rate;
    float gain;
} Voice;

/* Real-time Continuous XY Synth State */
typedef struct {
    float target_freq;   /* Hz (e.g. 55.0 -> 880.0) */
    float current_freq;
    float target_amp;    /* 0.0 -> 1.0 (controlled by Y axis) */
    float current_amp;
    int mode;            /* 0 = Biomorphic Buchla, 1 = Harmonic Drone, 2 = Crystal FM */

    double phase1;
    double phase2;
    double phase3;
    double lfo_phase;
} ContinuousSynth;

static SampleBuffer g_samples[MAX_SAMPLES];
static Voice g_voices[MAX_VOICES];
static ContinuousSynth g_synth;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static AudioQueueRef g_queue = NULL;
static AudioQueueBufferRef g_buffers[NUM_BUFFERS];
static int g_running = 0;

static void audio_callback(void *user_data, AudioQueueRef queue, AudioQueueBufferRef buf) {
    int16_t *out = (int16_t *)buf->mAudioData;
    int frames = BUFFER_FRAMES;

    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < frames; i++) {
        float mix_l = 0.0f;
        float mix_r = 0.0f;

        /* 1. Mix active 4/4 Drum Machine voices */
        for (int v = 0; v < MAX_VOICES; v++) {
            if (!g_voices[v].active) continue;
            int sid = g_voices[v].sample_id;
            if (sid < 0 || sid >= MAX_SAMPLES || !g_samples[sid].data) {
                g_voices[v].active = 0;
                continue;
            }
            int idx = (int)g_voices[v].pos;
            if (idx >= g_samples[sid].length - 1) {
                g_voices[v].active = 0;
                continue;
            }
            double frac = g_voices[v].pos - (double)idx;
            float s0 = g_samples[sid].data[idx];
            float s1 = g_samples[sid].data[idx + 1];
            float val = (s0 + (float)frac * (s1 - s0)) * g_voices[v].gain;
            mix_l += val;
            mix_r += val;
            g_voices[v].pos += g_voices[v].rate;
        }

        /* 2. Continuous Real-Time XY Pad Synthesizer (Click-free per-sample smoothing) */
        g_synth.current_freq += 0.0035f * (g_synth.target_freq - g_synth.current_freq);
        g_synth.current_amp  += 0.0025f * (g_synth.target_amp  - g_synth.current_amp);

        if (g_synth.current_amp > 0.0002f) {
            float f = g_synth.current_freq;
            float amp = g_synth.current_amp;

            g_synth.lfo_phase += TWO_PI * 0.42 / SAMPLE_RATE;
            if (g_synth.lfo_phase >= TWO_PI) g_synth.lfo_phase -= TWO_PI;
            float lfo = (float)sin(g_synth.lfo_phase);

            float syn_l = 0.0f;
            float syn_r = 0.0f;

            if (g_synth.mode == 0) {
                /* Mode 0: Biomorphic Buchla Wavefolder (Y intensity also opens harmonic fold) */
                g_synth.phase1 += TWO_PI * f / SAMPLE_RATE;
                g_synth.phase2 += TWO_PI * (f * 0.5) / SAMPLE_RATE;
                g_synth.phase3 += TWO_PI * (f * 1.502) / SAMPLE_RATE;
                if (g_synth.phase1 >= TWO_PI) g_synth.phase1 -= TWO_PI;
                if (g_synth.phase2 >= TWO_PI) g_synth.phase2 -= TWO_PI;
                if (g_synth.phase3 >= TWO_PI) g_synth.phase3 -= TWO_PI;

                float fold = 1.0f + amp * (2.6f + 0.4f * lfo);
                float raw = (float)sin(g_synth.phase1) + 0.35f * amp * (float)sin(g_synth.phase3);
                float folded = (float)sin(raw * fold);
                float sub = 0.45f * (float)sin(g_synth.phase2);

                syn_l = (folded * 0.65f + sub) * amp * 0.55f;
                syn_r = (folded * 0.65f + sub * 0.95f) * amp * 0.55f;

            } else if (g_synth.mode == 1) {
                /* Mode 1: Organic Glass / Harmonic Drone (Detuned Stereo Chorus + Fifth/Octave) */
                g_synth.phase1 += TWO_PI * (f * (1.0 + 0.003 * lfo)) / SAMPLE_RATE;
                g_synth.phase2 += TWO_PI * (f * (1.0 - 0.003 * lfo)) / SAMPLE_RATE;
                g_synth.phase3 += TWO_PI * (f * 1.4983) / SAMPLE_RATE; /* fifth */
                if (g_synth.phase1 >= TWO_PI) g_synth.phase1 -= TWO_PI;
                if (g_synth.phase2 >= TWO_PI) g_synth.phase2 -= TWO_PI;
                if (g_synth.phase3 >= TWO_PI) g_synth.phase3 -= TWO_PI;

                float osc1 = (float)sin(g_synth.phase1) + 0.3f * amp * (float)sin(g_synth.phase1 * 2.0);
                float osc2 = (float)sin(g_synth.phase2) + 0.3f * amp * (float)sin(g_synth.phase2 * 2.0);
                float fifth = (0.15f + 0.35f * amp) * (float)sin(g_synth.phase3);

                syn_l = (osc1 * 0.6f + fifth * 0.4f) * amp * 0.55f;
                syn_r = (osc2 * 0.6f + fifth * 0.4f) * amp * 0.55f;

            } else {
                /* Mode 2: Post-Nature Crystal FM (Y intensity controls FM index & shimmer) */
                double mod_ratio = 1.41421356; /* sqrt(2) crystalline ratio */
                g_synth.phase1 += TWO_PI * f / SAMPLE_RATE;
                g_synth.phase2 += TWO_PI * (f * mod_ratio) / SAMPLE_RATE;
                g_synth.phase3 += TWO_PI * (f * 2.002) / SAMPLE_RATE;
                if (g_synth.phase1 >= TWO_PI) g_synth.phase1 -= TWO_PI;
                if (g_synth.phase2 >= TWO_PI) g_synth.phase2 -= TWO_PI;
                if (g_synth.phase3 >= TWO_PI) g_synth.phase3 -= TWO_PI;

                float mod_idx = 0.2f + amp * amp * 3.8f;
                float mod = (float)sin(g_synth.phase2);
                float car_l = (float)sin(g_synth.phase1 + mod_idx * mod);
                float car_r = (float)sin(g_synth.phase1 + mod_idx * mod * 0.92f + 0.25f * lfo);
                float oct = 0.28f * amp * (float)sin(g_synth.phase3);

                syn_l = (car_l * 0.65f + oct) * amp * 0.50f;
                syn_r = (car_r * 0.65f + oct) * amp * 0.50f;
            }

            mix_l += syn_l;
            mix_r += syn_r;
        }

        /* Soft clip stereo output */
        if (mix_l > 1.0f) mix_l = 1.0f - 1.0f / (1.0f + mix_l * 2.0f);
        else if (mix_l < -1.0f) mix_l = -1.0f + 1.0f / (1.0f - mix_l * 2.0f);
        else mix_l *= 0.78f;

        if (mix_r > 1.0f) mix_r = 1.0f - 1.0f / (1.0f + mix_r * 2.0f);
        else if (mix_r < -1.0f) mix_r = -1.0f + 1.0f / (1.0f - mix_r * 2.0f);
        else mix_r *= 0.78f;

        int32_t pcm_l = (int32_t)(mix_l * 32767.0f);
        int32_t pcm_r = (int32_t)(mix_r * 32767.0f);
        if (pcm_l > 32767) pcm_l = 32767;
        if (pcm_l < -32768) pcm_l = -32768;
        if (pcm_r > 32767) pcm_r = 32767;
        if (pcm_r < -32768) pcm_r = -32768;

        out[i * 2]     = (int16_t)pcm_l;
        out[i * 2 + 1] = (int16_t)pcm_r;
    }
    pthread_mutex_unlock(&g_lock);

    buf->mAudioDataByteSize = frames * 2 * sizeof(int16_t);
    if (g_running) {
        AudioQueueEnqueueBuffer(queue, buf, 0, NULL);
    }
}

int mixer_init(void) {
    if (g_running) return 0;

    memset(g_samples, 0, sizeof(g_samples));
    memset(g_voices, 0, sizeof(g_voices));
    memset(&g_synth, 0, sizeof(g_synth));
    g_synth.target_freq = 220.0f;
    g_synth.current_freq = 220.0f;
    g_synth.target_amp = 0.0f;
    g_synth.current_amp = 0.0f;
    g_synth.mode = 0;

    AudioStreamBasicDescription fmt = {0};
    fmt.mSampleRate = SAMPLE_RATE;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    fmt.mBitsPerChannel = 16;
    fmt.mChannelsPerFrame = 2;
    fmt.mBytesPerFrame = 4;
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerPacket = 4;

    OSStatus status = AudioQueueNewOutput(&fmt, audio_callback, NULL, NULL, NULL, 0, &g_queue);
    if (status != noErr) return -1;

    g_running = 1;
    int buf_bytes = BUFFER_FRAMES * 4;
    for (int i = 0; i < NUM_BUFFERS; i++) {
        AudioQueueAllocateBuffer(g_queue, buf_bytes, &g_buffers[i]);
        memset(g_buffers[i]->mAudioData, 0, buf_bytes);
        g_buffers[i]->mAudioDataByteSize = buf_bytes;
        AudioQueueEnqueueBuffer(g_queue, g_buffers[i], 0, NULL);
    }

    status = AudioQueueStart(g_queue, NULL);
    return (status == noErr) ? 0 : -2;
}

int mixer_load_sample(int sample_id, const float *data, int length) {
    if (sample_id < 0 || sample_id >= MAX_SAMPLES || length <= 0) return -1;
    float *copy = (float *)malloc(sizeof(float) * length);
    if (!copy) return -2;
    memcpy(copy, data, sizeof(float) * length);

    pthread_mutex_lock(&g_lock);
    if (g_samples[sample_id].data) {
        free(g_samples[sample_id].data);
    }
    g_samples[sample_id].data = copy;
    g_samples[sample_id].length = length;
    pthread_mutex_unlock(&g_lock);
    return 0;
}

void mixer_trigger(int sample_id, float gain, float rate) {
    if (sample_id < 0 || sample_id >= MAX_SAMPLES) return;
    pthread_mutex_lock(&g_lock);
    int chosen = -1;
    double max_pos = -1.0;
    int oldest = 0;
    for (int v = 0; v < MAX_VOICES; v++) {
        if (!g_voices[v].active) {
            chosen = v;
            break;
        }
        if (g_voices[v].pos > max_pos) {
            max_pos = g_voices[v].pos;
            oldest = v;
        }
    }
    if (chosen < 0) chosen = oldest;

    g_voices[chosen].sample_id = sample_id;
    g_voices[chosen].pos = 0.0;
    g_voices[chosen].rate = (rate > 0.2f && rate < 4.0f) ? (double)rate : 1.0;
    g_voices[chosen].gain = gain;
    g_voices[chosen].active = 1;
    pthread_mutex_unlock(&g_lock);
}

void mixer_set_synth(float freq, float amp, int mode) {
    if (freq < 20.0f) freq = 20.0f;
    if (freq > 4000.0f) freq = 4000.0f;
    if (amp < 0.0f) amp = 0.0f;
    if (amp > 1.2f) amp = 1.2f;
    pthread_mutex_lock(&g_lock);
    g_synth.target_freq = freq;
    g_synth.target_amp = amp;
    g_synth.mode = mode;
    pthread_mutex_unlock(&g_lock);
}

void mixer_stop_all(void) {
    pthread_mutex_lock(&g_lock);
    for (int v = 0; v < MAX_VOICES; v++) {
        g_voices[v].active = 0;
    }
    g_synth.target_amp = 0.0f;
    pthread_mutex_unlock(&g_lock);
}

void mixer_shutdown(void) {
    if (!g_running) return;
    g_running = 0;
    if (g_queue) {
        AudioQueueStop(g_queue, true);
        AudioQueueDispose(g_queue, true);
        g_queue = NULL;
    }
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_SAMPLES; i++) {
        if (g_samples[i].data) {
            free(g_samples[i].data);
            g_samples[i].data = NULL;
        }
    }
    pthread_mutex_unlock(&g_lock);
}
