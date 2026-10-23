/*
 * BIOMORPHIC (sintesi) — Delicate Ambient Electronic Edition (C++20 / Cocoa / Vision)
 * Minimalist Swiss Editorial Style ("RIVERBELT" warm paper & charcoal ink palette)
 * Inspired by Brian Eno, Ryuichi Sakamoto & Alva Noto:
 *
 * - 001 — RITMO MICRO-AMBIENT (4 VOCI DELICATE):
 *     1. PULSO  (battito sub-sinusoidale morbido a 52 Hz, attacco vellutato)
 *     2. LEGNO  (tocco soffice di marimba/legno smorzato)
 *     3. SOFFIO (fruscio d'aria / vinile filtrato)
 *     4. GOCCIA (micro-armonica cristallina in alta frequenza)
 *
 * - 002 — TIMBRO AMBIENT (TEMPO REALE):
 *     FILTRO (morbidezza e calore), DECAY (durata code ambient), SPAZIO (nuvola di riverbero/delay).
 *
 * - 003 — MANI (INTERAZIONE CON I POLPASTRELLI):
 *     * RIQUADRO SINISTRO (MANO SX · ACCORDI AMBIENT):
 *         - 1–4 dita scelgono l'accordo ambient (DO maj9, SOL add9, LA min9, MI min9)
 *         - Asse X: a sinistra suono LISCIO e puro, verso destra VIBRATO dolce su nastro
 *         - Asse Y: calore/apertura del pad armonico
 *     * RIQUADRO SUPERIORE DESTRO (DX · ATMOSFERA ARMONIA):
 *         4 modulazioni ambient delicate sugli accordi:
 *         - FIORITURA (fioritura lenta delle note dell'accordo)
 *         - RESPIRO   (onda lenta di filtro e volume come un respiro)
 *         - CRISTALLO (armoniche eteree di quinta e ottava)
 *         - NASTRO    (calda deriva analogica stile Sakamoto)
 *     * RIQUADRO INFERIORE DESTRO (MANO DX · TASTIERA POLPASTRELLI - SCALA DO):
 *         Basta passare o appoggiare i POLPASTRELLI della mano destra sulle 8 note
 *         della scala di DO (DO, RE, MI, FA, SOL, LA, SI, DO5) per suonarle e sostenerle.
 */

#import <AudioToolbox/AudioToolbox.h>
#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

#include "mediapipe_hand_tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <utility>
#include <vector>

static constexpr int SAMPLE_RATE = 44100;
static constexpr int NUM_AUDIO_BUFFERS = 3;
static constexpr int AUDIO_BUFFER_FRAMES = 512;
static constexpr double TWO_PI = 6.28318530717958647692;
static constexpr int MAX_VOICES = 12;
static constexpr int NUM_DRUM_TRACKS = 4;
static constexpr int SCOPE_SAMPLES = 128;

// Normalized bounds of the Quadrant Boxes inside the camera view (top-down Y: 0=top, 1=bottom)
static constexpr float LEFT_BOX_X0   = 0.04f;
static constexpr float LEFT_BOX_X1   = 0.47f;
static constexpr float LEFT_BOX_Y0   = 0.08f;
static constexpr float LEFT_BOX_Y1   = 0.92f;

static constexpr float RIGHT_BOX_X0  = 0.53f;
static constexpr float RIGHT_BOX_X1  = 0.96f;
static constexpr float GLITCH_BOX_Y0 = 0.08f;
static constexpr float GLITCH_BOX_Y1 = 0.31f;
static constexpr float NOTES_BOX_Y0  = 0.37f;
static constexpr float NOTES_BOX_Y1  = 0.92f;

static constexpr int NUM_SCALE_NOTES = 8;

struct ScaleNote {
    const char *name;
    const char *short_name;
    float freq;
};

// C Major Scale (Scala di DO Maggiore: DO, RE, MI, FA, SOL, LA, SI, DO5)
static constexpr ScaleNote QUADRANT_NOTES[NUM_SCALE_NOTES] = {
    {"DO 4",  "DO",  261.63f},
    {"RE 4",  "RE",  293.66f},
    {"MI 4",  "MI",  329.63f},
    {"FA 4",  "FA",  349.23f},
    {"SOL 4", "SOL", 392.00f},
    {"LA 4",  "LA",  440.00f},
    {"SI 4",  "SI",  493.88f},
    {"DO 5",  "DO5", 523.25f}
};

static const char *GLITCH_MODE_NAMES[4] = {
    "FIORITURA", "RESPIRO", "CRISTALLO", "NASTRO"
};

static const char *DRUM_TRACK_NAMES[NUM_DRUM_TRACKS] = {
    "PULSO", "LEGNO", "SOFFIO", "GOCCIA"
};

// 5 Fingertip landmark indices (Polpastrelli)
static constexpr int FINGERTIP_INDICES[5] = {
    mediapipe_hands::INDEX_FINGER_TIP,
    mediapipe_hands::MIDDLE_FINGER_TIP,
    mediapipe_hands::RING_FINGER_TIP,
    mediapipe_hands::PINKY_TIP,
    mediapipe_hands::THUMB_TIP
};

struct ChordDef {
    const char *name;
    float freqs[4];
};

// Delicate, open ambient voicings (Sakamoto / Eno style)
static constexpr ChordDef CHORDS[4] = {
    {"DO MAJ9", {130.81f, 196.00f, 246.94f, 293.66f}}, // C2, G3, B3, D4
    {"SOL ADD9",{98.00f,  146.83f, 220.00f, 246.94f}}, // G2, D3, A3, B3
    {"LA MIN9", {110.00f, 164.81f, 246.94f, 261.63f}}, // A2, E3, B3, C4
    {"MI MIN9", {82.41f,  146.83f, 185.00f, 246.94f}}  // E2, D3, F#3, B3
};

// Delicate Ambient Felted-Glass / Sine-Bell Voice triggered by fingertips
struct AmbientNoteVoice {
    bool active = false;
    double t = 0.0;
    float base_freq = 440.0f;
    float gain = 0.6f;
    float pan = 0.5f;
    double ph1 = 0.0;
    double ph2 = 0.0;
    double ph3 = 0.0;
};

static constexpr int DELAY_BUF_SAMPLES = 65536; // ~1.48 sec lush ambient cloud buffer

// =====================================================================
// REAL-TIME COREAUDIO AMBIENT ENGINE
// =====================================================================

struct AudioSynthState {
    std::mutex mtx;

    // 4 Delicate Micro-Ambient Percussion Samples
    std::vector<float> drum_samples[NUM_DRUM_TRACKS];
    double drum_pos[NUM_DRUM_TRACKS] = {-1.0, -1.0, -1.0, -1.0};
    float drum_gain[NUM_DRUM_TRACKS] = {0.72f, 0.58f, 0.44f, 0.52f};
    std::array<bool, NUM_DRUM_TRACKS> drum_muted = {false, false, false, false};

    bool is_playing = true;
    int bpm = 84; // Contemplative ambient tempo
    float swing = 0.08f;
    std::array<std::array<int, 16>, NUM_DRUM_TRACKS> grid = {{
        {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0}, // PULSO
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, // LEGNO
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}, // SOFFIO
        {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1}  // GOCCIA
    }};
    int current_step = 0;
    int displayed_step = 0;
    double step_sample_counter = 0.0;

    // Real-time Ambient Sound Macros (002 — TIMBRO)
    float master_filter = 0.72f; // Warmth / Brilliance
    float master_decay  = 0.68f; // Lush ambient decay length
    float master_space  = 0.56f; // Ambient cloud reverb/delay wet mix

    // Left Hand Ambient Pad + Gentle Vibrato (X) + Warmth (Y)
    int active_chord = 0;
    float target_chord_freqs[4] = {130.81f, 196.00f, 246.94f, 293.66f};
    float current_chord_freqs[4] = {130.81f, 196.00f, 246.94f, 293.66f};
    float target_chord_amp = 0.0f;
    float current_chord_amp = 0.0f;
    double chord_phases[4] = {0.0, 0.0, 0.0, 0.0};
    double chord_shimmer_phases[4] = {0.0, 0.0, 0.0, 0.0};
    double chord_lfo = 0.0;

    float target_chord_vibrato = 0.0f;
    float current_chord_vibrato = 0.0f;
    float target_chord_brightness = 0.60f;
    float current_chord_brightness = 0.60f;
    float chord_lp_l = 0.0f;
    float chord_lp_r = 0.0f;

    // Top-Right Delicate Ambient Harmonic Variations
    bool glitch_active = false;
    int glitch_mode = 0;   // 0=FIORITURA, 1=RESPIRO, 2=CRISTALLO, 3=NASTRO
    float glitch_x = 0.5f;
    float current_glitch_mix = 0.0f;
    double bloom_phase = 0.0;
    double tape_drift_phase = 0.0;

    // Polyphonic Delicate Ambient Voices + Per-Note Fingertip Sustain Gates
    std::array<AmbientNoteVoice, MAX_VOICES> voices{};
    std::array<float, NUM_SCALE_NOTES> note_target_sustain{};
    std::array<float, NUM_SCALE_NOTES> note_current_sustain{};
    std::array<double, NUM_SCALE_NOTES> sustain_phases_l{};
    std::array<double, NUM_SCALE_NOTES> sustain_phases_r{};
    double sustain_lfo = 0.0;

    // Master Filter & Lush Multi-Tap Stereo Cloud Diffuser
    float master_lp_l = 0.0f;
    float master_lp_r = 0.0f;
    std::array<float, DELAY_BUF_SAMPLES> delay_l{};
    std::array<float, DELAY_BUF_SAMPLES> delay_r{};
    int delay_write_idx = 0;
    float delay_fb_lp_l = 0.0f;
    float delay_fb_lp_r = 0.0f;

    // Live Oscilloscope Ring Buffer
    std::array<float, SCOPE_SAMPLES> scope_ring{};
    int scope_write_idx = 0;

    AudioQueueRef queue = nullptr;
    AudioQueueBufferRef buffers[NUM_AUDIO_BUFFERS];
    bool running = false;

    void generateDrumSamples() {
        // 0: PULSO (Soft, warm 52 Hz heartbeat sub-sine with pillowy 8ms fade-in, zero harsh click)
        {
            int n = (int)(SAMPLE_RATE * 0.36f);
            drum_samples[0].resize(n);
            double ph = 0.0;
            float peak = 0.001f;
            for (int i = 0; i < n; ++i) {
                double t = (double)i / SAMPLE_RATE;
                double freq = 50.0 + 38.0 * std::exp(-t * 28.0);
                ph += TWO_PI * freq / SAMPLE_RATE;
                double attack = 1.0 - std::exp(-t * 260.0);
                double env = attack * std::exp(-t * 9.5);
                float s = (float)((std::sin(ph) + 0.08 * std::sin(2.0 * ph)) * env);
                drum_samples[0][i] = s;
                peak = std::max(peak, std::fabs(s));
            }
            for (float &s : drum_samples[0]) s = (s / peak) * 0.80f;
        }

        // 1: LEGNO (Delicate warm wooden kalimba/marimba tap at 293.66 Hz D4 + soft 5th)
        {
            int n = (int)(SAMPLE_RATE * 0.20f);
            drum_samples[1].resize(n);
            double ph1 = 0.0, ph2 = 0.0;
            float peak = 0.001f;
            for (int i = 0; i < n; ++i) {
                double t = (double)i / SAMPLE_RATE;
                double f1 = 293.66 + 24.0 * std::exp(-t * 90.0);
                double f2 = f1 * 2.98;
                ph1 += TWO_PI * f1 / SAMPLE_RATE;
                ph2 += TWO_PI * f2 / SAMPLE_RATE;
                double attack = 1.0 - std::exp(-t * 450.0);
                double s = (std::sin(ph1) * std::exp(-t * 26.0) +
                            0.22 * std::sin(ph2) * std::exp(-t * 85.0)) * attack;
                drum_samples[1][i] = (float)s;
                peak = std::max(peak, std::fabs((float)s));
            }
            for (float &s : drum_samples[1]) s = (s / peak) * 0.65f;
        }

        // 2: SOFFIO (Gentle, warm band-passed air/vinyl brush whisper)
        {
            int n = (int)(SAMPLE_RATE * 0.18f);
            drum_samples[2].resize(n);
            float lp = 0.0f, hp_prev = 0.0f, hp = 0.0f;
            uint32_t rng = 0x2468ACE1u;
            float peak = 0.001f;
            for (int i = 0; i < n; ++i) {
                double t = (double)i / SAMPLE_RATE;
                rng = rng * 1664525u + 1013904223u;
                float white = ((float)(rng & 0xFFFF) / 32768.0f) - 1.0f;
                hp = 0.78f * (hp + white - hp_prev);
                hp_prev = white;
                lp += 0.18f * (hp - lp);
                double env = std::sin(3.14159265 * std::min(1.0, t / 0.18)) * std::exp(-t * 14.0);
                float s = (float)(lp * env);
                drum_samples[2][i] = s;
                peak = std::max(peak, std::fabs(s));
            }
            for (float &s : drum_samples[2]) s = (s / peak) * 0.45f;
        }

        // 3: GOCCIA (Delicate ambient harmonic glass ping at G5 784 Hz)
        {
            int n = (int)(SAMPLE_RATE * 0.28f);
            drum_samples[3].resize(n);
            double ph1 = 0.0, ph2 = 0.0;
            float peak = 0.001f;
            for (int i = 0; i < n; ++i) {
                double t = (double)i / SAMPLE_RATE;
                ph1 += TWO_PI * 783.99 / SAMPLE_RATE;
                ph2 += TWO_PI * (783.99 * 2.0) / SAMPLE_RATE;
                double attack = 1.0 - std::exp(-t * 380.0);
                double env = attack * std::exp(-t * 16.0);
                float s = (float)((std::sin(ph1) + 0.18 * std::sin(ph2) * std::exp(-t * 35.0)) * env);
                drum_samples[3][i] = s;
                peak = std::max(peak, std::fabs(s));
            }
            for (float &s : drum_samples[3]) s = (s / peak) * 0.52f;
        }
    }

    void triggerDrum(int track, float accent = 1.0f) {
        if (track < 0 || track >= NUM_DRUM_TRACKS || drum_muted[track]) return;
        drum_pos[track] = 0.0;
        drum_gain[track] = 0.68f * accent;
    }

    void setChord(int chordIdx) {
        active_chord = std::clamp(chordIdx, 0, 4);
        if (active_chord == 0) {
            target_chord_amp = 0.0f;
        } else {
            const auto &ch = CHORDS[active_chord - 1];
            for (int k = 0; k < 4; ++k) {
                target_chord_freqs[k] = ch.freqs[k];
            }
            target_chord_amp = 0.34f;
        }
    }

    void triggerAmbientNote(int noteIdx, float intensity) {
        noteIdx = std::clamp(noteIdx, 0, NUM_SCALE_NOTES - 1);
        int chosen = 0;
        double max_t = -1.0;
        for (int i = 0; i < MAX_VOICES; ++i) {
            if (!voices[i].active) {
                chosen = i;
                break;
            }
            if (voices[i].t > max_t) {
                max_t = voices[i].t;
                chosen = i;
            }
        }
        auto &v = voices[chosen];
        v.active = true;
        v.t = 0.0;
        v.base_freq = QUADRANT_NOTES[noteIdx].freq;
        v.gain = std::clamp(intensity, 0.15f, 1.0f);
        v.pan = 0.25f + 0.50f * ((float)noteIdx / (float)(NUM_SCALE_NOTES - 1));
        v.ph1 = 0.0;
        v.ph2 = 0.0;
        v.ph3 = 0.0;
    }
};

static void coreAudioCallback(void *user_data, AudioQueueRef queue, AudioQueueBufferRef buf) {
    AudioSynthState *st = static_cast<AudioSynthState *>(user_data);
    int16_t *out = static_cast<int16_t *>(buf->mAudioData);

    std::lock_guard<std::mutex> lock(st->mtx);
    double base_step_samples = (double)SAMPLE_RATE * 60.0 / (std::max(40, st->bpm) * 4.0);
    const double dt = 1.0 / SAMPLE_RATE;

    float decay_mult = 0.55f + 1.45f * st->master_decay;
    float filter_macro = std::clamp(st->master_filter, 0.12f, 1.0f);
    float space_mix = std::clamp(st->master_space, 0.0f, 1.0f);

    // Lush ambient stereo cloud delay times
    int delay_samples_l = std::clamp((int)(base_step_samples * 3.0), 4000, DELAY_BUF_SAMPLES - 4);
    int delay_samples_r = std::clamp((int)(base_step_samples * 4.5), 6000, DELAY_BUF_SAMPLES - 4);
    int tap2_samples_l  = std::clamp((int)(base_step_samples * 1.75), 2500, DELAY_BUF_SAMPLES - 4);
    int tap2_samples_r  = std::clamp((int)(base_step_samples * 2.25), 3200, DELAY_BUF_SAMPLES - 4);

    for (int i = 0; i < AUDIO_BUFFER_FRAMES; ++i) {
        if (st->is_playing) {
            if (st->step_sample_counter <= 0.0) {
                int s = st->current_step;
                st->displayed_step = s;
                float accent = (s % 4 == 0) ? 1.0f : 0.76f;
                for (int tr = 0; tr < NUM_DRUM_TRACKS; ++tr) {
                    if (st->grid[tr][s]) {
                        st->triggerDrum(tr, accent);
                    }
                }
                double swing_offset = ((s % 2) == 0) ? (1.0 + 0.38 * st->swing) : (1.0 - 0.38 * st->swing);
                st->current_step = (s + 1) % 16;
                st->step_sample_counter += base_step_samples * swing_offset;
            }
            st->step_sample_counter -= 1.0;
        }

        float mix_l = 0.0f;
        float mix_r = 0.0f;
        float ambient_send_l = 0.0f;
        float ambient_send_r = 0.0f;

        // 1. Delicate Micro-Ambient Percussion Bus
        double drum_step_inc = 1.0 / std::clamp((double)decay_mult, 0.65, 1.40);
        for (int tr = 0; tr < NUM_DRUM_TRACKS; ++tr) {
            if (st->drum_pos[tr] >= 0.0 && !st->drum_muted[tr]) {
                int idx = (int)st->drum_pos[tr];
                const auto &samp = st->drum_samples[tr];
                if (idx < (int)samp.size() - 1) {
                    float v = samp[idx] * st->drum_gain[tr];
                    float pan_l = (tr == 2) ? 0.80f : ((tr == 3) ? 1.15f : 1.0f);
                    float pan_r = (tr == 2) ? 1.15f : ((tr == 3) ? 0.80f : 1.0f);
                    mix_l += v * pan_l;
                    mix_r += v * pan_r;
                    if (tr >= 1) {
                        ambient_send_l += v * pan_l * 0.35f;
                        ambient_send_r += v * pan_r * 0.35f;
                    }
                    st->drum_pos[tr] += drum_step_inc;
                } else {
                    st->drum_pos[tr] = -1.0;
                }
            }
        }

        // 2. Left Hand Ambient Pad (Pure Sine + Soft Harmonics, Gentle Vibrato on Right)
        st->current_chord_amp += 0.0006f * (st->target_chord_amp - st->current_chord_amp);
        st->current_chord_vibrato += 0.0012f * (st->target_chord_vibrato - st->current_chord_vibrato);
        st->current_chord_brightness += 0.0012f * (st->target_chord_brightness - st->current_chord_brightness);
        float target_g = st->glitch_active ? 1.0f : 0.0f;
        st->current_glitch_mix += 0.0015f * (target_g - st->current_glitch_mix);

        if (st->current_chord_amp > 0.0002f) {
            float vib = st->current_chord_vibrato;
            // Gentle, musical vibrato (3.0 Hz -> 5.2 Hz, subtle depth)
            double vib_rate_hz = 3.0 + 2.2 * vib;
            st->chord_lfo += TWO_PI * vib_rate_hz * dt;
            if (st->chord_lfo >= TWO_PI) st->chord_lfo -= TWO_PI;
            float lfo_sin = (float)std::sin(st->chord_lfo);
            float pitch_vib_depth = 0.0095f * vib * vib; // Delicate, intimate pitch vibrato
            float amp_vib_mod = 1.0f - 0.14f * vib * (0.5f * (1.0f - (float)std::cos(st->chord_lfo)));

            // Slow meditative cycle for Top-Right Ambient Variations
            double bloom_hz = (0.45 + 1.35 * st->glitch_x);
            st->bloom_phase += bloom_hz * dt;
            if (st->bloom_phase >= 4.0) st->bloom_phase -= 4.0;

            st->tape_drift_phase += TWO_PI * (0.65 + 1.4 * st->glitch_x) * dt;
            if (st->tape_drift_phase >= TWO_PI) st->tape_drift_phase -= TWO_PI;

            float g_mix = st->current_glitch_mix;
            float tape_drift = (g_mix > 0.01f && st->glitch_mode == 3)
                ? (float)(g_mix * (0.0065 * std::sin(st->tape_drift_phase) + 0.003 * std::cos(st->tape_drift_phase * 2.1)))
                : 0.0f;

            float ch_l = 0.0f;
            float ch_r = 0.0f;
            for (int k = 0; k < 4; ++k) {
                st->current_chord_freqs[k] += 0.0025f * (st->target_chord_freqs[k] - st->current_chord_freqs[k]);
                float detune = (k % 2 == 0) ? 0.9992f : 1.0008f;
                float f = st->current_chord_freqs[k] * detune * (1.0f + pitch_vib_depth * lfo_sin + tape_drift);
                st->chord_phases[k] += TWO_PI * f * dt;
                if (st->chord_phases[k] >= TWO_PI) st->chord_phases[k] -= TWO_PI;

                // Ethereal fifth/octave crystal harmonic
                float crystal_ratio = (k % 2 == 0) ? 2.0f : 1.4983f;
                st->chord_shimmer_phases[k] += TWO_PI * (f * crystal_ratio) * dt;
                if (st->chord_shimmer_phases[k] >= TWO_PI) st->chord_shimmer_phases[k] -= TWO_PI;

                float warm_2nd = (0.08f + 0.18f * st->current_chord_brightness) * filter_macro;
                float voice = (float)std::sin(st->chord_phases[k]) +
                              warm_2nd * (float)std::sin(st->chord_phases[k] * 2.0);

                // Mode 2: CRISTALLO — delicate high fifth & octave glass overtones
                if (g_mix > 0.01f && st->glitch_mode == 2) {
                    float cryst = (float)std::sin(st->chord_shimmer_phases[k]) * 0.32f;
                    voice += cryst * g_mix;
                }

                // Mode 0: FIORITURA — slow overlapping cosine blooming of each chord note
                if (g_mix > 0.01f && st->glitch_mode == 0) {
                    double dist = std::fabs(st->bloom_phase - (k + 0.5));
                    if (dist > 2.0) dist = 4.0 - dist;
                    float bloom_env = (float)std::max(0.15, 0.5 * (1.0 + std::cos(std::min(1.0, dist) * 3.14159265)));
                    voice *= (1.0f - g_mix) + g_mix * (bloom_env * 1.45f);
                }

                ch_l += voice * ((k % 2 == 0) ? 0.27f : 0.18f);
                ch_r += voice * ((k % 2 == 0) ? 0.18f : 0.27f);
            }

            ch_l *= amp_vib_mod;
            ch_r *= amp_vib_mod;

            // Mode 1: RESPIRO — slow ocean-wave breathing of stereo pad & filter
            float breath_mod = 0.0f;
            if (g_mix > 0.01f && st->glitch_mode == 1) {
                float b_l = 0.55f + 0.45f * (float)std::sin(TWO_PI * (st->bloom_phase * 0.25));
                float b_r = 0.55f + 0.45f * (float)std::cos(TWO_PI * (st->bloom_phase * 0.25));
                ch_l *= (1.0f - g_mix) + g_mix * b_l;
                ch_r *= (1.0f - g_mix) + g_mix * b_r;
                breath_mod = g_mix * 0.15f * b_l;
            }

            float ch_cut = std::clamp(0.03f + 0.22f * st->current_chord_brightness * filter_macro + breath_mod, 0.02f, 0.45f);
            st->chord_lp_l += ch_cut * (ch_l - st->chord_lp_l);
            st->chord_lp_r += ch_cut * (ch_r - st->chord_lp_r);

            float pad_l = st->chord_lp_l * st->current_chord_amp;
            float pad_r = st->chord_lp_r * st->current_chord_amp;
            mix_l += pad_l;
            mix_r += pad_r;
            ambient_send_l += pad_l * (0.45f + 0.30f * g_mix);
            ambient_send_r += pad_r * (0.45f + 0.30f * g_mix);
        }

        // 3A. Delicate Ambient Felted-Piano / Glass-Sine Notes (triggered by fingertips passing over keys)
        float notes_l = 0.0f;
        float notes_r = 0.0f;

        double max_note_t = 1.65 * decay_mult;
        double note_decay_rate = 3.2 / decay_mult;

        for (int v = 0; v < MAX_VOICES; ++v) {
            auto &vc = st->voices[v];
            if (!vc.active) continue;

            double t = vc.t;
            if (t > max_note_t) {
                vc.active = false;
                continue;
            }

            vc.ph1 += TWO_PI * vc.base_freq * dt;
            vc.ph2 += TWO_PI * (vc.base_freq * 2.0) * dt;
            vc.ph3 += TWO_PI * (vc.base_freq * 3.0) * dt;
            if (vc.ph1 >= TWO_PI) vc.ph1 -= TWO_PI;
            if (vc.ph2 >= TWO_PI) vc.ph2 -= TWO_PI;
            if (vc.ph3 >= TWO_PI) vc.ph3 -= TWO_PI;

            // Soft 18ms raised-cosine-like fade-in (no harsh click) + long singing decay
            double attack = 1.0 - std::exp(-t * 140.0);
            double env_fund = attack * std::exp(-t * note_decay_rate);
            // Upper harmonics decay gently faster like a felted piano / Rhodes bell
            double env_harm2 = attack * std::exp(-t * (note_decay_rate * 2.4));
            double env_harm3 = attack * std::exp(-t * (note_decay_rate * 4.5));

            double tone = std::sin(vc.ph1) * env_fund +
                          (0.18 * filter_macro * vc.gain) * std::sin(vc.ph2) * env_harm2 +
                          (0.05 * filter_macro * vc.gain) * std::sin(vc.ph3) * env_harm3;

            float sig = (float)(tone * (0.20 + 0.30 * vc.gain));
            notes_l += sig * (1.0f - 0.45f * vc.pan);
            notes_r += sig * (0.55f + 0.45f * vc.pan);

            vc.t += dt;
        }

        // 3B. Continuous Gentle Sustain for fingertips resting on the C Major Scale keys
        st->sustain_lfo += TWO_PI * 3.6 * dt;
        if (st->sustain_lfo >= TWO_PI) st->sustain_lfo -= TWO_PI;
        float sus_lfo = (float)std::sin(st->sustain_lfo);

        for (int n = 0; n < NUM_SCALE_NOTES; ++n) {
            float target_s = st->note_target_sustain[n];
            float rate = (target_s > st->note_current_sustain[n]) ? 0.0020f : (0.0006f / decay_mult);
            st->note_current_sustain[n] += rate * (target_s - st->note_current_sustain[n]);

            float env = st->note_current_sustain[n];
            if (env > 0.0004f) {
                double f = QUADRANT_NOTES[n].freq * (1.0 + 0.0014 * sus_lfo);
                st->sustain_phases_l[n] += TWO_PI * (f * 0.999) * dt;
                st->sustain_phases_r[n] += TWO_PI * (f * 1.001) * dt;
                if (st->sustain_phases_l[n] >= TWO_PI) st->sustain_phases_l[n] -= TWO_PI;
                if (st->sustain_phases_r[n] >= TWO_PI) st->sustain_phases_r[n] -= TWO_PI;

                float harm = 0.12f * filter_macro;
                float sl = (float)std::sin(st->sustain_phases_l[n]) + harm * (float)std::sin(st->sustain_phases_l[n] * 2.0);
                float sr = (float)std::sin(st->sustain_phases_r[n]) + harm * (float)std::sin(st->sustain_phases_r[n] * 2.0);

                notes_l += sl * env * 0.22f;
                notes_r += sr * env * 0.22f;
            }
        }

        mix_l += notes_l;
        mix_r += notes_r;
        ambient_send_l += notes_l * 0.75f;
        ambient_send_r += notes_r * 0.75f;

        // 4. Lush Multi-Tap Ambient Cloud Diffuser (SPAZIO)
        int r1_l = (st->delay_write_idx - delay_samples_l + DELAY_BUF_SAMPLES) % DELAY_BUF_SAMPLES;
        int r1_r = (st->delay_write_idx - delay_samples_r + DELAY_BUF_SAMPLES) % DELAY_BUF_SAMPLES;
        int r2_l = (st->delay_write_idx - tap2_samples_l + DELAY_BUF_SAMPLES) % DELAY_BUF_SAMPLES;
        int r2_r = (st->delay_write_idx - tap2_samples_r + DELAY_BUF_SAMPLES) % DELAY_BUF_SAMPLES;

        float cloud_l = 0.65f * st->delay_l[r1_l] + 0.45f * st->delay_r[r2_r];
        float cloud_r = 0.65f * st->delay_r[r1_r] + 0.45f * st->delay_l[r2_l];

        float fb = 0.30f + 0.46f * space_mix;
        // Warm low-pass damping in the feedback loop so echoes melt into a soft ambient pad
        st->delay_fb_lp_l += 0.16f * ((ambient_send_l + cloud_r * fb) - st->delay_fb_lp_l);
        st->delay_fb_lp_r += 0.16f * ((ambient_send_r + cloud_l * fb) - st->delay_fb_lp_r);

        st->delay_l[st->delay_write_idx] = std::clamp(st->delay_fb_lp_l, -1.2f, 1.2f);
        st->delay_r[st->delay_write_idx] = std::clamp(st->delay_fb_lp_r, -1.2f, 1.2f);
        st->delay_write_idx = (st->delay_write_idx + 1) % DELAY_BUF_SAMPLES;

        mix_l += cloud_l * (0.68f * space_mix);
        mix_r += cloud_r * (0.68f * space_mix);

        // Master Warmth Low-Pass Filter
        float master_cut = std::clamp(0.06f + 0.75f * filter_macro * filter_macro, 0.06f, 0.85f);
        st->master_lp_l += master_cut * (mix_l - st->master_lp_l);
        st->master_lp_r += master_cut * (mix_r - st->master_lp_r);

        if ((i & 7) == 0) {
            st->scope_ring[st->scope_write_idx] = 0.5f * (st->master_lp_l + st->master_lp_r);
            st->scope_write_idx = (st->scope_write_idx + 1) % SCOPE_SAMPLES;
        }

        auto softClip = [](float x) -> int16_t {
            float y = std::tanh(x * 0.85f);
            int32_t p = (int32_t)(y * 32760.0f);
            return (int16_t)std::clamp(p, -32768, 32767);
        };

        out[i * 2]     = softClip(st->master_lp_l);
        out[i * 2 + 1] = softClip(st->master_lp_r);
    }

    buf->mAudioDataByteSize = AUDIO_BUFFER_FRAMES * 4;
    if (st->running) {
        AudioQueueEnqueueBuffer(queue, buf, 0, nullptr);
    }
}

static bool initCoreAudio(AudioSynthState &st) {
    st.generateDrumSamples();

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
// VISUAL RIPPLE STRUCT
// =====================================================================

struct WaterRipple {
    float local_x;
    float local_y;
    double start_time;
};

// =====================================================================
// SWISS EDITORIAL AMBIENT VIEW (FINGERTIP KEYBOARD EDITION)
// =====================================================================

@interface BiomorphicEssentialView : NSView {
@public
    AudioSynthState *audio_;
    mediapipe_hands::HandTracker *handTracker_;
    mediapipe_hands::HandTrackingState latestHand_;

    std::array<uint32_t, mediapipe_hands::CAM_PREVIEW_W * mediapipe_hands::CAM_PREVIEW_H> camPixels_;
    CGColorSpaceRef colorSpace_;
    CGContextRef camBitmapCtx_;

    bool chordHold_;
    int lockedChord_;
    int activeSlider_;

    float leftLocalX_;
    float leftLocalY_;
    float leftVibrato_;
    float leftBrightness_;

    bool inGlitchBox_;
    bool mouseGlitchActive_;
    int glitchMode_;
    float glitchLocalX_;

    // Per-fingertip previous note column for smooth fingertip keyboard playing
    std::array<int, 5> prevTipNote_;
    std::array<double, NUM_SCALE_NOTES> lastNoteTriggerTime_;
    std::array<bool, NUM_SCALE_NOTES> activeKeyColumns_;

    float rightLocalX_;
    float rightLocalY_;
    int currentNoteIdx_;
    float currentIntensity_;

    std::vector<WaterRipple> ripples_;
    NSTimer *timer_;
}
- (instancetype)initWithFrame:(NSRect)frame
                        audio:(AudioSynthState *)audio
                  handTracker:(mediapipe_hands::HandTracker *)tracker;
@end

@implementation BiomorphicEssentialView

- (BOOL)isFlipped {
    return YES;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (instancetype)initWithFrame:(NSRect)frame
                        audio:(AudioSynthState *)audio
                  handTracker:(mediapipe_hands::HandTracker *)tracker {
    self = [super initWithFrame:frame];
    if (self) {
        [self setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];
        audio_ = audio;
        handTracker_ = tracker;
        chordHold_ = false;
        lockedChord_ = 0;
        activeSlider_ = 0;
        leftLocalX_ = 0.15f;
        leftLocalY_ = 0.65f;
        leftVibrato_ = 0.0f;
        leftBrightness_ = 0.65f;
        inGlitchBox_ = false;
        mouseGlitchActive_ = false;
        glitchMode_ = 0;
        glitchLocalX_ = 0.5f;
        prevTipNote_.fill(-1);
        lastNoteTriggerTime_.fill(0.0);
        activeKeyColumns_.fill(false);
        rightLocalX_ = 0.50f;
        rightLocalY_ = 0.65f;
        currentNoteIdx_ = 0;
        currentIntensity_ = 0.65f;

        camPixels_.fill(0xEAE7E0FFu);
        colorSpace_ = CGColorSpaceCreateDeviceRGB();
        CGBitmapInfo bmpInfo = (CGBitmapInfo)kCGImageAlphaNoneSkipLast | (CGBitmapInfo)kCGBitmapByteOrder32Big;
        camBitmapCtx_ = CGBitmapContextCreate(
            camPixels_.data(),
            mediapipe_hands::CAM_PREVIEW_W,
            mediapipe_hands::CAM_PREVIEW_H,
            8,
            mediapipe_hands::CAM_PREVIEW_W * 4,
            colorSpace_,
            bmpInfo
        );

        timer_ = [NSTimer scheduledTimerWithTimeInterval:(1.0 / 30.0)
                                                  target:self
                                                selector:@selector(onTick:)
                                                userInfo:nil
                                                 repeats:YES];
    }
    return self;
}

- (void)dealloc {
    if (timer_) {
        [timer_ invalidate];
        timer_ = nil;
    }
    if (camBitmapCtx_) {
        CGContextRelease(camBitmapCtx_);
        camBitmapCtx_ = nullptr;
    }
    if (colorSpace_) {
        CGColorSpaceRelease(colorSpace_);
        colorSpace_ = nullptr;
    }
}

- (void)triggerNoteIdx:(int)noteIdx localX:(float)lx localY:(float)ly {
    noteIdx = std::clamp(noteIdx, 0, NUM_SCALE_NOTES - 1);
    lx = std::clamp(lx, 0.0f, 1.0f);
    ly = std::clamp(ly, 0.15f, 1.0f);

    rightLocalX_ = lx;
    rightLocalY_ = ly;
    currentNoteIdx_ = noteIdx;
    currentIntensity_ = ly;

    double now = CACurrentMediaTime();
    if (now - lastNoteTriggerTime_[noteIdx] < 0.11) {
        return;
    }
    lastNoteTriggerTime_[noteIdx] = now;

    if (audio_) {
        std::lock_guard<std::mutex> lock(audio_->mtx);
        audio_->triggerAmbientNote(noteIdx, ly);
    }

    ripples_.push_back({(noteIdx + 0.5f) / (float)NUM_SCALE_NOTES, ly, now});
    if (ripples_.size() > 14) {
        ripples_.erase(ripples_.begin());
    }
}

- (void)onTick:(NSTimer *)timer {
    (void)timer;
    @autoreleasepool {
        if (handTracker_) {
            latestHand_ = handTracker_->getStateAndPixels(camPixels_.data());
        }

        double now = CACurrentMediaTime();

        // 1. LEFT QUADRANT BOX -> Ambient Chord (1..4 fingers) + Smooth-to-Vibrato (X) + Warmth (Y)
        int desiredChord = lockedChord_;
        if (latestHand_.left_hand.detected) {
            int fingers = std::clamp(latestHand_.left_hand.open_fingers, 0, 4);
            if (fingers >= 1) {
                desiredChord = fingers;
                if (chordHold_) lockedChord_ = fingers;
            } else if (!chordHold_) {
                desiredChord = 0;
            }

            float lx = (latestHand_.left_hand.norm_x - LEFT_BOX_X0) / (LEFT_BOX_X1 - LEFT_BOX_X0);
            float ly = (latestHand_.left_hand.norm_y - (1.0f - LEFT_BOX_Y1)) / (LEFT_BOX_Y1 - LEFT_BOX_Y0);
            leftLocalX_ = std::clamp(lx, 0.0f, 1.0f);
            leftLocalY_ = std::clamp(ly, 0.0f, 1.0f);
            leftVibrato_ = std::clamp((leftLocalX_ - 0.22f) / 0.78f, 0.0f, 1.0f);
            leftBrightness_ = std::clamp(leftLocalY_, 0.12f, 1.0f);
        } else if (!chordHold_) {
            desiredChord = 0;
            leftVibrato_ *= 0.92f;
        }

        // 2. RIGHT HAND -> Fingertips (polpastrelli) play the C Major Scale keys just by passing over them!
        bool handGlitch = false;
        std::array<float, NUM_SCALE_NOTES> newSustain{};
        newSustain.fill(0.0f);
        activeKeyColumns_.fill(false);

        if (latestHand_.right_hand.detected) {
            const auto &rh = latestHand_.right_hand;
            // Check if the index fingertip (8) or palm is inside the Top-Right Ambient Variation Box
            const auto &idxTip = rh.landmarks[mediapipe_hands::INDEX_FINGER_TIP];
            float top_y = (idxTip.confidence > 0.2f) ? idxTip.y : (1.0f - rh.norm_y);
            float top_x = (idxTip.confidence > 0.2f) ? idxTip.x : rh.norm_x;

            if (top_y >= GLITCH_BOX_Y0 - 0.02f && top_y <= GLITCH_BOX_Y1 + 0.02f &&
                top_x >= RIGHT_BOX_X0 - 0.02f && top_x <= RIGHT_BOX_X1 + 0.02f) {
                handGlitch = true;
                float gx = std::clamp((top_x - RIGHT_BOX_X0) / (RIGHT_BOX_X1 - RIGHT_BOX_X0), 0.0f, 1.0f);
                glitchLocalX_ = gx;
                glitchMode_ = std::clamp((int)(gx * 4.0f), 0, 3);
                prevTipNote_.fill(-1);
            } else {
                // Check all 5 fingertips (polpastrelli) passing over the 8 C-Major Scale columns!
                // To avoid 5 fingers triggering all at once when the hand is flat, we prioritize
                // the primary extended fingertips (Index #8 and Middle #12, plus Thumb #4 if in box)
                for (int f = 0; f < 2; ++f) { // Index fingertip (#8) & Middle fingertip (#12)
                    int lmIdx = FINGERTIP_INDICES[f];
                    const auto &tip = rh.landmarks[lmIdx];
                    if (tip.confidence <= 0.22f) {
                        prevTipNote_[f] = -1;
                        continue;
                    }

                    if (tip.x >= RIGHT_BOX_X0 && tip.x <= RIGHT_BOX_X1 &&
                        tip.y >= NOTES_BOX_Y0 && tip.y <= NOTES_BOX_Y1) {
                        float lx = std::clamp((tip.x - RIGHT_BOX_X0) / (RIGHT_BOX_X1 - RIGHT_BOX_X0), 0.0f, 1.0f);
                        float ly = std::clamp(1.0f - (tip.y - NOTES_BOX_Y0) / (NOTES_BOX_Y1 - NOTES_BOX_Y0), 0.15f, 1.0f);
                        int noteIdx = std::clamp((int)(lx * NUM_SCALE_NOTES), 0, NUM_SCALE_NOTES - 1);

                        activeKeyColumns_[noteIdx] = true;
                        newSustain[noteIdx] = std::max(newSustain[noteIdx], ly);

                        if (f == 0) {
                            rightLocalX_ = lx;
                            rightLocalY_ = ly;
                            currentNoteIdx_ = noteIdx;
                            currentIntensity_ = ly;
                        }

                        // Trigger delicate note when the fingertip enters the box or crosses into a new note column!
                        if (noteIdx != prevTipNote_[f]) {
                            [self triggerNoteIdx:noteIdx localX:lx localY:ly];
                            prevTipNote_[f] = noteIdx;
                        }
                    } else {
                        prevTipNote_[f] = -1;
                    }
                }
            }
        } else {
            prevTipNote_.fill(-1);
        }

        inGlitchBox_ = handGlitch || mouseGlitchActive_;

        if (audio_) {
            std::lock_guard<std::mutex> lock(audio_->mtx);
            audio_->setChord(desiredChord);
            audio_->target_chord_vibrato = leftVibrato_;
            audio_->target_chord_brightness = leftBrightness_;
            audio_->glitch_active = inGlitchBox_;
            audio_->glitch_mode = glitchMode_;
            audio_->glitch_x = glitchLocalX_;
            audio_->note_target_sustain = newSustain;
        }

        ripples_.erase(
            std::remove_if(ripples_.begin(), ripples_.end(),
                           [now](const WaterRipple &r) { return (now - r.start_time) > 0.90; }),
            ripples_.end());

        [self setNeedsDisplay:YES];
    }
}

// ---------------------------------------------------------------------
// DRAWING HELPERS
// ---------------------------------------------------------------------

- (void)drawDividerAtY:(CGFloat)y width:(CGFloat)w {
    [[NSColor colorWithSRGBRed:0.15 green:0.14 blue:0.13 alpha:1.0] setFill];
    NSRectFill(NSMakeRect(0, y, w, 1.0));
}

- (void)drawSectionHeader:(NSString *)title atY:(CGFloat)y width:(CGFloat)w {
    NSDictionary *attrs = @{
        NSFontAttributeName : [NSFont systemFontOfSize:10.5 weight:NSFontWeightMedium],
        NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.09 green:0.09 blue:0.08 alpha:1.0],
        NSKernAttributeName : @(1.1)
    };
    [title drawAtPoint:NSMakePoint(20, y) withAttributes:attrs];
    [@"–" drawAtPoint:NSMakePoint(w - 28, y - 1) withAttributes:attrs];
}

- (void)drawCheckboxLabel:(NSString *)label checked:(BOOL)checked atX:(CGFloat)x atY:(CGFloat)y {
    NSDictionary *lblAttrs = @{
        NSFontAttributeName : [NSFont systemFontOfSize:10.0 weight:NSFontWeightRegular],
        NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.46 green:0.44 blue:0.41 alpha:1.0],
        NSKernAttributeName : @(0.6)
    };
    [label drawAtPoint:NSMakePoint(x, y + 1) withAttributes:lblAttrs];

    NSRect box = NSMakeRect(x + 106, y, 15, 15);
    NSBezierPath *bp = [NSBezierPath bezierPathWithRoundedRect:box xRadius:3.0 yRadius:3.0];
    if (checked) {
        [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0] setFill];
        [bp fill];
        NSBezierPath *chk = [NSBezierPath bezierPath];
        [chk setLineWidth:1.7];
        [chk setLineCapStyle:NSLineCapStyleRound];
        [chk moveToPoint:NSMakePoint(box.origin.x + 3.5, box.origin.y + 8.0)];
        [chk lineToPoint:NSMakePoint(box.origin.x + 6.5, box.origin.y + 11.0)];
        [chk lineToPoint:NSMakePoint(box.origin.x + 11.5, box.origin.y + 4.5)];
        [[NSColor whiteColor] setStroke];
        [chk stroke];
    } else {
        [[NSColor colorWithSRGBRed:0.86 green:0.85 blue:0.82 alpha:1.0] setFill];
        [bp fill];
    }
}

- (void)drawCompactSlider:(NSString *)label
                normValue:(float)norm
                valueText:(NSString *)valText
                      atX:(CGFloat)x
                      atY:(CGFloat)y
                    width:(CGFloat)w {
    norm = std::clamp(norm, 0.0f, 1.0f);
    NSDictionary *lblAttrs = @{
        NSFontAttributeName : [NSFont systemFontOfSize:9.5 weight:NSFontWeightMedium],
        NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.42 green:0.40 blue:0.38 alpha:1.0],
        NSKernAttributeName : @(0.5)
    };
    [label drawAtPoint:NSMakePoint(x, y + 2) withAttributes:lblAttrs];

    CGFloat trackX = x + 64.0;
    CGFloat trackW = std::max(40.0, w - 104.0);
    CGFloat trackY = y + 5.0;
    CGFloat trackH = 6.0;

    NSBezierPath *bgPath = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(trackX, trackY, trackW, trackH) xRadius:3.0 yRadius:3.0];
    [[NSColor colorWithSRGBRed:0.86 green:0.85 blue:0.82 alpha:1.0] setFill];
    [bgPath fill];

    CGFloat activeW = std::max(6.0, (double)(norm * trackW));
    NSBezierPath *fgPath = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(trackX, trackY, activeW, trackH) xRadius:3.0 yRadius:3.0];
    [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0] setFill];
    [fgPath fill];

    CGFloat thumbW = 18.0;
    CGFloat thumbH = 12.0;
    CGFloat thumbX = trackX + norm * (trackW - thumbW);
    NSBezierPath *thumbPath = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(thumbX, y + 2.0, thumbW, thumbH) xRadius:6.0 yRadius:6.0];
    [[NSColor whiteColor] setFill];
    [thumbPath fill];
    [[NSColor colorWithSRGBRed:0.78 green:0.76 blue:0.73 alpha:1.0] setStroke];
    [thumbPath setLineWidth:1.0];
    [thumbPath stroke];

    NSDictionary *valAttrs = @{
        NSFontAttributeName : [NSFont monospacedDigitSystemFontOfSize:10.0 weight:NSFontWeightRegular],
        NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0]
    };
    NSSize valSize = [valText sizeWithAttributes:valAttrs];
    [valText drawAtPoint:NSMakePoint(x + w - valSize.width, y + 1.5) withAttributes:valAttrs];
}

- (NSRect)cameraRectForBounds:(NSRect)bounds {
    CGFloat margin = 20.0;
    CGFloat topY = 368.0;
    CGFloat bottomReserved = 56.0;
    CGFloat availW = std::max(260.0, bounds.size.width - margin * 2.0);
    CGFloat availH = std::max(200.0, bounds.size.height - topY - bottomReserved);
    return NSMakeRect(margin, topY, availW, availH);
}

- (NSRect)leftQuadrantRectInCamera:(NSRect)camRect {
    return NSMakeRect(
        camRect.origin.x + LEFT_BOX_X0 * camRect.size.width,
        camRect.origin.y + LEFT_BOX_Y0 * camRect.size.height,
        (LEFT_BOX_X1 - LEFT_BOX_X0) * camRect.size.width,
        (LEFT_BOX_Y1 - LEFT_BOX_Y0) * camRect.size.height
    );
}

- (NSRect)rightGlitchRectInCamera:(NSRect)camRect {
    return NSMakeRect(
        camRect.origin.x + RIGHT_BOX_X0 * camRect.size.width,
        camRect.origin.y + GLITCH_BOX_Y0 * camRect.size.height,
        (RIGHT_BOX_X1 - RIGHT_BOX_X0) * camRect.size.width,
        (GLITCH_BOX_Y1 - GLITCH_BOX_Y0) * camRect.size.height
    );
}

- (NSRect)rightQuadrantRectInCamera:(NSRect)camRect {
    return NSMakeRect(
        camRect.origin.x + RIGHT_BOX_X0 * camRect.size.width,
        camRect.origin.y + NOTES_BOX_Y0 * camRect.size.height,
        (RIGHT_BOX_X1 - RIGHT_BOX_X0) * camRect.size.width,
        (NOTES_BOX_Y1 - NOTES_BOX_Y0) * camRect.size.height
    );
}

- (void)drawHandSkeleton:(const mediapipe_hands::SingleHandState &)hand
                  inRect:(NSRect)camRect
                  isLeft:(BOOL)isLeft {
    if (!hand.detected) return;

    NSBezierPath *bones = [NSBezierPath bezierPath];
    [bones setLineWidth:1.4];
    for (const auto &edge : mediapipe_hands::HAND_CONNECTIONS) {
        const auto &p0 = hand.landmarks[edge.first];
        const auto &p1 = hand.landmarks[edge.second];
        if (p0.confidence > 0.2f && p1.confidence > 0.2f) {
            CGFloat x0 = camRect.origin.x + p0.x * camRect.size.width;
            CGFloat y0 = camRect.origin.y + p0.y * camRect.size.height;
            CGFloat x1 = camRect.origin.x + p1.x * camRect.size.width;
            CGFloat y1 = camRect.origin.y + p1.y * camRect.size.height;
            [bones moveToPoint:NSMakePoint(x0, y0)];
            [bones lineToPoint:NSMakePoint(x1, y1)];
        }
    }
    [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.85] setStroke];
    [bones stroke];

    for (int j = 0; j < mediapipe_hands::NUM_LANDMARKS; ++j) {
        const auto &lm = hand.landmarks[j];
        if (lm.confidence > 0.2f) {
            CGFloat jx = camRect.origin.x + lm.x * camRect.size.width;
            CGFloat jy = camRect.origin.y + lm.y * camRect.size.height;
            bool isActiveFingertip = (!isLeft) &&
                (j == mediapipe_hands::INDEX_FINGER_TIP || j == mediapipe_hands::MIDDLE_FINGER_TIP);
            if (isActiveFingertip) {
                // Highlight the playing fingertips (polpastrelli) with a double ring
                NSRect outerR = NSMakeRect(jx - 6.0, jy - 6.0, 12.0, 12.0);
                NSBezierPath *op = [NSBezierPath bezierPathWithOvalInRect:outerR];
                [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.95] setFill];
                [op fill];
                [[NSColor whiteColor] setStroke];
                [op setLineWidth:1.6];
                [op stroke];
            } else {
                NSRect nRect = NSMakeRect(jx - 2.8, jy - 2.8, 5.6, 5.6);
                NSBezierPath *np = [NSBezierPath bezierPathWithOvalInRect:nRect];
                if (isLeft) {
                    [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0] setFill];
                    [np fill];
                } else {
                    [[NSColor whiteColor] setFill];
                    [np fill];
                    [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0] setStroke];
                    [np setLineWidth:1.1];
                    [np stroke];
                }
            }
        }
    }
}

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    @autoreleasepool {
        NSRect bounds = [self bounds];
        CGFloat W = bounds.size.width;
        double now = CACurrentMediaTime();

        // Warm paper background (#F3F1EC)
        [[NSColor colorWithSRGBRed:0.953 green:0.945 blue:0.925 alpha:1.0] setFill];
        NSRectFill(bounds);

        bool isPlaying = true;
        int bpm = 84;
        float swing = 0.08f;
        float filtMacro = 0.72f;
        float decayMacro = 0.68f;
        float spaceMacro = 0.56f;
        int currentStep = 0;
        int activeChord = 0;
        std::array<std::array<int, 16>, NUM_DRUM_TRACKS> grid{};
        std::array<bool, NUM_DRUM_TRACKS> drumMuted{};
        std::array<float, SCOPE_SAMPLES> scope{};
        int scopeIdx = 0;

        if (audio_) {
            std::lock_guard<std::mutex> lock(audio_->mtx);
            isPlaying = audio_->is_playing;
            bpm = audio_->bpm;
            swing = audio_->swing;
            filtMacro = audio_->master_filter;
            decayMacro = audio_->master_decay;
            spaceMacro = audio_->master_space;
            currentStep = audio_->displayed_step;
            activeChord = audio_->active_chord;
            grid = audio_->grid;
            drumMuted = audio_->drum_muted;
            scope = audio_->scope_ring;
            scopeIdx = audio_->scope_write_idx;
        }

        // =============================================================
        // HEADER + REAL-TIME OSCILLOSCOPE
        // =============================================================
        NSDictionary *titleAttrs = @{
            NSFontAttributeName : [NSFont systemFontOfSize:18.0 weight:NSFontWeightBold],
            NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.09 green:0.09 blue:0.08 alpha:1.0],
            NSKernAttributeName : @(-0.4)
        };
        [@"BIOMORPHIC" drawAtPoint:NSMakePoint(20, 14) withAttributes:titleAttrs];

        NSFont *italicSerif = [NSFont fontWithName:@"TimesNewRomanPS-ItalicMT" size:18.0];
        if (!italicSerif) italicSerif = [NSFont systemFontOfSize:17.0 weight:NSFontWeightLight];
        [@"(sintesi)" drawAtPoint:NSMakePoint(145, 14) withAttributes:@{
            NSFontAttributeName : italicSerif,
            NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.09 green:0.09 blue:0.08 alpha:1.0]
        }];

        NSRect scopeRect = NSMakeRect(230, 14, std::max(80.0, W - 290.0), 24.0);
        NSBezierPath *scopePath = [NSBezierPath bezierPath];
        [scopePath setLineWidth:1.2];
        for (int s = 0; s < SCOPE_SAMPLES; ++s) {
            int idx = (scopeIdx + s) % SCOPE_SAMPLES;
            float val = std::clamp(scope[idx] * 1.6f, -1.0f, 1.0f);
            CGFloat sx = scopeRect.origin.x + (scopeRect.size.width * s) / (CGFloat)(SCOPE_SAMPLES - 1);
            CGFloat sy = scopeRect.origin.y + scopeRect.size.height * (0.5 - 0.45 * val);
            if (s == 0) [scopePath moveToPoint:NSMakePoint(sx, sy)];
            else [scopePath lineToPoint:NSMakePoint(sx, sy)];
        }
        [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.75] setStroke];
        [scopePath stroke];

        NSRect closeBtn = NSMakeRect(W - 46, 13, 26, 26);
        [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0] setFill];
        NSRectFill(closeBtn);
        [@"×" drawAtPoint:NSMakePoint(closeBtn.origin.x + 8.5, closeBtn.origin.y + 3.5) withAttributes:@{
            NSFontAttributeName : [NSFont systemFontOfSize:14.0 weight:NSFontWeightMedium],
            NSForegroundColorAttributeName : [NSColor whiteColor]
        }];

        const char *chordLbl = (activeChord >= 1 && activeChord <= 4) ? CHORDS[activeChord - 1].name : "NESSUNO";
        const char *glitchStatus = inGlitchBox_ ? GLITCH_MODE_NAMES[glitchMode_] : "OFF";
        NSString *subText = [NSString stringWithFormat:@"SX: %s (VIB %d%%)  ·  ATMOSFERA: %s  ·  POLPASTRELLI DX: %s",
                             chordLbl,
                             (int)std::round(leftVibrato_ * 100.0f),
                             glitchStatus,
                             QUADRANT_NOTES[currentNoteIdx_].name];
        [subText drawAtPoint:NSMakePoint(20, 42) withAttributes:@{
            NSFontAttributeName : [NSFont systemFontOfSize:10.0 weight:NSFontWeightRegular],
            NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.46 green:0.44 blue:0.41 alpha:1.0],
            NSKernAttributeName : @(0.5)
        }];

        [self drawDividerAtY:64 width:W];

        // =============================================================
        // 001 — RITMO MICRO-AMBIENT (4 VOCI DELICATE)
        // =============================================================
        [self drawSectionHeader:@"001 — RITMO MICRO-AMBIENT" atY:74 width:W];
        [self drawCheckboxLabel:@"IN MOVIMENTO" checked:isPlaying atX:20 atY:98];

        CGFloat halfW = (W - 56.0) * 0.5;
        float bpmNorm = (bpm - 50.0f) / 90.0f;
        [self drawCompactSlider:@"BPM"
                      normValue:bpmNorm
                      valueText:[NSString stringWithFormat:@"%d", bpm]
                            atX:165
                            atY:98
                          width:halfW - 75.0];

        [self drawCompactSlider:@"SWING"
                      normValue:(swing / 0.50f)
                      valueText:[NSString stringWithFormat:@"%d%%", (int)std::round(swing * 100.0f)]
                            atX:165 + halfW - 60.0
                            atY:98
                          width:W - (185 + halfW - 60.0)];

        CGFloat gridX = 88.0;
        CGFloat gridW = W - 108.0;
        CGFloat gap = 3.0;
        CGFloat cellW = (gridW - gap * 15.0) / 16.0;
        CGFloat rowH = 14.0;
        CGFloat rowGap = 4.0;
        CGFloat gridTopY = 122.0;

        for (int tr = 0; tr < NUM_DRUM_TRACKS; ++tr) {
            CGFloat ry = gridTopY + tr * (rowH + rowGap);
            bool muted = drumMuted[tr];

            NSString *trName = [NSString stringWithUTF8String:DRUM_TRACK_NAMES[tr]];
            [trName drawAtPoint:NSMakePoint(20, ry + 0.5) withAttributes:@{
                NSFontAttributeName : [NSFont systemFontOfSize:9.0 weight:NSFontWeightMedium],
                NSForegroundColorAttributeName : muted
                    ? [NSColor colorWithSRGBRed:0.68 green:0.66 blue:0.63 alpha:1.0]
                    : [NSColor colorWithSRGBRed:0.12 green:0.11 blue:0.10 alpha:1.0],
                NSKernAttributeName : @(0.5)
            }];

            NSRect mDot = NSMakeRect(72, ry + 4.0, 6.0, 6.0);
            [[NSColor colorWithSRGBRed:(muted ? 0.80 : 0.12)
                                 green:(muted ? 0.78 : 0.11)
                                  blue:(muted ? 0.75 : 0.10)
                                 alpha:1.0] setFill];
            [[NSBezierPath bezierPathWithOvalInRect:mDot] fill];

            for (int s = 0; s < 16; ++s) {
                CGFloat cx = gridX + s * (cellW + gap);
                NSBezierPath *cp = [NSBezierPath bezierPathWithRoundedRect:NSMakeRect(cx, ry, cellW, rowH)
                                                                   xRadius:2.0
                                                                   yRadius:2.0];
                if (grid[tr][s]) {
                    [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:(muted ? 0.30 : 1.0)] setFill];
                } else {
                    [[NSColor colorWithSRGBRed:(s % 4 == 0 ? 0.82 : 0.87)
                                         green:(s % 4 == 0 ? 0.80 : 0.86)
                                          blue:(s % 4 == 0 ? 0.77 : 0.83)
                                         alpha:1.0] setFill];
                }
                [cp fill];
            }
        }

        if (isPlaying) {
            CGFloat cx = gridX + currentStep * (cellW + gap);
            NSRect dotR = NSMakeRect(cx + cellW * 0.5 - 2.0, gridTopY + NUM_DRUM_TRACKS * (rowH + rowGap) + 1.0, 4.0, 4.0);
            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0] setFill];
            [[NSBezierPath bezierPathWithOvalInRect:dotR] fill];
        }

        [self drawDividerAtY:204 width:W];

        // =============================================================
        // 002 — TIMBRO E SPAZIO AMBIENT (TEMPO REALE)
        // =============================================================
        [self drawSectionHeader:@"002 — TIMBRO E SPAZIO AMBIENT" atY:214 width:W];

        CGFloat col3W = (W - 64.0) / 3.0;
        [self drawCompactSlider:@"CALORE"
                      normValue:filtMacro
                      valueText:[NSString stringWithFormat:@"%d%%", (int)std::round(filtMacro * 100.0f)]
                            atX:20.0
                            atY:238.0
                          width:col3W];

        [self drawCompactSlider:@"DECAY"
                      normValue:decayMacro
                      valueText:[NSString stringWithFormat:@"%d%%", (int)std::round(decayMacro * 100.0f)]
                            atX:32.0 + col3W
                            atY:238.0
                          width:col3W];

        [self drawCompactSlider:@"SPAZIO"
                      normValue:spaceMacro
                      valueText:[NSString stringWithFormat:@"%d%%", (int)std::round(spaceMacro * 100.0f)]
                            atX:44.0 + col3W * 2.0
                            atY:238.0
                          width:col3W];

        [self drawDividerAtY:266 width:W];

        // =============================================================
        // 003 — MANI (SX: ACCORDI AMBIENT · DX: POLPASTRELLI SU SCALA DO)
        // =============================================================
        [self drawSectionHeader:@"003 — MANI (SX: ACCORDI · DX: POLPASTRELLI SU SCALA DO)" atY:276 width:W];

        const char *chordTags[4] = {"1 · DO MAJ9", "2 · SOL ADD9", "3 · LA MIN9", "4 · MI MIN9"};
        CGFloat pillGap = 6.0;
        CGFloat pillW = (W - 40.0 - pillGap * 3.0) / 4.0;
        for (int c = 0; c < 4; ++c) {
            NSRect pRect = NSMakeRect(20.0 + c * (pillW + pillGap), 300.0, pillW, 24.0);
            bool isAct = (activeChord == c + 1);
            [[NSColor colorWithSRGBRed:(isAct ? 0.10 : 0.86)
                                 green:(isAct ? 0.09 : 0.85)
                                  blue:(isAct ? 0.08 : 0.82)
                                 alpha:1.0] setFill];
            NSRectFill(pRect);

            NSString *tStr = [NSString stringWithUTF8String:chordTags[c]];
            NSDictionary *tAttrs = @{
                NSFontAttributeName : [NSFont systemFontOfSize:10.0 weight:NSFontWeightMedium],
                NSForegroundColorAttributeName : isAct ? [NSColor whiteColor]
                                                       : [NSColor colorWithSRGBRed:0.25 green:0.24 blue:0.22 alpha:1.0],
                NSKernAttributeName : @(0.5)
            };
            NSSize tSz = [tStr sizeWithAttributes:tAttrs];
            [tStr drawAtPoint:NSMakePoint(pRect.origin.x + (pillW - tSz.width) * 0.5, pRect.origin.y + 5.0)
               withAttributes:tAttrs];
        }

        for (int g = 0; g < 4; ++g) {
            NSRect gRect = NSMakeRect(20.0 + g * (pillW + pillGap), 330.0, pillW, 22.0);
            bool isGAct = inGlitchBox_ && (glitchMode_ == g);
            [[NSColor colorWithSRGBRed:(isGAct ? 0.10 : 0.90)
                                 green:(isGAct ? 0.09 : 0.89)
                                  blue:(isGAct ? 0.08 : 0.86)
                                 alpha:1.0] setFill];
            NSRectFill(gRect);
            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.35] setStroke];
            [[NSBezierPath bezierPathWithRect:gRect] stroke];

            NSString *gStr = [NSString stringWithFormat:@"%s", GLITCH_MODE_NAMES[g]];
            NSDictionary *gAttrs = @{
                NSFontAttributeName : [NSFont systemFontOfSize:9.0 weight:NSFontWeightMedium],
                NSForegroundColorAttributeName : isGAct ? [NSColor whiteColor]
                                                        : [NSColor colorWithSRGBRed:0.20 green:0.19 blue:0.18 alpha:1.0],
                NSKernAttributeName : @(0.5)
            };
            NSSize gSz = [gStr sizeWithAttributes:gAttrs];
            [gStr drawAtPoint:NSMakePoint(gRect.origin.x + (pillW - gSz.width) * 0.5, gRect.origin.y + 4.5)
               withAttributes:gAttrs];
        }

        NSRect camRect = [self cameraRectForBounds:bounds];
        NSRect leftBox = [self leftQuadrantRectInCamera:camRect];
        NSRect glitchBox = [self rightGlitchRectInCamera:camRect];
        NSRect rightBox = [self rightQuadrantRectInCamera:camRect];

        if (latestHand_.camera_ready && camBitmapCtx_) {
            CGImageRef img = CGBitmapContextCreateImage(camBitmapCtx_);
            if (img) {
                CGContextRef ctx = [[NSGraphicsContext currentContext] CGContext];
                CGContextSaveGState(ctx);
                CGContextTranslateCTM(ctx, camRect.origin.x, camRect.origin.y + camRect.size.height);
                CGContextScaleCTM(ctx, 1.0, -1.0);
                CGContextDrawImage(ctx, CGRectMake(0, 0, camRect.size.width, camRect.size.height), img);
                CGContextRestoreGState(ctx);
                CGImageRelease(img);
            }
        } else {
            [[NSColor colorWithSRGBRed:0.92 green:0.91 blue:0.88 alpha:1.0] setFill];
            NSRectFill(camRect);
        }

        [[NSColor colorWithSRGBRed:0.15 green:0.14 blue:0.13 alpha:1.0] setStroke];
        [[NSBezierPath bezierPathWithRect:camRect] stroke];

        // 2. LEFT BOX ("MANO SX · ACCORDI": LISCIO -> VIBRATO DOLCE)
        {
            bool hasLeft = latestHand_.left_hand.detected;

            CGFloat midLeftX = leftBox.origin.x + leftBox.size.width * 0.35;
            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.22] setStroke];
            [NSBezierPath strokeLineFromPoint:NSMakePoint(midLeftX, leftBox.origin.y)
                                      toPoint:NSMakePoint(midLeftX, leftBox.origin.y + leftBox.size.height)];

            NSDictionary *zoneAttrs = @{
                NSFontAttributeName : [NSFont systemFontOfSize:8.5 weight:NSFontWeightMedium],
                NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.65],
                NSKernAttributeName : @(0.5)
            };
            [@"← LISCIO" drawAtPoint:NSMakePoint(leftBox.origin.x + 8, leftBox.origin.y + 22) withAttributes:zoneAttrs];
            [@"VIBRATO →" drawAtPoint:NSMakePoint(leftBox.origin.x + leftBox.size.width - 64, leftBox.origin.y + 22) withAttributes:zoneAttrs];

            NSBezierPath *lbp = [NSBezierPath bezierPathWithRect:leftBox];
            [lbp setLineWidth:hasLeft ? 2.0 : 1.2];
            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:(hasLeft ? 0.95 : 0.65)] setStroke];
            [lbp stroke];

            NSRect tagR = NSMakeRect(leftBox.origin.x, leftBox.origin.y, 136, 18);
            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.88] setFill];
            NSRectFill(tagR);
            [@"MANO SX · ACCORDI" drawAtPoint:NSMakePoint(tagR.origin.x + 8, tagR.origin.y + 2.5) withAttributes:@{
                NSFontAttributeName : [NSFont systemFontOfSize:9.0 weight:NSFontWeightMedium],
                NSForegroundColorAttributeName : [NSColor whiteColor],
                NSKernAttributeName : @(0.6)
            }];

            int vibPct = (int)std::round(leftVibrato_ * 100.0f);
            NSString *sxInfo = hasLeft
                ? [NSString stringWithFormat:@"%s · %s",
                   chordLbl,
                   (vibPct < 5) ? "LISCIO" : [[NSString stringWithFormat:@"VIB %d%%", vibPct] UTF8String]]
                : @"SX: LISCIO · DX: VIBRATO";
            NSRect botR = NSMakeRect(leftBox.origin.x + 6, leftBox.origin.y + leftBox.size.height - 22, 150, 17);
            [[NSColor colorWithSRGBRed:0.95 green:0.94 blue:0.92 alpha:0.88] setFill];
            NSRectFill(botR);
            [sxInfo drawAtPoint:NSMakePoint(botR.origin.x + 6, botR.origin.y + 2.0) withAttributes:@{
                NSFontAttributeName : [NSFont systemFontOfSize:8.5 weight:NSFontWeightMedium],
                NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0],
                NSKernAttributeName : @(0.4)
            }];

            if (hasLeft) {
                CGFloat lx = leftBox.origin.x + leftLocalX_ * leftBox.size.width;
                CGFloat ly = leftBox.origin.y + (1.0f - leftLocalY_) * leftBox.size.height;
                NSBezierPath *lDot = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(lx - 5.5, ly - 5.5, 11.0, 11.0)];
                [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0] setFill];
                [lDot fill];
                [[NSColor whiteColor] setStroke];
                [lDot setLineWidth:1.5];
                [lDot stroke];
            }
        }

        // 3A. TOP-RIGHT AMBIENT VARIATIONS BOX ("DX · ATMOSFERA ARMONIA")
        {
            for (int g = 0; g < 4; ++g) {
                CGFloat cw = glitchBox.size.width / 4.0;
                NSRect cellR = NSMakeRect(glitchBox.origin.x + g * cw, glitchBox.origin.y, cw, glitchBox.size.height);
                bool activeCell = inGlitchBox_ && (glitchMode_ == g);
                if (activeCell) {
                    [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.82] setFill];
                    NSRectFill(cellR);
                } else {
                    [[NSColor colorWithSRGBRed:0.95 green:0.94 blue:0.92 alpha:0.35] setFill];
                    NSRectFill(cellR);
                }
                if (g > 0) {
                    [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.30] setStroke];
                    [NSBezierPath strokeLineFromPoint:NSMakePoint(cellR.origin.x, glitchBox.origin.y)
                                              toPoint:NSMakePoint(cellR.origin.x, glitchBox.origin.y + glitchBox.size.height)];
                }

                NSString *gName = [NSString stringWithUTF8String:GLITCH_MODE_NAMES[g]];
                NSDictionary *gAttrs = @{
                    NSFontAttributeName : [NSFont systemFontOfSize:8.0 weight:NSFontWeightBold],
                    NSForegroundColorAttributeName : activeCell
                        ? [NSColor whiteColor]
                        : [NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.85],
                    NSKernAttributeName : @(0.3)
                };
                NSSize gSz = [gName sizeWithAttributes:gAttrs];
                [gName drawAtPoint:NSMakePoint(cellR.origin.x + (cw - gSz.width) * 0.5,
                                               glitchBox.origin.y + glitchBox.size.height - 16.0)
                    withAttributes:gAttrs];
            }

            NSBezierPath *gbp = [NSBezierPath bezierPathWithRect:glitchBox];
            [gbp setLineWidth:inGlitchBox_ ? 2.2 : 1.2];
            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:(inGlitchBox_ ? 1.0 : 0.72)] setStroke];
            [gbp stroke];

            NSRect gTagR = NSMakeRect(glitchBox.origin.x, glitchBox.origin.y, 148, 17);
            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.90] setFill];
            NSRectFill(gTagR);
            [@"DX · ATMOSFERA ARMONIA" drawAtPoint:NSMakePoint(gTagR.origin.x + 7, gTagR.origin.y + 2.0) withAttributes:@{
                NSFontAttributeName : [NSFont systemFontOfSize:8.5 weight:NSFontWeightMedium],
                NSForegroundColorAttributeName : [NSColor whiteColor],
                NSKernAttributeName : @(0.5)
            }];
        }

        // 3B. BOTTOM-RIGHT FINGERTIP KEYBOARD ("MANO DX · TASTIERA SCALA DO")
        {
            bool hasRight = latestHand_.right_hand.detected && !inGlitchBox_;
            CGFloat colW = rightBox.size.width / (CGFloat)NUM_SCALE_NOTES;

            for (int n = 0; n < NUM_SCALE_NOTES; ++n) {
                CGFloat nx = rightBox.origin.x + colW * n;
                if (activeKeyColumns_[n]) {
                    [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.25] setFill];
                    NSRectFill(NSMakeRect(nx, rightBox.origin.y, colW, rightBox.size.height));
                }
                if (n > 0) {
                    [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.22] setStroke];
                    [NSBezierPath strokeLineFromPoint:NSMakePoint(nx, rightBox.origin.y)
                                              toPoint:NSMakePoint(nx, rightBox.origin.y + rightBox.size.height)];
                }

                NSString *nStr = [NSString stringWithUTF8String:QUADRANT_NOTES[n].short_name];
                NSDictionary *nAttrs = @{
                    NSFontAttributeName : [NSFont systemFontOfSize:8.5 weight:NSFontWeightMedium],
                    NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.80]
                };
                NSSize nSz = [nStr sizeWithAttributes:nAttrs];
                [nStr drawAtPoint:NSMakePoint(nx + (colW - nSz.width) * 0.5, rightBox.origin.y + 20.0)
                   withAttributes:nAttrs];
            }

            NSBezierPath *rbp = [NSBezierPath bezierPathWithRect:rightBox];
            [rbp setLineWidth:hasRight ? 2.0 : 1.2];
            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:(hasRight ? 0.95 : 0.65)] setStroke];
            [rbp stroke];

            NSRect tagR = NSMakeRect(rightBox.origin.x, rightBox.origin.y, 152, 17);
            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:0.88] setFill];
            NSRectFill(tagR);
            [@"DX · POLPASTRELLI SCALA DO" drawAtPoint:NSMakePoint(tagR.origin.x + 7, tagR.origin.y + 2.0) withAttributes:@{
                NSFontAttributeName : [NSFont systemFontOfSize:8.5 weight:NSFontWeightMedium],
                NSForegroundColorAttributeName : [NSColor whiteColor],
                NSKernAttributeName : @(0.5)
            }];

            NSString *dxInfo = [NSString stringWithFormat:@"%s · %d%%",
                                QUADRANT_NOTES[currentNoteIdx_].name,
                                (int)std::round(currentIntensity_ * 100.0f)];
            NSRect botR = NSMakeRect(rightBox.origin.x + 6, rightBox.origin.y + rightBox.size.height - 21, 108, 16);
            [[NSColor colorWithSRGBRed:0.95 green:0.94 blue:0.92 alpha:0.88] setFill];
            NSRectFill(botR);
            [dxInfo drawAtPoint:NSMakePoint(botR.origin.x + 6, botR.origin.y + 1.5) withAttributes:@{
                NSFontAttributeName : [NSFont systemFontOfSize:8.5 weight:NSFontWeightMedium],
                NSForegroundColorAttributeName : [NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0],
                NSKernAttributeName : @(0.4)
            }];
        }

        // 4. Draw Left & Right Hand Skeletons
        [self drawHandSkeleton:latestHand_.left_hand inRect:camRect isLeft:YES];
        [self drawHandSkeleton:latestHand_.right_hand inRect:camRect isLeft:NO];

        // 5. Delicate Expanding Ripples inside Right Box
        for (const auto &rp : ripples_) {
            double age = now - rp.start_time;
            float prog = (float)std::clamp(age / 0.90, 0.0, 1.0);
            CGFloat rx = rightBox.origin.x + rp.local_x * rightBox.size.width;
            CGFloat ry = rightBox.origin.y + (1.0f - rp.local_y) * rightBox.size.height;

            CGFloat rad1 = 6.0 + prog * 46.0;
            CGFloat rad2 = 3.0 + prog * 24.0;
            CGFloat alpha = (1.0 - prog) * 0.75;

            [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:alpha] setStroke];
            NSBezierPath *c1 = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(rx - rad1, ry - rad1, rad1 * 2.0, rad1 * 2.0)];
            [c1 setLineWidth:1.3];
            [c1 stroke];

            NSBezierPath *c2 = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(rx - rad2, ry - rad2, rad2 * 2.0, rad2 * 2.0)];
            [c2 setLineWidth:0.9];
            [c2 stroke];
        }

        // =============================================================
        // FOOTER BAR ("BLOCCA ACCORDO", "SVUOTA", "RIGENERA")
        // =============================================================
        CGFloat footY = camRect.origin.y + camRect.size.height + 14.0;
        [self drawCheckboxLabel:@"BLOCCA ACCORDO" checked:chordHold_ atX:20 atY:footY + 6];

        NSRect clearBtn = NSMakeRect(W - 226, footY, 94, 28);
        NSRect regenBtn = NSMakeRect(W - 124, footY, 104, 28);

        NSDictionary *btnTxtAttrs = @{
            NSFontAttributeName : [NSFont systemFontOfSize:10.0 weight:NSFontWeightMedium],
            NSForegroundColorAttributeName : [NSColor whiteColor],
            NSKernAttributeName : @(0.8)
        };

        [[NSColor colorWithSRGBRed:0.66 green:0.64 blue:0.61 alpha:1.0] setFill];
        NSRectFill(clearBtn);
        [@"SVUOTA" drawAtPoint:NSMakePoint(clearBtn.origin.x + 22, clearBtn.origin.y + 7) withAttributes:btnTxtAttrs];

        [[NSColor colorWithSRGBRed:0.10 green:0.09 blue:0.08 alpha:1.0] setFill];
        NSRectFill(regenBtn);
        [@"RIGENERA" drawAtPoint:NSMakePoint(regenBtn.origin.x + 20, regenBtn.origin.y + 7) withAttributes:btnTxtAttrs];
    }
}

// ---------------------------------------------------------------------
// INTERACTION
// ---------------------------------------------------------------------

- (void)applySliderDragAtX:(CGFloat)mx width:(CGFloat)W {
    if (!audio_ || activeSlider_ == 0) return;
    CGFloat halfW = (W - 56.0) * 0.5;
    CGFloat col3W = (W - 64.0) / 3.0;

    std::lock_guard<std::mutex> lock(audio_->mtx);
    if (activeSlider_ == 1) {
        CGFloat trackX = 165.0 + 64.0;
        CGFloat trackW = std::max(40.0, (halfW - 75.0) - 104.0);
        float norm = std::clamp((float)((mx - trackX) / trackW), 0.0f, 1.0f);
        audio_->bpm = (int)std::round(50.0f + norm * 90.0f);
    } else if (activeSlider_ == 2) {
        CGFloat x0 = 165.0 + halfW - 60.0;
        CGFloat w0 = W - (185.0 + halfW - 60.0);
        CGFloat trackX = x0 + 64.0;
        CGFloat trackW = std::max(40.0, w0 - 104.0);
        float norm = std::clamp((float)((mx - trackX) / trackW), 0.0f, 1.0f);
        audio_->swing = norm * 0.50f;
    } else if (activeSlider_ == 3) {
        CGFloat trackX = 20.0 + 64.0;
        CGFloat trackW = std::max(40.0, col3W - 104.0);
        audio_->master_filter = std::clamp((float)((mx - trackX) / trackW), 0.12f, 1.0f);
    } else if (activeSlider_ == 4) {
        CGFloat trackX = 32.0 + col3W + 64.0;
        CGFloat trackW = std::max(40.0, col3W - 104.0);
        audio_->master_decay = std::clamp((float)((mx - trackX) / trackW), 0.10f, 1.0f);
    } else if (activeSlider_ == 5) {
        CGFloat trackX = 44.0 + col3W * 2.0 + 64.0;
        CGFloat trackW = std::max(40.0, col3W - 104.0);
        audio_->master_space = std::clamp((float)((mx - trackX) / trackW), 0.0f, 1.0f);
    }
}

- (void)mouseDown:(NSEvent *)event {
    NSPoint pt = [self convertPoint:[event locationInWindow] fromView:nil];
    NSRect bounds = [self bounds];
    CGFloat W = bounds.size.width;

    if (NSPointInRect(pt, NSMakeRect(W - 46, 13, 26, 26))) {
        [NSApp terminate:nil];
        return;
    }

    if (NSPointInRect(pt, NSMakeRect(20, 94, 135, 22))) {
        std::lock_guard<std::mutex> lock(audio_->mtx);
        audio_->is_playing = !audio_->is_playing;
        audio_->current_step = 0;
        audio_->step_sample_counter = 0.0;
        return;
    }

    CGFloat halfW = (W - 56.0) * 0.5;
    if (pt.y >= 94 && pt.y <= 118 && pt.x >= 165) {
        if (pt.x < 165 + halfW - 65.0) activeSlider_ = 1;
        else activeSlider_ = 2;
        [self applySliderDragAtX:pt.x width:W];
        return;
    }

    CGFloat gridX = 88.0;
    CGFloat gridW = W - 108.0;
    CGFloat gap = 3.0;
    CGFloat cellW = (gridW - gap * 15.0) / 16.0;
    CGFloat rowH = 14.0;
    CGFloat rowGap = 4.0;
    CGFloat gridTopY = 122.0;

    for (int tr = 0; tr < NUM_DRUM_TRACKS; ++tr) {
        CGFloat ry = gridTopY + tr * (rowH + rowGap);
        if (pt.y >= ry - 1.0 && pt.y <= ry + rowH + 1.0) {
            if (pt.x >= 18.0 && pt.x < gridX - 4.0) {
                std::lock_guard<std::mutex> lock(audio_->mtx);
                audio_->drum_muted[tr] = !audio_->drum_muted[tr];
                return;
            }
            if (pt.x >= gridX && pt.x <= gridX + gridW) {
                int s = (int)((pt.x - gridX) / (cellW + gap));
                if (s >= 0 && s < 16) {
                    std::lock_guard<std::mutex> lock(audio_->mtx);
                    audio_->grid[tr][s] = audio_->grid[tr][s] ? 0 : 1;
                    if (audio_->grid[tr][s]) audio_->triggerDrum(tr, 1.0f);
                }
                return;
            }
        }
    }

    CGFloat col3W = (W - 64.0) / 3.0;
    if (pt.y >= 232 && pt.y <= 260) {
        if (pt.x < 28.0 + col3W) activeSlider_ = 3;
        else if (pt.x < 40.0 + col3W * 2.0) activeSlider_ = 4;
        else activeSlider_ = 5;
        [self applySliderDragAtX:pt.x width:W];
        return;
    }

    CGFloat pillGap = 6.0;
    CGFloat pillW = (W - 40.0 - pillGap * 3.0) / 4.0;
    for (int c = 0; c < 4; ++c) {
        NSRect pRect = NSMakeRect(20.0 + c * (pillW + pillGap), 300.0, pillW, 24.0);
        if (NSPointInRect(pt, pRect)) {
            if (lockedChord_ == c + 1 && chordHold_) {
                lockedChord_ = 0;
                chordHold_ = false;
            } else {
                lockedChord_ = c + 1;
                chordHold_ = true;
            }
            return;
        }
    }

    for (int g = 0; g < 4; ++g) {
        NSRect gRect = NSMakeRect(20.0 + g * (pillW + pillGap), 330.0, pillW, 22.0);
        if (NSPointInRect(pt, gRect)) {
            if (mouseGlitchActive_ && glitchMode_ == g) {
                mouseGlitchActive_ = false;
            } else {
                mouseGlitchActive_ = true;
                glitchMode_ = g;
                glitchLocalX_ = (g + 0.5f) / 4.0f;
            }
            return;
        }
    }

    NSRect camRect = [self cameraRectForBounds:bounds];
    NSRect leftBox = [self leftQuadrantRectInCamera:camRect];
    NSRect glitchBox = [self rightGlitchRectInCamera:camRect];
    NSRect rightBox = [self rightQuadrantRectInCamera:camRect];

    if (NSPointInRect(pt, leftBox)) {
        float lx = std::clamp((float)((pt.x - leftBox.origin.x) / leftBox.size.width), 0.0f, 1.0f);
        float ly = std::clamp(1.0f - (float)((pt.y - leftBox.origin.y) / leftBox.size.height), 0.1f, 1.0f);
        leftLocalX_ = lx;
        leftLocalY_ = ly;
        leftVibrato_ = std::clamp((lx - 0.22f) / 0.78f, 0.0f, 1.0f);
        leftBrightness_ = ly;
        lockedChord_ = (lockedChord_ + 1) % 5;
        chordHold_ = (lockedChord_ > 0);
        return;
    }

    if (NSPointInRect(pt, glitchBox)) {
        float gx = std::clamp((float)((pt.x - glitchBox.origin.x) / glitchBox.size.width), 0.0f, 1.0f);
        int clickedMode = std::clamp((int)(gx * 4.0f), 0, 3);
        if (mouseGlitchActive_ && glitchMode_ == clickedMode) {
            mouseGlitchActive_ = false;
        } else {
            mouseGlitchActive_ = true;
            glitchMode_ = clickedMode;
            glitchLocalX_ = gx;
        }
        return;
    }

    if (NSPointInRect(pt, rightBox)) {
        float lx = std::clamp((float)((pt.x - rightBox.origin.x) / rightBox.size.width), 0.0f, 1.0f);
        float ly = std::clamp(1.0f - (float)((pt.y - rightBox.origin.y) / rightBox.size.height), 0.15f, 1.0f);
        int noteIdx = std::clamp((int)(lx * NUM_SCALE_NOTES), 0, NUM_SCALE_NOTES - 1);
        [self triggerNoteIdx:noteIdx localX:lx localY:ly];
        return;
    }

    CGFloat footY = camRect.origin.y + camRect.size.height + 14.0;
    if (NSPointInRect(pt, NSMakeRect(20, footY, 150, 26))) {
        chordHold_ = !chordHold_;
        if (!chordHold_) lockedChord_ = 0;
        return;
    }

    if (NSPointInRect(pt, NSMakeRect(W - 226, footY, 94, 28))) {
        std::lock_guard<std::mutex> lock(audio_->mtx);
        for (int tr = 0; tr < NUM_DRUM_TRACKS; ++tr) audio_->grid[tr].fill(0);
        lockedChord_ = 0;
        chordHold_ = false;
        mouseGlitchActive_ = false;
        return;
    }
    if (NSPointInRect(pt, NSMakeRect(W - 124, footY, 104, 28))) {
        std::lock_guard<std::mutex> lock(audio_->mtx);
        for (int s = 0; s < 16; ++s) {
            audio_->grid[0][s] = (s == 0 || (s == 10 && std::rand() % 100 < 60)) ? 1 : 0;
            audio_->grid[1][s] = (s == 4 || s == 12 || (std::rand() % 100 < 12)) ? 1 : 0;
            audio_->grid[2][s] = (s % 4 == 2 || (std::rand() % 100 < 18)) ? 1 : 0;
            audio_->grid[3][s] = (s == 7 || s == 15 || (std::rand() % 100 < 15)) ? 1 : 0;
        }
        audio_->triggerDrum(0, 1.0f);
        return;
    }
}

- (void)mouseDragged:(NSEvent *)event {
    NSPoint pt = [self convertPoint:[event locationInWindow] fromView:nil];
    NSRect bounds = [self bounds];
    CGFloat W = bounds.size.width;
    if (activeSlider_ != 0) {
        [self applySliderDragAtX:pt.x width:W];
        return;
    }
    NSRect camRect = [self cameraRectForBounds:bounds];
    NSRect rightBox = [self rightQuadrantRectInCamera:camRect];
    if (NSPointInRect(pt, rightBox)) {
        float lx = std::clamp((float)((pt.x - rightBox.origin.x) / rightBox.size.width), 0.0f, 1.0f);
        float ly = std::clamp(1.0f - (float)((pt.y - rightBox.origin.y) / rightBox.size.height), 0.15f, 1.0f);
        int noteIdx = std::clamp((int)(lx * NUM_SCALE_NOTES), 0, NUM_SCALE_NOTES - 1);
        [self triggerNoteIdx:noteIdx localX:lx localY:ly];
    }
}

- (void)mouseUp:(NSEvent *)event {
    (void)event;
    activeSlider_ = 0;
}

- (void)keyDown:(NSEvent *)event {
    if ([[event charactersIgnoringModifiers] isEqualToString:@" "]) {
        std::lock_guard<std::mutex> lock(audio_->mtx);
        audio_->is_playing = !audio_->is_playing;
        audio_->current_step = 0;
        audio_->step_sample_counter = 0.0;
    }
}

@end

// =====================================================================
// APPLICATION DELEGATE & MAIN
// =====================================================================

@interface AppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate> {
@public
    NSWindow *window_;
    AudioSynthState audio_;
    mediapipe_hands::HandTracker handTracker_;
}
@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    (void)notification;
    std::srand((unsigned)std::time(nullptr));

    initCoreAudio(audio_);
    handTracker_.start();

    NSRect frame = NSMakeRect(0, 0, 560, 820);
    NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                       NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;

    window_ = [[NSWindow alloc] initWithContentRect:frame
                                          styleMask:style
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    [window_ setTitle:@"BIOMORPHIC"];
    [window_ setMinSize:NSMakeSize(480, 680)];
    [window_ center];
    [window_ setDelegate:self];

    BiomorphicEssentialView *view = [[BiomorphicEssentialView alloc] initWithFrame:frame
                                                                             audio:&audio_
                                                                       handTracker:&handTracker_];
    [window_ setContentView:view];
    [window_ makeFirstResponder:view];
    [window_ makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    (void)sender;
    return YES;
}

- (void)applicationWillTerminate:(NSNotification *)notification {
    (void)notification;
    handTracker_.stop();
    shutdownCoreAudio(audio_);
}

@end

int main(int argc, const char *argv[]) {
    (void)argc;
    (void)argv;
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        AppDelegate *delegate = [[AppDelegate alloc] init];
        [app setDelegate:delegate];
        [app run];
    }
    return 0;
}
