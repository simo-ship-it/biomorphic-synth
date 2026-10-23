"""
Biomorphic & Microsound DSP Synthesis Engine (Pure Python)
Generates WAV audio buffers for:
- IDM / Microsound 4/4 Drum Machine voices (Ikeda / Alva Noto / Organic drops)
- Physical Modeling & Buchla-style Melodic Plucks (Modal / Karplus-Strong / FM)
- Granular Synthesis Clouds & Organic Textures (TouchDesigner aesthetic)
"""

import array
import math
import os
import random
import wave

SAMPLE_RATE = 44100


def _save_wav(filepath: str, samples: list[float], normalize: bool = True, max_amp: float = 0.88) -> str:
    """Save float samples (-1.0 to 1.0) to a 16-bit mono WAV file."""
    if not samples:
        samples = [0.0]

    peak = max(abs(s) for s in samples) or 1.0
    scale = (max_amp / peak) if (normalize and peak > 0) else 1.0

    pcm = array.array("h")
    for s in samples:
        val = int(max(-1.0, min(1.0, s * scale)) * 32767.0)
        pcm.append(val)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return filepath


# =====================================================================
# 1. DRUM MACHINE VOICES (IDM / Microsound / Organic Electronica)
# =====================================================================

def synth_sub_pulse(filepath: str) -> str:
    """
    Ryoji Ikeda / Alva Noto style sub-bass pulse with a sharp 2ms micro-click transient.
    """
    duration = 0.28
    n_samples = int(SAMPLE_RATE * duration)
    samples = []
    phase = 0.0

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # Pitch envelope drops rapidly from 150Hz to 46Hz
        freq = 46.0 + 110.0 * math.exp(-t * 48.0)
        phase += 2.0 * math.pi * freq / SAMPLE_RATE

        # Deep pure sine body + subtle 2nd harmonic saturation
        body = math.sin(phase) + 0.18 * math.sin(2.0 * phase)
        env = math.exp(-t * 11.5)

        # Ultra-crisp digital click at t < 2.5ms for TD audio-reactive transients
        click = 0.9 * math.sin(2.0 * math.pi * 2400.0 * t) * math.exp(-t * 900.0)
        samples.append(body * env + click)

    return _save_wav(filepath, samples)


def synth_water_droplet(filepath: str) -> str:
    """
    Organic water droplet (Biomorphic Ambient): upward bubble pitch bend + resonant tail.
    """
    duration = 0.22
    n_samples = int(SAMPLE_RATE * duration)
    samples = []
    phase = 0.0
    phase2 = 0.0

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # Classic Minnaert bubble frequency curve: sweeps UP as bubble detaches
        freq = 380.0 + 920.0 * (1.0 - math.exp(-t * 32.0)) * math.exp(-t * 8.0)
        phase += 2.0 * math.pi * freq / SAMPLE_RATE
        phase2 += 2.0 * math.pi * (freq * 2.756) / SAMPLE_RATE

        env = (1.0 - math.exp(-t * 450.0)) * math.exp(-t * 22.0)
        sig = math.sin(phase) * env + 0.25 * math.sin(phase2) * math.exp(-t * 45.0)
        samples.append(sig)

    return _save_wav(filepath, samples)


def synth_glitch_click(filepath: str) -> str:
    """
    Microsound impulse / static electricity snap (Alva Noto / Microsound school).
    """
    duration = 0.09
    n_samples = int(SAMPLE_RATE * duration)
    samples = []
    rng = random.Random(42)

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # Primary impulse + secondary micro-echoes at 12ms and 29ms
        imp1 = math.exp(-t * 480.0) * math.sin(2.0 * math.pi * 5200.0 * t)
        imp2 = math.exp(-abs(t - 0.014) * 700.0) * math.sin(2.0 * math.pi * 7800.0 * t) * 0.6
        imp3 = math.exp(-abs(t - 0.031) * 900.0) * (rng.uniform(-1.0, 1.0)) * 0.45
        samples.append(imp1 + imp2 + imp3)

    return _save_wav(filepath, samples)


def synth_hollow_wood(filepath: str) -> str:
    """
    Physical Modeling Synthesis: Modal resonator simulating a struck hollow wooden tube
    (inspired by Mutable Instruments Rings / Chromaphone).
    """
    duration = 0.26
    n_samples = int(SAMPLE_RATE * duration)
    samples = []
    # Inharmonic modal ratios of a hollow wooden bar/membrane
    modes = [
        (195.0, 1.00, 18.0),
        (195.0 * 1.58, 0.65, 28.0),
        (195.0 * 2.32, 0.42, 42.0),
        (195.0 * 3.41, 0.22, 65.0),
        (195.0 * 4.85, 0.12, 95.0),
    ]
    rng = random.Random(108)

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        val = 0.0
        for freq, amp, decay in modes:
            val += amp * math.sin(2.0 * math.pi * freq * t) * math.exp(-t * decay)
        # Short woody strike exciter
        exciter = rng.uniform(-0.5, 0.5) * math.exp(-t * 320.0)
        samples.append(val + exciter)

    return _save_wav(filepath, samples)


def synth_crystal_tick(filepath: str) -> str:
    """
    Eco-Futurism / Post-Nature crystalline metallic micro-hat using high-ratio FM synthesis.
    """
    duration = 0.18
    n_samples = int(SAMPLE_RATE * duration)
    samples = []

    carrier = 1480.0
    mod_freq = carrier * 1.4142  # irrational ratio (sqrt(2)) for glassy spectrum

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        mod_index = 4.5 * math.exp(-t * 42.0)
        mod = math.sin(2.0 * math.pi * mod_freq * t)
        sig = math.sin(2.0 * math.pi * carrier * t + mod_index * mod)
        shimmer = 0.35 * math.sin(2.0 * math.pi * 4120.0 * t + 2.0 * mod)
        env = math.exp(-t * 28.0)
        samples.append((sig + shimmer) * env)

    return _save_wav(filepath, samples)


def synth_insect_grain(filepath: str) -> str:
    """
    Granular field-recording texture: micro-grains of band-limited rustling / insect stridulation.
    """
    duration = 0.24
    n_samples = int(SAMPLE_RATE * duration)
    samples = [0.0] * n_samples
    rng = random.Random(77)

    # Spawn 14 microscopic grains (4ms - 15ms) with Hann window
    for g in range(14):
        start_idx = int(rng.uniform(0.0, duration * 0.78) * SAMPLE_RATE)
        grain_len = int(rng.uniform(0.004, 0.015) * SAMPLE_RATE)
        grain_freq = rng.uniform(2200.0, 6400.0)
        grain_amp = rng.uniform(0.35, 1.0) * (1.0 - (start_idx / n_samples) * 0.6)

        for j in range(grain_len):
            if start_idx + j >= n_samples:
                break
            # Hann window
            win = 0.5 * (1.0 - math.cos(2.0 * math.pi * j / max(1, grain_len - 1)))
            t_g = j / SAMPLE_RATE
            carrier = math.sin(2.0 * math.pi * grain_freq * t_g)
            noise = rng.uniform(-0.6, 0.6)
            samples[start_idx + j] += grain_amp * win * (0.7 * carrier + 0.3 * noise)

    return _save_wav(filepath, samples)


# =====================================================================
# 2. MELODIC SYNTH VOICES (Physical Modeling & Buchla / Barbieri Plucks)
# =====================================================================

def synth_melody_note(filepath: str, freq: float, timbre: str = "modal") -> str:
    """
    Synthesize a melodic note with organic physical modeling or wavefolded harmonics.
    timbre options:
      - 'modal': Mutable Instruments Rings-style modal resonator (glass/marimba/string)
      - 'buchla': Kaitlyn Aurelia Smith / Caterina Barbieri style wavefolded West-Coast pluck
      - 'crystal': Post-Nature FM bell + sub-harmonic shimmer
    """
    duration = 0.85
    n_samples = int(SAMPLE_RATE * duration)
    samples = []

    if timbre == "modal":
        # Modal resonator with slight detuned sympathetic strings
        modes = [
            (freq, 1.0, 4.2),
            (freq * 1.003, 0.55, 3.8),
            (freq * 2.0, 0.45, 7.5),
            (freq * 2.998, 0.22, 11.0),
            (freq * 4.71, 0.14, 18.0),  # metallic/woody overtone
        ]
        for i in range(n_samples):
            t = i / SAMPLE_RATE
            att = 1.0 - math.exp(-t * 260.0)
            val = 0.0
            for m_freq, amp, dec in modes:
                val += amp * math.sin(2.0 * math.pi * m_freq * t) * math.exp(-t * dec)
            samples.append(val * att)

    elif timbre == "buchla":
        # West-Coast Low-Pass Gate + Wavefolder
        for i in range(n_samples):
            t = i / SAMPLE_RATE
            lpg_env = (1.0 - math.exp(-t * 320.0)) * math.exp(-t * 5.8)
            fold_amount = 1.0 + 3.2 * math.exp(-t * 9.5)
            raw = math.sin(2.0 * math.pi * freq * t) + 0.35 * math.sin(2.0 * math.pi * (freq * 1.5) * t)
            # Buchla-like sine wavefolder
            folded = math.sin(raw * fold_amount * 1.4)
            sub = 0.4 * math.sin(2.0 * math.pi * (freq * 0.5) * t) * math.exp(-t * 4.5)
            samples.append((folded + sub) * lpg_env)

    else:  # 'crystal'
        for i in range(n_samples):
            t = i / SAMPLE_RATE
            att = 1.0 - math.exp(-t * 180.0)
            mod_idx = 2.6 * math.exp(-t * 8.5)
            mod = math.sin(2.0 * math.pi * (freq * 3.0) * t)
            shimmer = 0.28 * math.sin(2.0 * math.pi * (freq * 4.005) * t) * math.exp(-t * 6.0)
            car = math.sin(2.0 * math.pi * freq * t + mod_idx * mod) * math.exp(-t * 4.0)
            samples.append((car + shimmer) * att)

    return _save_wav(filepath, samples)


# =====================================================================
# 3. TEXTURE / GRANULAR / SOUNDSCAPE FX PADS
# =====================================================================

def synth_granular_cloud(filepath: str) -> str:
    """
    Granular Synthesis Cloud (1.8s):
    Deconstructs harmonic water/wind resonance into 45 overlapping micro-grains (15-90ms).
    """
    duration = 1.8
    n_samples = int(SAMPLE_RATE * duration)
    samples = [0.0] * n_samples
    rng = random.Random(303)

    # Pentatonic D minor / Dorian frequencies for organic cloud
    scale_freqs = [146.83, 174.61, 220.0, 261.63, 293.66, 349.23, 440.0, 523.25, 587.33, 659.25]

    for _ in range(55):
        start_idx = int(rng.uniform(0.0, duration * 0.82) * SAMPLE_RATE)
        grain_len = int(rng.uniform(0.025, 0.110) * SAMPLE_RATE)
        base_f = rng.choice(scale_freqs) * rng.uniform(0.994, 1.006)
        amp = rng.uniform(0.2, 0.55)
        chirp = rng.uniform(-0.08, 0.12)

        for j in range(grain_len):
            if start_idx + j >= n_samples:
                break
            win = 0.5 * (1.0 - math.cos(2.0 * math.pi * j / max(1, grain_len - 1)))
            t_g = j / SAMPLE_RATE
            f_inst = base_f * (1.0 + chirp * (j / grain_len))
            sig = math.sin(2.0 * math.pi * f_inst * t_g) + 0.3 * math.sin(2.0 * math.pi * f_inst * 2.01 * t_g)
            samples[start_idx + j] += amp * win * sig

    return _save_wav(filepath, samples)


def synth_bio_spores(filepath: str) -> str:
    """
    Bio-Sonification Spore Burst (1.4s):
    Rapid organic cluster of water-like pings and micro-clicks following a Fibonacci/golden-ratio spiral.
    """
    duration = 1.4
    n_samples = int(SAMPLE_RATE * duration)
    samples = [0.0] * n_samples
    phi = 1.61803398875

    for k in range(22):
        t_onset = ((k * phi) % 1.0) * (duration * 0.82)
        start_idx = int(t_onset * SAMPLE_RATE)
        ping_len = int(0.085 * SAMPLE_RATE)
        freq = 320.0 * (1.0 + ((k * 7) % 12) * 0.14)

        for j in range(ping_len):
            if start_idx + j >= n_samples:
                break
            t_local = j / SAMPLE_RATE
            env = (1.0 - math.exp(-t_local * 380.0)) * math.exp(-t_local * 38.0)
            f_bend = freq + 280.0 * math.exp(-t_local * 60.0)
            samples[start_idx + j] += 0.5 * env * math.sin(2.0 * math.pi * f_bend * t_local)

    return _save_wav(filepath, samples)


def synth_ikeda_datastream(filepath: str) -> str:
    """
    Ryoji Ikeda 'Test Pattern' / 'Dataplex' style rhythmic binary sine + white noise burst (1.1s).
    """
    duration = 1.1
    n_samples = int(SAMPLE_RATE * duration)
    samples = []
    rng = random.Random(10101)
    pattern = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
    step_len = duration / len(pattern)

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        step = min(len(pattern) - 1, int(t / step_len))
        local_t = t - step * step_len
        if pattern[step]:
            # High-frequency 10kHz data tone + 110Hz sub + noise gate
            gate = 1.0 if local_t < step_len * 0.65 else 0.0
            hi = 0.32 * math.sin(2.0 * math.pi * 4800.0 * t)
            sub = 0.65 * math.sin(2.0 * math.pi * 73.42 * t)
            burst = 0.28 * rng.uniform(-1.0, 1.0) if local_t < 0.012 else 0.0
            samples.append((hi + sub + burst) * gate)
        else:
            samples.append(0.0)

    return _save_wav(filepath, samples)


def synth_sub_nature_drone(filepath: str) -> str:
    """
    Deep evolving Biomorphic Ambient pad/drone (3.2s) in D minor 9th.
    """
    duration = 3.2
    n_samples = int(SAMPLE_RATE * duration)
    samples = []
    freqs = [73.42, 110.0, 146.83, 147.15, 220.0, 261.63, 329.63]

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        # Smooth fade-in and fade-out envelope
        env = math.sin(math.pi * (t / duration)) ** 0.75
        lfo = 1.0 + 0.004 * math.sin(2.0 * math.pi * 0.45 * t)
        val = 0.0
        for idx, f in enumerate(freqs):
            mod = math.sin(2.0 * math.pi * (0.2 + idx * 0.11) * t)
            val += (0.35 / (1.0 + idx * 0.35)) * math.sin(2.0 * math.pi * f * lfo * t + 0.4 * mod)
        samples.append(val * env)

    return _save_wav(filepath, samples)


def synth_field_wind_water(filepath: str) -> str:
    """
    Organic Electronica texture (2.2s): Resonant swept noise + micro-droplets simulating wind & cave water.
    """
    duration = 2.2
    n_samples = int(SAMPLE_RATE * duration)
    samples = []
    rng = random.Random(909)
    lp1 = 0.0
    lp2 = 0.0

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        env = math.sin(math.pi * (t / duration))
        white = rng.uniform(-1.0, 1.0)
        # Resonant lowpass filter modulated by a slow Perlin-like wave
        cutoff = 0.04 + 0.03 * math.sin(2.0 * math.pi * 0.7 * t) + 0.015 * math.cos(2.0 * math.pi * 1.9 * t)
        lp1 += cutoff * (white - lp1)
        lp2 += cutoff * (lp1 - lp2)
        shimmer = 0.15 * math.sin(2.0 * math.pi * 587.33 * t) * (0.5 + 0.5 * math.sin(2.0 * math.pi * 2.5 * t))
        samples.append((lp2 * 2.4 + shimmer) * env)

    return _save_wav(filepath, samples)


def _ensure_wav(filepath: str, gen_fn, *args, **kwargs) -> str:
    if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
        return filepath
    return gen_fn(filepath, *args, **kwargs)


def build_sound_bank(assets_dir: str) -> dict:
    """
    Generate all drum, melody, and texture WAV files into `assets_dir` (cached on disk) and return paths.
    """
    os.makedirs(assets_dir, exist_ok=True)

    drums = [
        {"id": "sub_pulse", "name": "01 · Ikeda Sub Pulse", "color": "#00f5d4", "path": _ensure_wav(os.path.join(assets_dir, "drum_sub.wav"), synth_sub_pulse)},
        {"id": "water_drop", "name": "02 · Bio Water Drop", "color": "#00bbf9", "path": _ensure_wav(os.path.join(assets_dir, "drum_drop.wav"), synth_water_droplet)},
        {"id": "hollow_wood", "name": "03 · Modal Wood Body", "color": "#9b5de5", "path": _ensure_wav(os.path.join(assets_dir, "drum_wood.wav"), synth_hollow_wood)},
        {"id": "glitch_click", "name": "04 · Alva Glitch Click", "color": "#f15bb5", "path": _ensure_wav(os.path.join(assets_dir, "drum_click.wav"), synth_glitch_click)},
        {"id": "crystal_tick", "name": "05 · Post-Nature Crystal", "color": "#fee440", "path": _ensure_wav(os.path.join(assets_dir, "drum_crystal.wav"), synth_crystal_tick)},
        {"id": "insect_grain", "name": "06 · Insect Field Grain", "color": "#38b000", "path": _ensure_wav(os.path.join(assets_dir, "drum_grain.wav"), synth_insect_grain)},
    ]

    # D Dorian / Biomorphic scale (8 notes from D3 to D5)
    scale_notes = [
        ("D3", 146.83),
        ("F3", 174.61),
        ("A3", 220.00),
        ("C4", 261.63),
        ("D4", 293.66),
        ("E4", 329.63),
        ("G4", 392.00),
        ("A4", 440.00),
    ]

    melodies = {"modal": [], "buchla": [], "crystal": []}
    for timbre in melodies.keys():
        for note_name, freq in scale_notes:
            fname = f"mel_{timbre}_{note_name}.wav"
            fpath = _ensure_wav(os.path.join(assets_dir, fname), synth_melody_note, freq, timbre=timbre)
            melodies[timbre].append({
                "note": note_name,
                "freq": freq,
                "timbre": timbre,
                "path": fpath,
            })

    textures = [
        {"id": "granular_cloud", "name": "Granular Cloud", "subtitle": "55 Micro-Grains", "key": "1", "color": "#00bbf9", "path": _ensure_wav(os.path.join(assets_dir, "fx_granular.wav"), synth_granular_cloud)},
        {"id": "bio_spores", "name": "Bio-Spores", "subtitle": "Fibonacci Pings", "key": "2", "color": "#38b000", "path": _ensure_wav(os.path.join(assets_dir, "fx_spores.wav"), synth_bio_spores)},
        {"id": "ikeda_data", "name": "Ikeda Datastream", "subtitle": "Microsound Pulse", "key": "3", "color": "#f15bb5", "path": _ensure_wav(os.path.join(assets_dir, "fx_ikeda.wav"), synth_ikeda_datastream)},
        {"id": "sub_drone", "name": "Biomorphic Drone", "subtitle": "D Min9 Breath", "key": "4", "color": "#9b5de5", "path": _ensure_wav(os.path.join(assets_dir, "fx_drone.wav"), synth_sub_nature_drone)},
        {"id": "field_wind", "name": "Cave Wind & Water", "subtitle": "Organic Field", "key": "5", "color": "#00f5d4", "path": _ensure_wav(os.path.join(assets_dir, "fx_field.wav"), synth_field_wind_water)},
    ]

    return {
        "drums": drums,
        "melodies": melodies,
        "textures": textures,
    }
