"""
Ultra-Low-Latency CoreAudio RAM Mixer + Continuous XY Synth (`libnative_mixer.dylib` via ctypes).
- Preloads drum machine WAV buffers into native C memory
- Exposes `set_continuous_synth(freq, amp, mode)` for phase-continuous real-time XY mouse synthesis
"""

import array
import ctypes
import math
import os
import time
import wave


class AudioEngine:
    def __init__(self):
        self._lib = None
        self._sample_ids: dict[str, int] = {}
        self._next_id = 0

        # Audio-reactive energy bands (0.0 -> 2.0+) for TouchDesigner visualizer
        self.sub_energy = 0.0
        self.mid_energy = 0.0
        self.high_energy = 0.0
        self.synth_amp = 0.0
        self.synth_freq = 220.0
        self.last_trigger_color = "#00f5d4"
        self.last_trigger_label = "COREAUDIO READY"
        self._last_update = time.perf_counter()

        self._init_native_mixer()

    def _init_native_mixer(self) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dylib_path = os.path.join(base_dir, "libnative_mixer.dylib")
        if not os.path.exists(dylib_path):
            return

        lib = ctypes.CDLL(dylib_path)
        lib.mixer_init.argtypes = []
        lib.mixer_init.restype = ctypes.c_int

        lib.mixer_load_sample.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.mixer_load_sample.restype = ctypes.c_int

        lib.mixer_trigger.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float]
        lib.mixer_trigger.restype = None

        lib.mixer_set_synth.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int]
        lib.mixer_set_synth.restype = None

        lib.mixer_stop_all.argtypes = []
        lib.mixer_stop_all.restype = None

        lib.mixer_shutdown.argtypes = []
        lib.mixer_shutdown.restype = None

        if lib.mixer_init() == 0:
            self._lib = lib

    def preload_wav(self, wav_path: str) -> int:
        """Load a 16-bit mono WAV file into native C RAM once for instant triggering."""
        if wav_path in self._sample_ids:
            return self._sample_ids[wav_path]
        if not self._lib or not os.path.exists(wav_path):
            return -1

        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        pcm = array.array("h")
        pcm.frombytes(raw_bytes)
        n = len(pcm)
        FloatArray = ctypes.c_float * n
        float_buf = FloatArray(*(s / 32768.0 for s in pcm))

        sid = self._next_id
        self._next_id += 1
        self._lib.mixer_load_sample(sid, float_buf, n)
        self._sample_ids[wav_path] = sid
        return sid

    def preload_bank(self, bank: dict) -> None:
        """Preload all drum samples into RAM."""
        for d in bank.get("drums", []):
            self.preload_wav(d["path"])

    def play(
        self,
        wav_path: str,
        volume: float = 0.85,
        rate: float = 1.0,
        band: str = "mid",
        color: str = "#00f5d4",
        label: str = "",
    ) -> None:
        """Trigger a preloaded drum sample in RAM immediately."""
        sid = self._sample_ids.get(wav_path, -1)
        if sid < 0:
            sid = self.preload_wav(wav_path)

        vol = max(0.05, min(1.5, volume))
        r = max(0.5, min(2.0, rate))

        if self._lib and sid >= 0:
            self._lib.mixer_trigger(sid, ctypes.c_float(vol), ctypes.c_float(r))

        if band == "sub":
            self.sub_energy = min(2.2, self.sub_energy + 1.15 * vol)
            self.high_energy = min(2.0, self.high_energy + 0.35 * vol)
        elif band == "high":
            self.high_energy = min(2.4, self.high_energy + 1.25 * vol)
            self.mid_energy = min(2.0, self.mid_energy + 0.30 * vol)
        else:
            self.mid_energy = min(2.2, self.mid_energy + 1.10 * vol)
            self.high_energy = min(2.0, self.high_energy + 0.25 * vol)

        self.last_trigger_color = color
        if label:
            self.last_trigger_label = label

    def set_continuous_synth(
        self,
        freq: float,
        amp: float,
        mode: int = 0,
        color: str = "#00f5d4",
        label: str = "",
    ) -> None:
        """Update the continuous XY Pad synthesizer in real time."""
        self.synth_freq = freq
        self.synth_amp = amp
        if self._lib:
            self._lib.mixer_set_synth(ctypes.c_float(freq), ctypes.c_float(amp), ctypes.c_int(mode))
        if amp > 0.02:
            self.last_trigger_color = color
            if label:
                self.last_trigger_label = label

    def step_physics(self) -> tuple[float, float, float]:
        """Decay transient envelopes and blend continuous XY synth energy."""
        now = time.perf_counter()
        dt = min(0.1, max(0.001, now - self._last_update))
        self._last_update = now

        self.sub_energy *= math.exp(-dt * 5.5)
        self.mid_energy *= math.exp(-dt * 7.2)
        self.high_energy *= math.exp(-dt * 12.0)

        # Feed continuous XY synth energy into the reactive bands based on frequency & amplitude
        if self.synth_amp > 0.01:
            norm_f = max(0.0, min(1.0, (self.synth_freq - 55.0) / 825.0))
            sub_contrib = self.synth_amp * (1.0 - norm_f) * 1.3
            mid_contrib = self.synth_amp * 1.2
            high_contrib = self.synth_amp * norm_f * 1.4
            self.sub_energy = max(self.sub_energy, sub_contrib)
            self.mid_energy = max(self.mid_energy, mid_contrib)
            self.high_energy = max(self.high_energy, high_contrib)

        return self.sub_energy, self.mid_energy, self.high_energy

    def stop_all(self) -> None:
        if self._lib:
            self._lib.mixer_stop_all()
        self.sub_energy = 0.0
        self.mid_energy = 0.0
        self.high_energy = 0.0
        self.synth_amp = 0.0

    def shutdown(self) -> None:
        if self._lib:
            self._lib.mixer_shutdown()
            self._lib = None
