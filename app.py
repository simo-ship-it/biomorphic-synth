"""
BIOMORPHIC // TD-SYNTH (Continuous XY Pad + 4/4 Microsound Sequencer)
Interactive Python Music Creation App inspired by the TouchDesigner & MUTEK Digital Art Scene:
- 01 // 4/4 Microsound & Organic Drum Machine (16 steps)
- 02 // Continuous 2D XY Mouse Synthesizer (X = Frequency 55Hz–880Hz, Y = Intensity/Harmonics)
- Right Panel // TouchDesigner Audio-Reactive 3D Point Cloud & CHOP Telemetry
"""

import math
import os
import random
import threading
import time
import tkinter as tk

from audio_player import AudioEngine
from synth_engine import build_sound_bank

# Palette (TouchDesigner / Derivative Dark UI aesthetic)
BG_MAIN = "#0b0d12"
BG_PANEL = "#12161f"
BG_CELL_OFF = "#1b2130"
BG_CELL_BEAT = "#222a3d"
BORDER_DIM = "#2a3348"
TEXT_MAIN = "#e2e8f0"
TEXT_MUTED = "#7c8ba1"
ACCENT_CYAN = "#00f5d4"
ACCENT_BLUE = "#00bbf9"
ACCENT_MAGENTA = "#f15bb5"
ACCENT_PURPLE = "#9b5de5"
ACCENT_GREEN = "#38b000"
ACCENT_YELLOW = "#fee440"


class BiomorphicSynthApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BIOMORPHIC // TouchDesigner 4/4 Sequencer & Continuous XY Synth")
        self.root.configure(bg=BG_MAIN)
        self.root.geometry("1220x800")
        self.root.minsize(1080, 720)

        # Build or load cached WAV sound bank & preload into C CoreAudio RAM mixer
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        self.bank = build_sound_bank(assets_dir)
        self.audio = AudioEngine()
        self.audio.preload_bank(self.bank)

        # Sequencer state: 6 drum tracks x 16 steps (4/4 time, 4 beats of 4 sixteenth notes)
        self.num_tracks = len(self.bank["drums"])
        self.num_steps = 16
        self.grid = [[0] * self.num_steps for _ in range(self.num_tracks)]
        self.muted = [False] * self.num_tracks

        self.is_playing = False
        self.current_step = 0
        self.bpm = 108
        self.micro_jitter = 0.06
        self.bio_sonification_active = False
        self._alive = True
        self._ui_step_dirty = False

        # Continuous XY Synth Pad state
        # mode: 0 = Buchla Wavefold, 1 = Harmonic Drone, 2 = Post-Nature Crystal FM
        self.synth_mode = 0
        self.xy_active = False       # True while mouse button is held or HOLD is active
        self.xy_hold = False         # Latch/Hold toggle so sound stays continuous hands-free
        self.xy_norm_x = 0.5         # 0.0 (left, low freq) -> 1.0 (right, high freq)
        self.xy_norm_y = 0.65        # 0.0 (bottom, silent) -> 1.0 (top, max intensity)
        self.xy_freq = 220.0
        self.xy_amp = 0.0
        self.xy_trail: list[tuple[float, float]] = []

        # Cached Canvas item IDs for instant (<0.2ms) in-place UI updates
        self._seq_cell_rects = {}
        self._seq_cell_dots = {}
        self._seq_beat_Texts = {}
        self._seq_mute_rects = {}
        self._seq_mute_texts = {}
        self._seq_name_rects = {}
        self._seq_name_texts = {}

        # Bio-sonification simulated plant sensor state
        self.bio_phase = 0.0
        self.bio_history = [0.0] * 48

        # TouchDesigner Visualizer 3D point cloud state
        self.vis_nodes = self._init_vis_nodes(28)
        self.vis_angle = 0.0

        # Build UI and load initial organic pattern
        self._build_ui()
        self.load_preset("cooper")
        self._bind_keyboard()

        # Start high-precision background sequencer clock thread
        self._clock_thread = threading.Thread(target=self._precision_clock_loop, daemon=True)
        self._clock_thread.start()

        # Start 30 FPS audio-reactive visual loop
        self._animate_visualizer()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_vis_nodes(self, count: int) -> list[dict]:
        rng = random.Random(108)
        nodes = []
        phi = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(count):
            y = 1.0 - (i / float(count - 1)) * 2.0
            radius = math.sqrt(max(0.0, 1.0 - y * y))
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            nodes.append({
                "bx": x, "by": y, "bz": z,
                "seed": rng.uniform(0.0, 10.0),
            })
        return nodes

    # =================================================================
    # UI CONSTRUCTION
    # =================================================================

    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG_PANEL, height=62, highlightthickness=1, highlightbackground=BORDER_DIM)
        header.pack(fill=tk.X, padx=14, pady=(12, 6))

        title_frame = tk.Frame(header, bg=BG_PANEL)
        title_frame.pack(side=tk.LEFT, padx=14, pady=8)

        tk.Label(
            title_frame,
            text="BIOMORPHIC // COREAUDIO LOW-LATENCY SYNTH",
            font=("Menlo", 14, "bold"),
            fg=ACCENT_CYAN,
            bg=BG_PANEL,
        ).pack(anchor="w")
        tk.Label(
            title_frame,
            text="4/4 Microsound Drum Machine · Continuous 2D XY Synth Pad · Bio-Sonification",
            font=("Menlo", 10),
            fg=TEXT_MUTED,
            bg=BG_PANEL,
        ).pack(anchor="w")

        transport = tk.Frame(header, bg=BG_PANEL)
        transport.pack(side=tk.RIGHT, padx=14, pady=8)

        self.play_btn = self._make_pill_button(
            transport, "▶ PLAY 4/4 [SPACE]", ACCENT_CYAN, "#061014", self.toggle_play, width=165
        )
        self.play_btn.pack(side=tk.LEFT, padx=6)

        self.hold_btn = self._make_pill_button(
            transport, "🔒 XY HOLD: OFF", "#22303c", TEXT_MAIN, self.toggle_xy_hold, width=155
        )
        self.hold_btn.pack(side=tk.LEFT, padx=6)

        self.bio_btn = self._make_pill_button(
            transport, "☘ BIO-SONIFY: OFF", "#22303c", TEXT_MAIN, self.toggle_bio_sonification, width=165
        )
        self.bio_btn.pack(side=tk.LEFT, padx=6)

        bpm_frame = tk.Frame(transport, bg=BG_PANEL)
        bpm_frame.pack(side=tk.LEFT, padx=(12, 4))
        self.bpm_label = tk.Label(
            bpm_frame, text=f"BPM: {self.bpm}", font=("Menlo", 10, "bold"), fg=TEXT_MAIN, bg=BG_PANEL
        )
        self.bpm_label.pack(anchor="w")
        self.bpm_scale = tk.Scale(
            bpm_frame,
            from_=68,
            to=155,
            orient=tk.HORIZONTAL,
            showvalue=False,
            length=110,
            bg=BG_PANEL,
            fg=ACCENT_CYAN,
            troughcolor=BG_CELL_OFF,
            highlightthickness=0,
            command=self._on_bpm_change,
        )
        self.bpm_scale.set(self.bpm)
        self.bpm_scale.pack()

        workspace = tk.Frame(self.root, bg=BG_MAIN)
        workspace.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)

        left_col = tk.Frame(workspace, bg=BG_MAIN)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right_col = tk.Frame(workspace, bg=BG_PANEL, width=370, highlightthickness=1, highlightbackground=BORDER_DIM)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_col.pack_propagate(False)

        # --- LEFT TOP: 01 // 4/4 DRUM SEQUENCER ---
        seq_panel = tk.Frame(left_col, bg=BG_PANEL, highlightthickness=1, highlightbackground=BORDER_DIM)
        seq_panel.pack(fill=tk.X, pady=(0, 10))

        seq_top = tk.Frame(seq_panel, bg=BG_PANEL)
        seq_top.pack(fill=tk.X, padx=12, pady=(10, 4))

        tk.Label(
            seq_top,
            text="01 // MICROSOUND & ORGANIC DRUM MACHINE (4/4 · 16 STEPS)",
            font=("Menlo", 11, "bold"),
            fg=TEXT_MAIN,
            bg=BG_PANEL,
        ).pack(side=tk.LEFT)

        preset_bar = tk.Frame(seq_top, bg=BG_PANEL)
        preset_bar.pack(side=tk.RIGHT)

        tk.Label(preset_bar, text="PRESET:", font=("Menlo", 9), fg=TEXT_MUTED, bg=BG_PANEL).pack(side=tk.LEFT, padx=4)
        for label, key in [
            ("Ikeda/Noto", "ikeda"),
            ("Bio-Fluid", "bio"),
            ("Cooper 4/4", "cooper"),
            ("Mutate 🎲", "mutate"),
            ("Clear", "clear"),
        ]:
            btn = self._make_small_tag(preset_bar, label, lambda k=key: self.load_preset(k))
            btn.pack(side=tk.LEFT, padx=3)

        self.seq_canvas = tk.Canvas(
            seq_panel,
            bg=BG_PANEL,
            height=255,
            highlightthickness=0,
            cursor="hand2",
        )
        self.seq_canvas.pack(fill=tk.X, padx=12, pady=(4, 10))
        self.seq_canvas.bind("<ButtonPress-1>", self._on_seq_click)
        self.seq_canvas.bind("<Configure>", lambda e: self._rebuild_sequencer_geometry())

        # --- LEFT BOTTOM (EXPANDED): 02 // CONTINUOUS 2D XY MOUSE SYNTH PAD ---
        xy_panel = tk.Frame(left_col, bg=BG_PANEL, highlightthickness=1, highlightbackground=BORDER_DIM)
        xy_panel.pack(fill=tk.BOTH, expand=True)

        xy_top = tk.Frame(xy_panel, bg=BG_PANEL)
        xy_top.pack(fill=tk.X, padx=12, pady=(10, 4))

        tk.Label(
            xy_top,
            text="02 // CONTINUOUS XY SYNTH PAD (X: FREQ 55–880 Hz · Y: INTENSITY & HARMONICS)",
            font=("Menlo", 11, "bold"),
            fg=TEXT_MAIN,
            bg=BG_PANEL,
        ).pack(side=tk.LEFT)

        timbre_bar = tk.Frame(xy_top, bg=BG_PANEL)
        timbre_bar.pack(side=tk.RIGHT)
        tk.Label(timbre_bar, text="ENGINE:", font=("Menlo", 9), fg=TEXT_MUTED, bg=BG_PANEL).pack(
            side=tk.LEFT, padx=4
        )

        self.timbre_buttons = {}
        for m_id, m_label in [
            (0, "Buchla Wavefold"),
            (1, "Harmonic Drone"),
            (2, "Crystal FM"),
        ]:
            b = self._make_small_tag(timbre_bar, m_label, lambda mid=m_id: self.set_synth_mode(mid))
            b.pack(side=tk.LEFT, padx=3)
            self.timbre_buttons[m_id] = b
        self._refresh_timbre_buttons()

        self.xy_canvas = tk.Canvas(
            xy_panel,
            bg="#080b11",
            highlightthickness=1,
            highlightbackground=BORDER_DIM,
            cursor="crosshair",
        )
        self.xy_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))

        # Mouse events for continuous XY control
        self.xy_canvas.bind("<ButtonPress-1>", self._on_xy_press)
        self.xy_canvas.bind("<B1-Motion>", self._on_xy_drag)
        self.xy_canvas.bind("<ButtonRelease-1>", self._on_xy_release)
        # Right-click toggles Hold/Drone lock at current position
        self.xy_canvas.bind("<ButtonPress-2>", lambda e: self.toggle_xy_hold())
        self.xy_canvas.bind("<ButtonPress-3>", lambda e: self.toggle_xy_hold())

        # --- RIGHT COLUMN: TOUCHDESIGNER AUDIO-REACTIVE VISUALIZER ---
        vis_header = tk.Frame(right_col, bg=BG_PANEL)
        vis_header.pack(fill=tk.X, padx=12, pady=(10, 4))
        tk.Label(
            vis_header,
            text="TD // AUDIO ANALYSIS & BIOMESH",
            font=("Menlo", 11, "bold"),
            fg=ACCENT_CYAN,
            bg=BG_PANEL,
        ).pack(anchor="w")
        tk.Label(
            vis_header,
            text="Real-time CHOP Spectrum + Reactive SOP Point Cloud",
            font=("Menlo", 9),
            fg=TEXT_MUTED,
            bg=BG_PANEL,
        ).pack(anchor="w")

        self.vis_canvas = tk.Canvas(
            right_col,
            bg="#080a0f",
            highlightthickness=1,
            highlightbackground=BORDER_DIM,
        )
        self.vis_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))

    # =================================================================
    # CUSTOM CANVAS BUTTON HELPERS
    # =================================================================

    def _make_pill_button(self, parent, text: str, bg_color: str, fg_color: str, command, width: int = 150):
        c = tk.Canvas(
            parent,
            width=width,
            height=34,
            bg=BG_PANEL,
            highlightthickness=0,
            cursor="hand2",
        )
        rect = c.create_rectangle(2, 2, width - 2, 32, fill=bg_color, outline=BORDER_DIM, width=1)
        lbl = c.create_text(width // 2, 17, text=text, fill=fg_color, font=("Menlo", 10, "bold"))
        c.bind("<ButtonPress-1>", lambda e: command())
        c._rect = rect
        c._lbl = lbl
        return c

    def _update_pill_button(self, canvas_btn, text: str, bg_color: str, fg_color: str):
        canvas_btn.itemconfig(canvas_btn._rect, fill=bg_color)
        canvas_btn.itemconfig(canvas_btn._lbl, text=text, fill=fg_color)

    def _make_small_tag(self, parent, text: str, command):
        w = max(68, len(text) * 8 + 18)
        c = tk.Canvas(parent, width=w, height=24, bg=BG_PANEL, highlightthickness=0, cursor="hand2")
        rect = c.create_rectangle(1, 1, w - 1, 23, fill=BG_CELL_OFF, outline=BORDER_DIM, width=1)
        lbl = c.create_text(w // 2, 12, text=text, fill=TEXT_MAIN, font=("Menlo", 9))
        c.bind("<ButtonPress-1>", lambda e: command())
        c._rect = rect
        c._lbl = lbl
        return c

    def _refresh_timbre_buttons(self):
        mode_colors = {0: ACCENT_CYAN, 1: ACCENT_PURPLE, 2: ACCENT_YELLOW}
        for mid, btn in self.timbre_buttons.items():
            if mid == self.synth_mode:
                col = mode_colors.get(mid, ACCENT_CYAN)
                btn.itemconfig(btn._rect, fill=col, outline="#ffffff")
                btn.itemconfig(btn._lbl, fill="#080a0f")
            else:
                btn.itemconfig(btn._rect, fill=BG_CELL_OFF, outline=BORDER_DIM)
                btn.itemconfig(btn._lbl, fill=TEXT_MUTED)

    # =================================================================
    # 01 // FAST IN-PLACE SEQUENCER RENDERING
    # =================================================================

    def _rebuild_sequencer_geometry(self):
        c = self.seq_canvas
        c.delete("all")
        self._seq_cell_rects.clear()
        self._seq_cell_dots.clear()
        self._seq_beat_Texts.clear()
        self._seq_mute_rects.clear()
        self._seq_mute_texts.clear()
        self._seq_name_rects.clear()
        self._seq_name_texts.clear()

        w = max(680, c.winfo_width())
        h = max(240, c.winfo_height())

        label_w = 215
        grid_w = w - label_w - 10
        step_w = grid_w / self.num_steps
        row_h = (h - 28) / self.num_tracks

        for s in range(self.num_steps):
            x0 = label_w + s * step_w
            is_downbeat = (s % 4 == 0)
            beat_txt = f"{(s // 4) + 1}.{(s % 4) + 1}"
            tid = c.create_text(
                x0 + step_w / 2,
                12,
                text=beat_txt,
                fill=TEXT_MAIN if is_downbeat else TEXT_MUTED,
                font=("Menlo", 8, "bold" if is_downbeat else "normal"),
            )
            self._seq_beat_Texts[s] = tid

        for r, drum in enumerate(self.bank["drums"]):
            y0 = 26 + r * row_h
            y1 = y0 + row_h - 5
            color = drum["color"]

            nrect = c.create_rectangle(4, y0, label_w - 58, y1, fill="#171c28", outline=color, width=1)
            ntxt = c.create_text(12, (y0 + y1) / 2, text=drum["name"], anchor="w", fill=TEXT_MAIN, font=("Menlo", 9, "bold"))
            self._seq_name_rects[r] = nrect
            self._seq_name_texts[r] = ntxt

            mx0, mx1 = label_w - 54, label_w - 30
            mrect = c.create_rectangle(mx0, y0, mx1, y1, fill=BG_CELL_OFF, outline=BORDER_DIM)
            mtxt = c.create_text((mx0 + mx1) / 2, (y0 + y1) / 2, text="M", fill=TEXT_MUTED, font=("Menlo", 9, "bold"))
            self._seq_mute_rects[r] = mrect
            self._seq_mute_texts[r] = mtxt

            rx0, rx1 = label_w - 27, label_w - 5
            c.create_rectangle(rx0, y0, rx1, y1, fill=BG_CELL_OFF, outline=BORDER_DIM)
            c.create_text((rx0 + rx1) / 2, (y0 + y1) / 2, text="🎲", fill=TEXT_MAIN, font=("Menlo", 9))

            for s in range(self.num_steps):
                sx0 = label_w + s * step_w + 2
                sx1 = label_w + (s + 1) * step_w - 2
                cx = (sx0 + sx1) / 2
                cy = (y0 + y1) / 2
                crect = c.create_rectangle(sx0, y0, sx1, y1, fill=BG_CELL_OFF, outline=BORDER_DIM, width=1)
                cdot = c.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#ffffff", outline="", state="hidden")
                self._seq_cell_rects[(r, s)] = crect
                self._seq_cell_dots[(r, s)] = cdot

        self._draw_sequencer()

    def _draw_sequencer(self):
        if not self._seq_cell_rects:
            self._rebuild_sequencer_geometry()
            return

        c = self.seq_canvas
        for s in range(self.num_steps):
            is_downbeat = (s % 4 == 0)
            is_current = (self.is_playing and s == self.current_step)
            col = ACCENT_CYAN if is_current else (TEXT_MAIN if is_downbeat else TEXT_MUTED)
            c.itemconfig(self._seq_beat_Texts[s], fill=col)

        for r, drum in enumerate(self.bank["drums"]):
            color = drum["color"]
            is_muted = self.muted[r]
            c.itemconfig(
                self._seq_name_rects[r],
                fill="#171c28" if not is_muted else "#11141c",
                outline=color if not is_muted else BORDER_DIM,
            )
            c.itemconfig(self._seq_name_texts[r], fill=TEXT_MAIN if not is_muted else TEXT_MUTED)
            c.itemconfig(self._seq_mute_rects[r], fill=ACCENT_MAGENTA if is_muted else BG_CELL_OFF)
            c.itemconfig(self._seq_mute_texts[r], fill="#0b0d12" if is_muted else TEXT_MUTED)

            for s in range(self.num_steps):
                active = self.grid[r][s] == 1
                is_playhead = self.is_playing and (s == self.current_step)
                is_beat_group = (s // 4) % 2 == 0

                if active:
                    fill_col = color if not is_muted else "#3a4256"
                    outline_col = "#ffffff" if is_playhead else color
                else:
                    fill_col = "#283248" if is_playhead else (BG_CELL_BEAT if is_beat_group else BG_CELL_OFF)
                    outline_col = ACCENT_CYAN if is_playhead else BORDER_DIM

                c.itemconfig(
                    self._seq_cell_rects[(r, s)],
                    fill=fill_col,
                    outline=outline_col,
                    width=2 if is_playhead else 1,
                )
                c.itemconfig(
                    self._seq_cell_dots[(r, s)],
                    state="normal" if (active and not is_muted) else "hidden",
                )

    def _on_seq_click(self, event):
        c = self.seq_canvas
        w = max(680, c.winfo_width())
        h = max(240, c.winfo_height())
        label_w = 215
        grid_w = w - label_w - 10
        step_w = grid_w / self.num_steps
        row_h = (h - 28) / self.num_tracks

        if event.y < 26:
            return
        r = int((event.y - 26) // row_h)
        if not (0 <= r < self.num_tracks):
            return

        if event.x < label_w - 58:
            self._trigger_drum(r)
            return

        if label_w - 54 <= event.x <= label_w - 30:
            self.muted[r] = not self.muted[r]
            self._draw_sequencer()
            return

        if label_w - 27 <= event.x <= label_w - 5:
            density = 0.25 if r in (0, 2) else 0.38
            self.grid[r] = [1 if random.random() < density else 0 for _ in range(self.num_steps)]
            if r == 0:
                self.grid[r][0] = 1
            self._draw_sequencer()
            return

        if event.x >= label_w:
            s = int((event.x - label_w) // step_w)
            if 0 <= s < self.num_steps:
                self.grid[r][s] = 0 if self.grid[r][s] else 1
                if self.grid[r][s] == 1:
                    self._trigger_drum(r)
                self._draw_sequencer()

    def _trigger_drum(self, track_idx: int, step_accent: bool = False):
        drum = self.bank["drums"][track_idx]
        band_map = {
            0: "sub",
            1: "mid",
            2: "mid",
            3: "high",
            4: "high",
            5: "high",
        }
        rate = 1.0 + random.uniform(-self.micro_jitter, self.micro_jitter) if track_idx > 0 else 1.0
        vol = 0.95 if step_accent else 0.78
        self.audio.play(
            drum["path"],
            volume=vol,
            rate=rate,
            band=band_map.get(track_idx, "mid"),
            color=drum["color"],
            label=drum["name"],
        )

    # =================================================================
    # 02 // CONTINUOUS 2D XY SYNTH PAD (MOUSE CONTROL)
    # =================================================================

    def set_synth_mode(self, mode: int):
        self.synth_mode = mode
        self._refresh_timbre_buttons()
        if self.xy_active or self.xy_hold:
            self._apply_xy_synth()

    def toggle_xy_hold(self):
        self.xy_hold = not self.xy_hold
        if self.xy_hold:
            self.xy_active = True
            self._update_pill_button(self.hold_btn, "🔒 XY HOLD: ON", ACCENT_PURPLE, "#ffffff")
            self._apply_xy_synth()
        else:
            self.xy_active = False
            self._update_pill_button(self.hold_btn, "🔒 XY HOLD: OFF", "#22303c", TEXT_MAIN)
            self._silence_xy_synth()

    def _update_xy_from_mouse(self, x_px: float, y_px: float):
        w = max(100, self.xy_canvas.winfo_width())
        h = max(100, self.xy_canvas.winfo_height())

        # Normalize X (0.0 left -> 1.0 right) and Y (0.0 bottom -> 1.0 top)
        nx = max(0.0, min(1.0, x_px / w))
        ny = max(0.0, min(1.0, 1.0 - (y_px / h)))

        self.xy_norm_x = nx
        self.xy_norm_y = ny
        self._apply_xy_synth()

    def _apply_xy_synth(self):
        # Exponential pitch mapping across 4 octaves: 55 Hz (A1) -> 880 Hz (A5)
        self.xy_freq = 55.0 * (2.0 ** (4.0 * self.xy_norm_x))
        self.xy_amp = self.xy_norm_y

        mode_Colors = {0: ACCENT_CYAN, 1: ACCENT_PURPLE, 2: ACCENT_YELLOW}
        mode_names = {0: "BUCHLA", 1: "DRONE", 2: "CRYSTAL FM"}
        col = mode_Colors.get(self.synth_mode, ACCENT_CYAN)
        mname = mode_names.get(self.synth_mode, "SYNTH")

        self.audio.set_continuous_synth(
            freq=self.xy_freq,
            amp=self.xy_amp,
            mode=self.synth_mode,
            color=col,
            label=f"{mname} // {self.xy_freq:.1f} Hz ({int(self.xy_amp * 100)}%)",
        )

        self.xy_trail.append((self.xy_norm_x, self.xy_norm_y))
        if len(self.xy_trail) > 26:
            self.xy_trail.pop(0)

    def _silence_xy_synth(self):
        self.xy_amp = 0.0
        self.audio.set_continuous_synth(freq=self.xy_freq, amp=0.0, mode=self.synth_mode)

    def _on_xy_press(self, event):
        self.xy_active = True
        self._update_xy_from_mouse(event.x, event.y)

    def _on_xy_drag(self, event):
        if self.xy_active:
            self._update_xy_from_mouse(event.x, event.y)

    def _on_xy_release(self, event):
        if not self.xy_hold:
            self.xy_active = False
            self._silence_xy_synth()

    def _draw_xy_pad(self):
        c = self.xy_canvas
        c.delete("all")
        w = max(400, c.winfo_width())
        h = max(220, c.winfo_height())

        mode_colors = {0: ACCENT_CYAN, 1: ACCENT_PURPLE, 2: ACCENT_YELLOW}
        active_col = mode_colors.get(self.synth_mode, ACCENT_CYAN)

        # 1. Vertical Octave / Frequency Reference Grid (55, 110, 220, 440, 880 Hz)
        octaves = [
            (0.00, "55 Hz (A1)"),
            (0.25, "110 Hz (A2)"),
            (0.50, "220 Hz (A3)"),
            (0.75, "440 Hz (A4)"),
            (1.00, "880 Hz (A5)"),
        ]
        for nx, lbl in octaves:
            gx = max(2, min(w - 2, nx * w))
            c.create_line(gx, 0, gx, h, fill="#151c28", width=1)
            anchor = "w" if nx < 0.1 else ("e" if nx > 0.9 else "center")
            tx = gx + 6 if nx < 0.1 else (gx - 6 if nx > 0.9 else gx)
            c.create_text(tx, h - 12, text=lbl, anchor=anchor, fill="#4a5568", font=("Menlo", 8))

        # 2. Horizontal Intensity Grid (25%, 50%, 75%)
        for ny, ilbl in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%")]:
            gy = (1.0 - ny) * h
            c.create_line(0, gy, w, gy, fill="#131924", width=1, dash=(3, 5))
            c.create_text(10, gy - 8, text=f"INTENSITY {ilbl}", anchor="w", fill="#384357", font=("Menlo", 8))

        # Axis Hints
        c.create_text(
            w - 12, 16,
            text="CLICK & DRAG MOUSE  |  RIGHT-CLICK = HOLD CONTINUOUS SOUND",
            anchor="e",
            fill=TEXT_MUTED,
            font=("Menlo", 9),
        )

        # 3. Draw Motion Trail
        if len(self.xy_trail) >= 2:
            pts = []
            for tx, ty in self.xy_trail:
                pts.extend([tx * w, (1.0 - ty) * h])
            c.create_line(
                *pts,
                fill=active_col if (self.xy_active or self.bio_sonification_active) else "#253042",
                width=2,
                smooth=True,
            )

        # 4. Draw Active Cursor Crosshair & Resonant Wave Rings
        cx = self.xy_norm_x * w
        cy = (1.0 - self.xy_norm_y) * h
        is_sounding = self.xy_active or self.xy_hold or self.bio_sonification_active

        if is_sounding and self.xy_amp > 0.01:
            # Crosshairs
            c.create_line(cx, 0, cx, h, fill="#1e2d42", width=1)
            c.create_line(0, cy, w, cy, fill="#1e2d42", width=1)

            # Pulsing concentric rings proportional to intensity (Y)
            pulse = 14.0 + self.xy_amp * 38.0 + 6.0 * math.sin(time.perf_counter() * 14.0)
            c.create_oval(
                cx - pulse, cy - pulse, cx + pulse, cy + pulse,
                outline=active_col, width=1.5,
            )
            r_inner = 7.0 + self.xy_amp * 10.0
            c.create_oval(
                cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                fill=active_col, outline="#ffffff", width=1.5,
            )

            # Live HUD Readout next to cursor
            hud_x = cx + 18 if cx < w - 180 else cx - 18
            hud_anchor = "w" if cx < w - 180 else "e"
            c.create_text(
                hud_x, cy - 10,
                text=f"{self.xy_freq:.1f} Hz",
                anchor=hud_anchor,
                fill="#ffffff",
                font=("Menlo", 11, "bold"),
            )
            c.create_text(
                hud_x, cy + 8,
                text=f"INTENSITY: {int(self.xy_amp * 100)}%",
                anchor=hud_anchor,
                fill=active_col,
                font=("Menlo", 9, "bold"),
            )
        else:
            # Idle cursor node
            c.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#222c3c", outline=TEXT_MUTED, width=1.5)
            c.create_text(
                cx, cy - 16,
                text=" trascina qui il mouse ",
                fill=TEXT_MUTED,
                font=("Menlo", 9),
            )

    # =================================================================
    # HIGH-PRECISION AUDIO SEQUENCER CLOCK THREAD
    # =================================================================

    def _precision_clock_loop(self):
        next_step_time = time.perf_counter()
        while self._alive:
            if not self.is_playing:
                time.sleep(0.005)
                next_step_time = time.perf_counter()
                continue

            now = time.perf_counter()
            if now >= next_step_time:
                step = self.current_step
                is_downbeat = (step % 4 == 0)

                for r in range(self.num_tracks):
                    if self.grid[r][step] == 1 and not self.muted[r]:
                        self._trigger_drum(r, step_accent=is_downbeat)

                self._ui_step_dirty = True
                self.current_step = (step + 1) % self.num_steps
                step_sec = 60.0 / max(40.0, float(self.bpm)) / 4.0
                next_step_time += step_sec
                if now - next_step_time > step_sec:
                    next_step_time = now + step_sec
            else:
                sleep_dur = min(0.002, max(0.0005, next_step_time - now))
                time.sleep(sleep_dur)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.current_step = 0
            self._update_pill_button(self.play_btn, "■ STOP 4/4 [SPACE]", ACCENT_MAGENTA, "#ffffff")
        else:
            self._update_pill_button(self.play_btn, "▶ PLAY 4/4 [SPACE]", ACCENT_CYAN, "#061014")
            self._draw_sequencer()

    def _on_bpm_change(self, val):
        self.bpm = int(float(val))
        self.bpm_label.configure(text=f"BPM: {self.bpm}")

    def toggle_bio_sonification(self):
        self.bio_sonification_active = not self.bio_sonification_active
        if self.bio_sonification_active:
            self._update_pill_button(self.bio_btn, "☘ BIO-SONIFY: ON", ACCENT_GREEN, "#061014")
        else:
            self._update_pill_button(self.bio_btn, "☘ BIO-SONIFY: OFF", "#22303c", TEXT_MAIN)
            if not self.xy_active and not self.xy_hold:
                self._silence_xy_synth()

    def load_preset(self, preset_name: str):
        if preset_name == "clear":
            self.grid = [[0] * 16 for _ in range(self.num_tracks)]
        elif preset_name == "ikeda":
            self.grid = [
                [1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 1, 0],
                [0, 0, 0, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0],
                [0, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0,  1, 0, 0, 0],
                [1, 0, 1, 1,  0, 1, 0, 1,  1, 0, 1, 0,  0, 1, 1, 1],
                [0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 1, 0,  0, 1, 0, 1],
                [0, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0],
            ]
        elif preset_name == "bio":
            self.grid = [
                [1, 0, 0, 0,  0, 0, 1, 0,  1, 0, 0, 0,  0, 0, 0, 0],
                [0, 0, 1, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 1, 1],
                [0, 0, 0, 0,  1, 0, 0, 1,  0, 0, 0, 0,  1, 0, 0, 0],
                [0, 1, 0, 0,  0, 0, 0, 1,  0, 1, 0, 0,  0, 0, 1, 0],
                [1, 0, 0, 1,  0, 0, 1, 0,  0, 1, 0, 0,  1, 0, 0, 1],
                [0, 1, 1, 0,  0, 1, 0, 0,  1, 0, 0, 1,  0, 1, 0, 0],
            ]
        elif preset_name == "cooper":
            self.grid = [
                [1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0,  1, 0, 0, 0],
                [0, 0, 0, 1,  0, 0, 1, 0,  0, 0, 0, 1,  0, 1, 0, 0],
                [0, 0, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0,  1, 0, 0, 1],
                [0, 1, 0, 1,  0, 0, 1, 1,  0, 1, 0, 0,  0, 1, 1, 0],
                [0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 1, 1],
                [1, 0, 0, 0,  0, 1, 0, 0,  1, 0, 0, 1,  0, 0, 1, 0],
            ]
        elif preset_name == "mutate":
            densities = [0.28, 0.25, 0.22, 0.40, 0.35, 0.30]
            for r in range(self.num_tracks):
                self.grid[r] = [1 if random.random() < densities[r] else 0 for _ in range(16)]
            self.grid[0][0] = 1
            self.grid[0][8] = 1

        self._draw_sequencer()

    # =================================================================
    # TOUCHDESIGNER-STYLE REAL-TIME AUDIO-REACTIVE VISUALIZER (30 FPS)
    # =================================================================

    def _animate_visualizer(self):
        if not self._alive:
            return
        now = time.perf_counter()

        if self._ui_step_dirty:
            self._ui_step_dirty = False
            self._draw_sequencer()

        # If Bio-Sonification is ON and user is not manually dragging, move the XY synth organically
        if self.bio_sonification_active and not self.xy_active:
            bx = 0.5 + 0.36 * math.sin(now * 0.55) + 0.12 * math.cos(now * 1.31)
            by = 0.55 + 0.30 * math.cos(now * 0.42) + 0.12 * math.sin(now * 1.73)
            self.xy_norm_x = max(0.02, min(0.98, bx))
            self.xy_norm_y = max(0.10, min(0.95, by))
            self._apply_xy_synth()

        # Render the 2D XY Synth Pad
        self._draw_xy_pad()

        sub, mid, high = self.audio.step_physics()
        total_energy = sub + mid + high

        self.bio_phase += 0.05
        bio_val = (
            0.5 * math.sin(self.bio_phase)
            + 0.3 * math.sin(self.bio_phase * 1.618)
            + 0.2 * math.cos(self.bio_phase * 2.414)
        )
        self.bio_history.append(bio_val + 0.35 * mid)
        if len(self.bio_history) > 48:
            self.bio_history.pop(0)

        c = self.vis_canvas
        c.delete("all")
        w = max(320, c.winfo_width())
        h = max(420, c.winfo_height())
        cx, cy = w * 0.5, h * 0.43

        c.create_line(0, cy, w, cy, fill="#131926", width=1)
        c.create_line(cx, 0, cx, h - 130, fill="#131926", width=1)

        halo_r = 78.0 + sub * 34.0
        c.create_oval(
            cx - halo_r, cy - halo_r, cx + halo_r, cy + halo_r,
            outline="#1c2a3e", width=1 + int(sub * 2.0),
        )

        self.vis_angle += 0.018 + total_energy * 0.022
        cos_a = math.cos(self.vis_angle)
        sin_a = math.sin(self.vis_angle)
        cos_b = math.cos(self.vis_angle * 0.63)
        sin_b = math.sin(self.vis_angle * 0.63)

        projected = []
        base_scale = min(w, h) * 0.27

        for node in self.vis_nodes:
            disp = (
                1.0
                + 0.22 * sub * math.sin(node["seed"] + now * 4.0)
                + 0.26 * mid * math.cos(node["seed"] * 1.7 + now * 6.0)
                + 0.18 * high * math.sin(node["seed"] * 3.1 + now * 18.0)
            )
            nx = node["bx"] * disp
            ny = node["by"] * disp
            nz = node["bz"] * disp

            rx = nx * cos_a - nz * sin_a
            rz = nx * sin_a + nz * cos_a
            ry = ny * cos_b - rz * sin_b
            rz2 = ny * sin_b + rz * cos_b

            fov = 2.6 / (2.6 + rz2 * 0.7)
            px = cx + rx * base_scale * fov
            py = cy + ry * base_scale * fov
            projected.append((px, py, rz2, disp))

        thresh_sq = (52.0 + mid * 24.0) ** 2
        line_col = self.audio.last_trigger_color if total_energy > 0.25 else "#1f354d"
        for i in range(len(projected)):
            x1, y1, _, _ = projected[i]
            for j in range(i + 1, len(projected)):
                x2, y2, _, _ = projected[j]
                if (x1 - x2) ** 2 + (y1 - y2) ** 2 < thresh_sq:
                    c.create_line(x1, y1, x2, y2, fill=line_col, width=1)

        for px, py, rz2, disp in projected:
            r_dot = max(2.0, (2.8 - rz2 * 0.9) + high * 2.2)
            dot_col = "#ffffff" if high > 0.6 else (ACCENT_CYAN if disp > 1.05 else ACCENT_BLUE)
            c.create_oval(px - r_dot, py - r_dot, px + r_dot, py + r_dot, fill=dot_col, outline="")

        panel_y = h - 128
        c.create_rectangle(8, panel_y, w - 8, h - 8, fill="#0d111a", outline=BORDER_DIM)
        c.create_text(
            18, panel_y + 14,
            text=f"LAST EVENT // {self.audio.last_trigger_label}",
            anchor="w",
            fill=self.audio.last_trigger_color,
            font=("Menlo", 9, "bold"),
        )

        bands = [
            ("CHOP: SUB", sub, ACCENT_CYAN),
            ("CHOP: SYNTH", mid, ACCENT_PURPLE),
            ("CHOP: MICRO", high, ACCENT_MAGENTA),
        ]
        for b_idx, (b_name, b_val, b_col) in enumerate(bands):
            by = panel_y + 32 + b_idx * 18
            c.create_text(18, by + 5, text=b_name, anchor="w", fill=TEXT_MUTED, font=("Menlo", 8))
            bar_x0 = 96
            bar_max_w = w - 114
            c.create_rectangle(bar_x0, by, bar_x0 + bar_max_w, by + 10, fill="#161c28", outline="")
            fill_w = min(bar_max_w, max(2, int(bar_max_w * min(1.0, b_val / 1.8))))
            c.create_rectangle(bar_x0, by, bar_x0 + fill_w, by + 10, fill=b_col, outline="")

        wave_y0 = panel_y + 98
        c.create_text(18, wave_y0, text="BIO-SENSOR:", anchor="w", fill=ACCENT_GREEN, font=("Menlo", 8, "bold"))
        pts = []
        wx0 = 96
        ww = w - 114
        for idx, val in enumerate(self.bio_history):
            px = wx0 + (idx / max(1, len(self.bio_history) - 1)) * ww
            py = wave_y0 - val * 12.0
            pts.extend([px, py])
        if len(pts) >= 4:
            c.create_line(*pts, fill=ACCENT_GREEN if self.bio_sonification_active else "#2e4a3b", width=1.5, smooth=True)

        self.root.after(33, self._animate_visualizer)

    # =================================================================
    # KEYBOARD BINDINGS & CLEANUP
    # =================================================================

    def _bind_keyboard(self):
        self.root.bind("<space>", lambda e: self.toggle_play())

    def _on_close(self):
        self._alive = False
        self.is_playing = False
        self.audio.shutdown()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BiomorphicSynthApp(root)
    root.mainloop()
