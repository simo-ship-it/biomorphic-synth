# BIOMORPHIC (sintesi) — Ambient Hand-Tracked Synthesizer (C++20 / MediaPipe / Vision / CoreAudio)

**BIOMORPHIC (sintesi)** è un sintetizzatore elettronico ambient e drum machine micro-minimale in C++20 / Objective-C++ per macOS, in cui l'utente suona e scolpisce la musica in tempo reale tramite il movimento delle mani rilevato dalla videocamera (MediaPipe / Apple Vision 21-landmark hand tracking).

L'estetica visiva segue un design editoriale svizzero minimale (`#F3F1EC` carta calda e `#181715` inchiostro antracite), mentre il motore DSP in tempo reale (`CoreAudio` / `AudioToolbox`) è ispirato alle sonorità ambient e microsound di *Brian Eno*, *Ryuichi Sakamoto* e *Alva Noto*.

---

## Architettura Sonora e Interazione

### `001 — RITMO MICRO-AMBIENT`
Sequencer **4/4 a 16 step** con 4 voci percussive micro-ambient sintetizzate in tempo reale:
- **`PULSO`**: battito sub-sinusoidale vellutato a 50 Hz con attacco morbido.
- **`LEGNO`**: tocco smorzato di marimba/kalimba armonica.
- **`SOFFIO`**: fruscio d'aria e vinile filtrato passa-banda.
- **`GOCCIA`**: micro-armonica cristallina in alta frequenza.
- Include controlli di **`BPM`** (`50–140`), **`SWING`** (`0%–50%`) e **Mute** per ogni singola traccia.

### `002 — TIMBRO E SPAZIO AMBIENT (TEMPO REALE)`
Tre macro-controlli in tempo reale + oscilloscopio live nell'intestazione:
- **`CALORE`**: apertura del filtro passa-basso e ricchezza armonica su pad e note.
- **`DECAY`**: durata del decadimento e del rilascio delle note e delle micro-percussioni.
- **`SPAZIO`**: mix della nuvola di riverbero/delay multi-tap stereo con filtro caldo nel feedback.

### `003 — MANI (QUADRANTE CAMERA A 3 ZONE)`
La vista live della webcam (con filtro monocromatico caldo e tracciamento a 21 punti per mano) è divisa in tre riquadri interattivi:

1. **Riquadro Sinistro (`MANO SX · ACCORDI`)**:
   - **1–4 dita aperte** della mano sinistra scelgono l'accordo ambient (`1 = DO maj9`, `2 = SOL add9`, `3 = LA min9`, `4 = MI min9`).
   - **Asse Orizzontale (X)**: tenendo la mano a sinistra il pad suona **liscio e puro** (`LISCIO`), spostandola verso destra introduce progressivamente un **vibrato** dolce su nastro (`VIBRATO →`).
   - **Asse Verticale (Y)**: regola in tempo reale il calore e l'apertura del filtro sull'accordo.

2. **Riquadro Superiore Destro (`DX · ATMOSFERA ARMONIA`)**:
   - Portando la mano destra nel riquadro in alto si attivano 4 trasformazioni ambient sugli accordi della mano sinistra:
     - **`FIORITURA`**: fioritura lenta delle singole note dell'accordo.
     - **`RESPIRO`**: onda lenta di filtro e panning stereo.
     - **`CRISTALLO`**: armoniche eteree di quinta e ottava.
     - **`NASTRO`**: deriva analogica lenta su nastro.

3. **Riquadro Inferiore Destro (`DX · POLPASTRELLI SCALA DO`)**:
   - Tastiera divisa nelle **8 note della scala di DO Maggiore** (`DO 4`, `RE 4`, `MI 4`, `FA 4`, `SOL 4`, `LA 4`, `SI 4`, `DO 5`).
   - Basta **passare o appoggiare i polpastrelli della mano destra** sulle colonne per suonare e mantenere in **sustain** le note, regolando l'intensità con l'altezza (`Y`).

---

## Compilazione ed Esecuzione (macOS)

```bash
clang++ -std=c++20 -fobjc-arc -O3 \
  -Icpp_app \
  cpp_app/main.mm \
  cpp_app/mediapipe_hand_tracker.mm \
  -framework Cocoa \
  -framework QuartzCore \
  -framework AudioToolbox \
  -framework AVFoundation \
  -framework Vision \
  -framework CoreMedia \
  -framework CoreVideo \
  -framework Foundation \
  -Wl,-sectcreate,__TEXT,__info_plist,Info.plist \
  -o biomorphic_hand_synth

codesign -s - --force biomorphic_hand_synth
./biomorphic_hand_synth
```
