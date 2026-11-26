# Khmer 9:16 Video Production Pipeline

An automated vertical (`1080x1920`, `9:16`) Khmer video production toolkit combining zero-shot Khmer voice cloning (VoxCPM2), procedural BGM & frame-synced SFX synthesis, native macOS Khmer typography rendering, procedural 2D/3D stick-character animation, AI Image-to-Video (I2V) batch generation, automated QA gating, and an interactive browser preview player.

---

## Pipelines Included

### 1. Filial Duty vs Your Own Dreams (*កាតព្វកិច្ចកូន ឬ ក្តីសុបិនខ្លួនឯង*)
- **Format:** `1080x1920` (`9:16`), `30 fps`, `120.0s`
- **Visuals:** Procedural draw-path motion graphics, 3D cel-shaded expressive stick characters (`stickman_3d.py`), animated stickers (`sticker_layer.py`), and kinetic Khmer typography.
- **Audio:** Zero-shot Khmer narration (`VoxCPM2`), 3-act procedural BGM, and frame-synced emotion/UI SFX with automatic voice ducking.
- **Preview & QA:** Synced `subtitles.srt`, browser studio player (`index.html`), and frame-by-frame QA gate (`qa_gate.py`).

### 2. Topic 1 Reel — Payday Rich, Month-End Noodles (*ដើមខែសេដ្ឋី ចុងខែមីកញ្ចប់*)
- **Format:** `1080x1920` (`9:16`), `24 fps`, `75.0s` (13 scenes)
- **Visuals:** Batch Image-to-Video (I2V) clip generation via Google Veo, fal.ai, or Runway (`generate_topic1_i2v.py`), plus a local 2D cartoon-physics fallback animator (`animate_topic1_scenes.py`).
- **Audio:** Zero-shot Khmer voiceover cloned from `assets/chavchet_male.wav` (`topic1_generate_vo.py`), procedural BGM/SFX (`topic1_generate_audio.py`), and automatic ducking over native clip audio.

---

## Prerequisites

- **OS:** macOS (Apple Silicon recommended for MLX voice cloning and native AppKit Khmer text rendering; Linux/CUDA also supported for PyTorch `voxcpm`)
- **System Tools:** `ffmpeg` / `ffprobe`, Node.js & `npm`, Python 3.10+, Swift compiler (`swiftc` on macOS for Khmer text rendering)
- **Python Packages:** `numpy`, `scipy`, `soundfile`, `opencv-python`, `Pillow`, `requests`, `huggingface_hub`, plus either `mlx-audio` (Apple Silicon) or `voxcpm` + `torch` (CUDA)

---

## Setup

### 1. Verify system dependencies
```bash
npm run check
# or
bash setup_env.sh
```

### 2. Install Node & Python dependencies
```bash
npm install
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy soundfile opencv-python Pillow requests huggingface_hub
# Apple Silicon (MLX):
pip install mlx-audio
# Or CUDA / PyTorch:
# pip install voxcpm torch
```

### 3. Build the Khmer text renderer (macOS)
Compiles `render_khmer_text.swift` into `bin/khmer_text_render` (uses `KhmerMN-Bold` / `KhmerSangamMN` via AppKit for accurate Khmer shaping and subscripts):
```bash
mkdir -p bin
swiftc render_khmer_text.swift -o bin/khmer_text_render
```

### 4. Configure API keys (for Topic 1 I2V generation)
Copy `.env.example` to `.env` and set **one** provider key:
```bash
cp .env.example .env
```

Supported providers in `.env`:
| Variable | Provider |
| :--- | :--- |
| `GEMINI_API_KEY` | Google Veo (Gemini API) |
| `FAL_KEY` | fal.ai (Kling / Hailuo / Veo) |
| `RUNWAYML_API_SECRET` | Runway Gen-4 Turbo |

---

## Usage

### Pipeline 1: *Filial Duty vs Your Own Dreams* (120s)

1. **Generate Khmer narration** (zero-shot voice cloning via VoxCPM2):
   ```bash
   npm run audio
   # or: .venv/bin/python generate_audio.py
   ```
2. **Generate BGM & SFX** and sync subtitles/preview player:
   ```bash
   .venv/bin/python generate_bgm.py
   .venv/bin/python generate_sfx.py
   .venv/bin/python sync_khmer_subtitles.py
   ```
3. **Render the 120s 1080p30 procedural motion-graphics video** (writes `output/final_video.mp4`):
   ```bash
   .venv/bin/python render_local_25d.py
   ```
   *(To re-merge audio tracks onto an existing render, run `.venv/bin/python merge_audio_video.py`. For the external generative-video CLI blueprint, see `npm run render` / `render_video.sh`.)*
4. **Run the automated QA gate** (checks 1080×1920 @ 30fps, consistency, artifacting, grid alignment, and color LUT adherence; passes at `>= 8.0 / 10.0`):
   ```bash
   npm run qa
   # or: .venv/bin/python qa_gate.py --video output/final_video.mp4
   ```
5. **Preview in browser:** Open `index.html` via a local static server (e.g., `python3 -m http.server`) to inspect the rendered video with synced Khmer subtitles.

---

### Pipeline 2: *Topic 1 Reel* (75s)

1. **Generate animated scene clips** into `topic1_scenes/clips/`:
   - **Option A — Cloud I2V API** (requires key in `.env`):
     ```bash
     # Test a single scene first
     .venv/bin/python generate_topic1_i2v.py --scenes 00_hook

     # Generate all 13 scenes and assemble the final reel
     .venv/bin/python generate_topic1_i2v.py --force --assemble
     ```
   - **Option B — Local 2D motion-graphics animator** (offline, no API key):
     ```bash
     .venv/bin/python animate_topic1_scenes.py --force
     ```
2. **Assemble the 75s reel** with Khmer caption overlays (`output/topic1_reel_final.mp4`):
   ```bash
   .venv/bin/python render_topic1_reel.py
   ```
3. **Synthesize & mix Khmer voiceover** (clones `assets/chavchet_male.wav`, ducks native audio under speech, and remuxes losslessly):
   ```bash
   .venv/bin/python topic1_generate_vo.py
   ```
   *(Optional standalone procedural BGM + SFX mix: `.venv/bin/python topic1_generate_audio.py`.)*

---

## Project Structure

```text
.
├── assets/                       # Reference voice samples (.wav) for zero-shot cloning
├── bin/                          # Compiled native binaries (khmer_text_render, gitignored)
├── output/                       # Generated audio, video renders, and caches (gitignored)
├── topic1_scenes/                # Topic 1 keyframes, prompt manifests, VO script, & clips/
├── index.html                    # Interactive 9:16 preview player
├── package.json                  # npm scripts (check, audio, render, qa)
├── setup_env.sh                  # System dependency checker
├── render_khmer_text.swift       # macOS AppKit Khmer text-to-PNG renderer
├── prompts.json                  # Scene manifest & LUT config for Pipeline 1
├── script_with_emotions.md       # Pipeline 1 Khmer narration, emotion & SFX cue sheet
├── subtitles.srt                 # Synced Khmer subtitles for Pipeline 1
├── generate_audio.py             # VoxCPM2 Khmer narration generator (120s)
├── generate_bgm.py               # Procedural 3-act BGM synthesizer (120s)
├── generate_sfx.py               # Frame-synced Foley & emotion SFX generator (120s)
├── merge_audio_video.py          # 3-layer ducked audio mixer & video remuxer
├── motion_scenes.py              # Procedural draw-path scene renderer
├── stickman.py                   # 2D articulated stick-character renderer
├── stickman_3d.py                # 3D cel-shaded expressive stick-character engine
├── sticker_layer.py              # Keyframe-animated GIF sticker compositor
├── prepare_keyframes.py          # Oversampled 9:16 editorial keyframe generator
├── render_local_25d.py           # Full 120s local motion-graphic video renderer
├── render_video.sh               # Generative video CLI render & grade blueprint
├── sync_khmer_subtitles.py       # Syncs subtitles.srt & index.html to audio timings
├── qa_gate.py                    # Automated frame-by-frame QA scoring gate
├── generate_topic1_i2v.py        # Batch I2V generator (Veo / fal.ai / Runway)
├── animate_topic1_scenes.py      # Local 2D cartoon-FX per-scene animator
├── render_topic1_reel.py         # Topic 1 75s video-clip reel assembler
├── topic1_generate_audio.py      # Topic 1 procedural BGM & SFX generator
├── topic1_generate_vo.py         # Topic 1 Khmer VO cloning & auto-ducked mixer
├── .env.example                  # Template for I2V provider API keys
└── .gitignore                    # Excludes secrets, venv, node_modules, bin, & renders
```
