"""
Frame-Synced Motion-Graphic & Emotion SFX Generator (120.0s @ 48kHz Stereo)
Synthesizes crisp Foley & UI/Motion sound effects:
  - text_pop: bubble/pop transient synced to every kinetic Khmer text entrance
  - type_tick: subtle typewriter reveal clicks
  - draw_whoosh: resonant bandpass air/pencil sweep for draw-path animations
  - emotion SFX: scared_ping, curious_boing, sad_drop, angry_steam, tired_sigh,
                 comic_boing, loving_chime, confident_sparkle, excited_fanfare, footsteps
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000
DURATION = 120.0
N_SAMPLES = int(SR * DURATION)
OUT_SFX = Path("output/sfx.wav")


def _place(buf: np.ndarray, t_sec: float, sig: np.ndarray, pan: float = 0.0) -> None:
    i0 = int(t_sec * SR)
    if i0 < 0 or i0 >= len(buf):
        return
    i1 = min(len(buf), i0 + len(sig))
    seg = sig[: i1 - i0]
    angle = (pan + 1.0) * 0.25 * np.pi
    buf[i0:i1, 0] += seg * np.cos(angle)
    buf[i0:i1, 1] += seg * np.sin(angle)


def sfx_text_pop(amp: float = 0.45) -> np.ndarray:
    """Punchy upward pitch-swept pop for kinetic text entrances."""
    n = int(SR * 0.085)
    t = np.arange(n, dtype=np.float32) / SR
    f = 280.0 + 720.0 * (1.0 - np.exp(-t * 55.0))
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    env = np.minimum(1.0, t / 0.004) * np.exp(-t * 48.0)
    return (np.sin(phase) * env * amp).astype(np.float32)


def sfx_type_tick(amp: float = 0.14) -> np.ndarray:
    """Crisp short mechanical key tick for typewriter wipe."""
    n = int(SR * 0.024)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(13)
    noise = rng.normal(0, 1, n).astype(np.float32)
    sos = butter(2, [1800, 5500], btype="bandpass", fs=SR, output="sos")
    click = sosfilt(sos, noise) + 0.5 * np.sin(2 * np.pi * 2400 * t)
    return (click * np.exp(-t * 190.0) * amp).astype(np.float32)


def sfx_draw_whoosh(dur: float = 0.42, amp: float = 0.30) -> np.ndarray:
    """Filtered sweep for scene frame draw-path animations."""
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, n).astype(np.float32)
    sos_low = butter(2, 650.0, btype="low", fs=SR, output="sos")
    sos_hi = butter(2, [550.0, 2600.0], btype="bandpass", fs=SR, output="sos")
    low_body = sosfilt(sos_low, noise)
    hi_air = sosfilt(sos_hi, noise)
    blend = (1.0 - t / dur) * low_body + (t / dur) * hi_air * 1.4
    env = np.sin(np.pi * np.clip(t / dur, 0.0, 1.0)) ** 1.4
    return (blend * env * amp).astype(np.float32)


def sfx_curious_boing(amp: float = 0.34) -> np.ndarray:
    n = int(SR * 0.28)
    t = np.arange(n, dtype=np.float32) / SR
    f = 370.0 + 260.0 * np.sin(np.pi * np.minimum(1.0, t / 0.18))
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    return (np.sin(phase) * np.exp(-t * 10.5) * amp).astype(np.float32)


def sfx_scared_ping(amp: float = 0.32) -> np.ndarray:
    n = int(SR * 0.35)
    t = np.arange(n, dtype=np.float32) / SR
    vib = 1.0 + 0.025 * np.sin(2 * np.pi * 16.0 * t)
    sig = np.sin(2 * np.pi * 880.0 * vib * t) + 0.5 * np.sin(2 * np.pi * 1318.5 * t)
    return (sig * np.minimum(1.0, t / 0.01) * np.exp(-t * 9.0) * amp).astype(np.float32)


def sfx_sad_drop(amp: float = 0.32) -> np.ndarray:
    n = int(SR * 0.25)
    t = np.arange(n, dtype=np.float32) / SR
    f = 620.0 * np.exp(-t * 4.5) + 180.0
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    return (np.sin(phase) * np.exp(-t * 12.0) * amp).astype(np.float32)


def sfx_angry_steam(amp: float = 0.28) -> np.ndarray:
    n = int(SR * 0.34)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(77)
    noise = rng.normal(0, 1, n).astype(np.float32)
    sos = butter(2, [2200, 6800], btype="bandpass", fs=SR, output="sos")
    hiss = sosfilt(sos, noise)
    env = np.minimum(1.0, t / 0.025) * np.exp(-t * 7.5)
    return (hiss * env * amp).astype(np.float32)


def sfx_comic_boing(amp: float = 0.36) -> np.ndarray:
    n = int(SR * 0.32)
    t = np.arange(n, dtype=np.float32) / SR
    f = 220.0 + 380.0 * (1.0 - np.exp(-t * 18.0)) + 28.0 * np.sin(2 * np.pi * 18.0 * t)
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    return (np.sin(phase) * np.exp(-t * 8.0) * amp).astype(np.float32)


def sfx_sparkle_chime(base_hz: float = 784.0, amp: float = 0.28) -> np.ndarray:
    n = int(SR * 0.75)
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.zeros(n, dtype=np.float32)
    for idx, mult in enumerate((1.0, 1.25, 1.5, 2.0)):
        delay = idx * 0.065
        mask = t >= delay
        tt = np.maximum(0.0, t - delay)
        sig[mask] += np.sin(2 * np.pi * (base_hz * mult) * tt[mask]) * np.exp(-tt[mask] * 5.2) * 0.35
    return (sig * amp).astype(np.float32)


def sfx_footstep_soft(amp: float = 0.16) -> np.ndarray:
    n = int(SR * 0.07)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(99)
    thud = np.sin(2 * np.pi * 115.0 * np.exp(-t * 25.0) * t) + 0.35 * rng.normal(0, 1, n)
    sos = butter(2, 480.0, btype="low", fs=SR, output="sos")
    return (sosfilt(sos, thud) * np.exp(-t * 45.0) * amp).astype(np.float32)


def generate_sfx() -> None:
    OUT_SFX.parent.mkdir(parents=True, exist_ok=True)
    buf = np.zeros((N_SAMPLES, 2), dtype=np.float32)

    pop = sfx_text_pop()
    tick = sfx_type_tick()
    whoosh = sfx_draw_whoosh()

    # 1. Draw-path whoosh on every scene cut + keyframed sticker swoops (birdcage, rocket, torn paper)
    for t_w in (0.10, 2.25, 6.25, 19.25, 46.25, 61.05, 80.55, 81.20, 102.55):
        _place(buf, t_w, whoosh, pan=-0.15 if t_w % 2 < 1 else 0.15)

    # 2. Text Pop + Typewriter ticks synced to KINETIC_TEXT_CUES in render_local_25d.py
    text_starts = [0.35, 6.60, 13.00, 19.80, 34.20, 46.70, 53.60, 61.50, 70.80, 81.00, 92.20, 103.00]
    for i, ts in enumerate(text_starts):
        _place(buf, ts, pop, pan=0.0)
        for k in range(4):
            _place(buf, ts + 0.10 + k * 0.11, tick, pan=-0.1 + 0.06 * k)

    # 3. Stick-Character Emotion & Giphy Sticker SFX cues
    _place(buf, 1.40, sfx_scared_ping(), pan=-0.2)     # 01_intro SCARED "!"
    _place(buf, 3.35, sfx_sad_drop(), pan=0.1)         # 01_intro -> lock slam & SAD
    _place(buf, 8.20, sfx_curious_boing(), pan=0.2)    # 02_verse_1a CURIOUS "?"
    _place(buf, 13.30, sfx_curious_boing(), pan=-0.2)  # 02_verse_1a Giphy "?" swoop
    _place(buf, 25.5, sfx_sad_drop(), pan=0.0)         # 03_verse_1b SAD character enter
    _place(buf, 35.5, sfx_angry_steam(), pan=-0.25)    # 03_verse_1b ANGRY fire & steam
    _place(buf, 36.1, sfx_angry_steam(), pan=0.25)
    _place(buf, 47.0, sfx_comic_boing(), pan=-0.2)     # 04_hook_midpoint Money GIF bounce
    _place(buf, 54.5, sfx_scared_ping(), pan=0.2)      # 04_hook_midpoint SCARED reaction
    _place(buf, 66.8, sfx_sparkle_chime(659.25), pan=0.15)  # 05_verse_2a Lightbulb realization
    _place(buf, 74.5, sfx_sparkle_chime(587.33), pan=-0.15) # 05_verse_2a Hearts & RELAXED
    _place(buf, 92.5, sfx_sparkle_chime(880.0), pan=0.0)    # 06_verse_2b CONFIDENT bridge
    _place(buf, 103.8, sfx_sparkle_chime(1046.5), pan=0.0)  # 07_outro Confetti & EXCITED fanfare

    # 4. Rhythmic footsteps synced to 3D stick-man walking across the bridge (86.5s..102.0s)
    step_wave = sfx_footstep_soft()
    for step_idx in range(25):
        ts = 86.8 + step_idx * 0.62
        if ts < 102.0:
            _place(buf, ts, step_wave, pan=-0.25 + (step_idx / 25.0) * 0.5)

    peak = float(np.max(np.abs(buf)))
    if peak > 0.85:
        buf = buf * (0.85 / peak)

    sf.write(str(OUT_SFX), buf, samplerate=SR, subtype="PCM_16")
    print(f"[SFX] Generated {OUT_SFX} ({DURATION:.1f}s stereo @ {SR}Hz, peak={np.max(np.abs(buf)):.3f})")


if __name__ == "__main__":
    generate_sfx()
