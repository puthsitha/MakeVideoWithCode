"""
Topic 1: ដើមខែសេដ្ឋី ចុងខែមីកញ្ចប់ — Full 75.0s BGM + Frame-Synced SFX Generator
Produces:
  - output/topic1_bgm.wav       (75.0s @ 48kHz Stereo)
  - output/topic1_sfx.wav       (75.0s @ 48kHz Stereo)
  - output/topic1_audio_mix.wav (75.0s @ 48kHz Stereo mastered mix)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000
DURATION = 75.0
N_SAMPLES = int(SR * DURATION)

OUT_DIR = Path("output")
OUT_BGM = OUT_DIR / "topic1_bgm.wav"
OUT_SFX = OUT_DIR / "topic1_sfx.wav"
OUT_MIX = OUT_DIR / "topic1_audio_mix.wav"


def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def place(buf: np.ndarray, t_sec: float, sig: np.ndarray, pan: float = 0.0) -> None:
    i0 = int(t_sec * SR)
    if i0 < 0 or i0 >= len(buf):
        return
    i1 = min(len(buf), i0 + len(sig))
    seg = sig[: i1 - i0]
    angle = (pan + 1.0) * 0.25 * np.pi
    buf[i0:i1, 0] += seg * np.cos(angle)
    buf[i0:i1, 1] += seg * np.sin(angle)


# -------------------- SYNTH INSTRUMENTS --------------------

def synth_kick(amp: float = 0.42) -> np.ndarray:
    n = int(SR * 0.16)
    t = np.arange(n, dtype=np.float32) / SR
    f = 48.0 + 110.0 * np.exp(-t * 38.0)
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    env = np.minimum(1.0, t / 0.003) * np.exp(-t * 18.0)
    return (np.sin(phase) * env * amp).astype(np.float32)


def synth_snare(amp: float = 0.26, seed: int = 1) -> np.ndarray:
    n = int(SR * 0.13)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n).astype(np.float32)
    sos = butter(2, [1200, 6500], btype="bandpass", fs=SR, output="sos")
    body = np.sin(2.0 * np.pi * 195.0 * t) * np.exp(-t * 32.0)
    snap = sosfilt(sos, noise) * np.exp(-t * 24.0)
    return ((0.45 * body + 0.55 * snap) * amp).astype(np.float32)


def synth_hihat(amp: float = 0.10, dur: float = 0.045, seed: int = 2) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n).astype(np.float32)
    sos = butter(2, 6800.0, btype="high", fs=SR, output="sos")
    return (sosfilt(sos, noise) * np.exp(-t * 75.0) * amp).astype(np.float32)


def synth_funk_bass(freq: float, dur: float, amp: float = 0.30) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    sig = (
        0.65 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-t * 8.0)
        + 0.12 * np.sin(2 * np.pi * freq * 3.0 * t) * np.exp(-t * 14.0)
    )
    env = np.minimum(1.0, t / 0.008) * np.exp(-t * 4.2) * np.minimum(1.0, (dur - t) / 0.03)
    return (sig * env * amp).astype(np.float32)


def synth_clav_stab(freqs: List[float], dur: float, amp: float = 0.16) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.zeros(n, dtype=np.float32)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2.0 * t)
    sig /= max(1, len(freqs))
    env = np.minimum(1.0, t / 0.005) * np.exp(-t * 11.0)
    return (sig * env * amp).astype(np.float32)


def synth_sad_piano(freq: float, dur: float, amp: float = 0.24) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    sig = (
        0.70 * np.sin(2 * np.pi * freq * t)
        + 0.22 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-2.6 * t)
        + 0.08 * np.sin(2 * np.pi * freq * 3.0 * t) * np.exp(-5.0 * t)
    )
    env = np.minimum(1.0, t / 0.016) * np.exp(-1.5 * t) * np.minimum(1.0, (dur - t) / 0.08)
    return (sig * env * amp).astype(np.float32)


def synth_sad_violin(freq: float, dur: float, amp: float = 0.16) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    vib = 1.0 + 0.009 * np.sin(2 * np.pi * 5.4 * t) * np.minimum(1.0, t / 0.35)
    phase = 2.0 * np.pi * np.cumsum(freq * vib) / SR
    sig = 0.62 * np.sin(phase) + 0.26 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
    env = np.minimum(1.0, t / 0.28) * np.minimum(1.0, (dur - t) / 0.28)
    return (sig * env * amp).astype(np.float32)


def synth_lofi_chord(freqs: List[float], dur: float, amp: float = 0.18) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.zeros(n, dtype=np.float32)
    for f in freqs:
        trem = 1.0 + 0.06 * np.sin(2 * np.pi * 3.5 * t)
        sig += (0.78 * np.sin(2 * np.pi * f * t) + 0.22 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-3.0 * t)) * trem
    sig /= max(1, len(freqs))
    env = np.minimum(1.0, t / 0.025) * np.exp(-0.9 * t) * np.minimum(1.0, (dur - t) / 0.12)
    return (sig * env * amp).astype(np.float32)


def synth_chime(freq: float, dur: float = 1.6, amp: float = 0.15) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    sig = 0.72 * np.sin(2 * np.pi * freq * t) + 0.28 * np.sin(2 * np.pi * freq * 2.003 * t)
    env = np.minimum(1.0, t / 0.01) * np.exp(-2.4 * t)
    return (sig * env * amp).astype(np.float32)


# -------------------- BGM COMPOSITION (75s) --------------------

def build_bgm() -> np.ndarray:
    buf = np.zeros((N_SAMPLES, 2), dtype=np.float32)
    kick = synth_kick()
    snare = synth_snare()
    hh = synth_hihat()

    # 1. PAYDAY FUNK GROOVE (0.0s - 18.95s) @ ~120 BPM (0.5s beat, 0.25s 8th)
    bass_riff = [45, 45, 48, 50, 52, 50, 48, 43]  # A minor funk groove
    stab_chords = [
        [69, 72, 76, 79],  # Am7
        [67, 71, 74, 77],  # G7
        [65, 69, 72, 76],  # Fmaj7
        [64, 68, 71, 74],  # E7
    ]
    step = 0.25
    n_steps = int(18.85 / step)
    for i in range(n_steps):
        t = i * step
        bar = i // 8
        pos = i % 8
        if pos in (0, 3, 4):
            place(buf, t, kick, pan=0.0)
        if pos in (2, 6):
            place(buf, t, snare, pan=0.05)
        place(buf, t, hh, pan=-0.22 if i % 2 == 0 else 0.22)
        if pos in (0, 2, 3, 5, 6, 7):
            note = bass_riff[pos] + (2 if bar % 4 == 1 else 0)
            place(buf, t, synth_funk_bass(midi_to_hz(note), 0.24, amp=0.28), pan=-0.08)
        if pos in (0, 3, 6):
            ch = stab_chords[bar % len(stab_chords)]
            place(buf, t, synth_clav_stab([midi_to_hz(m) for m in ch], 0.22, amp=0.14), pan=0.25 if pos == 3 else -0.20)

    # 2. DAY 12 SHOCK: 19.0s - 21.2s is DEAD SILENCE for comic punchline!
    # 21.2s - 25.8s: subtle eerie low drone
    i0, i1 = int(21.2 * SR), int(25.8 * SR)
    t_d = np.arange(i1 - i0, dtype=np.float32) / SR
    drone = (np.sin(2 * np.pi * 65.4 * t_d) + 0.4 * np.sin(2 * np.pi * 98.0 * t_d)) * 0.06
    drone *= np.minimum(1.0, t_d / 0.5) * np.minimum(1.0, (4.6 - t_d) / 0.6)
    buf[i0:i1, 0] += drone
    buf[i0:i1, 1] += drone

    # 3. COMEDIC SAD PIANO & VIOLIN (26.0s - 38.9s)
    sad_chords = [
        ([50, 57, 62, 65], 69),  # Dm -> A4
        ([46, 53, 58, 62], 65),  # Bb -> F4
        ([43, 50, 55, 58], 67),  # Gm -> G4
        ([45, 52, 57, 61], 64),  # A7 -> E4
    ]
    for idx in range(4):
        t_bar = 26.0 + idx * 3.2
        notes, melody = sad_chords[idx]
        for s_i, m in enumerate(notes):
            place(buf, t_bar + s_i * 0.45, synth_sad_piano(midi_to_hz(m), 2.4, amp=0.22), pan=-0.2 + s_i * 0.13)
        # Sad violin enters at 30.0s (Scene 4B coffee downgrade -> 4C noodles)
        if t_bar >= 29.0:
            place(buf, t_bar + 0.2, synth_sad_violin(midi_to_hz(melody + 12), 2.8, amp=0.14), pan=0.18)

    # 4. LIGHTBULB + 3 JARS + DAY 25 + OUTRO: UPLIFTING LO-FI GROOVE (39.5s - 75.0s)
    lofi_chords = [
        [53, 57, 60, 64, 67],  # Fmaj9
        [52, 55, 59, 62, 67],  # Em7
        [50, 53, 57, 60, 64],  # Dm9
        [48, 52, 55, 59, 62],  # Cmaj9
    ]
    lofi_bass = [41, 40, 38, 36]
    bar_len = 2.4
    t_cur = 39.6
    bar_i = 0
    soft_kick = synth_kick(amp=0.24)
    soft_snare = synth_snare(amp=0.13, seed=7)
    soft_hh = synth_hihat(amp=0.055, seed=9)
    while t_cur < 73.5:
        ch = lofi_chords[bar_i % len(lofi_chords)]
        bs = lofi_bass[bar_i % len(lofi_bass)]
        place(buf, t_cur, synth_lofi_chord([midi_to_hz(m) for m in ch], bar_len + 0.2, amp=0.20), pan=-0.1)
        place(buf, t_cur, synth_funk_bass(midi_to_hz(bs), bar_len * 0.8, amp=0.18), pan=0.0)
        # Gentle lo-fi beat starts from 44.0s (3 jars) until 71.5s
        if 44.0 <= t_cur < 71.5:
            place(buf, t_cur + 0.0, soft_kick, pan=0.0)
            place(buf, t_cur + 0.6, soft_snare, pan=0.08)
            place(buf, t_cur + 1.2, soft_kick, pan=0.0)
            place(buf, t_cur + 1.5, soft_kick, pan=0.0)
            place(buf, t_cur + 1.8, soft_snare, pan=-0.08)
            for h_i in range(8):
                place(buf, t_cur + h_i * 0.3, soft_hh, pan=-0.18 if h_i % 2 == 0 else 0.18)
        # Warm uplifting shimmer chimes from 59.0s (Day 25 smile)
        if t_cur >= 58.8:
            place(buf, t_cur + 0.3, synth_chime(midi_to_hz(ch[-1] + 12), 1.8, amp=0.11), pan=0.28)
            place(buf, t_cur + 1.5, synth_chime(midi_to_hz(ch[-2] + 12), 1.6, amp=0.09), pan=-0.28)
        t_cur += bar_len
        bar_i += 1

    # Warm master EQ shelf + smooth 2.0s fade-out at end
    sos = butter(2, 6500.0, btype="low", fs=SR, output="sos")
    buf[:, 0] = sosfilt(sos, buf[:, 0])
    buf[:, 1] = sosfilt(sos, buf[:, 1])

    fo = int(2.0 * SR)
    buf[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]

    # Enforce strict silence during 19.05s - 21.15s
    buf[int(19.05 * SR) : int(21.15 * SR)] = 0.0

    peak = float(np.max(np.abs(buf)))
    if peak > 0:
        buf *= 0.36 / peak
    return buf


# -------------------- SFX SYNTHESIS (75s) --------------------

def sfx_thump(amp: float = 0.65) -> np.ndarray:
    n = int(SR * 0.25)
    t = np.arange(n, dtype=np.float32) / SR
    f = 42.0 + 130.0 * np.exp(-t * 28.0)
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    return (np.sin(phase) * np.exp(-t * 14.0) * amp).astype(np.float32)


def sfx_sizzle(dur: float = 1.1, amp: float = 0.20, seed: int = 11) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n).astype(np.float32)
    sos = butter(2, [3200, 9500], btype="bandpass", fs=SR, output="sos")
    env = np.minimum(1.0, t / 0.06) * np.minimum(1.0, (dur - t) / 0.20)
    return (sosfilt(sos, noise) * env * amp).astype(np.float32)


def sfx_slurp(amp: float = 0.32) -> np.ndarray:
    n = int(SR * 0.45)
    t = np.arange(n, dtype=np.float32) / SR
    f = 320.0 + 950.0 * (t / 0.45) ** 1.4
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    rng = np.random.default_rng(19)
    air = rng.normal(0, 0.25, n).astype(np.float32)
    env = np.sin(np.pi * t / 0.45)
    return ((0.6 * np.sin(phase) + air) * env * amp).astype(np.float32)


def sfx_chaching(amp: float = 0.42) -> np.ndarray:
    n = int(SR * 0.65)
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.zeros(n, dtype=np.float32)
    # Mechanical drawer slide + bright double bell
    mask1 = t >= 0.08
    t1 = np.maximum(0.0, t - 0.08)
    sig[mask1] += (np.sin(2 * np.pi * 2093 * t1[mask1]) + 0.5 * np.sin(2 * np.pi * 3136 * t1[mask1])) * np.exp(-t1[mask1] * 7.5)
    mask2 = t >= 0.16
    t2 = np.maximum(0.0, t - 0.16)
    sig[mask2] += (np.sin(2 * np.pi * 2637 * t2[mask2]) + 0.4 * np.sin(2 * np.pi * 3951 * t2[mask2])) * np.exp(-t2[mask2] * 6.0)
    return (sig * amp).astype(np.float32)


def sfx_whoosh(dur: float = 0.34, amp: float = 0.36, seed: int = 23) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n).astype(np.float32)
    sos = butter(2, [450, 2800], btype="bandpass", fs=SR, output="sos")
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.3
    return (sosfilt(sos, noise) * env * amp).astype(np.float32)


def sfx_qr_beep(amp: float = 0.42) -> np.ndarray:
    n = int(SR * 0.16)
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.sin(2 * np.pi * 1760 * t) + 0.3 * np.sin(2 * np.pi * 3520 * t)
    env = np.minimum(1.0, t / 0.005) * np.minimum(1.0, (0.16 - t) / 0.02)
    return (sig * env * amp).astype(np.float32)


def sfx_pop(pitch_lo: float = 280.0, pitch_hi: float = 820.0, amp: float = 0.44) -> np.ndarray:
    n = int(SR * 0.09)
    t = np.arange(n, dtype=np.float32) / SR
    f = pitch_lo + (pitch_hi - pitch_lo) * (1.0 - np.exp(-t * 50.0))
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    return (np.sin(phase) * np.exp(-t * 42.0) * amp).astype(np.float32)


def sfx_moto_vroom(amp: float = 0.28) -> np.ndarray:
    n = int(SR * 0.75)
    t = np.arange(n, dtype=np.float32) / SR
    f = 95.0 + 135.0 * np.sin(np.pi * t / 0.75)
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    sig = np.tanh(2.2 * np.sin(phase) + 0.8 * np.sin(2 * phase)) * 0.5
    env = np.sin(np.pi * t / 0.75) ** 0.8
    return (sig * env * amp).astype(np.float32)


def sfx_record_scratch(amp: float = 0.48) -> np.ndarray:
    n = int(SR * 0.32)
    t = np.arange(n, dtype=np.float32) / SR
    f = 1600.0 * np.exp(-t * 8.5) + 180.0 + 420.0 * np.sin(2 * np.pi * 26 * t)
    phase = 2.0 * np.pi * np.cumsum(np.maximum(60.0, f)) / SR
    rng = np.random.default_rng(31)
    grit = rng.normal(0, 0.25, n).astype(np.float32)
    env = np.minimum(1.0, t / 0.01) * np.exp(-t * 6.5)
    return ((0.7 * np.sin(phase) + grit) * env * amp).astype(np.float32)


def sfx_error_buzz(amp: float = 0.36) -> np.ndarray:
    n = int(SR * 0.38)
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.sign(np.sin(2 * np.pi * 145 * t)) * 0.35 + np.sin(2 * np.pi * 138 * t) * 0.45
    sos = butter(2, 900.0, btype="low", fs=SR, output="sos")
    env = np.minimum(1.0, t / 0.01) * np.minimum(1.0, (0.38 - t) / 0.05)
    return (sosfilt(sos, sig) * env * amp).astype(np.float32)


def sfx_ghost_woo(amp: float = 0.26) -> np.ndarray:
    n = int(SR * 1.1)
    t = np.arange(n, dtype=np.float32) / SR
    f = 540.0 + 140.0 * np.sin(2 * np.pi * 2.2 * t) - 120.0 * (t / 1.1)
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    env = np.sin(np.pi * t / 1.1) ** 1.2
    return (np.sin(phase) * env * amp).astype(np.float32)


def sfx_chat_ping(amp: float = 0.38) -> np.ndarray:
    n = int(SR * 0.18)
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.sin(2 * np.pi * 1046.5 * t) * np.exp(-t * 20.0) + 0.6 * np.sin(2 * np.pi * 1568.0 * np.maximum(0, t - 0.05)) * np.exp(-np.maximum(0, t - 0.05) * 18.0)
    return (sig * amp).astype(np.float32)


def sfx_tear_drop(amp: float = 0.38) -> np.ndarray:
    n = int(SR * 0.14)
    t = np.arange(n, dtype=np.float32) / SR
    f = 420.0 + 680.0 * (t / 0.14) ** 1.6
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    return (np.sin(phase) * np.exp(-t * 22.0) * amp).astype(np.float32)


def sfx_snap(amp: float = 0.45) -> np.ndarray:
    n = int(SR * 0.05)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(55)
    noise = rng.normal(0, 1, n).astype(np.float32)
    sos = butter(2, [1500, 5200], btype="bandpass", fs=SR, output="sos")
    return (sosfilt(sos, noise) * np.exp(-t * 110.0) * amp).astype(np.float32)


def sfx_ding(freq: float = 1318.5, amp: float = 0.40) -> np.ndarray:
    n = int(SR * 0.85)
    t = np.arange(n, dtype=np.float32) / SR
    sig = 0.7 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2.0 * t)
    return (sig * np.minimum(1.0, t / 0.004) * np.exp(-t * 4.2) * amp).astype(np.float32)


def sfx_coin_clink(pitch_mult: float = 1.0, amp: float = 0.32) -> np.ndarray:
    n = int(SR * 0.24)
    t = np.arange(n, dtype=np.float32) / SR
    f1, f2 = 2450.0 * pitch_mult, 3680.0 * pitch_mult
    sig = 0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)
    return (sig * np.minimum(1.0, t / 0.003) * np.exp(-t * 18.0) * amp).astype(np.float32)


def sfx_padlock_click(amp: float = 0.48) -> np.ndarray:
    n = int(SR * 0.12)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.default_rng(88)
    c1 = np.sin(2 * np.pi * 1400 * t) * np.exp(-t * 85.0)
    t2 = np.maximum(0.0, t - 0.045)
    c2 = (np.sin(2 * np.pi * 2200 * t2) + 0.3 * rng.normal(0, 1, n)) * (t >= 0.045) * np.exp(-t2 * 95.0)
    return ((c1 + c2) * amp).astype(np.float32)


def build_sfx() -> np.ndarray:
    buf = np.zeros((N_SAMPLES, 2), dtype=np.float32)

    # Scene 0 (0-3s): Impact thump + sizzling grill -> sad slurp
    place(buf, 0.02, sfx_thump(), pan=0.0)
    place(buf, 0.18, sfx_sizzle(1.0, amp=0.18), pan=-0.2)
    place(buf, 1.60, sfx_slurp(), pan=0.15)

    # Scene 1 (3-9s): Cha-ching, cowboy whip whoosh, sharp QR BEEP
    place(buf, 3.15, sfx_pop(320, 880), pan=0.0)
    place(buf, 3.45, sfx_chaching(), pan=0.1)
    place(buf, 5.20, sfx_whoosh(0.28, amp=0.38), pan=-0.2)
    place(buf, 5.60, sfx_qr_beep(), pan=0.15)
    place(buf, 6.20, sfx_pop(360, 920), pan=0.0)

    # Scene 2A (9-12s): Straw slurp + whip-pan whoosh
    place(buf, 9.35, sfx_slurp(amp=0.28), pan=0.0)
    place(buf, 11.65, sfx_whoosh(0.30, amp=0.36), pan=0.2)

    # Scene 2B (12-15s): Pop + sizzling BBQ grill
    place(buf, 12.15, sfx_pop(300, 860), pan=0.0)
    place(buf, 12.25, sfx_sizzle(1.8, amp=0.16, seed=14), pan=0.0)

    # Scene 2C (15-19s): Mini moto vroom + rapid pop-pop-pop parcels dropping
    place(buf, 15.15, sfx_moto_vroom(), pan=-0.25)
    for idx, dt in enumerate([16.0, 16.45, 16.9, 17.35, 17.8, 18.2]):
        place(buf, dt, sfx_pop(260 + idx * 35, 760 + idx * 50, amp=0.38), pan=-0.3 + idx * 0.12)

    # Scene 3 (19-26s): Heavy slam + record scratch -> 2s SILENCE -> error bzzzt + ghostly wooo
    place(buf, 18.95, sfx_thump(amp=0.72), pan=0.0)
    place(buf, 19.00, sfx_record_scratch(), pan=0.0)
    # 19.35s - 21.15s completely silent!
    place(buf, 21.25, sfx_error_buzz(), pan=0.0)
    place(buf, 22.10, sfx_ghost_woo(), pan=0.12)

    # Scene 4A (26-30s): Chat pings + frantic tap
    place(buf, 26.25, sfx_chat_ping(), pan=-0.2)
    place(buf, 26.95, sfx_chat_ping(), pan=0.2)
    place(buf, 28.00, sfx_pop(520, 1100, amp=0.40), pan=0.0)

    # Scene 4B (30-34s): Double cartoon poof-poof
    place(buf, 31.00, sfx_whoosh(0.18, amp=0.32, seed=41), pan=-0.25)
    place(buf, 32.15, sfx_whoosh(0.18, amp=0.32, seed=42), pan=0.25)

    # Scene 4C (34-39s): Crunch-snap + tear drop
    place(buf, 34.45, sfx_snap(amp=0.40), pan=-0.1)
    place(buf, 37.40, sfx_tear_drop(), pan=0.15)

    # Scene 5 (39-44s): Finger snap + bright ding + VHS rewind whoosh
    place(buf, 39.25, sfx_snap(amp=0.52), pan=0.0)
    place(buf, 39.50, sfx_ding(1318.5, amp=0.44), pan=0.0)
    place(buf, 42.90, sfx_whoosh(0.45, amp=0.38, seed=61), pan=0.0)

    # Scene 6 (44-59s): Rhythmic glass coin clink-clink-clink + padlock click
    coin_times = [45.2, 45.7, 46.2, 48.2, 48.7, 49.2, 51.4, 51.9, 52.4, 54.0, 54.5, 55.0]
    for i, ct in enumerate(coin_times):
        place(buf, ct, sfx_coin_clink(0.95 + (i % 3) * 0.08), pan=-0.35 + (i % 3) * 0.35)
    place(buf, 56.60, sfx_padlock_click(), pan=0.30)
    place(buf, 56.85, sfx_ding(1568.0, amp=0.28), pan=0.30)

    # Scene 7 (59-68s): Soft page flip + warm chime
    place(buf, 59.10, sfx_whoosh(0.22, amp=0.24, seed=71), pan=-0.1)
    place(buf, 60.20, sfx_ding(1046.5, amp=0.30), pan=0.15)
    place(buf, 60.55, sfx_ding(1318.5, amp=0.26), pan=-0.15)

    # Scene 8 (68-75s): Double soft pop on text reveal + clean chime tail
    place(buf, 68.35, sfx_pop(340, 820, amp=0.36), pan=0.0)
    place(buf, 70.40, sfx_pop(380, 900, amp=0.36), pan=0.0)
    place(buf, 72.00, sfx_ding(1568.0, amp=0.26), pan=0.0)

    peak = float(np.max(np.abs(buf)))
    if peak > 0.85:
        buf *= 0.85 / peak
    return buf


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bgm = build_bgm()
    sfx = build_sfx()

    sf.write(str(OUT_BGM), bgm, samplerate=SR, subtype="PCM_16")
    sf.write(str(OUT_SFX), sfx, samplerate=SR, subtype="PCM_16")

    mix = bgm + sfx * 0.92
    peak = float(np.max(np.abs(mix)))
    if peak > 0.92:
        mix *= 0.92 / peak
    sf.write(str(OUT_MIX), mix, samplerate=SR, subtype="PCM_16")

    print(f"[AUDIO] Saved {OUT_BGM}, {OUT_SFX}, and {OUT_MIX} ({DURATION:.1f}s @ {SR}Hz Stereo)")


if __name__ == "__main__":
    main()
