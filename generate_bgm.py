"""
Content-Synced Emotional Background Music Generator (120.0s @ 48kHz Stereo)
Act I  (0-44s):  Contemplative D-minor felt piano + warm low pad + subtle 60 BPM clock tick
Act II (44-60s): Playful pizzicato/marimba rhythmic bounce for the satirical twist
Act III(60-120s): Uplifting Bbmaj7 -> C -> Fmaj7 resolution with warm chimes & string swell
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000
DURATION = 120.0
N_SAMPLES = int(SR * DURATION)
OUT_BGM = Path("output/bgm.wav")


def midi_to_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def synth_felt_piano(freq: float, dur: float, amp: float = 0.28) -> np.ndarray:
    """Warm muted felt-piano note with soft hammer transient and harmonic roll-off."""
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    sig = (
        0.65 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-2.8 * t)
        + 0.10 * np.sin(2 * np.pi * freq * 3.0 * t) * np.exp(-5.0 * t)
    )
    attack = np.minimum(1.0, t / 0.018)
    decay = np.exp(-1.55 * t)
    fade_out = np.minimum(1.0, (dur - t) / 0.08)
    return (sig * attack * decay * fade_out * amp).astype(np.float32)


def synth_pizzicato(freq: float, dur: float, amp: float = 0.24) -> np.ndarray:
    """Light witty plucked string/marimba note for the mid-point satirical beat."""
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    sig = 0.72 * np.sin(2 * np.pi * freq * t) + 0.28 * np.sin(2 * np.pi * freq * 2.0 * t)
    env = np.minimum(1.0, t / 0.006) * np.exp(-9.5 * t)
    return (sig * env * amp).astype(np.float32)


def synth_chime(freq: float, dur: float, amp: float = 0.16) -> np.ndarray:
    """Warm bell/celesta shimmer for Act III dawn breakthrough."""
    n = int(SR * dur)
    t = np.arange(n, dtype=np.float32) / SR
    sig = 0.7 * np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2.002 * t)
    env = np.minimum(1.0, t / 0.015) * np.exp(-1.9 * t)
    return (sig * env * amp).astype(np.float32)


def synth_clock_tick(amp: float = 0.065) -> np.ndarray:
    """Soft muffled wooden clock tick symbolizing ticking parental expectations."""
    n = int(SR * 0.045)
    t = np.arange(n, dtype=np.float32) / SR
    click = np.sin(2 * np.pi * 620.0 * t) * np.exp(-130.0 * t) + 0.4 * np.sin(2 * np.pi * 1180.0 * t) * np.exp(-190.0 * t)
    return (click * amp).astype(np.float32)


def add_note(buf: np.ndarray, start_sec: float, note: np.ndarray, pan: float = 0.0) -> None:
    """Mix mono note into stereo buffer with constant-power panning (-1..1)."""
    i0 = int(start_sec * SR)
    if i0 >= len(buf) or i0 < 0:
        return
    i1 = min(len(buf), i0 + len(note))
    seg = note[: i1 - i0]
    angle = (pan + 1.0) * 0.25 * np.pi
    buf[i0:i1, 0] += seg * np.cos(angle)
    buf[i0:i1, 1] += seg * np.sin(angle)


def add_warm_pad(buf: np.ndarray, start_sec: float, end_sec: float, freqs: List[float], amp: float = 0.075) -> None:
    """Lush detuned stereo string/organ pad."""
    i0 = int(start_sec * SR)
    i1 = min(len(buf), int(end_sec * SR))
    n = i1 - i0
    if n <= 0:
        return
    t = np.arange(n, dtype=np.float32) / SR
    dur = n / SR
    env = np.minimum(1.0, t / 1.1) * np.minimum(1.0, (dur - t) / 1.2)
    l_sig = np.zeros(n, dtype=np.float32)
    r_sig = np.zeros(n, dtype=np.float32)
    for f in freqs:
        l_sig += np.sin(2 * np.pi * (f * 0.9985) * t) + 0.25 * np.sin(2 * np.pi * f * 2.0 * t)
        r_sig += np.sin(2 * np.pi * (f * 1.0015) * t) + 0.25 * np.sin(2 * np.pi * f * 2.0 * t)
    scale = (amp / max(1, len(freqs))) * env
    buf[i0:i1, 0] += l_sig * scale
    buf[i0:i1, 1] += r_sig * scale


def generate_bgm() -> None:
    OUT_BGM.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.zeros((N_SAMPLES, 2), dtype=np.float32)

    # Act I (0-44s): Contemplative Dm -> Bb -> F -> C progression
    act1_chords: List[Tuple[List[int], List[int]]] = [
        ([50, 57, 62, 65], [38]),  # Dm
        ([46, 53, 58, 62], [34]),  # Bb
        ([41, 48, 53, 57], [41]),  # F
        ([48, 55, 60, 64], [36]),  # C
    ]
    tick_wave = synth_clock_tick(0.06)
    for bar_idx in range(11):  # 11 * 4s = 44s
        t_bar = bar_idx * 4.0
        arp_notes, bass_notes = act1_chords[bar_idx % len(act1_chords)]
        add_warm_pad(stereo, t_bar, t_bar + 4.3, [midi_to_hz(b) for b in bass_notes] + [midi_to_hz(arp_notes[0])], amp=0.08)
        add_note(stereo, t_bar, synth_felt_piano(midi_to_hz(bass_notes[0]), 3.8, amp=0.22), pan=-0.1)
        # Gentle 8-step felt piano arpeggio across 4s bar
        pattern = [0, 1, 2, 3, 2, 1, 2, 3]
        for step, idx in enumerate(pattern):
            t_note = t_bar + step * 0.5
            pan = -0.25 + (step / 7.0) * 0.5
            add_note(stereo, t_note, synth_felt_piano(midi_to_hz(arp_notes[idx]), 1.8, amp=0.19), pan=pan)
        # Subtle clock ticks during filial-duty pressure (6s - 43s)
        if 4.0 <= t_bar < 42.0:
            for beat in range(4):
                add_note(stereo, t_bar + beat * 1.0, tick_wave, pan=0.28 if beat % 2 == 0 else -0.28)

    # Act II (44-60s): Playful, slightly ironic pizzicato bounce
    act2_chords = [
        [50, 57, 62, 65],  # Dm
        [55, 62, 67, 70],  # Gm
        [57, 64, 69, 73],  # A7
        [50, 57, 62, 65],  # Dm
    ]
    for bar_idx in range(4):  # 4 * 4s = 16s (44..60s)
        t_bar = 44.0 + bar_idx * 4.0
        notes = act2_chords[bar_idx % len(act2_chords)]
        add_warm_pad(stereo, t_bar, t_bar + 4.1, [midi_to_hz(notes[0] - 12)], amp=0.05)
        for step in range(8):
            m = notes[step % len(notes)] + (12 if step % 4 == 2 else 0)
            add_note(stereo, t_bar + step * 0.5, synth_pizzicato(midi_to_hz(m), 0.7, amp=0.20), pan=(-0.3 if step % 2 == 0 else 0.3))

    # Act III (60-120s): Warm empathetic turn -> Dawn blueprint -> Inspiring resolution
    act3_chords: List[Tuple[List[int], List[int]]] = [
        ([46, 53, 58, 62, 65], [34]),  # Bbmaj7
        ([48, 55, 60, 64, 67], [36]),  # C
        ([50, 57, 62, 65, 69], [38]),  # Dm7
        ([41, 48, 53, 57, 64], [41]),  # Fmaj7
    ]
    for bar_idx in range(15):  # 15 * 4s = 60s (60..120s)
        t_bar = 60.0 + bar_idx * 4.0
        arp_notes, bass_notes = act3_chords[bar_idx % len(act3_chords)]
        swell = 0.08 + 0.05 * min(1.0, bar_idx / 10.0)
        add_warm_pad(
            stereo,
            t_bar,
            min(DURATION, t_bar + 4.5),
            [midi_to_hz(bass_notes[0]), midi_to_hz(arp_notes[1]), midi_to_hz(arp_notes[3])],
            amp=swell,
        )
        add_note(stereo, t_bar, synth_felt_piano(midi_to_hz(bass_notes[0]), 3.9, amp=0.24), pan=-0.12)
        for step, idx in enumerate([0, 1, 2, 3, 4, 3, 2, 4]):
            add_note(
                stereo,
                t_bar + step * 0.5,
                synth_felt_piano(midi_to_hz(arp_notes[idx]), 2.1, amp=0.20),
                pan=-0.22 + step * 0.06,
            )
        # Uplifting sunrise chimes starting at Act III Verse 2B (84s+)
        if t_bar >= 84.0:
            add_note(stereo, t_bar + 0.0, synth_chime(midi_to_hz(arp_notes[-1] + 12), 2.6, amp=0.14), pan=0.25)
            add_note(stereo, t_bar + 2.0, synth_chime(midi_to_hz(arp_notes[2] + 12), 2.4, amp=0.11), pan=-0.25)

    # Warm low-pass shelf + gentle stereo room delay
    sos = butter(2, 5200.0, btype="low", fs=SR, output="sos")
    stereo[:, 0] = sosfilt(sos, stereo[:, 0])
    stereo[:, 1] = sosfilt(sos, stereo[:, 1])

    delay_n = int(0.19 * SR)
    wet = np.zeros_like(stereo)
    wet[delay_n:, 0] = stereo[:-delay_n, 1] * 0.24
    wet[delay_n:, 1] = stereo[:-delay_n, 0] * 0.24
    stereo = stereo + wet

    # Fade in (1.0s) and fade out (2.5s)
    fi = int(1.0 * SR)
    fo = int(2.5 * SR)
    stereo[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
    stereo[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, None]

    # Normalize BGM cleanly to -12 dBFS peak
    peak = float(np.max(np.abs(stereo)))
    if peak > 0:
        stereo = stereo * (0.25 / peak)

    sf.write(str(OUT_BGM), stereo, samplerate=SR)
    print(f"[BGM] Generated {OUT_BGM} ({DURATION:.1f}s stereo @ {SR}Hz)")


if __name__ == "__main__":
    generate_bgm()
