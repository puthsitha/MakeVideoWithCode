"""
Merge output/narration.wav + output/bgm.wav (ducked) + output/sfx.wav into output/merged_audio.wav,
then remux into output/final_video.mp4 using Apple AudioToolbox AAC (aac_at).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d

VO_PATH = Path("output/narration.wav")
BGM_PATH = Path("output/bgm.wav")
SFX_PATH = Path("output/sfx.wav")
VIDEO_PATH = Path("output/final_video.mp4")
MERGED_WAV = Path("output/merged_audio.wav")
TEMP_MP4 = Path("output/final_video_tmp.mp4")


def _to_stereo(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.stack([arr, arr], axis=1)
    return arr[:, :2]


def build_merged_audio() -> int:
    vo, sr_vo = sf.read(str(VO_PATH), dtype="float32")
    bgm, sr_bgm = sf.read(str(BGM_PATH), dtype="float32")
    if sr_vo != sr_bgm:
        raise ValueError(f"Sample rate mismatch: {sr_vo} vs {sr_bgm}")

    vo_st = _to_stereo(vo)
    bgm_st = _to_stereo(bgm)

    if SFX_PATH.exists():
        sfx, sr_sfx = sf.read(str(SFX_PATH), dtype="float32")
        if sr_sfx != sr_vo:
            raise ValueError(f"SFX sample rate mismatch: {sr_sfx} vs {sr_vo}")
        sfx_st = _to_stereo(sfx)
    else:
        sfx_st = np.zeros_like(vo_st)

    n = min(len(vo_st), len(bgm_st), len(sfx_st))
    vo_st = vo_st[:n].copy()
    bgm_st = bgm_st[:n].copy()
    sfx_st = sfx_st[:n].copy()

    # Normalize narration cleanly (~-1.5 dBFS peak)
    vo_peak = float(np.max(np.abs(vo_st)))
    if vo_peak > 0:
        vo_st *= 0.85 / vo_peak

    # Normalize BGM gently before ducking
    bgm_peak = float(np.max(np.abs(bgm_st)))
    if bgm_peak > 0:
        bgm_st *= 0.52 / bgm_peak

    # Normalize SFX cleanly so pops/whooshes/emotion SFX are crisp without overpowering speech
    sfx_peak = float(np.max(np.abs(sfx_st)))
    if sfx_peak > 0:
        sfx_st *= 0.42 / sfx_peak

    # Smooth speech-presence envelope to duck BGM under narration
    vo_mono = np.mean(np.abs(vo_st), axis=1)
    speech_env = maximum_filter1d(vo_mono, size=int(0.35 * sr_vo))
    speech_env = uniform_filter1d(speech_env, size=int(0.40 * sr_vo))
    duck_factor = np.clip(speech_env / 0.035, 0.0, 1.0)
    bgm_gain = 0.55 - 0.36 * duck_factor

    mixed = vo_st + bgm_st * bgm_gain[:, None] + sfx_st

    mixed_peak = float(np.max(np.abs(mixed)))
    if mixed_peak > 0.891:
        mixed = np.tanh(mixed / mixed_peak * 1.15) * 0.891

    sf.write(str(MERGED_WAV), mixed, samplerate=sr_vo, subtype="PCM_16")
    print(f"[AUDIO] Saved merged narration+BGM+SFX track: {MERGED_WAV} ({n/sr_vo:.2f}s @ {sr_vo}Hz)")
    return sr_vo


def remux_video_with_audio(sample_rate: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(VIDEO_PATH),
        "-i",
        str(MERGED_WAV),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac_at",
        "-b:a",
        "192k",
        "-ar",
        str(sample_rate),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-shortest",
        str(TEMP_MP4),
    ]
    subprocess.check_call(cmd)
    shutil.move(str(TEMP_MP4), str(VIDEO_PATH))
    print(f"[MUX] Merged video + narration + BGM + SFX into: {VIDEO_PATH}")


if __name__ == "__main__":
    sr = build_merged_audio()
    remux_video_with_audio(sr)
