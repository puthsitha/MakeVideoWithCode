"""
Topic 4: Khmer Voiceover Generator & Video Audio Mixer
Synthesizes all 10 Khmer voiceover segments for Topic 4 (លុយតូចៗដែលស៊ីលុយធំ)
using VoxCPM2 MLX cloned from `assets/chavchet_male.wav`.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d

HF_TORCH_REPO_ID = "openbmb/VoxCPM2"
HF_MLX_REPO_ID = "mlx-community/VoxCPM2-8bit"
TARGET_SAMPLE_RATE = 48000
TARGET_DURATION_SEC = 68.0

VOICE_SAMPLE_PATH = Path("assets/king_male.wav")
SEGMENTS_DIR = Path("output/topic4_vo_segments_km")
VO_TIMELINE_PATH = Path("output/topic4_vo_khmer.wav")
TIMING_REPORT_PATH = Path("output/topic4_vo_timings.json")
NATIVE_AUDIO_BACKUP = Path("output/topic4_native_audio_only.wav")
MERGED_AUDIO_PATH = Path("output/topic4_final_audio_with_vo.wav")
FINAL_VIDEO_PATH = Path("output/topic4_reel_final.mp4")

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

TOPIC4_VO_SEGMENTS: List[Dict[str, object]] = [
    {
        "id": "00_hook",
        "start": 0.0,
        "end": 4.0,
        "lead_in": 0.10,
        "text": "មិនដែលទិញរបស់ថ្លៃផង... តែហេតុអ្វីលុយចេះតែបាត់រហូត?",
    },
    {
        "id": "01_empty_piggy",
        "start": 4.0,
        "end": 10.0,
        "lead_in": 0.20,
        "text": "រាល់ខែមិនដែលទិញឡាន ឬទូរស័ព្ទថ្មីទេ តែក្រឡុកកូនជ្រូកមើល នៅតែស្ងាត់ឈឹង!",
    },
    {
        "id": "02_xray_morning",
        "start": 10.0,
        "end": 16.0,
        "lead_in": 0.20,
        "text": "សាកឆ្លុះមើលជីវិតប្រចាំថ្ងៃ។ ព្រឹកឡើងកាហ្វេមួយកែវ គិតថាលុយតូចទេ មិនអីទេ។",
    },
    {
        "id": "03_xray_afternoon",
        "start": 16.0,
        "end": 22.0,
        "lead_in": 0.20,
        "text": "ដល់រសៀល ឃ្លានតែគុជ បូកថ្លៃដឹកដល់កន្លែងធ្វើការ លុយក៏ហើរចេញតិចៗទៀត។",
    },
    {
        "id": "04_xray_night",
        "start": 22.0,
        "end": 28.0,
        "lead_in": 0.20,
        "text": "យប់ឡើងអង្គុយអូសទូរស័ព្ទ ឃើញរបស់តូចៗគួរឱ្យស្រឡាញ់ ក៏ចុចទិញលេងៗទៀត។",
    },
    {
        "id": "05_mountain_drop",
        "start": 28.0,
        "end": 36.0,
        "lead_in": 0.45,
        "text": "តែពេលបូកលុយតូចៗទាំងអស់នោះពេញមួយឆ្នាំ... ភ្ញាក់ផ្អើល! វាស្មើនឹងម៉ូតូថ្មីមួយគ្រឿងតែម្តង!",
    },
    {
        "id": "06_patch_leaks",
        "start": 36.0,
        "end": 42.0,
        "lead_in": 0.20,
        "text": "ទូកលិច មិនមែនដោយសាររលកធំទេ តែដោយសាររន្ធលេចតូចៗ។ ដូច្នេះត្រូវបិទរន្ធលេចជាមុន!",
    },
    {
        "id": "07_smart_swaps",
        "start": 42.0,
        "end": 51.0,
        "lead_in": 0.25,
        "text": "សាកឆុងកាហ្វេខ្លួនឯងខ្លះ គិតម្ភៃបួនម៉ោងមុនចុចទិញរបស់មិនចាំបាច់ ហើយកត់ត្រាចំណាយតូចៗរាល់ថ្ងៃ។",
    },
    {
        "id": "08_piggy_grows",
        "start": 51.0,
        "end": 61.0,
        "lead_in": 0.25,
        "text": "អ្នកមិនបាច់រឹតត្បិតរហូតពិបាកខ្លួនទេ គ្រាន់តែចេះគ្រប់គ្រងលុយតូចៗឱ្យបានល្អ លុយសន្សំរបស់អ្នកនឹងកើនឡើងយ៉ាងឆាប់រហ័ស។",
    },
    {
        "id": "09_outro",
        "start": 61.0,
        "end": 68.0,
        "lead_in": 0.20,
        "text": "ចេះមើលថែលុយតូច ទើបមានលុយធំ។ ចុះរាល់ថ្ងៃអ្នកអស់លុយច្រើនលើអីជាងគេ? ខំមិនប្រាប់ផងណា!",
    },
]


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def ensure_voxcpm2_local(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        local_dir = snapshot_download(repo_id=repo_id, local_files_only=True)
        print(f"[OK] Found local VoxCPM2 cache at: {local_dir}")
        return Path(local_dir)
    except LocalEntryNotFoundError:
        local_dir = snapshot_download(repo_id=repo_id, local_files_only=False)
        return Path(local_dir)


def prepare_reference_voice(src_path: Path) -> Path:
    if not src_path.exists():
        raise FileNotFoundError(f"Voice sample not found: {src_path}")
    out_ref = Path(f"output/ref_{src_path.stem}_mono.wav")
    out_ref.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_path), "-t", "14.0", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
        str(out_ref),
    ])
    return out_ref


def time_stretch_ffmpeg(wav: np.ndarray, sr: int, speed_factor: float) -> np.ndarray:
    if speed_factor <= 1.005:
        return wav
    filters: List[str] = []
    rem = speed_factor
    while rem > 2.0:
        filters.append("atempo=2.0")
        rem /= 2.0
    filters.append(f"atempo={rem:.4f}")
    with tempfile.TemporaryDirectory() as td:
        in_p, out_p = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(in_p), wav, samplerate=sr, subtype="PCM_16")
        subprocess.check_call([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(in_p), "-af", ",".join(filters), "-ar", str(sr), "-ac", "1",
            str(out_p),
        ])
        stretched, _ = sf.read(str(out_p), dtype="float32")
    return stretched.flatten()


def normalize_segment_rms(wav: np.ndarray, target_dbfs: float = -16.5) -> np.ndarray:
    if wav.size == 0:
        return wav
    active = wav[np.abs(wav) > 0.01]
    rms = float(np.sqrt(np.mean(active * active))) if active.size > 100 else float(np.sqrt(np.mean(wav * wav)))
    if rms < 1e-6:
        return wav
    gain = float(np.clip((10 ** (target_dbfs / 20.0)) / rms, 0.25, 4.0))
    out = wav * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.89:
        out = out * (0.89 / peak)
    return out.astype(np.float32)


def synthesize_voiceover() -> int:
    ref_wav = prepare_reference_voice(VOICE_SAMPLE_PATH)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    use_mlx = is_apple_silicon()
    if use_mlx:
        from mlx_audio.tts.utils import load_model

        model_path = ensure_voxcpm2_local(HF_MLX_REPO_ID)
        print(f"[TTS-MLX] Loading {HF_MLX_REPO_ID} for Topic 4 narration...")
        model = load_model(str(model_path))
        sr = int(getattr(model, "sample_rate", TARGET_SAMPLE_RATE))
    else:
        from voxcpm import VoxCPM

        model_path = ensure_voxcpm2_local(HF_TORCH_REPO_ID)
        model = VoxCPM.from_pretrained(str(model_path), load_denoiser=False)
        sr = int(getattr(getattr(model, "tts_model", None), "sample_rate", TARGET_SAMPLE_RATE))

    timeline = np.zeros(int(round(TARGET_DURATION_SEC * sr)), dtype=np.float32)
    timings = []

    for seg in TOPIC4_VO_SEGMENTS:
        sid = str(seg["id"])
        start_s, end_s = float(seg["start"]), float(seg["end"])
        lead_in = float(seg["lead_in"])
        slot_s = end_s - start_s
        text = str(seg["text"])

        print(f"[TTS-KM] Synthesizing {sid} ({start_s:.1f}s - {end_s:.1f}s)...")
        if use_mlx:
            chunks = [
                np.asarray(res.audio, dtype=np.float32).flatten()
                for res in model.generate(text=text, ref_audio=str(ref_wav), cfg_value=2.0, inference_timesteps=10)
            ]
            wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        else:
            wav = np.asarray(
                model.generate(text=text, reference_wav_path=str(ref_wav), cfg_value=2.0, inference_timesteps=10),
                dtype=np.float32,
            ).flatten()

        nz = np.where(np.abs(wav) > 0.008)[0]
        if len(nz) > 0:
            wav = wav[max(0, int(nz[0]) - int(0.04 * sr)) : min(len(wav), int(nz[-1]) + int(0.12 * sr))]

        wav = normalize_segment_rms(wav)
        raw_dur = len(wav) / sr
        sf.write(str(SEGMENTS_DIR / f"{sid}_raw.wav"), wav, samplerate=sr, subtype="PCM_16")

        tail_pad = 0.85 if sid == "09_outro" else 0.15
        avail_s = max(0.5, slot_s - lead_in - tail_pad)
        speed = 1.0
        if raw_dur > avail_s:
            if lead_in > 0.15 and raw_dur <= (slot_s - 0.10 - tail_pad):
                lead_in = max(0.05, slot_s - tail_pad - raw_dur)
                avail_s = slot_s - lead_in - tail_pad
            if raw_dur > avail_s:
                speed = raw_dur / avail_s
                wav = time_stretch_ffmpeg(wav, sr, speed)

        fade_n = min(int(0.025 * sr), len(wav) // 4)
        if fade_n > 0:
            ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            wav[:fade_n] *= ramp
            wav[-fade_n:] *= ramp[::-1]

        max_samples = int(round((slot_s - lead_in) * sr))
        wav = wav[:max_samples]
        place_idx = int(round((start_s + lead_in) * sr))
        end_idx = min(place_idx + len(wav), len(timeline))
        timeline[place_idx:end_idx] += wav[: end_idx - place_idx]

        fit_dur = len(wav) / sr
        timings.append(
            {
                "id": sid,
                "slot_start": start_s,
                "slot_end": end_s,
                "lead_in": round(lead_in, 2),
                "raw_sec": round(raw_dur, 2),
                "fitted_sec": round(fit_dur, 2),
                "speed_factor": round(speed, 2),
                "text": text,
            }
        )
        print(f"  -> {sid}: raw={raw_dur:.2f}s, fit={fit_dur:.2f}s (slot={slot_s:.1f}s, speed={speed:.2f}x)")

    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 0.89:
        timeline = timeline * (0.89 / peak)

    sf.write(str(VO_TIMELINE_PATH), timeline, samplerate=sr, subtype="PCM_16")
    TIMING_REPORT_PATH.write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Topic 4 Khmer voiceover saved to: {VO_TIMELINE_PATH} ({TARGET_DURATION_SEC:.1f}s @ {sr}Hz)")
    return sr


def mix_voiceover_and_remux(vo_sr: int) -> None:
    if not FINAL_VIDEO_PATH.exists():
        print(f"[INFO] {FINAL_VIDEO_PATH} does not exist yet — run render_topic4_reel.py once clips are ready.")
        return

    if not NATIVE_AUDIO_BACKUP.exists():
        subprocess.check_call([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(FINAL_VIDEO_PATH), "-vn", "-ac", "2", "-ar", str(TARGET_SAMPLE_RATE),
            str(NATIVE_AUDIO_BACKUP),
        ])

    bg, _ = sf.read(str(NATIVE_AUDIO_BACKUP), dtype="float32")
    vo_raw, sr_vo = sf.read(str(VO_TIMELINE_PATH), dtype="float32")
    if sr_vo != TARGET_SAMPLE_RATE:
        with tempfile.TemporaryDirectory() as td:
            res_p = Path(td) / "vo_48k.wav"
            subprocess.check_call([
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(VO_TIMELINE_PATH), "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
                str(res_p),
            ])
            vo_raw, _ = sf.read(str(res_p), dtype="float32")

    sr = TARGET_SAMPLE_RATE
    vo_mono = vo_raw if vo_raw.ndim == 1 else np.mean(vo_raw, axis=1)
    bg_st = bg[:, :2] if bg.ndim == 2 else np.stack([bg, bg], axis=1)
    n = min(len(bg_st), len(vo_mono))
    bg_st, vo_mono = bg_st[:n].copy(), vo_mono[:n].copy()
    vo_st = np.stack([vo_mono, vo_mono], axis=1)

    env = maximum_filter1d(np.abs(vo_mono), size=int(0.28 * sr))
    env = uniform_filter1d(env, size=int(0.32 * sr))
    duck = np.clip(env / 0.030, 0.0, 1.0)
    bg_gain = 0.78 - 0.54 * duck

    mixed = vo_st * 1.02 + bg_st * bg_gain[:, None]
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.912:
        mixed = np.tanh(mixed / peak * 1.15) * 0.912 / np.tanh(1.15)

    sf.write(str(MERGED_AUDIO_PATH), mixed, samplerate=sr, subtype="PCM_16")
    tmp_mp4 = FINAL_VIDEO_PATH.with_name("topic4_reel_final_tmp.mp4")
    subprocess.check_call([
        FFMPEG, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(FINAL_VIDEO_PATH), "-i", str(MERGED_AUDIO_PATH),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2",
        "-movflags", "+faststart", "-shortest", str(tmp_mp4),
    ])
    shutil.move(str(tmp_mp4), str(FINAL_VIDEO_PATH))
    print(f"[DONE] Remuxed Topic 4 VO + native SFX/BGM into: {FINAL_VIDEO_PATH}")


if __name__ == "__main__":
    sr = synthesize_voiceover()
    mix_voiceover_and_remux(sr)
