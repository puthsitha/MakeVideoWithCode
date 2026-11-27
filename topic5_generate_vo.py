"""
Topic 5: Dual-Voice Khmer Voiceover Generator & Video Audio Mixer
Synthesizes all 10 Khmer voiceover segments for Topic 5 (ក្តីស្រឡាញ់មិនមែនទាល់តែរបស់ថ្លៃ)
using VoxCPM2 MLX alternating smoothly between:
  - Male   (`boy`) : assets/kid_boy.wav
  - Female (`girl`): assets/kid_girl.wav
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
TARGET_DURATION_SEC = 70.0

VOICE_SAMPLES: Dict[str, Path] = {
    "boy": Path("assets/kid_boy.wav"),
    "girl": Path("assets/kid_girl.wav"),
}

SEGMENTS_DIR = Path("output/topic5_vo_segments_km")
VO_TIMELINE_PATH = Path("output/topic5_vo_khmer.wav")
TIMING_REPORT_PATH = Path("output/topic5_vo_timings.json")
NATIVE_AUDIO_BACKUP = Path("output/topic5_native_audio_only.wav")
MERGED_AUDIO_PATH = Path("output/topic5_final_audio_with_vo.wav")
FINAL_VIDEO_PATH = Path("output/topic5_reel_final.mp4")

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

TOPIC5_VO_SEGMENTS: List[Dict[str, object]] = [
    {
        "id": "00_hook",
        "voice": "girl",
        "start": 0.0,
        "end": 4.0,
        "lead_in": 0.10,
        "text": "ក្តីស្រឡាញ់មិនមែនទាល់តែរបស់ថ្លៃទេ... តែជាការយកចិត្តទុកដាក់ពិតប្រាកដ!",
    },
    {
        "id": "01_social_comparison",
        "voice": "girl",
        "start": 4.0,
        "end": 10.0,
        "lead_in": 0.20,
        "text": "អូសមើលហ្វេសប៊ុកឃើញគេបង្ហោះបាច់ផ្កាធំៗ ហាងថ្លៃៗ និងកាដូប្រ៊េនៗ...",
    },
    {
        "id": "02_feeling_small",
        "voice": "girl",
        "start": 10.0,
        "end": 16.0,
        "lead_in": 0.20,
        "text": "ពេលខ្លះ ក៏លួចប្រៀបធៀបក្នុងចិត្ត ហើយមានអារម្មណ៍តូចចិត្តតិចៗដែរ។",
    },
    {
        "id": "03_rainy_arrival",
        "voice": "boy",
        "start": 16.0,
        "end": 22.0,
        "lead_in": 0.25,
        "text": "តែពេលមេឃភ្លៀងធ្លាក់ខ្លាំង អ្នកដែលជិះម៉ូតូកាត់ភ្លៀងមករកភ្លាម គឺជាគាត់។",
    },
    {
        "id": "04_raincoat_helmet",
        "voice": "girl",
        "start": 22.0,
        "end": 29.0,
        "lead_in": 0.20,
        "text": "មិនខ្វល់ខ្លួនឯងទទឹកទេ គាត់ប្រញាប់ពាក់អាវភ្លៀង និងមួកសុវត្ថិភាពឱ្យអូនមុនគេ។",
    },
    {
        "id": "05_warm_street_snack",
        "voice": "boy",
        "start": 29.0,
        "end": 36.0,
        "lead_in": 0.20,
        "text": "ហើយថែមទាំងឆ្លៀតទិញរបស់ញ៉ាំក្តៅៗដែលអូនចូលចិត្ត លាក់ទុកក្នុងអាវកុំឱ្យត្រជាក់ទៀត។",
    },
    {
        "id": "06_bubble_pops",
        "voice": "girl",
        "start": 36.0,
        "end": 43.0,
        "lead_in": 0.25,
        "text": "ភ្លាមនោះ ភាពប្រៀបធៀបទាំងអស់ក៏រលាយបាត់... ព្រោះអ្វីដែលអូនមាន វាកក់ក្តៅជាងអ្វីទាំងអស់។",
    },
    {
        "id": "07_dry_hair_care",
        "voice": "boy",
        "start": 43.0,
        "end": 51.0,
        "lead_in": 0.20,
        "text": "ទោះជាគ្រាន់តែអង្គុយញ៉ាំគុយទាវតូបតូចធម្មតា តែបានហុចក្រដាសជូតសក់ និងចាប់សាច់ដាក់ចានឱ្យគ្នា... នោះហើយជាសុភមង្គល។",
    },
    {
        "id": "08_warm_ride_home",
        "voice": "girl",
        "start": 51.0,
        "end": 62.0,
        "lead_in": 0.25,
        "text": "របស់ថ្លៃៗអាចទិញបានតែភាពសប្បាយមួយភ្លែតទេ... តែមនុស្សដែលតែងតែបារម្ភ និងយកចិត្តទុកដាក់លើអ្នករាល់ថ្ងៃ ទើបជាកាដូដ៏មានតម្លៃបំផុត។",
    },
    {
        "id": "09_outro",
        "voice": "boy",
        "start": 62.0,
        "end": 70.0,
        "lead_in": 0.25,
        "text": "កុំភ្លេចថែរក្សាមនុស្សដែលស្រឡាញ់អ្នកដោយចិត្តស្មោះ។ ជួយថែកគាត់ខាងក្រោមផងណា!",
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


def prepare_reference_voice(role: str, src_path: Path) -> Path:
    if not src_path.exists():
        raise FileNotFoundError(f"Voice sample not found: {src_path}")
    out_ref = Path(f"output/ref_{role}_{src_path.stem}_mono.wav")
    out_ref.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_path), "-t", "14.0", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
        str(out_ref),
    ])
    print(f"[VOICE:{role.upper()}] Prepared mono 48kHz reference from {src_path} -> {out_ref}")
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


def synthesize_dual_voiceover() -> int:
    ref_map = {role: prepare_reference_voice(role, path) for role, path in VOICE_SAMPLES.items()}
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    use_mlx = is_apple_silicon()
    if use_mlx:
        from mlx_audio.tts.utils import load_model

        model_path = ensure_voxcpm2_local(HF_MLX_REPO_ID)
        print(f"[TTS-MLX] Loading {HF_MLX_REPO_ID} for Topic 5 dual-voice Khmer narration...")
        model = load_model(str(model_path))
        sr = int(getattr(model, "sample_rate", TARGET_SAMPLE_RATE))
    else:
        from voxcpm import VoxCPM

        model_path = ensure_voxcpm2_local(HF_TORCH_REPO_ID)
        model = VoxCPM.from_pretrained(str(model_path), load_denoiser=False)
        sr = int(getattr(getattr(model, "tts_model", None), "sample_rate", TARGET_SAMPLE_RATE))

    timeline = np.zeros(int(round(TARGET_DURATION_SEC * sr)), dtype=np.float32)
    timings = []

    for seg in TOPIC5_VO_SEGMENTS:
        sid = str(seg["id"])
        role = str(seg["voice"])
        ref_wav = ref_map[role]
        start_s, end_s = float(seg["start"]), float(seg["end"])
        lead_in = float(seg["lead_in"])
        slot_s = end_s - start_s
        text = str(seg["text"])

        print(f"[TTS-KM:{role.upper()}] Synthesizing {sid} ({start_s:.1f}s - {end_s:.1f}s)...")
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
        sf.write(str(SEGMENTS_DIR / f"{sid}_{role}_raw.wav"), wav, samplerate=sr, subtype="PCM_16")

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
                "voice": role,
                "sample": VOICE_SAMPLES[role].name,
                "slot_start": start_s,
                "slot_end": end_s,
                "lead_in": round(lead_in, 2),
                "raw_sec": round(raw_dur, 2),
                "fitted_sec": round(fit_dur, 2),
                "speed_factor": round(speed, 2),
                "text": text,
            }
        )
        print(f"  -> {sid} [{role}]: raw={raw_dur:.2f}s, fit={fit_dur:.2f}s (slot={slot_s:.1f}s, speed={speed:.2f}x)")

    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 0.89:
        timeline = timeline * (0.89 / peak)

    sf.write(str(VO_TIMELINE_PATH), timeline, samplerate=sr, subtype="PCM_16")
    TIMING_REPORT_PATH.write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Topic 5 dual-voice narration saved to: {VO_TIMELINE_PATH} ({TARGET_DURATION_SEC:.1f}s @ {sr}Hz)")
    return sr


def mix_voiceover_and_remux(vo_sr: int) -> None:
    if not FINAL_VIDEO_PATH.exists():
        print(f"[INFO] {FINAL_VIDEO_PATH} does not exist yet — run render_topic5_reel.py once clips are ready.")
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
    tmp_mp4 = FINAL_VIDEO_PATH.with_name("topic5_reel_final_tmp.mp4")
    subprocess.check_call([
        FFMPEG, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(FINAL_VIDEO_PATH), "-i", str(MERGED_AUDIO_PATH),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2",
        "-movflags", "+faststart", "-shortest", str(tmp_mp4),
    ])
    shutil.move(str(tmp_mp4), str(FINAL_VIDEO_PATH))
    print(f"[DONE] Remuxed Topic 5 dual-voice VO + native SFX/BGM into: {FINAL_VIDEO_PATH}")


if __name__ == "__main__":
    sr = synthesize_dual_voiceover()
    mix_voiceover_and_remux(sr)
