"""
Topic 2: Dual-Voice Khmer Voiceover Generator & Video Audio Mixer
1. Synthesizes all 10 Khmer voiceover segments for Topic 2 using VoxCPM2 Zero-Shot
   Voice Cloning (MLX `mlx-community/VoxCPM2-8bit`), alternating smoothly between:
     - Male   (`M`): assets/chavchet_male.wav
     - Female (`F`): assets/khunneary_female.wav
2. Fits each segment cleanly into its exact scene slot (pitch-preserving ffmpeg atempo
   only when slightly longer than the slot — never cutting off words).
3. Saves isolated dual-voice track (`output/topic2_vo_khmer.wav`) and timing report
   (`output/topic2_vo_timings.json`). If `output/topic2_reel_final.mp4` already exists,
   also ducks native SFX/BGM underneath the dual-voice track and remuxes losslessly.
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
TARGET_DURATION_SEC = 72.0

VOICE_SAMPLES: Dict[str, Path] = {
    "male": Path("assets/chavchet_male.wav"),
    "female": Path("assets/khunneary_female.wav"),
}

SEGMENTS_DIR = Path("output/topic2_vo_segments_km")
VO_TIMELINE_PATH = Path("output/topic2_vo_khmer.wav")
TIMING_REPORT_PATH = Path("output/topic2_vo_timings.json")
NATIVE_AUDIO_BACKUP = Path("output/topic2_native_audio_only.wav")
MERGED_AUDIO_PATH = Path("output/topic2_final_audio_with_vo.wav")
FINAL_VIDEO_PATH = Path("output/topic2_reel_final.mp4")

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

TOPIC2_VO_SEGMENTS: List[Dict[str, object]] = [
    {
        "id": "00_hook",
        "voice": "female",
        "start": 0.0,
        "end": 4.0,
        "lead_in": 0.12,
        "text": "ធ្លាប់អត់? អង្គុយជិតគ្នាតែមួយតុ... តែអារម្មណ៍ដូចនៅឆ្ងាយគ្នាណាស់។",
    },
    {
        "id": "01_date_night_start",
        "voice": "male",
        "start": 4.0,
        "end": 10.0,
        "lead_in": 0.20,
        "text": "ណាត់គ្នាញ៉ាំបាយល្ងាច ព្រោះចង់ចំណាយពេលល្អៗជាមួយគ្នា បន្ទាប់ពីហត់ពេញមួយថ្ងៃ។",
    },
    {
        "id": "02_notification_ping",
        "voice": "male",
        "start": 10.0,
        "end": 16.0,
        "lead_in": 0.35,
        "text": "តែគ្រាន់តែសារលោតមកមួយភ្លែត... ខ្ញុំក៏លើកទូរស័ព្ទមើល ហើយអូនក៏លើកមើលតាមដែរ។",
    },
    {
        "id": "03_two_worlds",
        "voice": "female",
        "start": 16.0,
        "end": 24.0,
        "lead_in": 0.25,
        "text": "យូរៗទៅ ម្នាក់ៗជាប់ក្នុងពិភពរៀងខ្លួន។ ញញឹមជាមួយអេក្រង់ទូរស័ព្ទ តែស្ងាត់ជ្រងំជាមួយមនុស្សនៅចំពោះមុខ។",
    },
    {
        "id": "04_missed_moment",
        "voice": "female",
        "start": 24.0,
        "end": 32.0,
        "lead_in": 0.25,
        "text": "ពេលអូនព្យាយាមរករឿងនិយាយ តែបងឆ្លើយទាំងភ្នែកមិនដកពីអេក្រង់... ការសន្ទនាក៏រលត់ទៅបន្តិចម្តងៗ។",
    },
    {
        "id": "05_cracking_heart",
        "voice": "male",
        "start": 32.0,
        "end": 39.0,
        "lead_in": 0.25,
        "text": "ពេលខ្លះ អ្វីដែលធ្វើឱ្យគូស្នេហ៍ឃ្លាតឆ្ងាយ មិនមែនជារឿងធំទេ តែជាការមិនបានយកចិត្តទុកដាក់គ្នា។",
    },
    {
        "id": "06_phone_facedown",
        "voice": "male",
        "start": 39.0,
        "end": 46.0,
        "lead_in": 0.35,
        "text": "សាកងើបមុខឡើងមើលមនុស្សក្បែរខ្លួន ហើយផ្កាប់ទូរស័ព្ទចុះលើតុមួយភ្លែតមើល៍។",
    },
    {
        "id": "07_small_khmer_gesture",
        "voice": "female",
        "start": 46.0,
        "end": 54.0,
        "lead_in": 0.25,
        "text": "គ្រាន់តែចាក់តែឱ្យគ្នា ហុចស្លាបព្រា ឬចាប់ម្ហូបដាក់ចាន... រឿងតូចៗប៉ុណ្ណឹង ក៏ធ្វើឱ្យចិត្តកក់ក្តៅវិញដែរ។",
    },
    {
        "id": "08_wall_dissolves",
        "voice": "male",
        "start": 54.0,
        "end": 64.0,
        "lead_in": 0.30,
        "text": "ពេលទាំងពីរនាក់ព្រមដាក់ទូរស័ព្ទចុះ ជញ្ជាំងមើលមិនឃើញក៏រលាយបាត់។ សំណើច និងភាពកក់ក្តៅពិតប្រាកដ ក៏ត្រឡប់មកវិញភ្លាម។",
    },
    {
        "id": "09_outro",
        "voice": "female",
        "start": 64.0,
        "end": 72.0,
        "lead_in": 0.25,
        "text": "កុំឱ្យទូរស័ព្ទដណ្តើមពេលវេលាមនុស្សដែលអ្នកស្រឡាញ់។ កុំភ្លេចថែកមនុស្សដែលអ្នកចង់ញ៉ាំបាយជាមួយផងណា!",
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
        print(f"[DOWNLOAD] Downloading {repo_id} from Hugging Face...")
        local_dir = snapshot_download(repo_id=repo_id, local_files_only=False)
        return Path(local_dir)


def prepare_reference_voice(role: str, src_path: Path) -> Path:
    """Convert sample voice to clean <=14s mono 48kHz WAV for optimal VoxCPM2 cloning stability."""
    if not src_path.exists():
        raise FileNotFoundError(f"Voice sample not found: {src_path}")
    out_ref = Path(f"output/ref_{role}_{src_path.stem}_mono.wav")
    out_ref.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_path),
        "-t", "14.0",
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
        str(out_ref),
    ]
    subprocess.check_call(cmd)
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
    af = ",".join(filters)

    with tempfile.TemporaryDirectory() as td:
        in_p = Path(td) / "in.wav"
        out_p = Path(td) / "out.wav"
        sf.write(str(in_p), wav, samplerate=sr, subtype="PCM_16")
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(in_p), "-af", af, "-ar", str(sr), "-ac", "1",
            str(out_p),
        ]
        subprocess.check_call(cmd)
        stretched, _ = sf.read(str(out_p), dtype="float32")
    return stretched.flatten()


def normalize_segment_rms(wav: np.ndarray, target_dbfs: float = -16.5) -> np.ndarray:
    if wav.size == 0:
        return wav
    active = wav[np.abs(wav) > 0.01]
    rms = float(np.sqrt(np.mean(active * active))) if active.size > 100 else float(np.sqrt(np.mean(wav * wav)))
    if rms < 1e-6:
        return wav
    target = 10 ** (target_dbfs / 20.0)
    gain = float(np.clip(target / rms, 0.25, 4.0))
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
        print(f"[TTS-MLX] Loading {HF_MLX_REPO_ID} for Topic 2 dual-voice Khmer narration...")
        model = load_model(str(model_path))
        sr = int(getattr(model, "sample_rate", TARGET_SAMPLE_RATE))
    else:
        from voxcpm import VoxCPM

        model_path = ensure_voxcpm2_local(HF_TORCH_REPO_ID)
        model = VoxCPM.from_pretrained(str(model_path), load_denoiser=False)
        sr = int(getattr(getattr(model, "tts_model", None), "sample_rate", TARGET_SAMPLE_RATE))

    timeline = np.zeros(int(round(TARGET_DURATION_SEC * sr)), dtype=np.float32)
    timings = []

    for seg in TOPIC2_VO_SEGMENTS:
        sid = str(seg["id"])
        role = str(seg["voice"])
        ref_wav = ref_map[role]
        start_s = float(seg["start"])
        end_s = float(seg["end"])
        lead_in = float(seg["lead_in"])
        slot_s = end_s - start_s
        text = str(seg["text"])

        print(f"[TTS-KM:{role.upper()}] Synthesizing {sid} ({start_s:.1f}s - {end_s:.1f}s)...")
        if use_mlx:
            chunks = [
                np.asarray(res.audio, dtype=np.float32).flatten()
                for res in model.generate(
                    text=text,
                    ref_audio=str(ref_wav),
                    cfg_value=2.0,
                    inference_timesteps=10,
                )
            ]
            wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        else:
            wav = np.asarray(
                model.generate(
                    text=text,
                    reference_wav_path=str(ref_wav),
                    cfg_value=2.0,
                    inference_timesteps=10,
                ),
                dtype=np.float32,
            ).flatten()

        nz = np.where(np.abs(wav) > 0.008)[0]
        if len(nz) > 0:
            i0 = max(0, int(nz[0]) - int(0.04 * sr))
            i1 = min(len(wav), int(nz[-1]) + int(0.12 * sr))
            wav = wav[i0:i1]

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
        if len(wav) > max_samples:
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
    print(f"[DONE] Dual-voice Khmer narration saved to: {VO_TIMELINE_PATH} ({TARGET_DURATION_SEC:.1f}s @ {sr}Hz)")
    return sr


def mix_voiceover_and_remux(vo_sr: int) -> None:
    if not FINAL_VIDEO_PATH.exists():
        print(f"[INFO] {FINAL_VIDEO_PATH} does not exist yet — run render_topic2_reel.py once clips are ready.")
        return

    if not NATIVE_AUDIO_BACKUP.exists():
        subprocess.check_call([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(FINAL_VIDEO_PATH),
            "-vn", "-ac", "2", "-ar", str(TARGET_SAMPLE_RATE),
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
    bg_st = bg_st[:n].copy()
    vo_mono = vo_mono[:n].copy()
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
    tmp_mp4 = FINAL_VIDEO_PATH.with_name("topic2_reel_final_tmp.mp4")
    subprocess.check_call([
        FFMPEG, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(FINAL_VIDEO_PATH),
        "-i", str(MERGED_AUDIO_PATH),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2",
        "-movflags", "+faststart", "-shortest",
        str(tmp_mp4),
    ])
    shutil.move(str(tmp_mp4), str(FINAL_VIDEO_PATH))
    print(f"[DONE] Remuxed Topic 2 dual-voice VO + native SFX/BGM into: {FINAL_VIDEO_PATH}")


def main() -> None:
    sr = synthesize_dual_voiceover()
    mix_voiceover_and_remux(sr)


if __name__ == "__main__":
    main()
