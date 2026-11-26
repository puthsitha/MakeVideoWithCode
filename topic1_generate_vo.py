"""
Topic 1: Khmer Voiceover Generator & Video Audio Mixer
1. Synthesizes all 13 Khmer voiceover segments for Topic 1 using VoxCPM2 Zero-Shot
   Voice Cloning (MLX `mlx-community/VoxCPM2-8bit`) cloned from `assets/chavchet_male.wav`.
2. Fits each segment cleanly into its exact scene slot (using pitch-preserving ffmpeg
   `atempo` only when slightly longer than the slot — never cutting off words).
3. Mixes the Khmer voiceover over the native SFX/BGM audio inside `output/topic1_reel_final.mp4`
   with smooth automatic ducking underneath speech, and remuxes with `-c:v copy` (lossless video).
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
TARGET_DURATION_SEC = 75.0

VOICE_SAMPLE_PATH = Path("assets/chavchet_male.wav")
SEGMENTS_DIR = Path("output/topic1_vo_segments_km")
VO_TIMELINE_PATH = Path("output/topic1_vo_khmer.wav")
TIMING_REPORT_PATH = Path("output/topic1_vo_timings.json")
NATIVE_AUDIO_BACKUP = Path("output/topic1_native_audio_only.wav")
MERGED_AUDIO_PATH = Path("output/topic1_final_audio_with_vo.wav")
FINAL_VIDEO_PATH = Path("output/topic1_reel_final.mp4")

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# Khmer TTS-optimized script: all numbers spelled out in Khmer words, timed to all 13 scenes.
TOPIC1_VO_SEGMENTS: List[Dict[str, object]] = [
    {
        "id": "00_hook",
        "start": 0.0,
        "end": 3.0,
        "lead_in": 0.05,
        "text": "ដើមខែសេដ្ឋី... តែចុងខែមីកញ្ចប់!",
    },
    {
        "id": "01_payday_khqr",
        "start": 3.0,
        "end": 9.0,
        "lead_in": 0.15,
        "text": "ថ្ងៃទីមួយ ប្រាក់ខែទើបតែចូលភ្លាម អារម្មណ៍ដូចសេដ្ឋី! ឃើញអីស្កេនភ្លាម មិនបាច់សួរតម្លៃទេ។",
    },
    {
        "id": "02a_iced_coffee",
        "start": 9.0,
        "end": 12.0,
        "lead_in": 0.10,
        "text": "កាហ្វេប្រ៊េនថ្លៃៗ ផឹករាល់ថ្ងៃមិនខ្វល់។",
    },
    {
        "id": "02b_treat_bbq",
        "start": 12.0,
        "end": 15.0,
        "lead_in": 0.10,
        "text": "ជួបមិត្តភក្តិភ្លាម ល្ងាចនេះប៉ាវមួយតុតែម្តង!",
    },
    {
        "id": "02c_delivery_parcels",
        "start": 15.0,
        "end": 19.0,
        "lead_in": 0.10,
        "text": "ដេកលេងទូរស័ព្ទទំនេរៗ ចុចទិញអនឡាញដឹកដល់ផ្ទះពេញបន្ទប់។",
    },
    {
        "id": "03_day12_shock",
        "start": 19.0,
        "end": 26.0,
        "lead_in": 0.90,  # Let the record scratch hit cleanly before speaking!
        "text": "បើកមើលគណនីធនាគារភ្លាម... ហាស? ទើបតែថ្ងៃទីដប់ពីរសោះ លុយក្នុងកុងសល់តែពីរពាន់ប្រាំរយរៀល!",
    },
    {
        "id": "04a_airplane_mode",
        "start": 26.0,
        "end": 30.0,
        "lead_in": 0.12,
        "text": "មិត្តភក្តិឆាតហៅជុំគ្នាល្ងាចនេះ... ប្រញាប់បិទសេវាពួនសិន!",
    },
    {
        "id": "04b_coffee_downgrade",
        "start": 30.0,
        "end": 34.0,
        "lead_in": 0.12,
        "text": "ពីកាហ្វេប្រ៊េន មកកាហ្វេថង់ ចុងក្រោយផឹកទឹកសុទ្ធ។",
    },
    {
        "id": "04c_noodles_bbq_photo",
        "start": 34.0,
        "end": 39.0,
        "lead_in": 0.15,
        "text": "ដល់ម៉ោងបាយល្ងាច កាច់មីកញ្ចប់ពាក់កណ្តាល មើលរូបសាច់អាំងក្លែមទឹកភ្នែក។",
    },
    {
        "id": "05_lightbulb_idea",
        "start": 39.0,
        "end": 44.0,
        "lead_in": 0.15,
        "text": "ឈប់សិន! ប្រសិនបើអ្នកមិនចង់ជួបរឿងបែបនេះរាល់ខែទេ សាកប្តូរវិធីគ្រប់គ្រងលុយថ្មី។",
    },
    {
        "id": "06_three_jars",
        "start": 44.0,
        "end": 59.0,
        "lead_in": 0.20,
        "text": (
            "ពេលប្រាក់ខែចូលភ្លាម សាកបែងចែកលុយជាបីកែវ។ "
            "ហាសិបភាគរយ សម្រាប់ចំណាយចាំបាច់ប្រចាំថ្ងៃ ដូចជាថ្លៃផ្ទះ និងអាហារ។ "
            "សាមសិបភាគរយ សម្រាប់កម្សាន្ត កាហ្វេ ឬទិញរបស់ដែលអ្នកចូលចិត្ត។ "
            "ហើយម្ភៃភាគរយទៀត ចាក់សោទុកសន្សំសម្រាប់អនាគត។"
        ),
    },
    {
        "id": "07_new_day25",
        "start": 59.0,
        "end": 68.0,
        "lead_in": 0.20,
        "text": (
            "ដល់ថ្ងៃទីម្ភៃប្រាំម្តងនេះ បើកមើលទូរស័ព្ទដោយញញឹម។ "
            "នៅតែមានបាយឆ្ងាញ់ញ៉ាំធម្មតា ហើយលុយសន្សំក៏កាន់តែកើនឡើងជារៀងរាល់ខែ។"
        ),
    },
    {
        "id": "08_outro",
        "start": 68.0,
        "end": 75.0,
        "lead_in": 0.20,
        "text": "ចេះបែងចែក ទើបចុងខែមិនដាច់លុយ។ ចុះអ្នកវិញ រាល់ខែដាច់លុយតាំងពីថ្ងៃទីប៉ុន្មាន? ខំមិនខាងក្រោមផងណា!",
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


def prepare_reference_voice(src_path: Path) -> Path:
    """Convert sample voice to clean mono WAV for optimal VoxCPM2 cloning stability."""
    if not src_path.exists():
        raise FileNotFoundError(f"Voice sample not found: {src_path}")
    out_ref = Path("output/ref_chavchet_male_mono.wav")
    out_ref.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_path),
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
        str(out_ref),
    ]
    subprocess.check_call(cmd)
    print(f"[VOICE] Prepared mono 48kHz reference from {src_path} -> {out_ref}")
    return out_ref


def time_stretch_ffmpeg(wav: np.ndarray, sr: int, speed_factor: float) -> np.ndarray:
    """Pitch-preserving ffmpeg atempo stretch so words are never abruptly cut off."""
    if speed_factor <= 1.005:
        return wav
    # Chain atempo filters if factor > 2.0
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


def synthesize_voiceover() -> int:
    ref_wav = prepare_reference_voice(VOICE_SAMPLE_PATH)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    use_mlx = is_apple_silicon()
    if use_mlx:
        from mlx_audio.tts.utils import load_model

        model_path = ensure_voxcpm2_local(HF_MLX_REPO_ID)
        print(f"[TTS-MLX] Loading {HF_MLX_REPO_ID} with voice '{VOICE_SAMPLE_PATH.name}'...")
        model = load_model(str(model_path))
        sr = int(getattr(model, "sample_rate", TARGET_SAMPLE_RATE))
    else:
        from voxcpm import VoxCPM

        model_path = ensure_voxcpm2_local(HF_TORCH_REPO_ID)
        model = VoxCPM.from_pretrained(str(model_path), load_denoiser=False)
        sr = int(getattr(getattr(model, "tts_model", None), "sample_rate", TARGET_SAMPLE_RATE))

    timeline = np.zeros(int(round(TARGET_DURATION_SEC * sr)), dtype=np.float32)
    timings = []

    for seg in TOPIC1_VO_SEGMENTS:
        sid = str(seg["id"])
        start_s = float(seg["start"])
        end_s = float(seg["end"])
        lead_in = float(seg["lead_in"])
        slot_s = end_s - start_s
        text = str(seg["text"])

        print(f"[TTS-KM] Synthesizing {sid} ({start_s:.1f}s - {end_s:.1f}s)...")
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

        # Trim leading/trailing silence cleanly
        nz = np.where(np.abs(wav) > 0.008)[0]
        if len(nz) > 0:
            i0 = max(0, int(nz[0]) - int(0.04 * sr))
            i1 = min(len(wav), int(nz[-1]) + int(0.12 * sr))
            wav = wav[i0:i1]

        wav = normalize_segment_rms(wav)
        raw_dur = len(wav) / sr
        sf.write(str(SEGMENTS_DIR / f"{sid}_raw.wav"), wav, samplerate=sr, subtype="PCM_16")

        # Leave small breathing margin at end of each slot (and 1.2s tail fade on outro)
        tail_pad = 0.8 if sid == "08_outro" else 0.12
        avail_s = max(0.5, slot_s - lead_in - tail_pad)
        speed = 1.0
        if raw_dur > avail_s:
            # Reduce lead_in first if possible before applying gentle atempo
            if lead_in > 0.15 and raw_dur <= (slot_s - 0.10 - tail_pad):
                lead_in = max(0.05, slot_s - tail_pad - raw_dur)
                avail_s = slot_s - lead_in - tail_pad
            if raw_dur > avail_s:
                speed = raw_dur / avail_s
                wav = time_stretch_ffmpeg(wav, sr, speed)

        # Gentle 25ms edge fades
        fade_n = min(int(0.025 * sr), len(wav) // 4)
        if fade_n > 0:
            ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            wav[:fade_n] *= ramp
            wav[-fade_n:] *= ramp[::-1]

        # Enforce strict slot boundary
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
    print(f"[DONE] Khmer voiceover timeline saved to: {VO_TIMELINE_PATH} ({TARGET_DURATION_SEC:.1f}s @ {sr}Hz)")
    return sr


def mix_voiceover_and_remux(vo_sr: int) -> None:
    """Duck native SFX/BGM under the Khmer voiceover and remux output/topic1_reel_final.mp4."""
    if not FINAL_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Expected rendered reel at {FINAL_VIDEO_PATH}")

    # Preserve clean backup of the native SFX/BGM track (before VO is mixed in)
    if not NATIVE_AUDIO_BACKUP.exists():
        subprocess.check_call([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(FINAL_VIDEO_PATH),
            "-vn", "-ac", "2", "-ar", str(TARGET_SAMPLE_RATE),
            str(NATIVE_AUDIO_BACKUP),
        ])
        print(f"[BACKUP] Saved native SFX/BGM audio -> {NATIVE_AUDIO_BACKUP}")

    bg, sr_bg = sf.read(str(NATIVE_AUDIO_BACKUP), dtype="float32")
    vo_raw, sr_vo = sf.read(str(VO_TIMELINE_PATH), dtype="float32")

    # Resample VO to 48kHz if model emitted a different sample rate
    if sr_vo != TARGET_SAMPLE_RATE:
        with tempfile.TemporaryDirectory() as td:
            res_p = Path(td) / "vo_48k.wav"
            subprocess.check_call([
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(VO_TIMELINE_PATH), "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
                str(res_p),
            ])
            vo_raw, sr_vo = sf.read(str(res_p), dtype="float32")

    sr = TARGET_SAMPLE_RATE
    vo_mono = vo_raw if vo_raw.ndim == 1 else np.mean(vo_raw, axis=1)
    bg_st = bg[:, :2] if bg.ndim == 2 else np.stack([bg, bg], axis=1)

    n = min(len(bg_st), len(vo_mono))
    bg_st = bg_st[:n].copy()
    vo_mono = vo_mono[:n].copy()
    vo_st = np.stack([vo_mono, vo_mono], axis=1)

    # Smooth speech-presence envelope to duck native SFX/BGM cleanly while narrator speaks
    env = maximum_filter1d(np.abs(vo_mono), size=int(0.28 * sr))
    env = uniform_filter1d(env, size=int(0.32 * sr))
    duck = np.clip(env / 0.030, 0.0, 1.0)
    # Full 0.78 when no speech (so SFX & record scratch pop!), ducked to 0.24 during speech
    bg_gain = 0.78 - 0.54 * duck

    # Present, clear Khmer voiceover on top of ducked SFX/BGM
    mixed = vo_st * 1.02 + bg_st * bg_gain[:, None]

    # Soft peak limit at -0.8 dBFS (0.912)
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.912:
        mixed = np.tanh(mixed / peak * 1.15) * 0.912 / np.tanh(1.15)

    sf.write(str(MERGED_AUDIO_PATH), mixed, samplerate=sr, subtype="PCM_16")
    print(f"[MIX] Saved VO + ducked native SFX/BGM -> {MERGED_AUDIO_PATH}")

    # Remux losslessly (-c:v copy) into output/topic1_reel_final.mp4
    tmp_mp4 = FINAL_VIDEO_PATH.with_name("topic1_reel_final_tmp.mp4")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(FINAL_VIDEO_PATH),
        "-i", str(MERGED_AUDIO_PATH),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(sr), "-ac", "2",
        "-movflags", "+faststart", "-shortest",
        str(tmp_mp4),
    ]
    subprocess.check_call(cmd)
    shutil.move(str(tmp_mp4), str(FINAL_VIDEO_PATH))
    print(f"[DONE] Remuxed Khmer VO + native SFX/BGM into: {FINAL_VIDEO_PATH}")


def main() -> None:
    sr = synthesize_voiceover()
    mix_voiceover_and_remux(sr)


if __name__ == "__main__":
    main()
