"""
VoxCPM2 Zero-Shot Voice Cloning & Narration Generator (Khmer Language)
Supports Apple Silicon MLX (`mlx-community/VoxCPM2-8bit` via `mlx-audio`)
and CUDA PyTorch (`openbmb/VoxCPM2` via `voxcpm`).
Topic: Filial Duty vs Your Own Dreams (ភាសាខ្មែរ)
Duration Target: 120 seconds | Sample Rate: 48 kHz
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List

HF_TORCH_REPO_ID = "openbmb/VoxCPM2"
HF_MLX_REPO_ID = "mlx-community/VoxCPM2-8bit"
TARGET_SAMPLE_RATE = 48000
TARGET_DURATION_SEC = 120.0
ASSETS_DIR = Path("assets")
OUTPUT_DIR = Path("output/audio_segments_km")
FINAL_NARRATION_PATH = Path("output/narration.wav")
TIMING_REPORT_PATH = Path("output/audio_timings.json")

# Khmer TTS-optimized script: all numbers spelled out in Khmer words, natural breathing pauses.
SCRIPT_SEGMENTS: List[Dict[str, object]] = [
    {
        "id": "01_intro",
        "start": 0.0,
        "end": 6.5,
        "style": "(intimate, gripping, emotional pattern-interrupt tone)",
        "text": "ឪពុកម្តាយស្រឡាញ់អ្នក... ប៉ុន្តែហេតុអ្វីបានជាក្តីសុបិនរបស់ពួកគាត់ ធ្វើឱ្យអ្នកមានអារម្មណ៍ដូចជាប់ទ្រុង?",
    },
    {
        "id": "02_verse_1a",
        "start": 6.5,
        "end": 22.0,
        "style": "(reflective, steady storytelling pace)",
        "text": (
            "តាំងពីអ្នកមិនទាន់ស្គាល់ឈ្មោះខ្លួនឯងផង ព្រួញពីរត្រូវបានកំណត់អនាគតអ្នករួចទៅហើយ។ "
            "មួយចង្អុលទៅគ្រូពេទ្យ។ មួយទៀតចង្អុលទៅមេធាវី។ "
            "មានសុវត្ថិភាព មានកិត្តិយស និងត្រូវបានគាំទ្រជាស្រេច។"
        ),
    },
    {
        "id": "03_verse_1b",
        "start": 22.0,
        "end": 44.0,
        "style": "(empathetic, slightly heavier emotional weight)",
        "text": (
            "អស់រយៈពេលជាច្រើនឆ្នាំ កតញ្ញូតាធម៌មានន័យថាការយល់ព្រម។ "
            "អ្នកមើលឃើញការខិតខំធ្វើការទាំងយប់ទាំងថ្ងៃរបស់ពួកគាត់ ហើយភ្លាមៗនោះ "
            "ការដឹងគុណក្លាយជាខ្សែចងមើលមិនឃើញ... ទាញដៃអ្នកទៅកាន់ជីវិតដែលអ្នកមិនដែលជ្រើសរើស។ "
            "ពេលអ្នកដទៃសម្រេចចិត្តធំៗទាំងអស់ជំនួសអ្នក "
            "អ្នកភ្ញាក់ឡើងនៅអាយុម្ភៃប្រាំឆ្នាំ ដោយមិនដឹងថាត្រូវសម្រេចចិត្តដោយខ្លួនឯងយ៉ាងណាឡើយ។"
        ),
    },
    {
        "id": "04_hook_midpoint",
        "start": 44.0,
        "end": 60.0,
        "style": "(direct, candid, touch of dry ironic humor)",
        "text": (
            "សូមសួរដោយស្មោះត្រង់មួយសំណួរ។ តើពួកគាត់កំពុងចិញ្ចឹមមនុស្សម្នាក់ "
            "ឬកំពុងបីអគារការិយាល័យនៅក្នុងភួយទារក? "
            "ព្រោះបើអាជីពរបស់អ្នកគ្រាន់តែជាពានរង្វាន់សំណព្វចិត្តរបស់ពួកគាត់... "
            "តើអ្នកណាជាអ្នកទៅធ្វើការនៅព្រឹកថ្ងៃចន្ទពិតប្រាកដ?"
        ),
    },
    {
        "id": "05_verse_2a",
        "start": 60.0,
        "end": 83.0,
        "style": "(warm, grounded, reassuring clarity)",
        "text": (
            "នេះគឺជាការពិតដែលគ្មាននរណាប្រាប់អ្នកនៅតុបាយឡើយ។ "
            "ការជ្រើសរើសផ្លូវដើរដោយខ្លួនឯង មិនមែនជាការក្បត់ទេ។ "
            "ការគោរពការលះបង់របស់ពួកគាត់ មានន័យថាការមិនខ្ជះខ្ជាយសេរីភាពដែលពួកគាត់តស៊ូទិញឱ្យអ្នក។ "
            "ដំបូន្មានរបស់ពួកគាត់កើតចេញពីការភ័យខ្លាចភាពងងឹត "
            "ព្រោះពួកគាត់ធ្លាប់ឆ្លងកាត់ពិភពលោកដែលគ្មានសំណាញ់សុវត្ថិភាព។"
        ),
    },
    {
        "id": "06_verse_2b",
        "start": 83.0,
        "end": 104.5,
        "style": "(confident, practical, inspiring momentum)",
        "text": (
            "ដូច្នេះ ចូរឈប់ឈ្លោះប្រកែកក្នុងចិត្ត ហើយចាប់ផ្តើមកសាងភស្តុតាងក្រោមពន្លឺថ្ងៃ។ "
            "រក្សាសន្តិភាពក្នុងគ្រួសារខណៈពេលអ្នកពង្រឹងជំនាញ សន្សំប្រាក់ទុករស់បានប្រាំមួយខែ "
            "ហើយបង្ហាញផែនការពិតប្រាកដដល់ពួកគាត់ មិនមែនត្រឹមតែក្តីប្រាថ្នាទេ។ "
            "ពេលភាពភ័យខ្លាចជួបការត្រៀមខ្លួន ការស្រឡាញ់នឹងមានកន្លែងដកដង្ហើម។"
        ),
    },
    {
        "id": "07_outro",
        "start": 104.5,
        "end": 120.0,
        "style": "(warm, decisive, clear call-to-action)",
        "text": (
            "អ្នកមានជីវិតតែមួយគត់ដើម្បីចុះហត្ថលេខាដោយខ្លួនឯង។ "
            "ដូច្នេះ សូមប្រាប់ខ្ញុំនៅក្នុងមតិយោបល់ឥឡូវនេះ... "
            "តើអាជីពក្នុងក្តីសុបិនមួយណាដែលអ្នកកំពុងចាប់ផ្តើមទាមទារសម្រាប់ខ្លួនឯងនៅថ្ងៃនេះ?"
        ),
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


def resolve_voice_sample(assets_dir: Path = ASSETS_DIR) -> Path:
    import soundfile as sf

    preferred = assets_dir / "cg_voice_sample.wav"
    candidates: List[Path] = [preferred] if preferred.exists() else []
    exts = (".wav", ".mp3", ".flac", ".m4a")
    if assets_dir.exists():
        candidates.extend(sorted(p for p in assets_dir.iterdir() if p.suffix.lower() in exts and p != preferred))
    for fallback_name in ("voice_sample.wav", "voice_sample.mp3"):
        fb = Path(fallback_name)
        if fb.exists():
            candidates.append(fb)
    if not candidates:
        raise FileNotFoundError(f"No voice sample audio found in '{assets_dir}/' or project root.")

    source = candidates[0]
    print(f"[VOICE] Using reference audio: {source}")
    data, sr = sf.read(str(source), dtype="float32")
    dur = len(data) / sr
    if dur > 16.0:
        # Take clean 14s window (4s..18s) for optimal VoxCPM2 cloning stability & speed
        start_s = 4.0 if dur >= 18.0 else 0.0
        clip = data[int(start_s * sr) : int((start_s + 14.0) * sr)]
        ref_out = Path("output/ref_voice_14s.wav")
        ref_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(ref_out), clip, samplerate=sr)
        print(f"[VOICE] Extracted 14.0s clean reference slice -> {ref_out}")
        return ref_out
    return source


def _fit_clip_to_slot(wav, slot_samples: int, sample_rate: int):
    import numpy as np

    fade_n = min(int(0.03 * sample_rate), len(wav) // 4)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        wav[:fade_n] *= ramp
        wav[-fade_n:] *= ramp[::-1]

    if len(wav) <= slot_samples:
        return wav
    trimmed = wav[:slot_samples].copy()
    if fade_n > 0:
        trimmed[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    return trimmed


def synthesize_narration() -> None:
    import numpy as np
    import soundfile as sf

    ref_wav_path = resolve_voice_sample()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_NARRATION_PATH.parent.mkdir(parents=True, exist_ok=True)

    use_mlx = is_apple_silicon()
    if use_mlx:
        from mlx_audio.tts.utils import load_model

        model_path = ensure_voxcpm2_local(HF_MLX_REPO_ID)
        print(f"[TTS-MLX] Loading {HF_MLX_REPO_ID} for Khmer narration...")
        model = load_model(str(model_path))
        sample_rate = int(getattr(model, "sample_rate", TARGET_SAMPLE_RATE))
    else:
        from voxcpm import VoxCPM

        model_path = ensure_voxcpm2_local(HF_TORCH_REPO_ID)
        model = VoxCPM.from_pretrained(str(model_path), load_denoiser=False)
        sample_rate = int(getattr(getattr(model, "tts_model", None), "sample_rate", TARGET_SAMPLE_RATE))

    timeline = np.zeros(int(TARGET_DURATION_SEC * sample_rate), dtype=np.float32)
    timings = []

    for seg in SCRIPT_SEGMENTS:
        seg_id = str(seg["id"])
        start_sec = float(seg["start"])
        end_sec = float(seg["end"])
        slot_sec = end_sec - start_sec
        text_only = str(seg["text"])
        prompt = f"{seg['style']} {text_only}"

        print(f"[TTS-KM] Synthesizing {seg_id} ({start_sec:.1f}s - {end_sec:.1f}s)...")
        if use_mlx:
            chunks = [
                np.asarray(res.audio, dtype=np.float32).flatten()
                for res in model.generate(
                    text=text_only,
                    ref_audio=str(ref_wav_path),
                    cfg_value=2.0,
                    inference_timesteps=10,
                )
            ]
            wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        else:
            wav = np.asarray(
                model.generate(
                    text=prompt,
                    reference_wav_path=str(ref_wav_path),
                    cfg_value=2.0,
                    inference_timesteps=10,
                ),
                dtype=np.float32,
            ).flatten()

        abs_w = np.abs(wav)
        nz = np.where(abs_w > 0.008)[0]
        if len(nz) > 0:
            i0 = max(0, int(nz[0]) - int(0.05 * sample_rate))
            i1 = min(len(wav), int(nz[-1]) + int(0.15 * sample_rate))
            wav = wav[i0:i1]

        raw_dur = len(wav) / sample_rate
        sf.write(str(OUTPUT_DIR / f"{seg_id}.wav"), wav, samplerate=sample_rate)

        slot_samples = int(slot_sec * sample_rate)
        fitted = _fit_clip_to_slot(wav, slot_samples, sample_rate)
        start_idx = int(start_sec * sample_rate)
        offset = int(0.2 * sample_rate) if start_sec > 0 and (len(fitted) + int(0.25 * sample_rate) <= slot_samples) else 0
        place_idx = start_idx + offset
        end_idx = min(place_idx + len(fitted), len(timeline))
        timeline[place_idx:end_idx] = fitted[: end_idx - place_idx]

        timings.append(
            {
                "id": seg_id,
                "slot_start": start_sec,
                "slot_end": end_sec,
                "offset": round(offset / sample_rate, 3),
                "raw_speech_sec": round(raw_dur, 2),
                "fitted_sec": round(len(fitted) / sample_rate, 2),
            }
        )
        print(f"  -> {seg_id}: generated {raw_dur:.2f}s (slot {slot_sec:.1f}s)")

    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 0.0:
        timeline = timeline * (0.891 / peak)

    sf.write(str(FINAL_NARRATION_PATH), timeline, samplerate=sample_rate)
    TIMING_REPORT_PATH.write_text(json.dumps(timings, indent=2, ensure_ascii=False))
    print(f"[DONE] Khmer narration saved to: {FINAL_NARRATION_PATH} ({TARGET_DURATION_SEC:.1f}s @ {sample_rate}Hz)")


if __name__ == "__main__":
    synthesize_narration()
