"""
Topic 5: ក្តីស្រឡាញ់មិនមែនទាល់តែរបស់ថ្លៃ — Full 70.0s Video-Clip Reel Assembler
Combines all 10 animated scene clips from topic5_scenes/clips/scene_<id>.mp4
(with smooth Ken Burns + sparkle fallback for any scene that only has a .jpg),
levels and ducks native SFX/BGM under output/topic5_vo_khmer.wav (if present),
upscales to 1080x1920 @ 24fps with Lanczos, overlays bold Khmer captions via
bin/khmer_text_render, and exports output/topic5_reel_final.mp4.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter1d, uniform_filter1d

W, H = 1080, 1920
FPS = 24
SR = 48000
SCENE_DIR = Path("topic5_scenes")
CLIPS_DIR = SCENE_DIR / "clips"
MANIFEST_PATH = SCENE_DIR / "topic5_prompts.json"
KHMER_RENDER_BIN = Path("bin/khmer_text_render")
TEXT_CACHE_DIR = Path("output/topic5_text_pngs")
VO_PATH = Path("output/topic5_vo_khmer.wav")
OUT_VIDEO = Path("output/topic5_reel_final.mp4")

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def render_khmer_badge(
    text: str,
    cache_name: str,
    font_size: int = 50,
    hex_color: str = "FFFFFF",
    bg_rgba: Tuple[int, int, int, int] = (32, 34, 52, 215),
    max_width: int = 1010,
) -> Image.Image:
    TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_png = TEXT_CACHE_DIR / f"{cache_name}.png"
    subprocess.check_call(
        [str(KHMER_RENDER_BIN), text, str(font_size), str(max_width), hex_color, str(raw_png)],
        stdout=subprocess.DEVNULL,
    )
    txt_im = Image.open(raw_png).convert("RGBA")
    pad_x, pad_y = 26, 8
    bw, bh = txt_im.width + pad_x * 2, txt_im.height + pad_y * 2
    badge = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    draw.rounded_rectangle(
        [0, 0, bw - 1, bh - 1],
        radius=24,
        fill=bg_rgba,
        outline=(255, 255, 255, 75),
        width=3,
    )
    badge.alpha_composite(txt_im, (pad_x, pad_y))
    return badge


def alpha_composite_np(
    frame_rgb: np.ndarray,
    badge_rgba: Image.Image,
    cx: int,
    cy: int,
    scale: float = 1.0,
    opacity: float = 1.0,
) -> None:
    if opacity <= 0.01 or scale <= 0.05:
        return
    b = (
        badge_rgba.resize(
            (max(4, int(round(badge_rgba.width * scale))), max(4, int(round(badge_rgba.height * scale)))),
            Image.Resampling.LANCZOS,
        )
        if abs(scale - 1.0) > 0.005
        else badge_rgba
    )
    arr = np.asarray(b, dtype=np.float32)
    bh, bw = arr.shape[:2]
    x0, y0 = cx - bw // 2, cy - bh // 2
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(W, x0 + bw), min(H, y0 + bh)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    bx0, by0 = fx0 - x0, fy0 - y0
    src_rgb = arr[by0 : by0 + (fy1 - fy0), bx0 : bx0 + (fx1 - fx0), :3]
    alpha = (arr[by0 : by0 + (fy1 - fy0), bx0 : bx0 + (fx1 - fx0), 3:4] / 255.0) * opacity
    dst = frame_rgb[fy0:fy1, fx0:fx1].astype(np.float32)
    frame_rgb[fy0:fy1, fx0:fx1] = np.clip(src_rgb * alpha + dst * (1.0 - alpha), 0, 255).astype(np.uint8)


def ease_out_back(u: float) -> float:
    u = max(0.0, min(1.0, u))
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * ((u - 1.0) ** 3) + c1 * ((u - 1.0) ** 2)


def render_animated_still_fallback(img_path: Path, target_frames: int) -> List[np.ndarray]:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise RuntimeError(f"Could not read fallback still: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sh, sw = rgb.shape[:2]
    overscan = 1.08
    ow, oh = int(round(W * overscan)), int(round(H * overscan))
    sc = max(ow / sw, oh / sh)
    base = cv2.resize(rgb, (int(round(sw * sc)), int(round(sh * sc))), interpolation=cv2.INTER_LANCZOS4)
    bx0, by0 = (base.shape[1] - ow) // 2, (base.shape[0] - oh) // 2
    base = base[by0 : by0 + oh, bx0 : bx0 + ow]

    rng = np.random.default_rng(55)
    n_sparks = 16
    sx = rng.uniform(W * 0.15, W * 0.85, n_sparks)
    sy = rng.uniform(H * 0.36, H * 0.65, n_sparks)
    sv = rng.uniform(14.0, 30.0, n_sparks)
    sp = rng.uniform(0.0, 2.0 * np.pi, n_sparks)
    sr_px = rng.uniform(3.0, 6.0, n_sparks)

    frames: List[np.ndarray] = []
    for f in range(target_frames):
        u = f / max(1, target_frames - 1)
        zoom = 1.0 + 0.045 * (0.5 - 0.5 * np.cos(np.pi * u))
        cw, ch = int(round(W * overscan / zoom)), int(round(H * overscan / zoom))
        x0, y0 = (ow - cw) // 2, (oh - ch) // 2
        frame = cv2.resize(base[y0 : y0 + ch, x0 : x0 + cw], (W, H), interpolation=cv2.INTER_LANCZOS4)
        t_s = f / FPS
        overlay = frame.copy()
        for i in range(n_sparks):
            px = int(round(sx[i] + 10.0 * np.sin(t_s * 1.3 + sp[i])))
            py = int(round(sy[i] - sv[i] * t_s))
            if py < int(H * 0.12):
                py += int(H * 0.45)
            r = max(2, int(round(sr_px[i] * (0.65 + 0.5 * (0.5 + 0.5 * np.sin(t_s * 3.8 + sp[i]))))))
            cv2.circle(overlay, (px, py), r + 3, (255, 224, 150), -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), max(1, r - 1), (255, 250, 225), -1, cv2.LINE_AA)
        frames.append(cv2.addWeighted(overlay, 0.30, frame, 0.70, 0.0))
    return frames


def read_clip_frames_native(clip_path: Path, fallback_img: Path, target_frames: int) -> List[np.ndarray]:
    if not clip_path.exists():
        print(f"  [FALLBACK] {clip_path.name} missing -> animating {fallback_img.name} smoothly ({target_frames} frames)")
        return render_animated_still_fallback(fallback_img, target_frames)
    cap = cv2.VideoCapture(str(clip_path))
    frames: List[np.ndarray] = []
    while len(frames) < target_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sh, sw = rgb.shape[:2]
        if sw != W or sh != H:
            sc = max(W / sw, H / sh)
            nw, nh = int(round(sw * sc)), int(round(sh * sc))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            x0, y0 = (nw - W) // 2, (nh - H) // 2
            rgb = rgb[y0 : y0 + H, x0 : x0 + W]
        frames.append(rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {clip_path}")
    while len(frames) < target_frames:
        frames.append(frames[-1].copy())
    return frames[:target_frames]


def extract_clip_pcm(clip_path: Path, dur_sec: float) -> np.ndarray:
    n_samples = int(round(dur_sec * SR))
    raw = subprocess.check_output([
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", str(clip_path), "-t", f"{dur_sec:.6f}",
        "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "pipe:1",
    ])
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    if audio.size % 2 != 0:
        audio = audio[:-1]
    stereo = audio.reshape(-1, 2) if audio.size else np.zeros((0, 2), dtype=np.float32)
    if len(stereo) < n_samples:
        stereo = np.vstack([stereo, np.zeros((n_samples - len(stereo), 2), dtype=np.float32)])
    return stereo[:n_samples]


def level_clip_audio(stereo: np.ndarray) -> np.ndarray:
    if stereo.size == 0:
        return stereo
    mono_sq = np.mean(stereo * stereo, axis=1)
    active = mono_sq[mono_sq > (10 ** (-45 / 10))]
    rms = float(np.sqrt(np.mean(active))) if active.size > SR * 0.1 else float(np.sqrt(np.mean(mono_sq)))
    if rms < 1e-6:
        return stereo
    gain = float(np.clip((10 ** (-18.5 / 20.0)) / rms, 10 ** (-6.0 / 20.0), 10 ** (9.0 / 20.0)))
    leveled = stereo * gain
    peak = float(np.max(np.abs(leveled)))
    if peak > 0.891:
        leveled = np.tanh(leveled * (0.95 / peak) * 1.15) * 0.891 / np.tanh(1.15)
    return leveled.astype(np.float32)


def build_final_audio(scenes: List[dict], out_wav: Path) -> None:
    segments: List[np.ndarray] = []
    fade_n = int(0.04 * SR)

    for idx, s in enumerate(scenes):
        sid = s["id"]
        dur = float(s["end"] - s["start"])
        n_samples = int(round(dur * SR))
        clip_path = CLIPS_DIR / f"scene_{sid}.mp4"
        if clip_path.exists():
            seg = level_clip_audio(extract_clip_pcm(clip_path, dur))
        elif segments:
            prev = segments[-1]
            tail = prev[-min(len(prev), int(4.0 * SR)) :].copy()
            bed = np.vstack([tail, tail[::-1].copy()])
            seg = np.vstack([bed] * ((n_samples // len(bed)) + 2))[:n_samples] * 0.72
            print(f"  [AUDIO FALLBACK] Extended warm music bed for missing {clip_path.name}")
        else:
            seg = np.zeros((n_samples, 2), dtype=np.float32)

        if idx > 0 and len(seg) > fade_n * 2:
            seg[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)[:, None]
        if idx < len(scenes) - 1 and len(seg) > fade_n * 2:
            seg[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)[:, None]
        if idx == len(scenes) - 1:
            end_fade = min(len(seg), int(1.8 * SR))
            seg[-end_fade:] *= np.linspace(1.0, 0.0, end_fade, dtype=np.float32)[:, None]
        segments.append(seg)

    bg_st = np.vstack(segments)
    if VO_PATH.exists():
        vo, sr_vo = sf.read(str(VO_PATH), dtype="float32")
        if sr_vo != SR:
            raise ValueError(f"Expected {SR}Hz VO audio, got {sr_vo}Hz")
        vo_mono = vo if vo.ndim == 1 else np.mean(vo, axis=1)
        n = min(len(bg_st), len(vo_mono))
        bg_st, vo_mono = bg_st[:n], vo_mono[:n]
        vo_st = np.stack([vo_mono, vo_mono], axis=1)

        env = maximum_filter1d(np.abs(vo_mono), size=int(0.28 * SR))
        env = uniform_filter1d(env, size=int(0.32 * SR))
        duck = np.clip(env / 0.030, 0.0, 1.0)
        bg_gain = 0.78 - 0.54 * duck
        mixed = vo_st * 1.02 + bg_st * bg_gain[:, None]
    else:
        mixed = bg_st

    peak = float(np.max(np.abs(mixed)))
    if peak > 0.912:
        mixed = np.tanh(mixed / peak * 1.15) * 0.912 / np.tanh(1.15)
    sf.write(str(out_wav), mixed, samplerate=SR, subtype="PCM_16")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    scenes = manifest["scenes"]
    missing_both = [
        s["id"]
        for s in scenes
        if not (CLIPS_DIR / f"scene_{s['id']}.mp4").exists()
        and not (SCENE_DIR / s["file"]).exists()
    ]
    if missing_both:
        raise FileNotFoundError(f"Missing both MP4 clip and fallback still for scenes: {missing_both}")

    scene_badges: Dict[str, List[Image.Image]] = {}
    for s in scenes:
        sid = s["id"]
        badges: List[Image.Image] = []
        for idx, line in enumerate(s.get("khmer_text", [])):
            if sid in ("04_raincoat_helmet", "05_warm_street_snack", "06_bubble_pops", "07_dry_hair_care", "08_warm_ride_home", "09_outro"):
                bg = (96, 48, 38, 215)
            else:
                bg = (32, 34, 52, 215)
            fsize = 42 if (sid in ("08_warm_ride_home", "09_outro") and (sid == "08_warm_ride_home" or idx == 1)) else 48
            badges.append(render_khmer_badge(line, f"{sid}_{idx}", font_size=fsize, hex_color="FFFFFF", bg_rgba=bg))
        scene_badges[sid] = badges

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        audio_wav = Path(td) / "topic5_mix.wav"
        build_final_audio(scenes, audio_wav)

        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-i", str(audio_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
            "-movflags", "+faststart", "-shortest",
            str(OUT_VIDEO),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None

        prev_last: np.ndarray | None = None
        trans_frames = 5

        for s in scenes:
            sid = s["id"]
            dur = float(s["end"] - s["start"])
            n_frames = int(round(dur * FPS))
            frames = read_clip_frames_native(CLIPS_DIR / f"scene_{sid}.mp4", SCENE_DIR / s["file"], n_frames)
            badges = scene_badges[sid]

            for f, raw in enumerate(frames):
                t_local = f / FPS
                frame = raw.copy()
                if prev_last is not None and f < trans_frames:
                    a = (f + 1) / (trans_frames + 1)
                    frame = cv2.addWeighted(frame, a, prev_last, 1.0 - a, 0.0)

                if badges:
                    if sid == "00_hook" and len(badges) == 2:
                        positions = [(W // 2, 560), (W // 2, 1795)]
                        delays = [0.10, 1.35]
                    elif sid == "09_outro" and len(badges) == 2:
                        positions = [(W // 2, 360), (W // 2, 540)]
                        delays = [0.25, 2.20]
                    else:
                        positions = [(W // 2, 240)]
                        delays = [0.20]

                    for b_img, (bx, by), d_sec in zip(badges, positions, delays):
                        if t_local >= d_sec:
                            u_in = min(1.0, (t_local - d_sec) / 0.28)
                            sc = 0.65 + 0.35 * ease_out_back(u_in)
                            op = min(1.0, (t_local - d_sec) / 0.14)
                            alpha_composite_np(frame, b_img, bx, by, scale=sc, opacity=op)

                proc.stdin.write(frame.tobytes())
                if f == n_frames - 1:
                    prev_last = frame

        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("FFmpeg failed during Topic 5 reel render")

    print(f"[DONE] Combined all 10 Topic 5 clips into: {OUT_VIDEO}")


if __name__ == "__main__":
    main()
