"""
Topic 1: ដើមខែសេដ្ឋី ចុងខែមីកញ្ចប់ — Full 75.0s Video-Clip Reel Assembler
Combines all 13 animated scene clips from topic1_scenes/clips/scene_<id>.mp4
at native 1.0x speed (24fps, cleanly trimmed to each scene's manifest duration),
preserves and levels each clip's native SFX/BGM audio with clean joins (and a
hard smash-cut into 03_day12_shock), upscales to 1080x1920 with Lanczos,
overlays bold Khmer captions via bin/khmer_text_render, and exports
output/topic1_reel_final.mp4.
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
from PIL import Image, ImageDraw

W, H = 1080, 1920
FPS = 24
SR = 48000
SCENE_DIR = Path("topic1_scenes")
CLIPS_DIR = SCENE_DIR / "clips"
MANIFEST_PATH = SCENE_DIR / "topic1_prompts.json"
KHMER_RENDER_BIN = Path("bin/khmer_text_render")
TEXT_CACHE_DIR = Path("output/topic1_text_pngs")
OUT_VIDEO = Path("output/topic1_reel_final.mp4")

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def render_khmer_badge(
    text: str,
    cache_name: str,
    font_size: int = 54,
    hex_color: str = "FFFFFF",
    bg_rgba: Tuple[int, int, int, int] = (28, 22, 18, 205),
    max_width: int = 980,
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
    if abs(scale - 1.0) > 0.005:
        nw = max(4, int(round(badge_rgba.width * scale)))
        nh = max(4, int(round(badge_rgba.height * scale)))
        b = badge_rgba.resize((nw, nh), Image.Resampling.LANCZOS)
    else:
        b = badge_rgba
    arr = np.asarray(b, dtype=np.float32)
    bh, bw = arr.shape[:2]
    x0, y0 = cx - bw // 2, cy - bh // 2
    x1, y1 = x0 + bw, y0 + bh

    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(W, x1), min(H, y1)
    if fx1 <= fx0 or fy1 <= fy0:
        return

    bx0, by0 = fx0 - x0, fy0 - y0
    bx1, by1 = bx0 + (fx1 - fx0), by0 + (fy1 - fy0)
    src_rgb = arr[by0:by1, bx0:bx1, :3]
    alpha = (arr[by0:by1, bx0:bx1, 3:4] / 255.0) * opacity
    dst = frame_rgb[fy0:fy1, fx0:fx1].astype(np.float32)
    frame_rgb[fy0:fy1, fx0:fx1] = np.clip(src_rgb * alpha + dst * (1.0 - alpha), 0, 255).astype(np.uint8)


def ease_out_back(u: float) -> float:
    u = max(0.0, min(1.0, u))
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * ((u - 1.0) ** 3) + c1 * ((u - 1.0) ** 2)


def read_clip_frames_native(clip_path: Path, target_frames: int) -> List[np.ndarray]:
    """Read an MP4 clip at native 1.0x speed, Lanczos-upscale to 1080x1920, and trim/pad cleanly."""
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
        raise RuntimeError(f"Could not read any frames from {clip_path}")

    # Pad any 1-2 frame rounding shortfall cleanly by holding the last frame
    while len(frames) < target_frames:
        frames.append(frames[-1].copy())

    return frames[:target_frames]


def extract_clip_pcm(clip_path: Path, dur_sec: float) -> np.ndarray:
    """Extract trimmed 48kHz stereo float32 audio from a clip and return shape (N, 2)."""
    n_samples = int(round(dur_sec * SR))
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", str(clip_path),
        "-t", f"{dur_sec:.6f}",
        "-vn", "-ac", "2", "-ar", str(SR),
        "-f", "f32le", "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    if audio.size % 2 != 0:
        audio = audio[:-1]
    stereo = audio.reshape(-1, 2) if audio.size else np.zeros((0, 2), dtype=np.float32)

    if len(stereo) < n_samples:
        pad = np.zeros((n_samples - len(stereo), 2), dtype=np.float32)
        stereo = np.vstack([stereo, pad])
    else:
        stereo = stereo[:n_samples]
    return stereo


def level_clip_audio(stereo: np.ndarray, sid: str) -> np.ndarray:
    """Gently level clip loudness to ~ -19 dBFS active RMS while preserving dramatic silence."""
    if stereo.size == 0:
        return stereo

    # Measure active RMS (samples above -45 dBFS) so silence beats in 03_day12_shock stay accurate
    mono_sq = np.mean(stereo * stereo, axis=1)
    active = mono_sq[mono_sq > (10 ** (-45 / 10))]
    rms = float(np.sqrt(np.mean(active))) if active.size > SR * 0.1 else float(np.sqrt(np.mean(mono_sq)))
    if rms < 1e-6:
        return stereo

    target_rms = 10 ** (-18.5 / 20.0)
    gain = target_rms / rms
    # Clamp gain to keep dynamic character natural and avoid noise floor lift
    max_gain_db = 5.0 if sid == "03_day12_shock" else 9.0
    gain = float(np.clip(gain, 10 ** (-6.0 / 20.0), 10 ** (max_gain_db / 20.0)))

    leveled = stereo * gain
    # Gentle soft-knee peak limiting around -1.0 dBFS (0.891)
    peak = float(np.max(np.abs(leveled)))
    ceiling = 0.891
    if peak > ceiling:
        leveled = np.tanh(leveled * (0.95 / peak) * 1.15) * ceiling / np.tanh(1.15)
    return leveled.astype(np.float32)


def build_native_audio_track(scenes: List[dict], out_wav: Path) -> None:
    """Stitch all 13 clips' native audio tracks with smooth micro-fades and final fade-out."""
    segments: List[np.ndarray] = []
    fade_n = int(0.035 * SR)  # 35ms click-free edge ramp

    for idx, s in enumerate(scenes):
        sid = s["id"]
        dur = float(s["end"] - s["start"])
        clip_path = CLIPS_DIR / f"scene_{sid}.mp4"
        seg = level_clip_audio(extract_clip_pcm(clip_path, dur), sid)

        # Micro fade-in (skip hard smash-cut on 03_day12_shock and first frame of 00_hook)
        if idx > 0 and sid != "03_day12_shock" and len(seg) > fade_n * 2:
            ramp_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)[:, None]
            seg[:fade_n] *= ramp_in

        # Micro fade-out before next clip (hard cut right before 03_day12_shock)
        next_sid = scenes[idx + 1]["id"] if idx + 1 < len(scenes) else None
        if next_sid is not None and next_sid != "03_day12_shock" and len(seg) > fade_n * 2:
            ramp_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)[:, None]
            seg[-fade_n:] *= ramp_out

        # Smooth 1.5s fade-out at end of final outro scene
        if idx == len(scenes) - 1:
            end_fade = min(len(seg), int(1.5 * SR))
            ramp_end = np.linspace(1.0, 0.0, end_fade, dtype=np.float32)[:, None]
            seg[-end_fade:] *= ramp_end

        segments.append(seg)

    full = np.vstack(segments)
    # Write 16-bit PCM WAV via ffmpeg
    pcm16 = np.clip(full * 32767.0, -32768, 32767).astype("<i2")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "pipe:0",
        str(out_wav),
    ]
    subprocess.run(cmd, input=pcm16.tobytes(), check=True)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    scenes = manifest["scenes"]

    # 1. Verify all 13 clips exist
    missing = [s["id"] for s in scenes if not (CLIPS_DIR / f"scene_{s['id']}.mp4").exists()]
    if missing:
        raise FileNotFoundError(f"Missing scene video clips in {CLIPS_DIR}: {missing}")

    # 2. Pre-render Khmer caption badges
    print(f"[TEXT] Rendering Khmer caption badges for {len(scenes)} scene clips...")
    scene_badges: Dict[str, List[Image.Image]] = {}
    for s in scenes:
        sid = s["id"]
        badges: List[Image.Image] = []
        for idx, line in enumerate(s.get("khmer_text", [])):
            if sid == "03_day12_shock" and idx == 1:
                bg = (185, 32, 32, 225)
                fg = "FFE854"
            elif sid in ("06_three_jars", "07_new_day25", "08_outro"):
                bg = (22, 68, 52, 215)
                fg = "FFFFFF"
            else:
                bg = (30, 22, 18, 210)
                fg = "FFFFFF"
            fsize = 44 if (sid == "08_outro" and idx == 1) else (50 if sid in ("06_three_jars", "08_outro") else 54)
            badges.append(render_khmer_badge(line, f"{sid}_{idx}", font_size=fsize, hex_color=fg, bg_rgba=bg, max_width=1020))
        scene_badges[sid] = badges

    # 3. Build leveled native audio track from all 13 video clips
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        stitched_wav = Path(tmpdir) / "native_audio_mix.wav"
        print("[AUDIO] Extracting & leveling native SFX/BGM audio from all 13 clips...")
        build_native_audio_track(scenes, stitched_wav)

        # 4. Combine all 13 animated MP4 clips + transitions + Khmer overlays + native audio
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "warning", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-i", str(stitched_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
            "-movflags", "+faststart", "-shortest",
            str(OUT_VIDEO),
        ]

        total_dur = sum(float(s["end"] - s["start"]) for s in scenes)
        total_frames = int(round(total_dur * FPS))
        print(f"[COMBINE] Combining 13 clips ({total_frames} frames, {total_dur:.1f}s @ {FPS}fps, 1080x1920) -> {OUT_VIDEO}")

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None

        prev_last_frame: np.ndarray | None = None
        trans_frames = 4  # ~0.16s snappy crossfade at 24fps

        for s in scenes:
            sid = s["id"]
            dur = float(s["end"] - s["start"])
            n_frames = int(round(dur * FPS))
            clip_path = CLIPS_DIR / f"scene_{sid}.mp4"
            frames = read_clip_frames_native(clip_path, n_frames)
            badges = scene_badges[sid]

            for f, raw_frame in enumerate(frames):
                t_local = f / FPS
                frame = raw_frame.copy()

                if prev_last_frame is not None and f < trans_frames and sid != "03_day12_shock":
                    a = (f + 1) / (trans_frames + 1)
                    frame = cv2.addWeighted(frame, a, prev_last_frame, 1.0 - a, 0.0)

                if badges:
                    if sid == "00_hook" and len(badges) == 2:
                        positions = [(W // 2, 875), (W // 2, 1795)]
                        delays = [0.10, 1.05]
                    elif sid == "06_three_jars" and len(badges) == 3:
                        positions = [(W // 2, 175), (W // 2, 345), (W // 2, 1730)]
                        delays = [0.35, 4.2, 8.2]
                    elif sid == "08_outro" and len(badges) == 2:
                        positions = [(W // 2, 390), (W // 2, 575)]
                        delays = [0.30, 2.20]
                    elif len(badges) == 2:
                        positions = [(W // 2, 240), (W // 2, 1650)]
                        delays = [0.25, dur * 0.42]
                    else:
                        positions = [(W // 2, 245)]
                        delays = [0.22]

                    for b_img, (bx, by), d_sec in zip(badges, positions, delays):
                        if t_local >= d_sec:
                            u_in = min(1.0, (t_local - d_sec) / 0.28)
                            sc = 0.65 + 0.35 * ease_out_back(u_in)
                            op = min(1.0, (t_local - d_sec) / 0.14)
                            alpha_composite_np(frame, b_img, bx, by, scale=sc, opacity=op)

                proc.stdin.write(frame.tobytes())
                if f == n_frames - 1:
                    prev_last_frame = frame

        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg exited with code {ret}")

    print(f"[DONE] Combined all 13 scene clips into: {OUT_VIDEO}")


if __name__ == "__main__":
    main()
