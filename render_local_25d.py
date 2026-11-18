"""
Enhanced Motion-Graphic Video Renderer (1080x1920 @ 30fps, 120.0s)
- Pure draw-path scenes + halftone paper/blueprint grids + Giphy transparent stickers
- 3D cel-shaded emotional stick characters (stickman_3d.py)
- Kinetic Khmer typography with 3D extruded shadow, speech-bubble callouts, typewriter cursor & spring pop
- 3-layer merged audio: Khmer narration + ducked BGM + frame-synced SFX
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from generate_sfx import generate_sfx
from merge_audio_video import MERGED_WAV, build_merged_audio
from motion_scenes import (
    W,
    H,
    ease_out_back,
    ease_out_cubic,
    render_scene_frame,
)

FPS = 30
TOTAL_DURATION = 120.0
TOTAL_FRAMES = int(TOTAL_DURATION * FPS)
XFADE_FRAMES = 14

TEXT_DIR = Path("output/text_overlays")
KHMER_RENDER_BIN = Path("bin/khmer_text_render")
FINAL_VIDEO_PATH = Path("output/final_video.mp4")

SCENE_WINDOWS: List[Tuple[str, float, float]] = [
    ("01_intro", 0.0, 6.2),
    ("02_verse_1a", 6.2, 19.2),
    ("03_verse_1b", 19.2, 46.2),
    ("04_hook_midpoint", 46.2, 61.0),
    ("05_verse_2a", 61.0, 80.5),
    ("06_verse_2b", 80.5, 102.5),
    ("07_outro", 102.5, 120.0),
]


@dataclass
class KineticTextCue:
    key: str
    text: str
    start: float
    end: float
    color_hex: str
    shadow_hex: str
    font_size: int
    y_center: int
    style: str  # "type_wipe" | "bubble_tilt" | "spring_3d"
    rgba: np.ndarray | None = None
    shadow_rgba: np.ndarray | None = None


KINETIC_TEXT_CUES: List[KineticTextCue] = [
    KineticTextCue("km_01", "ក្តីស្រឡាញ់ ឬ ទ្រុង?", 0.35, 5.8, "#EDE3D2", "#C84B31", 58, 255, "type_wipe"),
    KineticTextCue("km_02", "អនាគតដែលគេកំណត់ឱ្យ", 6.6, 12.6, "#111318", "#D9822B", 50, 245, "spring_3d"),
    KineticTextCue("km_03", "គ្រូពេទ្យ  ឬ  មេធាវី?", 13.0, 18.8, "#111318", "#C84B31", 52, 245, "bubble_tilt"),
    KineticTextCue("km_04", "កតញ្ញូតាធម៌ ក្លាយជាខ្សែចង", 19.8, 33.5, "#EDE3D2", "#D9822B", 50, 245, "type_wipe"),
    KineticTextCue("km_05", "ភ្ញាក់ឡើងនៅអាយុម្ភៃប្រាំឆ្នាំ", 34.2, 45.6, "#EAB308", "#C84B31", 50, 245, "spring_3d"),
    KineticTextCue("km_06", "ចិញ្ចឹមមនុស្ស ឬ បីអគារ?", 46.7, 53.2, "#111318", "#D9822B", 50, 245, "bubble_tilt"),
    KineticTextCue("km_07", "អ្នកណាទៅធ្វើការថ្ងៃចន្ទ?", 53.6, 60.4, "#EDE3D2", "#C84B31", 52, 245, "type_wipe"),
    KineticTextCue("km_08", "ការជ្រើសខ្លួនឯង មិនមែនជាការក្បត់", 61.5, 70.2, "#EDE3D2", "#D9822B", 48, 235, "spring_3d"),
    KineticTextCue("km_09", "ពួកគាត់ធ្លាប់ឆ្លងភាពភ័យខ្លាច", 70.8, 79.8, "#EAB308", "#C84B31", 50, 235, "type_wipe"),
    KineticTextCue("km_10", "កសាងភស្តុតាង ក្រោមពន្លឺថ្ងៃ", 81.0, 91.5, "#111318", "#D9822B", 52, 245, "spring_3d"),
    KineticTextCue("km_11", "ផែនការពិត មិនមែនត្រឹមពាក្យ", 92.2, 101.8, "#111318", "#C84B31", 48, 245, "bubble_tilt"),
    KineticTextCue("km_12", "ជីវិតមួយនេះ ជាហត្ថលេខារបស់អ្នក", 103.0, 118.5, "#EDE3D2", "#D9822B", 52, 255, "spring_3d"),
]


def prepare_kinetic_text_assets() -> None:
    if not KHMER_RENDER_BIN.exists():
        raise FileNotFoundError(f"Missing Khmer renderer binary: {KHMER_RENDER_BIN}")
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for cue in KINETIC_TEXT_CUES:
        fg_png = TEXT_DIR / f"{cue.key}_fg.png"
        sh_png = TEXT_DIR / f"{cue.key}_sh.png"
        subprocess.check_call(
            [str(KHMER_RENDER_BIN), cue.text, str(cue.font_size), "860", cue.color_hex, str(fg_png)],
            stdout=subprocess.DEVNULL,
        )
        subprocess.check_call(
            [str(KHMER_RENDER_BIN), cue.text, str(cue.font_size), "860", cue.shadow_hex, str(sh_png)],
            stdout=subprocess.DEVNULL,
        )
        fg = cv2.imread(str(fg_png), cv2.IMREAD_UNCHANGED)
        sh = cv2.imread(str(sh_png), cv2.IMREAD_UNCHANGED)
        if fg is None or sh is None or fg.shape[2] != 4:
            raise RuntimeError(f"Failed loading Khmer RGBA PNGs for {cue.key}")
        cue.rgba = fg
        cue.shadow_rgba = sh


def _alpha_blit(frame: np.ndarray, rgba: np.ndarray, cx: int, cy: int, fade_out: float) -> Tuple[int, int, int, int]:
    th, tw = rgba.shape[:2]
    tx0, ty0 = cx - tw // 2, cy - th // 2
    tx1, ty1 = min(W, tx0 + tw), min(H, ty0 + th)
    sx0, sy0 = max(0, -tx0), max(0, -ty0)
    tx0, ty0 = max(0, tx0), max(0, ty0)
    cw, ch = tx1 - tx0, ty1 - ty0
    if cw <= 0 or ch <= 0:
        return (0, 0, 0, 0)
    sub = rgba[sy0 : sy0 + ch, sx0 : sx0 + cw]
    a = (sub[:, :, 3:4].astype(np.float32) / 255.0) * fade_out
    fg = sub[:, :, :3].astype(np.float32)
    bg = frame[ty0:ty1, tx0:tx1].astype(np.float32)
    frame[ty0:ty1, tx0:tx1] = np.clip(fg * a + bg * (1.0 - a), 0, 255).astype(np.uint8)
    return (tx0, ty0, tx1, ty1)


def apply_kinetic_motion_text(frame: np.ndarray, t_sec: float) -> np.ndarray:
    """Animate Khmer kinetic typography with 3D extruded shadow, speech bubble, and typewriter cursor."""
    cue = next((c for c in KINETIC_TEXT_CUES if c.start <= t_sec <= c.end), None)
    if cue is None or cue.rgba is None or cue.shadow_rgba is None:
        return frame

    dt_in = t_sec - cue.start
    dt_out = cue.end - t_sec
    fade_out = ease_out_cubic(dt_out / 0.32) if dt_out < 0.32 else 1.0
    if fade_out <= 0.02:
        return frame

    fg = cue.rgba.copy()
    sh = cue.shadow_rgba.copy()
    th, tw = fg.shape[:2]

    if cue.style == "spring_3d":
        scale = 0.42 + 0.58 * ease_out_back(min(1.0, dt_in / 0.46))
        angle = 0.0
        reveal_frac = 1.0
        slide_y = int(round((1.0 - ease_out_cubic(dt_in / 0.38)) * 32.0))
    elif cue.style == "bubble_tilt":
        p_in = ease_out_back(min(1.0, dt_in / 0.45))
        scale = 0.50 + 0.50 * p_in
        angle = math.sin(min(1.0, dt_in / 1.1) * math.pi) * 3.5 - 1.5 * min(1.0, dt_in / 0.5)
        reveal_frac = 1.0
        slide_y = 0
    else:  # type_wipe
        scale = 1.0
        angle = 0.0
        reveal_frac = ease_out_cubic(min(1.0, dt_in / 0.72))
        slide_y = 0

    if abs(scale - 1.0) > 1e-3 or abs(angle) > 0.1:
        mat = cv2.getRotationMatrix2D((tw * 0.5, th * 0.5), angle, scale)
        fg = cv2.warpAffine(fg, mat, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        sh = cv2.warpAffine(sh, mat, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    if reveal_frac < 0.995:
        cut_x = max(2, int(round(tw * reveal_frac)))
        fg[:, cut_x:, 3] = 0
        sh[:, cut_x:, 3] = 0
        if 4 <= cut_x < tw - 2:
            cv2.line(fg, (cut_x, 12), (cut_x, th - 12), (8, 179, 234, 255), 5, cv2.LINE_AA)

    cx = W // 2
    cy = cue.y_center + slide_y

    # Speech-bubble rounded pill backing (mg2_04 style) for bubble_tilt hooks
    if cue.style == "bubble_tilt":
        bw = min(W - 120, int(round((tw + 56) * min(1.0, scale))))
        bh = int(round((th + 22) * min(1.0, scale)))
        bx0, by0 = max(60, cx - bw // 2), max(110, cy - bh // 2)
        bx1, by1 = min(W - 60, cx + bw // 2), min(H - 220, cy + bh // 2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (242, 240, 234), -1)
        # Speech pointer triangle below bubble
        tri = np.array([(cx - 18, by1), (cx + 18, by1), (cx - 4, by1 + 24)], np.int32)
        cv2.fillPoly(overlay, [tri], (242, 240, 234), cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.94 * fade_out, frame, 1.0 - 0.94 * fade_out, 0.0)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (20, 22, 26), 4)
    else:
        # Extruded 3D hard shadow behind text (mg2_07 SCARCITY style)
        _alpha_blit(frame, sh, cx + 6, cy + 6, fade_out * 0.92)

    _, _, _, ty1 = _alpha_blit(frame, fg, cx, cy, fade_out)

    # Underline wipe accent for non-bubble styles
    if cue.style != "bubble_tilt" and ty1 > 0:
        bar_w = int(round(min(tw * 0.72, 660) * ease_out_cubic(dt_in / 0.50)))
        by = min(H - 180, ty1 + 4)
        cv2.line(frame, (cx - bar_w // 2, by), (cx + bar_w // 2, by), (43, 130, 217), 5, cv2.LINE_AA)

    return frame


def render_video() -> None:
    prepare_kinetic_text_assets()
    generate_sfx()
    sr = build_merged_audio()

    FINAL_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-i",
        str(MERGED_WAV),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-r",
        str(FPS),
        "-c:a",
        "aac_at",
        "-b:a",
        "192k",
        "-ar",
        str(sr),
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-t",
        str(int(TOTAL_DURATION)),
        str(FINAL_VIDEO_PATH),
    ]

    print(f"[RENDER] Streaming {TOTAL_FRAMES} enhanced motion-graphic frames (Draw-Paths + Giphy Stickers + 3D Stick-Men + Khmer Kinetic Text + Narration/BGM/SFX)...")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    for f_idx in range(TOTAL_FRAMES):
        t_sec = f_idx / FPS
        s_idx = len(SCENE_WINDOWS) - 1
        for i, (_, s, e) in enumerate(SCENE_WINDOWS):
            if s <= t_sec < e:
                s_idx = i
                break

        sid, s_start, s_end = SCENE_WINDOWS[s_idx]
        frame = render_scene_frame(sid, t_sec, t_sec - s_start, s_end - s_start)

        # Crisp sliding vertical wipe / crossfade transition (mg2_07 wipe style!)
        frames_left = int(round((s_end - t_sec) * FPS))
        if s_idx < len(SCENE_WINDOWS) - 1 and 0 < frames_left <= XFADE_FRAMES:
            n_sid, n_start, n_end = SCENE_WINDOWS[s_idx + 1]
            next_frame = render_scene_frame(n_sid, t_sec, max(0.0, t_sec - n_start), n_end - n_start)
            alpha = 1.0 - (frames_left / XFADE_FRAMES)
            wipe_y = int(round(H * alpha))
            frame[0:wipe_y, :] = next_frame[0:wipe_y, :]
            cv2.line(frame, (0, wipe_y), (W, wipe_y), (43, 130, 217), 6)

        frame = apply_kinetic_motion_text(frame, t_sec)

        if f_idx < 10:
            frame = (frame.astype(np.float32) * (f_idx / 10.0)).astype(np.uint8)
        elif f_idx > TOTAL_FRAMES - 18:
            rem = max(0, TOTAL_FRAMES - 1 - f_idx)
            frame = (frame.astype(np.float32) * (rem / 18.0)).astype(np.uint8)

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg exited with status {ret}")

    print(f"[DONE] Enhanced video with Giphy stickers & motion graphics saved to: {FINAL_VIDEO_PATH}")


if __name__ == "__main__":
    render_video()
