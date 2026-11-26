"""
Topic 1: Image-to-Video Per-Scene 2D Motion Animator
Generates 13 lively, dynamic 2D motion-graphics MP4 clips in topic1_scenes/clips/
from the 13 scene images in topic1_scenes/, with per-scene cartoon physics & FX
synced to each scene's beats (steam, sparkles, QR scan flash, comic speed lines,
hearts, falling tears, red phone glow, arcing coins, lightbulb burst, shakes).

Usage:
  .venv/bin/python animate_topic1_scenes.py          # render missing clips
  .venv/bin/python animate_topic1_scenes.py --force  # re-render all 13 clips
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

W, H = 1080, 1920
FPS = 30
SCENE_DIR = Path("topic1_scenes")
CLIPS_DIR = SCENE_DIR / "clips"
MANIFEST_PATH = SCENE_DIR / "topic1_prompts.json"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


def load_oversampled(img_path: Path, scale_factor: float = 1.22) -> np.ndarray:
    im = Image.open(img_path).convert("RGB")
    tw, th = int(W * scale_factor), int(H * scale_factor)
    sw, sh = im.size
    sc = max(tw / sw, th / sh)
    nw, nh = int(round(sw * sc)), int(round(sh * sc))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return np.asarray(im.crop((left, top, left + tw, top + th)), dtype=np.uint8)


def crop_camera(canvas: np.ndarray, zoom: float, dx: float = 0.0, dy: float = 0.0, rot_deg: float = 0.0) -> np.ndarray:
    ch, cw = canvas.shape[:2]
    crop_w = min(cw, max(W // 2, int(round(W * (1.22 / max(0.95, zoom))))))
    crop_h = min(ch, max(H // 2, int(round(H * (1.22 / max(0.95, zoom))))))
    x0 = max(0, min(cw - crop_w, int(round((cw - crop_w) * 0.5 + dx))))
    y0 = max(0, min(ch - crop_h, int(round((ch - crop_h) * 0.5 + dy))))
    patch = canvas[y0 : y0 + crop_h, x0 : x0 + crop_w]
    frame = cv2.resize(patch, (W, H), interpolation=cv2.INTER_LINEAR)
    if abs(rot_deg) > 0.05:
        M = cv2.getRotationMatrix2D((W * 0.5, H * 0.5), rot_deg, 1.04)
        frame = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return frame


# -------------------- 2D MOTION GRAPHICS FX HELPERS --------------------

def draw_sparkle(frame: np.ndarray, cx: int, cy: int, radius: float, color_rgb: Tuple[int, int, int], alpha: float = 0.9) -> None:
    if radius < 2 or alpha <= 0.02:
        return
    r = int(radius)
    r_in = max(2, int(r * 0.28))
    pts = np.array([
        [cx, cy - r], [cx + r_in, cy - r_in],
        [cx + r, cy], [cx + r_in, cy + r_in],
        [cx, cy + r], [cx - r_in, cy + r_in],
        [cx - r, cy], [cx - r_in, cy - r_in],
    ], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color_rgb)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def draw_steam_wisp(frame: np.ndarray, base_x: int, base_y: int, t: float, phase: float, height: float = 160.0, alpha: float = 0.45) -> None:
    u = (t * 0.65 + phase) % 1.0
    fade = math.sin(math.pi * u) * alpha
    if fade <= 0.02:
        return
    overlay = frame.copy()
    pts = []
    for k in range(10):
        frac = k / 9.0
        px = int(base_x + math.sin(frac * 4.5 + t * 4.0 + phase * 6.0) * 16.0)
        py = int(base_y - (u * height * 0.55) - frac * height * 0.55)
        pts.append([px, py])
    cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, (255, 255, 255), thickness=8, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, fade, frame, 1.0 - fade, 0.0, dst=frame)


def draw_radial_glow(frame: np.ndarray, cx: int, cy: int, radius: int, color_rgb: Tuple[int, int, int], alpha: float) -> None:
    if alpha <= 0.01 or radius < 4:
        return
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(W, cx + radius), min(H, cy + radius)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / float(radius)
    mask = np.clip(1.0 - dist, 0.0, 1.0) ** 2.0 * alpha
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    col = np.array(color_rgb, dtype=np.float32)[None, None, :]
    frame[y0:y1, x0:x1] = np.clip(roi * (1.0 - mask[..., None]) + col * mask[..., None], 0, 255).astype(np.uint8)


def draw_heart(frame: np.ndarray, cx: int, cy: int, size: float, alpha: float) -> None:
    if size < 4 or alpha <= 0.02:
        return
    overlay = frame.copy()
    r = int(size * 0.48)
    col = (255, 92, 128)
    cv2.circle(overlay, (cx - r // 2, cy - r // 3), r // 2, col, -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx + r // 2, cy - r // 3), r // 2, col, -1, cv2.LINE_AA)
    tri = np.array([[cx - r, cy - r // 6], [cx + r, cy - r // 6], [cx, cy + int(size * 0.72)]], dtype=np.int32)
    cv2.fillPoly(overlay, [tri], col, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def draw_teardrop(frame: np.ndarray, cx: int, cy: int, size: float, alpha: float = 0.88) -> None:
    if alpha <= 0.02:
        return
    overlay = frame.copy()
    r = int(size * 0.5)
    col = (95, 195, 255)
    cv2.circle(overlay, (cx, cy), r, col, -1, cv2.LINE_AA)
    tri = np.array([[cx - r, cy - 2], [cx + r, cy - 2], [cx, cy - int(size * 1.15)]], dtype=np.int32)
    cv2.fillPoly(overlay, [tri], col, cv2.LINE_AA)
    cv2.circle(overlay, (cx - r // 3, cy - r // 4), max(2, r // 4), (235, 248, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def draw_coin(frame: np.ndarray, cx: int, cy: int, r: int = 24) -> None:
    cv2.circle(frame, (cx, cy), r, (65, 42, 20), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r - 3, (255, 210, 62), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r - 9, (255, 238, 130), 2, cv2.LINE_AA)


def draw_puff_ring(frame: np.ndarray, cx: int, cy: int, u: float) -> None:
    if u < 0.0 or u > 1.0:
        return
    alpha = (1.0 - u) * 0.75
    rad = int(28 + u * 110)
    overlay = frame.copy()
    for a_i in range(8):
        ang = a_i * (2.0 * math.pi / 8.0) + u * 0.4
        px = int(cx + math.cos(ang) * rad)
        py = int(cy + math.sin(ang) * rad)
        cv2.circle(overlay, (px, py), max(4, int(24 * (1.0 - u * 0.6))), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


# -------------------- PER-SCENE ANIMATION CHOREOGRAPHY --------------------

def animate_frame(sid: str, canvas: np.ndarray, t: float, dur: float) -> np.ndarray:
    u = max(0.0, min(1.0, t / max(0.001, dur)))
    ease = 1.0 - (1.0 - u) ** 2

    if sid == "00_hook":
        # Punchy opening impact + lively BBQ steam & sparkles (top) + sad tears (bottom)
        shake = math.sin(t * 72.0) * 16.0 * max(0.0, 1.0 - t / 0.28)
        zoom = 1.02 + 0.09 * ease + 0.015 * math.sin(t * 2.0 * math.pi * 2.0) * max(0.0, 1.0 - t / 1.5)
        frame = crop_camera(canvas, zoom, dx=shake, dy=shake * 0.5)
        for i, bx in enumerate((410, 510, 610)):
            draw_steam_wisp(frame, bx, 690, t, i * 0.33, height=170, alpha=0.50)
        for i, (sx, sy) in enumerate([(220, 260), (880, 290), (830, 520), (260, 540)]):
            pulse = 0.5 + 0.5 * math.sin(t * 7.5 + i * 1.6)
            draw_sparkle(frame, sx, sy, 14 + 18 * pulse, (255, 248, 175), alpha=0.55 + 0.4 * pulse)
        # Bottom half teardrops falling
        for i, tx in enumerate((445, 625)):
            tu = (t * 1.15 + i * 0.5) % 1.0
            draw_teardrop(frame, tx, int(1310 + tu * 220), 18, alpha=math.sin(math.pi * tu) * 0.85)
        return frame

    if sid == "01_payday_khqr":
        # Bouncy payday groove + bright QR scan flash burst right at t=2.6s (global 5.6s QR BEEP)
        beat_bounce = abs(math.sin(t * math.pi * 2.0)) * 9.0
        zoom = 1.02 + 0.10 * ease
        frame = crop_camera(canvas, zoom, dx=-18 * ease, dy=-beat_bounce)
        # Golden sparkle bursts
        for i, (sx, sy) in enumerate([(190, 420), (870, 380), (240, 820), (850, 740), (540, 340)]):
            p = 0.5 + 0.5 * math.sin(t * 6.5 + i * 1.3)
            draw_sparkle(frame, sx, sy, 12 + 22 * p, (255, 250, 190), alpha=0.45 + 0.5 * p)
        # Big QR scan flash burst at t=2.55s..3.25s
        if 2.45 <= t <= 3.35:
            fu = (t - 2.45) / 0.90
            flash_a = math.sin(math.pi * fu) ** 0.7
            draw_radial_glow(frame, 725, 1015, int(240 + 280 * fu), (255, 255, 245), alpha=0.82 * flash_a)
            draw_sparkle(frame, 725, 1015, 70 + 95 * flash_a, (255, 255, 255), alpha=0.95 * flash_a)
        return frame

    if sid == "02a_iced_coffee":
        # Confident strutting walk-cycle head-bob + speed lines + glints
        bob = abs(math.sin(t * math.pi * 2.6)) * 18.0
        tilt = math.sin(t * math.pi * 2.6) * 1.3
        zoom = 1.08 + 0.06 * ease
        frame = crop_camera(canvas, zoom, dx=-55 + 110 * ease, dy=-bob, rot_deg=tilt)
        # Comic speed lines zipping left
        overlay = frame.copy()
        for i in range(9):
            ly = 340 + i * 145
            lx = int((W + 300) - ((t * 1650 + i * 240) % (W + 500)))
            cv2.line(overlay, (lx, ly), (lx + 190, ly - 24), (255, 255, 255), 4, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, dst=frame)
        # Sunglasses glint
        gp = max(0.0, math.sin(t * 5.0))
        draw_sparkle(frame, 560, 620, 28 * gp, (255, 255, 255), alpha=0.9 * gp)
        return frame

    if sid == "02b_treat_bbq":
        # Lively celebratory bounce + swirling BBQ steam + floating hearts & stars
        bounce = abs(math.sin(t * math.pi * 2.4)) * 10.0
        frame = crop_camera(canvas, 1.02 + 0.09 * ease, dy=-bounce)
        for i, bx in enumerate((420, 540, 660)):
            draw_steam_wisp(frame, bx, 1180, t, i * 0.33, height=200, alpha=0.48)
        for i, hx in enumerate((210, 340, 760, 880)):
            hu = (t * 0.55 + i * 0.27) % 1.0
            hy = int(860 - hu * 360)
            hx_w = int(hx + math.sin(t * 4.0 + i) * 18)
            if i % 2 == 0:
                draw_heart(frame, hx_w, hy, 34, alpha=math.sin(math.pi * hu) * 0.85)
            else:
                draw_sparkle(frame, hx_w, hy, 22, (255, 236, 120), alpha=math.sin(math.pi * hu) * 0.9)
        return frame

    if sid == "02c_delivery_parcels":
        # Rapid parcel-drop squish impacts synced to pop-pop-pop SFX (t=1.0, 1.45, 1.9, 2.35, 2.8, 3.2)
        imp = 0.0
        for pt in (1.0, 1.45, 1.9, 2.35, 2.8, 3.2):
            dt = t - pt
            if 0.0 <= dt < 0.22:
                imp += math.sin(dt / 0.22 * math.pi) * 20.0
        frame = crop_camera(canvas, 1.02 + 0.11 * ease + imp * 0.0012, dy=imp)
        # Starry phone glow pulse
        draw_radial_glow(frame, 485, 1170, 160, (255, 240, 160), alpha=0.22 + 0.12 * math.sin(t * 9.0))
        for i in range(4):
            p = 0.5 + 0.5 * math.sin(t * 8.0 + i * 1.5)
            draw_sparkle(frame, 360 + i * 85, 1080 + (i % 2) * 55, 14 + 14 * p, (255, 244, 150), alpha=0.8 * p)
        return frame

    if sid == "03_day12_shock":
        # Hard impact slam + frantic shock camera shake + scrolling panic lines + pulsing red phone glow + sliding sweat drop
        shake_x = math.sin(t * 78.0) * 26.0 * max(0.0, 1.0 - t / 0.65)
        shake_y = math.cos(t * 64.0) * 20.0 * max(0.0, 1.0 - t / 0.65)
        zoom = 1.04 + 0.11 * ease
        frame = crop_camera(canvas, zoom, dx=shake_x, dy=shake_y)
        # Dramatic downward panic speed lines
        overlay = frame.copy()
        for i in range(12):
            lx = 70 + i * 88
            ly = int((t * 950 + i * 190) % (H + 300)) - 200
            cv2.line(overlay, (lx, ly), (lx - 18, ly + 220), (35, 48, 65), 3, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0.0, dst=frame)
        # Red smartphone glow pulse (intensifies after 2s silence at t=2.25s!)
        red_p = 0.35 + 0.25 * math.sin(t * 7.5) if t >= 2.15 else 0.18
        draw_radial_glow(frame, 740, 980, 260, (255, 55, 55), alpha=red_p)
        # Sliding cartoon sweat drop
        su = (t * 0.45) % 1.0
        draw_teardrop(frame, 375, int(680 + su * 150), 20, alpha=math.sin(math.pi * su) * 0.85)
        return frame

    if sid == "04a_airplane_mode":
        # Shivering anxiety under blanket + pulsing blue airplane halo + bouncing red chat dots that pop away at t=2.0s
        shiver = math.sin(t * 42.0) * (5.5 if t < 2.0 else 2.5)
        frame = crop_camera(canvas, 1.02 + 0.08 * ease, dx=shiver)
        # Pulsing blue glow on airplane icon
        halo_a = 0.25 + 0.18 * math.sin(t * 6.0)
        if 1.95 <= t <= 2.45:
            halo_a = 0.65 * math.sin((t - 1.95) / 0.50 * math.pi)
        draw_radial_glow(frame, 785, 735, 220, (80, 185, 255), alpha=halo_a)
        if 1.95 <= t <= 2.45:
            draw_puff_ring(frame, 785, 735, (t - 1.95) / 0.50)
        return frame

    if sid == "04b_coffee_downgrade":
        # Pan left-to-right across the 3 drinks + cartoon poof rings synced to poof-poof SFX at t=1.0s & t=2.15s
        frame = crop_camera(canvas, 1.08 + 0.04 * ease, dx=-50 + 100 * ease)
        if 0.95 <= t <= 1.50:
            draw_puff_ring(frame, 540, 1260, (t - 0.95) / 0.55)
        if 2.10 <= t <= 2.65:
            draw_puff_ring(frame, 880, 1290, (t - 2.10) / 0.55)
        return frame

    if sid == "04c_noodles_bbq_photo":
        # Slow moody zoom + thin steam from noodles + warm BBQ phone glow + single falling tear synced to t=3.4s
        frame = crop_camera(canvas, 1.01 + 0.09 * ease, dx=10 * ease, dy=8 * ease)
        draw_steam_wisp(frame, 610, 1080, t, 0.1, height=130, alpha=0.38)
        draw_radial_glow(frame, 765, 965, 190, (255, 165, 70), alpha=0.22 + 0.08 * math.sin(t * 4.5))
        if t >= 2.8:
            tu = min(1.0, (t - 2.8) / 1.4)
            draw_teardrop(frame, 615, int(810 + tu * 165), 17, alpha=math.sin(math.pi * tu) * 0.92)
        return frame

    if sid == "05_lightbulb_idea":
        # Crisp energetic punch-in + bright lightbulb flash burst synced to finger snap + ding at t=0.5s!
        punch = 1.02 + 0.10 * ease + (0.04 * math.sin((t - 0.45) / 0.35 * math.pi) if 0.45 <= t <= 0.80 else 0.0)
        frame = crop_camera(canvas, punch, dy=-12 * ease)
        if t >= 0.42:
            burst = math.exp(-(t - 0.42) * 2.8)
            draw_radial_glow(frame, 750, 505, int(230 + 190 * (1.0 - burst)), (255, 240, 110), alpha=0.32 + 0.55 * burst)
            for i, (sx, sy) in enumerate([(610, 380), (885, 370), (890, 610), (615, 620)]):
                p = 0.5 + 0.5 * math.sin(t * 8.5 + i * 1.4)
                draw_sparkle(frame, sx, sy, 14 + 22 * p, (255, 226, 85), alpha=0.6 + 0.38 * p)
        return frame

    if sid == "06_three_jars":
        # Smooth uplifting zoom + animated golden coins arcing into the 3 jars + padlock sparkle click at t=12.6s!
        frame = crop_camera(canvas, 1.01 + 0.08 * ease)
        # Arcing coins synced to coin_times in topic1_generate_audio.py (local offsets from 44.0s)
        jar_targets = [(190, 1290), (540, 1290), (895, 1290)]
        local_coin_times = [1.2, 1.7, 2.2, 4.2, 4.7, 5.2, 7.4, 7.9, 8.4, 10.0, 10.5, 11.0]
        for idx, ct in enumerate(local_coin_times):
            cu = (t - (ct - 0.45)) / 0.55
            if 0.0 <= cu <= 1.0:
                tx, ty = jar_targets[idx % 3]
                sx, sy = 710, 960
                cx = int(sx + (tx - sx) * cu)
                cy = int(sy + (ty - sy) * cu - math.sin(math.pi * cu) * 220)
                draw_coin(frame, cx, cy, r=22)
        # Padlock golden sparkle burst at t=12.6s..13.8s
        if 12.55 <= t <= 14.0:
            pu = (t - 12.55) / 1.45
            pa = math.sin(math.pi * pu)
            draw_radial_glow(frame, 895, 1265, 180, (255, 235, 110), alpha=0.65 * pa)
            draw_sparkle(frame, 895, 1265, 45 + 45 * pa, (255, 255, 220), alpha=0.95 * pa)
        return frame

    if sid == "07_new_day25":
        # Warm peaceful zoom + gentle steam from rice/soup bowls + glowing green savings jar aura + sparkles
        frame = crop_camera(canvas, 1.01 + 0.07 * ease, dx=10 * ease, dy=-6 * ease)
        draw_steam_wisp(frame, 445, 1180, t, 0.0, height=130, alpha=0.40)
        draw_steam_wisp(frame, 635, 1210, t, 0.5, height=120, alpha=0.35)
        glow_p = 0.22 + 0.12 * math.sin(t * 3.8)
        draw_radial_glow(frame, 915, 1130, 210, (110, 240, 150), alpha=glow_p)
        for i, (sx, sy) in enumerate([(845, 950), (960, 975), (905, 895)]):
            p = 0.5 + 0.5 * math.sin(t * 5.0 + i * 2.0)
            draw_sparkle(frame, sx, sy, 10 + 14 * p, (235, 255, 180), alpha=0.45 + 0.45 * p)
        return frame

    if sid == "08_outro":
        # Gentle friendly hold/zoom + glowing jars with floating golden sparkles + smooth 1.5s fade-out at end
        frame = crop_camera(canvas, 1.00 + 0.05 * ease)
        for i, (jx, jy) in enumerate([(685, 1210), (815, 1210), (955, 1210)]):
            p = 0.5 + 0.5 * math.sin(t * 4.2 + i * 1.4)
            draw_radial_glow(frame, jx, jy, 120, (255, 245, 175), alpha=0.14 + 0.10 * p)
            draw_sparkle(frame, jx, int(jy - 115 - 25 * p), 10 + 12 * p, (255, 230, 110), alpha=0.5 + 0.45 * p)
        return frame

    return crop_camera(canvas, 1.01 + 0.06 * ease)


def render_scene_clip(scene: dict, force: bool = False) -> Path:
    sid = scene["id"]
    img_path = SCENE_DIR / scene["file"]
    clip_path = CLIPS_DIR / f"scene_{sid}.mp4"
    if clip_path.exists() and not force:
        print(f"[SKIP] Using existing clip: {clip_path}")
        return clip_path

    dur = float(scene["end"] - scene["start"])
    n_frames = int(round(dur * FPS))
    canvas = load_oversampled(img_path)

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "16",
        "-pix_fmt", "yuv420p", "-an",
        str(clip_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    for f in range(n_frames):
        frame = animate_frame(sid, canvas, f / FPS, dur)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg failed on {sid} with code {ret}")
    print(f"[CLIP] Animated {clip_path} ({dur:.1f}s, {n_frames} frames)")
    return clip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-render all 13 scene clips")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for scene in manifest["scenes"]:
        render_scene_clip(scene, force=args.force)
    print(f"[DONE] All {len(manifest['scenes'])} animated scene clips ready in {CLIPS_DIR}/")


if __name__ == "__main__":
    main()
