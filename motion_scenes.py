"""
Enhanced Motion-Graphic Draw-Path Scenes (v2: Dynamic Pacing & Keyframed Giphy Stickers)
Intentionally alternates between:
  - Pure Draw-Path + Keyframed GIF Frames (no stick characters)
  - Pure 3D Stick-Character Acting Frames (no GIF clutter)
  - Dynamic Character + Keyframed Multi-Motion GIF Frames
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import cv2
import numpy as np

from stickman_3d import draw_3d_stick_character
from sticker_layer import StickerBank

W, H = 1080, 1920

BGR_BG_DARK = (20, 18, 16)
BGR_BG_PAPER = (210, 226, 236)
BGR_PARCHMENT = (210, 227, 237)
BGR_INK = (20, 22, 26)
BGR_AMBER = (43, 130, 217)
BGR_GOLD = (8, 179, 234)
BGR_SLATE = (130, 107, 74)
BGR_CRIMSON = (49, 75, 200)

_STICKERS: StickerBank | None = None
_HALFTONE_MASK: np.ndarray | None = None


def _get_stickers() -> StickerBank:
    global _STICKERS
    if _STICKERS is None:
        _STICKERS = StickerBank()
    return _STICKERS


def _get_halftone_mask() -> np.ndarray:
    global _HALFTONE_MASK
    if _HALFTONE_MASK is not None:
        return _HALFTONE_MASK
    mask = np.zeros((H, W), dtype=np.float32)
    step = 24
    for y in range(0, H, step):
        ny = abs((y - H * 0.5) / (H * 0.5))
        for x in range(0, W, step):
            nx = abs((x - W * 0.5) / (W * 0.5))
            edge = max(0.0, math.hypot(nx, ny) - 0.55)
            r = int(min(9.0, edge * 11.0))
            if r >= 2:
                cv2.circle(mask, (x + (step // 2 if (y // step) % 2 else 0), y), r, 0.24, -1)
    _HALFTONE_MASK = mask[..., None]
    return _HALFTONE_MASK


def ease_out_cubic(t: float) -> float:
    u = 1.0 - float(np.clip(t, 0.0, 1.0))
    return 1.0 - u * u * u


def ease_out_back(t: float) -> float:
    u = float(np.clip(t, 0.0, 1.0)) - 1.0
    return 1.0 + 2.70158 * u * u * u + 1.70158 * u * u


def draw_path_polyline(
    img: np.ndarray,
    pts: Sequence[Tuple[float, float]],
    progress: float,
    color: Tuple[int, int, int],
    thickness: int = 4,
    glow: bool = False,
) -> None:
    p = float(np.clip(progress, 0.0, 1.0))
    if p <= 0.0 or len(pts) < 2:
        return
    segs = []
    total_len = 0.0
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        segs.append((pts[i], pts[i + 1], d))
        total_len += d
    if total_len <= 1e-6:
        return

    target = total_len * p
    acc = 0.0
    for a, b, d in segs:
        if acc >= target:
            break
        frac = min(1.0, (target - acc) / max(1e-6, d))
        bx = a[0] + (b[0] - a[0]) * frac
        by = a[1] + (b[1] - a[1]) * frac
        pa = (int(round(a[0])), int(round(a[1])))
        pb = (int(round(bx)), int(round(by)))
        if glow:
            cv2.line(img, pa, pb, (color[0] // 2, color[1] // 2, color[2] // 2), thickness + 8, cv2.LINE_AA)
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
        acc += d


def draw_curved_ribbon(
    img: np.ndarray,
    p0: Tuple[float, float],
    ctrl: Tuple[float, float],
    p1: Tuple[float, float],
    progress: float,
    color: Tuple[int, int, int],
    thickness: int = 12,
) -> None:
    steps = 28
    pts = [
        (
            (1 - i / steps) ** 2 * p0[0] + 2 * (1 - i / steps) * (i / steps) * ctrl[0] + (i / steps) ** 2 * p1[0],
            (1 - i / steps) ** 2 * p0[1] + 2 * (1 - i / steps) * (i / steps) * ctrl[1] + (i / steps) ** 2 * p1[1],
        )
        for i in range(steps + 1)
    ]
    draw_path_polyline(img, pts, progress, BGR_INK, thickness + 5)
    draw_path_polyline(img, pts, progress, color, thickness)


def draw_path_rect(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, progress: float, color: Tuple[int, int, int], thickness: int = 4) -> None:
    draw_path_polyline(img, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], progress, color, thickness)


def draw_zigzag_motif(img: np.ndarray, x0: int, y0: int, width: int, peaks: int, progress: float, color: Tuple[int, int, int]) -> None:
    step = width / max(1, peaks * 2)
    pts = [(x0 + i * step, y0 + (16 if i % 2 == 1 else -16)) for i in range(peaks * 2 + 1)]
    draw_path_polyline(img, pts, progress, color, thickness=6, glow=True)


def draw_hatched_circle(img: np.ndarray, center: Tuple[int, int], radius: int, t_sec: float, color: Tuple[int, int, int], alpha_prog: float) -> None:
    if alpha_prog <= 0.05:
        return
    cx, cy = center
    rot = t_sec * 0.42
    c, s = math.cos(rot), math.sin(rot)
    for k in range(-radius, radius + 1, 14):
        chord = math.sqrt(max(0.0, radius * radius - k * k))
        if chord < 4:
            continue
        p1 = (int(cx + (-chord * c - k * s) * alpha_prog), int(cy + (-chord * s + k * c) * alpha_prog))
        p2 = (int(cx + (chord * c - k * s) * alpha_prog), int(cy + (chord * s + k * c) * alpha_prog))
        cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)


def make_grid_canvas(t_sec: float, light_paper: bool = False, crimson_tint: bool = False) -> np.ndarray:
    if light_paper:
        base = BGR_BG_PAPER
        g_col = (182, 198, 208)
        dot_col = np.array([150, 165, 175], dtype=np.float32)
    elif crimson_tint:
        base = (24, 20, 48)
        g_col = (44, 38, 72)
        dot_col = np.array([14, 12, 22], dtype=np.float32)
    else:
        base = BGR_BG_DARK
        g_col = (38, 35, 31)
        dot_col = np.array([58, 54, 48], dtype=np.float32)

    frame = np.full((H, W, 3), base, dtype=np.uint8)
    step = 72
    ox = int((t_sec * 16.0) % step)
    oy = int((t_sec * 11.0) % step)
    for x in range(ox, W, step):
        cv2.line(frame, (x, 0), (x, H), g_col, 1)
    for y in range(oy, H, step):
        cv2.line(frame, (0, y), (W, y), g_col, 1)

    ht = _get_halftone_mask()
    return np.clip(frame.astype(np.float32) * (1.0 - ht) + dot_col * ht, 0, 255).astype(np.uint8)


def apply_torn_paper_split(frame: np.ndarray, open_prog: float) -> np.ndarray:
    p = ease_out_cubic(open_prog)
    if p <= 0.02:
        return frame
    half_gap = int(round(415 * p))
    left_pts, right_pts = [], []
    rng = np.random.default_rng(7)
    for y in range(0, H + 40, 40):
        left_pts.append((max(0, 540 - half_gap + int(rng.integers(-22, 22))), y))
        right_pts.append((min(W, 540 + half_gap + int(rng.integers(-22, 22))), y))
    poly = np.array(left_pts + list(reversed(right_pts)), np.int32)
    cv2.fillPoly(frame, [poly], (18, 17, 16))
    cv2.polylines(frame, [np.array(left_pts, np.int32)], False, (235, 240, 244), 10, cv2.LINE_AA)
    cv2.polylines(frame, [np.array(right_pts, np.int32)], False, (235, 240, 244), 10, cv2.LINE_AA)
    return frame


def render_scene_frame(scene_id: str, t_sec: float, scene_t: float, scene_dur: float) -> np.ndarray:
    prog = scene_t / max(0.01, scene_dur)
    stickers = _get_stickers()

    # =========================================================================
    # SCENE 01 (0.0 - 6.2s): Stick character only (0-2.2s) -> Dynamic Cage + Lock GIFs (2.2-6.2s)
    # =========================================================================
    if scene_id == "01_intro":
        frame = make_grid_canvas(t_sec, light_paper=False)
        draw_p = ease_out_cubic(scene_t / 1.6)
        draw_hatched_circle(frame, (540, 1020), 295, t_sec, (44, 48, 58), min(1.0, scene_t / 0.8))

        cv2.ellipse(frame, (540, 740), (265, 190), 0, 180, int(180 + 180 * draw_p), BGR_AMBER, 6, cv2.LINE_AA)
        cv2.ellipse(frame, (540, 1480), (265, 45), 0, 0, int(360 * draw_p), BGR_AMBER, 7, cv2.LINE_AA)
        for bx in range(300, 781, 60):
            dx = abs(bx - 540) / 265.0
            top_y = 740 - 190 * math.sqrt(max(0.0, 1.0 - dx * dx))
            draw_path_polyline(frame, [(bx, top_y), (bx, 1480)], max(0.0, (draw_p - 0.15) / 0.85), BGR_AMBER, 4, glow=True)

        draw_curved_ribbon(frame, (150, 390), (310, 510), (420, 560), ease_out_cubic((scene_t - 0.4) / 1.2), BGR_SLATE, 10)
        draw_curved_ribbon(frame, (930, 390), (770, 510), (660, 560), ease_out_cubic((scene_t - 0.4) / 1.2), BGR_SLATE, 10)

        emo = "scared" if scene_t < 3.4 else "sad"
        trem = math.sin(t_sec * 24.0) * 4.0 if emo == "scared" else 0.0
        draw_3d_stick_character(
            frame, hip_xy=(540 + trem, 1265), scale=1.18, emotion=emo,
            yaw_deg=math.sin(t_sec * 1.4) * 6.0,
            l_arm=(-18, -110, 22) if emo == "scared" else (20, 15, 5),
            r_arm=(18, 110, 22) if emo == "scared" else (-20, -15, 5),
            t_sec=t_sec,
        )

        # Dynamic keyframed Giphy stickers enter at 2.2s (after clean stick-character intro)
        stickers.overlay_keyframed(
            frame, "01_birdcage",
            [(2.2, 1180, 480, 75, 18, 0.0), (2.9, 835, 620, 225, -4, 0.95), (5.5, 810, 595, 245, 3, 0.95), (6.2, 810, 560, 140, 0, 0.0)],
            t_sec, scene_t, fps_gif=8.0,
        )
        stickers.overlay_keyframed(
            frame, "07_lock",
            [(2.8, 540, 1740, 80, -25, 0.0), (3.4, 540, 1485, 185, 6, 0.98), (3.8, 540, 1495, 165, 0, 0.98), (6.2, 540, 1520, 120, 0, 0.0)],
            t_sec, scene_t, fps_gif=10.0, bob_amp=2.0,
        )

    # =========================================================================
    # SCENE 02 (6.2 - 19.2s): ONLY Stick Character acting (0-6.5s local) -> Dynamic ? Sticker (6.5-13s local)
    # =========================================================================
    elif scene_id == "02_verse_1a":
        frame = make_grid_canvas(t_sec, light_paper=True)
        p_post = ease_out_cubic(scene_t / 1.0)
        p_left = ease_out_cubic((scene_t - 0.7) / 1.2)
        p_right = ease_out_cubic((scene_t - 1.6) / 1.2)

        draw_path_polyline(frame, [(540, 1540), (540, 520)], p_post, BGR_INK, 14)
        draw_path_polyline(frame, [(540, 1540), (540, 520)], p_post, BGR_AMBER, 8)
        draw_path_polyline(frame, [(540, 620), (210, 620), (130, 675), (210, 730), (540, 730)], p_left, BGR_INK, 6)
        draw_path_polyline(frame, [(265, 645), (265, 705)], p_left, BGR_CRIMSON, 5)
        draw_path_polyline(frame, [(240, 658), (290, 658)], p_left, BGR_CRIMSON, 5)

        draw_path_polyline(frame, [(540, 810), (870, 810), (950, 865), (870, 920), (540, 920)], p_right, BGR_INK, 6)
        draw_path_polyline(frame, [(810, 835), (810, 895)], p_right, BGR_SLATE, 5)
        draw_path_polyline(frame, [(775, 850), (845, 850)], p_right, BGR_SLATE, 5)
        draw_zigzag_motif(frame, 120, 1020, 230, 4, p_left, BGR_CRIMSON)

        look = math.sin(t_sec * 1.4)
        emo = "curious" if scene_t < 8.5 else "tired"
        draw_3d_stick_character(
            frame, hip_xy=(520, 1335), scale=1.22, emotion=emo,
            yaw_deg=look * 12.0, look_x=look * 0.9, look_y=-0.35,
            l_arm=(-16, 125, 25), r_arm=(32, -35, 10), t_sec=t_sec,
        )

        # Beat 2 (6.5-13.0s local): Dynamic keyframed Question & Star stickers zoom in
        stickers.overlay_keyframed(
            frame, "02_question",
            [(6.4, 1120, 980, 75, 22, 0.0), (7.1, 800, 1110, 225, -8, 1.0), (9.8, 770, 1050, 175, 6, 1.0), (13.0, 850, 960, 90, 0, 0.0)],
            t_sec, scene_t, fps_gif=9.0,
        )

    # =========================================================================
    # SCENE 03 (19.2 - 46.2s):
    #   Beat 1 (0-6.0s local): NO STICK CHARACTER — Torn Paper + Clock & Money GIFs + Ribbons!
    #   Beat 2 (6.0-16.0s local): ONLY STICK CHARACTERS — Parents tugging SAD child (no GIFs)
    #   Beat 3 (16.0-27.0s local): ANGRY -> TIRED Stick Character + Fire & Lock GIFs!
    # =========================================================================
    elif scene_id == "03_verse_1b":
        frame = make_grid_canvas(t_sec, light_paper=True)
        frame = apply_torn_paper_split(frame, scene_t / 1.3)
        p_str = ease_out_cubic((scene_t - 0.8) / 1.4)
        draw_hatched_circle(frame, (540, 1060), 320, t_sec, (46, 48, 56), min(1.0, (scene_t - 0.4) / 0.9))

        tag_xs = [250, 410, 670, 830]
        for idx, tx in enumerate(tag_xs):
            sway = math.sin(t_sec * 1.8 + idx) * 10.0
            draw_path_polyline(frame, [(tx, 500), (tx + sway, 730)], p_str, BGR_AMBER, 4, glow=True)
            draw_path_rect(frame, tx + sway - 36, 730, tx + sway + 36, 825, p_str, BGR_PARCHMENT, 4)

        if scene_t < 6.0:
            # Beat 1: NO stick characters! Dynamic Clock & Money GIFs travel across torn void
            draw_curved_ribbon(frame, (170, 900), (540, 1260), (910, 900), p_str, BGR_CRIMSON, 12)
            stickers.overlay_keyframed(
                frame, "08_clock",
                [(0.5, 540, 1040, 70, -25, 0.0), (1.3, 540, 1050, 290, 0, 0.96), (5.3, 540, 1020, 260, 10, 0.96), (6.0, 540, 900, 110, 0, 0.0)],
                t_sec, scene_t, fps_gif=12.0,
            )
            stickers.overlay_keyframed(
                frame, "03_money",
                [(1.2, 200, 1260, 110, -14, 0.0), (2.2, 320, 1190, 215, 6, 0.96), (5.2, 760, 1190, 225, -6, 0.96), (6.0, 920, 1140, 100, 0, 0.0)],
                t_sec, scene_t, fps_gif=12.0,
            )
        else:
            # Beat 2 & 3: Stick Characters enter cleanly!
            draw_curved_ribbon(frame, (170, 950), (310, 1180), (480, 1290), p_str, BGR_CRIMSON, 10)
            draw_curved_ribbon(frame, (910, 950), (770, 1180), (600, 1290), p_str, BGR_AMBER, 10)
            tug = math.sin(t_sec * 1.9)
            emo = "sad" if scene_t < 16.0 else ("angry" if scene_t < 21.5 else "tired")
            child = draw_3d_stick_character(
                frame, hip_xy=(540 + tug * 22.0, 1345), scale=1.08, emotion=emo, lean_deg=tug * 7.0,
                l_arm=(-65 - tug * 12, -18, 12), r_arm=(65 - tug * 12, 18, 12), t_sec=t_sec,
            )
            p_left = draw_3d_stick_character(
                frame, hip_xy=(195, 1265), scale=1.02, emotion="confident", override_color="sad",
                yaw_deg=25.0, r_arm=(85 + tug * 10, 22, 18), t_sec=t_sec,
            )
            p_right = draw_3d_stick_character(
                frame, hip_xy=(885, 1265), scale=1.02, emotion="confident", override_color="sad",
                yaw_deg=-25.0, l_arm=(-85 + tug * 10, -22, 18), t_sec=t_sec,
            )
            draw_path_polyline(frame, [p_left["r_hand"], child["l_hand"]], p_str, BGR_AMBER, 4, glow=True)
            draw_path_polyline(frame, [p_right["l_hand"], child["r_hand"]], p_str, BGR_AMBER, 4, glow=True)

            # Beat 3 (16.0-27.0s): Fire sticker rises on ANGRY, then Clock hovers on TIRED
            if scene_t >= 16.0:
                stickers.overlay_keyframed(
                    frame, "09_fire",
                    [(16.0, 540, 1680, 180, 0, 0.0), (16.8, 540, 1520, 340, 0, 0.92), (21.0, 540, 1500, 310, 0, 0.85), (22.0, 540, 1620, 160, 0, 0.0)],
                    t_sec, scene_t, fps_gif=18.0, bob_amp=2.0,
                )
                stickers.overlay_keyframed(
                    frame, "08_clock",
                    [(21.2, 540, 520, 80, -15, 0.0), (22.2, 540, 610, 210, 0, 0.92), (27.0, 540, 580, 150, 8, 0.0)],
                    t_sec, scene_t, fps_gif=12.0,
                )

    # =========================================================================
    # SCENE 04 (46.2 - 61.0s):
    #   Beat 1 (0-6.8s local): NO STICK CHARACTER — Pure Satire Skyscraper vs Cradle + Flying Money GIF!
    #   Beat 2 (6.8-14.8s local): ONLY Stick Characters reacting (HAPPY parent vs SCARED child)!
    # =========================================================================
    elif scene_id == "04_hook_midpoint":
        frame = make_grid_canvas(t_sec, crimson_tint=True)
        p_draw = ease_out_cubic(scene_t / 1.3)
        draw_zigzag_motif(frame, 150, 560, 780, 8, p_draw, BGR_CRIMSON)

        draw_path_rect(frame, 165, 650, 365, 1020, p_draw, BGR_GOLD, 5)
        for wy in range(690, 980, 58):
            draw_path_rect(frame, 198, wy, 240, wy + 34, max(0.0, p_draw - 0.2), BGR_PARCHMENT, 3)
            draw_path_rect(frame, 290, wy, 332, wy + 34, max(0.0, p_draw - 0.2), BGR_PARCHMENT, 3)

        cv2.ellipse(frame, (815, 860), (130, 80), 0, 0, int(180 * p_draw), BGR_PARCHMENT, 5, cv2.LINE_AA)
        draw_path_polyline(frame, [(685, 860), (945, 860)], p_draw, BGR_AMBER, 5)

        if scene_t < 6.8:
            # Beat 1: NO stick character! Dynamic orbiting Money GIFs + bold scale pulse
            stickers.overlay_keyframed(
                frame, "03_money",
                [(0.3, 180, 1260, 110, -20, 0.0), (1.2, 540, 1180, 290, 8, 1.0), (3.8, 540, 830, 240, -6, 1.0), (6.0, 820, 1160, 260, 10, 1.0), (6.8, 980, 1220, 90, 0, 0.0)],
                t_sec, scene_t, fps_gif=14.0,
            )
        else:
            # Beat 2: ONLY Stick Characters reacting
            bounce = abs(math.sin(t_sec * 4.5)) * 18.0
            draw_3d_stick_character(
                frame, hip_xy=(290, 1345 - bounce), scale=1.08, emotion="happy",
                yaw_deg=18.0, l_arm=(-130, -20, 20), r_arm=(60, 45, 25), l_leg=(22, -25, 0), r_leg=(-18, 25, 0), t_sec=t_sec,
            )
            draw_3d_stick_character(
                frame, hip_xy=(770, 1345), scale=1.08, emotion="scared",
                yaw_deg=-18.0, l_arm=(-24, -105, 24), r_arm=(24, 105, 24), t_sec=t_sec,
            )

    # =========================================================================
    # SCENE 05 (61.0 - 80.5s):
    #   0-5.5s: ONLY SAD Stick Character + rain window (no GIF)
    #   5.5-13.5s: Animated Rainstorm GIF + sweeping Lightbulb GIF!
    #   13.5-19.5s: LOVING -> RELAXED Stick Character + rising Hearts GIF!
    # =========================================================================
    elif scene_id == "05_verse_2a":
        frame = make_grid_canvas(t_sec, light_paper=False)
        p_draw = ease_out_cubic(scene_t / 1.4)
        draw_path_rect(frame, 520, 490, 910, 870, p_draw, BGR_AMBER, 5)
        draw_path_polyline(frame, [(715, 490), (715, 870)], p_draw, BGR_SLATE, 4)
        draw_path_polyline(frame, [(520, 680), (910, 680)], p_draw, BGR_SLATE, 4)
        if p_draw > 0.6:
            for rx in range(555, 890, 48):
                ry = int(510 + ((t_sec * 210 + rx) % 330))
                if ry + 26 < 860:
                    cv2.line(frame, (rx, ry), (rx - 10, ry + 24), BGR_PARCHMENT, 2, cv2.LINE_AA)

        draw_path_polyline(frame, [(130, 1460), (950, 1460)], p_draw, BGR_PARCHMENT, 6)
        draw_path_polyline(frame, [(210, 1460), (190, 1180), (290, 1090)], p_draw, BGR_AMBER, 6, glow=True)
        if p_draw > 0.4:
            cone = frame.copy()
            cv2.fillPoly(cone, [np.array([(290, 1090), (760, 1460), (160, 1460)], np.int32)], (38, 95, 165), cv2.LINE_AA)
            frame = cv2.addWeighted(cone, 0.22 * p_draw, frame, 1.0 - 0.22 * p_draw, 0.0)

        # Dynamic keyframed Giphy stickers in Beats 2 & 3
        stickers.overlay_keyframed(
            frame, "10_rain",
            [(5.4, 960, 560, 120, 0, 0.0), (6.3, 715, 650, 320, 0, 0.90), (13.0, 715, 650, 310, 0, 0.85), (14.2, 715, 580, 160, 0, 0.0)],
            t_sec, scene_t, fps_gif=12.0, bob_amp=3.0,
        )
        stickers.overlay_keyframed(
            frame, "04_lightbulb",
            [(5.5, 160, 940, 80, -18, 0.0), (6.4, 280, 745, 225, 4, 1.0), (12.0, 300, 710, 240, -4, 1.0), (13.2, 280, 640, 95, 0, 0.0)],
            t_sec, scene_t, fps_gif=5.0,
        )
        stickers.overlay_keyframed(
            frame, "05_hearts",
            [(12.5, 540, 1250, 220, 0, 0.0), (14.0, 540, 1020, 440, 0, 0.95), (19.5, 540, 880, 520, 0, 0.0)],
            t_sec, scene_t, fps_gif=14.0, bob_amp=4.0,
        )

        emo = "sad" if scene_t < 7.0 else ("loving" if scene_t < 14.5 else "relaxed")
        draw_3d_stick_character(
            frame, hip_xy=(540, 1300), scale=1.14, emotion=emo,
            yaw_deg=-10.0, look_x=0.35, look_y=-0.35,
            l_arm=(-15, 110, 20) if emo != "relaxed" else (-115, -90, 10),
            r_arm=(25, -25, 10) if emo != "relaxed" else (115, 90, 10),
            t_sec=t_sec,
        )

    # =========================================================================
    # SCENE 06 (80.5 - 102.5s):
    #   Beat 1 (0-6.0s local): NO STICK CHARACTER — Pure Blueprint Arch Construction + Rocket GIF!
    #   Beat 2 (6.0-22.0s local): ONLY CONFIDENT 3D Stick Character walking across bridge (zero GIF clutter)!
    # =========================================================================
    elif scene_id == "06_verse_2b":
        frame = make_grid_canvas(t_sec, light_paper=True)
        p_draw = ease_out_cubic(scene_t / 2.2)
        bridge_y = 1230
        sun_c = (740, 740)
        for rad in (75, 135, 205, 285):
            cv2.circle(frame, sun_c, int(rad * p_draw), BGR_AMBER, 3, cv2.LINE_AA)
        cv2.circle(frame, sun_c, int(52 * p_draw), BGR_GOLD, -1, cv2.LINE_AA)

        draw_path_polyline(frame, [(40, bridge_y), (220, bridge_y), (280, 1680)], p_draw, BGR_INK, 7)
        draw_path_polyline(frame, [(1040, bridge_y), (860, bridge_y), (800, 1680)], p_draw, BGR_INK, 7)
        draw_path_polyline(frame, [(210, bridge_y), (870, bridge_y)], p_draw, BGR_INK, 12)
        draw_path_polyline(frame, [(210, bridge_y), (870, bridge_y)], p_draw, BGR_AMBER, 6)
        cv2.ellipse(frame, (540, bridge_y + 18), (320, 135), 0, 180, int(180 + 180 * p_draw), BGR_CRIMSON, 6, cv2.LINE_AA)
        for sx in range(280, 801, 65):
            t_arch = abs(sx - 540) / 320.0
            ay = bridge_y + 153 - 135 * math.sqrt(max(0.0, 1.0 - t_arch * t_arch))
            draw_path_polyline(frame, [(sx, bridge_y + 4), (sx, ay)], max(0.0, p_draw - 0.25), BGR_INK, 3)

        if scene_t < 6.0:
            # Beat 1: NO stick character! Dynamic flying Rocket + Lightbulb GIFs
            stickers.overlay_keyframed(
                frame, "11_rocket",
                [(0.3, 120, 1140, 120, -25, 0.0), (1.2, 260, 1020, 210, -15, 1.0), (4.8, 760, 620, 250, -5, 1.0), (6.0, 980, 440, 130, 5, 0.0)],
                t_sec, scene_t, fps_gif=14.0, bob_amp=4.0,
            )
            stickers.overlay_keyframed(
                frame, "04_lightbulb",
                [(0.6, 250, 760, 90, 0, 0.0), (1.5, 260, 730, 200, 0, 0.95), (5.2, 260, 710, 220, 0, 0.95), (6.0, 260, 670, 90, 0, 0.0)],
                t_sec, scene_t, fps_gif=5.0,
            )
        else:
            # Beat 2: ONLY CONFIDENT Stick Character walking cleanly across completed bridge
            walk_prog = (scene_t - 6.0) / max(0.1, (scene_dur - 6.0))
            walk_x = 240.0 + walk_prog * 580.0
            phase = t_sec * 5.8
            stride = math.sin(phase) * 32.0
            bob = abs(math.sin(phase)) * 7.0
            hero = draw_3d_stick_character(
                frame, hip_xy=(walk_x, bridge_y - 118 - bob), scale=0.95, emotion="confident",
                yaw_deg=62.0, lean_deg=6.0, l_arm=(-stride * 0.85, -18, 12), r_arm=(48 + stride * 0.35, -35, 18),
                l_leg=(stride, max(0.0, -math.cos(phase) * 26.0), 8), r_leg=(-stride, max(0.0, math.cos(phase) * 26.0), -8),
                t_sec=t_sec,
            )
            rh = hero["r_hand"]
            cv2.line(frame, (int(rh[0] - 6), int(rh[1] + 6)), (int(rh[0] + 40), int(rh[1] - 28)), BGR_INK, 12, cv2.LINE_AA)
            cv2.line(frame, (int(rh[0] - 6), int(rh[1] + 6)), (int(rh[0] + 40), int(rh[1] - 28)), BGR_GOLD, 7, cv2.LINE_AA)

    # =========================================================================
    # SCENE 07 (102.5 - 120.0s): Stick Character Finale + Dynamic Confetti & Star Bursts!
    # =========================================================================
    elif scene_id == "07_outro":
        frame = make_grid_canvas(t_sec, light_paper=False)
        p_draw = ease_out_cubic(scene_t / 1.5)
        for ang in (-35, 0, 35):
            rad = math.radians(ang - 90)
            draw_path_polyline(
                frame,
                [(540, 980), (540 + math.cos(rad) * 430, 980 + math.sin(rad) * 430)],
                p_draw, BGR_AMBER if ang == 0 else BGR_SLATE, 6, glow=(ang == 0),
            )
        draw_hatched_circle(frame, (540, 820), 220, t_sec, (56, 82, 118), min(1.0, scene_t / 0.8))

        # Dynamic keyframed Confetti & Stars scaling & orbiting
        stickers.overlay_keyframed(
            frame, "06_confetti",
            [(0.2, 540, 980, 360, 0, 0.0), (1.2, 540, 880, 780, 0, 0.96), (15.5, 540, 850, 840, 0, 0.96), (17.5, 540, 800, 920, 0, 0.0)],
            t_sec, scene_t, fps_gif=15.0, bob_amp=0.0,
        )
        stickers.overlay_keyframed(
            frame, "12_star",
            [(1.0, 220, 780, 70, -25, 0.0), (2.2, 240, 760, 220, 12, 0.95), (9.0, 260, 730, 245, -10, 0.95), (17.0, 240, 700, 120, 25, 0.0)],
            t_sec, scene_t, fps_gif=12.0,
        )
        stickers.overlay_keyframed(
            frame, "12_star",
            [(1.4, 860, 780, 70, 25, 0.0), (2.6, 840, 760, 220, -12, 0.95), (9.5, 820, 730, 245, 10, 0.95), (17.0, 840, 700, 120, -25, 0.0)],
            t_sec, scene_t, fps_gif=12.0,
        )

        jump = abs(math.sin(t_sec * 4.0)) * 34.0 if scene_t < 8.5 else 0.0
        emo = "excited" if scene_t < 8.5 else "confident"
        draw_3d_stick_character(
            frame, hip_xy=(540, 1320 - jump), scale=1.24, emotion=emo,
            yaw_deg=math.sin(t_sec * 1.5) * 10.0,
            l_arm=(-140, -15, 18) if emo == "excited" else (-28, 12, 6),
            r_arm=(140, 15, 18) if emo == "excited" else (115, -35, 22),
            l_leg=(26, -32, 0) if emo == "excited" else (14, -4, 4),
            r_leg=(-26, 32, 0) if emo == "excited" else (-14, 4, -4),
            t_sec=t_sec,
        )

    return frame
