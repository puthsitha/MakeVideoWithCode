"""
3D Cel-Shaded Glossy Stick-Character Engine (inspired by 10-emotion reference sheet)
Projects 3D joint coordinates (x, y, z) with depth sorting, thick ink outlines,
volumetric capsule/sphere 3-tone shading + glossy specular highlights,
and animated emotion props (!, ?, rain cloud, steam puffs, hearts, sunglasses, sparkles, Zzz).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

BGR_INK = (16, 18, 22)
BGR_WHITE = (250, 250, 252)

# Emotion -> (base BGR, shadow BGR, highlight BGR)
EMOTION_PALETTES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]] = {
    "angry":     ((48, 56, 232),   (28, 30, 155),   (125, 138, 255)),   # Red
    "happy":     ((24, 145, 250),  (12, 92, 175),   (115, 205, 255)),   # Orange
    "curious":   ((38, 202, 252),  (18, 135, 178),  (145, 236, 255)),   # Yellow
    "relaxed":   ((74, 172, 68),   (42, 112, 38),   (148, 225, 145)),   # Green
    "sad":       ((230, 136, 32),  (150, 82, 16),   (255, 196, 112)),   # Blue
    "scared":    ((182, 48, 148),  (112, 24, 92),   (232, 132, 212)),   # Purple
    "loving":    ((136, 72, 242),  (88, 36, 168),   (198, 155, 255)),   # Pink
    "excited":   ((222, 202, 42),  (142, 130, 20),  (255, 244, 140)),   # Cyan
    "confident": ((196, 98, 24),   (124, 58, 12),   (248, 165, 96)),    # Deep Blue
    "tired":     ((236, 232, 224), (168, 165, 158), (255, 255, 255)),   # Off-White
}


def _rot_y(pt: Tuple[float, float, float], yaw_deg: float) -> Tuple[float, float, float]:
    r = math.radians(yaw_deg)
    c, s = math.cos(r), math.sin(r)
    x, y, z = pt
    return (x * c + z * s, y, -x * s + z * c)


def _proj(pt: Tuple[float, float, float], origin: Tuple[float, float], scale: float) -> Tuple[float, float, float]:
    # Gentle perspective projection so front limbs look subtly larger/closer
    x, y, z = pt
    persp = 900.0 / (900.0 - z * scale)
    return (origin[0] + x * scale * persp, origin[1] + y * scale * persp, z)


def _draw_3d_capsule(
    img: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    radius: int,
    palette: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
) -> None:
    """Render a 3D-shaded capsule limb with ink outline, shadow body, core fill, and specular streak."""
    base, shadow, hi = palette
    a = (int(round(p0[0])), int(round(p0[1])))
    b = (int(round(p1[0])), int(round(p1[1])))
    r = max(4, radius)

    # 1. Outer thick dark outline
    cv2.line(img, a, b, BGR_INK, r * 2 + 10, cv2.LINE_AA)
    # 2. Shadow side base
    cv2.line(img, a, b, shadow, r * 2, cv2.LINE_AA)
    # 3. Lit main volume shifted toward top-left light source
    off = max(2, int(round(r * 0.24)))
    a_lit = (a[0] - off, a[1] - off)
    b_lit = (b[0] - off, b[1] - off)
    cv2.line(img, a_lit, b_lit, base, max(4, int(round(r * 1.55))), cv2.LINE_AA)
    # 4. Specular highlight stripe
    off_hi = max(2, int(round(r * 0.36)))
    a_hi = (a[0] - off_hi, a[1] - off_hi)
    b_hi = (b[0] - off_hi, b[1] - off_hi)
    cv2.line(img, a_hi, b_hi, hi, max(2, int(round(r * 0.42))), cv2.LINE_AA)


def _draw_3d_foot(
    img: np.ndarray,
    pt: Tuple[float, float],
    facing: float,
    radius: int,
    palette: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
) -> None:
    """Rounded boot-like foot matching the reference character design."""
    base, shadow, hi = palette
    cx, cy = int(round(pt[0] + facing * radius * 0.65)), int(round(pt[1]))
    rx, ry = int(round(radius * 1.65)), int(round(radius * 0.85))
    cv2.ellipse(img, (cx, cy), (rx + 5, ry + 5), 0, 0, 360, BGR_INK, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, shadow, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx - 2, cy - 2), (max(3, rx - 3), max(3, ry - 3)), 0, 0, 360, base, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx - int(rx * 0.25), cy - int(ry * 0.32)), (max(2, rx // 3), max(2, ry // 4)), 0, 0, 360, hi, -1, cv2.LINE_AA)


def _draw_3d_head_and_face(
    img: np.ndarray,
    center: Tuple[float, float],
    radius: int,
    emotion: str,
    palette: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
    look_x: float = 0.0,
    look_y: float = 0.0,
    t_sec: float = 0.0,
) -> None:
    base, shadow, hi = palette
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = max(18, radius)

    # 3D shaded sphere head
    cv2.circle(img, (cx, cy), r + 6, BGR_INK, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, shadow, -1, cv2.LINE_AA)
    cv2.circle(img, (cx - int(r * 0.10), cy - int(r * 0.10)), int(r * 0.88), base, -1, cv2.LINE_AA)
    # Glossy crescent highlight top-left
    cv2.ellipse(
        img,
        (cx - int(r * 0.34), cy - int(r * 0.36)),
        (max(4, int(r * 0.34)), max(3, int(r * 0.18))),
        -35,
        0,
        360,
        hi,
        -1,
        cv2.LINE_AA,
    )

    # Face center shifted subtly for 3D look direction
    fx = cx + int(round(look_x * r * 0.22))
    fy = cy + int(round(look_y * r * 0.16))
    eye_dx = int(round(r * 0.34))
    eye_y = fy - int(round(r * 0.06))
    eye_r = max(5, int(round(r * 0.19)))

    if emotion == "confident":
        # Cool dark sunglasses with white diagonal sheen
        gw, gh = int(r * 0.42), int(r * 0.25)
        cv2.line(img, (fx - r + 4, eye_y - 4), (fx + r - 4, eye_y - 4), BGR_INK, 5, cv2.LINE_AA)
        for sx in (fx - int(r * 0.40), fx + int(r * 0.40)):
            cv2.rectangle(img, (sx - gw // 2, eye_y - gh // 2), (sx + gw // 2, eye_y + gh // 2), BGR_INK, -1)
            cv2.line(img, (sx - gw // 4, eye_y - gh // 3), (sx + gw // 5, eye_y + gh // 3), BGR_WHITE, 2, cv2.LINE_AA)
        # Confident smirk
        cv2.ellipse(img, (fx + 3, fy + int(r * 0.32)), (int(r * 0.26), int(r * 0.14)), -8, 15, 165, BGR_INK, 4, cv2.LINE_AA)
        return

    if emotion == "happy" or emotion == "excited":
        if emotion == "happy":
            # Joyful curved eyes ^ ^
            for sx in (fx - eye_dx, fx + eye_dx):
                cv2.ellipse(img, (sx, eye_y + 3), (eye_r, eye_r), 0, 195, 345, BGR_INK, 4, cv2.LINE_AA)
        else:
            # Big sparkling open eyes
            for sx in (fx - eye_dx, fx + eye_dx):
                cv2.circle(img, (sx, eye_y), eye_r + 2, BGR_INK, -1, cv2.LINE_AA)
                cv2.circle(img, (sx, eye_y), eye_r, BGR_WHITE, -1, cv2.LINE_AA)
                cv2.circle(img, (sx + 1, eye_y + 1), max(3, eye_r // 2), BGR_INK, -1, cv2.LINE_AA)
        # Wide open smiling mouth with red tongue
        mw, mh = int(r * 0.46), int(r * 0.34)
        my = fy + int(r * 0.24)
        cv2.ellipse(img, (fx, my), (mw, mh), 0, 0, 180, BGR_INK, -1, cv2.LINE_AA)
        cv2.ellipse(img, (fx, my + int(mh * 0.32)), (int(mw * 0.55), int(mh * 0.45)), 0, 0, 180, (70, 72, 230), -1, cv2.LINE_AA)
        return

    if emotion == "relaxed":
        # Peaceful closed eyelids + gentle smile
        for sx in (fx - eye_dx, fx + eye_dx):
            cv2.ellipse(img, (sx, eye_y), (eye_r, int(eye_r * 0.65)), 0, 15, 165, BGR_INK, 4, cv2.LINE_AA)
        cv2.ellipse(img, (fx, fy + int(r * 0.28)), (int(r * 0.26), int(r * 0.14)), 0, 15, 165, BGR_INK, 4, cv2.LINE_AA)
        return

    if emotion == "loving":
        # One warm closed wink eye, one sparkling eye, gentle kiss mouth
        cv2.ellipse(img, (fx - eye_dx, eye_y + 2), (eye_r, int(eye_r * 0.7)), 0, 195, 345, BGR_INK, 4, cv2.LINE_AA)
        cv2.circle(img, (fx + eye_dx, eye_y), eye_r, BGR_INK, -1, cv2.LINE_AA)
        cv2.circle(img, (fx + eye_dx, eye_y), eye_r - 2, BGR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, (fx + eye_dx - 1, eye_y), max(3, eye_r // 2), BGR_INK, -1, cv2.LINE_AA)
        cv2.ellipse(img, (fx, fy + int(r * 0.28)), (int(r * 0.24), int(r * 0.14)), 0, 15, 165, BGR_INK, 4, cv2.LINE_AA)
        return

    # Standard cartoon eyes (white sclera + expressive pupils + eyelids/brows)
    for sx in (fx - eye_dx, fx + eye_dx):
        cv2.circle(img, (sx, eye_y), eye_r + 2, BGR_INK, -1, cv2.LINE_AA)
        cv2.circle(img, (sx, eye_y), eye_r, BGR_WHITE, -1, cv2.LINE_AA)
        pr = max(3, int(eye_r * (0.42 if emotion == "scared" else 0.58)))
        px = sx + int(round(look_x * eye_r * 0.35))
        py = eye_y + int(round(look_y * eye_r * 0.35))
        cv2.circle(img, (px, py), pr, BGR_INK, -1, cv2.LINE_AA)
        cv2.circle(img, (px - 2, py - 2), max(1, pr // 3), BGR_WHITE, -1, cv2.LINE_AA)

    if emotion == "angry":
        # Sharp V-shaped angry brows + grit mouth
        cv2.line(img, (fx - eye_dx - eye_r, eye_y - eye_r - 6), (fx - 4, eye_y - eye_r + 4), BGR_INK, 5, cv2.LINE_AA)
        cv2.line(img, (fx + eye_dx + eye_r, eye_y - eye_r - 6), (fx + 4, eye_y - eye_r + 4), BGR_INK, 5, cv2.LINE_AA)
        cv2.rectangle(img, (fx - int(r * 0.26), fy + int(r * 0.30)), (fx + int(r * 0.26), fy + int(r * 0.44)), BGR_WHITE, -1)
        cv2.rectangle(img, (fx - int(r * 0.26), fy + int(r * 0.30)), (fx + int(r * 0.26), fy + int(r * 0.44)), BGR_INK, 3)
    elif emotion == "sad":
        # Droopy inner-up slanted brows + frown
        cv2.line(img, (fx - eye_dx - eye_r, eye_y - eye_r + 4), (fx - 6, eye_y - eye_r - 5), BGR_INK, 4, cv2.LINE_AA)
        cv2.line(img, (fx + eye_dx + eye_r, eye_y - eye_r + 4), (fx + 6, eye_y - eye_r - 5), BGR_INK, 4, cv2.LINE_AA)
        cv2.ellipse(img, (fx, fy + int(r * 0.44)), (int(r * 0.24), int(r * 0.14)), 0, 195, 345, BGR_INK, 4, cv2.LINE_AA)
    elif emotion == "scared":
        # High worried eyebrows + wavy shocked open mouth
        cv2.ellipse(img, (fx - eye_dx, eye_y - eye_r - 5), (eye_r, 5), 0, 195, 345, BGR_INK, 3, cv2.LINE_AA)
        cv2.ellipse(img, (fx + eye_dx, eye_y - eye_r - 5), (eye_r, 5), 0, 195, 345, BGR_INK, 3, cv2.LINE_AA)
        cv2.ellipse(img, (fx, fy + int(r * 0.36)), (int(r * 0.22), int(r * 0.16)), 0, 0, 360, BGR_INK, -1, cv2.LINE_AA)
        cv2.ellipse(img, (fx, fy + int(r * 0.39)), (int(r * 0.13), int(r * 0.08)), 0, 0, 360, (70, 72, 220), -1, cv2.LINE_AA)
    elif emotion == "curious":
        # One raised curious brow + small thoughtful mouth
        cv2.ellipse(img, (fx - eye_dx, eye_y - eye_r - 4), (eye_r, 5), 0, 195, 345, BGR_INK, 3, cv2.LINE_AA)
        cv2.ellipse(img, (fx + eye_dx, eye_y - eye_r - 8), (eye_r, 6), 0, 195, 345, BGR_INK, 4, cv2.LINE_AA)
        cv2.line(img, (fx - int(r * 0.16), fy + int(r * 0.34)), (fx + int(r * 0.14), fy + int(r * 0.31)), BGR_INK, 4, cv2.LINE_AA)
    elif emotion == "tired":
        # Half-closed heavy eyelids + yawning mouth
        for sx in (fx - eye_dx, fx + eye_dx):
            cv2.rectangle(img, (sx - eye_r, eye_y - eye_r), (sx + eye_r, eye_y + 1), base, -1)
            cv2.line(img, (sx - eye_r, eye_y + 1), (sx + eye_r, eye_y + 1), BGR_INK, 4, cv2.LINE_AA)
        cv2.ellipse(img, (fx, fy + int(r * 0.35)), (int(r * 0.16), int(r * 0.20)), 0, 0, 360, BGR_INK, -1, cv2.LINE_AA)
        cv2.ellipse(img, (fx, fy + int(r * 0.37)), (int(r * 0.10), int(r * 0.12)), 0, 0, 360, (80, 70, 210), -1, cv2.LINE_AA)


def _draw_emotion_props(img: np.ndarray, head_pt: Tuple[float, float], r: int, emotion: str, t_sec: float) -> None:
    hx, hy = int(round(head_pt[0])), int(round(head_pt[1]))
    bob = math.sin(t_sec * 3.5) * 8.0

    if emotion == "scared":
        # Floating bold "!" above head
        px, py = hx + int(r * 0.85), hy - int(r * 1.45 + abs(bob))
        cv2.line(img, (px, py - 44), (px, py - 14), BGR_INK, 14, cv2.LINE_AA)
        cv2.line(img, (px, py - 44), (px, py - 14), (232, 132, 212), 8, cv2.LINE_AA)
        cv2.circle(img, (px, py), 8, BGR_INK, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), 5, (232, 132, 212), -1, cv2.LINE_AA)

    elif emotion == "curious":
        # Floating "?" mark above head
        px, py = hx + int(r * 0.35), hy - int(r * 1.75 + bob)
        cv2.ellipse(img, (px, py - 24), (16, 16), 0, 175, 390, BGR_INK, 12, cv2.LINE_AA)
        cv2.ellipse(img, (px, py - 24), (16, 16), 0, 175, 390, (38, 202, 252), 6, cv2.LINE_AA)
        cv2.line(img, (px + 4, py - 10), (px, py - 2), BGR_INK, 12, cv2.LINE_AA)
        cv2.line(img, (px + 4, py - 10), (px, py - 2), (38, 202, 252), 6, cv2.LINE_AA)
        cv2.circle(img, (px, py + 12), 7, BGR_INK, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py + 12), 4, (38, 202, 252), -1, cv2.LINE_AA)

    elif emotion == "sad":
        # Dark rain cloud + dripping blue raindrops above head
        cx, cy = hx, hy - int(r * 1.85)
        for ox, oy, cr in [(-24, 4, 18), (0, -6, 24), (24, 4, 18)]:
            cv2.circle(img, (cx + ox, cy + oy), cr + 4, BGR_INK, -1, cv2.LINE_AA)
            cv2.circle(img, (cx + ox, cy + oy), cr, (75, 78, 86), -1, cv2.LINE_AA)
        for idx, rx in enumerate((-20, 0, 20)):
            drop_y = cy + 28 + int(((t_sec * 95 + idx * 14) % 42))
            cv2.line(img, (cx + rx, drop_y), (cx + rx, drop_y + 10), (245, 180, 90), 3, cv2.LINE_AA)

    elif emotion == "angry":
        # Twin white steam puffs popping left & right of head
        puff = abs(math.sin(t_sec * 6.0)) * 8.0
        for side in (-1, 1):
            px = hx + side * int(r * 1.15)
            py = hy - int(r * 1.25 + puff)
            for ox, oy, pr in [(-8, 4, 11), (6, -4, 14), (10, 6, 9)]:
                cv2.circle(img, (px + ox * side, py + oy), pr + 3, BGR_INK, -1, cv2.LINE_AA)
                cv2.circle(img, (px + ox * side, py + oy), pr, BGR_WHITE, -1, cv2.LINE_AA)

    elif emotion == "loving":
        # Floating pink heart above head
        px, py = hx + int(r * 0.75), hy - int(r * 1.45 - bob)
        cv2.circle(img, (px - 9, py - 4), 11, BGR_INK, -1, cv2.LINE_AA)
        cv2.circle(img, (px + 9, py - 4), 11, BGR_INK, -1, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([(px - 20, py), (px + 20, py), (px, py + 22)], np.int32)], BGR_INK, cv2.LINE_AA)
        cv2.circle(img, (px - 9, py - 4), 8, (136, 72, 242), -1, cv2.LINE_AA)
        cv2.circle(img, (px + 9, py - 4), 8, (136, 72, 242), -1, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([(px - 16, py), (px + 16, py), (px, py + 18)], np.int32)], (136, 72, 242), cv2.LINE_AA)

    elif emotion in ("confident", "excited"):
        # 4-point glowing star sparkle
        sx, sy = hx + int(r * 1.25), hy - int(r * 1.1 + bob * 0.5)
        sp = int(16 + math.sin(t_sec * 5.0) * 4)
        pts = np.array([(sx, sy - sp), (sx + 5, sy - 5), (sx + sp, sy), (sx + 5, sy + 5), (sx, sy + sp), (sx - 5, sy + 5), (sx - sp, sy), (sx - 5, sy - 5)], np.int32)
        cv2.fillPoly(img, [pts], BGR_WHITE, cv2.LINE_AA)


def draw_3d_stick_character(
    img: np.ndarray,
    hip_xy: Tuple[float, float],
    scale: float = 1.0,
    emotion: str = "curious",
    yaw_deg: float = 0.0,
    lean_deg: float = 0.0,
    head_tilt_deg: float = 0.0,
    look_x: float = 0.0,
    look_y: float = 0.0,
    l_arm: Tuple[float, float, float] = (28.0, 22.0, 8.0),    # (pitch_deg, elbow_flex_deg, z_deg)
    r_arm: Tuple[float, float, float] = (-28.0, -22.0, -8.0),
    l_leg: Tuple[float, float, float] = (14.0, -6.0, 4.0),
    r_leg: Tuple[float, float, float] = (-14.0, 6.0, -4.0),
    t_sec: float = 0.0,
    override_color: str | None = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Draw a volumetric 3D-projected cel-shaded capsule stick character.
    Returns 2D screen coordinates of key joints (head, l_hand, r_hand, hip).
    """
    pal = EMOTION_PALETTES[override_color or emotion]

    # 3D skeleton lengths (local coords, y-down)
    torso_len = 104.0
    head_r = max(24, int(round(44.0 * scale)))
    limb_r = max(7, int(round(13.5 * scale)))
    torso_r = max(9, int(round(16.5 * scale)))

    # Torso shoulder top in 3D
    lr = math.radians(lean_deg)
    hip_3d = (0.0, 0.0, 0.0)
    sh_3d = (math.sin(lr) * torso_len, -math.cos(lr) * torso_len, 0.0)
    hr = math.radians(lean_deg + head_tilt_deg)
    head_3d = (sh_3d[0] + math.sin(hr) * 56.0, sh_3d[1] - math.cos(hr) * 56.0, 0.0)

    def limb_3d(start: Tuple[float, float, float], ang: Tuple[float, float, float], l1: float, l2: float, side_x: float):
        a1 = math.radians(ang[0] + lean_deg)
        az = math.radians(ang[2])
        mid = (
            start[0] + side_x + math.sin(a1) * l1 * math.cos(az),
            start[1] + math.cos(a1) * l1,
            start[2] + math.sin(az) * l1,
        )
        a2 = math.radians(ang[0] + ang[1] + lean_deg)
        end = (
            mid[0] + math.sin(a2) * l2 * math.cos(az),
            mid[1] + math.cos(a2) * l2,
            mid[2] + math.sin(az) * l2 * 1.1,
        )
        return mid, end

    l_knee_3, l_foot_3 = limb_3d(hip_3d, l_leg, 62.0, 60.0, -10.0)
    r_knee_3, r_foot_3 = limb_3d(hip_3d, r_leg, 62.0, 60.0, 10.0)
    l_elb_3, l_hand_3 = limb_3d(sh_3d, l_arm, 54.0, 50.0, -14.0)
    r_elb_3, r_hand_3 = limb_3d(sh_3d, r_arm, 54.0, 50.0, 14.0)

    # Rotate whole character in 3D space around Y-axis, then project to 2D
    def to_2d(p):
        return _proj(_rot_y(p, yaw_deg), hip_xy, scale)

    p_hip = to_2d(hip_3d)
    p_sh = to_2d(sh_3d)
    p_head = to_2d(head_3d)
    p_lk, p_lf = to_2d(l_knee_3), to_2d(l_foot_3)
    p_rk, p_rf = to_2d(r_knee_3), to_2d(r_foot_3)
    p_le, p_lh = to_2d(l_elb_3), to_2d(l_hand_3)
    p_re, p_rh = to_2d(r_elb_3), to_2d(r_hand_3)

    # Depth-sort segments back-to-front using midpoint Z
    segments: List[Tuple[float, str, Tuple]] = [
        ((p_hip[2] + p_lk[2]) * 0.5, "leg", (p_hip[:2], p_lk[:2], p_lf[:2], -1.0 if yaw_deg < 0 else 1.0)),
        ((p_hip[2] + p_rk[2]) * 0.5, "leg", (p_hip[:2], p_rk[:2], p_rf[:2], 1.0 if yaw_deg >= 0 else -1.0)),
        ((p_sh[2] + p_le[2]) * 0.5, "arm", (p_sh[:2], p_le[:2], p_lh[:2])),
        ((p_sh[2] + p_re[2]) * 0.5, "arm", (p_sh[:2], p_re[:2], p_rh[:2])),
        ((p_hip[2] + p_sh[2]) * 0.5, "torso", (p_hip[:2], p_sh[:2])),
    ]
    segments.sort(key=lambda item: item[0])

    for _, kind, data in segments:
        if kind == "leg":
            a, b, c, facing = data
            _draw_3d_capsule(img, a, b, limb_r, pal)
            _draw_3d_capsule(img, b, c, limb_r, pal)
            _draw_3d_foot(img, c, facing, limb_r, pal)
        elif kind == "arm":
            a, b, c = data
            _draw_3d_capsule(img, a, b, max(5, limb_r - 2), pal)
            _draw_3d_capsule(img, b, c, max(5, limb_r - 2), pal)
            # Spherical cartoon hand
            hx, hy = int(round(c[0])), int(round(c[1]))
            hr_pix = max(6, limb_r)
            cv2.circle(img, (hx, hy), hr_pix + 4, BGR_INK, -1, cv2.LINE_AA)
            cv2.circle(img, (hx, hy), hr_pix, pal[0], -1, cv2.LINE_AA)
            cv2.circle(img, (hx - 2, hy - 2), max(2, hr_pix // 2), pal[2], -1, cv2.LINE_AA)
        elif kind == "torso":
            a, b = data
            _draw_3d_capsule(img, a, b, torso_r, pal)

    # Head drawn above shoulders with face & floating emotion prop
    _draw_3d_head_and_face(img, p_head[:2], head_r, emotion, pal, look_x=look_x, look_y=look_y, t_sec=t_sec)
    _draw_emotion_props(img, p_head[:2], head_r, emotion, t_sec)

    return {
        "head": p_head[:2],
        "shoulder": p_sh[:2],
        "l_hand": p_lh[:2],
        "r_hand": p_rh[:2],
        "hip": p_hip[:2],
    }
