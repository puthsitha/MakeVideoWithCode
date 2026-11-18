"""
2D Stick-Man Character Animation Engine for 1080x1920 @ 30fps
Draws articulated, expressive 2D stick figures acting out each narrative beat
with dark outline halo + parchment/amber inner strokes for crisp readability.
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

# BGR colors matching Nocturnal Amber & Slate palette
BGR_OUTLINE = (17, 19, 24)
BGR_PARCHMENT = (210, 227, 237)
BGR_AMBER = (43, 130, 217)
BGR_SLATE = (130, 107, 74)
BGR_CRIMSON = (49, 75, 200)


def _limb(
    img: np.ndarray,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    a = (int(round(p0[0])), int(round(p0[1])))
    b = (int(round(p1[0])), int(round(p1[1])))
    cv2.line(img, a, b, BGR_OUTLINE, thickness + 6, cv2.LINE_AA)
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def _joint_dot(
    img: np.ndarray,
    pt: Tuple[float, float],
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    p = (int(round(pt[0])), int(round(pt[1])))
    cv2.circle(img, p, radius + 2, BGR_OUTLINE, -1, cv2.LINE_AA)
    cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)


def draw_stickman(
    img: np.ndarray,
    hip: Tuple[float, float],
    scale: float = 1.0,
    color: Tuple[int, int, int] = BGR_PARCHMENT,
    lean_deg: float = 0.0,
    head_tilt_deg: float = 0.0,
    l_arm_angles: Tuple[float, float] = (25.0, 15.0),   # (upper_deg from down, forearm_deg from upper)
    r_arm_angles: Tuple[float, float] = (-25.0, -15.0),
    l_leg_angles: Tuple[float, float] = (14.0, -6.0),
    r_leg_angles: Tuple[float, float] = (-14.0, 6.0),
    mood: str = "neutral",
) -> dict:
    """Render an articulated 2D stick figure and return key joint positions."""
    hx, hy = hip
    t_len = 96.0 * scale
    u_arm = 48.0 * scale
    f_arm = 44.0 * scale
    thigh = 56.0 * scale
    shin = 54.0 * scale
    head_r = max(12, int(round(25.0 * scale)))
    thick = max(4, int(round(8.5 * scale)))

    rad_lean = math.radians(lean_deg)
    sh_x = hx + math.sin(rad_lean) * t_len
    sh_y = hy - math.cos(rad_lean) * t_len
    neck = (sh_x, sh_y - 12.0 * scale)

    rad_head = math.radians(lean_deg + head_tilt_deg)
    head_c = (
        neck[0] + math.sin(rad_head) * (head_r + 6.0 * scale),
        neck[1] - math.cos(rad_head) * (head_r + 6.0 * scale),
    )

    def chain(start: Tuple[float, float], a1: float, a2: float, l1: float, l2: float):
        r1 = math.radians(a1 + lean_deg)
        mid = (start[0] + math.sin(r1) * l1, start[1] + math.cos(r1) * l1)
        r2 = math.radians(a1 + a2 + lean_deg)
        end = (mid[0] + math.sin(r2) * l2, mid[1] + math.cos(r2) * l2)
        return mid, end

    l_knee, l_foot = chain(hip, l_leg_angles[0], l_leg_angles[1], thigh, shin)
    r_knee, r_foot = chain(hip, r_leg_angles[0], r_leg_angles[1], thigh, shin)
    l_elbow, l_hand = chain((sh_x, sh_y), l_arm_angles[0], l_arm_angles[1], u_arm, f_arm)
    r_elbow, r_hand = chain((sh_x, sh_y), r_arm_angles[0], r_arm_angles[1], u_arm, f_arm)

    # Legs & Torso
    _limb(img, hip, l_knee, color, thick)
    _limb(img, l_knee, l_foot, color, thick)
    _limb(img, hip, r_knee, color, thick)
    _limb(img, r_knee, r_foot, color, thick)
    _limb(img, hip, (sh_x, sh_y), color, thick + 1)
    _limb(img, (sh_x, sh_y), neck, color, max(3, thick - 1))

    # Arms
    _limb(img, (sh_x, sh_y), l_elbow, color, max(3, thick - 1))
    _limb(img, l_elbow, l_hand, color, max(3, thick - 1))
    _limb(img, (sh_x, sh_y), r_elbow, color, max(3, thick - 1))
    _limb(img, r_elbow, r_hand, color, max(3, thick - 1))

    # Hands & Feet caps
    for pt in (l_hand, r_hand, l_foot, r_foot):
        _joint_dot(img, pt, max(3, int(round(5 * scale))), color)

    # Head circle (filled dark with crisp colored border)
    hc = (int(round(head_c[0])), int(round(head_c[1])))
    cv2.circle(img, hc, head_r + 5, BGR_OUTLINE, -1, cv2.LINE_AA)
    cv2.circle(img, hc, head_r, BGR_OUTLINE, -1, cv2.LINE_AA)
    cv2.circle(img, hc, head_r, color, max(3, int(round(5 * scale))), cv2.LINE_AA)

    # Simple expressive eyes / brow
    eye_dx = int(round(8 * scale))
    eye_y = hc[1] - int(round(2 * scale))
    eye_r = max(2, int(round(2.8 * scale)))
    cv2.circle(img, (hc[0] - eye_dx, eye_y), eye_r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (hc[0] + eye_dx, eye_y), eye_r, color, -1, cv2.LINE_AA)

    if mood == "sad":
        cv2.ellipse(img, (hc[0], hc[1] + int(9 * scale)), (int(7 * scale), int(4 * scale)), 0, 195, 345, color, 2, cv2.LINE_AA)
    elif mood == "happy":
        cv2.ellipse(img, (hc[0], hc[1] + int(6 * scale)), (int(8 * scale), int(5 * scale)), 0, 15, 165, color, 2, cv2.LINE_AA)
    elif mood == "surprised":
        cv2.circle(img, (hc[0], hc[1] + int(9 * scale)), max(3, int(4 * scale)), color, 2, cv2.LINE_AA)

    return {"head": head_c, "shoulder": (sh_x, sh_y), "l_hand": l_hand, "r_hand": r_hand, "hip": hip}


def animate_scene_stickmen(frame: np.ndarray, scene_id: str, t_sec: float, scene_prog: float) -> np.ndarray:
    """Composite animated 2D stick-man characters tailored to each scene's narrative action."""
    if scene_id == "01_intro":
        # Protagonist stick-man at lower-center foreground reaching subtly up toward cage bars
        wave = math.sin(t_sec * 2.2) * 8.0
        draw_stickman(
            frame,
            hip=(540, 1545),
            scale=1.22,
            color=BGR_PARCHMENT,
            lean_deg=math.sin(t_sec * 1.4) * 2.0,
            head_tilt_deg=-6.0,
            l_arm_angles=(-125 + wave, 35),
            r_arm_angles=(125 - wave, -35),
            l_leg_angles=(12, -5),
            r_leg_angles=(-12, 5),
            mood="sad",
        )

    elif scene_id == "02_verse_1a":
        # Thoughtful stick-man centered below crossroads pondering left vs right
        sway = math.sin(t_sec * 1.6) * 4.0
        look = math.sin(t_sec * 1.1) * 9.0
        draw_stickman(
            frame,
            hip=(540, 1525),
            scale=1.18,
            color=BGR_AMBER,
            lean_deg=sway * 0.4,
            head_tilt_deg=look,
            l_arm_angles=(-16, 118),     # hand touching chin thoughtfully
            r_arm_angles=(32, -45),
            l_leg_angles=(14, -4),
            r_leg_angles=(-14, 4),
            mood="sad",
        )

    elif scene_id == "03_verse_1b":
        # Two parent stick-figures left/right tugging glowing amber strings to central child stick-man
        tug = math.sin(t_sec * 1.8)
        child_joints = draw_stickman(
            frame,
            hip=(540 + tug * 18.0, 1540),
            scale=1.05,
            color=BGR_PARCHMENT,
            lean_deg=tug * 6.5,
            l_arm_angles=(-55 - tug * 12, -20),
            r_arm_angles=(55 - tug * 12, 20),
            mood="sad",
        )
        p_left = draw_stickman(
            frame,
            hip=(225, 1465),
            scale=1.15,
            color=BGR_SLATE,
            lean_deg=-5.0,
            l_arm_angles=(-20, 10),
            r_arm_angles=(78 + tug * 12, 22),
            mood="neutral",
        )
        p_right = draw_stickman(
            frame,
            hip=(855, 1465),
            scale=1.15,
            color=BGR_SLATE,
            lean_deg=5.0,
            l_arm_angles=(-78 + tug * 12, -22),
            r_arm_angles=(20, -10),
            mood="neutral",
        )
        # Glowing amber tether strings between parent hands & child hands
        for a, b in [(p_left["r_hand"], child_joints["l_hand"]), (p_right["l_hand"], child_joints["r_hand"])]:
            pa = (int(a[0]), int(a[1]))
            pb = (int(b[0]), int(b[1]))
            cv2.line(frame, pa, pb, BGR_OUTLINE, 6, cv2.LINE_AA)
            cv2.line(frame, pa, pb, BGR_AMBER, 3, cv2.LINE_AA)

    elif scene_id == "04_hook_midpoint":
        # Two satirical parent stick-figures below: left proudly holding tiny skyscraper, right rocking crying baby
        bounce = abs(math.sin(t_sec * 4.2)) * 10.0
        left_p = draw_stickman(
            frame,
            hip=(335, 1495 - bounce),
            scale=1.12,
            color=BGR_AMBER,
            lean_deg=-4.0,
            l_arm_angles=(-65, -45),
            r_arm_angles=(70, 45),
            mood="happy",
        )
        # Tiny 2D skyscraper prop cradled cleanly in front of chest (below head)
        bx = int((left_p["l_hand"][0] + left_p["r_hand"][0]) * 0.5)
        by = int((left_p["l_hand"][1] + left_p["r_hand"][1]) * 0.5) + 46
        cv2.rectangle(frame, (bx - 24, by - 58), (bx + 24, by + 14), BGR_OUTLINE, -1)
        cv2.rectangle(frame, (bx - 22, by - 56), (bx + 22, by + 12), BGR_SLATE, -1)
        cv2.rectangle(frame, (bx - 22, by - 56), (bx + 22, by + 12), BGR_PARCHMENT, 2)
        for wy in range(by - 48, by + 4, 15):
            for wx in (bx - 12, bx + 4):
                cv2.rectangle(frame, (wx, wy), (wx + 8, wy + 8), BGR_AMBER, -1)

        right_p = draw_stickman(
            frame,
            hip=(745, 1495 - abs(math.cos(t_sec * 4.2)) * 8.0),
            scale=1.12,
            color=BGR_PARCHMENT,
            lean_deg=4.0,
            l_arm_angles=(-60, -40),
            r_arm_angles=(60, 40),
            mood="surprised",
        )
        cx = int((right_p["l_hand"][0] + right_p["r_hand"][0]) * 0.5)
        cy = int((right_p["l_hand"][1] + right_p["r_hand"][1]) * 0.5)
        cv2.ellipse(frame, (cx, cy), (28, 16), -12, 0, 360, BGR_CRIMSON, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx - 16, cy - 8), 11, BGR_PARCHMENT, -1, cv2.LINE_AA)

    elif scene_id == "05_verse_2a":
        # Stick-man standing/leaning quietly beside study desk looking up toward rain window
        tilt = -12.0 + math.sin(t_sec * 1.2) * 3.0
        draw_stickman(
            frame,
            hip=(235, 1520),
            scale=1.12,
            color=BGR_PARCHMENT,
            lean_deg=4.0,
            head_tilt_deg=tilt,
            l_arm_angles=(-18, 95),
            r_arm_angles=(28, -22),
            mood="neutral",
        )

    elif scene_id == "06_verse_2b":
        # Walking stick-man crossing the glowing golden bridge from x=270 to x=800 with rolled blueprint!
        walk_x = 270.0 + scene_prog * 530.0
        phase = t_sec * 5.2
        stride = math.sin(phase) * 28.0
        knee_bend_l = max(0.0, -math.cos(phase) * 24.0)
        knee_bend_r = max(0.0, math.cos(phase) * 24.0)
        bob = abs(math.sin(phase)) * 6.0

        hero = draw_stickman(
            frame,
            hip=(walk_x, 1085 - bob),
            scale=0.92,
            color=BGR_PARCHMENT,
            lean_deg=7.0,
            l_arm_angles=(-stride * 0.8, -18),
            r_arm_angles=(55 + stride * 0.35, -35),
            l_leg_angles=(stride, knee_bend_l),
            r_leg_angles=(-stride, knee_bend_r),
            mood="happy",
        )
        # Rolled glowing blueprint held in right hand
        rh = hero["r_hand"]
        cv2.line(
            frame,
            (int(rh[0] - 8), int(rh[1] + 6)),
            (int(rh[0] + 38), int(rh[1] - 26)),
            BGR_OUTLINE,
            10,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (int(rh[0] - 8), int(rh[1] + 6)),
            (int(rh[0] + 38), int(rh[1] - 26)),
            BGR_AMBER,
            6,
            cv2.LINE_AA,
        )
        # Parents on left bank relaxing posture
        draw_stickman(frame, hip=(120, 1095), scale=0.88, color=BGR_SLATE, l_arm_angles=(-12, 8), r_arm_angles=(16, -8), mood="happy")
        draw_stickman(frame, hip=(195, 1100), scale=0.84, color=BGR_SLATE, l_arm_angles=(-16, 8), r_arm_angles=(12, -8), mood="happy")

    elif scene_id == "07_outro":
        # Hero stick-man walking forward at center raising arm toward golden horizon
        lift = min(1.0, scene_prog * 2.2)
        draw_stickman(
            frame,
            hip=(540, 1510 - scene_prog * 35.0),
            scale=1.22,
            color=BGR_AMBER,
            lean_deg=-2.0,
            l_arm_angles=(-24, -10),
            r_arm_angles=(25 + lift * 115.0, -22),
            l_leg_angles=(14, -4),
            r_leg_angles=(-14, 4),
            mood="happy",
        )

    return frame
