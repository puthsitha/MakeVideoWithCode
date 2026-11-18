"""
Dynamic Keyframe-Animated Giphy Sticker Loader & Compositor
Supports:
  - Multi-keyframe interpolation (x, y, target_width, rotation_deg, alpha) with cosine easing
  - Automatic loading of all *.gif stickers from output/stickers/
  - Clean RGBA -> BGRA alpha compositing onto 1080x1920 frames
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageSequence

STICKER_DIR = Path("output/stickers")

# Keyframe tuple: (t_local, cx, cy, target_w, angle_deg, alpha)
StickerKF = Tuple[float, float, float, float, float, float]


def _smooth_interp(a: float, b: float, u: float) -> float:
    e = 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, u)))
    return a + (b - a) * e


class StickerBank:
    def __init__(self) -> None:
        self._frames: Dict[str, List[np.ndarray]] = {}
        if not STICKER_DIR.exists():
            return
        for path in sorted(STICKER_DIR.glob("*.gif")):
            im = Image.open(path)
            seq: List[np.ndarray] = []
            for fr in ImageSequence.Iterator(im):
                rgba = np.asarray(fr.convert("RGBA"), dtype=np.uint8)
                seq.append(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            if seq:
                self._frames[path.stem] = seq

    def has(self, name: str) -> bool:
        return name in self._frames

    def overlay_keyframed(
        self,
        canvas: np.ndarray,
        name: str,
        keyframes: Sequence[StickerKF],
        t_sec: float,
        local_t: float,
        fps_gif: float = 11.0,
        bob_amp: float = 6.0,
    ) -> None:
        """Animate a Giphy sticker along multi-keyframe (t, x, y, width, angle, alpha) paths."""
        seq = self._frames.get(name)
        if not seq or not keyframes:
            return
        if local_t < keyframes[0][0] or local_t > keyframes[-1][0]:
            return

        # Find surrounding keyframes
        kf0 = keyframes[0]
        kf1 = keyframes[-1]
        for i in range(len(keyframes) - 1):
            if keyframes[i][0] <= local_t <= keyframes[i + 1][0]:
                kf0, kf1 = keyframes[i], keyframes[i + 1]
                break

        dt = max(1e-5, kf1[0] - kf0[0])
        u = (local_t - kf0[0]) / dt
        cx = int(round(_smooth_interp(kf0[1], kf1[1], u)))
        cy = int(round(_smooth_interp(kf0[2], kf1[2], u) + math.sin(t_sec * 3.1) * bob_amp))
        target_w = max(8, int(round(_smooth_interp(kf0[3], kf1[3], u))))
        angle = _smooth_interp(kf0[4], kf1[4], u)
        alpha_mult = float(np.clip(_smooth_interp(kf0[5], kf1[5], u), 0.0, 1.0))
        if alpha_mult <= 0.02:
            return

        idx = int(max(0.0, local_t) * fps_gif) % len(seq)
        bgra = seq[idx]
        sh, sw = bgra.shape[:2]
        nh = max(8, int(round(target_w * (sh / sw))))
        resized = cv2.resize(bgra, (target_w, nh), interpolation=cv2.INTER_AREA)

        if abs(angle) > 0.4:
            diag = int(math.ceil(math.hypot(target_w, nh))) + 4
            pad = np.zeros((diag, diag, 4), dtype=np.uint8)
            ox, oy = (diag - target_w) // 2, (diag - nh) // 2
            pad[oy : oy + nh, ox : ox + target_w] = resized
            mat = cv2.getRotationMatrix2D((diag * 0.5, diag * 0.5), angle, 1.0)
            resized = cv2.warpAffine(
                pad, mat, (diag, diag),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
            )
            nh, target_w = resized.shape[:2]

        H, W = canvas.shape[:2]
        x0, y0 = cx - target_w // 2, cy - nh // 2
        x1, y1 = min(W, x0 + target_w), min(H, y0 + nh)
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        cw, ch = x1 - x0, y1 - y0
        if cw <= 0 or ch <= 0:
            return

        sub = resized[sy0 : sy0 + ch, sx0 : sx0 + cw]
        a = (sub[:, :, 3:4].astype(np.float32) / 255.0) * alpha_mult
        fg = sub[:, :, :3].astype(np.float32)
        bg = canvas[y0:y1, x0:x1].astype(np.float32)
        canvas[y0:y1, x0:x1] = np.clip(fg * a + bg * (1.0 - a), 0, 255).astype(np.uint8)
