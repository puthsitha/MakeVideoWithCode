"""
Prepare 7 Oversampled (1296x2304, 9:16) Zero-Text Editorial Keyframes
Blends generated 9:16 scenes with cleaned, re-graded, and composited reference media
and procedural 2.5D chiaroscuro/dawn artwork aligned to Nocturnal Amber & Slate LUT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1296, 2304  # 1.2x oversampled 1080x1920 for lossless 2.5D Ken Burns pan/zoom

ARTIFACT_DIR = Path("/Users/htb/.gemini/antigravity/brain/68c527a2-5972-4930-add9-2f9701374137")
REF_DIR = ARTIFACT_DIR / ".user_uploaded"
OUT_DIR = Path("output/keyframes")

OBSIDIAN = (17, 19, 24)
PARCHMENT = (237, 227, 210)
AMBER = (217, 130, 43)
SLATE = (74, 107, 130)
CRIMSON = (200, 75, 49)


def add_paper_grain(arr: np.ndarray, strength: float = 5.5, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, strength, size=(arr.shape[0], arr.shape[1], 1)).astype(np.float32)
    return np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_radial_vignette(img: Image.Image, edge_color: Tuple[int, int, int] = OBSIDIAN, strength: float = 0.78) -> Image.Image:
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / (w * 0.55)
    ny = (yy - h * 0.48) / (h * 0.52)
    r = np.clip(np.sqrt(nx * nx + ny * ny), 0.0, 1.35)
    alpha = np.clip((r - 0.50) / 0.78, 0.0, 1.0) ** 1.6 * strength

    base = np.asarray(img, dtype=np.float32)
    bg = np.full_like(base, edge_color, dtype=np.float32)
    blended = base * (1.0 - alpha[..., None]) + bg * alpha[..., None]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def fit_portrait_canvas(src: Image.Image, target_w: int = W, target_h: int = H) -> Image.Image:
    sw, sh = src.size
    scale = max(target_w / sw, target_h / sh)
    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - target_w) // 2
    top = (nh - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def create_gradient_canvas(top_col: Tuple[int, int, int], mid_col: Tuple[int, int, int], bot_col: Tuple[int, int, int]) -> Image.Image:
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
    t1 = np.clip(yy / 0.48, 0.0, 1.0)
    t2 = np.clip((yy - 0.48) / 0.52, 0.0, 1.0)
    top_a = np.array(top_col, dtype=np.float32)
    mid_a = np.array(mid_col, dtype=np.float32)
    bot_a = np.array(bot_col, dtype=np.float32)
    grad = np.where(yy < 0.48, top_a * (1 - t1) + mid_a * t1, mid_a * (1 - t2) + bot_a * t2)
    canvas = np.repeat(grad, W, axis=1).astype(np.uint8)
    return Image.fromarray(add_paper_grain(canvas, strength=4.0))


def soft_feathered_mask(size: Tuple[int, int], border: int = 72, blur: int = 28) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 255)
    md = ImageDraw.Draw(mask)
    for i in range(border):
        val = int(255 * ((i / border) ** 1.4))
        md.rectangle([i, i, w - 1 - i, h - 1 - i], outline=val)
    return mask.filter(ImageFilter.GaussianBlur(blur))


def build_01_intro() -> Image.Image:
    return fit_portrait_canvas(Image.open(ARTIFACT_DIR / "01_intro_1794980628987.jpg").convert("RGB"))


def build_02_verse_1a() -> Image.Image:
    """Clean 'DOCTOR'/'LAWYER' text from Ref 1 while keeping caduceus & scales icons crisp."""
    raw = cv2.imread(str(REF_DIR / "media_1794976632831_e1c9879b.jpg"))

    # Top sign interior ("DOCTOR": x=216..408, y=38..108)
    patch_top = raw[36:50, 225:395].copy()
    raw[38:108, 216:408] = cv2.resize(patch_top, (408 - 216, 108 - 38), interpolation=cv2.INTER_LINEAR)

    # Bottom sign interior ("LAWYER": x=258..404, y=162..224) while preserving arrow tip outline
    patch_bot = raw[158:170, 260:370].copy()
    raw[162:224, 258:402] = cv2.resize(patch_bot, (402 - 258, 224 - 162), interpolation=cv2.INTER_LINEAR)
    # Redraw crisp pointed right-arrow border from (384,158)->(414,193)->(384,228)
    cv2.line(raw, (384, 158), (414, 193), (68, 88, 108), 2, cv2.LINE_AA)
    cv2.line(raw, (414, 193), (384, 228), (68, 88, 108), 2, cv2.LINE_AA)

    im = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    stage = create_gradient_canvas((26, 24, 21), (178, 166, 142), (20, 22, 27))
    target_w = int(W * 0.96)
    target_h = int(target_w * im.height / im.width)
    card = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    mask = soft_feathered_mask(card.size, border=96, blur=36)
    stage.paste(card, ((W - target_w) // 2, int(H * 0.51 - target_h * 0.5)), mask)
    return apply_radial_vignette(stage, OBSIDIAN, strength=0.80)


def build_03_verse_1b() -> Image.Image:
    return fit_portrait_canvas(Image.open(ARTIFACT_DIR / "03_verse_1b_v2_1794980599224.jpg").convert("RGB"))


def build_04_hook_midpoint() -> Image.Image:
    """Frame satirical skyscraper-cradle Ref 4 onto bold crimson-obsidian 9:16 stage."""
    im = Image.open(REF_DIR / "media_1794976655715_a1fbcee5.jpg").convert("RGB")
    stage = create_gradient_canvas((32, 14, 15), (176, 36, 28), OBSIDIAN)
    target_w = int(W * 0.96)
    target_h = int(target_w * im.height / im.width)
    card = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    mask = soft_feathered_mask(card.size, border=68, blur=26)
    stage.paste(card, ((W - target_w) // 2, int(H * 0.47 - target_h * 0.5)), mask)
    return apply_radial_vignette(stage, OBSIDIAN, strength=0.72)


def build_05_verse_2a() -> Image.Image:
    """Clean text-free desk youth from Ref 2 + glowing amber lamp cone & rain flashback window."""
    raw = cv2.imread(str(REF_DIR / "media_1794976641512_92f2dddc.jpg"))
    # Crop desk & youth region starting at y=354 (x=0..162, y=354..494)
    crop = raw[354:494, 0:162].copy()

    # Build soft alpha mask of ONLY the lamp, boy, and desk (excluding speech clouds & mug text)
    keep_mask = np.ones((crop.shape[0], crop.shape[1]), dtype=np.float32) * 255.0
    keep_mask[0:62, 0:92] = 0.0     # top-left cloud
    keep_mask[56:82, 42:74] = 0.0    # question mark
    keep_mask[0:92, 130:162] = 0.0   # right cloud
    keep_mask[0:18, 92:130] = 0.0    # top "?" dot
    keep_mask[102:140, 130:162] = 0.0  # mug label
    keep_mask = cv2.GaussianBlur(keep_mask, (21, 21), 8.0)

    desk_im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    desk_alpha = Image.fromarray(np.clip(keep_mask, 0, 255).astype(np.uint8), mode="L")
    stage = create_gradient_canvas(OBSIDIAN, (22, 27, 36), (13, 14, 18))

    # Draw crisp glowing rain window in upper-right Rule of Thirds
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    wx0, wy0, wx1, wy1 = int(W * 0.46), int(H * 0.16), int(W * 0.86), int(H * 0.40)
    od.rounded_rectangle([wx0, wy0, wx1, wy1], radius=24, fill=(48, 72, 92, 135), outline=(217, 130, 43, 185), width=4)
    wmid_x, wmid_y = (wx0 + wx1) // 2, (wy0 + wy1) // 2
    od.line([(wmid_x, wy0), (wmid_x, wy1)], fill=(20, 24, 32, 190), width=6)
    od.line([(wx0, wmid_y), (wx1, wmid_y)], fill=(20, 24, 32, 190), width=6)
    by = wy1 - 14
    od.pieslice([wmid_x - 130, by - 255, wmid_x + 130, by - 65], 180, 360, fill=(14, 17, 22, 235))
    od.line([(wmid_x, by - 160), (wmid_x, by - 45)], fill=(14, 17, 22, 235), width=5)
    od.polygon([(wmid_x - 82, by), (wmid_x - 55, by - 148), (wmid_x - 16, by - 148), (wmid_x - 8, by)], fill=(14, 17, 22, 240))
    od.ellipse([wmid_x - 56, by - 188, wmid_x - 14, by - 146], fill=(14, 17, 22, 240))
    od.polygon([(wmid_x + 8, by), (wmid_x + 18, by - 136), (wmid_x + 56, by - 136), (wmid_x + 80, by)], fill=(14, 17, 22, 240))
    od.ellipse([wmid_x + 16, by - 174, wmid_x + 56, by - 136], fill=(14, 17, 22, 240))
    for rx in range(wx0 + 24, wx1 - 18, 36):
        for ry in range(wy0 + 22, wy1 - 45, 58):
            od.line([(rx, ry), (rx - 14, ry + 34)], fill=(237, 227, 210, 95), width=2)

    overlay = overlay.filter(ImageFilter.GaussianBlur(2))
    stage = Image.alpha_composite(stage.convert("RGBA"), overlay).convert("RGB")

    tw = int(W * 0.86)
    th = int(tw * desk_im.height / desk_im.width)
    desk_scaled = desk_im.resize((tw, th), Image.Resampling.LANCZOS)
    alpha_scaled = desk_alpha.resize((tw, th), Image.Resampling.LANCZOS)
    # Combine custom cloud-erasure alpha with outer soft feather
    outer_mask = soft_feathered_mask(desk_scaled.size, border=72, blur=24)
    combined_alpha = Image.fromarray(
        np.minimum(np.asarray(alpha_scaled), np.asarray(outer_mask)).astype(np.uint8),
        mode="L",
    )
    stage.paste(desk_scaled, ((W - tw) // 2, int(H * 0.63 - th * 0.42)), combined_alpha)
    return apply_radial_vignette(stage, OBSIDIAN, strength=0.76)


def build_06_verse_2b() -> Image.Image:
    """Procedural 2.5D graphic-novel illustration: dawn blueprint arch bridge across canyon."""
    stage = create_gradient_canvas((22, 28, 40), (146, 94, 46), (17, 19, 24))
    rgba = stage.convert("RGBA")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    horizon_y = int(H * 0.382)  # Golden Ratio horizon
    sun_cx, sun_cy = int(W * 0.68), horizon_y - 24

    for r, a in [(480, 26), (330, 46), (205, 75), (105, 135)]:
        d.ellipse([sun_cx - r, sun_cy - r, sun_cx + r, sun_cy + r], fill=(237, 227, 210, a))
    d.ellipse([sun_cx - 64, sun_cy - 64, sun_cx + 64, sun_cy + 64], fill=(255, 242, 214, 215))

    # Foreground left & right canyon cliffs at lower Golden Ratio y=0.618
    bridge_y = int(H * 0.618)

    # Distant slate canyon ridges anchored cleanly down to bridge horizon
    d.polygon([(0, horizon_y + 20), (340, horizon_y - 58), (740, bridge_y), (0, bridge_y)], fill=(45, 62, 78, 225))
    d.polygon([(460, bridge_y), (960, horizon_y - 68), (W, horizon_y + 18), (W, bridge_y)], fill=(38, 52, 66, 225))
    d.polygon([(0, bridge_y), (int(W * 0.24), bridge_y), (int(W * 0.31), H), (0, H)], fill=(17, 19, 24, 255))
    d.polygon([(int(W * 0.76), bridge_y), (W, bridge_y), (W, H), (int(W * 0.69), H)], fill=(17, 19, 24, 255))

    bx0, bx1 = int(W * 0.20), int(W * 0.80)
    # Drafting blueprint grid
    for gx in range(bx0, bx1 + 1, 64):
        d.line([(gx, bridge_y - 160), (gx, bridge_y + 240)], fill=(217, 130, 43, 36), width=2)
    for gy in range(bridge_y - 160, bridge_y + 240, 64):
        d.line([(bx0 - 30, gy), (bx1 + 30, gy)], fill=(217, 130, 43, 36), width=2)

    # Bridge deck & true supporting arch springing upward from canyon walls to deck center
    d.line([(bx0, bridge_y), (bx1, bridge_y)], fill=(237, 227, 210, 240), width=11)
    d.line([(bx0, bridge_y + 13), (bx1, bridge_y + 13)], fill=(217, 130, 43, 215), width=5)
    # Upper-half ellipse bounding [bx0, bridge_y + 14 .. bx1, bridge_y + 270] curves gracefully under deck
    d.arc([bx0, bridge_y + 16, bx1, bridge_y + 280], 180, 360, fill=(217, 130, 43, 245), width=9)

    # Vertical blueprint struts connecting deck to arch
    cx_mid = (bx0 + bx1) * 0.5
    half_span = (bx1 - bx0) * 0.5
    ry = 132.0
    cy_arch = bridge_y + 16 + ry
    for sx in range(bx0 + 48, bx1 - 36, 72):
        dx = (sx - cx_mid) / half_span
        if abs(dx) < 0.98:
            ay = cy_arch - ry * float(np.sqrt(1.0 - dx * dx))
            if ay > bridge_y + 14:
                d.line([(sx, bridge_y + 12), (sx, int(ay))], fill=(237, 227, 210, 175), width=3)

    # Protagonist silhouette walking confidently across bridge with rolled blueprint
    px, py = int(W * 0.56), bridge_y - 6
    d.polygon([(px - 22, py), (px - 14, py - 110), (px + 14, py - 110), (px + 24, py)], fill=(17, 19, 24, 245))
    d.ellipse([px - 16, py - 146, px + 16, py - 112], fill=(17, 19, 24, 245))
    d.line([(px + 10, py - 85), (px + 48, py - 115)], fill=(237, 227, 210, 240), width=7)

    # Calm parents on left bank
    for ox, h_scale in [(int(W * 0.11), 1.0), (int(W * 0.17), 0.94)]:
        cy = bridge_y - 4
        hh = int(125 * h_scale)
        d.polygon([(ox - 20, cy), (ox - 14, cy - hh), (ox + 14, cy - hh), (ox + 20, cy)], fill=(32, 44, 56, 250))
        d.ellipse([ox - 15, cy - hh - 32, ox + 15, cy - hh], fill=(32, 44, 56, 250))

    comp = Image.alpha_composite(rgba, layer).convert("RGB")
    arr = add_paper_grain(np.asarray(comp), strength=5.0, seed=19)
    return apply_radial_vignette(Image.fromarray(arr), OBSIDIAN, strength=0.66)


def build_07_outro() -> Image.Image:
    """Clean text-free solitary crossroads figure from Ref 2 on golden-hour 9:16 finale stage."""
    raw = cv2.imread(str(REF_DIR / "media_1794976641512_92f2dddc.jpg"))
    crop = raw[336:418, 210:370].copy()
    cross_im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    stage = create_gradient_canvas(OBSIDIAN, (132, 82, 34), OBSIDIAN)
    rgba = stage.convert("RGBA")
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = W // 2, int(H * 0.50)
    for r, a in [(580, 30), (420, 58), (260, 100)]:
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(217, 130, 43, a))
    glow = glow.filter(ImageFilter.GaussianBlur(28))
    stage = Image.alpha_composite(rgba, glow).convert("RGB")

    tw = int(W * 0.94)
    th = int(tw * cross_im.height / cross_im.width)
    scaled = cross_im.resize((tw, th), Image.Resampling.LANCZOS)
    mask = soft_feathered_mask(scaled.size, border=96, blur=36)
    stage.paste(scaled, ((W - tw) // 2, int(H * 0.51 - th * 0.5)), mask)
    return apply_radial_vignette(stage, OBSIDIAN, strength=0.80)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "01_intro.png": build_01_intro,
        "02_verse_1a.png": build_02_verse_1a,
        "03_verse_1b.png": build_03_verse_1b,
        "04_hook_midpoint.png": build_04_hook_midpoint,
        "05_verse_2a.png": build_05_verse_2a,
        "06_verse_2b.png": build_06_verse_2b,
        "07_outro.png": build_07_outro,
    }
    for fname, fn in builders.items():
        out_path = OUT_DIR / fname
        fn().save(out_path, format="PNG")
        print(f"[KEYFRAME] Saved {out_path} ({W}x{H})")


if __name__ == "__main__":
    main()
