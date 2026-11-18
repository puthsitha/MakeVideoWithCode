"""
Frame-by-Frame QA Review Gate (1.0 - 10.0 Scale, >= 8.0 Pass Threshold)
Validates 1080x1920 @ 30fps video for visual consistency, anti-morphing,
zero artifacting / zero baked text, mathematical grid alignment, and color LUT adherence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

PASS_THRESHOLD = 8.0
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = 30.0
TARGET_DURATION_SEC = 120.0

# Nocturnal Amber & Slate Editorial LUT palette (hex -> RGB)
TARGET_PALETTE_HEX = ["#111318", "#EDE3D2", "#D9822B", "#4A6B82", "#C84B31"]

# Mathematical focal coordinates for 1080x1920 (Rule of Thirds, Center Axis, Golden Ratio)
PHI = 1.61803398875
THIRD_X = (TARGET_WIDTH / 3.0, 2.0 * TARGET_WIDTH / 3.0, TARGET_WIDTH / 2.0)  # 360, 720, 540
THIRD_Y = (
    TARGET_HEIGHT / 3.0,
    2.0 * TARGET_HEIGHT / 3.0,
    TARGET_HEIGHT / PHI,
    TARGET_HEIGHT - (TARGET_HEIGHT / PHI),
)  # 640, 1280, 1186.6, 733.4

SCENE_RANGES: List[Tuple[str, float, float]] = [
    ("01_intro", 0.0, 6.2),
    ("02_verse_1a", 6.2, 19.2),
    ("03_verse_1b", 19.2, 46.2),
    ("04_hook_midpoint", 46.2, 61.0),
    ("05_verse_2a", 61.0, 80.5),
    ("06_verse_2b", 80.5, 102.5),
    ("07_outro", 102.5, 120.0),
]


@dataclass
class QAReport:
    score: float
    passed: bool
    specs_ok: bool
    consistency_score: float
    artifact_score: float
    alignment_score: float
    color_score: float
    failing_scenes: List[str]
    details: Dict[str, object]


def _hex_to_lab_array(hex_colors: List[str]) -> np.ndarray:
    rgb_list = []
    for hx in hex_colors:
        h = hx.lstrip("#")
        rgb_list.append([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)])
    rgb_arr = np.uint8([rgb_list])
    lab_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return lab_arr


TARGET_PALETTE_LAB = _hex_to_lab_array(TARGET_PALETTE_HEX)


def probe_video_specs(video_path: Path) -> Dict[str, object]:
    """Verify 1080x1920 resolution, 30fps, and ~120s duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(video_path),
    ]
    raw = subprocess.check_output(cmd, text=True)
    stream = json.loads(raw)["streams"][0]
    num, den = map(float, stream["r_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    width = int(stream["width"])
    height = int(stream["height"])
    duration = float(stream.get("duration", 0.0))

    specs_ok = (
        width == TARGET_WIDTH
        and height == TARGET_HEIGHT
        and abs(fps - TARGET_FPS) <= 0.1
        and abs(duration - TARGET_DURATION_SEC) <= 1.5
    )
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "duration": round(duration, 2),
        "specs_ok": specs_ok,
    }


def score_frame_alignment(gray: np.ndarray) -> float:
    """Measure saliency centroid alignment to Rule-of-Thirds / Golden Ratio & 10% safe area."""
    h, w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.hypot(gx, gy)
    total_energy = float(np.sum(energy)) + 1e-6

    ys, xs = np.mgrid[0:h, 0:w]
    cx = float(np.sum(xs * energy) / total_energy) * (TARGET_WIDTH / w)
    cy = float(np.sum(ys * energy) / total_energy) * (TARGET_HEIGHT / h)

    min_dx = min(abs(cx - tx) / TARGET_WIDTH for tx in THIRD_X)
    min_dy = min(abs(cy - ty) / TARGET_HEIGHT for ty in THIRD_Y)
    dist = float(np.hypot(min_dx, min_dy))

    # Safe margin bonus/penalty (10% edge border)
    in_safe = (0.10 * TARGET_WIDTH <= cx <= 0.90 * TARGET_WIDTH) and (
        0.10 * TARGET_HEIGHT <= cy <= 0.90 * TARGET_HEIGHT
    )
    base = max(1.0, 10.0 - dist * 22.0)
    return min(10.0, base if in_safe else max(1.0, base - 2.0))


def score_frame_color_lut(bgr: np.ndarray) -> float:
    """Measure minimum CIELAB Delta-E against Nocturnal Amber & Slate Editorial LUT."""
    small = cv2.resize(bgr, (135, 240), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    # Distance from each sampled pixel to nearest target LUT swatch
    dists = np.linalg.norm(lab[:, None, :] - TARGET_PALETTE_LAB[None, :, :], axis=2)
    mean_min_dist = float(np.mean(np.min(dists, axis=1)))
    # Typical calibrated editorial frames score <= 28 Delta-E on mean nearest swatch
    return float(np.clip(10.5 - (mean_min_dist / 8.0), 1.0, 10.0))


def score_frame_artifacts_and_text(gray: np.ndarray) -> float:
    """Score sharpness/cleanliness and penalize high-density text-like glyph bands."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = np.clip((lap_var / 120.0) * 9.0, 2.0, 10.0)

    # Bottom 22% subtitle zone check: detect unexpected crisp horizontal text-like edges
    h, _ = gray.shape
    sub_zone = gray[int(h * 0.78) : int(h * 0.93), :]
    edges = cv2.Canny(sub_zone, 120, 240)
    edge_density = float(np.mean(edges > 0))
    text_penalty = 3.0 if edge_density > 0.14 else 0.0
    return float(np.clip(sharpness - text_penalty, 1.0, 10.0))


def analyze_video(video_path: Path, sample_interval_sec: float = 1.0) -> QAReport:
    """Run frame-by-frame QA evaluation across all scenes."""
    specs = probe_video_specs(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    step_frames = max(1, int(round(fps * sample_interval_sec)))

    consistency_scores: List[float] = []
    artifact_scores: List[float] = []
    alignment_scores: List[float] = []
    color_scores: List[float] = []
    scene_scores: Dict[str, List[float]] = {name: [] for name, _, _ in SCENE_RANGES}

    prev_hist = None
    prev_scene = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step_frames == 0:
            t_sec = frame_idx / fps
            scene_name = next(
                (name for name, s, e in SCENE_RANGES if s <= t_sec < e),
                SCENE_RANGES[-1][0],
            )
            resized = cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None and scene_name == prev_scene:
                corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
                c_score = float(np.clip(corr * 10.0, 1.0, 10.0))
            else:
                c_score = 9.0  # Scene cut boundary

            a_score = score_frame_artifacts_and_text(gray)
            m_score = score_frame_alignment(gray)
            l_score = score_frame_color_lut(resized)

            consistency_scores.append(c_score)
            artifact_scores.append(a_score)
            alignment_scores.append(m_score)
            color_scores.append(l_score)

            frame_avg = 0.25 * (c_score + a_score + m_score + l_score)
            scene_scores[scene_name].append(frame_avg)

            prev_hist = hist
            prev_scene = scene_name
        frame_idx += 1

    cap.release()

    if not consistency_scores:
        raise RuntimeError("No frames were sampled from the video.")

    c_mean = float(np.mean(consistency_scores))
    a_mean = float(np.mean(artifact_scores))
    m_mean = float(np.mean(alignment_scores))
    l_mean = float(np.mean(color_scores))

    total_score = 0.25 * (c_mean + a_mean + m_mean + l_mean)
    if not specs["specs_ok"]:
        total_score = min(total_score, 6.5)

    failing_scenes = [
        name for name, vals in scene_scores.items() if vals and float(np.mean(vals)) < PASS_THRESHOLD
    ]

    passed = bool(total_score >= PASS_THRESHOLD and specs["specs_ok"])
    return QAReport(
        score=round(total_score, 2),
        passed=passed,
        specs_ok=bool(specs["specs_ok"]),
        consistency_score=round(c_mean, 2),
        artifact_score=round(a_mean, 2),
        alignment_score=round(m_mean, 2),
        color_score=round(l_mean, 2),
        failing_scenes=failing_scenes,
        details={"specs": specs, "sampled_frames": len(consistency_scores)},
    )


def run_qa_gate(video_path: Path, regenerate_cmd: str | None, max_retries: int) -> int:
    """Execute QA gate; trigger regeneration loop if score < 8.0."""
    for attempt in range(1, max_retries + 2):
        print(f"\n[QA] Attempt {attempt}: Analyzing {video_path}...")
        report = analyze_video(video_path)
        print(json.dumps(report.__dict__, indent=2))

        if report.passed:
            print(f"[QA PASS] Score {report.score:.2f} >= {PASS_THRESHOLD:.1f}. Cleared for release.")
            return 0

        print(
            f"[QA FAIL] Score {report.score:.2f} < {PASS_THRESHOLD:.1f}. "
            f"Failing scenes: {report.failing_scenes}"
        )
        if attempt > max_retries or not regenerate_cmd:
            break

        failed_arg = ",".join(report.failing_scenes) if report.failing_scenes else "all"
        cmd = regenerate_cmd.format(scenes=failed_arg, attempt=attempt)
        print(f"[REGEN] Triggering regeneration loop: {cmd}")
        subprocess.check_call(cmd, shell=True)

    print("[QA BLOCKED] Video failed to reach >= 8.0 threshold within retry limit.")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-by-Frame Video QA Review Gate")
    parser.add_argument("--video", type=Path, default=Path("output/final_video.mp4"))
    parser.add_argument(
        "--regenerate-cmd",
        type=str,
        default="bash render_video.sh --scenes {scenes} --attempt {attempt}",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    sys.exit(run_qa_gate(args.video, args.regenerate_cmd, args.max_retries))


if __name__ == "__main__":
    main()
