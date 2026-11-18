#!/usr/bin/env bash
set -euo pipefail

# Video Render & Assembly Pipeline (1080x1920 @ 30fps, 9:16, 120s)
# Usage: bash render_video.sh [--scenes all|01_intro,...] [--attempt 1]

SCENES_ARG="all"
ATTEMPT_ARG="1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenes) SCENES_ARG="$2"; shift 2 ;;
    --attempt) ATTEMPT_ARG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p output/scenes output/graded
NEGATIVE_PROMPT="Do NOT bake, burn, or embed any text, captions, or subtitles into the video frames."

echo "=== [RENDER] Attempt #${ATTEMPT_ARG} | Target Scenes: ${SCENES_ARG} ==="

# Step 1: Invoke generative video engine per scene from prompts.json
# Replace `ai-video-gen` with your active provider CLI (e.g., Runway/Kling/Veo/Wan/ComfyUI wrapper)
python3 - <<PY
import json, subprocess, sys
from pathlib import Path

manifest = json.loads(Path("prompts.json").read_text())
target_scenes = "${SCENES_ARG}".split(",") if "${SCENES_ARG}" != "all" else [s["id"] for s in manifest["scenes"]]
neg_prompt = manifest["negative_prompt"]

for scene in manifest["scenes"]:
    sid = scene["id"]
    if sid not in target_scenes:
        continue
    dur = int(scene["end"] - scene["start"])
    out_clip = Path(f"output/scenes/{sid}.mp4")
    cmd = [
        "ai-video-gen",
        "--prompt", f"{scene['prompt']} Alignment: {scene['alignment']}. Color LUT: {manifest['color_lut']}.",
        "--negative-prompt", neg_prompt,
        "--aspect-ratio", "9:16",
        "--resolution", "1080x1920",
        "--fps", "30",
        "--duration", str(dur),
        "--seed", str(4200 + int("${ATTEMPT_ARG}")),
        "--output", str(out_clip),
    ]
    print("[GENERATOR CMD]", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    # Uncomment when your local/API video generator CLI is configured:
    # subprocess.check_call(cmd)
PY

# Step 2: Enforce 1080x1920 @ 30fps, Nocturnal Amber & Slate Color Grade, and Mux Audio
CONCAT_LIST="output/concat_list.txt"
: > "$CONCAT_LIST"

for clip in output/scenes/*.mp4; do
  [[ -e "$clip" ]] || continue
  base="$(basename "$clip")"
  graded="output/graded/$base"

  # Enforce strict 1080x1920, 30fps, SAR 1:1, and warm amber/slate cinematic contrast curve
  ffmpeg -y -i "$clip" \
    -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1,eq=contrast=1.08:saturation=1.06:gamma=0.98,colorbalance=rs=0.04:gs=0.01:bs=-0.05:rh=0.05:gh=0.02:bh=-0.03,format=yuv420p" \
    -c:v libx264 -preset slow -crf 18 -r 30 -an \
    "$graded"

  echo "file '$(pwd)/$graded'" >> "$CONCAT_LIST"
done

if [[ -s "$CONCAT_LIST" && -f "output/narration.wav" ]]; then
  ffmpeg -y -f concat -safe 0 -i "$CONCAT_LIST" -i "output/narration.wav" \
    -map 0:v:0 -map 1:a:0 \
    -c:v copy -c:a aac -b:a 192k -ar 48000 \
    -t 120 -shortest \
    "output/final_video.mp4"

  echo "[RENDER DONE] output/final_video.mp4 created. Running QA Gate..."
  python3 qa_gate.py --video "output/final_video.mp4"
else
  echo "[INFO] Scene clips (output/scenes/*.mp4) or output/narration.wav not yet present; command blueprint verified."
fi
