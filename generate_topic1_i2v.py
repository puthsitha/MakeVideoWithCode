"""
Topic 1: Hands-Free Batch Image-to-Video (I2V) Generator
Sends all 13 scene images in topic1_scenes/ + their paired motion_prompt from
topic1_scenes/topic1_prompts.json to your chosen I2V API provider, saves the
animated MP4 clips into topic1_scenes/clips/scene_<id>.mp4, and optionally
runs render_topic1_reel.py to assemble output/topic1_reel_final.mp4.

Supported providers (auto-detected from .env or --provider):
  - veo    : Google Gemini Veo API (requires GEMINI_API_KEY in .env)
  - fal    : fal.ai Kling / Hailuo / Veo (requires FAL_KEY in .env)
  - runway : Runway Gen-4 Turbo (requires RUNWAYML_API_SECRET in .env)

Usage:
  .venv/bin/python generate_topic1_i2v.py --scenes 00_hook           # test 1 scene first
  .venv/bin/python generate_topic1_i2v.py --force --assemble        # generate all 13 & build reel
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests

SCENE_DIR = Path("topic1_scenes")
CLIPS_DIR = SCENE_DIR / "clips"
MANIFEST_PATH = SCENE_DIR / "topic1_prompts.json"
ENV_PATH = Path(".env")


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def img_to_b64(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode("ascii")


def download_file(url: str, out_path: Path, headers: Optional[Dict[str, str]] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers or {}, stream=True, timeout=180) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)


# -------------------- PROVIDER 1: GOOGLE VEO (GEMINI API) --------------------

def run_veo(img_path: Path, prompt: str, out_path: Path, model: str, dur_sec: int) -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")

    base = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base}/models/{model}:predictLongRunning"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    clip_dur = 5 if dur_sec <= 5 else 8
    payload = {
        "instances": [
            {
                "prompt": prompt,
                "image": {
                    "bytesBase64Encoded": img_to_b64(img_path),
                    "mimeType": "image/jpeg",
                },
            }
        ],
        "parameters": {
            "aspectRatio": "9:16",
            "durationSeconds": clip_dur,
            "personGeneration": "allow_adult",
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Veo submit failed ({resp.status_code}): {resp.text[:500]}")
    op_name = resp.json()["name"]
    print(f"  [VEO] Operation started: {op_name}")

    poll_url = f"{base}/{op_name}"
    while True:
        time.sleep(8)
        pr = requests.get(poll_url, headers={"x-goog-api-key": api_key}, timeout=60)
        pr.raise_for_status()
        data = pr.json()
        if data.get("done"):
            if "error" in data:
                raise RuntimeError(f"Veo error: {data['error']}")
            samples = (
                data.get("response", {})
                .get("generateVideoResponse", {})
                .get("generatedSamples", [])
            )
            if not samples:
                raise RuntimeError(f"Veo returned no video samples: {json.dumps(data)[:400]}")
            video_uri = samples[0]["video"]["uri"]
            download_file(video_uri, out_path, headers={"x-goog-api-key": api_key})
            return
        print("  [VEO] Waiting for render...")


# -------------------- PROVIDER 2: FAL.AI (KLING / HAILUO / VEO) --------------------

def run_fal(img_path: Path, prompt: str, out_path: Path, model: str, dur_sec: int) -> None:
    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        raise RuntimeError("Missing FAL_KEY in .env")

    headers = {"Authorization": f"Key {fal_key}", "Content-Type": "application/json"}
    data_uri = f"data:image/jpeg;base64,{img_to_b64(img_path)}"
    payload = {
        "prompt": prompt,
        "image_url": data_uri,
        "duration": "5" if dur_sec <= 6 else "10",
        "aspect_ratio": "9:16",
    }
    submit_url = f"https://queue.fal.run/{model}"
    resp = requests.post(submit_url, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"fal.ai submit failed ({resp.status_code}): {resp.text[:500]}")
    job = resp.json()
    status_url = job["status_url"]
    response_url = job["response_url"]
    print(f"  [FAL] Queued job: {job.get('request_id', '')}")

    while True:
        time.sleep(6)
        sr = requests.get(status_url, headers=headers, timeout=60)
        sr.raise_for_status()
        st = sr.json().get("status")
        if st == "COMPLETED":
            rr = requests.get(response_url, headers=headers, timeout=60)
            rr.raise_for_status()
            res = rr.json()
            video_url = res.get("video", {}).get("url") or res.get("video_url")
            if not video_url:
                raise RuntimeError(f"fal.ai returned no video URL: {json.dumps(res)[:400]}")
            download_file(video_url, out_path)
            return
        if st in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"fal.ai job {st}: {sr.text[:400]}")
        print(f"  [FAL] Status: {st}...")


# -------------------- PROVIDER 3: RUNWAY (GEN-4 TURBO) --------------------

def run_runway(img_path: Path, prompt: str, out_path: Path, model: str, dur_sec: int) -> None:
    api_key = os.environ.get("RUNWAYML_API_SECRET")
    if not api_key:
        raise RuntimeError("Missing RUNWAYML_API_SECRET in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Runway-Version": "2024-11-06",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "promptImage": f"data:image/jpeg;base64,{img_to_b64(img_path)}",
        "promptText": prompt,
        "ratio": "720:1280",
        "duration": 5 if dur_sec <= 6 else 10,
    }
    resp = requests.post("https://api.dev.runwayml.com/v1/image_to_video", headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Runway submit failed ({resp.status_code}): {resp.text[:500]}")
    task_id = resp.json()["id"]
    print(f"  [RUNWAY] Task started: {task_id}")

    while True:
        time.sleep(6)
        tr = requests.get(f"https://api.dev.runwayml.com/v1/tasks/{task_id}", headers=headers, timeout=60)
        tr.raise_for_status()
        data = tr.json()
        st = data.get("status")
        if st == "SUCCEEDED":
            video_url = data["output"][0]
            download_file(video_url, out_path)
            return
        if st in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Runway task {st}: {data}")
        print(f"  [RUNWAY] Status: {st}...")


def detect_provider() -> Optional[str]:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "veo"
    if os.environ.get("FAL_KEY"):
        return "fal"
    if os.environ.get("RUNWAYML_API_SECRET"):
        return "runway"
    return None


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["veo", "fal", "runway"], default=None)
    parser.add_argument("--model", default=None, help="Override default provider model")
    parser.add_argument("--scenes", default="all", help="Comma-separated scene IDs or 'all'")
    parser.add_argument("--force", action="store_true", help="Overwrite existing clips in topic1_scenes/clips/")
    parser.add_argument("--assemble", action="store_true", help="Run render_topic1_reel.py after generating clips")
    args = parser.parse_args()

    provider = args.provider or detect_provider()
    if not provider:
        print(
            "[ERROR] No API key found in .env. Add one of:\n"
            "  GEMINI_API_KEY=...        (Google Veo)\n"
            "  FAL_KEY=...               (fal.ai: Kling / Hailuo / Veo)\n"
            "  RUNWAYML_API_SECRET=...   (Runway Gen-4 Turbo)",
            file=sys.stderr,
        )
        sys.exit(2)

    default_models = {
        "veo": "veo-3.0-fast-generate-001",
        "fal": "fal-ai/kling-video/v2.1/standard/image-to-video",
        "runway": "gen4_turbo",
    }
    model = args.model or default_models[provider]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    target_ids = None if args.scenes == "all" else {s.strip() for s in args.scenes.split(",")}

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[I2V] Provider={provider} | Model={model}")

    for s in manifest["scenes"]:
        sid = s["id"]
        if target_ids is not None and sid not in target_ids:
            continue
        img_path = SCENE_DIR / s["file"]
        out_path = CLIPS_DIR / f"scene_{sid}.mp4"
        if out_path.exists() and not args.force:
            print(f"[SKIP] {out_path} already exists (pass --force to overwrite)")
            continue
        dur = int(round(s["end"] - s["start"]))
        prompt = s["motion_prompt"]
        print(f"[GEN] {sid} ({dur}s) <- {img_path.name}")
        if provider == "veo":
            run_veo(img_path, prompt, out_path, model, dur)
        elif provider == "fal":
            run_fal(img_path, prompt, out_path, model, dur)
        elif provider == "runway":
            run_runway(img_path, prompt, out_path, model, dur)
        print(f"  [SAVED] {out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")

    if args.assemble:
        print("[ASSEMBLE] Running render_topic1_reel.py...")
        subprocess.check_call([sys.executable, "render_topic1_reel.py"])


if __name__ == "__main__":
    main()
