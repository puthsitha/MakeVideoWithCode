#!/usr/bin/env bash
set -euo pipefail

OS_NAME="$(uname -s)"
echo "=== Video Pipeline System Dependency Check ($OS_NAME) ==="

MISSING=0

# 1. Check Node.js & npm
if command -v node >/dev/null 2>&1; then
  echo "[OK] node: $(node -v)"
else
  echo "[MISSING] node is not installed."
  echo "  -> macOS (Homebrew): brew install node"
  echo "  -> Ubuntu/Debian:    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs"
  MISSING=1
fi

if command -v npm >/dev/null 2>&1; then
  echo "[OK] npm: $(npm -v)"
else
  echo "[MISSING] npm is not installed (bundled with Node.js)."
  MISSING=1
fi

# 2. Check FFmpeg & FFprobe
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  echo "[OK] ffmpeg: $(ffmpeg -version | head -n 1)"
  echo "[OK] ffprobe: $(ffprobe -version | head -n 1)"
else
  echo "[MISSING] ffmpeg / ffprobe not found."
  if [[ "$OS_NAME" == "Darwin" ]]; then
    echo "  -> Install on macOS: brew install ffmpeg"
  else
    echo "  -> Install on Linux: sudo apt-get update && sudo apt-get install -y ffmpeg"
  fi
  MISSING=1
fi

# 3. Check fluent-ffmpeg & Node packages
if [[ -f "package.json" ]]; then
  if node -e "require.resolve('fluent-ffmpeg')" >/dev/null 2>&1; then
    echo "[OK] fluent-ffmpeg is installed in node_modules."
  else
    echo "[MISSING] fluent-ffmpeg not installed locally."
    echo "  -> Run: npm install fluent-ffmpeg"
    MISSING=1
  fi
else
  echo "[MISSING] package.json not found. Run: npm init -y && npm install fluent-ffmpeg"
  MISSING=1
fi

# 4. Check Python3 & pip
if command -v python3 >/dev/null 2>&1; then
  echo "[OK] python3: $(python3 --version)"
else
  echo "[MISSING] python3 is not installed."
  echo "  -> macOS: brew install python@3.12"
  echo "  -> Linux: sudo apt-get install -y python3 python3-pip"
  MISSING=1
fi

# 5. Check Python multimedia & QA libraries
python3 - <<'PY' || MISSING=1
import importlib
required = ["huggingface_hub", "soundfile", "numpy", "cv2"]
for pkg in required:
    try:
        importlib.import_module(pkg)
        print(f"[OK] python package: {pkg}")
    except ImportError:
        pip_name = "opencv-python" if pkg == "cv2" else pkg
        print(f"[MISSING] python package: {pkg} -> pip install {pip_name}")
        raise SystemExit(1)
PY

if [[ "$MISSING" -eq 0 ]]; then
  echo "=== All core system dependencies are ready. ==="
else
  echo ""
  echo "One-liner quick install (macOS):"
  echo "  brew install node ffmpeg && npm install fluent-ffmpeg && pip install voxcpm huggingface_hub soundfile torch numpy opencv-python"
  echo "One-liner quick install (Ubuntu/Debian):"
  echo "  sudo apt-get update && sudo apt-get install -y ffmpeg nodejs npm python3-pip && npm install fluent-ffmpeg && pip install voxcpm huggingface_hub soundfile torch numpy opencv-python"
  exit 1
fi
