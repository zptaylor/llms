#!/usr/bin/env bash

set -euo pipefail

VLLM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="/home/USER/vllm-qwen38-build"
VENV_PATH="/home/USER/nixos-config/vllm-qwen38/.venv"

if [[ -z "${HF_TOKEN:-}" ]] && command -v sops >/dev/null 2>&1 \
   && [[ -f "$VLLM_DIR/../secrets.yaml" ]]; then
  HF_TOKEN="$(sops -d --extract '["hf_token"]' "$VLLM_DIR/../secrets.yaml" 2>/dev/null || true)"
fi

echo "=== vLLM Qwen3.8-27B Setup Script ==="
echo "Build directory: $BUILD_DIR"
echo "Venv path: $VENV_PATH"
echo ""

echo ">>> Step 1: Setting up build directory"
mkdir -p "$BUILD_DIR"

if [ ! -d "$BUILD_DIR/scripts" ]; then
    echo "Copying vLLM source to build directory..."
    cp -r "$VLLM_DIR"/{scripts,patches,run.sh,README.md} "$BUILD_DIR"/
    cp -r "$VLLM_DIR"/patches/* "$BUILD_DIR"/patches/ 2>/dev/null || true
fi

echo ">>> Step 2: Creating venv at expected path"
if [ ! -d "$VENV_PATH" ]; then
    python3.12 -m venv "$VENV_PATH"
else
    echo "Venv already exists at $VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel

pip install uv

echo ">>> Step 3: Building vLLM with patches"
cd "$VLLM_DIR"
bash scripts/build.sh

echo ">>> Step 4: Fetching model with HF_TOKEN"
HF_TOKEN="$HF_TOKEN" bash scripts/model.sh fetch

echo ">>> Step 5: Starting vLLM server"
echo "The server will be available at http://0.0.0.0:19622"
echo "Press Ctrl+C to stop..."
bash scripts/serve.sh
