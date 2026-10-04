#!/usr/bin/env bash


set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VLLM_BASE="$(cat "$HERE/patches/vllm/BASE_COMMIT.txt" | head -1)"
KERNELS_BASE="$(cat "$HERE/patches/vllm-xpu-kernels/BASE_COMMIT.txt" | head -1)"

echo "== humble-b70 build =="
echo "vLLM base: $VLLM_BASE"
echo "kernels base: $KERNELS_BASE"

uv venv --python 3.11 .venv
if [[ ! -d "$HERE/src/vllm" ]]; then
    git clone https://github.com/vllm-project/vllm.git "$HERE/src/vllm"
    git -C "$HERE/src/vllm" checkout "$VLLM_BASE"
fi

if ! git -C "$HERE/src/vllm" apply --check "$HERE/patches/vllm/"*.patch 2>/dev/null; then
    echo "vLLM patches already applied (or conflict) — skipping apply."
fi

git -C "$HERE/src/vllm" apply "$HERE/patches/vllm/"*.patch 2>/dev/null || true

if [[ ! -d "$HERE/src/vllm-xpu-kernels" ]]; then
    git clone https://github.com/vllm-project/vllm-xpu-kernels.git "$HERE/src/vllm-xpu-kernels"
    git -C "$HERE/src/vllm-xpu-kernels" checkout "$KERNELS_BASE"
fi

git -C "$HERE/src/vllm-xpu-kernels" apply --check "$HERE/patches/vllm-xpu-kernels/"*.patch 2>/dev/null || true
git -C "$HERE/src/vllm-xpu-kernels" apply "$HERE/patches/vllm-xpu-kernels/"*.patch 2>/dev/null || true

uv venv --python 3.12 "$HERE/.venv"

"$HERE/.venv/bin/pip" install --extra-index-url \
    https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
    "torch==2.13.0+xpu"

export MAX_JOBS="${MAX_JOBS:-6}"
export VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf
export VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf

"$HERE/.venv/bin/pip" install numpy "cmake==3.31.8" ninja \
    "setuptools>=77,<80" setuptools-scm wheel build

(cd "$HERE/src/vllm-xpu-kernels" && \
    "$HERE/.venv/bin/python" setup.py bdist_wheel \
    --dist-dir "$HERE/dist" --py-limited-api=cp38)

"$HERE/.venv/bin/pip" install --no-build-isolation -e "$HERE/src/vllm"
"$HERE/.venv/bin/pip" install --force-reinstall --no-deps "$HERE/dist"/vllm_xpu_kernels-*.whl

echo "== done. venv at $HERE/.venv =="
echo "Next: bash scripts/model.sh fetch ; bash scripts/serve.sh"
