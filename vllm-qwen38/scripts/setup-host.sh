#!/usr/bin/env bash


set -euo pipefail

echo "== Intel graphics runtime install (NixOS) =="


echo "== verify =="
xpu-smi --list-gpus 2>/dev/null || echo "xpu-smi not present yet — install intel-gpu-tools or xpu-smi from the Intel oneAPI/BMC packages"

ls -la /run/opengl-driver/lib/libze_intel_gpu.so.* 2>/dev/null | tail -1 || \
echo "libze_intel_gpu not found — check driver package"

echo "== NOTE =="
echo "A custom xe-next kernel was used by the reference machine; distro"
echo "kernels also work if recent enough for Battlemage. If B70 is not"
echo "enumerated (xpu-smi empty), you need a newer xe driver line."
echo "Re-run: bash scripts/verify-install.sh --host-only"
