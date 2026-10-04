#!/usr/bin/env bash


set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
    build)
        echo "Building vLLM Qwen3.8 environment..."
        INTEL_ONEAPI_ROOT=$(find /nix/store -maxdepth 1 -name 'intel-oneapi-toolkit-*' -type d | head -1)
        if [ -n "$INTEL_ONEAPI_ROOT" ] && [ -f "$INTEL_ONEAPI_ROOT/setvars.sh" ]; then
            source "$INTEL_ONEAPI_ROOT/setvars.sh" --force 2>/dev/null || true
        else
            source /run/current-system/sw/opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
        fi
        bash "$HERE/scripts/build.sh"
        ;;
    fetch)
        echo "Fetching model..."
        HF_TOKEN="${HF_TOKEN:-}" bash "$HERE/scripts/model.sh" fetch
        bash "$HERE/scripts/model.sh" verify
        ;;
    serve)
        echo "Starting vLLM server..."
        INTEL_ONEAPI_ROOT=$(find /nix/store -maxdepth 1 -name 'intel-oneapi-toolkit-*' -type d | head -1)
        if [ -n "$INTEL_ONEAPI_ROOT" ] && [ -f "$INTEL_ONEAPI_ROOT/setvars.sh" ]; then
            source "$INTEL_ONEAPI_ROOT/setvars.sh" --force 2>/dev/null || true
        else
            source /run/current-system/sw/opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
        fi
        shift
        bash "$HERE/scripts/serve.sh" "$@"
        ;;
    verify)
        echo "Verifying installation..."
        bash "$HERE/scripts/verify-install.sh"
        ;;
    bench)
        echo "Running benchmark..."
        shift
        bash "$HERE/scripts/bench-strict.sh" "$@"
        ;;
    quality)
        echo "Running quality gate..."
        bash "$HERE/scripts/quality-gate.sh"
        ;;
    setup)
        echo "Running full setup (build + fetch + verify)..."
        INTEL_ONEAPI_ROOT=$(find /nix/store -maxdepth 1 -name 'intel-oneapi-toolkit-*' -type d | head -1)
        if [ -n "$INTEL_ONEAPI_ROOT" ] && [ -f "$INTEL_ONEAPI_ROOT/setvars.sh" ]; then
            source "$INTEL_ONEAPI_ROOT/setvars.sh" --force 2>/dev/null || true
        else
            source /run/current-system/sw/opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
        fi
        bash "$HERE/scripts/build.sh"
        HF_TOKEN="${HF_TOKEN:-}" bash "$HERE/scripts/model.sh" fetch
        bash "$HERE/scripts/model.sh" verify
        bash "$HERE/scripts/verify-install.sh"
        bash "$HERE/scripts/quality-gate.sh"
        echo "Setup complete! Run './run.sh serve' to start the server."
        ;;
    *)
        echo "Usage: $0 {build|fetch|serve|verify|bench|quality|setup}"
        echo ""
        echo "Commands:"
        echo "  build   - Build vLLM with patches and kernels"
        echo "  fetch   - Fetch and verify the Qwen3.8-27B model"
        echo "  serve   - Start the vLLM server (passes args to serve.sh)"
        echo "  verify  - Verify installation (host drivers, ops, canary, smoke)"
        echo "  bench   - Run cold workload benchmark (passes args to bench-strict.sh)"
        echo "  quality - Run quality gate (arithmetic + code canaries)"
        echo "  setup   - Run full setup: build + fetch + verify + quality"
        exit 1
        ;;
esac
