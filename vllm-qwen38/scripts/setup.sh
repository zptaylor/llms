#!/usr/bin/env bash
set -euo pipefail

REPO_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="/home/USER/vllm-qwen38-build"
MODEL_DIR="/home/USER/models/johannrplaster/Qwen3.8-27B-Uncensored-int4-AutoRound"
ONEAPI_TOOLKIT="$(readlink -f /nix/store/*intel-oneapi-toolkit*/ 2>/dev/null | head -1 || true)"
ONEAPI_TOOLKIT="${ONEAPI_TOOLKIT:-}"

log()  { printf '\033[1;34m>>\033[0m %s\n' "$*"; }
step() { printf '\n\033[1;33m== %s ==\033[0m\n' "$*"; }
die()  { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; exit 1; }

if [[ -z "${HF_TOKEN:-}" ]] && command -v sops >/dev/null 2>&1; then
  if [[ -f "$REPO_SRC/../secrets.yaml" ]]; then
    log "Reading HF token from sops (secrets.yaml -> hf_token)"
    HF_TOKEN="$(sops -d --extract '["hf_token"]' "$REPO_SRC/../secrets.yaml" 2>/dev/null || true)"
    if [[ -z "$HF_TOKEN" ]]; then
      log "  no token extracted (sops key absent or no age key); model fetch may hit gated repo"
    fi
  fi
fi

step "1/4 Re-initialising writable build dir: $BUILD_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cp -r "$REPO_SRC"/{scripts,patches,README.md,run.sh,nixos-module.nix} "$BUILD_DIR"/ 2>/dev/null || true
chmod -R u+rwX "$BUILD_DIR"
chown -R "${USER:-USER}:${USER:-USER}" "$BUILD_DIR" 2>/dev/null || true
ls -ld "$BUILD_DIR"

step "2/4 Intel oneAPI environment"
if [[ -n "$ONEAPI_TOOLKIT" ]] && [[ -f "$ONEAPI_TOOLKIT/setvars.sh" ]]; then
  log "Sourcing $ONEAPI_TOOLKIT/setvars.sh"
  source "$ONEAPI_TOOLKIT/setvars.sh" --force >/dev/null 2>&1 || true
else
  log "oneAPI setvars.sh not found under /nix/store/*intel-oneapi-toolkit*/; compile may fail"
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  step "3/4 Building venv + vLLM (+ xpu kernels) — this is the long step"
  log "Build dir: $BUILD_DIR  (build.sh writes \$HERE/.venv, \$HERE/src, \$HERE/dist)"
  cd "$BUILD_DIR"
  bash scripts/build.sh
else
  log "SKIP_BUILD=1 — leaving existing venv as-is"
fi

if [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  step "4/4 Ensuring model present at $MODEL_DIR"
  EXPECTED_SHARDS=$(grep -o 'model-0000[0-9]-of-0000[0-9]' \
    "$MODEL_DIR/model.safetensors.index.json" 2>/dev/null | sort -u | wc -l)
  [[ "$EXPECTED_SHARDS" =~ ^[0-9]+$ ]] || EXPECTED_SHARDS=7
  PRESENT_SHARDS=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)
  if [[ "$PRESENT_SHARDS" -ge "$EXPECTED_SHARDS" ]] && [[ "$PRESENT_SHARDS" -gt 0 ]]; then
    log "Model already present ($PRESENT_SHARDS shards) — skipping download"
  else
    log "Model missing/incomplete ($PRESENT_SHARDS/$EXPECTED_SHARDS shards) — downloading pinned revision"
    cd "$BUILD_DIR"
    export HF_TOKEN="${HF_TOKEN:-}"
    bash scripts/model.sh fetch
  fi
else
  log "SKIP_MODEL=1 — leaving existing model as-is"
fi

step "Done"
log "venv: $BUILD_DIR/.venv"
log "model: $MODEL_DIR"
log "Next:"
log "  1) flip services.vllm-qwen38.enable = true in hosts/b70-host/configuration.nix"
log "  2) sudo nixos-rebuild switch --flake .#ai"
log "  3) sudo systemctl start vllm-qwen38 ; journalctl -u vllm-qwen38 -f"
log "Test: curl -X POST http://ai:19622/v1/chat/completions ..."
