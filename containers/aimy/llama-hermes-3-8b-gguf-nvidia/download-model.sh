#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="NousResearch/Hermes-3-Llama-3.1-8B-GGUF"
MODEL_DIRNAME="Hermes-3-Llama-3.1-8B-GGUF"
FILENAME="Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
EXPECTED_BYTES=4920733824
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
TARGET_DIR="$MODELS_DIR/$MODEL_DIRNAME"
TARGET_FILE="$TARGET_DIR/$FILENAME"
HF_REVISION="${HF_REVISION:-}"

log() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

download() {
  local url="$1" out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    die "need curl or wget to download"
  fi
}

main() {
  if [ -f "$TARGET_FILE" ] && [ "$(stat -c%s "$TARGET_FILE" 2>/dev/null)" = "$EXPECTED_BYTES" ]; then
    log "already downloaded: $TARGET_FILE"
    return 0
  fi

  mkdir -p "$TARGET_DIR"

  local base="https://huggingface.co/$MODEL_REPO"
  local rev="main"
  [ -n "$HF_REVISION" ] && rev="$HF_REVISION"
  local url="$base/resolve/$rev/$FILENAME"

  log "downloading $url -> $TARGET_FILE"
  local tmp="$TARGET_FILE.tmp.$$"
  download "$url" "$tmp"
  mv "$tmp" "$TARGET_FILE"

  local got
  got="$(stat -c%s "$TARGET_FILE")"
  if [ "$got" != "$EXPECTED_BYTES" ]; then
    die "size mismatch: got $got, expected $EXPECTED_BYTES (revision changed?)"
  fi
  log "done: $TARGET_FILE ($got bytes)"
}

main "$@"
