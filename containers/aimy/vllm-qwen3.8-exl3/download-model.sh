#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="turboderp/Qwen3.8-27B-exl3"
HF_REVISION_DEFAULT="113cf7ab958054860e43fb7f3063b1af19171095"
MODEL_DIRNAME="qwen3.8-exl3"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
TARGET_DIR="$MODELS_DIR/$MODEL_DIRNAME"
HF_REVISION="${HF_REVISION:-$HF_REVISION_DEFAULT}"

HF_VENV_DIR="${HF_VENV_DIR:-$HOME/.hf-cli-venv}"
SECRETS_YAML="${SECRETS_YAML:-}"

log() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

repo_root() {
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/secrets.yaml" ] && [ -d "$dir/.git" ]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  die "cannot locate nixos-config repo root (secrets.yaml + .git not found)"
}

model_present() {
  [ -f "$TARGET_DIR/config.json" ] && ls "$TARGET_DIR"/*.safetensors >/dev/null 2>&1
}

REVISION_MARKER="$TARGET_DIR/.hf-revision"

revision_matches() {
  [ -f "$REVISION_MARKER" ] && [ "$(cat "$REVISION_MARKER")" = "$HF_REVISION" ]
}

ensure_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
    return 0
  fi

  if [ ! -x "$HF_VENV_DIR/bin/hf" ]; then
    log "installing huggingface-hub (hf CLI) into $HF_VENV_DIR ..."
    if ! command -v python3 >/dev/null 2>&1; then
      die "python3 not found; install python3 first (e.g. nix shell nixpkgs#python3)"
    fi
    python3 -m venv "$HF_VENV_DIR"
    "$HF_VENV_DIR/bin/pip" install -q --upgrade "huggingface_hub[cli]"
  fi
  HF_CLI="$HF_VENV_DIR/bin/hf"
}

load_hf_token() {
  [ -n "${HF_TOKEN:-}" ] && return 0

  local repo val
  repo="$(repo_root)"
  [ -n "$SECRETS_YAML" ] && repo="$(cd "$(dirname "$SECRETS_YAML")" && pwd)"
  SECRETS_YAML="${SECRETS_YAML:-$repo/secrets.yaml}"

  if [ -f "$SECRETS_YAML" ] && command -v sops >/dev/null 2>&1; then
    val="$(sops --decrypt --extract '["hf_token"]' "$SECRETS_YAML" 2>/dev/null || true)"
    case "$val" in
      HF_TOKEN=*) HF_TOKEN="${val#HF_TOKEN=}"; log "HF_TOKEN read from sops hf_token";;
    esac
  fi

  if [ -z "${HF_TOKEN:-}" ] && [ -f "$(dirname "${BASH_SOURCE[0]}")/.env" ]; then
    HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "$(dirname "${BASH_SOURCE[0]}")/.env" | tail -1)"
  fi

  [ -n "${HF_TOKEN:-}" ] || log "note: no HF_TOKEN found (checkpoint is public; continuing anonymously)"
}

main() {
  if model_present; then
    if revision_matches; then
      log "already downloaded: $TARGET_DIR (revision $HF_REVISION)"
      return 0
    fi
    if [ -f "$REVISION_MARKER" ]; then
      have="$(cat "$REVISION_MARKER")"
      die "$TARGET_DIR exists but is revision '$have', not the pinned '$HF_REVISION'.
       EXL3 weights are kernel-specific, so these are NOT interchangeable.
       Either: rm -rf '$TARGET_DIR' and re-run, or set HF_REVISION='$have' if you
       have independently confirmed that revision is servable."
    fi
    die "$TARGET_DIR exists but carries no .hf-revision marker, so its provenance
       is unknown -- it was not fetched by this script, and EXL3 weights are
       kernel-specific. Either: rm -rf '$TARGET_DIR' and re-run, or, if you have
       independently confirmed the on-disk weights are revision '$HF_REVISION',
       record it with: printf '%s\\n' '$HF_REVISION' > '$REVISION_MARKER'"
  fi

  ensure_hf_cli
  load_hf_token

  log "downloading $MODEL_REPO @ $HF_REVISION into $TARGET_DIR ..."
  local cmd=("$HF_CLI" "download" "$MODEL_REPO" "--local-dir" "$TARGET_DIR" --revision "$HF_REVISION")
  HF_TOKEN="${HF_TOKEN:-}" "${cmd[@]}"

  if ! model_present; then
    die "download completed but checkpoint not found under $TARGET_DIR (check revision / token)"
  fi
  printf '%s\n' "$HF_REVISION" > "$REVISION_MARKER"
  log "done: $MODEL_REPO present at $TARGET_DIR (revision $HF_REVISION)"
  log "mount it read-only as :/model (already wired in docker-compose.yml)"
}

main "$@"
