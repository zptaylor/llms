#!/usr/bin/env bash
set -euo pipefail

LITELLM_PORT="${LITELLM_PORT:-4000}"
ADMIN_API="http://127.0.0.1:${LITELLM_PORT}/model/new"
MODEL_NAME="${MODEL_NAME:-local-or-free}"

log() { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or put it in the litellm .env)"
command -v curl >/dev/null 2>&1 || die "curl required"

curl -sf -m 5 "http://127.0.0.1:8000/v1/models" >/dev/null \
  || warn "local vLLM backend on :8000 not reachable yet (is a vllm-* container up?)"

log "registering '$MODEL_NAME' with LiteLLM at $ADMIN_API ..."
resp="$(curl -s -X POST "$ADMIN_API" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model_name\": \"$MODEL_NAME\",
    \"litellm_params\": {
      \"model\": \"openai//model\",
      \"api_base\": \"http://host.docker.internal:8000/v1\",
      \"api_key\": \"sk-no-key-required\"
    },
    \"model_info\": {
      \"fallbacks\": [\"local\", \"free\"]
    }
  }")"
echo "$resp"

log "verify:"
curl -s -X GET "http://127.0.0.1:${LITELLM_PORT}/v1/models" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  | python3 -c 'import sys,json; [print("  ",m["id"]) for m in json.load(sys.stdin).get("data",[])]' \
  || log "  (could not list models — check LiteLLM is healthy)"

log "done. Point clients at '$MODEL_NAME' for local-first-with-free-fallback routing."
