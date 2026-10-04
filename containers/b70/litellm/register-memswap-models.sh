#!/usr/bin/env bash
set -euo pipefail

LITELLM_PORT="${LITELLM_PORT:-4000}"
BASE="http://127.0.0.1:${LITELLM_PORT}"
LIST_API="$BASE/v1/models"
NEW_API="$BASE/model/new"

log() { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or put it in the litellm .env)"
command -v curl >/dev/null 2>&1 || die "curl required"

list_models() {
  curl -s -X GET "$LIST_API" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    | python3 -c 'import sys,json
try:
    print("\n".join(m["id"] for m in json.load(sys.stdin).get("data", [])))
except Exception:
    pass' 2>/dev/null || true
}

register_one() {
  local name="$1" port="$2" fallbacks="$3"

  if list_models | grep -qx "$name"; then
    log "'$name' is already registered — leaving it unchanged."
    return 0
  fi

  curl -sf -m 5 "http://127.0.0.1:${port}/v1/models" >/dev/null \
    || warn "'$name' upstream on :${port} not reachable yet"

  log "registering '$name' -> openai//model @ host.docker.internal:${port} ..."
  resp="$(curl -s -X POST "$NEW_API" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model_name\": \"$name\",
      \"litellm_params\": {
        \"model\": \"openai//model\",
        \"api_base\": \"http://host.docker.internal:${port}/v1\",
        \"api_key\": \"sk-no-key-required\"
      },
      \"model_info\": {
        \"fallbacks\": $fallbacks
      }
    }")"
  printf '%s\n' "$resp"

  if ! list_models | grep -qx "$name"; then
    die "'$name' did not appear in /v1/models — check the response above"
  fi
  log "'$name' registered and visible in /v1/models."
}

register_one "nex-n2.5-mini" "8003" "[]"
register_one "b70" "8000" '["nex-n2.5-mini"]'

log ""
log "next: make sure the pre-call hook is active (MEMSWAP_ENABLED=true) and the"
log "coordinator status is readable:"
log "  bash ../../../scripts/vllm-memswap.sh status"
log "  curl -s http://127.0.0.1:4000/v1/chat/completions -H 'Authorization: Bearer \$LITELLM_MASTER_KEY' \\"
log "    -H 'Content-Type: application/json' -d '{\"model\":\"b70\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'"
