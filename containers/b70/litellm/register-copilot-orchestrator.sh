#!/usr/bin/env bash
set -euo pipefail

LITELLM_PORT="${LITELLM_PORT:-4000}"
ADMIN_API="http://127.0.0.1:${LITELLM_PORT}/model/new"
LIST_API="http://127.0.0.1:${LITELLM_PORT}/v1/models"

log() { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or set it in the litellm .env)"
command -v curl >/dev/null 2>&1 || die "curl required"

declare -a COPILOT_MODELS=(
  "copilot-sonnet-5|claude-sonnet-5|200000|8192"
  "copilot-haiku|claude-haiku-4.5|200000|8192"
  "copilot-gpt-5-mini|gpt-5-mini|128000|8192"
)

register_one() {
  local name="$1" upstream_id="$2" ctx="$3" maxout="$4" body
  if curl -sf -m 15 -H "Authorization: Bearer $LITELLM_MASTER_KEY" "$LIST_API" \
      | grep -q "\"$name\""; then
    log "  already registered ('$name' present) — skipping."
    return 0
  fi
  log "registering '$name' -> 'github_copilot/$upstream_id' ..."
  body="{
    \"model_name\": \"$name\",
    \"litellm_params\": {
      \"model\": \"github_copilot/$upstream_id\"
    },
    \"model_info\": {
      \"max_tokens\": $maxout,
      \"max_input_tokens\": $ctx,
      \"max_output_tokens\": $maxout,
      \"input_cost_per_token\": 0,
      \"output_cost_per_token\": 0
    }
  }"
  curl -s -X POST "$ADMIN_API" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' \
    -d "$body"
  echo
}

for entry in "${COPILOT_MODELS[@]}"; do
  IFS='|' read -r name upstream ctx maxout <<< "$entry"
  register_one "$name" "$upstream" "$ctx" "$maxout"
done

log ""
log "register-copilot-orchestrator.sh done."
log "next: if this is the first run, do the one-time device-flow auth (see the"
log "header of this script), then test:"
log "  curl -s http://127.0.0.1:${LITELLM_PORT}/v1/chat/completions \\"
log "    -H 'Authorization: Bearer \$LITELLM_MASTER_KEY' -H 'Content-Type: application/json' \\"
log "    -d '{\"model\":\"copilot-sonnet-5\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}'"
