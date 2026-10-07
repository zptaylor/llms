#!/usr/bin/env bash
# Register the offload-gateway as a LiteLLM model group ('offload') that
# routes to the gateway on litellm's own network, then to an OpenRouter
# free model through the PIA tunnel. Idempotent: skips if already present.
#
# The gateway forwards the client 'model' verbatim to OpenRouter, so the
# row carries a concrete free-model id (swap with OFFLOAD_MODEL). When the
# tunnel is down or a leak is detected the gateway fails closed (503/421)
# and 'offload' is deliberately PINNED (no fallbacks): a blocked prompt must
# never be re-sent down another route (that would leak the secret off-tunnel),
# and a tunnel-down must not silently de-anonymize onto a direct backend.
# Clients that need a guaranteed answer use a different group (e.g. 'free').
#
# Run on ai-host:  bash containers/ai-host/litellm/register-offload.sh
set -euo pipefail

LITELLM_PORT="${LITELLM_PORT:-4000}"
ADMIN_API="http://127.0.0.1:${LITELLM_PORT}/model/new"
LIST_API="http://127.0.0.1:${LITELLM_PORT}/v1/models"
GW_PORT="${OFFLOAD_PORT:-8800}"
DEFAULT_MODEL="${OFFLOAD_MODEL:-liquid/lfm-2.5-2.6b:free}"

log() { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or put it in the litellm .env)"
command -v curl >/dev/null 2>&1 || die "curl required"

# The gateway authenticates with OFFLOAD_ADMIN_KEY. Read it from the offload
# .env (written by offload/setup-env.sh) so litellm's row can send it.
OFFLOAD_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../offload" && pwd)/.env"
[ -f "$OFFLOAD_ENV" ] || die "offload .env missing: $OFFLOAD_ENV (run containers/ai-host/offload/setup-env.sh first)"
OFFLOAD_ADMIN_KEY="$(grep -E '^OFFLOAD_ADMIN_KEY=' "$OFFLOAD_ENV" | cut -d= -f2-)"
[ -n "$OFFLOAD_ADMIN_KEY" ] || die "OFFLOAD_ADMIN_KEY empty in $OFFLOAD_ENV"

# Reach the gateway by container name on litellm's network.
GW_BASE="http://offload-gateway:${GW_PORT}"
log "registering 'offload' -> gateway ${GW_BASE} (model: $DEFAULT_MODEL, PIA-tunnelled, cost 0) ..."

body="{
  \"model_name\": \"offload\",
  \"litellm_params\": {
    \"model\": \"openai/${DEFAULT_MODEL}\",
    \"api_base\": \"${GW_BASE}/v1\",
    \"api_key\": \"${OFFLOAD_ADMIN_KEY}\"
  },
  \"model_info\": {
    \"mode\": \"chat\",
    \"max_input_tokens\": 32768,
    \"max_output_tokens\": 8192,
    \"input_cost_per_token\": 0,
    \"output_cost_per_token\": 0
  }
}"

# idempotency guard
if curl -sf -m 10 -H "Authorization: Bearer $LITELLM_MASTER_KEY" "$LIST_API" \
    | grep -q '"offload"'; then
  log "  'offload' already registered — skipping (delete via /model/delete to re-point it)."
else
  curl -s -X POST "$ADMIN_API" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' \
    -d "$body"
  echo
fi

log "verify:"
curl -s -X GET "$LIST_API" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  | python3 -c 'import sys,json; [print("  ",m["id"]) for m in json.load(sys.stdin).get("data",[]) if "offload" in m["id"].lower() or "free" in m["id"].lower()]' \
  || log "  (could not list models — check LiteLLM is healthy)"
log "done. Point clients at 'offload' (PIA-tunnelled, leak-scanned, cost 0)."
