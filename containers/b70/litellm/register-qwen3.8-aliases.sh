#!/usr/bin/env bash
set -euo pipefail

LITELLM_PORT="${LITELLM_PORT:-4000}"
API="http://127.0.0.1:${LITELLM_PORT}"
ADMIN_API="$API/model/new"
LIST_API="$API/v1/models"

log()  { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or put it in the litellm .env)"
command -v curl >/dev/null 2>&1 || die "curl required"

curl -sf -m 5 "http://127.0.0.1:8000/v1/models" >/dev/null \
  || warn "local vLLM backend on :8000 not reachable yet (is a vllm-qwen3.8-* container up?)"

ALIASES=(
  "qwen3.8-65k:57344"
  "qwen3.8-96k:90112"
)

existing="$(curl -s -X GET "$LIST_API" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  | python3 -c 'import sys,json; print("\n".join(m["id"] for m in json.load(sys.stdin).get("data",[])))' 2>/dev/null || true)"

if printf '%s\n' "$existing" | grep -qx "qwen3.8-131k"; then
  log "retiring old alias 'qwen3.8-131k' (superseded by qwen3.8-96k) ..."
  stale_id="$(curl -s -X GET "$API/model/info" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    | python3 -c '
import sys, json
d = json.load(sys.stdin)
for m in d.get("data", []):
    if m.get("model_name") == "qwen3.8-131k":
        print(m.get("model_info", {}).get("id", ""))
        break
')"
  if [ -n "$stale_id" ]; then
    curl -s -X POST "$API/model/delete" \
      -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
      -H 'Content-Type: application/json' \
      -d "{\"id\":\"$stale_id\"}"
    echo
    log "  retired $stale_id"
  else
    log "  qwen3.8-131k present in /v1/models but no /model/info row — nothing to retire"
  fi
fi

for pair in "${ALIASES[@]}"; do
  name="${pair%%:*}"
  window="${pair##*:}"

  if printf '%s\n' "$existing" | grep -qx "$name"; then
    log "alias '$name' already registered — leaving it unchanged."
    continue
  fi

  log "registering pinned alias '$name' (window $window) at $ADMIN_API ..."
  resp="$(curl -s -X POST "$ADMIN_API" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model_name\": \"$name\",
      \"litellm_params\": {
        \"model\": \"openai//model\",
        \"api_base\": \"http://host.docker.internal:8000/v1\",
        \"api_key\": \"sk-no-key-required\"
      },
      \"model_info\": {
        \"max_input_tokens\": $((window - 8192)),
        \"max_output_tokens\": 8192,
        \"input_cost_per_token\": 0,
        \"output_cost_per_token\": 0
        \"fallbacks\": []
      }
    }")"
  echo "$resp"
done

log "verify (qwen3.8* aliases):"
curl -s -X GET "$LIST_API" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  | python3 -c 'import sys,json; [print("  ",m["id"]) for m in json.load(sys.stdin).get("data",[]) if m["id"].startswith("qwen3.8")]' \
  || log "  (could not list models — check LiteLLM is healthy)"

log "done. pi's qwen3.8-65k / qwen3.8-96k now route to the local backend."
log "Reminder (2026-10-02): the GPTQ -mult / -single profiles these aliases were"
log "named for are retired — every qwen3.8* alias now lands on the single :8000"
log "owner (vllm-qwen3.8-exl3, 256K window), so their 64K/96K windows are pi-side"
log "settings, not backend ceilings. Re-run this script after a backend change."
