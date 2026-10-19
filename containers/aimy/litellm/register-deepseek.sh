#!/usr/bin/env bash
# Register the official DeepSeek API models as LiteLLM groups.
#
# The API currently advertises (GET https://api.deepseek.com/models):
#   deepseek-flash     DeepSeek-V4.1-Flash, 1,048,576 ctx, 393,216 max output,
#                      text+image, effort low/high/max
#   deepseek-v4-pro    DeepSeek-V4-Pro,    1,048,576 ctx, 393,216 max output,
#                      text only, effort low/high/max
#
# `deepseek-flash` is the group Reasonix uses; `deepseek-pro` keeps the optional
# Pro provider on the same proxy. The key is resolved by the container from
# DEEPSEEK_API_KEY (docker-compose passes it from this dir's .env).
#
# Idempotent: an existing group with the same name is left unchanged.
set -euo pipefail

LITELLM_PORT="${LITELLM_PORT:-4000}"
ADMIN_API="http://127.0.0.1:${LITELLM_PORT}/model/new"
LIST_API="http://127.0.0.1:${LITELLM_PORT}/v1/models"

log()  { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or source this dir's .env)"
command -v curl >/dev/null 2>&1 || die "curl required"

# name|upstream id|max input|max output|modalities|input $/tok|output $/tok|cache-read $/tok
# Prices: Reasonix list prices are CNY/1M (2/8 flash, 9/27 pro, cache 0.04/0.3);
# converted at ~7.14 CNY/USD. Update here if DeepSeek changes the list price.
declare -a DEEPSEEK_MODELS=(
  "deepseek-flash|deepseek-flash|1048576|393216|text,image|2.8e-07|1.12e-06|5.6e-09"
  "deepseek-pro|deepseek-v4-pro|1048576|393216|text|1.26e-06|3.78e-06|4.2e-08"
)

register_one() {
  local name="$1" upstream="$2" max_in="$3" max_out="$4" modalities="$5"
  local in_cost="$6" out_cost="$7" cache_cost="$8"
  local vision=false body

  case "$modalities" in
    *image*) vision=true ;;
  esac

  if curl -sf -m 15 -H "Authorization: Bearer $LITELLM_MASTER_KEY" "$LIST_API" \
      | grep -q "\"$name\""; then
    log "'$name' already registered — leaving it unchanged."
    return 0
  fi

  log "registering '$name' -> deepseek/$upstream ..."
  body="{
    \"model_name\": \"$name\",
    \"litellm_params\": {
      \"model\": \"deepseek/$upstream\",
      \"api_key\": \"os.environ/DEEPSEEK_API_KEY\"
    },
    \"model_info\": {
      \"max_tokens\": $max_out,
      \"max_input_tokens\": $max_in,
      \"max_output_tokens\": $max_out,
      \"input_cost_per_token\": $in_cost,
      \"output_cost_per_token\": $out_cost,
      \"cache_read_input_token_cost\": $cache_cost,
      \"supports_vision\": $vision,
      \"supports_function_calling\": true,
      \"supports_reasoning\": true,
      \"supports_prompt_caching\": true,
      \"fallbacks\": []
    }
  }"
  curl -s -X POST "$ADMIN_API" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' \
    -d "$body"
  echo
}

for entry in "${DEEPSEEK_MODELS[@]}"; do
  IFS='|' read -r name upstream max_in max_out modalities in_cost out_cost cache_cost <<< "$entry"
  register_one "$name" "$upstream" "$max_in" "$max_out" "$modalities" "$in_cost" "$out_cost" "$cache_cost"
done

log "verify 'deepseek-flash' through the proxy:"
curl -s -m 90 -X POST "http://127.0.0.1:${LITELLM_PORT}/v1/chat/completions" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-flash","messages":[{"role":"user","content":"Reply with exactly: FLASH-OK"}],"max_tokens":32,"temperature":0}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("  ", d["choices"][0]["message"]["content"] if "choices" in d else d)' \
  || warn "  chat completion failed — check podman logs litellm-ai-host | grep deepseek"

log "register-deepseek.sh done."
