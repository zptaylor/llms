#!/usr/bin/env bash
set -euo pipefail

LITELLM_PORT="${LITELLM_PORT:-4000}"
ADMIN_API="http://127.0.0.1:${LITELLM_PORT}/model/new"
LIST_API="http://127.0.0.1:${LITELLM_PORT}/v1/models"
API_BASE="https://integrate.api.nvidia.com/v1"

log() { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or set it in the litellm .env)"
command -v curl >/dev/null 2>&1 || die "curl required"

declare -a NVIDIA_MODELS=(
  "nv-deepseek-v4-flash|deepseek-ai/deepseek-v4-flash-0731|text"
  "nv-deepseek-v4-pro|deepseek-ai/deepseek-v4-pro-0813|text"
  "nv-gpt-oss-20b|openai/gpt-oss-20b|text"
  "nv-nemotron-lightning|nvidia/nemotron-3.5-lightning-30b-a3b|text"
  "nv-nemotron-super|nvidia/nemotron-3-super-120b-a12b|text"
  "nv-nemotron-ultra|nvidia/nemotron-3-ultra-550b-a55b|text"
  "nv-llama-11b-vision|meta/llama-3.2-11b-vision-instruct|text,image"
)

register_one() {
  local name="$1" upstream_id="$2" input_types="$3" body
  log "registering '$name' -> '$upstream_id' ..."
  input_types_json="[\"text\"]"
  [ "$input_types" = "text" ] || input_types_json="[\"text\",\"image\"]"
  body="{
    \"model_name\": \"$name\",
    \"litellm_params\": {
      \"model\": \"openai/$upstream_id\",
      \"api_base\": \"$API_BASE\",
      \"api_key\": \"os.environ/NVIDIA_API_KEY\"
    },
    \"model_info\": {
      \"max_tokens\": 32768,
      \"max_input_tokens\": 131072,
      \"max_output_tokens\": 32768,
      \"input_cost_per_token\": 0,
      \"output_cost_per_token\": 0
    }
  }"
  if curl -sf -m 15 -H "Authorization: Bearer $LITELLM_MASTER_KEY" "$LIST_API" \
      | grep -q "\"$name\""; then
    log "  already registered ('$name' present) — skipping."
    return 0
  fi
  curl -s -X POST "$ADMIN_API" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' \
    -d "$body"
  echo
}

[ -n "${NVIDIA_API_KEY:-}" ] && \
  warn "NVIDIA_API_KEY exported locally — verifying the backend is reachable." || true
curl -sf -m 10 "$API_BASE/models" -H "Authorization: Bearer ${NVIDIA_API_KEY:-os.environ/NVIDIA_API_KEY}" >/dev/null \
  || warn "NVIDIA endpoint ($API_BASE) not reachable with local NVIDIA_API_KEY (is containers/b70/litellm/.env set?)"

for entry in "${NVIDIA_MODELS[@]}"; do
  IFS='|' read -r name upstream input_types <<< "$entry"
  register_one "$name" "$upstream" "$input_types"
done

log "register-nvidia-free.sh done."
