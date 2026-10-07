#!/usr/bin/env bash
# Register the local hermes-agent gateway as a LiteLLM model group (`hermes`).
#
# The gateway is OpenAI-compatible on :8001 (:v1/models advertises `hermes-agent`
# and it echoes whatever model name it is sent). Its key comes from the SOPS
# secret `hermes_api_key`, written into this dir's .env as HERMES_API_KEY by
# homes/USER/credentials.nix and passed to the container by docker-compose.
#
# PINNED on purpose: no fallback, in LiteLLM *and* in config.yaml's `fallbacks:`
# (hermes mediates its own tool calls and memory; silently answering from a
# different backend would bypass the second brain). Do not add it to another
# group's fallback chain either.
set -euo pipefail

LITELLM_PORT="${LITELLM_PORT:-4000}"
API="http://127.0.0.1:${LITELLM_PORT}"
MODEL_NAME="hermes"
# Resolved by the litellm container (rootless pasta maps host.docker.internal to
# the host; the gateway listens on 0.0.0.0:8001 — see its docker-compose.yml).
HERMES_BASE="${HERMES_BASE:-http://host.docker.internal:8001/v1}"
# The gateway's agent loop needs far more than a chat turn's budget; these mirror
# the window pi advertises for the same backend.
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-57344}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"

log()  { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or put it in the litellm .env)"
command -v curl >/dev/null 2>&1 || die "curl required"

curl -sf -m 5 "http://127.0.0.1:8001/health" >/dev/null \
  || warn "hermes-agent gateway on :8001 not reachable yet (systemctl --user status hermes-agent)"

existing="$(curl -s -X GET "$API/v1/models" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  | python3 -c 'import sys,json; print("\n".join(m["id"] for m in json.load(sys.stdin).get("data",[])))' 2>/dev/null || true)"

if printf '%s\n' "$existing" | grep -qx "$MODEL_NAME"; then
  log "'$MODEL_NAME' already registered — leaving it unchanged."
else
  log "registering pinned group '$MODEL_NAME' -> $HERMES_BASE ..."
  resp="$(curl -s -X POST "$API/model/new" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model_name\": \"$MODEL_NAME\",
      \"litellm_params\": {
        \"model\": \"openai/hermes-agent\",
        \"api_base\": \"$HERMES_BASE\",
        \"api_key\": \"os.environ/HERMES_API_KEY\"
      },
      \"model_info\": {
        \"max_input_tokens\": $MAX_INPUT_TOKENS,
        \"max_output_tokens\": $MAX_OUTPUT_TOKENS,
        \"input_cost_per_token\": 0,
        \"output_cost_per_token\": 0,
        \"fallbacks\": []
      }
    }")"
  echo "$resp"
fi

log "verify (chat completion through the proxy):"
curl -s -X POST "$API/v1/chat/completions" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"hermes","messages":[{"role":"user","content":"Reply with exactly: PROXY-OK"}],"max_tokens":32}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  ", d["choices"][0]["message"]["content"] if "choices" in d else d)' \
  || warn "  (chat completion failed — check podman logs litellm-ai-host | grep hermes)"

log "done. '$MODEL_NAME' is pinned (no fallback); the hermes-agent keeps owning its own tool calls and memory."
