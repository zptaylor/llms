#!/usr/bin/env bash
set -euo pipefail

LITELLM_PORT="${LITELLM_PORT:-4000}"
VLLM_PORT="${VLLM_PORT:-8000}"
ADMIN_API="http://127.0.0.1:${LITELLM_PORT}/model/new"
MODELS_API="http://127.0.0.1:${LITELLM_PORT}/v1/models"
MODEL_NAME="${MODEL_NAME:-qwen3.8-exl3}"
UPSTREAM_MODEL="${UPSTREAM_MODEL:-/model}"
MAX_TOKENS="${MAX_TOKENS:-65536}"

log() { printf '%s\n' "$*"; }
warn() { printf 'warn: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or put it in containers/ai-host/litellm/.env)"
command -v curl >/dev/null 2>&1 || die "curl required"

backend_ids=""
if backend_ids="$(curl -sf -m 5 "http://127.0.0.1:${VLLM_PORT}/v1/models" \
      | python3 -c 'import sys,json; print(",".join(m["id"] for m in json.load(sys.stdin).get("data",[])))' 2>/dev/null)"; then
  log "backend on :${VLLM_PORT} serves: ${backend_ids:-<none>}"
  case ",${backend_ids}," in
    *",${UPSTREAM_MODEL},"*) : ;;
    *)
      warn "UPSTREAM_MODEL='${UPSTREAM_MODEL}' is NOT in that list."
      warn "Re-run with UPSTREAM_MODEL=<one of the ids above>."
      warn "Two causes, and they need opposite fixes: (a) you are not swapped in"
      warn "to -exl3 yet, so :${VLLM_PORT} is a sibling engine (correct behaviour --"
      warn "swap in first), or (b) this build ignores --served-model-name and"
      warn "serves its mount path (then set UPSTREAM_MODEL to what is printed)."
      ;;
  esac
else
  warn "backend on :${VLLM_PORT} not reachable (is vllm-qwen3.8-exl3 up? it takes ~6-7 min to JIT)"
fi

log "registering proxy id '$MODEL_NAME' -> openai/$UPSTREAM_MODEL on :${VLLM_PORT} ..."
body="$(mktemp)"
code="$(curl -s -o "$body" -w '%{http_code}' -X POST "$ADMIN_API" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model_name\": \"$MODEL_NAME\",
    \"litellm_params\": {
      \"model\": \"openai/$UPSTREAM_MODEL\",
      \"api_base\": \"http://host.docker.internal:${VLLM_PORT}/v1\",
      \"api_key\": \"sk-no-key-required\"
    },
    \"model_info\": {
      \"max_tokens\": ${MAX_TOKENS},
      \"max_input_tokens\": ${MAX_TOKENS},
      \"max_output_tokens\": ${MAX_TOKENS},
      \"input_cost_per_token\": 0,
      \"output_cost_per_token\": 0
    }
  }")" || code=000
cat "$body"; echo
rm -f "$body"

case "$code" in
  2*) : ;;
  000)
    die "could not reach the LiteLLM admin API at $ADMIN_API (is the proxy on :${LITELLM_PORT}?)" ;;
  *)
    warn "registration returned HTTP $code ('already exists' is the usual one)."
    warn "If it already exists, update it with /model/update or drop it first;"
    warn "otherwise check the payload above." ;;
esac

log "verify:"
listed="$(curl -s -X GET "$MODELS_API" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  | python3 -c 'import sys,json; print(",".join(m["id"] for m in json.load(sys.stdin).get("data",[])))' 2>/dev/null || true)"
if [ -z "$listed" ]; then
  warn "could not list models (proxy down, bad key, or non-JSON reply) — check LiteLLM is healthy, then re-run to verify"
else
  case ",${listed}," in
    *",${MODEL_NAME},"*)
      log "  OK: '$MODEL_NAME' is registered."
      log "  proxy ids now: $listed" ;;
    *)
      die "'$MODEL_NAME' is NOT in the proxy model list (got: $listed) — registration did not take effect" ;;
  esac
fi

log "done (see the verify result above)."
log "REMINDER: the pinned qwen3.8 / b70 / local groups reach THIS engine with no"
log "DB change, because the container overrides --served-model-name back to"
log "/model (the id those groups already use). Verified 2026-09-28: all of"
log "qwen3.8, local, b70 and qwen3.8-65k answered through :4000 against it."
log "Rollback is symmetric -- the GPTQ member serves /model too, so no row has"
log "to be rewritten in either direction. See the compose's SERVED ID OVERRIDE"
log "comment before changing that."
log "containers/ai-host/AGENTS.md asks for a new model to also be a fallback inside"
log "the 'local' router; that carve-out does NOT apply to a separate engine like"
log "this one, so it is INTENTIONALLY not done -- it would break the pin that"
log "stops a request landing on an unintended engine (see the carve-out in that"
log "file)."
