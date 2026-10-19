#!/usr/bin/env bash
# Put every :8000 deployment behind the same bouncer.
#
#   max_parallel_requests = 2   -> a third concurrent call gets a 429 before the
#                                  provider call. The router retries it with
#                                  backoff instead of queueing it inside vLLM.
#   timeout / stream_timeout    -> vLLM never times out a request, so this
#     = 1800                       bounds a dead turn holding one of the engine's
#                                  two sequence slots (30 min; a full-window
#                                  prefill + the 8192-token output cap measures
#                                  ~13 min).
#   num_retries = 2             -> three attempts total. LiteLLM's router owns
#                                  retries; `max_retries` is the provider SDK
#                                  knob and the router pins it to 0.
#
# The :8000 aliases (qwen3.8, qwen3.8-65k, qwen3.8-96k, auto, b70, local,
# qwen3.8-exl3) are separate DB deployments that all resolve to the same
# engine, so applying this to only one of them leaves the others unguarded.
#
#   bash apply-engine-guards.sh          # apply to every :8000 row
#   bash apply-engine-guards.sh --check  # report drift, change nothing
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITELLM_PORT="${LITELLM_PORT:-4000}"
BASE="http://127.0.0.1:${LITELLM_PORT}"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

MAX_PARALLEL_REQUESTS=2
REQUEST_TIMEOUT_S=1800
NUM_RETRIES=2

log() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [ -z "${LITELLM_MASTER_KEY:-}" ] && [ -f "$HERE/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$HERE/.env"
  set +a
fi
[ -n "${LITELLM_MASTER_KEY:-}" ] || die "LITELLM_MASTER_KEY not set (export it or put it in this dir's .env)"
command -v curl >/dev/null 2>&1 || die "curl required"
command -v python3 >/dev/null 2>&1 || die "python3 required"

read_rows() {
  curl -sS -f "$BASE/model/info" -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    | python3 -c '
import json, sys

def norm(v):
    if v is None:
        return ""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return str(v)

for d in json.load(sys.stdin).get("data", []):
    p = d.get("litellm_params") or {}
    if ":8000" not in str(p.get("api_base", "")):
        continue
    print("\t".join([
        d.get("model_name", ""),
        (d.get("model_info") or {}).get("id", ""),
        norm(p.get("max_parallel_requests")),
        norm(p.get("timeout")),
        norm(p.get("stream_timeout")),
        norm(p.get("num_retries")),
    ]))
'
}

has_guards() {
  [ "$1" = "$MAX_PARALLEL_REQUESTS" ] \
    && [ "$2" = "$REQUEST_TIMEOUT_S" ] \
    && [ "$3" = "$REQUEST_TIMEOUT_S" ] \
    && [ "$4" = "$NUM_RETRIES" ]
}

rows="$(read_rows)" || die "cannot read $BASE/model/info"
[ -n "$rows" ] || die "no deployments targeting :8000 found"

drift=0
while IFS=$'\t' read -r name id mpr timeout stream_timeout retries; do
  [ -n "$name" ] || continue
  if has_guards "$mpr" "$timeout" "$stream_timeout" "$retries"; then
    log "ok      $name"
    continue
  fi

  drift=1
  if [ "$CHECK" -eq 1 ]; then
    log "drift   $name (max_parallel_requests=$mpr timeout=$timeout stream_timeout=$stream_timeout num_retries=$retries)"
    continue
  fi

  [ -n "$id" ] || die "$name has no model id"
  code="$(curl -sS -o /tmp/litellm-guard-patch.json -w '%{http_code}' \
    -X PATCH "$BASE/model/$id/update" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' \
    -d "{\"litellm_params\":{\"max_parallel_requests\":$MAX_PARALLEL_REQUESTS,\"timeout\":$REQUEST_TIMEOUT_S,\"stream_timeout\":$REQUEST_TIMEOUT_S,\"num_retries\":$NUM_RETRIES}}")" \
    || die "$name: PATCH failed"
  [ "$code" = "200" ] || die "$name: PATCH returned HTTP $code ($(head -c 300 /tmp/litellm-guard-patch.json))"
  log "patched $name"
done <<< "$rows"

if [ "$CHECK" -eq 1 ]; then
  if [ "$drift" -ne 0 ]; then
    log "drift: run 'bash $HERE/apply-engine-guards.sh' to apply the guards"
    exit 1
  fi
  log "ok: every :8000 deployment carries the guards"
  exit 0
fi

# Re-read instead of trusting the PATCH responses.
rows="$(read_rows)" || die "cannot re-read $BASE/model/info after patching"
while IFS=$'\t' read -r name _id mpr timeout stream_timeout retries; do
  [ -n "$name" ] || continue
  has_guards "$mpr" "$timeout" "$stream_timeout" "$retries" \
    || die "$name still shows max_parallel_requests=$mpr timeout=$timeout stream_timeout=$stream_timeout num_retries=$retries"
done <<< "$rows"

log "ok: every :8000 deployment carries the guards"
log ""
log "verify:"
log "  bash $HERE/apply-engine-guards.sh --check"
log "  curl -s $BASE/router/settings -H 'Authorization: Bearer \$LITELLM_MASTER_KEY' | python3 -m json.tool"
