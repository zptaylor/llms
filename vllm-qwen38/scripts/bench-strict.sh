#!/usr/bin/env bash


set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TP=1; MTP=3; INT8=on; CTX=8192; KV=; PORT=19622; GPU=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp) TP="$2"; shift 2;;
        --mtp) MTP="$2"; shift 2;;
        --int8-head) INT8="$2"; shift 2;;
        --ctx) CTX="$2"; shift 2;;
        --kv) KV="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --gpu) GPU="$2"; shift 2;;
        *) echo "unknown: $1"; exit 2;;
    esac
done

MODEL="${MODEL:-$HERE/models/Qwen3.8-27B-Uncensored-int4-AutoRound}"

envs=(TP="$TP" PORT="$PORT" MAX_MODEL_LEN="$CTX"
    SPEC_TOKENS="$MTP" MODEL_DIR="$MODEL")

[[ "$INT8" == "on" ]] && envs+=(VLLM_XPU_LM_HEAD_INT8=1)
[[ -n "$KV" ]] && envs+=(KV_CACHE_DTYPE="$KV")

if [[ "$TP" == "1" ]]; then
    envs+=(ZE_AFFINITY_MASK="$GPU" GPU_MEMORY_UTILIZATION=0.88)
    envs+=(ONEAPI_DEVICE_SELECTOR="level_zero:0")
else
    envs+=(GPU_MEMORY_UTILIZATION=0.95)
fi

envs+=(VLLM_TARGET_DEVICE=xpu VLLM_XPU_ENABLE_XPU_GRAPH=1)
envs+=(CCL_TOPO_P2P_ACCESS=0 CCL_ZE_IPC_EXCHANGE=pidfd CCL_ATL_TRANSPORT=ofi)
envs+=(ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE)
envs+=(VLLM_XPU_HOST_STAGED_COLLECTIVES=1)

exec env "${envs[@]}" \
    "$HERE/.venv/bin/python" "$HERE/scripts/bench-strict.py" \
    --model "$MODEL" \
    --tp "$TP" \
    --mtp "$MTP" \
    --int8-head "$INT8" \
    --ctx "$CTX" \
    ${KV:+--kv "$KV"} \
    --port "$PORT" \
    --gpu "$GPU"
