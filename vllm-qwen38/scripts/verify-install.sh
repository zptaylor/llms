#!/usr/bin/env bash


set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST_ONLY=false
if [[ "${1:-}" == "--host-only" ]]; then
    HOST_ONLY=true
fi

echo "== humble-b70 verify =="

echo "--- host drivers ---"
xpu-smi --list-gpus 2>/dev/null || echo "xpu-smi not found"
ls -la /run/opengl-driver/lib/libze_intel_gpu.so.* 2>/dev/null | tail -1 || echo "libze_intel_gpu not found"

if $HOST_ONLY; then
    exit 0
fi

echo "--- venv ---"
[[ -d "$HERE/.venv" ]] || { echo "venv not found at $HERE/.venv"; exit 1; }
"$HERE/.venv/bin/python" -c "import torch; print('torch:', torch.__version__); print('xpu:', torch.xpu.is_available()); print('device:', torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'N/A')"

echo "--- ops registration ---"
"$HERE/.venv/bin/python" - <<'PY'
import torch
try:
    import vllm_xpu_kernels._xpu_C
    print('_xpu_C imported')
    ops = ['int8_gemm_w8a8', 'per_token_quant_int8_xpu']
    for op in ops:
        has = hasattr(torch.ops._xpu_C, op)
        print(f'  {op}: ' + ('OK' if has else 'MISSING'))
except Exception as e:
    print(f'vllm_xpu_kernels import failed: {e}')
    exit(1)
PY

echo "--- canary ---"
"$HERE/.venv/bin/python" -c "
import torch
from vllm_xpu_kernels._xpu_C import per_token_quant_int8_xpu
x = torch.randn(4, 4096, device='xpu', dtype=torch.float16)
q, s = per_token_quant_int8_xpu(x)
print('quantization ops working')"

echo "--- smoke generation ---"
"$HERE/.venv/bin/python" -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='$HERE/models/Qwen3.8-27B-Uncensored-int4-AutoRound',
    tensor_parallel_size=1,
    dtype='float16',
    max_model_len=128,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    trust_remote_code=True,
)
outputs = llm.generate(['Hello'], SamplingParams(max_tokens=5, temperature=0))
print('Generated:', outputs[0].outputs[0].text)
print('SMOKE OK')"

echo "== all checks passed =="
