

import argparse
import json
import os
import statistics
import time
from typing import List

import torch
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Strict cold benchmark")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--mtp", type=int, default=3, help="MTP speculative tokens")
    parser.add_argument("--int8-head", default="on", help="INT8 lm_head (on/off)")
    parser.add_argument("--ctx", type=int, default=8192, help="Max model len")
    parser.add_argument("--kv", default="", help="KV cache dtype (fp8 or empty)")
    parser.add_argument("--port", type=int, default=19622, help="Port")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for TP1")
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["VLLM_TARGET_DEVICE"] = "xpu"
    os.environ["VLLM_XPU_ENABLE_XPU_GRAPH"] = "1"
    os.environ["CCL_TOPO_P2P_ACCESS"] = "0"
    os.environ["CCL_ZE_IPC_EXCHANGE"] = "pidfd"
    os.environ["CCL_ATL_TRANSPORT"] = "ofi"
    os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"
    os.environ["VLLM_XPU_HOST_STAGED_COLLECTIVES"] = "1"

    if args.tp == 1:
        os.environ["ZE_AFFINITY_MASK"] = str(args.gpu)
        os.environ["GPU_MEMORY_UTILIZATION"] = "0.88"
        os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{args.gpu}"
    else:
        os.environ["GPU_MEMORY_UTILIZATION"] = "0.95"

    if args.int8_head == "on":
        os.environ["VLLM_XPU_LM_HEAD_INT8"] = "1"
    if args.kv:
        os.environ["KV_CACHE_DTYPE"] = args.kv

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="float16",
        max_model_len=args.ctx,
        gpu_memory_utilization=0.88 if args.tp == 1 else 0.95,
        enforce_eager=True,
        trust_remote_code=True,
        speculative_config={
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": args.mtp
        },
        quantization="gptq",
    )

    print("Warming up...")
    _ = llm.generate(
        ["Hello, this is a warmup prompt."],
        SamplingParams(max_tokens=10, temperature=0)
    )

    prompt = "The quick brown fox jumps over the lazy dog. " * 20
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0,
        ignore_eos=True,
    )

    print("Running cold benchmark...")
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end = time.perf_counter()

    total_time = end - start
    num_tokens = len(outputs[0].outputs[0].token_ids)

    decode_rate = num_tokens / total_time

    print(f"Total tokens: {num_tokens}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Decode rate: {decode_rate:.2f} tok/s")

    result = {
        "config": {
            "tp": args.tp,
            "mtp": args.mtp,
            "int8_head": args.int8_head,
            "ctx": args.ctx,
            "kv": args.kv,
        },
        "decode_tok_s": decode_rate,
        "total_tokens": num_tokens,
        "total_time_s": total_time,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()