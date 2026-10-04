

import sys
import torch


def test_arithmetic_canary():
    """Test the deterministic arithmetic canary: sum(i*i for i in range(4)) == 14"""
    result = sum(i * i for i in range(4))
    assert result == 14, f"Arithmetic canary failed: expected 14, got {result}"
    print("✓ Arithmetic canary passed: sum(i*i for i in range(4)) == 14")
    return True


def test_quantization_ops():
    """Test that quantization ops are registered and working."""
    try:
        from vllm_xpu_kernels._xpu_C import per_token_quant_int8_xpu, int8_gemm_w8a8
    except ImportError as e:
        print(f"✗ vllm_xpu_kernels import failed: {e}")
        return False

    x = torch.randn(4, 4096, device="xpu", dtype=torch.float16)
    q, s = per_token_quant_int8_xpu(x)
    assert q.shape == (4, 4096), f"Wrong quantized shape: {q.shape}"
    assert s.shape == (4,), f"Wrong scale shape: {s.shape}"
    assert q.dtype == torch.int8, f"Wrong quantized dtype: {q.dtype}"
    assert s.dtype == torch.float32, f"Wrong scale dtype: {s.dtype}"
    print("✓ per_token_quant_int8_xpu working")

    A = torch.randn(1, 4096, device="xpu", dtype=torch.int8)
    A_scale = torch.ones(1, device="xpu", dtype=torch.float32)
    B = torch.randn(4096, 256, device="xpu", dtype=torch.int8)
    B_scale = torch.ones(256, device="xpu", dtype=torch.float32)
    out = int8_gemm_w8a8(A, A_scale, B, B_scale, torch.float16, None)
    assert out.shape == (1, 256), f"Wrong GEMM output shape: {out.shape}"
    assert out.dtype == torch.float16, f"Wrong GEMM output dtype: {out.dtype}"
    print("✓ int8_gemm_w8a8 working")

    return True


def test_vllm_import():
    """Test that vLLM imports correctly with patches."""
    try:
        from vllm import LLM, SamplingParams
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
        print("✓ vLLM imports correctly")
        return True
    except Exception as e:
        print(f"✗ vLLM import failed: {e}")
        return False


def test_int8_lm_head_preparation():
    """Test that INT8 lm_head preparation code path is present."""
    try:
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
        import inspect
        source = inspect.getsource(UnquantizedEmbeddingMethod)
        assert "_maybe_prepare_xpu_int8_lm_head" in source, "INT8 lm_head preparation not found"
        assert "_maybe_prepare_xpu_int4_draft_lm_head" in source, "INT4 draft lm_head preparation not found"
        print("✓ INT8/INT4 lm_head preparation code present")
        return True
    except Exception as e:
        print(f"✗ INT8 lm_head preparation check failed: {e}")
        return False


def test_host_staged_collectives():
    """Test that host-staged collectives code is present."""
    try:
        from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
        import inspect
        source = inspect.getsource(XpuCommunicator)
        assert "_staged_all_reduce" in source, "Host-staged all_reduce not found"
        assert "_staged_all_gather" in source, "Host-staged all_gather not found"
        assert "_staged_reduce_scatter" in source, "Host-staged reduce_scatter not found"
        print("✓ Host-staged collectives code present")
        return True
    except Exception as e:
        print(f"✗ Host-staged collectives check failed: {e}")
        return False


def main():
    print("=== humble-b70 Quality Gate ===\n")

    tests = [
        ("Arithmetic Canary", test_arithmetic_canary),
        ("Quantization Ops", test_quantization_ops),
        ("vLLM Import", test_vllm_import),
        ("INT8/INT4 LM Head Preparation", test_int8_lm_head_preparation),
        ("Host-Staged Collectives", test_host_staged_collectives),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")

    if failed > 0:
        sys.exit(1)
    else:
        print("All quality gates passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()