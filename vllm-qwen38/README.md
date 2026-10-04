# vLLM Qwen3.8-27B on Intel Arc B70 (NixOS)

This directory contains a NixOS-native setup for running vLLM with Qwen3.8-27B-Uncensored-int4-AutoRound on a single Intel Arc Pro B70 (32 GB VRAM), based on the [humble-b70-llm](https://github.com/JP-devv/humble-b70-llm) reference implementation.

## Overview

The reference implementation (humble-b70-llm) achieves **66.84 tok/s (cold, TP1, INT8 head)** on a single Arc B70 by using:
- Patched vLLM (pinned to commit `8e6d8e4f6a0c84db0d79129ab648492edf640fe2`)
- Patched vllm-xpu-kernels (pinned to commit `27214a3d99c8c4f60f4a15ecb7d3c2febef5bc57`)
- INT8 W8A8 lm_head quantization (load-time)
- MTP3 speculative decoding
- Host-staged collectives (for TP2, not needed for TP1)
- FP16 KV cache (FP8 optional for long context)

This NixOS adaptation provides:
- NixOS module for declarative configuration
- Build scripts adapted for NixOS environment
- Systemd service (system and user variants)
- Automatic model fetching and verification

## Hardware Requirements

- **GPU**: Intel Arc Pro B70 (32 GB VRAM) — only verified GPU
- **RAM**: 64 GB recommended (32 GB works with `MAX_JOBS=3`)
- **Disk**: ~90 GB free (toolchain + 18 GB model + build tree)
- **OS**: NixOS (Ubuntu 26.04 reference; NixOS with recent kernel works)
- **Kernel**: Recent `xe` driver (6.10+ recommended for Battlemage)
- **ReBAR**: Must be enabled in BIOS for full 32 GB VRAM window

## Quickstart

### 1. Add to your NixOS configuration

```nix
# In your configuration.nix or a host-specific file
imports = [
  ./vllm-qwen38/nixos-module.nix
];

services.vllm-qwen38 = {
  enable = true;
  port = 19622;
  tp = 1;                    # Tensor parallel (1 for single GPU)
  mtp = 3;                   # MTP speculative tokens
  int8Head = true;           # Enable INT8 lm_head optimization
  maxModelLen = 8192;        # Max context length
  kvCacheDtype = null;       # Set to "fp8" for long context (>32K)
  gpuIndex = 0;              # GPU index (0 for first B70)
  hfToken = "your-hf-token"; # Required for gated model
};
```

### 2. Rebuild and activate

```bash
sudo nixos-rebuild switch --flake .#your-host
```

### 3. Build the environment (first boot)

The activation script will automatically build the vLLM environment on first boot. You can also build manually:

```bash
cd /home/USER/nixos-config/vllm-qwen38
source /run/current-system/sw/opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
bash scripts/build.sh
```

### 4. Fetch the model

```bash
cd /home/USER/nixos-config/vllm-qwen38
HF_TOKEN="your-hf-token" bash scripts/model.sh fetch
bash scripts/model.sh verify
```

### 5. Start the service

```bash
# System service (runs as USER)
sudo systemctl start vllm-qwen38
sudo systemctl enable vllm-qwen38

# Or user service (run as USER)
systemctl --user start vllm-qwen38
systemctl --user enable vllm-qwen38
```

### 6. Verify installation

```bash
bash scripts/verify-install.sh
bash scripts/quality-gate.sh
```

### 7. Run benchmark

```bash
bash scripts/bench-strict.sh
# Expected: ~66.84 tok/s (cold, TP1, INT8 head)
```

## Directory Structure

```
vllm-qwen38/
├── nixos-module.nix          # NixOS module
├── scripts/
│   ├── setup-host.sh         # Host driver verification
│   ├── build.sh              # Build vLLM + kernels
│   ├── model.sh              # Fetch/verify model
│   ├── serve.sh              # Launch server
│   ├── verify-install.sh     # Verify installation
│   ├── bench-strict.sh       # Cold benchmark
│   ├── bench-strict.py       # Benchmark implementation
│   ├── quality-gate.sh       # Quality gate runner
│   └── quality-gate.py       # Quality gate checks
├── patches/
│   ├── vllm/
│   │   ├── BASE_COMMIT.txt
│   │   └── 0001-humble-b70-production-deltas.patch
│   └── vllm-xpu-kernels/
│       ├── BASE_COMMIT.txt
│       └── 0001-int8-w8a8-lmhead-backport.patch
└── models/                   # Model directory (created at runtime)
```

## Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `VLLM_XPU_LM_HEAD_INT8=1` | Enable INT8 lm_head | on |
| `VLLM_XPU_DRAFT_LM_HEAD_INT4=1` | Enable INT4 draft lm_head | off |
| `VLLM_XPU_HOST_STAGED_COLLECTIVES=1` | Host-staged TP collectives | on |
| `VLLM_XPU_ENABLE_XPU_GRAPH=1` | Enable XPU graphs | on |
| `ZE_AFFINITY_MASK=0` | GPU selection for TP1 | 0 |
| `GPU_MEMORY_UTILIZATION=0.88` | VRAM utilization (TP1) | 0.88 |
| `KV_CACHE_DTYPE=fp8` | FP8 KV cache for long context | off |

## Expected Performance (Cold Workload)

| Config | Decode (tok/s) | Notes |
|--------|----------------|-------|
| TP1, MTP3, INT8 head | **66.84** | Reference cold workload |
| TP1, MTP3, FP16 head | ~57.8 | Without INT8 optimization |
| TP2, MTP3, INT8 head | **94.58** | Requires 2x B70 |

## Troubleshooting

### B70 not enumerated
```bash
xpu-smi --list-gpus
# If empty: check xe driver version, ReBAR enabled, kernel >= 6.10
```

### libstdc++ version mismatch
```bash
# If server crashes at dlopen with GLIBCXX_3.4.x error:
export LD_PRELOAD=/run/current-system/sw/lib/libstdc++.so.6
```

### Build OOM
```bash
# Reduce parallel jobs for lower RAM:
MAX_JOBS=3 bash scripts/build.sh
```

### Model fetch fails
```bash
# Ensure HF_TOKEN is set for gated model:
export HF_TOKEN="your-token"
bash scripts/model.sh fetch
```

## Differences from Reference

1. **NixOS-native**: Uses nixpkgs for toolchain instead of apt/oneAPI installer
2. **Systemd integration**: Both system and user service variants
3. **Activation script**: Auto-builds on first boot
4. **Firewall**: Automatic port opening
5. **Power management**: GPU power cap via systemd service

## Reference

- [humble-b70-llm](https://github.com/JP-devv/humble-b70-llm) — Original implementation
- [vLLM XPU docs](https://docs.vllm.ai/en/latest/getting_started/installation.html#intel-xpu)
- [Intel oneAPI on NixOS](https://nixos.wiki/wiki/Intel_oneAPI)

## License

Apache-2.0. All third-party bits are upstream open-source (vLLM, vllm-xpu-kernels, oneDNN, oneCCL). The model and its derived versions are Apache-2.