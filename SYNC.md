# llms (public mirror)

Auto-mirrored (redacted) subset of the private `nixos-config` repo: the
Podman containers that serve local LLM inference on an Intel Arc B70, plus
the host-side vLLM build tree.

- Last synced from private commit `8ca1470c3e3d`.
- How: `scripts/sync-public/sync-public-repo.sh` in the private repo runs a
  deterministic redactor (`scripts/sync-public/redact.py`) and pushes only
  when the redacted tree changes (idempotent — a no-op run commits nothing).
- The redactor strips secrets, personal identities, and network addresses of
  the source host. No secrets or private identifiers should be present.

Synced paths:
  - `containers/aimy/vllm-qwen3.8-exl3`
  - `containers/aimy/llama-hermes-3-8b-gguf-nvidia`
  - `containers/aimy/litellm`
  - `containers/aimy/memswap`
  - `vllm-qwen38`
