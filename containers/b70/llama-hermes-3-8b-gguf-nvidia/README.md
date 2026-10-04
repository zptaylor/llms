# Hermes-3-Llama-3.1-8B Q4_K_M (secondary model, NVIDIA GTX 1070 CUDA)

A small, fast **secondary** model for the B70 stack: NousResearch's
[Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
quantized to **Q4_K_M** GGUF and served by llama.cpp's OpenAI-compatible
[`llama-server`](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)
on the NVIDIA GTX 1070 via CUDA. Served model name: **`hermes-3-8b`**.

Use it for cheap side-tasks (routing, classification, short completions,
summaries) instead of the 27B primary — either directly at `:8010/v1` or through
the LiteLLM proxy (port 4000). It replaces the older `hermes-3-3b` (3B) as the
recommended small/secondary model: better quality at a still-small ~4.6 GiB
footprint, and the same 8 GB card can hold it alongside the KV cache when using
quantized (q4_0) KV.

The `-nvidia` suffix on the container name marks it as an **NVIDIA-card
container** (see `containers/b70/README.md`): it runs GPU-accelerated only in
headless mode (`ai-headless`), where the 1070 is not driving a
compositor.

## Why CUDA on the GTX 1070 (and not before)

This container originally ran on the CPU: the GTX 1070 was the
display/compositing GPU (`card1`, X11/KWin via the `legacy_580` driver) with no
container GPU runtime, so a CUDA container wasn't reachable — and compute would
have stolen cycles from the desktop compositor.

With the headless profile (`hosts/b70-host/headless.nix`) that changed:

- The 1070 is no longer bound to a display/compositor (no X, no Wayland, no
  Plasma).
- The proprietary `legacy_580` driver + `nvidia_uvm` are loaded, and
  `nvidia-container-toolkit` generates the CDI spec that rootless podman uses
  to expose the card to containers (`devices: ["nvidia.com/gpu=all"]`).
- `nvidia-persistenced` keeps the GPU warm between requests.

The **Intel Arc B70** remains untouched — it stays fully dedicated to the
primary vLLM model.

## Files

- `docker-compose.yml` — `llama-server` on host port **8010** (shared with
  `llama-qwen3.6-35b-a3b-nvidia`; only one runs at a time on the 8 GB GTX 1070),
  and `--parallel 2` with two **40K** sessions (`--ctx-size 81920` total — the
  measured maximum for this card; 96K OOMs). Pinned
  `ghcr.io/ggml-org/llama.cpp:full-cuda` image digest, **q4_0 KV cache**
  (~28.5 KiB/token — the dense 8B's fp16 KV would not fit next to the ~4.6 GiB
  weights), all layers offloaded (`--n-gpu-layers 99`), `--fit off` so the
  context is never silently shrunk.
- `download-model.sh` — downloads the public GGUF into
  `~/models/Hermes-3-Llama-3.1-8B-GGUF` and verifies its byte size.

## Prerequisites (headless host)

- Host switched to `ai-headless` (NVIDIA driver + CDI: see
  `hosts/b70-host/headless.nix`).
- **Not** for desktop mode: the 1070 drives the display there.

## Service (managed by NixOS)

This container is the declared owner of the shared 8010 port and runs as the
`llama-hermes-3-8b` **systemd user service** (defined in
`hosts/b70-host/configuration.nix`, wanted by `default.target`). It is
started at boot and survives reboots — which is what the `HOST-B`/`qwen3.8`
LiteLLM fallback chains rely on. The service guards on the GGUF being present
and stays inactive (no crash loop) until `download-model.sh` has run.

```bash
systemctl --user status   llama-hermes-3-8b
systemctl --user restart  llama-hermes-3-8b     # e.g. after changing this compose file
journalctl --user -u llama-hermes-3-8b -f
```

The other 1070 servers (`llama-hermes-3-3b`, `llama-qwen3.5-9b`) have NO
service by design — see the "port 8010 is shared" note below.

## Manual quickstart (if the service is stopped)

```bash
cd containers/b70/llama-hermes-3-8b-gguf-nvidia

./download-model.sh            # ~4.6 GiB, public, no token needed
podman-compose up -d           # the service does this at boot; manual = one-off

curl -s http://127.0.0.1:8010/v1/models      # shows "hermes-3-8b"
curl -s http://127.0.0.1:8010/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"hermes-3-8b","messages":[{"role":"user","content":"hi in 3 words"}]}'
```

`download-model.sh` env:

| Var | Default | Purpose |
|---|---|---|
| `MODELS_DIR` | `$HOME/models` | parent dir; GGUF lands in `$MODELS_DIR/Hermes-3-Llama-3.1-8B-GGUF` |
| `HF_REVISION` | *(latest)* | pin an exact HF revision |

## Registering it with LiteLLM

LiteLLM here is **DB-managed** (`store_model_in_db: true`; `model_list` is
empty), so a model is added via the Admin API, not YAML. Example:

```bash
curl -s http://127.0.0.1:4000/model/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{
    "model_name": "hermes-3-8b",
    "litellm_params": {
      "model": "openai/hermes-3-8b",
      "api_base": "http://host.docker.internal:8010/v1",
      "api_key": "sk-no-key-required"
    }
  }'
```

(Adjust `model_name`/routing groups as your LiteLLM conventions require; see
`../litellm/README.md`. The `HOST-B` group — hermes' default — should list
`hermes-3-8b` among its fallbacks instead of the older `hermes-3-3b`.)

## Notes

- Unlike the primary vLLM container, this one does not touch the Arc B70.
  It shares host port **8010** with `llama-qwen3.6-35b-a3b-nvidia`: both bind
  the SAME port so only ONE NVIDIA container can run at a time (the 8 GB GTX 1070
  cannot hold both at their full KV). Stop one before starting the other.
- **Context semantics (important):** `--ctx-size` is the TOTAL context shared
  across all slots — llama.cpp divides it by `--parallel`. The old config
  (`--ctx-size 8192 --parallel 2`) therefore served only **4096 tokens per
  session**, which is exactly the `ContextWindowExceededError (4096)` seen via
  pi/LiteLLM. The current config is `--ctx-size 81920 --parallel 2` → two
  40960-token slots (same 81920 total): hermes uses slot 1 (its compaction
  caps keep it at ~27-30K, well under 40K), and slot 2 stays free for a
  pi/pi-fallback session while the B70 vLLM pod is down. Reverting to
  `--parallel 1` would give one 80K session but removes that concurrent
  fallback capacity.
- **80K is the hard ceiling on this card** (measured, llama.cpp build 10588 and
  latest 10603 give identical results):

  | ctx (`--parallel 1`) | VRAM used / free | result |
  |---|---|---|
  | 32768 | 5849 / 2256 MiB | loads |
  | 81920 | 7817 / 288 MiB | loads; verified serving a 44,462-token prompt |
  | 98304 | — | `cudaMalloc failed: out of memory` |

  The wall is physical VRAM (4.68 GiB weights + ~28.5 KiB/token q4_0 KV), not
  the llama.cpp version — a newer build does not unlock more (tested
  `:full-cuda` build 10603, identical OOM at 96K).
- `--load-mode mlock` pins the ~4.6 GiB model in RAM (`--mlock` is deprecated).
  Note rootless podman may still warn `failed to mlock ... Cannot allocate
  memory` (RLIMIT_MEMLOCK) — that is a harmless soft failure.
- The 3B container (`llama-hermes-3-3b-gguf-nvidia`) remains available as a
  lighter fallback if the 8B ever needs to be stopped; LiteLLM's `HOST-B` group
  prefers the 8B now.
