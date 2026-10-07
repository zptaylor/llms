<!-- Summary: LiteLLM proxy on :4000 - DB model registry, required env keys, Redis cache, client wiring, and notes. -->
<!-- Map: 1-6 proxy, 7-25 model groups, 26-37 NVIDIA rows, 38-46 vLLM quirk, 47-62 model DB, 63-74 setup/keys, 75-90 redis/cache, 91+ client wiring & notes. -->
# LiteLLM proxy (port 4000)

OpenAI-compatible router in front of the local vLLM/llama.cpp servers and
cloud providers. All clients (pi, opencode, hermes-agent, copilot clients)
talk to this proxy; they never hit the engines directly.

**Model names live in the PostgreSQL database, not in YAML**
(`model_list: []` in `config.yaml`; `store_model_in_db: true`). The table
below is the reference set that must be present in the DB; exact
`litellm_params` to seed it are in `TODOS.md` next to this file.

## Model groups (what a client can name)

| Name | Resolves to | Purpose |
|---|---|---|
| `qwen3.8` | local vLLM `/model` :8000 (`vllm-qwen3.8-exl3`) | **pinned** primary local backend. No fallback at all — a down qwen3.8 surfaces the real error instead of silently serving another model |
| `qwen3.8-65k` / `qwen3.8-96k` | same `openai//model` backend :8000 | **pinned aliases for pi**; 65536 / 98304 windows, no fallback (fail loudly). Registered by `register-qwen3.8-aliases.sh`. They named the retired GPTQ `-mult`/`-single` profiles; the backend that owns :8000 serves both unchanged |
| `nex-n2.5-mini` | — | ⏹ **retired 2026-10-02** (`vllm-nex-n2.5-mini-sleep` moved to `containers/ai-host/.archive/`); a stale DB row should be removed with `/model/delete` |
| `b70` | whichever Arc B70 member is awake — today always `qwen3.8` :8000 | **sticky group**: `vram_swap.py` rewrites the request to the member that owns the card before routing. With the pair retired the DB row points at :8000 and the coordinator stays passive (keep `MEMSWAP_ENABLED=false`) |
| `local` | whichever local vLLM serves :8000 | interchangeable alias for local LLMs; **no cloud fallback** (privacy-pinned). Resolves to the `:8000` owner — the EXL3 engine, since the memswap pair was retired 2026-10-02 |
| `HOST-B` | hermes' group: `hermes-3-8b` :8010 → `qwen3.8` :8000 | used by hermes-agent |
| `hermes` | local hermes-agent gateway `http://host.docker.internal:8001/v1` (`model: openai/hermes-agent`) | **pinned** second-brain agent (memory, FTS session store, web research). Registered by `register-hermes.sh`; no fallback in either direction — hermes mediates its own tool calls, so silently answering from another backend would skip the memory layer. Reaches the host because the gateway binds `0.0.0.0:8001` (see `../hermes-agent/docker-compose.yml`) |
| `hermes-3-8b` | llama.cpp on GTX 1010… :8010 (NVIDIA) | hermes' primary backend + `orchestrator` target. (The smaller `hermes-3-3b` that once shared :8010 was removed — it left `/health` hanging when :8010 was down; see `config.yaml` fallbacks.) |
| `orchestrator` | `hermes-3-8b` :8010 | fast local model for routing/subtask handoff |
| `copilot-sonnet-5` (+ `copilot-haiku`, `copilot-gpt-5-mini`) | `github_copilot/claude-sonnet-5` etc. (enterprise va.ghe.com) | frontier **orchestrator tier**: strong model plans/routes, cheap local models do the bulk work. One-time device-flow auth via `register-copilot-orchestrator.sh`. Copilot "Auto" model-select is client-side only, not API-callable |
| `auto` | local vLLM `qwen3.8` :8000 | falls back openrouter → deepseek; on a **context-window-exceeded** it falls to `hermes-3-8b` (see the escape note below) |
| `free` | OpenRouter top-5 free router ($0) | falls back across all `*-free` models below |
| `deepseek` / `deepseek-reasoner` | `deepseek/deepseek-chat` / `-reasoner` | primary DeepSeek (chat) / reasoning (surfaces `reasoning_content`) |
| `openrouter` | `openrouter/qwen/qwen3-coder` | cloud |
| `nv-*` (8 rows) | NVIDIA NIM `https://integrate.api.nvidia.com/v1` | NVIDIA-hosted: `nv-deepseek-v4-flash`/`-pro`, `nv-gpt-oss-20b`, `nv-nemotron-lightning`/`-super`/`-ultra`, `nv-llama-11b-vision`. All 128K ctx; the vision row accepts images. Seeded by `register-nvidia-free.sh` (idempotent; only models returning 200 for the account's key) |

**Every** primary group (`qwen3.8`, `qwen3.8-65k`, `qwen3.8-96k`, `auto`, `b70`,
`local`) carries a `context_window_fallbacks` entry pointing at `hermes-3-8b`, and
the table is authoritative in `../local_models.py` (`CONTEXT_WINDOW_FALLBACKS`) —
the callbacks/compressor derive from it and `scripts/gen-model-group-config.py
--check` guards `config.yaml` against drift.

The entry **must exist for every nameable group**: the router matches by exact
string equality on the group name (`litellm/router.py:6960`), so a group with no
key silently re-raises the HTTP 400 (log tell: `Received Model Group=<name>`).
That was the bug when only `qwen3.8-65k`/`auto` were listed.

There is deliberately **no bigger rung**. `qwen3.8-96k` used to be the escape
target, but it resolves to the *same* `:8000` `api_base` as `qwen3.8`, so
escaping to it re-sends the identical prompt to the identical cap — litellm even
skips it as a duplicate target. And `hermes-3-8b` has a *smaller* window
(40960) than `qwen3.8` (65536), so this chain **cannot actually rescue an
oversized prompt** on this box; it exists so the router fails over instead of
hard-400ing on a missing key. Real oversize protection is pi's `contextWindow`
plus the compression hook.

Keep in mind: a prompt that *fits* but leaves no room for output does not 400 —
it comes back as `finish_reason=length` with ~1 token, which truncates tool
calls — so keep pi's `compaction.reserveTokens` conservative as well.

The `free` router fans out to the `*-free` rows (`glm-5.2-free`,
`gemma-4-31b-free`, `inkling-free`, `nemotron-super-free`, `gpt-oss-20b-free`);
the exact top-5 is refreshed when OpenRouter's free lineup changes (curl the
models API, pick the top-5 by quality, update the DB row).

## Model store (database)

- **Add/update/delete models:** `POST /model/new`, `/model/update`,
  `/model/delete` (or the Admin UI at `/ui`), e.g.
  `curl -X POST http://localhost:4000/model/new -H 'Authorization: Bearer $LITELLM_MASTER_KEY' -H 'Content-Type: application/json' -d '{...}'`.
- All **local-only** groups (`local`, `qwen3.8`, the pinned aliases,
  `nex-n2.5-mini`, `b70`) intentionally have **no cloud fallback** — a local
  request can never be routed to a remote model.
- Fallback routing between groups (e.g. `HOST-B`: hermes → qwen3.8) is DB-managed
  and must stay in sync with `config.yaml` `fallbacks:` + `custom_callbacks.py`
  `_FALLBACK_OVERRIDES`. The value must be a **list**; a mapping there is
  malformed and makes `GET /fallback/<group>` 500 (a stale
  `hermes: {request_timeout: 5}` entry did exactly that — and `hermes` is pinned
  now, so it has no entry at all). `scripts/test-support/test-hermes-agent-wiring.py`
  guards both.

## Setup

```bash
# .env needs: DEEPSEEK_API_KEY (present), OPENROUTER_API_KEY (REQUIRED for
# openrouter / all *-free models — an empty key breaks `auto` fallback with
# 401 "No cookie auth credentials"), MOONSHOT_API_KEY (for kimi-k3-256k)
# NVIDIA_API_KEY — REQUIRED for the nv-* rows (read from env at runtime);
#   after adding it, recreate the container so the key reaches the process:
#   podman-compose up -d litellm && ./register-nvidia-free.sh
podman-compose up -d
```

## Redis cache

A `redis` service (`litellm-redis`) runs in the compose network as LiteLLM's
cache backend (`litellm_settings.cache: true` + `cache_params: {type: redis,
host: redis, port: 6379, db: 0}`). It is internal-only (not published) and
persists AOF data to `./redis-data` (gitignored). The litellm container waits
on the redis healthcheck via `depends_on`.

## Client wiring

- **pi / opencode:** `dotfiles/pi-agent/agent/models.json.tpl`,
  `dotfiles/.config/opencode/opencode.json.tpl` — model ids point at the proxy
  (both also carry a `hermes` id for the group above).
- **hermes-agent:** `containers/ai-host/hermes-agent/hermes-config.yaml` uses
  `HOST-B` (and `orchestrator`). The reverse direction is the `hermes` group
  above: the agent is itself callable *through* the proxy, so clients that are
  not on ai-host can use its memory/session/web-research tools. Its key is the SOPS
  secret `hermes_api_key`, written into `.env` as `HERMES_API_KEY` by
  `homes/USER/credentials.nix` (the row reads `os.environ/HERMES_API_KEY`).
  Re-register with `bash register-hermes.sh` (needs `LITELLM_MASTER_KEY`).
- **Throughput/latency:** `HOST-B/scripts/litellm-tokspeed.sh` — `--live`
  streams to :8000 over Tailscale; `--audit` reads the in-container SQLite DB.

> **vLLM served-name quirk:** the Arc XPU vLLM fork **ignores
> `--served-model-name`** and registers the mount path `/model` as the served
> name (startup log: `served_model_name=/model`). The `qwen3.8` / `local` /
> `auto` rows therefore forward `litellm_params.model: openai//model` — that
> is what the fork accepts. If the image is ever upgraded to one that honors
> the flag, switch those rows back to `openai/qwen3.8`.

> **Guardrails:** `config.yaml` guardrails: (Presidio + regex) mask
> PII/PHI on all traffic and rehydrate it in responses — including inbound
> NVIDIA rows.

Usage:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
```
