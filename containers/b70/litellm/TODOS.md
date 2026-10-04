# LiteLLM — open TODOs

Models are DB-managed (`store_model_in_db: true`); seed/row definitions live in
`config.yaml` comments and the `register-*.sh` scripts next to this file, not
here. Historical "add model X" items are done and were removed from this list.

1. **Memswap pair promotion** (tracked in `docs/vllm-memswap.md`): flip
   `MEMSWAP_ENABLED=true` + `autostart_members` in `docker-compose.yml`, remove
   `-mult` from service ordering, and re-run `register-memswap-models.sh` +
   `register-qwen3.8-aliases.sh` once the pair is verified awake-and-swapping.
2. **Invert `HOST-B` fallback order** to `hermes-3-8b → qwen3.8` (currently
   reversed) — see `docs/vllm-xpu-swap-research.md` (qwen3.8's own capacity
   failure under concurrent agents). **Status 2026-10-02 (b70 agent):** repo
   config.yaml fallbacks already declare `HOST-B: [qwen3.8]` (line 25-26); only
   the DB row (`store_model_in_db: true`) still carries the old target —
   flipping it needs the litellm master key + DB access, which the agent
   container doesn't have. One-line fix on b70: `curl -X PATCH
   http://localhost:4000/fallback -H "Authorization: Bearer
   $LITELLM_MASTER_KEY" -d '{"fallbacks": {"HOST-B": ["hermes-3-8b"]}}'` (or
   via the UI).
3. ~~**qwen3.8-96k capacity**: ... with `b70`/`local` sticky semantics the 96k
   fallback can be retired.~~ **DONE**: `qwen3.8-96k` is no longer a
   `context_window_fallbacks` target (it shared `qwen3.8`'s `:8000` api_base, so
   it added no headroom). Every primary group now falls to `hermes-3-8b`; see
   `local_models.py` `ESCAPE_GROUP = None`. Remaining: no local backend has a
   window above 65536, so context-window escalation cannot rescue an oversized
   prompt — protection is pi's `contextWindow` + the compression hook.
4. **NVIDIA NIM free-tier rows**: `register-nvidia-free.sh` only seeds models
   returning 200 for the account key; re-run it when NVIDIA's free lineup
   changes (and update the `free` router's top-5 in the DB to match).
5. ~~**`-sleep` container promotion**~~ — **closed 2026-10-02**: both sleep-pair
   containers were retired into `containers/b70/.archive/` instead of being
   promoted, so there is nothing left to keep in sync. `vllm-qwen3.8-exl3` owns
   `:8000` on its own (see `containers/b70/README.md`).
