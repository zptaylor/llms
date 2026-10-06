<!-- Summary: vllm-qwen3.8-exl3 — the EXL3 (exl3xpu engine) route for Qwen3.8-27B on one Arc B70: why it is a separate engine, the pins, profiles, how to start it (unit ownership), memswap membership, measured throughput, traps. -->
<!-- Map: 1-14 what/why, 15-27 STATUS, 28-45 It is a different engine, not another flag-set, 46-59 The two pins (both verified to exist), 60-98 Profiles, 99-138 Launch, 139-158 What the recipe adds beyond the GPTQ siblings, 159-221 Traps already priced in, 222-260 Starting it — and the unit-ownership gotcha, 261-295 Surviving a reboot, 296-328 Pair membership (memswap), 329-360 Measured on b70-host, 361-485 Speculative decoding: OFF (2026-09-30) then ON at k=1, 486-525 Structured output: the Failed-to-advance-FSM line, 526-581 Structured output: json_schema is broken (open upstream), 582-620 Verification state, 621-712 Auto-mode tool-call leaks, 713-732 Rollback, 733-742 Sibling docs. -->
# vllm-qwen3.8-exl3 — Qwen3.8-27B EXL3 4.00 bpw on one B70 (host port 8000)

**The `:8000` backend.** This started as a *second* route to Qwen3.8-27B on the
Arc B70, alongside the GPTQ-Int4 containers (`../vllm-qwen3.8-mult/` and
siblings). On 2026-10-02 it stopped being an alternative: the whole GPTQ family,
the sleep-mode pair and `-heretic` were **retired into
[`../.archive/`](../.archive/)**, so this engine owns host `:8000` on its own
(see "Pair membership" below).

Recipe: `SergiioB/intel-arc-pro-b70-inference-cookbook`,
`docs/qwen38-27b/EXL3-XPU.md` @ `530ae03`.

## STATUS — live on `b70-host` since 2026-09-28

Running as the `:8000` backend, serving model id **`/model`** at the model's
native **262144** window, verified through the proxy for `qwen3.8`, `local`,
`b70`, `qwen3.8-65k`, `auto` and `qwen3.8-exl3`. **Tool calling verified**
(32/32 assertions, raw and through the proxy). **MTP speculative decoding is ON
at k=1** since 2026-10-02 (it was off from 2026-09-30 — see that section for the
KV pool it costs and the +59 % it buys): `GPU KV cache size: 303,056 tokens`, so
one full 262,144-token session still fits (1.16×). Boot **~6.5 min cold / ~2 min
warm**. It is **boot-enabled** — see "Surviving a reboot". Measured throughput
and the surprises found on the way: "Measured on b70-host" + "Traps" below. Rollback
is one command — see "Rollback".

## It is a different engine, not another flag-set

| | GPTQ siblings (`-mult`, …) | this container |
|---|---|---|
| Image | `vllm/vllm-openai-xpu` nightly | **`ghcr.io/0xsero/exl3xpu`** (digest-pinned) |
| Entrypoint | `vllm serve …` (flags) | `python3 scripts/serve.py <config> …` |
| Weights | `zrlu/SuperQwen3.8-27b-…GPTQ-Int4…` | **`turboderp/Qwen3.8-27B-exl3`** |
| Quant | GPTQ-Int4 + MTP4 + local patch scripts | **EXL3 trellis** 4.00 bpw, native ESIMD |
| Serving config | the `command:` in the compose | `models/qwen3.8-27b-exl3-4.00bpw/model.yaml` **inside the image** |
| Extra flags | free-form vLLM flags | **`--set KEY.PATH=VALUE`** into `model.yaml` (a bare vLLM flag is forwarded to `vllm serve` *after* the config, so it duplicates rather than overrides — see "Traps") |

`exl3xpu` (upstream: <https://github.com/0xSero/exl3xpu>) is a **vLLM XPU
plugin** that decodes EXL3 ([exllamav3](https://github.com/turboderp-org/exllamav3)
trellis quantization) with native ESIMD kernels on Battlemage; vLLM still
supplies scheduling, paged KV, GDN/attention kernels and the OpenAI API. It
needs **no patch scripts** — the whole GPTQ `patches/` machinery is absent here
on purpose.

## The two pins (both verified to exist)

- **Image:** `ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8`
  — vLLM `0.26.1.dev0+g568afb3a1`, ~32 GB on disk, attested build. Verify with
  `gh attestation verify oci://ghcr.io/0xsero/exl3xpu@sha256:21412bdd7535e9c653eeb3d099dce3bc79a83d440def2cd9556111c79c870fa8 -o 0xSero`.
- **Weights:** `turboderp/Qwen3.8-27B-exl3` rev
  `113cf7ab958054860e43fb7f3063b1af19171095` (branch `4.00bpw`) — 2 shards,
  **15.73 GiB**, public/ungated. `./download-model.sh` pins this exact revision.

EXL3 is **kernel-specific**: the engine decodes only the `mul1` codebook at
4/6 bpw. A newer revision of the repo is therefore *not* automatically
servable — do not bump `HF_REVISION` without re-checking upstream's
`model.yaml`.

## Profiles (two knobs, both `--set`)

(The compose also carries two non-profile `--set` overrides — `served_model_name`
and `max_num_batched_tokens` — each with its own reasoning comment there.)

| Profile | `vllm.gpu_memory_utilization` | `vllm.max_model_len` | Notes |
|---|---|---|---|
| **live default** | `0.94` | `262144` | The model's full native window, and what is deployed. Needs a **clean** card (the swap script removes the incumbent first). Recipe's proven boundary: a 261,920-token prompt answered correctly, fp8 KV pool 287,040 tokens — **reproduced here exactly** (`GPU KV cache size: 287,040 tokens`, max concurrency 1.09x). |
| co-tenant-safe | `0.90` | `65536` | The recipe's served default, for a card that is NOT clean. At 0.90 the KV budget cannot cover 262144 (measured: 9.1 GiB available vs 9.29 GiB needed, engine reports max 256000). |

**The window and the batch budget are coupled.** 262144 fits at 0.94 only
because `max_num_batched_tokens` is back at the recipe's **4096**: raising it to
8192 inflates the *vision encoder cache* enough that the GPTQ-budget check fails
again (measured: 9.29 GiB needed vs 9.1 GiB available). If you raise one, expect
to lower the other. And note that at 0.94 the KV budget has ~0.2 GiB of slack, so
the engine sits close to the line — at 0.90 the same failure appears immediately.

Re-measured 2026-10-02 **with MTP k=1** (`scripts/vllm-exl3-ab.sh apply batch8192`):
8192 *does* fit once the multimodal profiling budget is shrunk —
`--set vllm.limit_mm_per_prompt={"image":1,"video":1}`, and that key works (the
boot log then reads `'limit_mm_per_prompt': {'image': 1, 'video': 1}`) — but only
just: `GPU KV cache size: 265,174 tokens` = **1.01×** the 262,144 window, against
303,056 at 4096 + the recipe's 32/4. That is −37,882 tokens of pool for a knob
that buys prefill/activation budget only, and it leaves the prefix cache (72.5 %
hit on agent transcripts) nothing to work with — so it is **not** the config to
land. The batch-token cost dominates: the shrunk mm budget only partly offsets it.

`model.yaml` ships `0.965` + `262144`; **`0.965` OOMs** — measured startup-free
on this host family is ~29.5/31.9 GiB, so keep `≤ 0.94`. That is a different
rule from the `0.82` ceiling documented for the GPTQ siblings in
`../README.md`: that ceiling exists because *their* recipe's cudagraph capture
("decode, FULL") OOMs with a warm `torch.compile` cache. This engine's
`model.yaml` uses `cudagraph_mode: FULL_DECODE_ONLY` with an explicit
`cudagraph_capture_sizes` list, so the two numbers are not interchangeable.

Note that `max_model_len` does not resize the KV pool — it caps a single
sequence. Lowering it does not buy headroom, so do not reach for it as an OOM
workaround.

## Launch

**The card holds one engine — free it first.** Starting this on a card that
already runs a sibling fails to bind `:8000`, or worse, leaves two engines
fighting for the same 32 GB.

1. **As a memswap member (this is how the trial runs it).** `vllm-swap-restart.sh`
   stops the coordinator, removes whichever `vllm-*` is awake, verifies the card
   is actually free, cold-creates this member and waits for `/health`:

   ```bash
   systemctl --user stop vllm-memswap
   HEALTH_WAIT_S=1800 bash scripts/vllm-swap-restart.sh qwen3.8-exl3
   systemctl --user start vllm-memswap
   ```

   It needs `podman` on PATH, so run it from a login shell — and read "Starting
   it" below before doing the equivalent by hand, because a container started
   inside a short-lived unit gets **reaped**.

2. **As a plain profile swap** (if the coordinator is not in play):
   `HEALTH_WAIT_S=1200 bash scripts/swap-vllm-container.sh vllm-qwen3.8-exl3`.
   That script's health wait defaults to **600 s**, against a measured cold boot
   of **6.5 min (390 s)** — only ~3 min of margin, and none if the card is busy.

Weights and image:

```bash
./download-model.sh                      # ~15.7 GiB, pinned revision
podman pull ghcr.io/0xsero/exl3xpu@sha256:21412bdd…870fa8   # ~7.25 GiB compressed
```

Disk: **~40 GB** total (image 23.7 GB on disk as measured, + 15.7 GiB of
weights; the cookbook quotes ~32 GB for the image alone). `b70-host`'s `/` is a
1.9 TB disk that was at 97 % during this bring-up — check `df -h /` first, and
note the weights land in `~/models` (same filesystem). Pull with the *rootless*
store (the default here); under a containerd-backed daemon instead, force
overlay2 or use a big-disk `--data-root`, or the pull lands in
`/var/lib/containerd`.

## What the recipe adds beyond the GPTQ siblings

Measured by the cookbook against the GPTQ/W4A16 arms on the same card:

- **Correctness:** 6/6 on an exact-answer battery — the only clean Qwen3.8 arm
  on that host (the serving GPTQ artifact answered `30` where the answer is
  `14`).
- **Smallest resident footprint:** 15.7 GiB of weights ⇒ ~76K *more* fp8-KV
  tokens than an equal-fp8 AutoRound export of the same model.
- **Native 262,144 context on one card**, prefix cache 5.4× on a 10.5K shared
  prefix.
- **Hardest power-cap scaling measured** (+33–42 % at 230 W, paired medians).

Anti-hype, from the same table: it is **not faster** than the GPTQ champion at
decode (both ≈58 tok/s client wall at p~569/g128, ≈24.7 at p~8223), and the
`~400 GB/s` / `91 % lm_head` figures floating around are different cells on the
author's stack — the effective decode bandwidth here is ~240 GB/s @150 W →
~355 GB/s @230 W. The EXL3 decode numbers are **MTP chunk-rates** (accepted
steps), so they undercount generated tokens; compare only within that campaign.

## Traps already priced in

- **`serve.py` takes a config positional, not a vLLM flag list.** Overrides go
  through `--set KEY.PATH=VALUE` (what this container uses). The recipe says
  bare vLLM flags are *rejected*; upstream `scripts/serve.py` actually uses
  `parse_known_args` and forwards unrecognised args **verbatim, after** the
  `model.yaml` flags — so a bare `--max-model-len` wins by duplication instead
  of showing up as a config change. Prefer `--set`: it is the difference
  between "the config says 64K" and "something appended 4K and nobody can see
  where".
- **fp8 is the only KV dtype.** `bf16` KV is rejected at capture warmup.
- **Engine env lives in `model.yaml`, not in the compose.** `EXL3_INT8_PREFILL`,
  `EXL3_ONEDNN_ATTN`, `EXL3_KV_BLOCK_EXACT`, `EXL3_DRAFT_VOCAB`,
  `VLLM_XPU_ENABLE_XPU_GRAPH` are all set there and **override** any
  `-e`/`environment:` given at container level. To change one, use `--set` or a
  new image revision — editing `environment:` here will look correct and do
  nothing.
- **`/dev/dri/by-path` must be mounted** (it is): oneCCL enumerates the device
  through it. Upstream also passes `--group-add $(stat -c '%g' /dev/dri/renderD128)`;
  the recipe does, the sibling composes in this repo do not and work, so it is
  omitted here — add it under `group_add:` if the engine cannot open the render
  node.
- **`shm_size` is `8g`, not upstream's `32g`.** Upstream passes `--shm-size 32g`
  even on its single-card run, so this *is* a deliberate deviation, not a
  passive default: it is this repo's sibling convention, and on a 62 GiB host a
  32 GiB tmpfs is a real memory claim competing with the engine's own RAM use.
  Raise it if the engine ever reports a `/dev/shm` shortfall — but do it knowing
  you are reclaiming that memory from the host, not from the card.
- **Prefix caching is ON** (`model.yaml`: `enable_prefix_caching: true`, and it
  must be explicit because vLLM 0.26 leaves it off for hybrid GDN models). Do
  not benchmark cold cells without the entropy guard. Reuse is **block-aligned**,
  measured 2026-10-02: a ~960-token shared prefix got
  `prefix_cache_hits_total` **+0**, a ~6.1K-token one got **+3200 per call**
  (2 × the 1600-token block) — short shared prompts silently reuse nothing.
  Correctness with the cache hot: 6/6 shared-prefix cases retrieved a fact
  planted inside the *cached* prefix, flat repetition stats, and 800-token
  generations that ran to the cap with no mid-generation collapse. So the
  `compressed-tensors`/W4A16 repetition degeneration recorded in
  [`../.archive/vllm-qwen3.8-27b-uncensored-autoround/README.md`](../.archive/vllm-qwen3.8-27b-uncensored-autoround/README.md)
  is *not* a prefix-caching defect, and does not transfer to this route — there
  the cause was the export's calibrated FP8 KV scales.
- **Under spec decode, `min_p` and `logit_bias` are silently ignored** (vLLM
  logs it at boot). That is the one client-visible degradation lever on this
  route: a caller relying on either for repetition control gets nothing. Affects
  every quant route, not just this one.
- **`--reasoning-parser qwen3`, never `--reasoning-format deepseek`** — the
  latter crash-loops this image family (see `../../../AGENTS.md`).
- **The served id is overridden back to `/model`.** Measured: unlike the
  mainline XPU fork this repo assumes, **this engine honours
  `--served-model-name`** — it served `qwen3.8-27b-exl3` and 404'd `/model`. The
  compose therefore sets `--set served_model_name=/model` so the pinned LiteLLM
  groups (all `openai//model` @ `:8000`) keep working as drop-ins, and so
  rollback needs no DB rewrite. See the compose's `SERVED ID OVERRIDE` comment
  before changing it.
- **Two cache volumes exist so a *recreated* member starts warm.** Without
  `vllm-exl3-vllm-cache:/root/.cache/vllm` every `podman rm -f` + recreate
  re-pays torch.compile (measured 142.8 s of the boot; 6.5 min total cold vs
  **~2 min warm**). `podman start` of an existing container already keeps its
  writable layer — the volume is what survives a *recreate*.
- **One engine at a time.** `restart: "no"` is deliberate: this container shares
  `:8000` with three siblings and the memswap pair, so it must never auto-start
  and steal the card from the GPU-lease coordinator.

## Starting it — and the unit-ownership gotcha

**Start it from a systemd unit, not from an interactive shell**, or it dies the
moment your shell does. The repo's containers are declared as user units with
`Type=oneshot` + `RemainAfterExit=true` (see `hosts/b70-host/configuration.nix`),
which is exactly what keeps their cgroup alive after `ExecStart` returns.

Measured failure mode: starting it with a *transient* unit that exits reaps the
container — `systemd-run --user --unit=X bash -c 'podman-compose up -d'` logs
`SIGTERM` to the engine seconds after it first answers `/health`, because the
default `KillMode=control-group` kills everything left in the unit's cgroup, and
the container lives there. Symptom: `/v1/models` answers 200, then the engine
drains and dies, and the unit sits in `deactivating`.

The working shape on this host:

```bash
# 1) free :8000 (the coordinator must not race the recreation)
systemctl --user stop vllm-memswap
# 2) hand the card over -- removes the incumbent and cold-creates this member
HEALTH_WAIT_S=1800 bash scripts/vllm-swap-restart.sh qwen3.8-exl3
# 3) bring the lease owner back
systemctl --user start vllm-memswap
```

`vllm-swap-restart.sh` drives `podman`/`podman-compose` directly, so run it
somewhere those work (a normal login shell on the b70, or a unit). Doing the
equivalent by hand needs a unit that outlives the command:

```bash
systemd-run --user --unit=vllm-qwen3.8-exl3 --collect \
  --property=Type=oneshot --property=RemainAfterExit=yes \
  bash -c 'cd <repo>/containers/b70/vllm-qwen3.8-exl3 && podman-compose -f docker-compose.yml up -d'
```

Making it a **boot-enabled** backend (so it comes up by itself like `hermes` /
`glance`) is what `systemd.user.services.vllm-qwen38-exl3` in
`hosts/b70-host/configuration.nix` does — see "Surviving a reboot" below.

## Surviving a reboot

Declared in `hosts/b70-host/configuration.nix` as
`systemd.user.services.vllm-qwen38-exl3` (`Type=oneshot`,
`RemainAfterExit=true`, `wantedBy = default.target`, `ExecStartPre` guards the
checkpoint, `TimeoutStartSec=900` for the cold JIT boot). It is the **only**
boot-enabled `vllm-*` backend in this repo — every other one is deliberately
started by hand, because only one engine can own the card and *which* one is a
choice. This unit makes the choice explicit: **after a reboot EXL3 owns
`:8000`.**

Enable/disable and verify:

```bash
systemctl --user status vllm-qwen38-exl3                        # after a rebuild
systemctl --user disable --now vllm-qwen38-exl3                 # hand the slot back
bash scripts/vllm-swap-restart.sh qwen3.8                       # ...to the GPTQ member
```

Two things worth knowing:

- **Mutual exclusion.** The GPTQ members bind the same `:8000` and need the whole
  card. While this unit is enabled they cannot start; disable it first (the
  command above) and then bring the GPTQ member up explicitly.
- **The unit name has no dot** (`vllm-qwen38-exl3`, not `vllm-qwen3.8-exl3`):
  Nix would read the dotted form as attribute selection, and it matches the
  project name podman-compose already uses for this compose file
  (`podman-compose@vllm-qwen38-exl3.service`, network
  `vllm-qwen38-exl3_default`). This is also why the *container* is deliberately
  started under a unit of that name rather than by hand.

Applying it needs a rebuild (the declaration only takes effect after
`sudo nixos-rebuild switch --flake .#b70-host`); until then the container keeps
running only because it was started explicitly.

## Pair membership (memswap)

Registered in `../memswap/members.json` as member **`qwen3.8-exl3`** (aliases
`exl3`, `exl3xpu`, `qwen3.8-27b-exl3`). It used to share `:8000` with a
`qwen3.8` member (the `vllm-qwen3.8-sleep` container) — **that pair was retired
on 2026-10-02** and both containers moved to `../.archive/`, leaving this the
registry's only member. `default_model` now points here.

It is a **sleep-less member**, deliberately. The XPU sleep pool never booted on
this card (that is why the `-sleep` container was itself effectively sleep-less —
see `../../../docs/vllm-memswap.md`), so a member that cannot be parked is the
normal case here, not an exception. What that means:

- `container_sleep_capable()` in `scripts/vllm-memswapd.py` detects that the
  container command has no `--enable-sleep-mode` and **refuses to POST
  `/sleep`** — correct, because on this engine a `/sleep` POST would hang its
  core.
- `VLLM_SERVER_DEV_MODE` is deliberately **not** set. Dev mode is what mounts
  `/sleep`, and exposing it on a sleep-less engine is a foot-gun guarded only by
  the coordinator. Cost: `/is_sleeping` is absent (404).
- **Reporting artifact to know about:** with no `/is_sleeping`, the coordinator
  counts this member as `"sleeping": null`, so `GET :8002/status` reports
  `"awake": null` and `"all_asleep": true` **while the engine is in fact awake
  and serving**. It stays passive for those ticks rather than guessing, and
  `ensure` fails closed (refuses) — which is why the hook must stay off:
- **Do NOT set `MEMSWAP_ENABLED=true`** in `../litellm/docker-compose.yml` while
  this member owns the card. The pre-call hook would call `/ensure`, which fails
  closed on an unreadable member and would refuse every `qwen3.8`/`local`
  request. `MEMSWAP_ENABLED=false` is the live setting and is correct.

Swapping between the two members is a restart, not a sleep/wake:
`scripts/vllm-swap-restart.sh qwen3.8-exl3` ↔ `... qwen3.8`.

## Measured on `b70-host` (2026-09-28, one B70)

Cold boot **6.5 min**; warm (cache volume populated) **~2 min**. Two identical
consecutive boots measured 6.5 → 2 min, which is the whole point of the
`vllm-exl3-vllm-cache` volume below.

Greedy, `temperature 0`, streaming, `max_tokens` as noted:

| Cell | Result |
|---|---|
| `HOST-B/scripts/litellm-tokspeed.sh --live` (its prompt: *"List the prime numbers one per line."*, 400 tok) | **105.7 tok/s**, TTFT 0.09 s |
| ~600-token prose prompt, 400 tok | **66.4 tok/s** |
| ~600-token code prompt, 400 tok | **72.6 tok/s** |
| ~8K-token prompt | TTFT 1.0 s warm / **5.1 s cold**, then 76.8 tok/s |
| ~32K-token prompt | TTFT 1.0 s warm / **6.4 s cold**, then 72.2 tok/s |
| ~97K-token **unique** prompt ×3 concurrent (2026-10-03, MTP off, idle engine) | TTFT **67.6 / 134.6 / 201.7 s** — prefills are serialized (~1.45k tok/s), `Waiting` 2→1→0, 0 preemptions |
| Aggregate decode, C1 / C2 / C4 / C8 | **103 / 191 / 327 / 486 tok/s** |

Read this honestly:

- The repo's own benchmark cell is close to **best case**: a ~12-token prompt
  asking for a highly repetitive list is near-ideal for MTP3 speculative decode,
  so 105 tok/s is the ceiling, not the typical. The realistic-prompt cells
  (66–73 tok/s) are the comparable ones, and they line up with the cookbook's
  own "implied post-first ~65–71" for this route — i.e. **no faster than the
  GPTQ champion at decode**, exactly as the recipe says.
- Aggregate scaling is real and matches the recipe's shape (they report 365
  tok/s at C16 on prose-with-thinking; this prompt is easier).
- The warm TTFT figures are **prefix-cache hits** (`enable_prefix_caching: true`
  is on and the repeated filler prompt is cached). The cold numbers are the
  prefill cost.

## Speculative decoding: OFF on 2026-09-30 → **ON at k=1 since 2026-10-02**

The recipe's `model.yaml` enables MTP
(`speculative_config: {method: qwen3_5_mtp, num_speculative_tokens: 3}`). The
k=3 shape ran from 2026-09-28, was switched off on 2026-09-30 for the KV and
acceptance reasons below, and is back on at **k=1** as of 2026-10-02 — the k>1
re-run penalty is the part that did not survive contact with this workload.

**How the override works.** `scripts/serve.py:46-49` emits one flag per key of
the `vllm` block, and `--set` values go through `yaml.safe_load`, so a *key path*
(`vllm.speculative_config.num_speculative_tokens=1`) edits the dict that
`model.yaml` already defines and `method: qwen3_5_mtp` survives. The previous
setting was a whole-value `vllm.speculative_config=null`, which drops
`--speculative-config` entirely. Verified with `serve.py --print`: the flag
becomes `--speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":1}`
with every other argument byte-identical.

**Why k=3 was turned off (2026-09-30).** It cost on both axes, on exactly the
workload this box serves:

1. **KV pool.** This engine reported `kv_cache_size_tokens=292898` with
   `kv_cache_max_concurrency=1.1173`; requests queued (`Waiting: 1-2` for ~2 h)
   and `num_preemptions_total` reached 53. (The recipe's own note claims ~40 % of
   the pool — `docs/orcasaq2-27b-exl3-feasibility.md:125-127`.)
2. **Acceptance collapses where it matters.** Lifetime acceptance was healthy
   (282,630 / 684,672 = **41.3 %**), but in a long single-session decode the 10 s
   windows read `Accepted: 0, Drafted: 558`, mean acceptance length 1.00. The
   startup log names the cause: `exl3xpu: align-mode accepted-token sync skipped
   except on block crossings / batch changes` — with `block_size 1600` a stable
   batch crosses a block only every 1600 tokens, so the MTP head's state drifts
   and the drafts get rejected. vLLM also warns that `num_speculative_tokens > 1`
   re-runs the same MTP layer, so every extra draft is paid for at full price.

**What k=1 costs (measured 2026-10-02, from the boot log).** The price is the MTP
head's *weights*, not per-token KV — which is why k does not buy the pool back:

| | MTP off | **k=1 (live)** | k=3 (2026-09-30) |
|---|---|---|---|
| weights | 15.78 GiB | **17.34 GiB** (+1.56) | — |
| KV memory | 11.95 GiB | **10.42 GiB** | — |
| `GPU KV cache size` | 376,253 (1.44×) | **303,056 (1.16×)** | 292,898 (1.12×) |

Per-token KV stayed ~32 KiB in all three configurations, so the pool difference is
budget, not bytes-per-token: k=1 lands only **3.5 %** above the old k=3 pool while
paying for a third of the drafts. vLLM also logs `Add 3 padding layers, may waste
at most 6.25% KV cache memory`, and the draft head shares the target's embedding
and lm_head weights with the recipe's pruned draft vocab (`MTP draft head uses 512
of 1940 vocab blocks`). **303,056 tokens still holds one full 262,144-token
session** (1.16×) — that was the requirement — but a second long session has less
headroom than the MTP-off pool, so watch `Waiting` / `num_preemptions_total` when
2–3 agent sessions overlap.

**Measured 2026-10-03: that warning is the bug that shipped.** The gateway held
three long sessions at once (a 20-hour CLI session at 143–146k prompt tokens, the
30-min `oci-fleet-nixos` job at ~98–103k, `daily-local-model-trends` at ~56–72k —
demand ≈318k tokens against a 303,056 pool). The engine froze in
`Running: 1, Waiting: 2` all `reason="capacity"` with `Avg prompt throughput: 0.0
tokens/s` for minutes and `num_preemptions_total` climbing 137→139: the waiting
requests got *no* prefill, so hermes killed each stream at the end of its client
budget, re-sent the same prompt, and killed it again (the CLI session logged one
API call at `latency=2925.2s` and repeated `Stream stale for 900s … Killing
connection`). A clean-engine probe with three concurrent *unique* ~97k-token
prompts shows the second half of the problem — even with headroom, **serialized
prefill**: TTFT 67.6 / 134.6 / 201.7s at ~1.45k tok/s, `Waiting` 2→1→0, 0
preemptions. So depth *and* per-turn prefill time both count, which is why the fix
landed on the client side (`hermes model.context_length: 131072` → compacts at
96,000) rather than on this engine. Re-tested the pool lever the same day via
`scripts/vllm-exl3-ab.sh apply mtpoff`: `GPU KV cache size: 376,253` and all three
requests admitted, at the cost of **33.4 tok/s** C1 (`bench 3 300`) against 49.84
for k=1 — reverted, so k=1 stays the landed config.
**The harness gate cannot see any of this:** `gate` only checks that *one*
262,144-token session is resident, which 303,056 tokens (1.16×) satisfies — the
gate passes while the concurrent-agent case it is meant to protect is starving.

**What k=1 buys (measured 2026-10-02 with `bench/sweep.py`, synthetic greedy
prose).** Read the C1 row like-for-like: the MTP-off number is the engine's own
quiet-window (`Running: 1`) rate, because the harness cells below it were taken
while two agent streams were resident:

| Cell | MTP off | **k=1** |
|---|---|---|
| decode C1, per stream | 31.4 tok/s (engine window) | **49.84 tok/s** |
| acceptance length | — | **1.773** (lifetime 74 %: 9,644 / 13,037) |
| decode C2, per stream | 18.73 tok/s (contended) | 31.88 tok/s |
| TTFT C1 / C2 | 788 / 836 ms (contended) | 128 / 310 ms |
| cold prefill 4K / 32K | 1517 / 1474 (contended) | **2785 / 2447 tok/s** |

**+59 % per-stream at C1**, which is 89 % of the theoretical gain from a 1.773
acceptance length. The prefill rows are the clean measurement this repo was
missing and they now *beat* the recipe's reference cells (2421 / 2259), which
confirms the contended numbers were bandwidth sharing rather than slow kernels.
Aggregate under live load went from 53–76 tok/s to 66–114 tok/s at `Running: 3`,
and the new engine's cumulative per-stream ITL is 43.9 ms (22.8 tok/s) against the
old engine's 64.6 ms — different lifetimes, so treat that pair as directional.

**Caveats k=1 introduces** (all from the boot log):

- `min_p and logit_bias parameters won't work with speculative decoding` — a
  client that sets either now loses it silently.
- `max_num_scheduled_tokens is set to 4096 based on the speculative decoding
  settings… consider increasing max_num_batched_tokens` — draft slots want more
  than 4096, but that knob is coupled to the window (see "Profiles").
- `method 'qwen3_5_mtp' is deprecated and replaced with 'mtp'` — cosmetic; vLLM
  resolves the alias to `mtp` internally.
- Grammar × spec-decode was re-checked rather than assumed: an auto tool call
  returned `get_weather({"city":"Quito"})` with `finish_reason: tool_calls`,
  **0** `Failed to advance FSM` lines and 0 terminating-grammar errors.

**Rollback** (one line, then recreate — the compose is the source of truth). Put
the line back to `vllm.speculative_config=null`, then:

```bash
systemctl --user stop  vllm-memswap
systemctl --user restart vllm-qwen38-exl3
systemctl --user start vllm-memswap
```

`scripts/vllm-swap-restart.sh qwen3.8-exl3` is **not** the tool for a same-member
config change: it exits early with "already the awake member" when the target
container is the one running, so a recreate has to go through the unit (which is
also what keeps the container from being reaped — see "Starting it"). The
recipe's `EXL3_DRAFT_VOCAB` env is live again now that a drafter exists; it only
ever shaped the pruned draft head.


## Structured output: what the `Failed to advance FSM` log line means

If you see this in the engine log it is **expected and benign** — don't chase it:

```
(EngineCore pid=109) ERROR 09-28 19:12:16 [backend_xgrammar.py:162]
  Failed to advance FSM for request chatcmpl-... for tokens 271. Please file an issue.
```

**What it is.** `271` is a *token id*, not a position — in this checkpoint's
tokenizer id 271 decodes to `'\n\n'`. It comes from
`v1/structured_output/backend_xgrammar.py` → `accept_tokens()`, which returns
False when the grammar matcher rejects a token as vLLM advances the FSM over a
speculative-decode step. When reasoning ends **mid-step**, that step still
contains reasoning tokens (the `\n\n`), which are not grammar content.

**Why it is benign here.** vLLM has two reactions to a rejected token:
`v1/structured_output/__init__.py` *tolerates* it in the post-reasoning-end
window (the path that logs the line above), while
`v1/core/sched/scheduler.py` logs `"Unexpected: grammar rejected tokens …
Terminating request."` and fails the request. Measured on this deployment:
**0** occurrences of the terminating variant, **0** `AssertionError`, and forced
`tool_choice` stayed **6/6 well-formed** in both configurations (20/20 across all
runs). It is the residual of upstream
[#44006](https://github.com/vllm-project/vllm/issues/44006) ("speculative
decoding + strict tool calling failed to advance FSM"), closed via PR #44297; our
build contains that fix (the code cites `#44006` in
`trim_reasoning_for_advance` and carries the `post_reasoning_end_in_window`
tolerance). Upstream's own follow-up says the line *still* appears on v0.28.0
with MTP + `--reasoning-parser qwen3` + async scheduling + a grammar, but the
request is no longer terminated — i.e. exactly this configuration.

**Trigger: prefix caching × a grammar.** Recreated with
`--set vllm.enable_prefix_caching=false`: 12 grammar requests then produced
**0** FSM errors, against ~16 with prefix caching on. Prefix caching is left
**on** regardless (the recipe's setting, and the right one for agents' shared
growing transcripts — a 64.6% prefix hit rate was observed), because the line is
benign and silencing it would cost that reuse. The exact `--set` to trade the
other way is recorded in the compose comment above `--`.

## Structured output: `response_format: json_schema` IS broken (open upstream)

Separate from the log line, and **not** benign — but also **not live** today:

| Path (raw `:8000`, uncached, temperature 0) | Result |
|---|---|
| `response_format: {"type":"json_schema", …}` | **0/6 valid** (2/8 in an earlier distinct-prompt run) |
| forced `tool_choice` (tool calling) | **6/6 valid** (20/20 across all runs) |

The corruption is a **doubled leading brace**, e.g.
`{{"from":"Quito","to":"Suva","passengers":4}` — unparseable JSON. It is
produced by the **engine, not LiteLLM** (raw `:8000` and the proxy `:4000`
returned byte-identical malformed bodies), and it is **independent of prefix
caching** (still 0/6 with it disabled), so it is not a local misconfiguration. It
matches an **OPEN** upstream bug,
[#38696](https://github.com/vllm-project/vllm/issues/38696) — *"[Bug]: qwen3.5
when enable `response_format` json_schema outputs garbled spaces"* — same
qwen3_coder / qwen3-reasoning setup and model family.

**Why it needs no local fix yet:** nothing in this repo sends
`response_format` / `json_schema` / `guided_json` — not pi, not opencode, not
hermes, not the LiteLLM callbacks (grepped: zero hits). Agents use **tool
calling**, which is verified clean. So:

- **Use forced tool calling** (`tools` + `tool_choice`) if you need structured
  output from this backend. Avoid `response_format: json_schema` until #38696 is
  fixed upstream.
- Watch #38696 rather than patching locally. If a harness ever genuinely needs
  json_schema, the plausible knob is xgrammar's `disable_any_whitespace` compile
  option (the issue title blames xgrammar compiling JSON schemas with *unbounded
  whitespace*) — **untested** on this image.

Reproduce (~4 min):

```bash
python3 - <<'PY'
import json, urllib.request
p = {"model": "/model",
     "messages": [{"role": "user", "content": "Book a flight from Quito to Suva for 4 passengers."}],
     "response_format": {"type": "json_schema", "json_schema": {"name": "trip", "schema": {
        "type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"},
        "passengers": {"type": "integer"}}, "required": ["from", "to", "passengers"],
        "additionalProperties": False}}},
     "max_tokens": 2500, "temperature": 0}
for i in range(6):
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(p).encode(), headers={"Content-Type": "application/json"})
    b = json.loads(urllib.request.urlopen(r, timeout=600).read().decode())
    c = (b["choices"][0]["message"].get("content") or "").strip()
    try:
        json.loads(c); print(i, "ok")
    except Exception:
        print(i, "INVALID", repr(c[:40]))
PY
```

## Verification state

**Tool calling: VERIFIED (2026-09-28).** 32/32 assertions on the raw engine and
through the proxy for `qwen3.8`/`local`/`b70`: a forced
`tool_choice` returns a proper `tool_calls` array (`get_weather`, arguments
`{"city":"Paris"}`, valid JSON), **no XML leaks into `content`**, a returned tool
result chains into the right final answer, and reasoning stays in
`reasoning_content` rather than polluting the message.
**No template override is needed** — this was the trap I expected from the GPTQ
siblings, and it does not bite: the checkpoint's own `chat_template.jinja`
already carries the qwen3 `<tool_call>`/`<function=` scaffolding (5 of each, the
same shape as the retired GPTQ siblings'
[`../.archive/vllm-qwen3.8-mult/chat_template.jinja`](../.archive/vllm-qwen3.8-mult/chat_template.jinja)),
so `qwen3_coder` parses it as-is.

**Structured output:** `response_format: json_schema` is broken on this engine
and `tool_choice` is not — see the two sections above before wiring any harness
that wants machine-readable output.

Still **not** verified:

- **Image/video inputs.** The checkpoint has a vision tower and the recipe
  enables 32 images / 4 videos per prompt (and these are what size the encoder
  cache above); untested here.
- **Long-context quality: verified at ~100K, not at 262K.** A needle planted in
  the first line is retrieved correctly from a **100,111-token** prompt
  (varied filler, 3,771 distinct entries; 118 s wall), and the chain accepts far
  more than the old 65K cap — a 141,203-token prompt went through too. An
  earlier probe that appeared to fail was my own fault: it repeated one sentence
  ~3000 times, which the recipe warns collapses MTP acceptance, and it produced
  the degenerate answer you would expect from such input. So: 100K is solid,
  262144 is servable, and the region above ~140K is still unproven.

Registration is already done (`qwen3.8-exl3` row exists in the proxy DB):

```bash
./register-with-litellm.sh        # idempotent; needs LITELLM_MASTER_KEY
```

## Auto-mode tool-call leaks — the "dead turn" (measured 2026-09-29)

`--tool-call-parser qwen3_coder` does **not** always convert the model's
tool-call markup. When it loses one the response is still HTTP 200 with
`finish_reason: "stop"` and **no** `tool_calls`; the client (pi) sees an
ordinary text answer, has no tool to run and no error to retry, so it **ends the
turn** — the user must type "resume" to continue. The repo's own name for it is
a *malformed / leaked tool call* (`../litellm/custom_callbacks.py:62-80`).

Shape (pi session `01a0eae4`, 2026-09-29 13:36:24Z) — the engine consumed
`<tool_call>\n<function=bash>` and returned only the remainder as `content`, so
the leak **starts mid-tag**:

```
<parameter=command>
for d in /nix/store/*-source; do … done
</parameter>
<parameter=timeout>
120
</parameter>
</function>
</tool_call>
```

That mid-tag start is the whole reason this came back after the 2026-09-28
switch to this engine: with the previous `hermes` parser the parse died one step
earlier, so the leak still contained `<tool_call` — which is exactly what
`custom_callbacks.py`'s detector matches (`"<tool_call" in content`). Under
`qwen3_coder` that test is now **false**, so the corrective retry never fires.

**What the model actually emits** (recovered with `return_token_ids: true` +
`POST /detokenize`): the Hermes/Qwen-XML form
`<tool_call><function=NAME><parameter=K>VALUE</parameter></function></tool_call>`
-- and *never* JSON inside `<tool_call>`, which is what the retry note asks for.
Measured 30/30 `xml-function` (see below). A retry-with-note does produce a
clean call (`finish_reason=tool_calls`, arguments parse), so a resample is a
sound recovery; only the detector is broken.

### Reproducing / measuring it — and the parser A/B

`scripts/toolcall-leak-probe.py` fires N `tool_choice: auto` requests at
pi-sampling (temp 1.0, `enable_thinking`) and classifies each response
(`ok` / `bad-args` / `leak` / `no-call`), optionally recovering the raw text:

```bash
python3 scripts/toolcall-leak-probe.py --count 30 --raw   # baseline, live :8000
python3 scripts/toolcall-leak-probe.py --count 300 --max-leak-rate 0.01   # gate
```

`qwen3_coder` arm, 2026-09-29, clean card: **0 leaks / 30**, 0 `bad-args`, 30/30
`xml-function`. That is *not* a clean bill of health — the same backend leaked 1
turn in 282 on the live ~50K-token agent transcript, and every known leak was on
a long tool-heavy context, so a short-context probe under-samples it. The
`hermes` arm therefore needs the long-context replay (or several hundred
samples) before it can be judged.

Both arms are single-tenant (the card and `:8000`), so recreate, probe, revert.
`vllm-swap-restart.sh:78` is **not** the tool for this — it exits early with
"already the awake member" for the one member that already owns the card — so the
recreate goes through the unit:

```bash
# scripted one lever at a time: edits the compose, recreates via the unit, gates the
# 262,144-token pool, restores the pre-A/B compose on `revert`
bash scripts/vllm-exl3-ab.sh apply toolxml    # only qwen3_xml is scripted; `hermes` by hand
bash scripts/vllm-exl3-ab.sh probe 300        # the leak rate is this variant's gate
bash scripts/vllm-exl3-ab.sh revert

# or by hand — flip the command list's last `--` pair, then:
systemctl --user stop vllm-memswap
systemctl --user restart vllm-qwen38-exl3     # ~6 min cold, ~2 min warm
systemctl --user start vllm-memswap
python3 scripts/toolcall-leak-probe.py --count 300 --max-leak-rate 0.01
# then flip the arg back and recreate again before anything else uses :8000
```

An unknown parser name fails at engine *startup*, so confirm it is in the image
before flipping (`qwen3_xml` is **not** verified on this image — the arm measured
below was `hermes`):

```bash
podman run --rm --entrypoint python3 \
  "$(sed -n 's/^ *image: *//p' containers/b70/vllm-qwen3.8-exl3/docker-compose.yml)" -c \
  "import vllm.entrypoints.openai.tool_parsers as p,os;print(sorted(os.listdir(p.__path__[0])))"
```

Standing mitigations (independent of which parser wins): the malformed-tool-call
detector in `../litellm/custom_callbacks.py` (marks the tail shape as malformed
so the corrective retry fires) and the pi-side auto-nudge extension
(`dotfiles/pi-agent/agent/extensions/leaked-toolcall-nudge.js`) that re-sends
the corrective note so no human has to type "resume".

## Rollback

**There is no longer a second engine to hand the card back to.** The GPTQ member
this section used to swap to (`vllm-qwen3.8-sleep`, via
`scripts/vllm-swap-restart.sh qwen3.8`) was retired on 2026-10-02 and moved to
`../.archive/`, and that script exits early for a same-member target anyway.
Rolling back now means editing this container's `docker-compose.yml` back to the
previous flag set (or reverting the commit), then recreating it in place:

```bash
systemctl --user stop vllm-memswap
systemctl --user restart vllm-qwen38-exl3
systemctl --user start vllm-memswap
```

No LiteLLM change is needed in either direction: the pinned groups
(`qwen3.8`, `local`, `b70`, `auto`, `qwen3.8-65k`, `qwen3.8-96k`) are all
`openai//model` @ `:8000`, and this engine serves the id `/model` thanks to the
`served_model_name` override.

## Sibling docs

- The **retired** dual `-mult`/`-sleep` stack and why its ceiling was `0.82`:
  [`../README.md`](../README.md), [`../MODELS.md`](../MODELS.md) — the containers
  themselves are under [`../.archive/`](../.archive/).
- Router/registration rules for a new backend: [`../AGENTS.md`](../AGENTS.md).
- Why the B70 previously could not serve **any** EXL3 checkpoint, what of that
  verdict still holds (mixed codebooks, third-party kernels), and the remaining
  re-check:
  [`../../../docs/orcasaq2-27b-exl3-feasibility.md`](../../../docs/orcasaq2-27b-exl3-feasibility.md).
