"""Single source of truth for the local model groups on the Arc B70.

WHY THIS FILE EXISTS
--------------------
Before this, the same group names were hand-copied into four places and kept
drifting:

  1. config.yaml                          router_settings.{fallbacks,context_window_fallbacks}
  2. the DB row "router_settings"         (overrides #1 at runtime!)
  3. custom_callbacks.py                  _FALLBACK_OVERRIDES / _CONTEXT_WINDOW_*_OVERRIDES
  4. context_compressor.py                a hardcoded model.startswith("qwen3.8")
  5. dotfiles/pi-agent/agent/models.json.tpl   per-model contextWindow

That drift is the bug: the router matches a context_window fallback by EXACT
string equality on the model-group name (litellm/router.py:6960,
`list(item.keys())[0] == model_group`). A group that is missing a key -- or
spelled differently in one of the five files -- silently falls through to
`raise original_exception`, and the client sees a hard HTTP 400. There is no
warning; the only tell is `Received Model Group=<name>` /
`Available Model Group Fallbacks=None` in the router log.

So: declare each group ONCE here, and have the callbacks/compressor read it.
`scripts/gen-model-group-config.py --check` fails if anything drifts.

ADDING A MODEL
--------------
Append one entry to PRIMARY_GROUPS (routes to the :8000 B70 backend) or
SMALL_GROUPS (the little fallback), rebuild the proxy, done. Everything that
matters -- the compressor gate, the fallback maps, the /fallback wiring -- is
derived from this table.

Keep it dependency-free: this module is imported by litellm callbacks inside
the container, which has no repo on sys.path beyond this directory.
"""
from __future__ import annotations

SMALL_MODEL = "hermes-3-8b"

PRIMARY_GROUPS = (
    "qwen3.8",
    "qwen3.8-65k",
    "qwen3.8-96k",
    "auto",
    "b70",
    "local",
)

SMALL_GROUPS = (
    SMALL_MODEL,
    "HOST-B",
)

# The NVIDIA GTX 1070 engine (Ling-3.0-tiny, 8 GB, one slot) --
# containers/ai-host/llama-ling-3-tiny-gguf-nvidia. It is NOT a B70 group and must
# stay out of PRIMARY_GROUPS/COMPRESS_GROUPS: no fallback chain may land a B70
# request on the card, and the compression hook must never fire on a child.
# It is the `delegation` child target (see hermes-agent docs).
NVIDIA_GROUPS = ("ling-3.0-tiny",)

DEFAULT_BACKEND_WINDOW = 262144
BACKEND_WINDOW: dict[str, int] = {
    "qwen3.8-65k": 65536,
    "qwen3.8-96k": 98304,
    # 132K deployed (trained 131072 + --override-kv), not the 262144 default:
    # the child context budget is checked against this number.
    "ling-3.0-tiny": 135168,
}


def backend_window(model: str) -> int:
    """Total token window of the backend a group resolves to."""
    return BACKEND_WINDOW.get(model, DEFAULT_BACKEND_WINDOW)


# What the B70 engine holds *in flight* -- not what one request may be. The pool
# is 301,541 KV tokens at `vllm.gpu_memory_utilization=0.94`, max_model_len
# 262144, batch 4096, MTP k=1 and `max_num_seqs=2` (263,659 at batch 8192).
# Measured: containers/ai-host/vllm-qwen3.8-exl3/README.md, "Measured on ai-host".
# Two sequences are admitted at once, so a client that sizes its window for one
# session can put 2 x window on this pool. The guard in
# scripts/test-support/test-hermes-agent-wiring.py checks that product.
B70_KV_POOL_TOKENS = 301541
B70_MAX_NUM_SEQS = 2


ESCAPE_GROUP: str | None = None

COMPRESS_GROUPS = tuple(g for g in PRIMARY_GROUPS if g != ESCAPE_GROUP)

CONTEXT_WINDOW_FALLBACKS: dict[str, list[str]] = {
    _g: [SMALL_MODEL] for _g in PRIMARY_GROUPS
}

FALLBACKS: dict[str, list[str]] = {
    "HOST-B": ["qwen3.8"],
    "local": [SMALL_MODEL],
}


def is_primary_group(model: object) -> bool:
    """True if `model` names a group served by the primary B70 backend.

    Exact match first (the router's own semantics), then the qwen3.8 family
    prefix so a future `qwen3.8-128k` is covered without editing this table.
    """
    if not isinstance(model, str):
        return False
    if model in NVIDIA_GROUPS:
        return False
    return model in PRIMARY_GROUPS or model.startswith("qwen3.8")


def should_compress(model: object) -> bool:
    """True if the compression pre-call hook should consider this request."""
    if not isinstance(model, str):
        return False
    if model in SMALL_GROUPS or model in NVIDIA_GROUPS:
        return False
    return model in COMPRESS_GROUPS or model.startswith("qwen3.8")
