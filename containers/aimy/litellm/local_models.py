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

DEFAULT_BACKEND_WINDOW = 262144
BACKEND_WINDOW: dict[str, int] = {
    "qwen3.8-65k": 65536,
    "qwen3.8-96k": 98304,
}


def backend_window(model: str) -> int:
    """Total token window of the backend a group resolves to."""
    return BACKEND_WINDOW.get(model, DEFAULT_BACKEND_WINDOW)


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
    return model in PRIMARY_GROUPS or model.startswith("qwen3.8")


def should_compress(model: object) -> bool:
    """True if the compression pre-call hook should consider this request."""
    if not isinstance(model, str):
        return False
    if model in SMALL_GROUPS:
        return False
    return model in COMPRESS_GROUPS or model.startswith("qwen3.8")
