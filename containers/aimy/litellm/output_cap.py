"""Dynamic per-request output cap for the B70 (:8000) backend.

WHY THIS FILE EXISTS
--------------------
No client-side `max_tokens` is enforced on the `qwen3.8` route, and the engine
does not bound it either, so one runaway generation can hold the whole KV pool
and preempt/re-prefill everything behind it. Measured 2026-10-13: two requests
decoded at ~38 tok/s for hours with `request_success_total` frozen, and the
20-second pressure probe called every tick OK -- a 100K+ token answer that
nobody asked for is indistinguishable from a livelock at the client.

So every request that resolves to the :8000 engine is capped at what the pool
can actually hold, measured at request time:

    ceiling    = min(kv_cache_size_tokens, max_model_len) - prompt_tokens - margin
    available  = min(ceiling, pool - in_use - prompt_tokens - margin)   (only while
                                                           another request runs)
    max_tokens = min(client's own max_tokens, ceiling, available, hard_max) / n

`max_model_len` is a *second, harder* bound than the pool: the pool is routinely
larger than the served window (301,541 vs 262,144 on the promoted engine), and
vLLM rejects any `max_tokens` above the window outright -- measured 2026-10-14,
an injected `max_tokens=287252` came back
`max_tokens=287252 cannot be greater than max_model_len=max_total_tokens=262144`.

`hard_max` exists because the dynamic budget alone is not a cap: for a client
that sends no `max_tokens` it resolves to `window - prompt`, i.e. ~250K output
tokens, and with `max_num_seqs=2` one such generation holds one of the two
sequence slots for hours while every other client queues behind it. Measured
2026-10-15: a single
turn decoded for 78+ minutes and 827 sampled ticks showed requests waiting
behind it -- not a livelock, starvation by a permitted runaway.

Both prompt sources must be counted, or the budget lands over the window and the
engine rejects the request. `tools` is the one that bit: measured 2026-10-15,
the same messages tokenize to 62 tokens alone and 4,159 with a 19,788-byte tools
array -- +4,097 against a 2,048 margin -- so they ride along to `/tokenize`.

Below `FLOOR` the request is refused instead of sent: the engine is saturated,
and a truncated 1-token answer is worse than a clear retry. With
`max_num_seqs=2` the engine can run two sequences, so the `available` term
tightens a request that arrives while a sibling is actually decoding; a request
that is merely queued is not charged for a sibling that will finish before it
runs.

Two things this file deliberately does NOT do:

  * hardcode a pool size. The number comes from the engine's own
    `vllm:cache_config_info{kv_cache_size_tokens=...}`, so swapping the :8000
    engine (or its `max_num_batched_tokens`) needs no edit here.
  * estimate prompt length. Prompts go through the engine's `/tokenize`, the
    same rule the throughput harness follows.

Everything fails OPEN: an unreadable engine or a failed tokenize leaves the
request untouched. The call itself fails with a real error when the engine is
down, and a cap that breaks traffic is worse than a cap that misses a request.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

ENGINE_BASE = os.environ.get(
    "VLLM_ENGINE_BASE", "http://host.docker.internal:8000"
).rstrip("/")
# `/model` is the served id the :8000 compose pins -- every owner of that port
# (abliterated, stock, mixedk, davetha) mounts its checkpoint at /model.
ENGINE_MODEL_ID = os.environ.get("VLLM_ENGINE_MODEL_ID", "/model")

ENABLED = os.environ.get("VLLM_OUTPUT_CAP", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Tokens kept free so the pool never lands exactly on the boundary (a prefill
# in flight, a scheduler chunk, a sampler block that has not been counted yet).
MARGIN = int(os.environ.get("VLLM_OUTPUT_CAP_MARGIN", "2048"))
# Below this the request is refused rather than truncated.
FLOOR = int(os.environ.get("VLLM_OUTPUT_CAP_FLOOR", "1024"))
# Absolute ceiling on any single generation (0 disables it). The dynamic term
# alone permits `window - prompt` output, which for a client that sends no
# max_tokens is a multi-hour generation holding one of the two sequence slots.
HARD_MAX = int(os.environ.get("VLLM_OUTPUT_CAP_MAX", "8192"))
TIMEOUT_S = float(os.environ.get("VLLM_OUTPUT_CAP_TIMEOUT_S", "5"))
STATE_TTL_S = float(os.environ.get("VLLM_OUTPUT_CAP_STATE_TTL_S", "2"))

_state_lock = threading.Lock()
_state_cache: tuple[float, tuple[int, int, float] | None] = (0.0, None)


class Saturated(Exception):
    """The engine cannot serve this request; the caller answers 429."""


def _fetch(url: str, body: bytes | None = None) -> bytes:
    """Default transport. Indirected so the hermetic test never opens a socket."""
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read()


_FETCH = _fetch


def parse_metrics(text: str) -> tuple[int, int, float] | None:
    """`(pool_tokens, used_tokens, running)` from a vLLM `/metrics` body.

    None when the pool size is absent -- an older engine without
    `cache_config_info` means we cannot reason about the budget at all.
    """
    pool: int | None = None
    used_frac: float | None = None
    running = 0.0
    pool_label = 'kv_cache_size_tokens="'
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            i = line.find(pool_label)
            if i < 0:
                continue
            j = line.find('"', i + len(pool_label))
            if j < 0:
                continue
            try:
                pool = int(line[i + len(pool_label) : j])
            except ValueError:
                return None
        elif line.startswith("vllm:kv_cache_usage_perc{"):
            used_frac = _trailing_float(line)
        elif line.startswith("vllm:num_requests_running{"):
            running = _trailing_float(line) or 0.0
    if pool is None:
        return None
    used = int(pool * used_frac) if used_frac else 0
    return pool, used, running


def parse_models(text: str) -> int | None:
    """The served window (`max_model_len`) from a `/v1/models` body."""
    try:
        for entry in json.loads(text).get("data", []):
            value = entry.get("max_model_len") if isinstance(entry, dict) else None
            if value:
                return int(value)
    except Exception:
        return None
    return None


def _trailing_float(line: str) -> float | None:
    try:
        return float(line.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return None


def engine_state(
    *, force: bool = False, base: str = ENGINE_BASE
) -> tuple[int, int, float, int] | None:
    """`(pool, used, running, max_model_len)`, memoized for `STATE_TTL_S`.

    Both reads come from the engine itself: the pool and occupancy from
    `/metrics`, the served window from `/v1/models`. A partial read is no state
    at all -- a budget computed without the window is one the engine rejects.
    """
    global _state_cache
    now = time.monotonic()
    with _state_lock:
        cached_at, cached = _state_cache
        if not force and cached is not None and now - cached_at < STATE_TTL_S:
            return cached
    try:
        state = parse_metrics(_FETCH(f"{base}/metrics").decode("utf-8", "replace"))
        window = parse_models(_FETCH(f"{base}/v1/models").decode("utf-8", "replace"))
    except Exception:
        state, window = None, None
    result = (*state, window) if state and window else None
    with _state_lock:
        _state_cache = (now, result)
    return result


def count_prompt_tokens(data: dict, *, base: str = ENGINE_BASE) -> int | None:
    """Prompt tokens for this request (messages/prompt + tools) via `/tokenize`."""
    body: dict = {"model": ENGINE_MODEL_ID}
    if isinstance(data.get("messages"), list):
        body["messages"] = data["messages"]
    elif data.get("prompt") is not None:
        body["prompt"] = data["prompt"]
    else:
        return None
    # The engine counts the tool schemas as prompt tokens; a budget computed
    # without them overruns the window (see the module docstring).
    if data.get("tools"):
        body["tools"] = data["tools"]
    try:
        out = json.loads(_FETCH(f"{base}/tokenize", json.dumps(body).encode()))
    except Exception:
        return None
    count = out.get("count")
    if count is None and isinstance(out.get("tokens"), list):
        count = len(out["tokens"])
    return int(count) if count is not None else None


def cap_for(
    prompt_tokens: int,
    *,
    pool: int,
    used_tokens: int,
    running: float,
    window: int,
    client_max: object = None,
    choices: object = None,
    margin: int = MARGIN,
    floor: int = FLOOR,
    hard_max: int = HARD_MAX,
) -> int | None:
    """The `max_tokens` to send, or None when the pool cannot serve the prompt.

    Pure: every input is explicit, so the guard test exercises it without an
    engine.
    """
    budget = min(pool, window) - prompt_tokens - margin
    if running >= 1:
        budget = min(budget, pool - used_tokens - prompt_tokens - margin)
    # `n` choices each hold their own sequence, so they share the budget.
    try:
        n = max(1, int(choices))
    except (TypeError, ValueError):
        n = 1
    budget //= n
    try:
        client_wants = int(client_max)
    except (TypeError, ValueError):
        client_wants = None
    if client_wants is not None:
        budget = min(budget, client_wants)
    # After the client's own value: the ceiling bounds an explicit large ask too,
    # which is what makes it a runaway guard rather than a suggestion.
    if hard_max and hard_max > 0:
        budget = min(budget, hard_max)
    return budget if budget >= floor else None


def apply(data: dict) -> int | None:
    """Cap this request in place. Returns the cap, None if untouched.

    Raises `Saturated` when the prompt alone leaves less than `FLOOR` to answer
    with. The caller owns the :8000-only gating and the error translation.
    """
    if not ENABLED or not isinstance(data, dict):
        return None
    state = engine_state()
    if state is None:
        return None
    pool, used, running, window = state
    prompt = count_prompt_tokens(data)
    if prompt is None:
        return None
    cap = cap_for(
        prompt,
        pool=pool,
        used_tokens=used,
        running=running,
        window=window,
        client_max=data.get("max_tokens"),
        choices=data.get("n"),
    )
    if cap is None:
        raise Saturated(
            f"the B70 engine is saturated: {prompt} prompt tokens against a "
            f"{pool}-token KV pool / {window}-token window ({used} in use, "
            f"{running:.0f} running) leaves less than {FLOOR} output tokens. "
            f"Retry when the in-flight turns finish, or send a shorter prompt."
        )
    data["max_tokens"] = cap
    return cap
