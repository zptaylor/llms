"""
Context-compression pre-call hook (hermes-qwen.md workflow).

When a chat completion routed to the PRIMARY local model (``qwen3.8`` on the
Intel Arc B70) carries an oversized prompt, this hook summarizes/prunes the
middle of the conversation history using the small, fast Hermes 3 8B model on
the NVIDIA GTX 1070 (host port :8010) BEFORE the request reaches qwen3.8. The
primary then sees a lean payload instead of an ever-growing history, keeping
TTFT low and KV-cache pressure bounded (see hermes-qwen.md items 5-6).

Why a direct HTTP call instead of ``litellm.acompletion(model="hermes-3-8b")``:
this proxy is DB-managed (``store_model_in_db: true``; config.yaml ``model_list``
is empty), so a bare ``litellm.acompletion`` inside a callback has NO access to
the proxy's router and cannot resolve ``hermes-3-8b`` by name. We therefore hit
the llama.cpp backend directly over the host-gateway -- the same ``api_base``
the DB row uses (see ``register-with-litellm.sh``):
    http://host.docker.internal:8010/v1
This also guarantees the compression call never re-enters the proxy's own
callback stack (no recursion).

Every path is defensive: any error leaves the original request untouched, so a
compression failure can NEVER break a real completion.
"""
import os

import httpx
from litellm.integrations.custom_logger import CustomLogger

try:
    from local_models import should_compress as _should_compress
except ImportError:
    def _should_compress(model: object) -> bool:
        return isinstance(model, str) and model.startswith("qwen3.8")

TOKEN_THRESHOLD = int(
    os.environ.get("CONTEXT_COMPRESS_THRESHOLD_TOKENS", "40000")
)
_CHARS_PER_TOKEN = 4

HERMES_API_BASE = os.environ.get(
    "CONTEXT_COMPRESS_HERMES_BASE", "http://host.docker.internal:8010/v1"
)
HERMES_MODEL = os.environ.get("CONTEXT_COMPRESS_HERMES_MODEL", "hermes-3-8b")

_HERMES_HISTORY_CHAR_BUDGET = int(
    os.environ.get("CONTEXT_COMPRESS_HISTORY_CHAR_BUDGET", "60000")
)

COMPRESSION_PROMPT = (
    "You are a technical context summarizer for a coding assistant. "
    "Summarize the conversation history below, keeping: "
    "1. Active file paths and relevant code snippets. "
    "2. Key technical tasks completed so far. "
    "3. Current bugs or remaining goals. "
    "Discard unnecessary tool output, logs, or conversational fluff. "
    "Return ONLY the concise summary, no preamble."
)


def _est_tokens(messages) -> int:
    total = 0
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
    return total // _CHARS_PER_TOKEN


def _cap_history(middle: list, budget: int) -> list:
    """Keep the most recent middle turns within a character budget.

    Older turns are less relevant than recent ones, so we walk backwards and
    keep the newest messages that fit, then restore chronological order.
    """
    kept, total = [], 0
    for m in reversed(middle):
        c = m.get("content")
        n = len(c) if isinstance(c, str) else 0
        kept.append(m)
        total += n
        if total >= budget:
            break
    kept.reverse()
    return kept


class ContextCompressorHandler(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict, cache, data, call_type, **kwargs
    ):
        if call_type != "completion":
            return data
        model = data.get("model")
        if not _should_compress(model):
            return data

        messages = data.get("messages") or []
        if not messages:
            return data

        if _est_tokens(messages) < TOKEN_THRESHOLD:
            return data

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= 1:
            return data

        first = non_system[0]
        last = non_system[-1]
        middle = non_system[1:-1]
        if not middle:
            return data

        middle = _cap_history(middle, _HERMES_HISTORY_CHAR_BUDGET)

        print(
            f"[context_compressor] compressing {len(middle)} middle msgs "
            f"(est {_est_tokens(messages)} tok) via {HERMES_MODEL}"
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{HERMES_API_BASE}/chat/completions",
                    json={
                        "model": HERMES_MODEL,
                        "messages": [
                            {"role": "system", "content": COMPRESSION_PROMPT},
                            {"role": "user", "content": f"History to compress:\n{middle}"},
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.2,
                    },
                )
                resp.raise_for_status()
                summary = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[context_compressor] compression failed, sending original: {e}")
            return data

        new_messages = list(system_msgs)
        if not system_msgs:
            new_messages.append(first)
        new_messages.append(
            {"role": "system", "content": f"[COMPRESSED HISTORY BRIEF]:\n{summary}"}
        )
        if last not in new_messages:
            new_messages.append(last)

        data["messages"] = new_messages
        print(
            f"[context_compressor] done: {len(messages)} -> {len(new_messages)} messages"
        )
        return data


context_compressor_handler = ContextCompressorHandler()
