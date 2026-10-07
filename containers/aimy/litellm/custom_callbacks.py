"""
LiteLLM custom / call hooks: prepend the SERVED-model tag to agent responses.

WHY (annotated for future agents):
----------------------------------
pi / opencode / hermes (WhatsApp) all talk to this LiteLLM proxy. They usually
ask for a *routed* MODEL GROUP, e.g. ``auto`` or ``free``. The router then
picks ONE backend deployment to serve the request. The response comes back with
``response.model`` = the *actual upstream deployment* (e.g.
``openrouter/...:free`` or ``openai//model``), while the client only ever
told LiteLLM the group name (``auto`` / ``free``). So the client can't see which
model actually answered.

This hook prepends a small ``[served: <deployment>]`` to the first assistant
message, uniformly, so every client (pi / opencode / hermes) sees who served
WITHOUT each client needing bespoke logic (the originally-discussed "Setup 2").

NOTES / CONSTRAINTS:
- ``data["model"]``   = the model GROUP the client requested (e.g. ``auto``).
- ``response.model``  = the real deployment served (post-router resolution).
- Only prepend when served != requested group, so exact/direct calls (e.g.
  ``qwen3.8``) are left clean.
- Every hook is defensive: any error is swallowed, so a callback can NEVER break
  a real completion.

LITELLM WIRING (config.yaml) -- the DOCUMENTED pattern:
-----------------------------------------------------
Use ``litellm_settings.callbacks`` pointing at a MODULE PATH that resolves to an
INSTANCE of this CustomLogger subclass. The docs warn the proxy ONLY dispatches
CustomLogger *instances* (pointing at the bare class silently doesn't run the
hooks). So this module creates one at import time:
    proxy_handler_instance = AsyncPrependServedModel()
and config.yaml has:
    litellm_settings:
      callbacks: custom_callbacks.proxy_handler_instance

Hook used:
- ``async_pre_call_hook(...)``  -> runs BEFORE the LLM call (streaming or not)
  and can MODIFY the outgoing request. Used to normalize the message list so the
  local qwen3.8 chat template never sees a ``system`` message that is not
  first (its tokenizer raises "System message must be at the beginning").
- ``async_post_call_success_hook(data, user_api_key_dict, response)``  ->
  proxy-only, runs after a NON-STREAMING successful call and can MODIFY + return
  the outgoing response. This is the hook that actually rewrites what the client
  receives.
- ``async_post_call_streaming_iterator_hook(...)``  -> rewrites the FIRST stream
  chunk for STREAMING responses (pi / opencode / hermes stream by default).
"""
import contextvars
import json
import re

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse, ModelResponseStream
from typing import Any, AsyncGenerator, Optional

import local_models as _local_models

_MAX_TOOL_RETRIES = 3
_tool_retry_ctx: contextvars.ContextVar[int] = contextvars.ContextVar(
    "litellm_tool_retry_count", default=0
)

_LEAKED_TAIL_RE = re.compile(
    r"</(?:tool_call|function|parameter)>\s*$|<(?:parameter|function)=[^<>]*>\s*$"
)


class AsyncPrependServedModel(CustomLogger):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _normalize_system_messages(messages: list) -> list:
        """
        Qwen3's tokenizer chat template REJECTS a request whose ``system``
        message is not the FIRST message ("System message must be at the
        beginning"). pi / opencode / hermes can occasionally emit a ``system``
        message mid-list (e.g. after tool results or when reconstructing
        history), which makes the local vLLM backend hard-fail.

        This rewrites the list so every ``system`` message is merged into a
        single ``system`` message at position 0, preserving the order of all
        other (user / assistant / tool) messages. It is safe for every
        OpenAI-compatible backend (all expect system-first) and is a no-op when
        the list already begins with a system message and contains no others.
        """
        if not messages or messages[0].get("role") == "system":
            if not any(
                m.get("role") == "system" for m in messages[1:]
            ):
                return messages

        head, tail = [], []
        for m in messages:
            if m.get("role") == "system":
                head.append(m.get("content", ""))
            else:
                tail.append(m)
        if head:
            tail.insert(0, {"role": "system", "content": "\n\n".join(head)})
        return tail

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
        **kwargs,
    ) -> dict:
        """
        Runs for every chat/completion request before it is sent to a backend.
        Reorder/merge ``system`` messages to the front so the Qwen3 tokenizer
        never rejects the payload. Defensive: any error leaves the request
        untouched (never break a live request).

        NOTE: ``**kwargs`` is intentional. The proxy image is a rolling
        ``main-latest`` tag and LiteLLM has changed the arguments it passes to
        pre-call hooks across releases (notably the ``cache`` keyword). Keeping
        ``**kwargs`` here absorbs any extra keyword the running build forwards
        so a hook-contract change can never crash the proxy with a
        "got an unexpected keyword argument" error.
        """
        try:
            if data and isinstance(data.get("messages"), list):
                data["messages"] = self._normalize_system_messages(
                    data["messages"]
                )
        except Exception:
            pass
        return data

    @staticmethod
    def _served_tag(requested: str, served: str) -> Optional[str]:
        """Return the prepend string, or None if there is nothing to annotate.

        NOTE: The ``[served: <model>]`` prefix is intentionally DISABLED (always
        returns None). Users found the tag noisy in agent responses. If you ever
        want it back, restore the logic below:
            requested = (requested or "").strip()
            served = (served or "").strip()
            if not served or served == requested:
                return None
            return f"[served: {served}] "
        """
        return None

    @staticmethod
    def _mark_pii_entities(text: str) -> str:
        """Return a copy of ``text`` with any PII entities wrapped in a shield emoji."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            try:
                nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
                analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
            except Exception:
                analyzer = AnalyzerEngine()

            results = analyzer.analyze(text=text, language="en")
            if not results:
                return text

            results = sorted(results, key=lambda r: r.start, reverse=True)
            marked = text
            for r in results:
                entity_text = text[r.start:r.end]
                wrapped = f"🛡️{entity_text}🛡️"
                marked = marked[:r.start] + wrapped + marked[r.end:]
            return marked
        except Exception:
            return text

    @staticmethod
    def _prepend_first_content(response: ModelResponse, tag: str) -> None:
        """In-place: prepend ``tag`` to the first assistant text message and mark any PII."""
        for choice in getattr(response, "choices", None) or []:
            msg = getattr(choice, "message", None)
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content:
                if content.startswith("[served:"):
                    return
                marked_content = self._mark_pii_entities(content)
                msg.content = tag + marked_content
                return

    @staticmethod
    def _is_malformed_toolcall(
        content: str,
        saw_tool_calls: bool,
        finish: Optional[str],
        args_ok: Optional[bool] = None,
    ) -> bool:
        """
        True when a stream is NOT a clean, usable tool call.

        Two failure shapes the local qwen3.8 model produces:
          (a) the backend parser DROPS the call entirely -> the raw
              ``<tool_call`` text leaks into ``content`` and no structured
              ``tool_calls`` arrive; or
          (b) structured ``tool_calls`` DO arrive but their ``arguments`` are
              malformed (extra trailing data / truncated JSON), which LiteLLM
              then fails to parse ("Failed to parse tool call arguments").

        Shape (a) has two sub-shapes, and both are recognised by the markup
        residue at the END of the message:

          a1. the WHOLE block survives, so ``content`` ends
              ``…</tool_call>`` (how the old ``hermes`` parser failed -- it died
              at the opener, so the opener came through too); or
          a2. the engine consumed the opener and aborted further in, leaving
              only a mid-tag TAIL -- e.g. ``content`` begins at
              ``<parameter=command>`` and still ends ``…</tool_call>``. That is
              measured behaviour on this engine (``qwen3_coder``), and the
              opener-only test this replaces returned False for it, so the turn
              died silently and a human had to type "resume" to continue.

        Tail-anchoring is deliberate. The old test was a bare substring search
        for ``<tool_call`` ANYWHERE in the content, which fires on any prose
        answer that merely *names* the tags (this repo writes about them
        constantly) -- and firing there REPLACES a legitimate text answer with a
        resampled tool call. A dropped call is always the last thing the model
        emitted, so the tail is both sufficient and precise.
        """
        if not saw_tool_calls:
            return bool(_LEAKED_TAIL_RE.search(content or ""))
        if args_ok is not None:
            return not args_ok
        return False

    async def _retry_toolcall(
        self, request_data: dict
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Re-issue a tool-calling request whose previous completion leaked a
        malformed ``<tool_call>`` into content. Appends a corrective user note
        and streams the fresh completion. Never raises; on any failure yields
        nothing so the caller falls back to the original (bad) stream.
        """
        try:
            msgs = list(request_data.get("messages") or [])
            msgs = self._normalize_system_messages(msgs)
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        "[System: your previous reply was discarded because it "
                        "contained a malformed tool call. Output EXACTLY ONE tool "
                        "call as VALID JSON inside <tool_call></tool_call>, with "
                        "exactly two keys: \"name\" (string) and \"arguments\" "
                        "(a NESTED object whose keys are the parameter names). "
                        "Example: <tool_call>{\"name\": \"list_dir\", "
                        "\"arguments\": {\"path\": \"/tmp\"}}</tool_call>. "
                        "Do not output prose before or after the tool call."
                    ),
                }
            )
            token = _tool_retry_ctx.set(_tool_retry_ctx.get() + 1)
            try:
                kwargs = {
                    "model": request_data.get("model"),
                    "messages": msgs,
                    "tools": request_data.get("tools"),
                    "tool_choice": "auto",
                    "stream": True,
                    "max_tokens": request_data.get("max_tokens"),
                }
                for k in ("temperature", "top_p"):
                    if request_data.get(k) is not None:
                        kwargs[k] = request_data[k]
                api_key = request_data.get("api_key")
                if api_key:
                    kwargs["api_key"] = api_key
                stream = await litellm.acompletion(**kwargs)
                async for chunk in stream:
                    yield chunk
            finally:
                _tool_retry_ctx.reset(token)
        except Exception:
            return

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        """
        Runs after a successful NON-STREAMING LLM call, before the response is
        returned to the client. Returning the (modified) response replaces what
        the proxy sends back. See docs.litellm.ai/docs/proxy/call_hooks.
        """
        try:
            requested = (data or {}).get("model", "")
            served = getattr(response, "model", "") or requested
            tag = self._served_tag(requested, served)
            if tag:
                self._prepend_first_content(response, tag)
        except Exception:
            pass
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Streaming hook.

        - For requests WITHOUT ``tools``: stream straight through, tagging the
          first content delta (existing behavior).
        - For requests WITH ``tools``: buffer the whole stream so we can detect
          a tool call the backend parser dropped (malformed ``<tool_call>``
          leaked into content, no structured ``tool_calls``). If that happens
          (and we have retry budget), re-issue the request with a corrective
          note and stream the retried response. Otherwise yield the buffered
          chunks (tagging the first content delta as usual).
        """
        requested = (request_data or {}).get("model", "")
        has_tools = bool((request_data or {}).get("tools"))

        if not has_tools:
            tagged = False
            try:
                first = await response.__anext__()
                served = getattr(first, "model", "") or requested
                tag = self._served_tag(requested, served)
                idx = 0
                while True:
                    chunk = first if idx == 0 else await response.__anext__()
                    idx += 1
                    try:
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)
                        if tag and isinstance(content, str) and content and not tagged:
                            marked_content = self._mark_pii_entities(content)
                            delta.content = tag + marked_content
                            tagged = True
                        elif isinstance(content, str) and content:
                            delta.content = self._mark_pii_entities(content)
                    except Exception:
                        pass
                    yield chunk
            except StopAsyncIteration:
                return
            except Exception:
                return

        chunks = []
        content = ""
        saw_tool_calls = False
        finish = None
        served = requested
        tool_args: dict = {}
        try:
            async for chunk in response:
                chunks.append(chunk)
                try:
                    if not served:
                        served = getattr(chunk, "model", "") or requested
                    ch = chunk.choices[0]
                    delta = getattr(ch, "delta", None)
                    dc = getattr(delta, "content", None)
                    if isinstance(dc, str):
                        content += dc
                    tcs = getattr(delta, "tool_calls", None)
                    if tcs:
                        saw_tool_calls = True
                        for tc in tcs:
                            fn = getattr(tc, "function", None)
                            if fn is None:
                                continue
                            frag = getattr(fn, "arguments", None)
                            if not isinstance(frag, str):
                                continue
                            idx = getattr(tc, "index", 0)
                            tool_args[idx] = tool_args.get(idx, "") + frag
                    fr = getattr(ch, "finish_reason", None)
                    if fr:
                        finish = fr
                except Exception:
                    pass
        except StopAsyncIteration:
            pass
        except Exception:
            pass

        args_ok = True
        if saw_tool_calls and tool_args:
            for blob in tool_args.values():
                try:
                    json.loads(blob)
                except Exception:
                    args_ok = False
                    break

        malformed = (
            self._is_malformed_toolcall(
                content, saw_tool_calls, finish, args_ok=args_ok
            )
            and _tool_retry_ctx.get() < _MAX_TOOL_RETRIES
        )
        if malformed:
            retried = False
            async for chunk in self._retry_toolcall(request_data):
                retried = True
                yield chunk
            if retried:
                return

        tagged = False
        tag = self._served_tag(requested, served)
        for chunk in chunks:
            try:
                delta = chunk.choices[0].delta
                dc = getattr(delta, "content", None)
                if tag and isinstance(dc, str) and dc and not tagged:
                    delta.content = tag + dc
                    tagged = True
            except Exception:
                pass
            yield chunk


_FALLBACK_OVERRIDES = dict(_local_models.FALLBACKS)

_CONTEXT_WINDOW_FALLBACK_OVERRIDES = dict(_local_models.CONTEXT_WINDOW_FALLBACKS)


def _ensure_router_fallbacks():
    """Register model-group fallbacks on the live proxy router at import time.

    Why: DB-managed group fallbacks (``model_info.fallbacks``) have been
    observed NOT to route on litellm main-latest -- requests to such groups
    fail with ``Available Model Group Fallbacks=None`` instead of failing
    over (and ``litellm_settings.fallbacks`` in config.yaml showed up as
    router_settings.fallbacks=null too). The reliable channel in this build
    is the ``POST /fallback`` admin endpoint, which writes the router's
    fallback map at request time. (The pre-startup wiring attempt below is
    kept as a belt-and-braces backup; /fallback survives restarts only via
    the proxy's own persistence, so if that ever stops working, point the
    hermes-agent unit or a small oneshot at the curl in
    docker-compose.yml's comment.)
    """
    try:
        import json as _json
        import os as _os
        import urllib.request

        key = _os.environ.get("LITELLM_MASTER_KEY") or ""
        base = "http://127.0.0.1:4000"
        import time as _time
        for _ in range(60):
            try:
                req = urllib.request.Request(base + "/health/liveliness")
                with urllib.request.urlopen(req, timeout=2):
                    break
            except Exception:
                _time.sleep(1)
        else:
            return
        def _post_fallback(model: str, targets: list, fallback_type: str) -> None:
            body = _json.dumps(
                {
                    "model": model,
                    "fallback_models": list(targets),
                    "fallback_type": fallback_type,
                }
            ).encode()
            req = urllib.request.Request(
                base + "/fallback",
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + key,
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=10):
                    pass
            except Exception:
                pass

        for group, targets in _FALLBACK_OVERRIDES.items():
            _post_fallback(group, targets, "general")
        for group, targets in _CONTEXT_WINDOW_FALLBACK_OVERRIDES.items():
            _post_fallback(group, targets, "context_window")
    except Exception:
        pass


def _wire_router_fallbacks_directly():
    """Backup channel: call the Router.add_model_group_fallbacks directly.

    Attempted BEFORE the HTTP fallback above in case the admin API is not
    reachable yet; usually a no-op on main-latest (the router picks up
    /fallback writes, direct DB rows and YAML router_settings.fallbacks
    have both been observed ignored).
    """
    try:
        import litellm as _litellm
        import yaml as _yaml
        from litellm.router import Router

        def _cfg_fallbacks():
            try:
                with open("/app/config.yaml") as f:
                    cfg = _yaml.safe_load(f) or {}
                fb = (cfg.get("litellm_settings") or {}).get("fallbacks")
                if isinstance(fb, list):
                    return fb
                return None
            except Exception:
                return None

        def _fallbacks_map(fb):
            if isinstance(fb, dict):
                return {k: (v if isinstance(v, list) else [v]) for k, v in fb.items()}
            out = {}
            if isinstance(fb, list):
                for item in fb:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            out[k] = v if isinstance(v, list) else [v]
            return out

        desired = dict(_FALLBACK_OVERRIDES)
        for k, v in _fallbacks_map(_cfg_fallbacks() or {}).items():
            desired.setdefault(k, v)

        r = getattr(_litellm, "router", None)
        if isinstance(r, Router):
            for group, targets in desired.items():
                r.add_model_group_fallbacks(model_group=group, fallbacks=targets)
            return

        import gc as _gc

        for obj in _gc.get_objects():
            if type(obj).__name__ == "Router":
                for group, targets in desired.items():
                    try:
                        obj.add_model_group_fallbacks(model_group=group, fallbacks=targets)
                    except Exception:
                        pass
                return
    except Exception:
        pass


proxy_handler_instance = AsyncPrependServedModel()

import threading as _threading
_threading.Thread(
    target=_ensure_router_fallbacks,
    name="router-fallback-wiring",
    daemon=True,
).start()
_wire_router_fallbacks_directly()


from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    _OPTIONAL_PresidioPIIMasking,
)

_UNPROTECTED_SUFFIX = "-unprotected"

def _unprotected_route(data: dict) -> bool:
    """True when the requested route name opts out of the global guardrails."""
    if not isinstance(data, dict):
        return False
    model = data.get("model")
    return isinstance(model, str) and model.endswith(_UNPROTECTED_SUFFIX)

class UnprotectedAwarePresidio(_OPTIONAL_PresidioPIIMasking):
    """Presidio guardrail that bypasses any `-unprotected` route on every hook."""

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type, **kwargs):
        if _unprotected_route(data):
            return data
        return await super().async_pre_call_hook(
            user_api_key_dict, cache, data, call_type, **kwargs
        )

    async def async_post_call_success_hook(self, data, user_api_key_dict, response, **kwargs):
        if _unprotected_route(data):
            return response
        return await super().async_post_call_success_hook(
            data, user_api_key_dict, response, **kwargs
        )

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict, response, request_data, **kwargs
    ):
        if _unprotected_route(request_data):
            async for chunk in response:
                yield chunk
            return
        async for chunk in super().async_post_call_streaming_iterator_hook(
            user_api_key_dict, response, request_data, **kwargs
        ):
            yield chunk

    async def async_logging_hook(self, kwargs, result, call_type, **kw):
        if _unprotected_route(kwargs if isinstance(kwargs, dict) else {}):
            return kwargs, result
        return await super().async_logging_hook(kwargs, result, call_type, **kw)

litellm_presidio = UnprotectedAwarePresidio


import os as _os
import sqlite3 as _sqlite3
import hashlib as _hashlib
from datetime import datetime as _datetime
from pathlib import Path as _Path

_AUDIT_DB_PATH = _os.environ.get("LITELLM_AUDIT_DB", "/litellm/audit.db")


_request_start_times: dict = {}

def _init_audit_db():
    """Create the audit database and table if they don't exist."""
    try:
        db_path = _Path(_AUDIT_DB_PATH)
        if db_path.parent and not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = _sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                api_key_hash TEXT,
                requested_model TEXT NOT NULL,
                served_model TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                duration_ms INTEGER,
                status TEXT DEFAULT 'success',
                error_message TEXT,
                user_id TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_served_model ON requests(served_model)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_requested_model ON requests(requested_model)")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[audit] Failed to init audit DB: {e}")

def _log_request(
    api_key_hash: str,
    requested_model: str,
    served_model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: int,
    status: str = "success",
    error_message: str = None,
    user_id: str = None,
):
    """Log a single LLM request to the audit database."""
    try:
        conn = _sqlite3.connect(_AUDIT_DB_PATH)
        conn.execute(
            """
            INSERT INTO requests 
            (timestamp, api_key_hash, requested_model, served_model,
             prompt_tokens, completion_tokens, total_tokens, duration_ms,
             status, error_message, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _datetime.utcnow().isoformat() + "Z",
                api_key_hash,
                requested_model,
                served_model,
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
                duration_ms,
                status,
                error_message,
                user_id,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[audit] Failed to log request: {e}")

_init_audit_db()

class AsyncAuditLogger(CustomLogger):
    """LiteLLM callback that logs every request to the audit database."""
    
    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
        **kwargs,
    ) -> dict:
        """Record the start time so we can calculate duration later."""
        try:
            _request_start_times[id(data)] = _datetime.utcnow().timestamp()
        except Exception:
            pass
        return data
    
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        """Log successful requests to the audit database."""
        try:
            requested = (data or {}).get("model", "")
            served = getattr(response, "model", "") or requested
            
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            
            start_time = _request_start_times.get(id(data))
            duration_ms = int((_datetime.utcnow().timestamp() - start_time) * 1000) if start_time else 0
            
            api_key = (data or {}).get("api_key")
            if not api_key:
                api_key = getattr(user_api_key_dict, "token", None)
            api_key_hash = _hashlib.sha256((api_key or "").encode()).hexdigest()[:16] if api_key else None
            
            user_id = getattr(user_api_key_dict, "user_id", None)
            
            _log_request(
                api_key_hash=api_key_hash,
                requested_model=requested,
                served_model=served,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                status="success",
                user_id=user_id,
            )
            
            import sys
            print(f"[audit] Logged request: {requested} -> {served} ({prompt_tokens}+{completion_tokens} tokens, {duration_ms}ms)", file=sys.stderr, flush=True)
        except Exception as e:
            import sys
            print(f"[audit] Failed to log request: {e}", file=sys.stderr, flush=True)
        
        return response
    
    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        user_api_key_dict: Any,
        original_exception=None,
        traceback_str=None,
        response=None,
        **kwargs,
    ) -> Any:
        """Log failed requests to the audit database.

        NOTE on signature: LiteLLM invokes this hook as
            async_post_call_failure_hook(request_data=..., user_api_key_dict=...,
                                         original_exception=..., traceback_str=...)
        (keyword args). Do NOT rename the parameters without matching that call.
        """
        try:
            data = request_data or {}
            requested = data.get("model", "")
            served = data.get("_served_model") or requested

            start_time = _request_start_times.get(id(data))
            duration_ms = int((_datetime.utcnow().timestamp() - start_time) * 1000) if start_time else 0

            api_key = data.get("api_key")
            if not api_key:
                api_key = getattr(user_api_key_dict, "token", None)
            api_key_hash = _hashlib.sha256((api_key or "").encode()).hexdigest()[:16] if api_key else None

            user_id = getattr(user_api_key_dict, "user_id", None)

            error_msg = "unknown error"
            if original_exception is not None:
                error_msg = getattr(original_exception, "message", None) or str(original_exception)
            elif response is not None:
                error_msg = getattr(response, "message", "") or getattr(response, "args", "")[:200]
            error_msg = str(error_msg)[:500]

            _log_request(
                api_key_hash=api_key_hash,
                requested_model=requested,
                served_model=served,
                prompt_tokens=0,
                completion_tokens=0,
                duration_ms=duration_ms,
                status="error",
                error_message=error_msg,
                user_id=user_id,
            )
        except Exception:
            pass

        return response


_proxy_handler_audit = AsyncAuditLogger()


_PATCHED = False

if not _PATCHED:
    try:
        from litellm.types.utils import CallTypes
        import litellm.llms.openai.chat.guardrail_translation as _gtm
        import litellm.llms.openai.chat.guardrail_translation.handler as _hmod

        class _SafePresidioHandler(_hmod.OpenAIChatCompletionsHandler):
            """OpenAI chat guardrail handler that never mutates the INPUT text.

            Output unmasking is inherited unchanged (``process_output_response``)
            -- this class only neutralises the input rewrite.
            """

            async def _apply_guardrail_responses_to_input_texts(self, *args, **kwargs):
                verbose_logger.debug(
                    "[custom_callbacks] no-op: skipping Presidio input text rehydration"
                )
                return None

            async def _apply_guardrail_responses_to_input_tool_calls(self, *args, **kwargs):
                verbose_logger.debug(
                    "[custom_callbacks] no-op: skipping Presidio input tool_call rehydration"
                )
                return None

        _gtm.guardrail_translation_mappings[CallTypes.completion] = _SafePresidioHandler
        _gtm.guardrail_translation_mappings[CallTypes.acompletion] = _SafePresidioHandler
        _hmod.OpenAIChatCompletionsHandler = _SafePresidioHandler

        _PATCHED = True
        verbose_logger.info(
            "[custom_callbacks] Presidio guardrail input-mutation patch applied "
            "(no-op) for OpenAI chat completions"
        )
    except Exception:
        verbose_logger.exception(
            "[custom_callbacks] Presidio guardrail input-mutation patch FAILED "
            "-- continuing with stock LiteLLM behaviour"
        )
