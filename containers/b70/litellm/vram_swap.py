"""vram_swap.py — LiteLLM pre-call hook that makes the requested model the awake one.

WHY
---
The Arc B70 has one 32 GB card and hosts two vLLM engines (see
../../docs/vllm-memswap.md): `qwen3.8` on :8000 and `nex-n2.5-mini` on :8003.
Both processes stay resident, but only one may be awake at a time — the other
parks its weights in host RAM (`POST /sleep?level=1`). So before a request is
forwarded upstream, whoever holds the card has to change.

That decision belongs to the coordinator alone (scripts/vllm-memswapd.py). This
hook is just the request-path client of it: it asks the coordinator to make the
requested model awake, and routes the request to whichever concrete member the
coordinator says is now serving.

Consequences worth knowing
--------------------------
* A request that requires a swap pays for it. Level-1 wakes copy weights from
  host RAM at roughly 1 s per 4 GiB measured on this card (~5 s for the 19 GiB
  qwen3.8 model), but a member whose container is *cold* has to boot first
  (minutes). Run `scripts/vllm-memswap.sh up` at boot so both members are warm
  and parked before the first request needs them.
* Sticky names (`b70`, `local`) resolve to whatever is currently awake, so a
  client that does not care which model answers cannot make alternating clients
  thrash the card. The hook rewrites `data["model"]` to the resolved concrete
  member, so the response also tells you which model actually answered.
* Models that are not memswap members (DeepSeek, OpenRouter, the NVIDIA rows,
  hermes-3-8b on the GTX 1070) are passed through untouched.

Failure policy
--------------
If the coordinator is unreachable we do NOT blindly pass the request through: a
sleeping engine accepts the request and then never schedules it (the scheduler
is paused), so the caller would hang until timeout. Instead we ask the engine
directly whether it is sleeping and fail fast with an actionable message when it
is — while still letting a genuinely-serving engine through, so a coordinator
outage does not by itself take down local inference.

Wiring: `litellm_settings.callbacks` in config.yaml must list the module-level
INSTANCE (`vram_swap.handler_instance`), matching the existing
`custom_callbacks.proxy_handler_instance` pattern.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("vram_swap")

COORDINATOR_URL = os.environ.get("MEMSWAP_URL", "http://host.docker.internal:8002").rstrip("/")
ENABLED = os.environ.get("MEMSWAP_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
ENGINE_HOST = os.environ.get("MEMSWAP_ENGINE_HOST", "host.docker.internal")
ENSURE_TIMEOUT_S = float(os.environ.get("MEMSWAP_ENSURE_TIMEOUT_S", "2400"))
NAMES_TTL_S = float(os.environ.get("MEMSWAP_NAMES_TTL_S", "60"))
MEMBERS_ENV = os.environ.get("MEMSWAP_MEMBERS", "qwen3.8:8000,nex-n2.5-mini:8003")
STICKY_ENV = os.environ.get("MEMSWAP_STICKY", "b70,local,auto-local")


def _bootstrap_names() -> dict[str, int]:
    names: dict[str, int] = {}
    for part in MEMBERS_ENV.split(","):
        name, _, port = part.strip().partition(":")
        if name and port.isdigit():
            names[name] = int(port)
    for name in STICKY_ENV.split(","):
        if name.strip():
            names.setdefault(name.strip(), -1)
    return names


def _http(method: str, url: str, timeout: float, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw else {}


def _ensure_http(url: str, timeout: float, body: dict) -> dict:
    """POST /ensure, treating an error status with a JSON verdict as a DECISION.

    A refusal (503 + {"ok": false, "error": ...}) is the coordinator answering
    "no", not the coordinator being broken — the caller must surface that reason
    verbatim rather than degrade to the outage path.
    """
    try:
        return _http("POST", url, timeout, body)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            payload = json.loads(raw)
        except ValueError:
            raise
        if isinstance(payload, dict) and "ok" in payload:
            return payload
        raise


class VramSwapLease(CustomLogger):
    """Pre-call hook: ensure the requested memswap member owns the card."""

    def __init__(self) -> None:
        super().__init__()
        self._names: dict[str, int] = {}
        self._names_at: float = 0.0
        self._lock = asyncio.Lock()

    async def _known_names(self, force: bool = False) -> dict[str, int]:
        if not force and self._names and (time.monotonic() - self._names_at) < NAMES_TTL_S:
            return self._names
        try:
            payload = await asyncio.to_thread(
                _http, "GET", f"{COORDINATOR_URL}/names", 5.0, None
            )
            names: dict[str, int] = {}
            ports: dict[str, int] = payload.get("members", {})
            names.update(ports)
            for alias, member in (payload.get("aliases") or {}).items():
                if member in ports:
                    names[alias] = ports[member]
            for sticky in payload.get("sticky") or []:
                names.setdefault(sticky, -1)
            self._names = names
            self._names_at = time.monotonic()
            log.info("vram_swap: coordinator names = %s", sorted(names))
        except Exception as exc:
            log.warning("vram_swap: could not read %s/names: %s", COORDINATOR_URL, exc)
            if not self._names:
                self._names = _bootstrap_names()
                self._names_at = time.monotonic()
        return self._names

    async def _ensure(self, model: str) -> dict:
        return await asyncio.to_thread(
            _ensure_http, f"{COORDINATOR_URL}/ensure", ENSURE_TIMEOUT_S, {"model": model}
        )

    async def _is_sleeping(self, port: int) -> bool | None:
        if port <= 0:
            return None
        try:
            payload = await asyncio.to_thread(
                _http, "GET", f"http://{ENGINE_HOST}:{port}/is_sleeping", 5.0, None
            )
            return bool(payload.get("is_sleeping"))
        except Exception:
            return None

    async def async_pre_call_hook(
        self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str, **kwargs
    ) -> dict:
        """See the module docstring. ``**kwargs`` absorbs contract drift in this
        rolling ``main-latest`` image, exactly as custom_callbacks.py does."""
        try:
            if not ENABLED:
                return data
            model = (data or {}).get("model")
            if not model:
                return data

            names = await self._known_names()
            if model not in names:
                names = await self._known_names(force=True)
                if model not in names:
                    return data

            async with self._lock:
                try:
                    result = await self._ensure(model)
                except Exception as exc:
                    result = await self._passthrough_or_fail(model, names.get(model, -1), exc)

            if not result.get("ok"):
                raise RuntimeError(
                    f"VRAM-swap coordinator refused to serve '{model}': {result.get('error')}"
                )
            target = result.get("awake")
            if target and target != model:
                log.info("vram_swap: '%s' resolved to '%s'", model, target)
                data["model"] = target
            return data
        except RuntimeError:
            raise
        except Exception as exc:
            log.exception("vram_swap: hook failed, passing through: %s", exc)
            return data

    async def _passthrough_or_fail(self, model: str, port: int, cause: Exception) -> dict:
        """Coordinator unreachable: pass through only if an engine is really awake."""
        if port == -1:
            awake = [
                name for name, p in self._names.items()
                if p > 0 and await self._is_sleeping(p) is False
            ]
            if len(awake) == 1:
                log.warning("vram_swap: coordinator down; sticky '%s' served by '%s'",
                            model, awake[0])
                return {"ok": True, "awake": awake[0], "degraded": True}
            raise RuntimeError(
                f"VRAM-swap coordinator unreachable at {COORDINATOR_URL} ({cause}); "
                f"cannot resolve '{model}' — engines awake: {awake or 'none'}. "
                "Start it on the host with: systemctl --user start vllm-memswap"
            )

        sleeping = await self._is_sleeping(port)
        if sleeping is False:
            log.warning("vram_swap: coordinator down but '%s' is awake; passing through", model)
            return {"ok": True, "awake": model, "degraded": True}
        raise RuntimeError(
            f"VRAM-swap coordinator unreachable at {COORDINATOR_URL} ({cause}) and "
            f"'{model}' is not confirmed awake"
            + ("" if sleeping is None else " (it is asleep — a request would hang until timeout)")
            + ". Start it on the host with: systemctl --user start vllm-memswap"
        )


handler_instance = VramSwapLease()
