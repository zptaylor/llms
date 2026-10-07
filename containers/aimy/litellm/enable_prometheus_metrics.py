"""Enable the Prometheus /metrics endpoint on the LiteLLM proxy.

WHY THIS FILE EXISTS
--------------------
The litellm ``main-latest`` image registers a ``PrometheusAuthMiddleware``
at app-construction time (``proxy_server.py`` -> ``app.add_middleware(...)``).
The middleware 401s every ``/metrics`` request unless the *live server
process*'s ``litellm.require_auth_for_metrics_endpoint`` attribute is
``False``. The middleware is unauthenticated-friendly only in that case; the
vLLM Prometheus sidecar (<LAN_IP>) scrapes ``litellm:4000/metrics`` with no
credentials, so every 15s scrape produced a "Malformed API Key" 401 in the
proxy logs.

In this image build the standard config keys for that flag are NOT wired to
the module attribute:

  * ``litellm_settings.require_auth_for_metrics_endpoint: false`` in
    config.yaml is parsed into the config dict but never applied to
    ``litellm.require_auth_for_metrics_endpoint`` (the loop in
    ``_update_config_fields`` only overrides keys in
    ``LITELLM_SETTINGS_SAFE_DB_OVERRIDES``, which does not include it, and
    ``get_config`` has no env-var overlay for it).
  * The ``LITELLM_REQUIRE_AUTH_FOR_METRICS_ENDPOINT`` env var is not read.
  * The ``/metrics`` ASGI endpoint is only mounted when
    ``initialize_callbacks_on_proxy`` (the ``callbacks:`` branch) sees a
    ``prometheus`` entry in the list; ``success_callback:`` only registers
    the logger and does NOT mount the endpoint (the mount call in that
    branch is unreachable for the dotted-classpath form, and a bare
    ``"prometheus"`` string is not a resolvable callback in this build).

So the only supported way to flip the flag + mount the endpoint in the live
process is to make it happen as a side effect of a callback that
``callbacks:`` *does* initialize. That is what this module does.

HOW IT WORKS
------------
``config.yaml`` lists ``enable_prometheus_metrics.enable_instance`` in
``litellm_settings.callbacks``. At startup the proxy's callback-init path
imports this module and instantiates ``_EnablePrometheusMetrics``. Its
``__init__`` runs in the live server process (single gunicorn worker,
``preload: true``) and:

  1. Sets ``litellm.require_auth_for_metrics_endpoint = False`` so the
     ``PrometheusAuthMiddleware`` (and ``user_api_key_auth``) stop
     demanding credentials on ``/metrics``.
  2. Calls ``PrometheusLogger._mount_metrics_endpoint()`` to actually mount
     the ``/metrics`` ASGI route on the FastAPI app. Without this the
     endpoint is 404 even when auth is disabled.

The class is a ``CustomLogger`` subclass with no-op handlers, so it is
harmless to the request path (the proxy dispatches CustomLogger *instances*;
ours does nothing on success/failure/post-call events).

ORDERING MATTER
---------------
The callbacks list is processed in order. This entry is listed AFTER
``custom_callbacks.proxy_handler_instance`` and
``custom_callbacks._proxy_handler_audit`` in config.yaml so the audit logger
and the served-model tagger are registered first (their order relative to
each other is what matters for request handling; this module has no
request-path effect and only needs to run once at startup).

NOTE ON THE 307
----------------
``app.mount("/metrics", ...)`` creates a prefix mount; a GET to exactly
``/metrics`` is followed by the metrics sub-app with a 307 to
``/metrics/``. Prometheus follows redirects by default, so scrapes succeed;
the previous 401 "Malformed API Key" errors are gone.
"""

from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger


class _EnablePrometheusMetrics(CustomLogger):
    """No-op callback that flips the /metrics auth flag and mounts the endpoint."""

    def __init__(self) -> None:
        super().__init__()
        litellm.require_auth_for_metrics_endpoint = False
        try:
            from litellm.integrations.prometheus import PrometheusLogger

            PrometheusLogger._mount_metrics_endpoint()
        except Exception as e:
            import sys

            print(
                f"enable_prometheus_metrics: failed to mount /metrics: {e}",
                file=sys.stderr,
            )

        # Register a real PrometheusLogger INSTANCE on litellm.callbacks.
        #
        # WHY NOT `success_callback: ["prometheus"]` IN config.yaml
        # -------------------------------------------------------
        # That is the documented way, and it does create the metric families —
        # but in this build it only ever leaves the *string* in the list. The
        # proxy's own startup line proves it:
        #     "Initialized Success Callbacks - ['prometheus']"
        # and the request path dispatches success/failure events to logger
        # *instances*, so nothing is ever observed: /metrics exports every
        # litellm_* family with HELP/TYPE and zero samples, and each litellm:*
        # recording rule in lts-rules.yml evaluates to nothing.
        #
        # Listing it in BOTH success_callback and failure_callback is worse: the
        # second instantiation dies with "Duplicated timeseries in
        # CollectorRegistry" (visible as a non-blocking ERROR at startup).
        #
        # One instance on litellm.callbacks is what actually records. It is also
        # the list this build's own PrometheusLogger.get_instance() searches
        # ("Find the PrometheusLogger instance from litellm.callbacks, if
        # registered") — that accessor is never called, but it documents the
        # intended home. Registered here rather than in config.yaml because this
        # callback is initialised from the `callbacks:` key, which precedes
        # success_callback/failure_callback: anything added earlier would be
        # discarded when those branches reset their lists.
        try:
            import sys

            from litellm.integrations.prometheus import PrometheusLogger

            try:
                _pl = PrometheusLogger()
            except Exception:
                # Already instantiated elsewhere in this process. Its __init__
                # (re)creates the metric objects, so a second call raises
                # "Duplicated timeseries in CollectorRegistry" — reuse the
                # registered one instead of giving up on instrumentation.
                _pl = PrometheusLogger.get_instance()
                if _pl is None:
                    raise
            if not isinstance(getattr(litellm, "callbacks", None), list):
                litellm.callbacks = []
            if _pl not in litellm.callbacks:
                litellm.callbacks.append(_pl)
            print(
                "enable_prometheus_metrics: registered "
                f"{type(_pl).__name__} on litellm.callbacks",
                file=sys.stderr,
            )
        except Exception as e:
            import sys

            print(
                f"enable_prometheus_metrics: failed to register the prometheus "
                f"logger (metrics will stay empty): {e}",
                file=sys.stderr,
            )


    async def async_success_handler(self, user_id=None, sr=None, call_type=None) -> None:
        return None

    async def async_failure_handler(self, user_id=None, sr=None, call_type=None) -> None:
        return None

    async def async_post_call_event_handler(self, user_id=None, response=None, call_type=None) -> None:
        return None

    def success_handler(self, user_id=None, sr=None, call_type=None) -> None:
        return None

    def failure_handler(self, user_id=None, sr=None, call_type=None) -> None:
        return None

    def post_call_event_handler(self, user_id=None, response=None, call_type=None) -> None:
        return None


enable_instance = _EnablePrometheusMetrics()
