"""
app.observability.telemetry
===========================
Telemetry — the one tracing facade of the service, backed by Langfuse (OpenTelemetry under the hood).

    trace(module, …)      : the root span of a run, named after its `ModuleName`; stamps the trace with
                            tags `module:<m>` / `kind:<k>` (+ `dataset:<name>@<version>`), the session id
                            (= execution id), the pipeline version, the release (= service version) and the
                            allowlisted identifiers, propagated to every child span. Called while a trace is
                            already open, it opens a child span instead (a workflow run inside another)
    span(name, …)         : a child span (`SpanName`) with sanitised metadata
    generation(name, …)   : a child generation — a model call with `model` and token `usage_details`
    handle.set / set_output / set_usage / record_error : what a span may carry, allowlisted and capped
    current_trace_id()    : for the `X-Trace-Id` header and job records
    query_input(query)    : what a retrieval trace records as its input — a hash and the length, or the
                            text when `capture_query_text` is on (an explicit, off-by-default switch)

Parent/child correlation rides on contextvars (OpenTelemetry's current span): a thread pool task bound
with `app.app_logging.setup.in_context` — or an executor call wrapped the same way — keeps its parent.

Failure policy: secondary path, always. The default `Telemetry` is a no-op (tests, scripts, the
evaluation harness, `LANGFUSE_ENABLED=false`). `LangfuseTelemetry` guards every SDK call: an exception
inside the SDK is logged (WARNING at most once a minute per operation) and the run continues with a
no-op handle; spans are exported from the SDK's own background thread with `LANGFUSE_TIMEOUT`;
`shutdown()` flushes within that timeout. A misconfiguration (enabled without keys, invalid sample
rate) raises at `configure_telemetry`, i.e. at startup, never mid-request.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

from app.app_logging.setup import get_logger, request_id_var
from app.observability.context import current_execution, trace_ids_var
from app.observability.modules import ModuleName, SpanName
from app.observability.sanitize import classify_error, query_fingerprint, sanitize_metadata

__all__ = [
    "LangfuseTelemetry", "Observation", "Telemetry", "configure_telemetry", "get_telemetry", "langfuse_environment",
    "reset_telemetry", "set_telemetry",
]

_WARN_INTERVAL_SECONDS = 60.0
_DEFAULT_ENVIRONMENT = "default"
_MAX_ENVIRONMENT_CHARS = 40
#: Trace-level identifiers propagated to every span (Langfuse caps each encoded value at 200 chars).
_PROPAGATED_KEYS = (
    "module_name", "execution_kind", "execution_id", "dataset_name", "dataset_version", "case_id", "attempt", "request_id",
    "job_id", "document_id", "collection_name", "content_hash", "pipeline_version", "config_fingerprint",
)


class Observation:
    """No-op handle; `LangfuseTelemetry` returns a subclass bound to a live observation."""

    trace_id: Optional[str] = None

    def set(self, **attrs: Any) -> None:
        """Allowlisted metadata on this observation."""

    def set_output(self, **attrs: Any) -> None:
        """Allowlisted output fields (ids, counts, scores — never text)."""

    def set_usage(self, **usage: int) -> None:
        """Token usage of a generation (`input`, `output`, `total`, …)."""

    def record_error(self, exc: BaseException) -> None:
        """Mark the observation failed with a sanitised error description."""


class Telemetry:
    """The no-op implementation and the interface every caller sees."""

    enabled = False

    def __init__(self, *, pipeline_version: str = "", config_fingerprint: str = "", capture_query_text: bool = False) -> None:
        self.pipeline_version = pipeline_version
        self.config_fingerprint = config_fingerprint
        self.capture_query_text = capture_query_text

    @contextmanager
    def trace(self, module: ModuleName, *, input: Optional[Dict[str, Any]] = None, **ids: Any) -> Iterator[Observation]:
        _require(module, ModuleName)
        yield _NULL

    @contextmanager
    def span(self, name: SpanName, **attrs: Any) -> Iterator[Observation]:
        _require(name, SpanName)
        yield _NULL

    @contextmanager
    def generation(self, name: SpanName, *, model: str, **attrs: Any) -> Iterator[Observation]:
        _require(name, SpanName)
        yield _NULL

    def current_trace_id(self) -> Optional[str]:
        return None

    def query_input(self, query: str, **extra: Any) -> Dict[str, Any]:
        """The trace input for a query: its fingerprint, or the text when capture is explicitly enabled."""
        payload: Dict[str, Any] = {"query": (query or "").strip()} if self.capture_query_text else query_fingerprint(query)
        payload.update(sanitize_metadata(extra))
        return payload

    @property
    def langfuse_client(self) -> Any:
        """The raw Langfuse client for dataset mirroring; None when disabled."""
        return None

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class LangfuseTelemetry(Telemetry):
    """Langfuse-backed implementation. Construct through `configure_telemetry`."""

    enabled = True

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: Optional[str],
        timeout_seconds: int,
        sample_rate: float,
        environment: str,
        release: str,
        pipeline_version: str,
        config_fingerprint: str,
        capture_query_text: bool,
        span_exporter: Any = None,
        tracer_provider: Any = None,
    ) -> None:
        super().__init__(pipeline_version=pipeline_version, config_fingerprint=config_fingerprint, capture_query_text=capture_query_text)
        # imported here: the SDK is only needed when enabled
        from langfuse import Langfuse, is_langfuse_span, propagate_attributes

        self._propagate = propagate_attributes
        self._log = get_logger(module="TELEMETRY")
        self._warned_at: Dict[str, float] = {}
        kwargs: Dict[str, Any] = dict(
            public_key=public_key, secret_key=secret_key, timeout=timeout_seconds, sample_rate=sample_rate,
            environment=environment, release=release, tracing_enabled=True,
            # Only spans this facade created leave the process. The SDK's default also exports third-party
            # GenAI instrumentation (the Mistral SDK instruments its OCR / file calls), which bypasses the
            # metadata allowlist and could carry document content.
            should_export_span=is_langfuse_span,
        )
        if host:
            kwargs["base_url"] = host
        if span_exporter is not None:
            kwargs["span_exporter"] = span_exporter
        if tracer_provider is not None:
            kwargs["tracer_provider"] = tracer_provider
        self._client = Langfuse(**kwargs)

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    @contextmanager
    def trace(self, module: ModuleName, *, input: Optional[Dict[str, Any]] = None, **ids: Any) -> Iterator[Observation]:
        _require(module, ModuleName)
        if self.current_trace_id():  # a workflow run inside another: one trace, the module as a child span
            with self._observe(str(module), "span", metadata=sanitize_metadata({"nested_module": module, **ids})) as handle:
                yield handle
            return

        execution = current_execution()
        execution_id = execution.execution_id or request_id_var.get()
        tags = [f"module:{module}", f"kind:{execution.execution_kind}"]
        if execution.dataset_name:
            tags.append(f"dataset:{execution.dataset_name}@{execution.dataset_version or 'current'}")
        metadata = sanitize_metadata({
            "module_name": module, "execution_kind": execution.execution_kind, "execution_id": execution_id,
            "dataset_name": execution.dataset_name, "dataset_version": execution.dataset_version, "case_id": execution.case_id,
            "attempt": execution.attempt, "request_id": request_id_var.get(),
            "pipeline_version": self.pipeline_version, "config_fingerprint": self.config_fingerprint, **ids,
        })
        propagated = {key: metadata[key] for key in _PROPAGATED_KEYS if key in metadata}

        context = self._guarded("propagate", lambda: self._propagate(
            session_id=execution_id, version=self.pipeline_version or None, tags=tags, metadata=propagated, trace_name=str(module),
        ))
        entered = context is not None and self._guarded("propagate", context.__enter__) is not _FAILED
        try:
            with self._observe(str(module), "span", input=input, metadata=metadata) as handle:
                collected = trace_ids_var.get()
                if collected is not None and handle.trace_id:
                    collected.append(handle.trace_id)
                yield handle
        finally:
            if entered:
                self._guarded("propagate", lambda: context.__exit__(None, None, None))

    @contextmanager
    def span(self, name: SpanName, **attrs: Any) -> Iterator[Observation]:
        _require(name, SpanName)
        with self._observe(str(name), "span", metadata=sanitize_metadata(attrs)) as handle:
            yield handle

    @contextmanager
    def generation(self, name: SpanName, *, model: str, **attrs: Any) -> Iterator[Observation]:
        _require(name, SpanName)
        with self._observe(str(name), "generation", metadata=sanitize_metadata(attrs), model=model) as handle:
            yield handle

    def current_trace_id(self) -> Optional[str]:
        # Read OpenTelemetry's current span directly: the SDK's accessor logs a warning when no span is active,
        # which is the normal state right before a root trace opens.
        result = self._guarded("trace_id", _otel_trace_id)
        return result if isinstance(result, str) else None

    @property
    def langfuse_client(self) -> Any:
        return self._client

    def flush(self) -> None:
        self._guarded("flush", self._client.flush)

    def shutdown(self) -> None:
        self._guarded("shutdown", self._client.shutdown)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @contextmanager
    def _observe(
        self, name: str, as_type: str, *, input: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Iterator[Observation]:
        """Open an observation as the current span; the body runs even when the SDK fails."""
        kwargs: Dict[str, Any] = {"name": name, "as_type": as_type}
        if input is not None:
            kwargs["input"] = input
        if metadata:
            kwargs["metadata"] = metadata
        if model:
            kwargs["model"] = model
        manager = self._guarded("start", lambda: self._client.start_as_current_observation(**kwargs))
        handle: Observation = _NULL
        if manager is not None and manager is not _FAILED:
            observation = self._guarded("start", manager.__enter__)
            if observation is _FAILED:
                manager = None
            else:
                handle = _LangfuseObservation(observation, self, self.current_trace_id())
        else:
            manager = None
        try:
            yield handle
        except BaseException as exc:
            handle.record_error(exc)
            raise
        finally:
            if manager is not None:
                # Always a clean exit: the error is recorded sanitised above, OpenTelemetry's own
                # record_exception (raw message + stack trace) never runs.
                self._guarded("end", lambda: manager.__exit__(None, None, None))

    def _guarded(self, operation: str, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except Exception as exc:  # the SDK must never break the run
            self._warn(operation, exc)
            return _FAILED

    def _warn(self, operation: str, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._warned_at.get(operation, 0.0) >= _WARN_INTERVAL_SECONDS:
            self._warned_at[operation] = now
            self._log.warning("Telemetry %s failed (%s): %s — continuing without it", operation, type(exc).__name__, exc)
        else:
            self._log.debug("Telemetry %s failed: %s", operation, exc)


class _LangfuseObservation(Observation):
    def __init__(self, observation: Any, telemetry: LangfuseTelemetry, trace_id: Optional[str]) -> None:
        self._observation = observation
        self._telemetry = telemetry
        self.trace_id = trace_id

    def set(self, **attrs: Any) -> None:
        metadata = sanitize_metadata(attrs)
        if metadata:
            self._telemetry._guarded("update", lambda: self._observation.update(metadata=metadata))

    def set_output(self, **attrs: Any) -> None:
        output = sanitize_metadata(attrs)
        if output:
            self._telemetry._guarded("update", lambda: self._observation.update(output=output))

    def set_usage(self, **usage: int) -> None:
        details = {key: int(value) for key, value in usage.items() if isinstance(value, int) and not isinstance(value, bool)}
        if details:
            self._telemetry._guarded("update", lambda: self._observation.update(usage_details=details))

    def record_error(self, exc: BaseException) -> None:
        described = classify_error(exc)
        status = f"{described.get('error_category', 'internal')}: {described.get('error_type', 'Exception')}"
        self._telemetry._guarded(
            "update", lambda: self._observation.update(level="ERROR", status_message=status, metadata=sanitize_metadata(described)),
        )


_NULL = Observation()
_FAILED = object()  # sentinel: a guarded call raised


def _otel_trace_id() -> Optional[str]:
    """The active span's trace id as Langfuse renders it (32 lowercase hex), or None outside any span."""
    from opentelemetry import trace as otel_trace_api

    context = otel_trace_api.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def _require(value: Any, kind: type) -> None:
    if not isinstance(value, kind):
        raise TypeError(f"expected a {kind.__name__}, got {value!r} — trace and span names come from app.observability.modules")


def langfuse_environment(app_env: str) -> str:
    """`APP_ENV` as Langfuse accepts it: lowercase [a-z0-9_-], ≤ 40 chars, not starting with 'langfuse'."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in (app_env or "").strip().lower())[:_MAX_ENVIRONMENT_CHARS]
    cleaned = cleaned.strip("-_")
    if not cleaned:
        return _DEFAULT_ENVIRONMENT
    if cleaned.startswith("langfuse"):
        cleaned = f"env-{cleaned}"[:_MAX_ENVIRONMENT_CHARS]
    return cleaned


# --------------------------------------------------------------------------- #
# Process-level instance
# --------------------------------------------------------------------------- #

_telemetry: Telemetry = Telemetry()


def get_telemetry() -> Telemetry:
    return _telemetry


def configure_telemetry(
    *,
    enabled: bool,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    timeout_seconds: int = 5,
    sample_rate: float = 1.0,
    environment: str = _DEFAULT_ENVIRONMENT,
    release: str = "",
    pipeline_version: str = "",
    config_fingerprint: str = "",
    capture_query_text: bool = False,
    span_exporter: Any = None,
    tracer_provider: Any = None,
) -> Telemetry:
    """
    Install the process telemetry. Disabled → the no-op. Enabled without both keys, or with a sample rate
    outside [0, 1], raises ValueError so a misconfigured service fails at startup.
    """
    global _telemetry
    if not enabled:
        _telemetry = Telemetry(pipeline_version=pipeline_version, config_fingerprint=config_fingerprint, capture_query_text=capture_query_text)
        return _telemetry
    if not public_key or not secret_key:
        raise ValueError("LANGFUSE_ENABLED=true requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"LANGFUSE_SAMPLE_RATE must be within [0, 1], got {sample_rate}")
    _telemetry = LangfuseTelemetry(
        public_key=public_key, secret_key=secret_key, host=host, timeout_seconds=timeout_seconds, sample_rate=sample_rate,
        environment=langfuse_environment(environment), release=release, pipeline_version=pipeline_version,
        config_fingerprint=config_fingerprint, capture_query_text=capture_query_text,
        span_exporter=span_exporter, tracer_provider=tracer_provider,
    )
    return _telemetry


def set_telemetry(instance: Telemetry) -> Telemetry:
    """Install a ready instance (tests inject one with an in-memory exporter through `create_app`)."""
    global _telemetry
    _telemetry = instance
    return instance


def reset_telemetry() -> None:
    """Tests: shut the current instance down and go back to the no-op."""
    global _telemetry
    _telemetry.shutdown()
    _telemetry = Telemetry()
