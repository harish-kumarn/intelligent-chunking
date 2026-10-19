"""
app.observability
=================
Module identity, execution context and the Langfuse-backed tracing facade. A leaf package next to
`app.logging` (it imports nothing else from `app`), so every layer may use it.

    modules    : ModuleName / SpanName / ExecutionKind — the only names a trace may carry
    context    : ExecutionContext (+ header parsing) carried in a context variable across threads
    telemetry  : Telemetry facade — `get_telemetry().trace(...)` / `.span(...)` / `.generation(...)`
    sanitize   : the metadata allowlist, error classification, query fingerprints
    versioning : pipeline_version and the configuration fingerprint stamped on every trace
"""

from app.observability.context import ExecutionContext, current_execution, execution_var, parse_execution_headers, trace_ids_var
from app.observability.modules import ExecutionKind, ModuleName, SpanName
from app.observability.telemetry import Telemetry, configure_telemetry, get_telemetry

__all__ = [
    "ExecutionContext", "ExecutionKind", "ModuleName", "SpanName", "Telemetry", "configure_telemetry", "current_execution",
    "execution_var", "get_telemetry", "parse_execution_headers", "trace_ids_var",
]
