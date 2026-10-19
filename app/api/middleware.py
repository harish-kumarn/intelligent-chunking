"""
app.api.middleware
==================
RequestContextMiddleware — one request, one identity.

    request id : a fresh uuid4 in `request_id_var` for every request (routes reuse it; logs carry it);
                 returned as `X-Request-Id`
    execution  : `X-Execution-Kind`, `X-Execution-Id`, `X-Dataset-Name`, `X-Dataset-Version`, `X-Case-Id`
                 → `ExecutionContext` in `execution_var`, stamped on every trace the request opens; an
                 invalid value is a 422 `INVALID_EXECUTION_HEADER` before any work
    trace ids  : `trace_ids_var` holds a list the telemetry facade appends each root trace to; the first
                 one is returned as `X-Trace-Id` (absent when tracing is off or the request opened none)

Pure ASGI (no body buffering). Contextvars set here reach the route and, through Starlette's
`run_in_threadpool` (contextvars are copied into the worker), the synchronous work it offloads.
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, Dict

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

from app.logging.setup import request_id_var
from app.observability.context import InvalidExecutionHeader, execution_var, parse_execution_headers, trace_ids_var

__all__ = ["REQUEST_ID_HEADER", "TRACE_ID_HEADER", "RequestContextMiddleware"]

REQUEST_ID_HEADER = "X-Request-Id"
TRACE_ID_HEADER = "X-Trace-Id"

Scope = Dict[str, Any]
Receive = Callable[[], Awaitable[Dict[str, Any]]]
Send = Callable[[Dict[str, Any]], Awaitable[None]]


class RequestContextMiddleware:
    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        try:
            execution = parse_execution_headers(headers)
        except InvalidExecutionHeader as exc:
            response = JSONResponse(
                status_code=422,
                content={"detail": {"error": {"code": "INVALID_EXECUTION_HEADER", "message": str(exc), "retryable": False}}},
            )
            await response(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        tokens = (request_id_var.set(request_id), execution_var.set(execution), trace_ids_var.set([]))

        async def send_with_identity(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers.append(REQUEST_ID_HEADER, request_id)
                trace_ids = trace_ids_var.get() or []
                if trace_ids:
                    response_headers.append(TRACE_ID_HEADER, trace_ids[0])
            await send(message)

        try:
            await self.app(scope, receive, send_with_identity)
        finally:
            request_id_var.reset(tokens[0])
            execution_var.reset(tokens[1])
            trace_ids_var.reset(tokens[2])
