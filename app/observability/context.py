"""
app.observability.context
=========================
ExecutionContext — who asked for a run and under which dataset, carried as a context variable next to
`request_id_var` / `document_id_var` so every trace the run opens is stamped with the same identifiers.

    execution_kind : production | ingestion_test | retrieval_test | evaluation   (`X-Execution-Kind`)
    execution_id   : groups traces of one run (an evaluation sweep, a job's attempts); the Langfuse
                     session id. Defaults to the request id, or the job id for a job   (`X-Execution-Id`)
    dataset_name / dataset_version / case_id : the ground-truth row a request replays   (`X-Dataset-*`, `X-Case-Id`)
    attempt        : the run's attempt number (ingestion jobs); 1 elsewhere

Headers are parsed by `parse_execution_headers` (the API middleware answers 422 on a bad value);
`to_dict` / `from_dict` persist the context with an ingestion job so a retry or a restart keeps it.
`trace_ids_var` is a per-request list the facade appends every root trace id to, so the HTTP layer
can return `X-Trace-Id` and a job can record the trace of each attempt.
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Mapping, Optional

from app.observability.modules import ExecutionKind

__all__ = [
    "EXECUTION_HEADERS", "ID_PATTERN", "ExecutionContext", "InvalidExecutionHeader", "current_execution",
    "execution_var", "parse_execution_headers", "trace_ids_var",
]

ID_PATTERN = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")
ID_RULE = "1-128 characters: letters, digits and . _ : @ -"

#: header → ExecutionContext field
EXECUTION_HEADERS: Dict[str, str] = {
    "x-execution-kind": "execution_kind",
    "x-execution-id": "execution_id",
    "x-dataset-name": "dataset_name",
    "x-dataset-version": "dataset_version",
    "x-case-id": "case_id",
}


class InvalidExecutionHeader(ValueError):
    """A request header carried a value outside the allowed form; `header` names it."""

    def __init__(self, header: str, message: str) -> None:
        self.header = header
        super().__init__(f"{header}: {message}")


@dataclass(frozen=True)
class ExecutionContext:
    execution_kind: ExecutionKind = ExecutionKind.PRODUCTION
    execution_id: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    case_id: Optional[str] = None
    attempt: int = 1

    def with_defaults(self, *, execution_id: Optional[str] = None, attempt: Optional[int] = None) -> "ExecutionContext":
        """Fill `execution_id` when the caller gave none; set `attempt` when given."""
        changes: Dict[str, Any] = {}
        if self.execution_id is None and execution_id:
            changes["execution_id"] = execution_id
        if attempt is not None:
            changes["attempt"] = attempt
        return replace(self, **changes) if changes else self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["execution_kind"] = self.execution_kind.value
        return data

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ExecutionContext":
        """Tolerant: a missing or unreadable record is the default context."""
        if not isinstance(data, Mapping):
            return cls()
        try:
            kind = ExecutionKind(str(data.get("execution_kind") or ExecutionKind.PRODUCTION))
        except ValueError:
            kind = ExecutionKind.PRODUCTION
        return cls(
            execution_kind=kind,
            execution_id=_optional_id(data.get("execution_id")),
            dataset_name=_optional_id(data.get("dataset_name")),
            dataset_version=_optional_id(data.get("dataset_version")),
            case_id=_optional_id(data.get("case_id")),
            attempt=max(1, int(data.get("attempt") or 1)),
        )


execution_var: contextvars.ContextVar[Optional[ExecutionContext]] = contextvars.ContextVar("execution", default=None)
trace_ids_var: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar("trace_ids", default=None)


def current_execution() -> ExecutionContext:
    return execution_var.get() or ExecutionContext()


def parse_execution_headers(headers: Mapping[str, str]) -> ExecutionContext:
    """
    The execution context a request declares; header names are matched case-insensitively.
    Raises InvalidExecutionHeader for an unknown kind or an id outside `ID_PATTERN`.
    """
    lowered = {str(key).lower(): value for key, value in headers.items()}
    values: Dict[str, Any] = {}
    for header, field in EXECUTION_HEADERS.items():
        raw = lowered.get(header)
        if raw is None or not str(raw).strip():
            continue
        raw = str(raw).strip()
        if field == "execution_kind":
            try:
                values[field] = ExecutionKind(raw.lower())
            except ValueError as exc:
                raise InvalidExecutionHeader(header, f"must be one of {[k.value for k in ExecutionKind]}") from exc
            continue
        if not ID_PATTERN.match(raw):
            raise InvalidExecutionHeader(header, f"must be {ID_RULE}")
        values[field] = raw
    return ExecutionContext(**values)


def _optional_id(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text if text and ID_PATTERN.match(text) else None
