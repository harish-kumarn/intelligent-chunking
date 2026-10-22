"""
app.app_logging.setup
=================
One application logger, two renderings.

    configure_logging : called once at startup; installs handlers on LOGGER_NAME
    get_logger        : per-module adapter that stamps `log_module` on every record
    request_id_var    : contextvars populated by the API layer so every record
    document_id_var     emitted during a request carries its identity
    in_context        : wraps a callable so a worker thread logs with the submitter's context

Development / debug environments render human-readable lines to stdout plus a
rotating file; every other environment renders one JSON object per line to stdout.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Callable, Optional, TypeVar

__all__ = ["LOGGER_NAME", "configure_logging", "document_id_var", "get_logger", "in_context", "request_id_var"]

LOGGER_NAME = "vision_rag_api"

_DEV_ENVIRONMENTS = frozenset({"development", "debug", "local"})
_LOG_FILE_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB per file before rotation
_LOG_FILE_BACKUPS = 5

# Structured fields a caller may pass via `extra={...}`; anything else is dropped.
_OPTIONAL_FIELDS = (
    "operation", "channel", "stage", "status", "duration_ms",
    "page_number", "page_count", "chunk_count",
    "error_type", "error_message",
    "external_service", "bucket", "s3_prefix", "uploaded_file_count",
    "collection_name", "version_id",
)

request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
document_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("document_id", default=None)


# --------------------------------------------------------------------------- #
# Filters & formatters
# --------------------------------------------------------------------------- #

class _ContextFilter(logging.Filter):
    """Copy the request/document context variables onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.document_id = document_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line — for log shippers."""

    def __init__(self, service: str, environment: str) -> None:
        super().__init__()
        self._service = service
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "service": self._service,
            "environment": self._environment,
            "log_module": getattr(record, "log_module", "UNKNOWN"),
            "request_id": getattr(record, "request_id", None),
            "document_id": getattr(record, "document_id", None),
            "message": record.getMessage(),
        }
        payload.update(_structured_fields(record))
        if record.exc_info and "error_type" not in payload:
            exc_type = record.exc_info[0]
            payload["error_type"] = exc_type.__name__ if exc_type else "Exception"
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single line — for local development."""

    def format(self, record: logging.LogRecord) -> str:
        fields = []
        for key in ("request_id", "document_id"):
            value = getattr(record, key, None)
            if value is not None:
                fields.append(f"{key}={value}")
        fields.extend(f"{key}={value}" for key, value in _structured_fields(record).items())
        context = f" | {' '.join(fields)}" if fields else ""
        return f"{record.levelname}: [{getattr(record, 'log_module', 'UNKNOWN')}]: {record.getMessage()}{context}"


def _structured_fields(record: logging.LogRecord) -> dict:
    return {key: getattr(record, key) for key in _OPTIONAL_FIELDS if getattr(record, key, None) is not None}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def configure_logging(service_name: str, environment: str, *, level: str = "INFO", log_dir: str = "logs") -> None:
    """Install stdout (+ rotating file in dev) handlers on the application logger. Idempotent."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    environment = environment.lower()
    dev = environment in _DEV_ENVIRONMENTS

    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter: logging.Formatter = ConsoleFormatter() if dev else JsonFormatter(service_name, environment)
    context_filter = _ContextFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    console.addFilter(context_filter)
    logger.addHandler(console)

    if dev:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{service_name}_{datetime.now():%d-%m-%Y}.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=_LOG_FILE_MAX_BYTES, backupCount=_LOG_FILE_BACKUPS, encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)


class _ModuleAdapter(logging.LoggerAdapter):
    """Stamps `log_module` while keeping any `extra={...}` the caller passes (stdlib adapters drop it before 3.13)."""

    def process(self, msg: str, kwargs: dict) -> tuple:
        kwargs["extra"] = {**self.extra, **(kwargs.get("extra") or {})}
        return msg, kwargs


def get_logger(module: str) -> logging.LoggerAdapter:
    """Adapter on the application logger that tags records with `log_module=MODULE`."""
    return _ModuleAdapter(logging.getLogger(LOGGER_NAME), {"log_module": module.upper()})


T = TypeVar("T")


def in_context(fn: Callable[..., T]) -> Callable[..., T]:
    """
    `fn` bound to the caller's contextvars, for thread pools: threads start with an empty context,
    so without this a page chunked on a worker thread would log without its document id.
    """
    context = contextvars.copy_context()

    def run(*args: object, **kwargs: object) -> T:
        return context.copy().run(fn, *args, **kwargs)  # one copy per call: a Context is not re-entrant

    return run
