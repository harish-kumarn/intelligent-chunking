"""
app.chunking.token_monitor
==========================
TokenUsageMonitor — per-page Gemini token accounting persisted to JSON.

    file shape : {"model", "started_at", "pages": {page: {"attempts": [...]}}, "summary": {...}}

Flushed after every `record()` so partial runs keep their accounting. A
monitor whose file already exists for the same model resumes its totals.
Thread-safe; flush failures are logged, never raised.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.logging.setup import get_logger
from app.utils.fs import atomic_write_json

__all__ = ["TokenUsageMonitor"]

_SUMMARY_KEYS = (
    "cached_input_tokens", "uncached_input_tokens", "total_input_tokens",
    "candidate_output_tokens", "thinking_tokens", "total_output_tokens", "total_tokens",
)


class TokenUsageMonitor:
    """Records token usage per page attempt and keeps a running summary."""

    def __init__(self, output_path: Path, model: str, logger: Optional[logging.LoggerAdapter] = None) -> None:
        self.output_path = Path(output_path)
        self._log = logger or get_logger(module="TOKEN_USAGE")
        self._lock = threading.Lock()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._resolve_existing(model) or self._new_data(model)
        self._flush()

    def record(
        self,
        *,
        page_number: int,
        attempt: int,
        prompt_tokens: int,
        candidate_tokens: int,
        cached_tokens: int = 0,
        thinking_tokens: int = 0,
    ) -> None:
        """Append one attempt's usage and flush."""
        entry = {
            "attempt": attempt,
            "cached_input_tokens": cached_tokens,
            "uncached_input_tokens": max(0, prompt_tokens - cached_tokens),
            "total_input_tokens": prompt_tokens,
            "candidate_output_tokens": candidate_tokens,
            "thinking_tokens": thinking_tokens,
            "total_output_tokens": candidate_tokens + thinking_tokens,
            "total_tokens": prompt_tokens + candidate_tokens + thinking_tokens,
            "recorded_at": _utcnow(),
        }
        with self._lock:
            self._data["pages"].setdefault(str(page_number), {"attempts": []})["attempts"].append(entry)
            summary = self._data["summary"]
            for key in _SUMMARY_KEYS:
                summary[key] += entry[key]
            self._flush()
        self._log.debug(
            "Page %d attempt %d: cached=%d uncached=%d output=%d thinking=%d",
            page_number, attempt, cached_tokens, entry["uncached_input_tokens"], candidate_tokens, thinking_tokens,
        )

    def summary(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._data["summary"])

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _new_data(self, model: str) -> Dict[str, Any]:
        return {"model": model, "started_at": _utcnow(), "pages": {}, "summary": {key: 0 for key in _SUMMARY_KEYS}}

    def _resolve_existing(self, model: str) -> Optional[Dict[str, Any]]:
        """Resume an on-disk file when it belongs to the same model and is well-formed."""
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            return None
        try:
            data = json.loads(self.output_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            self._log.warning("Ignoring unreadable token usage file %s: %s", self.output_path, exc)
            return None
        if data.get("model") != model or not isinstance(data.get("pages"), dict) or not isinstance(data.get("summary"), dict):
            return None
        data["summary"] = {key: int(data["summary"].get(key, 0)) for key in _SUMMARY_KEYS}
        self._log.info("Resuming token usage from %s", self.output_path)
        return data

    def _flush(self) -> None:
        try:
            atomic_write_json(self.output_path, self._data)
        except OSError as exc:
            self._log.error("Failed to write token usage to %s: %s", self.output_path, exc)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
