"""
app.concurrency
===============
ProcessingGate — single-flight admission for PDF ingestion.

A limit, not a queue: a request that cannot acquire the gate is rejected
immediately (HTTP 429). One document at a time keeps CPU/RAM bounded; retrieval
queries are not gated. Assumes one worker process (see Dockerfile).
"""

from __future__ import annotations

import threading

__all__ = ["ProcessingGate"]


class ProcessingGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        self._lock.release()

    @property
    def busy(self) -> bool:
        return self._lock.locked()
