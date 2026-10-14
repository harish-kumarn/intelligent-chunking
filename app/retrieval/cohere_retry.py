"""
app.retrieval.cohere_retry
==========================
Retry policy shared by every Cohere call (embed, rerank): which errors are
transient and how long to back off. Control flow stays with each caller —
embeddings raise when retries run out, the reranker degrades to fused order.
"""

from __future__ import annotations

from typing import Tuple

import cohere

__all__ = ["MAX_ATTEMPTS", "RETRYABLE_EXCEPTIONS", "backoff_seconds"]

MAX_ATTEMPTS = 7
_MAX_BACKOFF_SECONDS = 60
RETRYABLE_EXCEPTIONS: Tuple[type, ...] = tuple(
    exc for exc in (getattr(cohere, name, None) for name in ("TooManyRequestsError", "ServiceUnavailableError"))
    if isinstance(exc, type)
)


def backoff_seconds(attempt: int) -> int:
    """Exponential back-off for the 1-based `attempt`, capped at a minute."""
    return min(2 ** attempt, _MAX_BACKOFF_SECONDS)
