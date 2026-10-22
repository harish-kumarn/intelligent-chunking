"""
app.retrieval.reranker
======================
CohereReranker — reorders fused candidates by Cohere relevance.

    client  : optional; `enabled` is False without a key and reranking is a no-op
    run_fn  : caller-supplied awaitable offloader so the blocking call stays off the event loop

Failure policy: rate-limit / availability errors are retried with back-off; any
other failure logs a WARNING and returns the input order unchanged — the
caller detects "no rerank happened" by identity (`result is docs`). Candidates
without text are kept at the tail rather than dropped.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import cohere

from app.app_logging.setup import get_logger
from app.retrieval.cohere_retry import MAX_ATTEMPTS, RETRYABLE_EXCEPTIONS, backoff_seconds

__all__ = ["CohereReranker"]

Record = Dict[str, Any]
RunFn = Callable[..., Awaitable[Any]]


class CohereReranker:
    """Async wrapper over `cohere.Client.rerank`."""

    def __init__(self, api_key: Optional[str], model: str, logger: Optional[logging.LoggerAdapter] = None) -> None:
        self.model = model
        self._log = logger or get_logger(module="RERANKER")
        self._client = cohere.Client(api_key) if api_key else None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def rerank_async(self, query: str, docs: List[Record], top_k: int, run_fn: RunFn) -> List[Record]:
        """Returns re-scored copies best-first (+ text-less records appended); `docs` itself on failure/no-op."""
        rerankable = [d for d in docs if (d.get("text") or "").strip()]
        held_back = [d for d in docs if not (d.get("text") or "").strip()]
        if self._client is None or not rerankable:
            return docs
        if held_back:
            self._log.debug("%d candidate(s) without text bypass reranking", len(held_back))

        texts = [d["text"] for d in rerankable]
        top_n = max(1, min(top_k, len(texts)))
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await run_fn(
                    lambda: self._client.rerank(model=self.model, query=query, documents=texts, top_n=top_n),
                )
            except RETRYABLE_EXCEPTIONS as exc:
                if attempt == MAX_ATTEMPTS:
                    self._log.warning("Rerank gave up after %d throttled attempts: %s — keeping fused order", attempt, exc)
                    return docs
                wait = backoff_seconds(attempt)
                self._log.warning("Rerank throttled (attempt %d/%d): %s — retry in %ds", attempt, MAX_ATTEMPTS, exc, wait)
                await asyncio.sleep(wait)
                continue
            except Exception as exc:
                self._log.warning("Rerank failed (%s): %s — keeping fused order", type(exc).__name__, exc)
                return docs
            return [_to_reranked(rerankable[r.index], r.relevance_score) for r in response.results] + held_back
        return docs


def _to_reranked(record: Record, score: float) -> Record:
    return {**record, "rerank_score": score, "final_score": score, "retrieval_source": "reranked"}
