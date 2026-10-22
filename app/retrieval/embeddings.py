"""
app.retrieval.embeddings
========================
CohereEmbeddings — LangChain-compatible embeddings adapter over the Cohere embed API.

    embed_documents : batched (`batch_size` ≤ 96), input_type=search_document
    embed_query     : single text, input_type=search_query

Failure policy: rate-limit and availability errors are retried with exponential
back-off; anything else, and the last retry, propagates. A missing key fails
at construction.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, TypeVar

import cohere

from app.app_logging.setup import get_logger
from app.retrieval.cohere_retry import MAX_ATTEMPTS, RETRYABLE_EXCEPTIONS, backoff_seconds

__all__ = ["CohereEmbeddings"]

_LEGACY_MODEL_MARKER = "-v2"  # embed-*-v2.0 predates `input_type`; v3 and later require it

T = TypeVar("T")


class CohereEmbeddings:
    """Dense embeddings from Cohere (`embed_documents` / `embed_query`, the LangChain embeddings shape)."""

    def __init__(self, model: str, batch_size: int, api_key: Optional[str]) -> None:
        if not api_key:
            raise RuntimeError("COHERE_API_KEY is required for dense embeddings.")
        self.model = model
        self.batch_size = batch_size
        self._client = cohere.Client(api_key)
        self._log = get_logger(module="EMBEDDINGS")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self.batch_size], "search_document"))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text], "search_query")[0]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _embed_batch(self, texts: List[str], input_type: str) -> List[List[float]]:
        kwargs = {} if _LEGACY_MODEL_MARKER in self.model else {"input_type": input_type}
        response = self._with_retry(lambda: self._client.embed(model=self.model, texts=texts, **kwargs))
        return response.embeddings

    def _with_retry(self, call: Callable[[], T]) -> T:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return call()
            except RETRYABLE_EXCEPTIONS as exc:
                if attempt == MAX_ATTEMPTS:
                    raise
                wait = backoff_seconds(attempt)
                self._log.warning("Cohere embed throttled (attempt %d/%d): %s — retry in %ds", attempt, MAX_ATTEMPTS, exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")  # loop always returns or raises
