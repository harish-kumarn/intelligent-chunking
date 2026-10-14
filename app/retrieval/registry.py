"""
app.retrieval.registry
======================
RetrieverRegistry — one long-lived `EnhancedCohereMilvusRetriever` per collection.

A retriever owns a thread pool and live Milvus/Cohere clients, so it must not
be created per request. The registry creates each lazily on first use, hands
the same instance to every later query for that collection, evicts one when its
collection is dropped, and closes them all at shutdown. Thread-safe.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

from app.config.defaults import ChunkerConfig
from app.retrieval.config import RetrievalConfig
from app.retrieval.vector_db import EnhancedCohereMilvusRetriever

__all__ = ["RetrieverRegistry"]


class RetrieverRegistry:
    def __init__(
        self,
        base_config: ChunkerConfig,
        retrieval_config: Optional[RetrievalConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._base_config = base_config
        self._retrieval_config = retrieval_config or RetrievalConfig()
        self._logger = logger
        self._lock = threading.Lock()
        self._retrievers: Dict[str, EnhancedCohereMilvusRetriever] = {}

    def get(self, collection_name: str) -> EnhancedCohereMilvusRetriever:
        with self._lock:
            retriever = self._retrievers.get(collection_name)
            if retriever is None:
                retriever = EnhancedCohereMilvusRetriever(
                    collection_name,
                    config=self._base_config,
                    retrieval_config=self._retrieval_config,
                    logger=self._logger,
                )
                self._retrievers[collection_name] = retriever
            return retriever

    def evict(self, collection_name: str) -> bool:
        """Close and forget the retriever for `collection_name` (after a drop). True if one was cached."""
        with self._lock:
            retriever = self._retrievers.pop(collection_name, None)
        if retriever is None:
            return False
        retriever.close()
        return True

    def close_all(self) -> None:
        with self._lock:
            retrievers, self._retrievers = list(self._retrievers.values()), {}
        for retriever in retrievers:
            retriever.close()  # idempotent; each swallows its own client-close errors

    def __len__(self) -> int:
        return len(self._retrievers)
