"""
app.retrieval.reformulation
===========================
QueryReformulationService — rewrites a user query in the vocabulary of a collection's documents.

    S3ArtifactStore : the collection registry `_collections/{collection}.json` lists its documents; each
                      document's current version supplies its vocab_index.json. A collection with no
                      registry (ingested before it existed) is read as the single document named like it.
    Groq            : rewrites the query given the combined vocabulary
    caches          : one S3 store per (bucket, region); one combined vocabulary per (bucket, collection),
                      kept `_VOCAB_TTL_SECONDS` so a newly ingested document is picked up soon after

Main path
    reformulate ─► collection vocabulary (cached: registry ─► per document: current version ─► vocab) ─► Groq rewrite

Failure policy: every step degrades to returning the original query (logged at
WARNING/INFO with the stage) — reformulation must never block retrieval.
Concurrency: safe to share across threads; caches are guarded by a lock and
nothing per-request is stored on the instance.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from groq import Groq

from app.config.defaults import ChunkerConfig
from app.logging.setup import get_logger
from app.storage.s3_artifact_store import S3ArtifactStore

__all__ = ["QueryReformulationService", "flatten_vocabulary"]

_VOCAB_INDEX_RESOURCE = "vocab_index.json"
_SYSTEM_PROMPT = (
    "You are a precise search query reformulation assistant. You rewrite user queries to match "
    "document vocabulary. Output ONLY the rewritten query."
)
_USER_PROMPT = """\
You will be given a list of vocabulary index terms from a document collection, followed by a user search query.
Your task is to rewrite (reformulate) the user query to use the terminology present in the vocabulary index, which will improve retrieval accuracy (keyword/vector matching) in a RAG pipeline.

<vocabulary_index_terms>
{vocabulary}
</vocabulary_index_terms>

<user_query>
{query}
</user_query>

RULES:
1. Align the user's terms with synonyms, acronyms, or specific technical phrases from the vocabulary index.
2. Maintain the user's core search intent. Do not answer the query.
3. Keep the output query concise, clear, and optimized for search.
4. Do not include any fences, markdown, explanations, preamble, or trailing text. Output ONLY the reformulated query text.
"""
_QUOTES = ('"', "'")
_VOCAB_TTL_SECONDS = 60.0


def flatten_vocabulary(data: Any) -> str:
    """Every scalar in a (nested) JSON value, case-insensitively deduplicated, space-joined in first-seen order."""
    seen, terms = set(), []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            cleaned = str(value).strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                terms.append(cleaned)

    walk(data)
    return " ".join(terms)


class QueryReformulationService:
    """Process-scoped; construct once and share."""

    def __init__(self, config: ChunkerConfig, logger: Optional[logging.LoggerAdapter] = None) -> None:
        self._cfg = config
        self._log = logger or get_logger(module="REFORMULATION")
        self._lock = threading.Lock()
        self._stores: Dict[Tuple[str, Optional[str]], S3ArtifactStore] = {}
        self._vocab: Dict[Tuple[str, str], Tuple[float, str]] = {}  # (bucket, collection) → (expires, flattened terms)
        self._groq = Groq(api_key=config.groq_api_key) if config.groq_api_key else None
        if self._groq is None:
            self._log.warning("GROQ_API_KEY unset — query reformulation will return queries unchanged")

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def reformulate(self, query: str, collection_name: str, *, s3_bucket: str, AWS_DEFAULT_REGION: Optional[str]) -> str:
        """Returns the rewritten query, or `query` unchanged when any step is unavailable. Never raises."""
        query = query.strip()
        if self._groq is None:
            return query
        store = self._store(s3_bucket, AWS_DEFAULT_REGION)
        vocabulary = self._collection_vocabulary(store, s3_bucket, collection_name)
        if not vocabulary:
            return query

        try:
            rewritten = self._rewrite(query, vocabulary)
        except Exception as exc:
            self._skip("rewrite", "failed", collection_name, error=exc)
            return query
        self._log.info("Query reformulated: collection=%s %r -> %r", collection_name, query, rewritten)
        return rewritten or query

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _store(self, bucket: str, region: Optional[str]) -> S3ArtifactStore:
        key = (bucket, region)
        with self._lock:
            store = self._stores.get(key)
            if store is None:
                store = self._stores[key] = S3ArtifactStore(bucket=bucket, region=region)
        return store

    def _collection_vocabulary(self, store: S3ArtifactStore, bucket: str, collection_name: str) -> str:
        """Every registered document's vocabulary, flattened; "" (logged) when none can be read."""
        key = (bucket, collection_name)
        with self._lock:
            cached = self._vocab.get(key)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        try:
            documents = store.collection_documents(collection_name) or [collection_name]
        except Exception as exc:
            self._skip("collection_registry", "failed", collection_name, error=exc)
            return ""

        vocabularies = []
        for document_id in documents:
            try:
                current = store.get_current_version(document_id)
                version = current.get("version") if isinstance(current, dict) else None
                data = store.get_version_resource_json(document_id, version, _VOCAB_INDEX_RESOURCE) if version else None
            except Exception as exc:  # one unreadable document does not hide the others
                self._skip("vocabulary_load", "failed", collection_name, document_id=document_id, error=exc)
                continue
            if data is not None:
                vocabularies.append(data)

        vocabulary = flatten_vocabulary(vocabularies)
        if not vocabulary:
            self._skip("vocabulary_load", "empty", collection_name)
            return ""
        with self._lock:
            self._vocab[key] = (time.monotonic() + _VOCAB_TTL_SECONDS, vocabulary)
        return vocabulary

    def _rewrite(self, query: str, vocabulary: str) -> str:
        completion = self._groq.chat.completions.create(  # type: ignore[union-attr]
            model=self._cfg.requery_model,
            temperature=self._cfg.requery_temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT.format(vocabulary=vocabulary, query=query)},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in _QUOTES:
            text = text[1:-1].strip()
        return text

    def _skip(
        self, stage: str, status: str, collection_name: str, *, document_id: Optional[str] = None, error: Optional[Exception] = None,
    ) -> None:
        extra: Dict[str, Any] = {"channel": "retrieval", "stage": stage, "status": status, "collection_name": collection_name}
        if document_id:
            extra["document_id"] = document_id
        if error is not None:
            extra.update(error_type=type(error).__name__, error_message=str(error))
            self._log.warning("Query reformulation skipped at %s — returning original query", stage, extra=extra)
        else:
            self._log.info("Query reformulation skipped at %s (%s) — returning original query", stage, status, extra=extra)
