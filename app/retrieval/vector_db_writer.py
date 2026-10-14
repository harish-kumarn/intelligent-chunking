"""
app.retrieval.vector_db_writer
==============================
Write-path helpers for EnhancedCohereMilvusRetriever: Documents ─► Milvus rows; stale-row cleanup.

    prepare_rows    : pure — strip, truncate, key, coerce; no Milvus, no network
    delete_stale    : a document's rows whose key is no longer among its chunks (re-ingesting a
                      shorter version must not leave the old version's tail searchable)
    flush_and_load  : seal and load, so new rows are immediately searchable (secondary: logged)

Rows are keyed by `metadata["chunk_id"]` (or a stable content hash), so re-ingesting
a document replaces its rows instead of duplicating them. Fixed columns are declared
in app.retrieval.milvus_schema; every other metadata key lands in the dynamic field
as native JSON (lists stay lists).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from langchain_core.documents import Document
from pymilvus import MilvusClient

from app.retrieval.milvus_schema import (
    F_DENSE,
    F_DOCUMENT_ID,
    F_PAGE,
    F_PK,
    F_SPARSE,
    F_TEXT,
    MAX_PK_LEN,
    MAX_TEXT_BYTES,
    PAGE_NULL,
)

__all__ = ["DEFAULT_BATCH_SIZE", "PreparedRow", "delete_stale", "flush_and_load", "prepare_rows"]

DEFAULT_BATCH_SIZE = 256
_SCAN_BATCH = 1000  # rows per page when listing a document's keys; also the delete batch
_RESERVED_KEYS = frozenset({F_PK, F_DOCUMENT_ID, F_PAGE, F_TEXT, F_DENSE, F_SPARSE})


@dataclass(frozen=True)
class PreparedRow:
    """One Milvus row minus its dense vector (added once the texts are embedded)."""

    chunk_id: str
    text: str
    fields: Dict[str, Any]  # document_id, page and the dynamic metadata

    def with_vector(self, vector: List[float]) -> Dict[str, Any]:
        return {F_PK: self.chunk_id, F_TEXT: self.text, F_DENSE: vector, **self.fields}


def prepare_rows(docs: Sequence[Document], logger: logging.LoggerAdapter) -> List[PreparedRow]:
    """
    One row per non-empty document, in order. Raises ValueError on duplicate chunk ids —
    one upsert batch must not overwrite its own rows.
    """
    rows = [row for row in (_prepare(doc, logger) for doc in docs) if row is not None]
    duplicates = [cid for cid, count in Counter(r.chunk_id for r in rows).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate chunk_ids in batch: {duplicates[:5]}")
    return rows


def flush_and_load(client: MilvusClient, collection_name: str, logger: logging.LoggerAdapter) -> None:
    try:
        client.flush(collection_name)
        client.load_collection(collection_name)
    except Exception as exc:  # rows are written; they become searchable once Milvus seals them
        logger.warning("Flush/load after upsert failed for '%s': %s", collection_name, exc)


def delete_stale(client: MilvusClient, collection_name: str, document_id: str, keep: Set[str]) -> int:
    """Delete `document_id`'s rows whose primary key is not in `keep`; returns how many. Errors propagate."""
    iterator = client.query_iterator(
        collection_name, batch_size=_SCAN_BATCH, filter=f"{F_DOCUMENT_ID} == {json.dumps(document_id)}", output_fields=[F_PK],
    )
    stale: List[str] = []
    try:
        while True:
            page = iterator.next()
            if not page:
                break
            stale.extend(str(row[F_PK]) for row in page if str(row[F_PK]) not in keep)
    finally:
        iterator.close()
    for start in range(0, len(stale), _SCAN_BATCH):
        client.delete(collection_name, ids=stale[start:start + _SCAN_BATCH])
    return len(stale)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _prepare(doc: Document, logger: logging.LoggerAdapter) -> Optional[PreparedRow]:
    text = (doc.page_content or "").strip()
    if not text:
        return None
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        logger.warning("Truncating an oversized chunk to %d bytes to fit VARCHAR.", MAX_TEXT_BYTES)
        text = _truncate_utf8(text, MAX_TEXT_BYTES)

    metadata = dict(doc.metadata or {})
    document_id = str(metadata.get(F_DOCUMENT_ID) or "")[:MAX_PK_LEN]
    chunk_id = str(metadata.get(F_PK) or _stable_chunk_id(text, metadata))[:MAX_PK_LEN]
    extras = {
        key: val for key, val in metadata.items()
        if key not in _RESERVED_KEYS and val is not None and not str(key).startswith("$")
    }
    fields = {
        F_DOCUMENT_ID: document_id,
        F_PAGE: _coerce_page(metadata.get(F_PAGE)),
        **extras,
        "doc_length": len(text),
        "word_count": len(text.split()),
    }
    return PreparedRow(chunk_id=chunk_id, text=text, fields=fields)


def _stable_chunk_id(text: str, metadata: Dict[str, Any]) -> str:
    """Deterministic, collision-resistant id — stable across processes and restarts."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(text.encode("utf-8", errors="replace"))
    for key in ("source", F_DOCUMENT_ID, "file_name", F_PAGE):
        val = metadata.get(key)
        if val is not None:
            hasher.update(b"\x1f" + str(val).encode("utf-8", errors="replace"))
    return f"ck_{hasher.hexdigest()}"


def _truncate_utf8(text: str, limit: int) -> str:
    """Cut `text` to at most `limit` UTF-8 bytes without splitting a code point."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def _coerce_page(value: Any) -> int:
    try:
        return int(value) if value is not None else PAGE_NULL
    except (TypeError, ValueError):
        return PAGE_NULL
