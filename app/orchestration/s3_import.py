"""
app.orchestration.s3_import
===========================
S3DocumentImporter — index a document already published to S3 into Milvus, with no OCR, chunking or LLM call.

    resolve  : `{document_id}/current.json` → the live version (its manifest must exist)
    chunks   : `merged_chunks.json` as published (with its categories); when absent, `target_chunks.json`
               merged locally (pure code — the result carries no categories)
    language : resolve_document_language — the collection's first document fixes it; later ones must match
    index    : stage 8 of IngestPipeline — Cohere + Milvus upsert, the document's stale rows removed
    register : the document is added to `_collections/{collection}.json` (best-effort; feeds reformulation)

Main path
    import_document ─► load_published_chunks (current version ─► chunks) ─► language ─► index_merged_chunks

`load_published_chunks` is shared with app.orchestration.s3_grounding; `published_chunk_ids` wraps it for
dataset reference checks (app.datasets). Each import is one `s3_reindex` trace.

Apart from the collection registry, nothing is written to S3, and nothing touches the local disk.

Failure policy: a missing folder, version or chunk artifact raises `S3DocumentNotFoundError` (HTTP 404);
a collection that cannot take the language raises the app.retrieval.errors types (409); S3, Cohere and
Milvus errors propagate (500).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from botocore.exceptions import BotoCoreError, ClientError

from app.config.defaults import ChunkerConfig
from app.ingestion.pipeline import index_merged_chunks, resolve_document_language
from app.logging.setup import get_logger
from app.observability.modules import ModuleName
from app.observability.telemetry import get_telemetry
from app.orchestration.errors import S3DocumentNotFoundError
from app.postprocessing.merger import MergedChunk, merge_chunks
from app.storage.s3_artifact_store import S3ArtifactStore, register_in_collection
from app.storage.s3_dataset_store import DatasetStoreError

__all__ = ["S3DocumentImporter", "S3ImportResult", "load_published_chunks", "published_chunk_ids"]

_MERGED_CHUNKS = "merged_chunks.json"
_TARGET_CHUNKS = "target_chunks.json"
_MANIFEST = "manifest.json"
_PAGE_IMAGE = re.compile(r"^page_images/page_\d+\.jpeg$")


@dataclass(frozen=True)
class S3ImportResult:
    document_id: str
    version: str
    source_artifact: str          # merged_chunks.json, or target_chunks.json when it had to be merged here
    collection_name: str
    language: str
    chunk_count: int
    page_count: int
    classified: bool              # the chunks carry main/sub categories


class S3DocumentImporter:
    def __init__(self, store: S3ArtifactStore, config: ChunkerConfig, logger: Optional[logging.LoggerAdapter] = None) -> None:
        self._store = store
        self._cfg = config
        self._log = logger or get_logger(module="S3_IMPORT")

    def import_document(self, document_id: str, *, collection_name: str, language: Optional[str] = None) -> S3ImportResult:
        with get_telemetry().trace(
            ModuleName.S3_REINDEX, document_id=document_id, collection_name=collection_name, language_hint=language,
        ) as trace:
            version, merged, source = load_published_chunks(self._store, document_id, self._log)
            trace.set(version=version, source_artifact=source, chunk_count=len(merged))
            text = "\n".join(str(chunk.get("content") or "") for chunk in merged)
            document_language = resolve_document_language(text, language, collection_name, config=self._cfg, logger=self._log)
            self._log.info(
                "S3 import: document=%s version=%s source=%s chunks=%d language=%s collection=%s",
                document_id, version, source, len(merged), document_language, collection_name,
            )
            index_merged_chunks(
                merged, collection_name, document_id=document_id, language=document_language, config=self._cfg, logger=self._log,
            )
            register_in_collection(self._store, collection_name, document_id, language=document_language, logger=self._log)
            result = S3ImportResult(
                document_id=document_id,
                version=version,
                source_artifact=source,
                collection_name=collection_name,
                language=document_language,
                chunk_count=len(merged),
                page_count=self._page_count(document_id, version, merged),
                classified=any("main_category" in chunk for chunk in merged),
            )
            trace.set_output(
                status="completed", language=result.language, chunk_count=result.chunk_count, page_count=result.page_count,
                classified=result.classified, uploaded_to_milvus=True,
            )
        return result

    def _page_count(self, document_id: str, version: str, merged: List[MergedChunk]) -> int:
        """Rendered page images listed in the manifest; else the distinct pages the chunks came from."""
        manifest = self._store.get_version_resource_json(document_id, version, _MANIFEST) or {}
        files = manifest.get("files") if isinstance(manifest, dict) else None
        rendered = sum(1 for item in files or [] if isinstance(item, dict) and _PAGE_IMAGE.match(str(item.get("path", ""))))
        if rendered:
            return rendered
        return len({page for chunk in merged for page in (chunk.get("metadata") or {}).get("source_page_nos") or []})


def load_published_chunks(
    store: S3ArtifactStore, document_id: str, logger: logging.LoggerAdapter,
) -> Tuple[str, List[MergedChunk], str]:
    """(current version, merged chunks, source artifact) of a published document; S3DocumentNotFoundError when absent."""
    current = store.get_current_version(document_id)
    version = current.get("version") if isinstance(current, dict) else None
    if not version:
        raise S3DocumentNotFoundError(f"No published document '{document_id}' in S3 (missing or invalid current.json)")
    if not store.version_exists(document_id, version):
        raise S3DocumentNotFoundError(f"Version {version} of '{document_id}' is not completely uploaded (no manifest)")

    merged = _records(store.get_version_resource_json(document_id, version, _MERGED_CHUNKS))
    if merged:
        return version, merged, _MERGED_CHUNKS
    logger.info("S3: %s missing or empty for %s; merging %s", _MERGED_CHUNKS, document_id, _TARGET_CHUNKS)
    target = _records(store.get_version_resource_json(document_id, version, _TARGET_CHUNKS))
    if target:
        return version, merge_chunks(target, logger), _TARGET_CHUNKS
    raise S3DocumentNotFoundError(f"'{document_id}' version {version} has neither {_MERGED_CHUNKS} nor {_TARGET_CHUNKS} with chunks")


def published_chunk_ids(store: S3ArtifactStore, logger: logging.LoggerAdapter) -> Callable[[str], Tuple[str, Set[str]]]:
    """
    `document_id → (current version, its merged chunk ids)` for dataset reference checks
    (app.datasets.service.ResolveDocument); LookupError when the document is not published.
    """

    def resolve(document_id: str) -> Tuple[str, Set[str]]:
        try:
            version, merged, _ = load_published_chunks(store, document_id, logger)
        except S3DocumentNotFoundError as exc:
            raise LookupError(document_id) from exc
        except (BotoCoreError, ClientError) as exc:  # S3 down is a 502 for the dataset call, not an invalid dataset
            raise DatasetStoreError(f"could not read published document '{document_id}' from s3://{store.bucket}") from exc
        return version, {str(chunk.get("chunk_id")) for chunk in merged}

    return resolve


def _records(data: Any) -> List[Dict[str, Any]]:
    """`{"chunk1": {...}, …}` in file (document) order; anything else reads as no chunks."""
    if not isinstance(data, dict):
        return []
    return [record for record in data.values() if isinstance(record, dict) and record.get("chunk_id")]
