"""
app.datasets.langfuse_sync
==========================
Mirror one S3 dataset version into a Langfuse dataset, for Langfuse's experiment UI and item → trace links.

    dataset : named `{module_name}/{dataset_name}`; its metadata records the S3 object key, the dataset
              version and checksum it mirrors (and the schema id), so the mirror always names its source
    items   : one per row, id = sha256("{module}/{dataset}|{case_id}")[:32] — stable across versions, so a
              re-sync upserts in place (Langfuse upserts by id); the item metadata carries the version
    archive : items of the Langfuse dataset that the synced version does not contain are archived, so
              "latest" in Langfuse always equals exactly one S3 version; `synced_at` is recorded as the
              Langfuse version timestamp (`get_dataset(name, version=synced_at)` reproduces it)
    cap     : at most `max_items` rows are mirrored (one API call per item); larger datasets are skipped

S3 stays canonical — nothing is read back from Langfuse to decide anything.

Failure policy: secondary. Any failure (network, API, cap) is returned as a `SyncResult` with
`status="failed"` / `"skipped"` and logged; the dataset upload has already succeeded and stays so.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Set

from app.datasets.schema import CHUNK_ID_SEPARATOR
from app.observability.sanitize import classify_error

__all__ = ["SyncResult", "langfuse_dataset_name", "langfuse_item_id", "sync_dataset_version"]

_ARCHIVED = "ARCHIVED"
_ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class SyncResult:
    status: str                              # synced | skipped | failed
    item_count: int = 0
    archived_count: int = 0
    langfuse_dataset: Optional[str] = None
    synced_at: Optional[str] = None          # the Langfuse dataset version timestamp (UTC ISO-8601)
    reason: Optional[str] = None
    error_type: Optional[str] = None
    error_category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def langfuse_dataset_name(module_name: str, dataset_name: str) -> str:
    return f"{module_name}/{dataset_name}"


def langfuse_item_id(module_name: str, dataset_name: str, case_id: str) -> str:
    return hashlib.sha256(f"{module_name}/{dataset_name}|{case_id}".encode("utf-8")).hexdigest()[:32]


def sync_dataset_version(
    client: Any,
    *,
    module_name: str,
    dataset_name: str,
    dataset_version: str,
    checksum_sha256: str,
    schema_id: str,
    s3_bucket: str,
    csv_key: str,
    row_count: int,
    rows: Iterable[Mapping[str, str]],
    max_items: int,
    logger: logging.LoggerAdapter,
) -> SyncResult:
    name = langfuse_dataset_name(module_name, dataset_name)
    if row_count > max_items:
        logger.warning("Langfuse dataset sync skipped: %s has %d rows, cap is %d", name, row_count, max_items)
        return SyncResult("skipped", langfuse_dataset=name, reason=f"row_count {row_count} exceeds LANGFUSE_DATASET_SYNC_MAX_ITEMS {max_items}")

    synced: Set[str] = set()
    archived = 0
    try:
        client.create_dataset(
            name=name,
            description=f"Mirror of s3://{s3_bucket}/{csv_key} (schema {schema_id}); S3 is canonical.",
            metadata={
                "s3_bucket": s3_bucket, "s3_key": csv_key, "dataset_version": dataset_version, "checksum_sha256": checksum_sha256,
                "dataset_schema_version": schema_id, "module_name": module_name, "dataset_name": dataset_name,
            },
        )
        for row in rows:
            case_id = row.get("case_id", "")
            item_id = langfuse_item_id(module_name, dataset_name, case_id)
            client.create_dataset_item(
                dataset_name=name,
                id=item_id,
                input={"question": row.get("question", "")},
                expected_output={
                    "answer_chunk_ids": [part for part in row.get("answer_chunk_id", "").split(CHUNK_ID_SEPARATOR) if part],
                    "answer": row.get("answer", ""),
                },
                metadata={
                    "case_id": case_id, "document_id": row.get("document_id", ""), "page": row.get("page", ""),
                    "type": row.get("type", ""), "tag": row.get("tag", ""), "language": row.get("language", ""),
                    "dataset_version": dataset_version, "checksum_sha256": checksum_sha256,
                },
                status=_ACTIVE,
            )
            synced.add(item_id)
        for item in _existing_items(client, name):
            if item.id in synced or str(getattr(item, "status", _ACTIVE)) != _ACTIVE:
                continue
            client.create_dataset_item(
                dataset_name=name, id=item.id, input=item.input, expected_output=item.expected_output, metadata=item.metadata,
                status=_ARCHIVED,
            )
            archived += 1
        synced_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception as exc:  # secondary path: the S3 version is already published
        described = classify_error(exc)
        logger.warning(
            "Langfuse dataset sync failed: %s@%s after %d item(s): %s", name, dataset_version, len(synced), described.get("error_type"),
            extra={"error_type": described.get("error_type"), "status": "failed"},
        )
        return SyncResult(
            "failed", item_count=len(synced), archived_count=archived, langfuse_dataset=name,
            reason="langfuse_error", error_type=described.get("error_type"), error_category=described.get("error_category"),
        )
    logger.info("Langfuse dataset synced: %s@%s items=%d archived=%d", name, dataset_version, len(synced), archived)
    return SyncResult("synced", item_count=len(synced), archived_count=archived, langfuse_dataset=name, synced_at=synced_at)


def _existing_items(client: Any, name: str) -> Iterable[Any]:
    dataset = client.get_dataset(name)
    return list(getattr(dataset, "items", None) or [])
