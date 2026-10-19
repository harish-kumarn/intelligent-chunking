"""
app.datasets.service
====================
DatasetService — publish, inspect and promote ground-truth dataset versions.

    publish  : validate (schema, limits) ─► [verify references: every `document_id` is published and every
               `answer_chunk_id` is one of its merged chunks] ─► manifest ─► S3DatasetStore.put_version
               (immutable, content-addressed) ─► [promote] ─► [Langfuse mirror; `langfuse_sync.json` sidecar]
    info     : the current pointer and every complete version's manifest
    promote  : move `current.json` to a complete version

The reference check needs the published chunks of a document, which live above this package
(`app.orchestration.s3_import.load_published_chunks`): the caller injects `resolve_document`
(`document_id → (version, chunk ids)`, LookupError when unknown). Without it, `verify_references`
is refused rather than silently skipped.

Traced as `ModuleName.DATASET_MANAGEMENT` with `dataset_validate` / `dataset_store` /
`langfuse_dataset_sync` spans; row contents never reach a trace, only counts and ids.

Failure policy: validation and reference errors raise `DatasetValidationError` (422, issues without cell
values); storage errors raise the `app.storage.s3_dataset_store` types (409 / 404 / 502); the Langfuse
mirror is secondary (its outcome is reported, never raised).
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from app.datasets.langfuse_sync import SyncResult, sync_dataset_version
from app.datasets.schema import DatasetSchema, schema_for
from app.datasets.validation import DatasetValidationError, ValidatedDataset, ValidationIssue, validate_csv
from app.logging.setup import get_logger
from app.observability.modules import ModuleName, SpanName
from app.observability.telemetry import Telemetry, get_telemetry
from app.storage.s3_dataset_store import DatasetNotFoundError, S3DatasetStore, dataset_version_for

__all__ = ["DatasetPublication", "DatasetService", "ResolveDocument"]

ResolveDocument = Callable[[str], Tuple[str, Set[str]]]  # document_id → (published version, merged chunk ids)
MAX_REFERENCE_DOCUMENTS = 50
_SIDECAR_SYNC = "langfuse_sync.json"
_GENERATOR_UPLOAD = "upload"


@dataclass(frozen=True)
class DatasetPublication:
    module_name: str
    dataset_name: str
    dataset_version: str
    created: bool
    row_count: int
    checksum_sha256: str
    s3_keys: List[str]
    promoted: bool
    manifest: Dict[str, Any]
    langfuse_sync: Optional[Dict[str, Any]]


class DatasetService:
    def __init__(
        self,
        store: S3DatasetStore,
        *,
        max_rows: int,
        pipeline_version: str,
        service_version: str,
        resolve_document: Optional[ResolveDocument] = None,
        langfuse_sync_max_items: int = 5000,
        telemetry: Optional[Telemetry] = None,
        logger: Optional[logging.LoggerAdapter] = None,
    ) -> None:
        self._store = store
        self._max_rows = max_rows
        self._pipeline_version = pipeline_version
        self._service_version = service_version
        self._resolve_document = resolve_document
        self._sync_max_items = langfuse_sync_max_items
        self._telemetry = telemetry or get_telemetry()
        self._log = logger or get_logger(module="DATASETS")

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def publish(
        self,
        *,
        module: ModuleName,
        dataset_name: str,
        source: Path,
        purpose: Optional[str] = None,
        label: Optional[str] = None,
        collection_name: Optional[str] = None,
        promote: bool = False,
        verify_references: bool = True,
        sync_langfuse: bool = False,
        generator: Optional[Dict[str, Any]] = None,
        source_filename: Optional[str] = None,
    ) -> DatasetPublication:
        schema = schema_for(module)
        canonical_path = source.with_name(f"{source.stem}.canonical.csv")
        with self._telemetry.trace(
            ModuleName.DATASET_MANAGEMENT, dataset_name=dataset_name, collection_name=collection_name,
            dataset_schema_version=schema.schema_id, verify_references=verify_references,
        ) as trace:
            try:
                with self._telemetry.span(SpanName.DATASET_VALIDATE) as span:
                    validated = validate_csv(source, schema, canonical_path=canonical_path, max_rows=self._max_rows)
                    source_documents = self._verify_references(validated) if verify_references else None
                    span.set_output(row_count=validated.row_count, checksum_sha256=validated.checksum_sha256)

                manifest = self._manifest(
                    schema, validated, module=module, dataset_name=dataset_name, purpose=purpose, label=label,
                    collection_name=collection_name, source_documents=source_documents, generator=generator,
                    source_filename=source_filename,
                )
                with self._telemetry.span(SpanName.DATASET_STORE, dataset_version=manifest["dataset_version"]) as span:
                    stored = self._store.put_version(
                        str(module), dataset_name, csv_path=validated.canonical_path, checksum_sha256=validated.checksum_sha256,
                        manifest=manifest,
                    )
                    promoted = False
                    if promote:
                        self._store.promote(str(module), dataset_name, stored.dataset_version)
                        promoted = True
                    span.set_output(created=stored.created, promoted=promoted)

                sync: Optional[SyncResult] = None
                if sync_langfuse:
                    sync = self._sync(module, dataset_name, manifest, validated.canonical_path, stored.keys.csv_key)
            finally:
                canonical_path.unlink(missing_ok=True)

            trace.set_output(
                dataset_version=stored.dataset_version, created=stored.created, row_count=validated.row_count, promoted=promoted,
                langfuse_sync_status=sync.status if sync else None,
            )
        self._log.info(
            "Dataset %s: %s/%s@%s rows=%d promoted=%s langfuse=%s",
            "created" if stored.created else "unchanged", module, dataset_name, stored.dataset_version, validated.row_count,
            promoted, sync.status if sync else "off",
        )
        keys = [stored.keys.csv_key, stored.keys.manifest_key]
        return DatasetPublication(
            module_name=str(module), dataset_name=dataset_name, dataset_version=stored.dataset_version, created=stored.created,
            row_count=validated.row_count, checksum_sha256=validated.checksum_sha256, s3_keys=keys, promoted=promoted,
            manifest=manifest, langfuse_sync=sync.to_dict() if sync else None,
        )

    def info(self, module: ModuleName, dataset_name: str) -> Dict[str, Any]:
        versions = self._store.list_versions(str(module), dataset_name)
        current = self._store.get_current(str(module), dataset_name)
        if not versions and current is None:
            raise DatasetNotFoundError(f"no dataset {module}/{dataset_name} in s3://{self._store.bucket}")
        return {"module_name": str(module), "dataset_name": dataset_name, "current": current, "versions": versions}

    def version(self, module: ModuleName, dataset_name: str, dataset_version: str) -> Dict[str, Any]:
        manifest = self._store.get_manifest(str(module), dataset_name, dataset_version)
        if manifest is None:
            raise DatasetNotFoundError(f"no complete version {module}/{dataset_name}@{dataset_version}")
        keys = self._store.version_keys(str(module), dataset_name, dataset_version)
        return {"manifest": manifest, "s3_keys": [keys.csv_key, keys.manifest_key]}

    def promote(self, module: ModuleName, dataset_name: str, dataset_version: str) -> Dict[str, Any]:
        with self._telemetry.trace(ModuleName.DATASET_MANAGEMENT, dataset_name=dataset_name, dataset_version=dataset_version) as trace:
            current = self._store.promote(str(module), dataset_name, dataset_version)
            trace.set_output(promoted=True, dataset_version=dataset_version)
        return current

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _verify_references(self, validated: ValidatedDataset) -> List[Dict[str, Any]]:
        """Every document published, every chunk id among its merged chunks; else DatasetValidationError."""
        if self._resolve_document is None:
            raise DatasetValidationError([ValidationIssue(0, None, "REFERENCE_CHECK_UNAVAILABLE")])
        if len(validated.document_ids) > MAX_REFERENCE_DOCUMENTS:
            raise DatasetValidationError([ValidationIssue(0, "document_id", "TOO_MANY_DOCUMENTS")])
        issues: List[ValidationIssue] = []
        sources: List[Dict[str, Any]] = []
        for document_id in validated.document_ids:
            refs = validated.chunk_refs.get(document_id, {})
            try:
                version, chunk_ids = self._resolve_document(document_id)
            except LookupError:
                issues.append(ValidationIssue(min(refs.values(), default=0), "document_id", "UNKNOWN_DOCUMENT"))
                continue
            sources.append({"document_id": document_id, "version": version})
            issues.extend(
                ValidationIssue(row, "answer_chunk_id", "UNKNOWN_CHUNK_ID") for chunk_id, row in refs.items() if chunk_id not in chunk_ids
            )
        if issues:
            raise DatasetValidationError(sorted(issues, key=lambda issue: issue.row)[:50], truncated=len(issues) > 50)
        return sources

    def _manifest(
        self, schema: DatasetSchema, validated: ValidatedDataset, *, module: ModuleName, dataset_name: str, purpose: Optional[str],
        label: Optional[str], collection_name: Optional[str], source_documents: Optional[List[Dict[str, Any]]],
        generator: Optional[Dict[str, Any]], source_filename: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "module_name": str(module),
            "dataset_name": dataset_name,
            "dataset_version": dataset_version_for(validated.checksum_sha256),
            "dataset_schema_version": schema.schema_id,
            "purpose": (purpose or schema.purpose)[:500],
            "label": (label or "")[:200] or None,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "row_count": validated.row_count,
            "checksum_sha256": validated.checksum_sha256,
            "source_sha256": validated.source_sha256,
            "source_filename_sha256": hashlib.sha256(source_filename.encode("utf-8")).hexdigest()[:16] if source_filename else None,
            "source_documents": source_documents if source_documents is not None else [{"document_id": d, "version": None} for d in validated.document_ids],
            "references_verified": source_documents is not None,
            "collection_name": collection_name,
            "columns": list(schema.names),
            "pipeline_version": self._pipeline_version,
            "service_version": self._service_version,
            "generator": generator or {"kind": _GENERATOR_UPLOAD},
        }

    def _sync(self, module: ModuleName, dataset_name: str, manifest: Dict[str, Any], canonical_path: Path, csv_key: str) -> SyncResult:
        client = self._telemetry.langfuse_client
        with self._telemetry.span(SpanName.LANGFUSE_DATASET_SYNC) as span:
            if client is None:
                result = SyncResult("skipped", reason="langfuse_disabled")
            else:
                result = sync_dataset_version(
                    client, module_name=str(module), dataset_name=dataset_name, dataset_version=manifest["dataset_version"],
                    checksum_sha256=manifest["checksum_sha256"], schema_id=manifest["dataset_schema_version"],
                    s3_bucket=self._store.bucket, csv_key=csv_key, row_count=manifest["row_count"],
                    rows=_rows(canonical_path), max_items=self._sync_max_items, logger=self._log,
                )
            span.set_output(langfuse_sync_status=result.status, item_count=result.item_count)
        try:
            self._store.put_sidecar(str(module), dataset_name, manifest["dataset_version"], _SIDECAR_SYNC, {**result.to_dict(), "recorded_at": _now()})
        except Exception as exc:  # bookkeeping only
            self._log.warning("Langfuse sync sidecar not written for %s/%s: %s", module, dataset_name, exc)
        return result


def _rows(path: Path) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
