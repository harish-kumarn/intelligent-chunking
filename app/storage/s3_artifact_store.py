"""
app.storage.s3_artifact_store
=============================
S3ArtifactStore — content-hash-versioned publication of a document's artifact directory.

    version id : SHA-256 of the source PDF bytes (never the filename)
    layout     : {document_id}/versions/{hash}/<artifacts…>
                 {document_id}/versions/{hash}/manifest.json    ← written LAST; proves a complete upload
                 {document_id}/versions/{hash}/published.json   ← retention bookkeeping
                 {document_id}/current.json                     ← pointer to the live version
                 _collections/{collection_name}.json            ← documents indexed in that Milvus collection

Main path
    calculate_content_hash ─► upload_artifacts ─► publish_version ─► cleanup_old_versions
    add_version_artifacts : later outputs (e.g. groundtruth) added to a complete version and its manifest
    register_collection_document / collection_documents : the collection registry query reformulation reads

Failure policy: `upload_artifacts` / `publish_version` / `add_version_artifacts` raise `S3UploadError` /
`S3PublicationError` and never leave a half-published pointer — a retry either
re-uploads (no manifest) or refreshes only the files whose sha256 differs (manifest present).
`cleanup_old_versions` never raises. The current version is never deleted.
Concurrency: uploads fan out over a bounded thread pool; the boto3 client is
created lazily and is thread-safe.
Credentials: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` from the environment or
`.env` when set (see `app.config.aws`), otherwise boto3's standard chain.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config.aws import AwsCredentials
from app.logging.setup import get_logger

__all__ = [
    "PublishResult", "S3ArtifactError", "S3ArtifactStore", "S3PublicationError", "S3UploadError", "register_in_collection",
]

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB reads keep hashing memory flat for large PDFs
_DEFAULT_RETENTION_COUNT = 2    # current + one previous version
_DEFAULT_MAX_CONCURRENCY = 8
_DELETE_BATCH = 1000            # S3 DeleteObjects hard limit
_MANIFEST_NAME = "manifest.json"
_PUBLISHED_MARKER_NAME = "published.json"
_CURRENT_NAME = "current.json"
_COLLECTIONS_PREFIX = "_collections/"  # document ids never start with "_", so this never clashes with one
_MISSING_CODES = frozenset({"NoSuchKey", "404"})


class S3ArtifactError(Exception):
    """Base class for artifact storage failures."""


class S3UploadError(S3ArtifactError):
    """A version's artifact set did not upload completely."""


class S3PublicationError(S3ArtifactError):
    """A version could not be made current."""


@dataclass(frozen=True)
class PublishResult:
    document_id: str
    version: str
    already_existed: bool
    uploaded_file_count: int
    total_bytes: int


def _utcnow_iso() -> str:
    # Microseconds keep publishes within the same second orderable as plain strings (retention sorts on this).
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class S3ArtifactStore:
    """Credentials come from `AwsCredentials` (env / `.env`) or, when blank, boto3's standard chain."""

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        *,
        retention_count: int = _DEFAULT_RETENTION_COUNT,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        service_version: str = "unknown",
        credentials: Optional[AwsCredentials] = None,
    ) -> None:
        self._bucket = bucket
        self._region = region
        self._credentials = credentials
        self._retention_count = retention_count
        self._max_concurrency = max_concurrency
        self._service_version = service_version
        self._client: Any = None  # created on first use so constructing a store never touches the network
        self._log = get_logger(module="S3_ARTIFACTS")

    @property
    def client(self) -> Any:
        if self._client is None:
            credentials = self._credentials or AwsCredentials()
            region = self._region or credentials.default_region
            self._client = boto3.client("s3", region_name=region, **credentials.boto_kwargs())
            self._log.debug(
                "S3 client ready: bucket=%s region=%s credentials=%s",
                self._bucket, region, "static" if credentials.is_static else "boto3-chain",
            )
        return self._client

    # ------------------------------------------------------------------ #
    # Hashing & keys
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_content_hash(*, content: Optional[bytes] = None, file_path: Optional[Path] = None) -> str:
        """SHA-256 of the PDF, from in-memory bytes or a streamed file (exactly one must be given)."""
        if (content is None) == (file_path is None):
            raise ValueError("calculate_content_hash requires exactly one of content or file_path")
        digest = hashlib.sha256()
        if content is not None:
            digest.update(content)
        else:
            with open(file_path, "rb") as fh:  # type: ignore[arg-type]
                for block in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(block)
        return digest.hexdigest()

    def _version_prefix(self, document_id: str, content_hash: str) -> str:
        return f"{document_id}/versions/{content_hash}/"

    def _manifest_key(self, document_id: str, content_hash: str) -> str:
        return f"{self._version_prefix(document_id, content_hash)}{_MANIFEST_NAME}"

    def _published_marker_key(self, document_id: str, content_hash: str) -> str:
        return f"{self._version_prefix(document_id, content_hash)}{_PUBLISHED_MARKER_NAME}"

    def _current_key(self, document_id: str) -> str:
        return f"{document_id}/{_CURRENT_NAME}"

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def version_exists(self, document_id: str, content_hash: str) -> bool:
        """True only when a valid manifest exists — the sole proof of a complete upload."""
        return self._valid_manifest(document_id, content_hash) is not None

    def _valid_manifest(self, document_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
        manifest = self._get_json(self._manifest_key(document_id, content_hash))
        if isinstance(manifest, dict) and manifest.get("document_id") == document_id and manifest.get("version") == content_hash:
            return manifest
        return None

    def get_current_version(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self._get_json(self._current_key(document_id))

    def get_version_resource_json(self, document_id: str, content_hash: str, resource_name: str) -> Optional[Any]:
        """One JSON artifact from a version; None when missing or malformed."""
        return self._get_json(f"{self._version_prefix(document_id, content_hash)}{resource_name}")

    def _get_json(self, key: str) -> Optional[Any]:
        try:
            body = self.client.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in _MISSING_CODES:
                return None
            raise
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None  # a malformed artifact reads as absent; callers treat both the same way

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def upload_artifacts(self, local_dir: Path, document_id: str, content_hash: str) -> PublishResult:
        """
        Upload every (non-hidden) file under `local_dir`, then the manifest (each file listed with its sha256).

        A version that already exists is refreshed, not skipped: the same PDF ingested again with other
        options (classification, auto-grounding, …) produces other artifacts, and S3 must match what was
        indexed. Only files whose sha256 differs from the manifest (or that it lacks) are uploaded; files the
        manifest lists but `local_dir` lacks (e.g. groundtruth added later) are kept. Nothing changed → no write.
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise S3UploadError(f"artifact directory does not exist: {local_dir}")
        # Hidden files are temp files of atomic writes and workspace bookkeeping, never artifacts.
        files = sorted(
            path for path in local_dir.rglob("*")
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(local_dir).parts)
        )
        digests = {path: _file_sha256(path) for path in files}
        manifest = self._valid_manifest(document_id, content_hash)
        listed: Dict[str, Dict[str, Any]] = {
            item["path"]: item for item in (manifest or {}).get("files") or [] if isinstance(item, dict) and "path" in item
        }
        to_upload = [path for path in files if listed.get(path.relative_to(local_dir).as_posix(), {}).get("sha256") != digests[path]]
        if manifest is not None and not to_upload:
            self._log.info("Version already uploaded and unchanged: document=%s version=%s", document_id, content_hash)
            return PublishResult(document_id, content_hash, True, 0, 0)

        prefix = self._version_prefix(document_id, content_hash)
        started = time.monotonic()
        uploaded: List[Dict[str, Any]] = []
        failures: List[BaseException] = []
        with ThreadPoolExecutor(max_workers=self._max_concurrency, thread_name_prefix="s3-upload") as pool:
            futures = {pool.submit(self._upload_one, path, local_dir, prefix): path for path in to_upload}
            for future in as_completed(futures):
                try:
                    uploaded.append({**future.result(), "sha256": digests[futures[future]]})
                except (BotoCoreError, ClientError) as exc:
                    failures.append(exc)
        if failures:
            raise S3UploadError(
                f"{len(failures)} of {len(to_upload)} artifact file(s) failed to upload to s3://{self._bucket}/{prefix}"
            ) from failures[0]

        listed.update((item["path"], item) for item in uploaded)
        entries = sorted(listed.values(), key=lambda item: item["path"])
        total_bytes = sum(item.get("size_bytes", 0) for item in entries)
        now = _utcnow_iso()
        self._put_json(
            self._manifest_key(document_id, content_hash),
            {
                **(manifest or {}),
                "document_id": document_id,
                "source_pdf_sha256": content_hash,
                "version": content_hash,
                "service_version": self._service_version,
                "uploaded_at": (manifest or {}).get("uploaded_at", now),
                **({"updated_at": now} if manifest is not None else {}),
                "file_count": len(entries),
                "total_bytes": total_bytes,
                "files": entries,
            },
        )
        self._log.info(
            "Artifacts %s: document=%s files=%d/%d bytes=%d in %dms",
            "refreshed" if manifest is not None else "uploaded", document_id, len(uploaded), len(files),
            sum(item["size_bytes"] for item in uploaded), int((time.monotonic() - started) * 1000),
            extra={"bucket": self._bucket, "s3_prefix": prefix, "uploaded_file_count": len(uploaded)},
        )
        return PublishResult(document_id, content_hash, manifest is not None, len(uploaded), total_bytes)

    def publish_version(self, document_id: str, content_hash: str) -> None:
        """Point `current.json` at a completely uploaded version; then best-effort publish marker."""
        if not self.version_exists(document_id, content_hash):
            raise S3PublicationError(f"cannot publish {document_id}/{content_hash}: version is not completely uploaded")
        published_at = _utcnow_iso()
        try:
            self._put_json(
                self._current_key(document_id),
                {
                    "document_id": document_id,
                    "version": content_hash,
                    "published_at": published_at,
                    "artifact_path": self._version_prefix(document_id, content_hash),
                },
            )
        except (BotoCoreError, ClientError) as exc:
            raise S3PublicationError(f"failed to publish current version for {document_id}") from exc
        self._log.info("Version published: document=%s version=%s", document_id, content_hash, extra={"bucket": self._bucket})

        try:
            self._put_json(self._published_marker_key(document_id, content_hash), {"published_at": published_at})
        except (BotoCoreError, ClientError) as exc:
            self._log.warning("Publish marker write failed (retention bookkeeping only): document=%s %s", document_id, exc)

    def add_version_artifacts(self, document_id: str, content_hash: str, files: Sequence[Path]) -> List[str]:
        """
        Upload `files` to the top level of a complete version (replacing same-named ones), then rewrite its
        manifest to list them. Returns the uploaded keys. Raises S3UploadError; the manifest is untouched on failure.
        """
        manifest_key = self._manifest_key(document_id, content_hash)
        manifest = self._valid_manifest(document_id, content_hash)
        if manifest is None:
            raise S3UploadError(f"cannot add artifacts to {document_id}/{content_hash}: version is not completely uploaded")
        prefix = self._version_prefix(document_id, content_hash)
        entries = {item["path"]: item for item in manifest.get("files") or [] if isinstance(item, dict) and "path" in item}
        try:
            for path in files:
                entries[Path(path).name] = {**self._upload_one(Path(path), Path(path).parent, prefix), "sha256": _file_sha256(Path(path))}
            listed = sorted(entries.values(), key=lambda item: item["path"])
            manifest.update(
                files=listed, file_count=len(listed), total_bytes=sum(item.get("size_bytes", 0) for item in listed),
                updated_at=_utcnow_iso(),
            )
            self._put_json(manifest_key, manifest)
        except (BotoCoreError, ClientError) as exc:
            raise S3UploadError(f"failed to add artifacts to s3://{self._bucket}/{prefix}") from exc
        keys = [f"{prefix}{Path(path).name}" for path in files]
        self._log.info("Artifacts added: document=%s version=%s files=%s", document_id, content_hash, keys, extra={"bucket": self._bucket})
        return keys

    def _upload_one(self, path: Path, root: Path, prefix: str) -> Dict[str, Any]:
        relative = path.relative_to(root).as_posix()
        content_type, _ = mimetypes.guess_type(path.name)
        self.client.upload_file(
            str(path), self._bucket, f"{prefix}{relative}", ExtraArgs={"ContentType": content_type} if content_type else {},
        )
        return {"path": relative, "size_bytes": path.stat().st_size}

    def _put_json(self, key: str, payload: Dict[str, Any]) -> None:
        self.client.put_object(Bucket=self._bucket, Key=key, Body=json.dumps(payload).encode("utf-8"), ContentType="application/json")

    # ------------------------------------------------------------------ #
    # Collection registry — which documents a Milvus collection holds
    # ------------------------------------------------------------------ #

    def collection_documents(self, collection_name: str) -> List[str]:
        """Document ids registered for the collection, in registration order; [] when none."""
        registry = self._get_json(self._collection_key(collection_name))
        documents = registry.get("documents") if isinstance(registry, dict) else None
        return list(documents) if isinstance(documents, dict) else []

    def register_collection_document(self, collection_name: str, document_id: str, *, language: str) -> None:
        """Record that `document_id` is indexed in the collection (read-modify-write; ingestion is single-flight)."""
        key = self._collection_key(collection_name)
        registry = self._get_json(key)
        if not isinstance(registry, dict) or not isinstance(registry.get("documents"), dict):
            registry = {"collection_name": collection_name, "language": language, "documents": {}}
        now = _utcnow_iso()
        registry["documents"][document_id] = {"registered_at": now}
        registry["updated_at"] = now
        self._put_json(key, registry)

    def delete_collection_registry(self, collection_name: str) -> None:
        self.client.delete_object(Bucket=self._bucket, Key=self._collection_key(collection_name))

    @staticmethod
    def _collection_key(collection_name: str) -> str:
        return f"{_COLLECTIONS_PREFIX}{collection_name}.json"

    # ------------------------------------------------------------------ #
    # Retention
    # ------------------------------------------------------------------ #

    def cleanup_old_versions(self, document_id: str, current_hash: str) -> None:
        """Delete published versions beyond `retention_count`, newest kept. Never raises."""
        try:
            published = self._list_published_versions(document_id)
        except (BotoCoreError, ClientError) as exc:
            self._log.warning("Retention: failed to list versions for document=%s: %s", document_id, exc)
            return
        others = sorted((v for v in published if v[0] != current_hash), key=lambda item: item[1], reverse=True)
        for version, _ in others[max(self._retention_count - 1, 0):]:
            try:
                self.delete_version(document_id, version)
                self._log.info("Retention: deleted document=%s version=%s", document_id, version)
            except (BotoCoreError, ClientError) as exc:
                self._log.warning("Retention: failed to delete document=%s version=%s: %s", document_id, version, exc)

    def delete_version(self, document_id: str, content_hash: str) -> None:
        """Permanently delete one version's objects. Callers must ensure it is not current."""
        keys = self._list_keys(self._version_prefix(document_id, content_hash))
        for start in range(0, len(keys), _DELETE_BATCH):
            batch = keys[start:start + _DELETE_BATCH]
            self.client.delete_objects(Bucket=self._bucket, Delete={"Objects": [{"Key": key} for key in batch]})

    def _list_published_versions(self, document_id: str) -> List[Tuple[str, str]]:
        prefix = f"{document_id}/versions/"
        suffix = f"/{_PUBLISHED_MARKER_NAME}"
        versions = []
        for key in self._list_keys(prefix):
            if key.endswith(suffix):
                marker = self._get_json(key) or {}
                versions.append((key[len(prefix):-len(suffix)], marker.get("published_at", "")))
        return versions

    def _list_keys(self, prefix: str) -> List[str]:
        keys: List[str] = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self._bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys


def _file_sha256(path: Path) -> str:
    return S3ArtifactStore.calculate_content_hash(file_path=path)


def register_in_collection(
    store: S3ArtifactStore, collection_name: str, document_id: str, *, language: str, logger: logging.LoggerAdapter,
) -> None:
    """Best-effort `register_collection_document`: the registry only feeds query reformulation, so a failure is logged."""
    try:
        store.register_collection_document(collection_name, document_id, language=language)
    except (BotoCoreError, ClientError) as exc:
        logger.warning("Collection registry not updated: collection=%s document=%s: %s", collection_name, document_id, exc)
