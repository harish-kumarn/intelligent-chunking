"""
app.storage.s3_dataset_store
============================
S3DatasetStore — immutable, content-addressed ground-truth dataset versions.

    layout  : _datasets/{module_name}/{dataset_name}/
              ├── current.json                       ← the only mutable object: which version is "current"
              └── versions/{dataset_version}/
                  ├── dataset.csv                    canonical bytes (what the checksum covers)
                  ├── manifest.json                  written LAST; its presence proves a complete version
                  └── <sidecars>.json                e.g. langfuse_sync.json — bookkeeping, not dataset content
    version : `v-` + sha256(canonical csv)[:16] — the same content is always the same version, so a
              repeated upload is a no-op (`created=False`) and two different contents never share a version
    writes  : `dataset.csv` and `manifest.json` are PUT with `IfNoneMatch="*"` (S3 conditional write): an
              object that already exists is never overwritten. A 412 on an identical checksum is "already
              there"; on a different checksum it is `DatasetVersionConflictError`
    pointer : `promote` rewrites `current.json`; nothing moves it implicitly

The `_datasets/` prefix follows `_collections/`: document ids never start with `_`, so it can never clash
with a document folder (`{document_id}/…`).

Failure policy: a conflict raises `DatasetVersionConflictError` (409); any other S3 failure raises
`DatasetStoreError` (502, retryable) — a version is either complete (manifest present) or absent, never
half-written and pointed at. Reads of missing objects return None.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from botocore.exceptions import BotoCoreError, ClientError

from app.config.aws import AwsCredentials
from app.app_logging.setup import get_logger
from app.storage.s3_client import build_s3_client

__all__ = [
    "DATASET_NAME_PATTERN", "DATASET_VERSION_PATTERN", "DatasetNotFoundError", "DatasetStoreError", "DatasetVersionConflictError",
    "PutVersionResult", "S3DatasetStore", "VersionKeys", "dataset_version_for",
]

DATASETS_PREFIX = "_datasets/"
CSV_NAME = "dataset.csv"
MANIFEST_NAME = "manifest.json"
CURRENT_NAME = "current.json"
DATASET_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
DATASET_NAME_RULE = "3-64 characters: lowercase letters, digits, '_' and '-', starting with a letter"
DATASET_VERSION_PATTERN = re.compile(r"^v-[0-9a-f]{16}$")
_MISSING_CODES = frozenset({"NoSuchKey", "404", "NotFound"})
_PRECONDITION_CODES = frozenset({"PreconditionFailed", "412", "ConditionalRequestConflict"})
_READ_CHUNK = 1024 * 1024


class DatasetStoreError(Exception):
    """S3 did not complete a dataset read or write (retryable)."""

    code = "DATASET_STORE_UNAVAILABLE"
    http_status = 502


class DatasetVersionConflictError(DatasetStoreError):
    """A version with this id already holds different content; historical versions are never replaced."""

    code = "DATASET_VERSION_CONFLICT"
    http_status = 409


class DatasetNotFoundError(DatasetStoreError):
    code = "DATASET_NOT_FOUND"
    http_status = 404


@dataclass(frozen=True)
class VersionKeys:
    prefix: str
    csv_key: str
    manifest_key: str

    def sidecar(self, name: str) -> str:
        return f"{self.prefix}{name}"


@dataclass(frozen=True)
class PutVersionResult:
    dataset_version: str
    created: bool          # False: this exact content was already stored
    keys: VersionKeys


def dataset_version_for(checksum_sha256: str) -> str:
    return f"v-{checksum_sha256[:16]}"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class S3DatasetStore:
    def __init__(self, bucket: str, region: Optional[str] = None, *, credentials: Optional[AwsCredentials] = None, client: Any = None) -> None:
        self._bucket = bucket
        self._region = region
        self._credentials = credentials
        self._client: Any = client
        self._log = get_logger(module="S3_DATASETS")

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = build_s3_client(self._region, self._credentials or AwsCredentials())
        return self._client

    # ------------------------------------------------------------------ #
    # Keys
    # ------------------------------------------------------------------ #

    @staticmethod
    def dataset_prefix(module_name: str, dataset_name: str) -> str:
        return f"{DATASETS_PREFIX}{module_name}/{dataset_name}/"

    def version_keys(self, module_name: str, dataset_name: str, dataset_version: str) -> VersionKeys:
        prefix = f"{self.dataset_prefix(module_name, dataset_name)}versions/{dataset_version}/"
        return VersionKeys(prefix=prefix, csv_key=f"{prefix}{CSV_NAME}", manifest_key=f"{prefix}{MANIFEST_NAME}")

    def current_key(self, module_name: str, dataset_name: str) -> str:
        return f"{self.dataset_prefix(module_name, dataset_name)}{CURRENT_NAME}"

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def get_manifest(self, module_name: str, dataset_name: str, dataset_version: str) -> Optional[Dict[str, Any]]:
        manifest = self._get_json(self.version_keys(module_name, dataset_name, dataset_version).manifest_key)
        if isinstance(manifest, dict) and manifest.get("dataset_version") == dataset_version:
            return manifest
        return None

    def get_current(self, module_name: str, dataset_name: str) -> Optional[Dict[str, Any]]:
        current = self._get_json(self.current_key(module_name, dataset_name))
        return current if isinstance(current, dict) and current.get("dataset_version") else None

    def list_versions(self, module_name: str, dataset_name: str) -> List[Dict[str, Any]]:
        """Every complete version's manifest, newest first."""
        prefix = f"{self.dataset_prefix(module_name, dataset_name)}versions/"
        suffix = f"/{MANIFEST_NAME}"
        manifests: List[Dict[str, Any]] = []
        try:
            for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith(suffix):
                        manifest = self._get_json(key)
                        if isinstance(manifest, dict) and manifest.get("dataset_version"):
                            manifests.append(manifest)
        except (BotoCoreError, ClientError) as exc:
            raise DatasetStoreError(f"could not list dataset versions under s3://{self._bucket}/{prefix}") from exc
        return sorted(manifests, key=lambda item: str(item.get("created_at", "")), reverse=True)

    def object_sha256(self, key: str) -> Optional[str]:
        """Streamed sha256 of one object; None when it does not exist."""
        try:
            body = self.client.get_object(Bucket=self._bucket, Key=key)["Body"]
        except ClientError as exc:
            if _error_code(exc) in _MISSING_CODES:
                return None
            raise DatasetStoreError(f"could not read s3://{self._bucket}/{key}") from exc
        except BotoCoreError as exc:
            raise DatasetStoreError(f"could not read s3://{self._bucket}/{key}") from exc
        digest = hashlib.sha256()
        for block in iter(lambda: body.read(_READ_CHUNK), b""):
            digest.update(block)
        return digest.hexdigest()

    def download_csv(self, module_name: str, dataset_name: str, dataset_version: str, target: Path) -> Path:
        """Stream `dataset.csv` of a version to `target`; DatasetNotFoundError when absent."""
        key = self.version_keys(module_name, dataset_name, dataset_version).csv_key
        try:
            body = self.client.get_object(Bucket=self._bucket, Key=key)["Body"]
        except ClientError as exc:
            if _error_code(exc) in _MISSING_CODES:
                raise DatasetNotFoundError(f"no dataset.csv at s3://{self._bucket}/{key}") from exc
            raise DatasetStoreError(f"could not read s3://{self._bucket}/{key}") from exc
        except BotoCoreError as exc:
            raise DatasetStoreError(f"could not read s3://{self._bucket}/{key}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as out:
            for block in iter(lambda: body.read(_READ_CHUNK), b""):
                out.write(block)
        return target

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def put_version(
        self, module_name: str, dataset_name: str, *, csv_path: Path, checksum_sha256: str, manifest: Dict[str, Any],
    ) -> PutVersionResult:
        """
        Store one immutable version: `dataset.csv` then `manifest.json`, both conditional on absence.
        Identical content already stored → `created=False`; different content under the same version →
        DatasetVersionConflictError. The manifest must carry `dataset_version` and `checksum_sha256`.
        """
        version = str(manifest["dataset_version"])
        if manifest.get("checksum_sha256") != checksum_sha256:
            raise ValueError("manifest checksum does not match the dataset checksum")
        keys = self.version_keys(module_name, dataset_name, version)

        existing = self.get_manifest(module_name, dataset_name, version)
        if existing is not None:
            if existing.get("checksum_sha256") == checksum_sha256:
                self._log.info("Dataset version already stored: %s/%s@%s", module_name, dataset_name, version)
                return PutVersionResult(version, False, keys)
            raise DatasetVersionConflictError(f"{module_name}/{dataset_name}@{version} already holds different content")

        try:
            with open(csv_path, "rb") as body:
                self.client.put_object(
                    Bucket=self._bucket, Key=keys.csv_key, Body=body, ContentType="text/csv; charset=utf-8", IfNoneMatch="*",
                )
        except ClientError as exc:
            if _error_code(exc) not in _PRECONDITION_CODES:
                raise DatasetStoreError(f"could not write s3://{self._bucket}/{keys.csv_key}") from exc
            # A crashed earlier attempt left the CSV without a manifest: keep it only if it is this content.
            if self.object_sha256(keys.csv_key) != checksum_sha256:
                raise DatasetVersionConflictError(f"{keys.csv_key} exists with different content") from exc
        except BotoCoreError as exc:
            raise DatasetStoreError(f"could not write s3://{self._bucket}/{keys.csv_key}") from exc

        try:
            self._put_json(keys.manifest_key, manifest, conditional=True)
        except ClientError as exc:
            if _error_code(exc) not in _PRECONDITION_CODES:
                raise DatasetStoreError(f"could not write s3://{self._bucket}/{keys.manifest_key}") from exc
            stored = self.get_manifest(module_name, dataset_name, version)  # a concurrent writer finished first
            if stored is not None and stored.get("checksum_sha256") == checksum_sha256:
                return PutVersionResult(version, False, keys)
            raise DatasetVersionConflictError(f"{module_name}/{dataset_name}@{version} already holds different content") from exc
        except BotoCoreError as exc:
            raise DatasetStoreError(f"could not write s3://{self._bucket}/{keys.manifest_key}") from exc

        self._log.info(
            "Dataset version stored: %s/%s@%s rows=%s", module_name, dataset_name, version, manifest.get("row_count"),
            extra={"bucket": self._bucket, "s3_prefix": keys.prefix, "version_id": version},
        )
        return PutVersionResult(version, True, keys)

    def promote(self, module_name: str, dataset_name: str, dataset_version: str) -> Dict[str, Any]:
        """Point `current.json` at a complete version. DatasetNotFoundError when the version has no manifest."""
        manifest = self.get_manifest(module_name, dataset_name, dataset_version)
        if manifest is None:
            raise DatasetNotFoundError(f"{module_name}/{dataset_name}@{dataset_version} is not a complete dataset version")
        current = {
            "module_name": module_name, "dataset_name": dataset_name, "dataset_version": dataset_version,
            "checksum_sha256": manifest.get("checksum_sha256"), "promoted_at": _utcnow_iso(),
        }
        try:
            self._put_json(self.current_key(module_name, dataset_name), current)
        except (BotoCoreError, ClientError) as exc:
            raise DatasetStoreError(f"could not promote {module_name}/{dataset_name}@{dataset_version}") from exc
        self._log.info("Dataset promoted: %s/%s@%s", module_name, dataset_name, dataset_version, extra={"bucket": self._bucket})
        return current

    def put_sidecar(self, module_name: str, dataset_name: str, dataset_version: str, name: str, payload: Dict[str, Any]) -> str:
        """Bookkeeping next to a version (overwritten freely; never part of the dataset)."""
        key = self.version_keys(module_name, dataset_name, dataset_version).sidecar(name)
        try:
            self._put_json(key, payload)
        except (BotoCoreError, ClientError) as exc:
            raise DatasetStoreError(f"could not write s3://{self._bucket}/{key}") from exc
        return key

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _get_json(self, key: str) -> Optional[Any]:
        try:
            body = self.client.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if _error_code(exc) in _MISSING_CODES:
                return None
            raise DatasetStoreError(f"could not read s3://{self._bucket}/{key}") from exc
        except BotoCoreError as exc:
            raise DatasetStoreError(f"could not read s3://{self._bucket}/{key}") from exc
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None

    def _put_json(self, key: str, payload: Dict[str, Any], *, conditional: bool = False) -> None:
        kwargs: Dict[str, Any] = dict(
            Bucket=self._bucket, Key=key, Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        if conditional:
            kwargs["IfNoneMatch"] = "*"
        self.client.put_object(**kwargs)


def _error_code(exc: ClientError) -> str:
    error = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
    return str(error.get("Code", ""))
