"""
app.api.routes.datasets
=======================
Ground-truth datasets — versioned CSVs in S3 that an independent evaluation pipeline reads (docs/DATASETS.md).

    POST /v1/datasets                                              upload a CSV → immutable version (201; 200 when identical content exists)
    GET  /v1/datasets/{module_name}/{dataset_name}                 current pointer + every complete version
    GET  /v1/datasets/{module_name}/{dataset_name}/versions/{v}    one version's manifest
    POST /v1/datasets/{module_name}/{dataset_name}/current         promote a version

    stream the CSV to disk ─► DatasetService.publish on a worker thread (validate · verify references · store · [promote] · [Langfuse mirror])

`module_name` is a `ModuleName` (the canonical registry — anything else is a 422), `dataset_name` follows
`DatasetName`. The dataset bucket is `s3_bucket`, else `S3_DATASET_BUCKET`, else `S3_BUCKET`; references
(`document_id`, `answer_chunk_id`) are checked against the published documents of the artifact bucket
(`S3_BUCKET`, else the dataset bucket). These calls are light and do not take the ingestion gate.

Errors: 413 `DATASET_TOO_LARGE`; 422 `DATASET_INVALID` (with `issues: [{row, column, code}]` — never cell
values), `NO_DATASET_SCHEMA`; 409 `DATASET_VERSION_CONFLICT`; 404 `DATASET_NOT_FOUND`; 502
`DATASET_STORE_UNAVAILABLE` (retryable). Only validation, dispatch, response shaping and logging live here.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_logger, get_settings
from app.api.errors import http_error
from app.api.schemas import (
    CollectionName,
    DatasetInfoResponse,
    DatasetName,
    DatasetPromoteRequest,
    DatasetPromoteResponse,
    DatasetUploadResponse,
    DatasetVersion,
    DatasetVersionResponse,
    DatasetVersionSummary,
    ErrorResponse,
)
from app.api.uploads import StagedUpload, stage_upload
from app.config.settings import ServerSettings
from app.datasets.schema import NoDatasetSchemaError
from app.datasets.service import DatasetPublication, DatasetService
from app.datasets.validation import DatasetValidationError
from app.observability.modules import ModuleName
from app.orchestration.errors import ChannelError
from app.orchestration.s3_import import published_chunk_ids
from app.storage.s3_artifact_store import S3ArtifactStore
from app.storage.s3_dataset_store import DatasetNotFoundError, DatasetStoreError, DatasetVersionConflictError, S3DatasetStore

router = APIRouter(tags=["datasets"])

_ERROR_RESPONSES: dict = {
    400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
    422: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
}


@router.post("/v1/datasets", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED, responses=_ERROR_RESPONSES)
async def upload_dataset(
    response: Response,
    file: UploadFile,
    module_name: Annotated[ModuleName, Form()],
    dataset_name: Annotated[DatasetName, Form()],
    purpose: Optional[str] = Form(default=None, max_length=500),
    label: Optional[str] = Form(default=None, max_length=200),
    collection_name: Annotated[Optional[CollectionName], Form()] = None,
    promote: bool = Form(default=False),
    verify_references: bool = Form(default=True),
    sync_langfuse: bool = Form(default=False),
    s3_bucket: Optional[str] = Form(default=None),
    AWS_DEFAULT_REGION: Optional[str] = Form(default=None),
    settings: ServerSettings = Depends(get_settings),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> DatasetUploadResponse:
    service = dataset_service(settings, logger, s3_bucket=s3_bucket, region=AWS_DEFAULT_REGION)
    staged = await _receive_csv(file, settings)
    try:
        publication: DatasetPublication = await run_in_threadpool(
            _publish_or_raise, service, module=module_name, dataset_name=dataset_name, source=staged.path, purpose=purpose, label=label,
            collection_name=collection_name, promote=promote, verify_references=verify_references, sync_langfuse=sync_langfuse,
            source_filename=file.filename or None,
        )
    finally:
        staged.discard()
    logger.info(
        "dataset %s", "created" if publication.created else "unchanged",
        extra={"channel": "datasets", "stage": "upload", "status": "completed", "version_id": publication.dataset_version},
    )
    if not publication.created:
        response.status_code = status.HTTP_200_OK
    return DatasetUploadResponse(**dataset_summary(publication).model_dump(), manifest=publication.manifest)


@router.get("/v1/datasets/{module_name}/{dataset_name}", response_model=DatasetInfoResponse, responses=_ERROR_RESPONSES)
async def dataset_info(
    module_name: ModuleName,
    dataset_name: DatasetName,
    s3_bucket: Optional[str] = None,
    AWS_DEFAULT_REGION: Optional[str] = None,
    settings: ServerSettings = Depends(get_settings),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> DatasetInfoResponse:
    service = dataset_service(settings, logger, s3_bucket=s3_bucket, region=AWS_DEFAULT_REGION, with_references=False)
    info = await run_in_threadpool(_store_call, lambda: service.info(module_name, dataset_name))
    return DatasetInfoResponse(**info)


@router.get(
    "/v1/datasets/{module_name}/{dataset_name}/versions/{dataset_version}", response_model=DatasetVersionResponse, responses=_ERROR_RESPONSES,
)
async def dataset_version(
    module_name: ModuleName,
    dataset_name: DatasetName,
    dataset_version: DatasetVersion,
    s3_bucket: Optional[str] = None,
    AWS_DEFAULT_REGION: Optional[str] = None,
    settings: ServerSettings = Depends(get_settings),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> DatasetVersionResponse:
    service = dataset_service(settings, logger, s3_bucket=s3_bucket, region=AWS_DEFAULT_REGION, with_references=False)
    found = await run_in_threadpool(_store_call, lambda: service.version(module_name, dataset_name, dataset_version))
    return DatasetVersionResponse(module_name=str(module_name), dataset_name=dataset_name, dataset_version=dataset_version, **found)


@router.post("/v1/datasets/{module_name}/{dataset_name}/current", response_model=DatasetPromoteResponse, responses=_ERROR_RESPONSES)
async def promote_dataset_version(
    module_name: ModuleName,
    dataset_name: DatasetName,
    payload: DatasetPromoteRequest,
    settings: ServerSettings = Depends(get_settings),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> DatasetPromoteResponse:
    service = dataset_service(settings, logger, s3_bucket=payload.s3_bucket, region=payload.AWS_DEFAULT_REGION, with_references=False)
    current = await run_in_threadpool(_store_call, lambda: service.promote(module_name, dataset_name, payload.dataset_version))
    logger.info("dataset promoted", extra={"channel": "datasets", "stage": "promote", "status": "completed", "version_id": payload.dataset_version})
    return DatasetPromoteResponse(**current)


# --------------------------------------------------------------------------- #
# Helpers (also used by the S3 auto-grounding route)
# --------------------------------------------------------------------------- #


def dataset_service(
    settings: ServerSettings, logger: logging.LoggerAdapter, *, s3_bucket: Optional[str], region: Optional[str], with_references: bool = True,
) -> DatasetService:
    """A DatasetService on the dataset bucket; references resolve against the artifact bucket's published documents."""
    dataset_bucket = s3_bucket or settings.dataset_bucket
    if not dataset_bucket:
        raise http_error(400, "S3_BUCKET_REQUIRED", "s3_bucket is required: no S3_DATASET_BUCKET / S3_BUCKET default is configured")
    region = region or settings.AWS_DEFAULT_REGION
    resolve = None
    if with_references:
        artifacts = S3ArtifactStore(bucket=settings.s3_bucket or dataset_bucket, region=region, service_version=settings.service_version)
        resolve = published_chunk_ids(artifacts, logger)
    return DatasetService(
        S3DatasetStore(bucket=dataset_bucket, region=region),
        max_rows=settings.max_dataset_rows,
        pipeline_version=settings.effective_pipeline_version,
        service_version=settings.service_version,
        resolve_document=resolve,
        langfuse_sync_max_items=settings.langfuse_dataset_sync_max_items,
        logger=logger,
    )


def dataset_summary(publication: DatasetPublication) -> DatasetVersionSummary:
    return DatasetVersionSummary(
        module_name=publication.module_name, dataset_name=publication.dataset_name, dataset_version=publication.dataset_version,
        created=publication.created, row_count=publication.row_count, checksum_sha256=publication.checksum_sha256,
        s3_keys=publication.s3_keys, promoted=publication.promoted, langfuse_sync=publication.langfuse_sync,
    )


def dataset_http_error(exc: Exception) -> Optional[HTTPException]:
    """The HTTP error for a dataset failure, or None when `exc` is not one."""
    if isinstance(exc, DatasetValidationError):
        detail: Dict[str, Any] = {
            "error": {
                "code": exc.code, "message": str(exc), "retryable": False,
                "issues": [issue.to_dict() for issue in exc.issues], "truncated": exc.truncated,
            },
        }
        return HTTPException(status_code=exc.http_status, detail=detail)
    if isinstance(exc, NoDatasetSchemaError):
        return http_error(exc.http_status, exc.code, str(exc))
    if isinstance(exc, (DatasetVersionConflictError, DatasetNotFoundError)):
        return http_error(exc.http_status, exc.code, str(exc))
    if isinstance(exc, DatasetStoreError):
        return http_error(exc.http_status, exc.code, str(exc), retryable=True)
    return None


def _publish_or_raise(service: DatasetService, **kwargs: Any) -> DatasetPublication:
    try:
        return service.publish(**kwargs)
    except Exception as exc:
        mapped = dataset_http_error(exc)
        if mapped is None:
            raise
        raise mapped from exc


def _store_call(work: Any) -> Any:
    try:
        return work()
    except Exception as exc:
        mapped = dataset_http_error(exc)
        if mapped is None:
            raise
        raise mapped from exc


async def _receive_csv(file: UploadFile, settings: ServerSettings) -> StagedUpload:
    try:
        return await run_in_threadpool(
            stage_upload, file.file, file.filename or "", settings.uploads_dir, settings.max_dataset_upload_bytes, suffix=".csv",
        )
    except ChannelError as exc:
        raise http_error(413, "DATASET_TOO_LARGE", f"CSV exceeds the {settings.max_dataset_upload_bytes}-byte limit (MAX_DATASET_UPLOAD_BYTES)") from exc
