"""
app.api.routes.s3_documents
===========================
Work on a document whose artifacts are already published to S3, skipping OCR and Gemini chunking.

    POST /v1/documents/from-s3               : index the current version's merged chunks into Milvus
    POST /v1/documents/from-s3/autogrounding : auto-grounding alone; groundtruth_dataset.{json,csv}
                                               saved into the same S3 version (and its manifest)

    validate ─► acquire gate (429 if busy) ─► S3DocumentImporter / S3DocumentGrounder on a worker thread ─► response

The folder is `{folder_name}/current.json` in the bucket; its live version's `merged_chunks.json` (or, when
absent, its `target_chunks.json`, merged here) is the input. The import needs an explicit `collection_name`
(never the folder name), follows the collection's language rule (app.ingestion.pipeline.resolve_document_language)
and registers the document in the collection's S3 registry. Both share the single-flight gate with
`/v1/documents`, since they spend the same provider quota.

Only validation, dispatch, response shaping and logging live here.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, TypeVar

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_base_config, get_gate, get_logger, get_settings
from app.api.errors import http_error
from app.api.routes.documents import _language_or_400
from app.api.schemas import (
    ErrorResponse,
    S3DocumentImportRequest,
    S3DocumentImportResponse,
    S3DocumentRequest,
    S3GroundingResponse,
)
from app.concurrency import ProcessingGate
from app.config.defaults import ChunkerConfig
from app.config.settings import ServerSettings
from app.logging.setup import document_id_var, request_id_var
from app.orchestration.errors import S3DocumentNotFoundError
from app.orchestration.s3_grounding import S3DocumentGrounder, S3GroundingResult
from app.orchestration.s3_import import S3DocumentImporter, S3ImportResult
from app.retrieval.errors import RetrievalError
from app.storage.s3_artifact_store import S3ArtifactStore

router = APIRouter(tags=["documents"])

_RETRY_AFTER_SECONDS = 5
_ERROR_RESPONSES: dict = {
    400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse},
    429: {"model": ErrorResponse}, 500: {"model": ErrorResponse},
}
T = TypeVar("T")


@router.post("/v1/documents/from-s3", response_model=S3DocumentImportResponse, responses=_ERROR_RESPONSES)
async def import_document_from_s3(
    request: S3DocumentImportRequest,
    base_config: ChunkerConfig = Depends(get_base_config),
    gate: ProcessingGate = Depends(get_gate),
    settings: ServerSettings = Depends(get_settings),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> S3DocumentImportResponse:
    document_id = request.folder_name
    language = _language_or_400(request.language)
    importer = S3DocumentImporter(_store(request, settings), base_config, logger)
    result: S3ImportResult = await _run_gated(
        "s3_import", document_id, gate, logger,
        lambda: importer.import_document(document_id, collection_name=request.collection_name, language=language),
    )
    return S3DocumentImportResponse(
        document_id=result.document_id,
        status="completed",
        version=result.version,
        source_artifact=result.source_artifact,
        collection_name=result.collection_name,
        language=result.language,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        uploaded_to_milvus=True,
        classified=result.classified,
    )


@router.post("/v1/documents/from-s3/autogrounding", response_model=S3GroundingResponse, responses=_ERROR_RESPONSES)
async def ground_document_from_s3(
    request: S3DocumentRequest,
    base_config: ChunkerConfig = Depends(get_base_config),
    gate: ProcessingGate = Depends(get_gate),
    settings: ServerSettings = Depends(get_settings),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> S3GroundingResponse:
    document_id = request.folder_name
    grounder = S3DocumentGrounder(_store(request, settings), base_config, logger)
    result: S3GroundingResult = await _run_gated("s3_grounding", document_id, gate, logger, lambda: grounder.ground_document(document_id))
    return S3GroundingResponse(
        document_id=result.document_id,
        status="completed",
        version=result.version,
        source_artifact=result.source_artifact,
        page_count=result.page_count,
        question_count=result.question_count,
        failed_pages=result.failed_pages,
        artifacts=result.artifacts,
    )


def _store(request: S3DocumentRequest, settings: ServerSettings) -> S3ArtifactStore:
    bucket = request.s3_bucket or settings.s3_bucket
    if not bucket:
        raise http_error(400, "S3_BUCKET_REQUIRED", "s3_bucket is required: no S3_BUCKET default is configured")
    return S3ArtifactStore(
        bucket=bucket, region=request.AWS_DEFAULT_REGION or settings.AWS_DEFAULT_REGION, service_version=settings.service_version,
    )


async def _run_gated(channel: str, document_id: str, gate: ProcessingGate, logger: logging.LoggerAdapter, work: Callable[[], T]) -> T:
    """Run `work` on a worker thread behind the gate, mapping failures to the API's error bodies."""
    log_extra: dict[str, Any] = {"channel": channel}
    tokens = (request_id_var.set(uuid.uuid4().hex), document_id_var.set(document_id))
    try:
        if not gate.try_acquire():
            logger.warning("%s rejected: server busy", channel, extra={**log_extra, "stage": "admission", "status": "rejected"})
            raise http_error(429, "SERVER_BUSY", "Another document is processing; retry shortly.", retryable=True, retry_after=_RETRY_AFTER_SECONDS)

        started = time.monotonic()
        logger.info("%s started", channel, extra={**log_extra, "stage": "received", "status": "started"})
        try:
            result = await run_in_threadpool(work)
        except S3DocumentNotFoundError as exc:
            logger.warning("%s rejected: %s", channel, exc, extra={**log_extra, "stage": "resolve", "status": "failed"})
            raise http_error(404, exc.code, str(exc)) from exc
        except RetrievalError as exc:
            logger.warning("%s rejected: %s", channel, exc, extra={**log_extra, "stage": "index_check", "status": "failed"})
            raise http_error(409, exc.code, str(exc)) from exc
        except Exception as exc:
            logger.error(
                "%s failed", channel,
                extra={**log_extra, "stage": "processing", "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)},
                exc_info=True,
            )
            raise http_error(500, "PROCESSING_FAILED", f"{channel} failed: {exc}", retryable=True) from exc
        finally:
            gate.release()

        logger.info(
            "%s completed", channel,
            extra={**log_extra, "stage": "completed", "status": "completed", "duration_ms": int((time.monotonic() - started) * 1000)},
        )
        return result
    finally:
        request_id_var.reset(tokens[0])
        document_id_var.reset(tokens[1])
