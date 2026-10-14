"""
app.api.routes.documents
========================
`POST /v1/documents` — multipart PDF upload, processed synchronously behind the admission gate.

    stream to disk ─► acquire gate (429 if busy) ─► orchestrator.dispatch("pdf") on a worker thread ─► response

Fine for documents that finish within the caller's HTTP timeout; long ones (hundreds of pages) belong
on `POST /v1/ingestion-jobs`, which also retries and survives a restart.

`collection_name` is required (`CollectionName`: one lower-case spelling per collection) and is never
derived from the file name. A collection holds one language — the language of its first document. A later
document must be detected as that language, else 409 before chunking; `language` ("ja" / "en") is only a
hint for the first document of a new collection.
`classification` overrides `CLASSIFICATION_ENABLED` (stage 5: main/sub category per chunk).
`document_id` is the sanitised file stem; with S3 on, a *different* PDF already published under it (a
same-named file) is 409 `DOCUMENT_ID_CONFLICT` unless `replace=true`. A PDF yielding no chunk is 422 `NO_CHUNKS`.

Only validation, dispatch, response shaping and logging live here.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_gate, get_logger, get_orchestrator, get_settings
from app.api.errors import http_error
from app.api.schemas import CollectionName, DocumentProcessResponse, ErrorResponse
from app.api.uploads import StagedUpload, stage_upload
from app.concurrency import ProcessingGate
from app.config.settings import ServerSettings
from app.ingestion.errors import NoContentError
from app.ingestion.pipeline import IngestResult
from app.logging.setup import document_id_var, request_id_var
from app.orchestration.errors import ChannelError
from app.orchestration.orchestrator import ChannelOrchestrator
from app.retrieval.errors import RetrievalError
from app.retrieval.language import SUPPORTED_LANGUAGES, normalize_language
from app.utils.naming import sanitize_collection_name

router = APIRouter(tags=["documents"])

_RETRY_AFTER_SECONDS = 5


@router.post(
    "/v1/documents",
    response_model=DocumentProcessResponse,
    responses={
        400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
        429: {"model": ErrorResponse}, 500: {"model": ErrorResponse},
    },
)
async def create_document(
    file: UploadFile,
    collection_name: Annotated[CollectionName, Form()],
    autogrounding: bool = Form(default=False),
    language: Optional[str] = Form(default=None),
    classification: Optional[bool] = Form(default=None),
    s3_bucket: Optional[str] = Form(default=None),
    AWS_DEFAULT_REGION: Optional[str] = Form(default=None),
    replace: bool = Form(default=False),
    orchestrator: ChannelOrchestrator = Depends(get_orchestrator),
    gate: ProcessingGate = Depends(get_gate),
    settings: ServerSettings = Depends(get_settings),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> DocumentProcessResponse:
    filename = file.filename or ""
    document_id = document_id_for(filename)
    document_language = _language_or_400(language)
    tokens = (request_id_var.set(uuid.uuid4().hex), document_id_var.set(document_id))
    staged: Optional[StagedUpload] = None
    try:
        staged = await receive_upload(file, settings)
        if not gate.try_acquire():
            logger.warning("document rejected: server busy", extra={"channel": "pdf", "stage": "admission", "status": "rejected"})
            raise http_error(429, "SERVER_BUSY", "Another document is processing; retry shortly.", retryable=True, retry_after=_RETRY_AFTER_SECONDS)

        started = time.monotonic()
        logger.info("document processing started", extra={"channel": "pdf", "stage": "received", "status": "started"})
        try:
            result: IngestResult = await run_in_threadpool(
                orchestrator.dispatch,
                "pdf",
                document_id=document_id,
                filename=filename,
                source=staged.path,
                content_hash=staged.sha256,
                collection_name=collection_name,
                autogrounding=autogrounding,
                language=document_language,
                classification=classification,
                s3_bucket=s3_bucket,
                AWS_DEFAULT_REGION=AWS_DEFAULT_REGION,
                replace=replace,
            )
        except ChannelError as exc:
            logger.warning(
                "document rejected: %s", exc,
                extra={"channel": "pdf", "stage": "validation", "status": "failed", "error_type": type(exc).__name__},
            )
            raise http_error(exc.http_status, exc.code, str(exc)) from exc
        except NoContentError as exc:
            logger.warning("document rejected: %s", exc, extra={"channel": "pdf", "stage": "merge", "status": "failed"})
            raise http_error(422, exc.code, str(exc)) from exc
        except RetrievalError as exc:
            logger.warning(
                "document rejected: %s", exc,
                extra={"channel": "pdf", "stage": "index_check", "status": "failed", "error_type": type(exc).__name__},
            )
            raise http_error(409, exc.code, str(exc)) from exc
        except Exception as exc:
            logger.error(
                "document processing failed",
                extra={"channel": "pdf", "stage": "processing", "status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)},
                exc_info=True,
            )
            raise http_error(500, "PROCESSING_FAILED", f"Document processing failed: {exc}", retryable=True) from exc
        finally:
            gate.release()

        logger.info(
            "document processing completed",
            extra={
                "channel": "pdf", "stage": "completed", "status": "completed",
                "duration_ms": int((time.monotonic() - started) * 1000),
                "page_count": result.page_count, "chunk_count": result.chunk_count, "collection_name": result.collection_name,
            },
        )
        return document_response(document_id, result, debug=settings.debug)
    finally:
        if staged is not None:
            staged.discard()  # a no-op once the channel moved it into the workspace
        request_id_var.reset(tokens[0])
        document_id_var.reset(tokens[1])


async def receive_upload(file: UploadFile, settings: ServerSettings) -> StagedUpload:
    """Stream the upload into the staging directory (400 when it exceeds the size limit)."""
    try:
        return await run_in_threadpool(stage_upload, file.file, file.filename or "", settings.uploads_dir, settings.max_upload_size_bytes)
    except ChannelError as exc:
        raise http_error(400, exc.code, str(exc)) from exc


def document_response(document_id: str, result: IngestResult, *, debug: bool) -> DocumentProcessResponse:
    """The one place an IngestResult becomes the API's document response (also used by ingestion jobs)."""
    return DocumentProcessResponse(
        document_id=document_id,
        status="completed",
        collection_name=result.collection_name,
        language=result.language,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        uploaded_to_milvus=result.uploaded,
        classified=result.classification_path is not None,
        created_categories=result.created_categories,
        artifacts=_artifact_paths(result) if debug else None,
    )


def document_id_for(filename: str) -> str:
    stem = Path(filename).stem
    return sanitize_collection_name(stem) if stem else f"document_{uuid.uuid4().hex}"


def _language_or_400(language: Optional[str]) -> Optional[str]:
    code = normalize_language(language)
    if not code or code == "auto":
        return None
    if code not in SUPPORTED_LANGUAGES:
        raise http_error(400, "INVALID_LANGUAGE", f"language must be one of {sorted(SUPPORTED_LANGUAGES)} or omitted, got {language!r}")
    return code


def _artifact_paths(result: IngestResult) -> dict:
    return {
        "ocr_markdown_path": result.ocr_markdown_path,
        "target_chunks_path": result.target_chunks_path,
        "merged_chunks_path": result.merged_chunks_path,
        "vocab_index_path": result.vocab_index_path,
        "classification_path": result.classification_path,
        "groundtruth_path": result.groundtruth_path,
    }
