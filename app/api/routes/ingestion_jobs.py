"""
app.api.routes.ingestion_jobs
=============================
Asynchronous ingestion — for documents that take longer than any HTTP timeout (1,000 pages ≈ hours).

    POST /v1/ingestion-jobs           same form as POST /v1/documents → 202 + the job (200 when the
                                      same document and content already has an active job: attached)
    GET  /v1/ingestion-jobs/{job_id}  the job: status, stage, stage_done/stage_total, progress, result | error

Errors: 400 invalid upload / language; 422 missing or invalid `collection_name`; 409 the document is already being ingested
from another file; 429 another document is processing (`Retry-After`); 404 unknown job (finished
longer than `INGEST_JOB_RETENTION_HOURS` ago). Jobs are recorded on disk: after a restart they are
still readable, and the one that was running continues by itself (`INGEST_JOB_AUTO_RESUME`).
A job's own failure is its `error` (`retryable: true` → resubmitting resumes), never a status code
here — e.g. `DOCUMENT_ID_CONFLICT` (a different PDF under this file name; resubmit with replace=true) or
`NO_CHUNKS`; retryable failures are first retried by the service (`attempt` / `max_attempts`).
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Form, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_ingestion_jobs, get_logger, get_orchestrator, get_settings
from app.api.errors import http_error
from app.api.routes.documents import _language_or_400, document_id_for, document_response, receive_upload
from app.api.schemas import CollectionName, ErrorDetail, ErrorResponse, IngestionJobResponse
from app.config.settings import ServerSettings
from app.observability.context import current_execution
from app.orchestration.errors import ChannelError
from app.orchestration.jobs import IngestionJob, IngestionJobManager, IngestionRequest, JobConflictError, ServerBusyError
from app.orchestration.orchestrator import ChannelOrchestrator

router = APIRouter(tags=["ingestion-jobs"])

_RETRY_AFTER_SECONDS = 30  # a running ingestion takes minutes to hours


@router.post(
    "/v1/ingestion-jobs",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
async def create_ingestion_job(
    response: Response,
    file: UploadFile,
    collection_name: Annotated[CollectionName, Form()],
    autogrounding: bool = Form(default=False),
    language: Optional[str] = Form(default=None),
    classification: Optional[bool] = Form(default=None),
    s3_bucket: Optional[str] = Form(default=None),
    AWS_DEFAULT_REGION: Optional[str] = Form(default=None),
    replace: bool = Form(default=False),
    orchestrator: ChannelOrchestrator = Depends(get_orchestrator),
    jobs: IngestionJobManager = Depends(get_ingestion_jobs),
    settings: ServerSettings = Depends(get_settings),
    logger=Depends(get_logger),
) -> IngestionJobResponse:
    filename = file.filename or ""
    document_id = document_id_for(filename)
    document_language = _language_or_400(language)
    staged = await receive_upload(file, settings)
    request = IngestionRequest(
        document_id=document_id, filename=filename, content_hash=staged.sha256, collection_name=collection_name,
        autogrounding=autogrounding, language=document_language,
        classification=classification, s3_bucket=s3_bucket, AWS_DEFAULT_REGION=AWS_DEFAULT_REGION, replace=replace,
        execution=current_execution().to_dict(),  # the X-Execution-* headers travel with the job (and its restart)
    )
    try:
        job, created = await run_in_threadpool(jobs.submit, orchestrator, request, staged.path)  # validation opens the PDF
    except ChannelError as exc:
        raise http_error(exc.http_status, exc.code, str(exc)) from exc
    except JobConflictError as exc:
        raise http_error(409, "DOCUMENT_BUSY", str(exc)) from exc
    except ServerBusyError as exc:
        logger.warning("ingestion job rejected: server busy", extra={"channel": "pdf", "stage": "admission", "status": "rejected"})
        raise http_error(429, "SERVER_BUSY", str(exc), retryable=True, retry_after=_RETRY_AFTER_SECONDS) from exc
    if not created:
        response.status_code = status.HTTP_200_OK
    response.headers["Location"] = f"/v1/ingestion-jobs/{job.job_id}"
    return job_response(job, debug=settings.debug)


@router.get("/v1/ingestion-jobs/{job_id}", response_model=IngestionJobResponse, responses={404: {"model": ErrorResponse}})
def get_ingestion_job(
    job_id: str,
    jobs: IngestionJobManager = Depends(get_ingestion_jobs),
    settings: ServerSettings = Depends(get_settings),
) -> IngestionJobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise http_error(404, "JOB_NOT_FOUND", f"Ingestion job '{job_id}' not found (finished long ago, or the service restarted)")
    return job_response(job, debug=settings.debug)


def job_response(job: IngestionJob, *, debug: bool) -> IngestionJobResponse:
    return IngestionJobResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        status=job.status,
        stage=job.stage,
        stage_done=job.stage_done,
        stage_total=job.stage_total,
        progress=round(job.progress, 4),
        stages=list(job.stages),
        attempt=job.attempt,
        max_attempts=job.max_attempts,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result=document_response(job.document_id, job.result, debug=debug) if job.result is not None else None,
        error=ErrorDetail(code=job.error_code, message=job.error_message or "", retryable=job.retryable) if job.error_code else None,
        execution_id=job.execution_id,
        trace_ids=list(job.trace_ids),
    )
