"""
app.orchestration.jobs
======================
IngestionJobManager — runs an ingestion on a background thread so no HTTP request waits for it,
records every job on disk, retries a failed run, and continues a run a restart interrupted.

    submit  : validate ─► attach to the document's active job (same content) or ─► acquire the gate
              (single-flight with POST /v1/documents) ─► record ─► start the job thread ─► the job
    run     : admit the staged PDF into its workspace ─► attempts: channel.process(…, on_progress)
              ─► succeeded (IngestResult) | retryable failure → wait, run again | failed (error)
    recover : startup — jobs recorded as queued/running were cut off by a restart; the newest is
              continued from its workspace (`ingest_job_auto_resume`), any other is marked failed
    get     : the job as last updated; finished jobs are kept `ingest_job_retention_hours`

Attempts: a run that fails with a retryable error is run again after `retry_delay_seconds` ×
attempt, up to `max_attempts` runs in total (a run a restart interrupted counts as one). Every
run resumes from the workspace, so only the work in flight is repeated — e.g. the pages stage 3
could not chunk (app.ingestion.errors.IncompleteStageError).

Progress: the pipeline reports `(stage, done, total)`; `progress` is the overall fraction over the
stages this job will run (classification, auto-grounding, indexing and S3 publishing only when
enabled), each weighted by its usual share of the run time. It never goes backwards.

Durability: one JSON file per job under `ServerSettings.jobs_dir` (`JobStore`), rewritten on
every status change and at most every `_PERSIST_SECONDS` while progress moves. Without a
directory (tests) jobs live in memory only.

Identity: the record keeps the submitting request's id and execution context (`X-Execution-*` headers);
the job thread restores both, so its logs carry the request id and every attempt is one `document_ingestion`
trace in the session named by the execution id (the job id by default). The trace ids are recorded on the job.

Failure policy: `ChannelError` → failed, not retryable (the request is wrong, e.g. DOCUMENT_ID_CONFLICT);
`RetrievalError` / `NoContentError` → failed with its code, not retryable (e.g. the collection holds another
language; the PDF has no content); anything else →
`PROCESSING_FAILED`, retryable (the workspace is kept) — run again by the service unless no run could
succeed (a fatal Gemini error: invalid key, no permission, unknown model), which fails at once. Never raises out of the thread. Recording a
job is secondary: a write error is logged and the job runs on.
Concurrency: one lock guards the job table; a job's fields are written only by its own thread
(and read under the lock as a snapshot).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.chunking.chunker import is_fatal_gemini_error
from app.concurrency import ProcessingGate
from app.ingestion.errors import NoContentError
from app.ingestion.pipeline import STAGES, IngestResult
from app.logging.setup import document_id_var, request_id_var
from app.observability.context import ExecutionContext, execution_var, trace_ids_var
from app.orchestration.errors import ChannelError
from app.orchestration.orchestrator import ChannelOrchestrator
from app.retrieval.errors import RetrievalError
from app.utils.fs import atomic_write_json

__all__ = [
    "IngestionJob", "IngestionJobManager", "IngestionRequest", "JobConflictError", "JobStore", "ServerBusyError",
    "active_stages", "overall_progress",
]

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"
_ACTIVE = (QUEUED, RUNNING)
_SOURCE_TYPE = "pdf"
_PUBLISH = "publish"
# Relative share of a typical run; only ratios matter (chunking dominates: one vision call per page).
_STAGE_WEIGHTS: Dict[str, float] = {
    "ocr": 10, "chunk": 55, "merge": 1, "classify": 15, "vocab": 2, "ground": 10, "index": 5, _PUBLISH: 2,
}
_SECONDS_PER_HOUR = 3600
_PERSIST_SECONDS = 5.0  # progress-only updates are written to disk at most this often
_PROCESSING_FAILED = "PROCESSING_FAILED"
_INTERRUPTED = "INTERRUPTED"


class ServerBusyError(Exception):
    """Another document is being ingested."""


class JobConflictError(Exception):
    """The document already has an active job for different content."""


@dataclass(frozen=True)
class IngestionRequest:
    """What the caller asked for — recorded with the job so a restart can continue it."""

    document_id: str
    filename: str
    content_hash: str
    collection_name: Optional[str] = None   # the API always sets it; None only in records that predate it
    autogrounding: bool = False
    language: Optional[str] = None
    classification: Optional[bool] = None
    s3_bucket: Optional[str] = None
    AWS_DEFAULT_REGION: Optional[str] = None
    replace: bool = False                    # replace a different PDF already published under this document id
    execution: Optional[Dict[str, Any]] = None  # ExecutionContext.to_dict(): the X-Execution-* / X-Dataset-* / X-Case-Id headers

    _NOT_OPTIONS = ("document_id", "filename", "content_hash", "execution")

    def options(self) -> Dict[str, Any]:
        """The channel's `process` options."""
        return {key: value for key, value in asdict(self).items() if key not in self._NOT_OPTIONS}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestionRequest":
        """Tolerates records written by older versions (e.g. the removed `upload_vector_db`)."""
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(frozen=True)
class IngestionJob:
    job_id: str
    request: IngestionRequest
    stages: Tuple[str, ...]
    max_attempts: int = 1
    status: str = QUEUED
    attempt: int = 0
    stage: Optional[str] = None
    stage_done: int = 0
    stage_total: int = 0
    progress: float = 0.0
    created_at: str = field(default_factory=lambda: _now())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    finished_epoch: Optional[float] = None
    result: Optional[IngestResult] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: bool = False
    request_id: Optional[str] = None         # of the submitting request: the job thread logs and traces under it
    trace_ids: Tuple[str, ...] = ()          # one root trace per attempt (empty when tracing is off)

    @property
    def document_id(self) -> str:
        return self.request.document_id

    @property
    def content_hash(self) -> str:
        return self.request.content_hash

    @property
    def execution_id(self) -> str:
        """Groups the job's traces (Langfuse session): the caller's X-Execution-Id, else the job id."""
        return ExecutionContext.from_dict(self.request.execution).execution_id or self.job_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestionJob":
        result = data.get("result")
        known = {f.name for f in fields(cls)}
        return cls(**{
            **{key: value for key, value in data.items() if key in known},
            "request": IngestionRequest.from_dict(data["request"]),
            "stages": tuple(data["stages"]),
            "result": IngestResult(**result) if result else None,
            "trace_ids": tuple(data.get("trace_ids") or ()),
        })


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _failure(code: str, message: str, *, retryable: bool) -> Dict[str, Any]:
    """A failed job's final fields."""
    return dict(
        status=FAILED, finished_at=_now(), finished_epoch=time.time(), result=None,
        error_code=code, error_message=message, retryable=retryable,
    )


def active_stages(*, classify: bool, ground: bool, index: bool, publish: bool) -> Tuple[str, ...]:
    enabled = {"classify": classify, "ground": ground, "index": index}
    stages = tuple(stage for stage in STAGES if enabled.get(stage, True))
    return stages + ((_PUBLISH,) if publish else ())


def overall_progress(stages: Sequence[str], stage: str, done: int, total: int) -> float:
    """Weighted fraction of the run: finished stages + the current one's done/total."""
    if stage not in stages:
        return 0.0
    weights = [_STAGE_WEIGHTS.get(name, 1.0) for name in stages]
    index = stages.index(stage)
    fraction = min(max(done / total, 0.0), 1.0) if total > 0 else 0.0
    return (sum(weights[:index]) + weights[index] * fraction) / sum(weights)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class JobStore:
    """One JSON file per job in `directory`; with no directory, nothing is written (memory only)."""

    def __init__(self, directory: Optional[Path], logger: logging.LoggerAdapter) -> None:
        self._dir = Path(directory) if directory is not None else None
        self._log = logger

    def save(self, job: IngestionJob) -> None:
        if self._dir is None:
            return
        try:
            atomic_write_json(self._dir / f"{job.job_id}.json", job.to_dict())
        except OSError as exc:  # the job runs on; only its restart record is stale
            self._log.error("Could not record ingestion job %s: %s", job.job_id, exc)

    def delete(self, job_id: str) -> None:
        if self._dir is not None:
            (self._dir / f"{job_id}.json").unlink(missing_ok=True)

    def load_all(self) -> List[IngestionJob]:
        if self._dir is None or not self._dir.is_dir():
            return []
        jobs = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                jobs.append(IngestionJob.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, TypeError, KeyError) as exc:
                self._log.warning("Ignoring unreadable ingestion job record %s: %s", path, exc)
        return jobs


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class IngestionJobManager:
    """See the module docstring."""

    def __init__(
        self,
        gate: ProcessingGate,
        *,
        store: JobStore,
        retention_hours: float,
        max_attempts: int,
        retry_delay_seconds: float,
        s3_publish: bool,
        classify_default: bool,
        logger: logging.LoggerAdapter,
    ) -> None:
        self._gate = gate
        self._store = store
        self._retention_seconds = retention_hours * _SECONDS_PER_HOUR
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay_seconds
        self._s3_publish = s3_publish
        self._classify_default = classify_default
        self._log = logger
        self._lock = threading.Lock()
        self._jobs: Dict[str, IngestionJob] = {}
        self._persisted_at: Dict[str, float] = {}
        self._stopping = threading.Event()

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def submit(self, orchestrator: ChannelOrchestrator, request: IngestionRequest, source: Path) -> Tuple[IngestionJob, bool]:
        """
        (job, created). `source` is the staged upload; the job consumes it (it is removed when the
        submit is rejected or attaches). An active job for the same document and content is returned
        as is (a retried submit attaches to it). Raises ChannelError (invalid upload),
        JobConflictError, ServerBusyError.
        """
        started = False
        try:
            orchestrator.resolve(_SOURCE_TYPE).validate(request.filename, source)
            with self._lock:
                self._prune()
                active = self._active_job(request.document_id)
                if active is not None:
                    if active.content_hash != request.content_hash:
                        raise JobConflictError(f"Document '{request.document_id}' is already being ingested from another file")
                    self._log.info("Ingestion job resubmitted, attaching: job=%s document=%s", active.job_id, request.document_id)
                    return active, False
                if not self._gate.try_acquire():
                    raise ServerBusyError("Another document is processing; retry shortly.")
                job = IngestionJob(
                    job_id=uuid.uuid4().hex, request=request, stages=self._stages(request), max_attempts=self._max_attempts,
                    request_id=request_id_var.get(),
                )
                self._jobs[job.job_id] = job
            self._store.save(job)
            try:
                self._start(orchestrator, job.job_id, source)
            except BaseException:
                self._gate.release()
                with self._lock:
                    self._jobs.pop(job.job_id, None)
                self._store.delete(job.job_id)
                raise
            started = True
        finally:
            if not started:
                source.unlink(missing_ok=True)
        self._log.info("Ingestion job accepted: job=%s document=%s stages=%s", job.job_id, request.document_id, ",".join(job.stages))
        return job, True

    def get(self, job_id: str) -> Optional[IngestionJob]:
        with self._lock:
            self._prune()
            return self._jobs.get(job_id)

    def recover(self, orchestrator: ChannelOrchestrator, *, auto_resume: bool) -> Optional[IngestionJob]:
        """
        Startup: load the recorded jobs. A job still queued/running was cut off by a restart: the newest
        is continued from its workspace when `auto_resume` and attempts remain, every other one is marked
        failed (`INTERRUPTED`, retryable — resubmitting the PDF resumes it). Returns the continued job.
        """
        jobs = self._store.load_all()
        with self._lock:
            for job in jobs:
                self._jobs[job.job_id] = job
            self._prune()
            interrupted = sorted((j for j in self._jobs.values() if j.status in _ACTIVE), key=lambda j: j.created_at)
        resume = interrupted[-1] if interrupted and auto_resume else None
        if resume is not None and resume.attempt >= resume.max_attempts:
            self._log.error("Interrupted ingestion job %s already used %d attempt(s); not resuming", resume.job_id, resume.attempt)
            resume = None
        for job in interrupted:
            if job is not resume:
                self._fail(job.job_id, _INTERRUPTED, "Interrupted by a service restart; resubmit the PDF to resume", retryable=True)
        if resume is None or not self._gate.try_acquire():
            return None
        self._log.info("Resuming ingestion job interrupted by a restart: job=%s document=%s", resume.job_id, resume.document_id)
        try:
            self._start(orchestrator, resume.job_id, None)
        except BaseException:
            self._gate.release()
            raise
        return resume

    def stop(self) -> None:
        """Shutdown: a job waiting to retry stops waiting (its record stays active, so the next start resumes it)."""
        self._stopping.set()

    # ------------------------------------------------------------------ #
    # Job thread
    # ------------------------------------------------------------------ #

    def _start(self, orchestrator: ChannelOrchestrator, job_id: str, source: Optional[Path]) -> None:
        threading.Thread(
            target=self._run, args=(orchestrator, job_id, source), name=f"ingest-{job_id[:8]}", daemon=True,
        ).start()

    def _run(self, orchestrator: ChannelOrchestrator, job_id: str, source: Optional[Path]) -> None:
        """
        Gate held on entry; released on exit — before the final status is published, so a caller that
        sees the job finish can submit the next document at once. `source` is the staged upload, or
        None to continue a recorded job.
        """
        started = time.monotonic()
        job = self._snapshot(job_id)
        # A new thread starts without the request's context: restore it so logs and traces are attributable.
        document_id_var.set(job.document_id)
        request_id_var.set(job.request_id)
        execution_var.set(ExecutionContext.from_dict(job.request.execution).with_defaults(execution_id=job.job_id))
        outcome: Optional[Dict[str, Any]] = None
        try:
            outcome = self._execute(orchestrator, job, source)
        except Exception as exc:  # a bug in the manager itself: still a failed job, never a dead thread
            self._log.error("Ingestion job crashed: job=%s error=%s", job_id, exc, exc_info=True)
            outcome = _failure(_PROCESSING_FAILED, f"Document processing failed: {exc}", retryable=True)
        finally:
            self._gate.release()
            if source is not None:
                source.unlink(missing_ok=True)  # admitted sources were moved; this only removes a rejected one
        if outcome is None:
            return  # shutting down mid-wait: the record stays running, so the next start resumes it
        self._update(job_id, **outcome)
        final = self._snapshot(job_id)
        if final.status == SUCCEEDED and final.result is not None:
            self._log.info(
                "Ingestion job succeeded: job=%s pages=%d chunks=%d attempts=%d elapsed=%.0fs",
                job_id, final.result.page_count, final.result.chunk_count, final.attempt, time.monotonic() - started,
            )

    def _execute(self, orchestrator: ChannelOrchestrator, job: IngestionJob, source: Optional[Path]) -> Optional[Dict[str, Any]]:
        """Admit, then run attempts; returns the job's final fields (None when shutdown interrupted a retry wait)."""
        channel = orchestrator.resolve(_SOURCE_TYPE)
        request = job.request
        try:
            if source is not None:
                admitted = channel.admit(document_id=request.document_id, source=source, content_hash=request.content_hash)
            else:
                admitted = channel.admitted_source(document_id=request.document_id, content_hash=request.content_hash)
        except ChannelError as exc:
            return _failure(exc.code, str(exc), retryable=False)
        if admitted is None:
            return _failure(_INTERRUPTED, "The interrupted run's workspace is gone; resubmit the PDF", retryable=True)

        attempt = job.attempt
        base_execution = execution_var.get() or ExecutionContext()
        while True:
            attempt += 1
            self._update(job.job_id, status=RUNNING, attempt=attempt, started_at=job.started_at or _now())
            execution_var.set(base_execution.with_defaults(attempt=attempt))  # one trace per attempt, same session
            trace_ids_var.set(attempt_traces := [])
            try:
                result = channel.process(
                    document_id=request.document_id, source=admitted, content_hash=request.content_hash,
                    on_progress=lambda s, d, t: self._on_progress(job.job_id, s, d, t), **request.options(),
                )
            except (ChannelError, RetrievalError, NoContentError) as exc:
                self._record_traces(job.job_id, attempt_traces)
                return _failure(exc.code, str(exc), retryable=False)
            except Exception as exc:  # every pipeline failure is the job's outcome; the workspace is kept for a resume
                self._record_traces(job.job_id, attempt_traces)
                if attempt >= job.max_attempts or self._stopping.is_set() or is_fatal_gemini_error(exc):
                    self._log.error("Ingestion job failed: job=%s attempt=%d/%d error=%s", job.job_id, attempt, job.max_attempts, exc,
                                    exc_info=True)
                    return _failure(_PROCESSING_FAILED, f"Document processing failed: {exc}", retryable=True)
                delay = self._retry_delay * attempt
                self._log.warning(
                    "Ingestion job attempt %d/%d failed: job=%s error=%s — running again in %.0fs (resumes from the workspace)",
                    attempt, job.max_attempts, job.job_id, exc, delay, exc_info=True,
                )
                if self._stopping.wait(delay):
                    return None
                continue
            self._record_traces(job.job_id, attempt_traces)
            return dict(
                status=SUCCEEDED, finished_at=_now(), finished_epoch=time.time(), result=result, progress=1.0,
                error_code=None, error_message=None, retryable=False,
            )

    def _record_traces(self, job_id: str, trace_ids: List[str]) -> None:
        if not trace_ids:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            self._jobs[job_id] = updated = replace(job, trace_ids=job.trace_ids + tuple(trace_ids))
        self._store.save(updated)

    def _on_progress(self, job_id: str, stage: str, done: int, total: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            progress = max(job.progress, overall_progress(job.stages, stage, done, total))
            self._jobs[job_id] = updated = replace(job, stage=stage, stage_done=done, stage_total=total, progress=progress)
            due = time.monotonic() - self._persisted_at.get(job_id, 0.0) >= _PERSIST_SECONDS
            if due:
                self._persisted_at[job_id] = time.monotonic()
        if due:
            self._store.save(updated)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id] = updated = replace(self._jobs[job_id], **fields)
            self._persisted_at[job_id] = time.monotonic()
        self._store.save(updated)

    def _fail(self, job_id: str, code: str, message: str, *, retryable: bool) -> None:
        self._update(job_id, **_failure(code, message, retryable=retryable))

    def _snapshot(self, job_id: str) -> IngestionJob:
        with self._lock:
            return self._jobs[job_id]

    # ------------------------------------------------------------------ #
    # Helpers (lock held where noted)
    # ------------------------------------------------------------------ #

    def _stages(self, request: IngestionRequest) -> Tuple[str, ...]:
        return active_stages(
            classify=self._classify_default if request.classification is None else request.classification,
            ground=request.autogrounding,
            index=bool(request.collection_name),
            publish=self._s3_publish,
        )

    def _active_job(self, document_id: str) -> Optional[IngestionJob]:
        """Lock held."""
        return next((j for j in self._jobs.values() if j.document_id == document_id and j.status in _ACTIVE), None)

    def _prune(self) -> None:
        """Lock held: drop finished jobs older than the retention."""
        cutoff = time.time() - self._retention_seconds
        expired = [j.job_id for j in self._jobs.values() if j.finished_epoch is not None and j.finished_epoch < cutoff]
        for job_id in expired:
            del self._jobs[job_id]
            self._persisted_at.pop(job_id, None)
            self._store.delete(job_id)

