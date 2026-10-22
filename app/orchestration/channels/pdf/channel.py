"""
app.orchestration.channels.pdf.channel
======================================
PDFChannel — turns an uploaded PDF into an `IngestPipeline` run plus optional S3 publication.

    validate  : extension, size, `%PDF-` signature, and the file itself — PyMuPDF must open it, it
                must not be encrypted and must have 1..`max_pdf_pages` pages — before any paid work
    workspace : `<workspace root>/<document_id>/` (root: `DEBUG_WORKSPACE` in debug mode, else
                `INGEST_WORKSPACE_DIR`) holding the PDF, a content-hash marker and the artifacts.
                A different PDF under the same id (marker mismatch) starts clean; a matching one
                resumes (OCR windows, chunked pages, classifications). Production removes the
                workspace on success and keeps it on failure; workspaces idle longer than
                `ingest_workspace_retention_hours` are pruned before each admission. Debug keeps all.
    identity  : the server-generated `document_id` names the PDF. When S3 is on, its `current.json` is the
                record of which content holds that id: a different PDF under the same id (a same-named
                file) is rejected with `DocumentIdConflictError` (409) before any paid work, unless
                `replace=True`. Without S3 the service keeps no such record and the newer PDF replaces it.
    S3        : when `s3_upload_enabled`, the artifact directory is uploaded under the
                PDF's content hash, published as current, old versions are pruned, and the
                document is registered in its collection (`_collections/{collection}.json`)

Main path
    validate ─► admit (prune stale; reset or reuse the workspace; move the PDF in; write the marker)
             ─► process: [identity check] ─► IngestPipeline.run ─► [S3 upload → publish → retention → register] ─► remove workspace (production, success)

`process` is one `document_ingestion` trace (app.observability): the pipeline's stages are its child spans,
the S3 publication its `s3_publish` span; its output records page / chunk counts and the language.

The PDF is moved in only when the workspace does not hold it yet: replacing it would make every
artifact look older than the PDF and the resume would redo all of them. The marker is written
after the PDF, so a marker always vouches for a complete PDF.

Failure policy: validation raises `InvalidDocumentError` (HTTP 400); pipeline
and S3 upload/publish errors propagate (HTTP 500); S3 retention, the collection registry (it only
feeds query reformulation) and workspace pruning are best-effort.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from app.config.defaults import ChunkerConfig
from app.config.settings import ServerSettings
from app.ingestion.pipeline import IngestPipeline, IngestResult, ProgressCallback
from app.app_logging.setup import get_logger
from app.observability.modules import ModuleName, SpanName
from app.observability.telemetry import get_telemetry
from app.orchestration.errors import DocumentIdConflictError, InvalidDocumentError
from app.storage.s3_artifact_store import PublishResult, S3ArtifactStore, register_in_collection
from app.utils.fs import atomic_write_text, move_file

__all__ = ["PDFChannel"]

_PDF_MAGIC = b"%PDF-"
_MAGIC_WINDOW = 1024  # the PDF spec lets the header start anywhere in the first 1 KiB
_SOURCE_MARKER = ".source.sha256"  # content hash of the PDF this workspace belongs to
_SECONDS_PER_HOUR = 3600


class PDFChannel:
    source_type = "pdf"

    def __init__(self, settings: ServerSettings, base_config: ChunkerConfig, logger: Optional[logging.LoggerAdapter] = None) -> None:
        self._settings = settings
        self._base_config = base_config
        self._log = logger or get_logger(module="PDF_CHANNEL")

    # ------------------------------------------------------------------ #
    # Channel protocol
    # ------------------------------------------------------------------ #

    def validate(self, filename: str, source: Path) -> None:
        if not filename.lower().endswith(".pdf"):
            raise InvalidDocumentError(f"Expected a .pdf file, got {filename!r}")
        size = Path(source).stat().st_size
        if size == 0:
            raise InvalidDocumentError("Uploaded file is empty")
        if size > self._settings.max_upload_size_bytes:
            raise InvalidDocumentError(f"File size {size} bytes exceeds the {self._settings.max_upload_size_bytes}-byte limit")
        with open(source, "rb") as handle:
            if _PDF_MAGIC not in handle.read(_MAGIC_WINDOW):
                raise InvalidDocumentError("File does not look like a PDF (missing %PDF- signature)")
        self._validate_document(Path(source))

    def admit(self, *, document_id: str, source: Path, content_hash: str) -> Path:
        workspace = self._workspace(document_id, content_hash)
        pdf_path = workspace / f"{document_id}.pdf"
        if Path(source).resolve() != pdf_path.resolve():
            if pdf_path.exists():  # same content (the marker matched): keep the file and its mtime
                Path(source).unlink(missing_ok=True)
            else:
                move_file(Path(source), pdf_path)
        atomic_write_text(workspace / _SOURCE_MARKER, content_hash)  # also refreshes the mtime the pruning reads
        return pdf_path

    def admitted_source(self, *, document_id: str, content_hash: str) -> Optional[Path]:
        workspace = self._settings.workspace_root / document_id
        pdf_path = workspace / f"{document_id}.pdf"
        return pdf_path if _marker(workspace) == content_hash and pdf_path.is_file() else None

    def process(
        self,
        *,
        document_id: str,
        source: Path,
        content_hash: str,
        collection_name: Optional[str] = None,
        autogrounding: bool = False,
        language: Optional[str] = None,
        classification: Optional[bool] = None,
        s3_bucket: Optional[str] = None,
        AWS_DEFAULT_REGION: Optional[str] = None,
        replace: bool = False,
        on_progress: Optional[ProgressCallback] = None,
    ) -> IngestResult:
        store = self._resolve_store(s3_bucket, AWS_DEFAULT_REGION)  # validated before any work starts
        progress: ProgressCallback = on_progress or (lambda stage, done, total: None)
        workspace = Path(source).parent
        pipeline = IngestPipeline(self._base_config.model_copy(update={"resources_base": str(workspace)}), self._log)
        telemetry = get_telemetry()
        with telemetry.trace(
            ModuleName.DOCUMENT_INGESTION, document_id=document_id, collection_name=collection_name, content_hash=content_hash,
            language_hint=language, classification=classification, autogrounding=autogrounding, replace=replace,
            ocr_technique=self._base_config.ocr_technique, chunking_model=self._base_config.chunking_model,
        ) as trace:
            if store is not None and not replace:
                _check_identity(store, document_id, content_hash)
            try:
                result = pipeline.run(
                    Path(source),
                    collection_name=collection_name,
                    autogrounding=autogrounding,
                    language=language,
                    classification=classification,
                    on_progress=progress,
                )
                if store is not None:
                    progress("publish", 0, 1)
                    with telemetry.span(SpanName.S3_PUBLISH, version=content_hash) as span:
                        published = self._publish(store, Path(result.base_dir), document_id, content_hash)
                        span.set_output(uploaded_files=published.uploaded_file_count, already_existed=published.already_existed)
                    if result.collection_name:
                        register_in_collection(store, result.collection_name, document_id, language=result.language, logger=self._log)
                    progress("publish", 1, 1)
            except BaseException:
                if not self._settings.debug:
                    self._log.warning("Ingestion failed; workspace kept for a resume: %s", workspace)
                raise
            trace.set_output(
                status="completed", page_count=result.page_count, chunk_count=result.chunk_count, language=result.language,
                uploaded_to_milvus=result.uploaded, classified=result.classification_path is not None,
            )
        if not self._settings.debug:
            shutil.rmtree(workspace, ignore_errors=True)  # a finished document leaves nothing on local disk
        return result

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _validate_document(self, source: Path) -> None:
        try:
            with fitz.open(source) as document:
                encrypted, pages = document.needs_pass, document.page_count
        except Exception as exc:  # PyMuPDF's errors are not a stable hierarchy
            raise InvalidDocumentError(f"File cannot be read as a PDF: {exc}") from exc
        if encrypted:
            raise InvalidDocumentError("PDF is password-protected; upload an unencrypted copy")
        if pages == 0:
            raise InvalidDocumentError("PDF has no pages")
        if pages > self._settings.max_pdf_pages:
            raise InvalidDocumentError(f"PDF has {pages} pages; the limit is {self._settings.max_pdf_pages} (MAX_PDF_PAGES)")

    def _resolve_store(self, s3_bucket: Optional[str], AWS_DEFAULT_REGION: Optional[str]) -> Optional[S3ArtifactStore]:
        if not self._settings.s3_upload_enabled:
            return None
        bucket = s3_bucket or self._settings.s3_bucket
        if not bucket:
            raise InvalidDocumentError("s3_bucket is required: S3 upload is enabled and no S3_BUCKET default is configured")
        return S3ArtifactStore(
            bucket=bucket,
            region=AWS_DEFAULT_REGION or self._settings.AWS_DEFAULT_REGION,
            service_version=self._settings.service_version,
        )

    def _workspace(self, document_id: str, content_hash: str) -> Path:
        """The document's workspace, emptied when it belongs to other content; created if absent."""
        root = self._settings.workspace_root
        root.mkdir(parents=True, exist_ok=True)
        if not self._settings.debug:
            self._prune(root, keep=document_id)
        workspace = root / document_id
        if workspace.exists() and _marker(workspace) != content_hash:
            self._log.info("Workspace holds another version of %s; starting clean", document_id)
            shutil.rmtree(workspace, ignore_errors=True)
        if workspace.exists():
            self._log.info("Resuming from the workspace of a previous run: %s", workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _prune(self, root: Path, *, keep: str) -> None:
        """Remove workspaces no run touched within the retention (abandoned failures)."""
        cutoff = time.time() - self._settings.ingest_workspace_retention_hours * _SECONDS_PER_HOUR
        for entry in root.iterdir():
            marker = entry / _SOURCE_MARKER
            if entry.name == keep or not entry.is_dir() or not marker.exists():
                continue  # also skips the `_jobs` / `_uploads` folders, which have no marker
            try:
                if marker.stat().st_mtime < cutoff:
                    shutil.rmtree(entry)
                    self._log.info("Pruned abandoned ingestion workspace: %s", entry)
            except OSError as exc:  # best effort; the next run tries again
                self._log.warning("Could not prune workspace %s: %s", entry, exc)

    def _publish(self, store: S3ArtifactStore, artifact_dir: Path, document_id: str, content_hash: str) -> PublishResult:
        published = store.upload_artifacts(artifact_dir, document_id=document_id, content_hash=content_hash)
        store.publish_version(document_id=document_id, content_hash=content_hash)
        store.cleanup_old_versions(document_id=document_id, current_hash=content_hash)
        return published


def _check_identity(store: S3ArtifactStore, document_id: str, content_hash: str) -> None:
    """Raise DocumentIdConflictError when S3's current version of `document_id` is other content."""
    current = store.get_current_version(document_id)
    version = current.get("version") if isinstance(current, dict) else None
    if version and version != content_hash:
        raise DocumentIdConflictError(document_id)


def _marker(workspace: Path) -> Optional[str]:
    marker = workspace / _SOURCE_MARKER
    return marker.read_text().strip() if marker.is_file() else None
