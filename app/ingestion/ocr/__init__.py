"""
app.ingestion.ocr
=================
`run_ocr` — stage 1 of the pipeline, behind an engine switch.

    azure   → Azure AI Document Intelligence   (app.ingestion.ocr.azure)
    mistral → Mistral OCR                      (app.ingestion.ocr.mistral)

Which one runs comes from `ChunkerConfig.ocr_technique`, whose default is the
`ocr.technique` key in `config.yaml` and which `OCR_TECHNIQUE` can override.
The callable's signature, its artifacts (`<doc>.md` with one `# Page N` heading
per page, `pdf_images/page_{n}_img_{k}.{ext}`, markdown referencing those files
by name) and its failure policy are identical for both backends, so nothing
downstream — the pipeline included — knows which one ran.

A PDF longer than `ocr_window_pages` (or larger than `ocr_window_max_bytes`) is OCR'd in page
windows and stitched back (app.ingestion.ocr.windows); the artifacts are the same.

Failure policy: primary. Every backend call (the whole PDF, or one window) is retried up to
`ocr_max_retries` times with exponential back-off, except errors retrying cannot fix: missing
configuration (`OCRConfigurationError`), an unusable result (`OCRResultError`) and HTTP 4xx
answers other than 408/429. The last error propagates.

Backends are imported lazily: only the selected engine's SDK has to be installed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF

from app.config.defaults import ChunkerConfig
from app.ingestion.ocr.errors import OCRConfigurationError, OCRResultError
from app.logging.setup import get_logger

__all__ = ["OCRConfigurationError", "OCRResultError", "is_retryable_ocr_error", "run_ocr", "with_retries"]

OCRBackend = Callable[[Path, Path, Path, ChunkerConfig, Optional[logging.LoggerAdapter]], Path]

_RETRYABLE_CLIENT_STATUSES = frozenset({408, 429})
_MAX_BACKOFF_SECONDS = 300.0


def _resolve(technique: str) -> OCRBackend:
    """Import the selected backend, turning a missing SDK into an actionable message."""
    try:
        if technique == "azure":
            from app.ingestion.ocr.azure import run_azure_ocr

            return run_azure_ocr
        if technique == "mistral":
            from app.ingestion.ocr.mistral import run_mistral_ocr

            return run_mistral_ocr
    except ImportError as exc:  # the other engine's SDK may legitimately be absent
        raise OCRConfigurationError(
            f"OCR technique '{technique}' is selected but its dependencies are not installed: {exc}"
        ) from exc
    raise ValueError(f"Unknown OCR technique {technique!r}; expected 'azure' or 'mistral'.")


def run_ocr(
    pdf_path: Path,
    markdown_out: Path,
    images_dir: Path,
    config: ChunkerConfig,
    logger: Optional[logging.LoggerAdapter] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Run the configured OCR engine over `pdf_path` (in windows when long). Returns the markdown path."""
    log = logger or get_logger(module="OCR")
    technique = config.ocr_technique
    backend = with_retries(_resolve(technique), config, log)
    pages = _page_count(Path(pdf_path))
    windowed = pages is not None and (
        pages > config.ocr_window_pages or Path(pdf_path).stat().st_size > config.ocr_window_max_bytes
    )
    log.info("OCR technique: %s pages=%s windowed=%s", technique, pages if pages is not None else "unknown", windowed)
    if windowed:
        from app.ingestion.ocr.windows import run_windowed

        return run_windowed(backend, Path(pdf_path), Path(markdown_out), Path(images_dir), config, log, on_progress)
    result = backend(pdf_path, markdown_out, images_dir, config, log)
    if on_progress is not None:
        on_progress(1, 1)
    return result


def with_retries(backend: OCRBackend, config: ChunkerConfig, log: logging.LoggerAdapter) -> OCRBackend:
    """`backend` retried on transient errors (see the module's failure policy)."""

    def call(pdf_path: Path, markdown_out: Path, images_dir: Path, cfg: ChunkerConfig, logger: Optional[logging.LoggerAdapter]) -> Path:
        attempts = config.ocr_max_retries
        for attempt in range(1, attempts + 1):
            try:
                return backend(pdf_path, markdown_out, images_dir, cfg, logger)
            except Exception as exc:
                if attempt == attempts or not is_retryable_ocr_error(exc):
                    raise
                wait = min(config.ocr_retry_delay * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
                log.warning("OCR of %s failed (attempt %d/%d): %s — retrying in %.0fs", Path(pdf_path).name, attempt, attempts, exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")  # the loop returns or raises

    return call


def is_retryable_ocr_error(exc: BaseException) -> bool:
    if isinstance(exc, (OCRConfigurationError, OCRResultError, ValueError)):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in _RETRYABLE_CLIENT_STATUSES:
        return False
    return True


def _page_count(pdf_path: Path) -> Optional[int]:
    """None when PyMuPDF cannot open the file — the backend then reports the problem in its own terms."""
    try:
        with fitz.open(pdf_path) as document:
            return document.page_count
    except Exception:  # missing / damaged file: not ours to diagnose here
        return None
