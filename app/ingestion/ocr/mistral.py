"""
app.ingestion.ocr.mistral
=========================
Mistral OCR: PDF ─► markdown with one `# Page N` heading per page, plus extracted images.

Image references inside the markdown are rewritten from Mistral's opaque ids
to the on-disk file names (`page_{n}_img_{k}.{ext}`) so later stages can
resolve them. Errors propagate — OCR is the first, mandatory stage; `run_ocr` retries them.
Every call has `ocr_request_timeout`; the uploaded PDF is deleted from Mistral afterwards
(best effort), so no copy of the document outlives the run there.

Registered as the `mistral` backend; `app.ingestion.ocr.run_ocr` dispatches here.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import List, Optional

from mistralai import DocumentURLChunk, Mistral

from app.config.defaults import ChunkerConfig
from app.ingestion.ocr.errors import OCRConfigurationError, OCRResultError
from app.logging.setup import get_logger
from app.utils.fs import atomic_write_text

__all__ = ["run_mistral_ocr"]

_SIGNED_URL_EXPIRY_HOURS = 5
_DEFAULT_IMAGE_EXT = "bin"
_MS_PER_SECOND = 1000


def run_mistral_ocr(
    pdf_path: Path,
    markdown_out: Path,
    images_dir: Path,
    config: ChunkerConfig,
    logger: Optional[logging.LoggerAdapter] = None,
) -> Path:
    """upload ─► signed URL ─► OCR ─► write images + markdown. Returns the markdown path."""
    log = logger or get_logger(module="OCR")
    if not config.mistral_api_key:
        raise OCRConfigurationError("MISTRAL_API_KEY is required for OCR.")

    pdf_path, markdown_out, images_dir = Path(pdf_path), Path(markdown_out), Path(images_dir)
    client = Mistral(api_key=config.mistral_api_key, timeout_ms=int(config.ocr_request_timeout * _MS_PER_SECOND))

    log.info("OCR upload: %s (%d bytes)", pdf_path.name, pdf_path.stat().st_size)
    uploaded = client.files.upload(file={"file_name": pdf_path.name, "content": pdf_path.read_bytes()}, purpose="ocr")
    try:
        signed = client.files.get_signed_url(file_id=uploaded.id, expiry=_SIGNED_URL_EXPIRY_HOURS)
        result = client.ocr.process(
            document=DocumentURLChunk(document_url=signed.url),
            model=config.mistral_ocr_model,
            include_image_base64=True,
        )
    finally:
        _delete_upload(client, uploaded.id, log)
    if not result.pages:
        raise OCRResultError(f"Mistral OCR returned no pages for {pdf_path.name}.")

    images_dir.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []
    image_count = 0
    for page in result.pages:
        page_no = page.index + 1
        markdown = page.markdown or ""
        for k, image in enumerate(page.images or [], start=1):
            if not image.image_base64:
                continue
            filename = _write_image(images_dir, page_no, k, image.image_base64)
            markdown = re.sub(rf"!\[[^\]]*\]\({re.escape(str(image.id))}\)", f"![{filename}]({filename})", markdown)
            image_count += 1
        sections.append(f"\n\n# Page {page_no}\n\n{markdown.strip()}")

    atomic_write_text(markdown_out, "\n".join(sections))
    log.info("OCR complete: %d page(s), %d image(s) → %s", len(result.pages), image_count, markdown_out)
    return markdown_out


def _delete_upload(client: Mistral, file_id: str, log: logging.LoggerAdapter) -> None:
    """Secondary — a leftover file costs storage on Mistral's side, never the document."""
    try:
        client.files.delete(file_id=file_id)
    except Exception as exc:
        log.warning("Could not delete the uploaded PDF from Mistral (file_id=%s): %s", file_id, exc)


def _write_image(images_dir: Path, page_no: int, index: int, data_uri: str) -> str:
    """Decode a `data:<mime>;base64,<payload>` (or bare base64) string to disk; returns the file name."""
    extension = _DEFAULT_IMAGE_EXT
    payload = data_uri
    if data_uri.startswith("data:"):
        header, _, payload = data_uri.partition(",")
        extension = header.split(";")[0].split("/")[-1] or _DEFAULT_IMAGE_EXT
    filename = f"page_{page_no}_img_{index}.{extension}"
    (images_dir / filename).write_bytes(base64.b64decode(payload))
    return filename
