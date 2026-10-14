"""
app.ingestion.ocr.azure
=======================
Azure AI Document Intelligence: PDF ─► markdown with one `# Page N` heading per page,
plus extracted figures.

`prebuilt-layout` returns one markdown string for the whole document together with
per-page character spans; the spans are what lets us cut it back into the `# Page N`
sections the rest of the pipeline expects. Figures come back as bounding polygons
rather than bytes, so each one is cropped out of the source PDF with PyMuPDF and
written as `page_{n}_img_{k}.png` — the same file naming, the same in-markdown
`![file](file)` reference, and the same `pdf_images/` layout the Mistral backend
produces. Errors propagate — OCR is the first, mandatory stage; `run_ocr` retries them.
The analysis waits at most `ocr_request_timeout`.

Registered as the `azure` backend; `app.ingestion.ocr.run_ocr` dispatches here.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential

from app.config.defaults import ChunkerConfig
from app.ingestion.ocr.errors import OCRConfigurationError, OCRResultError
from app.logging.setup import get_logger
from app.utils.fs import atomic_write_text

__all__ = ["run_azure_ocr"]

_MARKDOWN_MODELS = ("prebuilt-layout",)  # the only models that accept outputContentFormat=markdown
_PAGE_BREAK = re.compile(r"(?m)^\s*<!--\s*PageBreak\s*-->\s*$")
_FIGURE_PADDING_PT = 4.0  # points of slack around a polygon so borders are not shaved off
_IMAGE_EXT = "png"


def run_azure_ocr(
    pdf_path: Path,
    markdown_out: Path,
    images_dir: Path,
    config: ChunkerConfig,
    logger: Optional[logging.LoggerAdapter] = None,
) -> Path:
    """analyze ─► split on page spans ─► crop figures + write markdown. Returns the markdown path."""
    log = logger or get_logger(module="OCR")
    if not config.azure_ocr_endpoint or not config.azure_ocr_key:
        raise OCRConfigurationError("AZURE_OCR_ENDPOINT and AZURE_OCR_KEY are required for Azure OCR.")

    pdf_path, markdown_out, images_dir = Path(pdf_path), Path(markdown_out), Path(images_dir)
    client = DocumentIntelligenceClient(
        endpoint=config.azure_ocr_endpoint.rstrip("/"),
        credential=AzureKeyCredential(config.azure_ocr_key),
    )

    log.info("OCR analyze: %s (%d bytes) model=%s", pdf_path.name, pdf_path.stat().st_size, config.azure_ocr_model)
    kwargs: Dict[str, Any] = {}
    if config.azure_ocr_model in _MARKDOWN_MODELS:
        kwargs["output_content_format"] = DocumentContentFormat.MARKDOWN
    poller = client.begin_analyze_document(
        config.azure_ocr_model,
        AnalyzeDocumentRequest(bytes_source=pdf_path.read_bytes()),
        **kwargs,
    )
    result = poller.result(timeout=config.ocr_request_timeout)
    if not poller.done():
        raise TimeoutError(f"Azure OCR did not finish {pdf_path.name} within {config.ocr_request_timeout:.0f}s")

    content = result.content or ""
    pages = list(result.pages or [])
    if not pages:
        raise OCRResultError(f"Azure OCR returned no pages for {pdf_path.name}.")

    figures_by_page = _figures_by_page(getattr(result, "figures", None) or [])
    images_dir.mkdir(parents=True, exist_ok=True)

    sections: List[str] = []
    image_count = 0
    with fitz.open(pdf_path) as document:
        for position, page in enumerate(pages):
            page_no = page.page_number or position + 1
            start, end = _page_bounds(pages, position, len(content))
            markdown = content[start:end]

            if config.azure_ocr_extract_figures and figures_by_page.get(page_no):
                markdown, written = _embed_figures(
                    markdown=markdown,
                    page_offset=start,
                    figures=figures_by_page[page_no],
                    page_no=page_no,
                    azure_page=page,
                    document=document,
                    images_dir=images_dir,
                    dpi=config.azure_ocr_figure_dpi,
                    log=log,
                )
                image_count += written

            markdown = _PAGE_BREAK.sub("", markdown)
            sections.append(f"\n\n# Page {page_no}\n\n{markdown.strip()}")

    atomic_write_text(markdown_out, "\n".join(sections))
    log.info("OCR complete: %d page(s), %d image(s) → %s", len(pages), image_count, markdown_out)
    return markdown_out


# --------------------------------------------------------------------------- #
# Page slicing
# --------------------------------------------------------------------------- #


def _page_bounds(pages: Sequence[Any], position: int, content_length: int) -> Tuple[int, int]:
    """
    Character range of one page inside `result.content`.

    A page's own spans can leave gaps (page headers and footers are spanned
    separately), so a page runs to wherever the next one starts; the last runs
    to the end of the document.
    """
    start = _span_start(pages[position])
    end = _span_start(pages[position + 1]) if position + 1 < len(pages) else content_length
    if start is None:
        return (0, content_length) if len(pages) == 1 else (0, end or content_length)
    return start, end if end is not None and end >= start else content_length


def _span_start(page: Any) -> Optional[int]:
    spans = getattr(page, "spans", None) or []
    return min(span.offset for span in spans) if spans else None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _figures_by_page(figures: Sequence[Any]) -> Dict[int, List[Any]]:
    """Group figures by the page their first bounding region sits on, in reading order."""
    grouped: Dict[int, List[Any]] = {}
    for figure in figures:
        regions = getattr(figure, "bounding_regions", None) or []
        if not regions:
            continue
        grouped.setdefault(regions[0].page_number, []).append(figure)
    return grouped


def _embed_figures(
    *,
    markdown: str,
    page_offset: int,
    figures: Sequence[Any],
    page_no: int,
    azure_page: Any,
    document: Any,
    images_dir: Path,
    dpi: int,
    log: logging.LoggerAdapter,
) -> Tuple[str, int]:
    """
    Crop every figure on this page and reference it from the markdown.

    Each crop is appended just after the figure's own span so the reference keeps
    its place in the reading order; insertions run back-to-front so that earlier
    offsets stay valid. Returns the rewritten markdown and the number of files written.
    """
    page_index = page_no - 1
    if page_index >= document.page_count:
        log.warning("OCR figures: page %d is outside the PDF (%d pages); skipped", page_no, document.page_count)
        return markdown, 0

    pdf_page = document[page_index]
    scale = _scale(azure_page, pdf_page)
    if scale is None:
        log.warning("OCR figures: page %d has no usable dimensions; figures skipped", page_no)
        return markdown, 0

    insertions: List[Tuple[int, str]] = []
    written = 0
    for index, figure in enumerate(figures, start=1):
        rect = _figure_rect(figure, page_no, scale, pdf_page.rect)
        if rect is None:
            continue
        filename = f"page_{page_no}_img_{index}.{_IMAGE_EXT}"
        pdf_page.get_pixmap(clip=rect, dpi=dpi).save(str(images_dir / filename))
        written += 1
        insertions.append((_figure_end(figure, page_offset, len(markdown)), f"\n\n![{filename}]({filename})\n\n"))

    for at, reference in sorted(insertions, key=lambda item: item[0], reverse=True):
        markdown = markdown[:at] + reference + markdown[at:]
    return markdown, written


def _scale(azure_page: Any, pdf_page: Any) -> Optional[Tuple[float, float]]:
    """
    Azure→PyMuPDF coordinate scale, derived from the two descriptions of the same page.

    Deriving it from the reported dimensions rather than assuming inches keeps the
    crop correct whatever unit the service reports and whatever rotation the page carries.
    """
    width, height = getattr(azure_page, "width", None), getattr(azure_page, "height", None)
    if not width or not height:
        return None
    return pdf_page.rect.width / width, pdf_page.rect.height / height


def _figure_rect(figure: Any, page_no: int, scale: Tuple[float, float], page_rect: Any) -> Optional[Any]:
    """Bounding box of every region this figure has on `page_no`, in PDF points, clipped to the page."""
    scale_x, scale_y = scale
    xs: List[float] = []
    ys: List[float] = []
    for region in getattr(figure, "bounding_regions", None) or []:
        if region.page_number != page_no:
            continue
        polygon = list(region.polygon or [])
        xs.extend(value * scale_x for value in polygon[0::2])
        ys.extend(value * scale_y for value in polygon[1::2])
    if not xs or not ys:
        return None

    rect = fitz.Rect(
        min(xs) - _FIGURE_PADDING_PT,
        min(ys) - _FIGURE_PADDING_PT,
        max(xs) + _FIGURE_PADDING_PT,
        max(ys) + _FIGURE_PADDING_PT,
    ) & page_rect
    return rect if not rect.is_empty else None


def _figure_end(figure: Any, page_offset: int, page_length: int) -> int:
    """Where the figure's markdown ends, as an offset into the page slice."""
    spans = getattr(figure, "spans", None) or []
    if not spans:
        return page_length
    absolute = max(span.offset + (span.length or 0) for span in spans)
    return max(0, min(absolute - page_offset, page_length))
