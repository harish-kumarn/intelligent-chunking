"""
app.ingestion.document_parser
=============================
Splits OCR markdown into per-page text keyed `page_{n}`.

With the PDF's page count the page map is made complete: every page 1..N is present (a page the
OCR engine skipped gets empty text — stage 3 still chunks it from its page image) and headings
beyond the PDF are dropped, both with a warning. Without it, the map is what the markdown holds.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional

from app.app_logging.setup import get_logger
from app.utils.fs import atomic_write_json

__all__ = ["complete_pages", "md_to_page_json", "split_md_pages"]

_PAGE_HEADING = re.compile(r"(?im)^\s*#{1,3}\s*Page\s+(\d+)\s*$")
_HORIZONTAL_RULE = re.compile(r"(?m)^\s*-{3,}\s*$")
_LISTED_PAGES = 20  # page numbers quoted in a warning


def split_md_pages(markdown: str) -> Dict[str, str]:
    """
    Map `page_{n}` → body text for every `# Page n` heading; empty when there are none.

    OCR emits the page headings in increasing order, so a heading whose number does not exceed the last
    accepted one (e.g. a "## Page 3" line inside a table of contents) is document text, not a page break —
    it never replaces the text of a page already read.
    """
    headings = []
    for match in _PAGE_HEADING.finditer(markdown):
        if not headings or int(match.group(1)) > int(headings[-1].group(1)):
            headings.append(match)
    pages: Dict[str, str] = {}
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        body = _HORIZONTAL_RULE.sub("", markdown[heading.end():end]).strip()
        pages[f"page_{heading.group(1)}"] = body
    return pages


def complete_pages(pages: Dict[str, str], page_count: int, log: logging.LoggerAdapter) -> Dict[str, str]:
    """Exactly `page_1`..`page_{page_count}`, in order: OCR gaps filled with "", extra headings dropped."""
    numbers = {int(key.split("_", 1)[1]) for key in pages}
    missing = [n for n in range(1, page_count + 1) if n not in numbers]
    extra = sorted(n for n in numbers if not 1 <= n <= page_count)
    if missing:
        log.warning("OCR returned no text for %d of %d page(s): %s — chunked from their page images only",
                    len(missing), page_count, missing[:_LISTED_PAGES])
    if extra:
        log.warning("OCR returned %d page heading(s) beyond the PDF's %d pages, dropped: %s", len(extra), page_count, extra[:_LISTED_PAGES])
    return {f"page_{n}": pages.get(f"page_{n}", "") for n in range(1, page_count + 1)}


def md_to_page_json(
    markdown_path: Path,
    output_path: Path,
    logger: Optional[logging.LoggerAdapter] = None,
    *,
    page_count: Optional[int] = None,
) -> Dict[str, str]:
    """Read markdown, split, [complete to `page_count`], persist as JSON, return the page map. Raises ValueError when no pages are found."""
    log = logger or get_logger(module="PARSER")
    pages = split_md_pages(Path(markdown_path).read_text(encoding="utf-8"))
    if not pages:
        raise ValueError(f"No page markers found in {markdown_path}; expected headings like '# Page 1'.")
    if page_count is not None:
        pages = complete_pages(pages, page_count, log)
    atomic_write_json(output_path, pages)
    log.info("Parsed %d page(s) → %s", len(pages), output_path)
    return pages
