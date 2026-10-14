"""
app.utils.pdf_utils
===================
PDF page rendering (PyMuPDF) and neighbour-page context trimming.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF

__all__ = ["render_pdf_pages", "split_content"]

_RENDER_DPI = 200  # sharp enough for OCR verification, small enough for one vision call per page
_PARAGRAPH_SEP = "\n\n"


def render_pdf_pages(pdf_path: Path, output_dir: Path, *, only_missing: bool = False) -> List[Path]:
    """
    Render every page to `output_dir/page_{n}.jpeg` (1-based) and return the paths in order.
    `only_missing` keeps existing non-empty images (a resumed run renders only what is left); each
    image is written to a temp name and renamed, so an interrupted render never leaves a half file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            out = output_dir / f"page_{index}.jpeg"
            paths.append(out)
            if only_missing and out.exists() and out.stat().st_size > 0:
                continue
            temp = output_dir / f".page_{index}.tmp.jpeg"
            page.get_pixmap(dpi=_RENDER_DPI).save(str(temp))
            temp.replace(out)
    return paths


def split_content(content: str, part: str) -> str:
    """
    Return the first ("start") or second ("end") half of `content`, split on a
    paragraph boundary when one exists. Gives the chunker boundary context
    from a neighbouring page at roughly half the token cost.
    """
    if part not in ("start", "end"):
        raise ValueError(f"part must be 'start' or 'end', got {part!r}")
    if not content:
        return ""
    paragraphs = [p for p in content.split(_PARAGRAPH_SEP) if p.strip()]
    if len(paragraphs) > 1:
        mid = len(paragraphs) // 2
        return _PARAGRAPH_SEP.join(paragraphs[:mid] if part == "start" else paragraphs[mid:])
    mid = len(content) // 2
    return content[:mid] if part == "start" else content[mid:]
