"""
app.ingestion.ocr.windows
=========================
`run_windowed` — OCR a long PDF as page windows and stitch the results into one document.

    plan   : page windows of `ocr_window_pages`; a window whose sub-PDF exceeds `ocr_window_max_bytes`
             is split in half until it fits (Mistral rejects > 50 MB / > 1,000 pages per request)
    run    : the selected backend over each window, `ocr_concurrency` at a time
    stitch : `# Page n` headings and `page_{n}_img_{k}` image names shifted by the window's offset,
             images moved into the document's image folder

Layout: `<markdown dir>/ocr_windows/p{first:05d}-{last:05d}/` holds `window.pdf`, `window.md`
and `images/`; a window whose `window.md` exists is reused, so a failed run redoes only the
windows that did not finish. Once the stitched markdown is written (atomically) the folder is
removed: the markdown is then the resume point, and the window PDFs never reach S3.

Failure policy: primary — the backend passed in is already retried (`run_ocr`); a window that
still fails propagates after the others finish (their results stay on disk for the retry). Pure
page arithmetic is in `plan_windows` / `shift_pages`.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import fitz  # PyMuPDF

from app.config.defaults import ChunkerConfig
from app.utils.fs import atomic_write_text

__all__ = ["OcrWindow", "plan_windows", "run_windowed", "shift_pages"]

Backend = Callable[[Path, Path, Path, ChunkerConfig, Optional[logging.LoggerAdapter]], Path]

_WINDOWS_DIR = "ocr_windows"
_PAGE_HEADING = re.compile(r"(?m)^(#\s*Page\s+)(\d+)(\s*)$")
_IMAGE_NAME = re.compile(r"page_(\d+)_img_(\d+)")


@dataclass(frozen=True)
class OcrWindow:
    first: int  # 1-based, inclusive
    last: int

    @property
    def name(self) -> str:
        return f"p{self.first:05d}-{self.last:05d}"


def page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as document:
        return document.page_count


def plan_windows(total_pages: int, window_pages: int) -> List[OcrWindow]:
    return [OcrWindow(first, min(first + window_pages - 1, total_pages)) for first in range(1, total_pages + 1, window_pages)]


def shift_pages(markdown: str, offset: int) -> str:
    """Window-local page numbers in headings and image names → document page numbers."""
    markdown = _PAGE_HEADING.sub(lambda m: f"{m.group(1)}{int(m.group(2)) + offset}{m.group(3)}", markdown)
    return _IMAGE_NAME.sub(lambda m: f"page_{int(m.group(1)) + offset}_img_{m.group(2)}", markdown)


def run_windowed(
    backend: Backend,
    pdf_path: Path,
    markdown_out: Path,
    images_dir: Path,
    config: ChunkerConfig,
    log: logging.LoggerAdapter,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """window ─► OCR (parallel, resumable) ─► stitch. Returns the markdown path."""
    root = markdown_out.parent / _WINDOWS_DIR
    windows = _fit_to_size(pdf_path, plan_windows(page_count(pdf_path), config.ocr_window_pages), root, config, log)
    done = sum(1 for window in windows if (root / window.name / "window.md").exists())
    log.info("OCR in %d window(s) of ≤ %d pages, %d already done", len(windows), config.ocr_window_pages, done)
    if on_progress is not None:
        on_progress(done, len(windows))

    lock = threading.Lock()

    def ocr(window: OcrWindow) -> None:
        nonlocal done
        folder = root / window.name
        if (folder / "window.md").exists():
            return
        partial = folder / "window.md.partial"
        backend(folder / "window.pdf", partial, folder / "images", config, log)
        partial.replace(folder / "window.md")  # only a finished window is ever reused
        with lock:
            done += 1
            finished = done
        log.info("OCR window %s done (%d/%d)", window.name, finished, len(windows))
        if on_progress is not None:
            on_progress(finished, len(windows))

    with ThreadPoolExecutor(max_workers=config.ocr_concurrency, thread_name_prefix="ocr") as pool:
        futures = [pool.submit(ocr, window) for window in windows]
    for future in futures:
        future.result()  # first failure propagates; finished windows stay for the retry

    images_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    for window in windows:
        folder = root / window.name
        offset = window.first - 1
        sections.append(shift_pages((folder / "window.md").read_text(encoding="utf-8"), offset))
        for image in sorted((folder / "images").glob("*")) if (folder / "images").is_dir() else []:
            shutil.copy2(image, images_dir / shift_pages(image.name, offset))
    atomic_write_text(markdown_out, "\n".join(sections))
    shutil.rmtree(root, ignore_errors=True)  # the stitched markdown is the resume point from here on
    log.info("OCR stitched: %d window(s) → %s", len(windows), markdown_out)
    return markdown_out


def _fit_to_size(pdf_path: Path, windows: List[OcrWindow], root: Path, config: ChunkerConfig, log: logging.LoggerAdapter) -> List[OcrWindow]:
    """Write each window's sub-PDF; split windows over the byte cap in half until they fit (or are one page)."""
    fitted: List[OcrWindow] = []
    pending = list(windows)
    with fitz.open(pdf_path) as document:
        while pending:
            window = pending.pop(0)
            path = root / window.name / "window.pdf"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with fitz.open() as part:
                    part.insert_pdf(document, from_page=window.first - 1, to_page=window.last - 1)
                    part.save(str(path), garbage=3, deflate=True)
            if path.stat().st_size > config.ocr_window_max_bytes and window.last > window.first:
                middle = (window.first + window.last) // 2
                log.info("OCR window %s is %d bytes; splitting", window.name, path.stat().st_size)
                shutil.rmtree(path.parent)
                pending[:0] = [OcrWindow(window.first, middle), OcrWindow(middle + 1, window.last)]
                continue
            fitted.append(window)
    return sorted(fitted, key=lambda w: w.first)
