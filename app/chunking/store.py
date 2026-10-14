"""
app.chunking.store
==================
ChunkStore — JSON file of per-page chunk records.

    file shape : {"chunk1": ChunkRecord, "chunk2": ChunkRecord, ...} in document order
                 (page, then the order the chunker emitted them)
    side file  : <name>.empty_pages.json — pages the chunker finished with zero chunks (blank or
                 boilerplate-only), so a resume does not pay for them again

Pages may be appended in any order (stage 3 chunks pages in parallel); `load` and the file
are always in document order, so the merger reads pages in sequence. A crashed run resumes
from `completed_pages()`. Appending a page replaces whatever the store held for it, so chunking
a page twice never duplicates its records.

Writes are atomic (temp file + rename), so a crash never leaves a truncated file behind. With
`flush_interval` > 0 appends are written at most that often (bounding the O(pages²) rewrite of a
1,000-page store) and `flush()` must be called when the run ends — pages appended after the last
write are simply chunked again on resume. The records file is written before the side file, so
the side file can only lag (an empty page is re-chunked), never claim a page whose chunks are lost.
Later stages may add keys to existing records with `annotate()` (e.g. classification); the
chunker's keys are never changed.

Thread-safe within one process: one lock guards the in-memory state and the files.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from app.chunking.schema import ChunkingResponse, ChunkRecord
from app.logging.setup import get_logger
from app.utils.fs import atomic_write_json

__all__ = ["ChunkStore"]

_KEY_PATTERN = re.compile(r"^chunk(\d+)$")
_EMPTY_PAGES_SUFFIX = ".empty_pages.json"


class ChunkStore:
    """Ordered, resumable store of raw chunk records for one document."""

    def __init__(self, path: Path, logger: Optional[logging.LoggerAdapter] = None, *, flush_interval: float = 0.0) -> None:
        self.path = Path(path)
        self.empty_pages_path = self.empty_pages_path_for(self.path)
        self._log = logger or get_logger(module="CHUNK_STORE")
        self._flush_interval = flush_interval
        self._lock = threading.Lock()
        self._records: Optional[List[ChunkRecord]] = None  # loaded lazily, then authoritative
        self._empty_pages: Optional[Set[int]] = None
        self._dirty = False
        self._last_write = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def empty_pages_path_for(path: Path) -> Path:
        """The side file recording pages chunked with zero chunks (`target_chunks.empty_pages.json`)."""
        return Path(path).with_name(f"{Path(path).stem}{_EMPTY_PAGES_SUFFIX}")

    def load(self) -> List[ChunkRecord]:
        """All records in document order; empty when the file is absent or blank."""
        with self._lock:
            return list(self._loaded())

    def completed_pages(self) -> Set[int]:
        """Pages with stored chunks, plus pages the chunker finished with none."""
        with self._lock:
            return {int(record["page_no"]) for record in self._loaded() if "page_no" in record} | self._loaded_empty()

    def append_page(self, response: ChunkingResponse, page_no: int) -> int:
        """Persist one page's chunks (any page order, replacing a previous result); returns how many were added."""
        records = response.to_records(page_no)
        if not records:
            self._log.info("Page %d produced 0 chunks (blank or boilerplate-only).", page_no)
        with self._lock:
            others = [record for record in self._loaded() if int(record.get("page_no") or 0) != page_no]
            self._records = _document_order(others + records)
            empty = self._loaded_empty()
            if records:
                empty.discard(page_no)
            else:
                empty.add(page_no)
            self._dirty = True
            if time.monotonic() - self._last_write >= self._flush_interval:
                self._write()
        self._log.debug("Stored %d chunk(s) for page %d → %s", len(records), page_no, self.path)
        return len(records)

    def annotate(self, fields_by_chunk_id: Mapping[str, Mapping[str, Any]]) -> int:
        """Merge extra fields into the records with these chunk ids; returns how many records changed."""
        with self._lock:
            updated = 0
            for record in self._loaded():
                fields = fields_by_chunk_id.get(record.get("chunk_id"))
                if fields:
                    record.update(fields)
                    updated += 1
            if updated:
                self._dirty = True
                self._write()
            return updated

    def flush(self) -> None:
        """Write pending appends now (end of stage 3, success or failure)."""
        with self._lock:
            if self._dirty:
                self._write()

    # ------------------------------------------------------------------ #
    # Internals (lock held)
    # ------------------------------------------------------------------ #

    def _loaded(self) -> List[ChunkRecord]:
        if self._records is None:
            if not self.path.exists() or self.path.stat().st_size == 0:
                self._records = []
            else:
                data: Dict[str, ChunkRecord] = json.loads(self.path.read_text(encoding="utf-8"))
                self._records = _document_order([data[key] for key in sorted(data, key=_key_index)])
        return self._records

    def _loaded_empty(self) -> Set[int]:
        if self._empty_pages is None:
            self._empty_pages = set()
            if self.empty_pages_path.exists():
                try:
                    self._empty_pages = {int(page) for page in json.loads(self.empty_pages_path.read_text(encoding="utf-8"))}
                except (ValueError, TypeError) as exc:  # an unreadable side file only costs a re-chunk of empty pages
                    self._log.warning("Ignoring unreadable %s: %s", self.empty_pages_path, exc)
        return self._empty_pages

    def _write(self, records: Optional[List[ChunkRecord]] = None) -> None:
        if records is not None:  # replace the whole store (tests, fixtures)
            self._records = _document_order(list(records))
            self._empty_pages = set()
        atomic_write_json(self.path, {f"chunk{i}": record for i, record in enumerate(self._loaded(), start=1)})
        empty = self._loaded_empty()
        if empty or self.empty_pages_path.exists():
            atomic_write_json(self.empty_pages_path, sorted(empty), indent=None)
        self._dirty = False
        self._last_write = time.monotonic()


def _document_order(records: List[ChunkRecord]) -> List[ChunkRecord]:
    """Stable sort by page: within a page the chunker's order is kept."""
    return sorted(records, key=lambda record: int(record.get("page_no") or 0))


def _key_index(key: str) -> int:
    match = _KEY_PATTERN.match(key)
    return int(match.group(1)) if match else 0
