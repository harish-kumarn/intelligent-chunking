"""
app.datasets.validation
=======================
`validate_csv` — stream an uploaded CSV through a `DatasetSchema` and write its canonical form.

    header   : exactly the schema's columns (any order; optional ones may be absent); unknown or
               duplicated names are rejected before a row is read
    rows     : per column — required, type (int / enum / id list), length, pattern; the key column must
               be unique; a cell that a spreadsheet would execute (`=…`, `@…`, `+cmd`, `-cmd`, control
               characters) is rejected — datasets are read by people and tools alike
    limits   : `max_rows` rows; the upload size is bounded before this runs (`stage_upload`)
    output   : canonical CSV at `canonical_path` — every schema column in schema order, cells trimmed,
               id lists normalised, UTF-8 without BOM, LF line endings — plus its sha256 and row count.
               Identical content therefore always yields the same bytes, hence the same dataset version

Memory is bounded by the key set (one string per row) and the chunk-reference index (document →
chunk id → first row), never by the file.

Failure policy: `DatasetValidationError` (HTTP 422) carries up to `max_issues` issues as
`{row, column, code}` — row numbers and codes only, never a cell value, so an error body or a log
line can not leak dataset content. Row 1 is the header.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.datasets.schema import CHUNK_ID_SEPARATOR, Column, DatasetSchema

__all__ = ["DatasetValidationError", "ValidatedDataset", "ValidationIssue", "validate_csv"]

DEFAULT_MAX_ISSUES = 50
_READ_BLOCK = 1024 * 1024
_FORMULA_PREFIXES = ("=", "@", "\t", "\r")
_SIGN_PREFIXES = ("+", "-")


@dataclass(frozen=True)
class ValidationIssue:
    row: int                 # 1-based line in the uploaded file; 1 = header, 0 = whole file
    column: Optional[str]
    code: str

    def to_dict(self) -> Dict[str, object]:
        return {"row": self.row, "column": self.column, "code": self.code}


class DatasetValidationError(Exception):
    code = "DATASET_INVALID"
    http_status = 422

    def __init__(self, issues: List[ValidationIssue], *, truncated: bool = False) -> None:
        self.issues = list(issues)
        self.truncated = truncated
        codes = sorted({issue.code for issue in self.issues})
        super().__init__(f"dataset rejected: {len(self.issues)}{'+' if truncated else ''} issue(s) ({', '.join(codes)})")


@dataclass(frozen=True)
class ValidatedDataset:
    schema_id: str
    row_count: int
    checksum_sha256: str                                     # of the canonical bytes
    source_sha256: str                                       # of the uploaded bytes
    canonical_path: Path
    document_ids: Tuple[str, ...]                            # distinct, first-seen order
    chunk_refs: Dict[str, Dict[str, int]] = field(default_factory=dict)  # document_id → chunk_id → first row


class _Issues:
    def __init__(self, limit: int) -> None:
        self.items: List[ValidationIssue] = []
        self.limit = limit
        self.truncated = False

    def add(self, row: int, column: Optional[str], code: str) -> None:
        if len(self.items) >= self.limit:
            self.truncated = True
            return
        self.items.append(ValidationIssue(row, column, code))

    @property
    def full(self) -> bool:
        return self.truncated

    def raise_if_any(self) -> None:
        if self.items:
            raise DatasetValidationError(self.items, truncated=self.truncated)

    def __bool__(self) -> bool:
        return bool(self.items)


def validate_csv(
    source: Path, schema: DatasetSchema, *, canonical_path: Path, max_rows: int, max_issues: int = DEFAULT_MAX_ISSUES,
) -> ValidatedDataset:
    """Validate `source` against `schema`; on success the canonical CSV is at `canonical_path`."""
    issues = _Issues(max_issues)
    source_sha256 = _file_sha256(source)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    keys: Set[str] = set()
    document_ids: List[str] = []
    seen_documents: Set[str] = set()
    chunk_refs: Dict[str, Dict[str, int]] = {}
    row_count = 0

    try:
        with open(source, "r", encoding="utf-8-sig", errors="strict", newline="") as handle, \
                open(canonical_path, "w", encoding="utf-8", newline="") as out:
            reader = csv.reader(handle, strict=True)  # bad quoting is an error, not a silently merged field
            writer = csv.writer(out, lineterminator="\n")
            header = next(reader, None)
            if header is None:
                issues.add(0, None, "EMPTY_FILE")
                positions = {}
            else:
                positions = _check_header([cell.strip() for cell in header], schema, issues)
            if not issues.items:
                writer.writerow(schema.names)
            for line_no, raw in enumerate(reader, start=2) if not issues.items else ():
                if not any(cell.strip() for cell in raw):
                    continue  # a blank line is not a row
                row_count += 1
                if row_count > max_rows:
                    issues.add(line_no, None, "TOO_MANY_ROWS")
                    break
                if len(raw) != len(header):
                    issues.add(line_no, None, "ROW_LENGTH_MISMATCH")
                    if issues.full:
                        break
                    continue
                values = {name: raw[index].strip() for name, index in positions.items()}
                canonical = _check_row(line_no, values, schema, issues)
                key = canonical.get(schema.key_column, "")
                if key:
                    if key in keys:
                        issues.add(line_no, schema.key_column, "DUPLICATE_CASE_ID")
                    keys.add(key)
                document_id = canonical.get("document_id", "")
                if document_id and document_id not in seen_documents:
                    seen_documents.add(document_id)
                    document_ids.append(document_id)
                if document_id and canonical.get("answer_chunk_id"):
                    refs = chunk_refs.setdefault(document_id, {})
                    for chunk_id in canonical["answer_chunk_id"].split(CHUNK_ID_SEPARATOR):
                        refs.setdefault(chunk_id, line_no)
                writer.writerow([canonical.get(name, "") for name in schema.names])
                if issues.full:
                    break
    except UnicodeDecodeError:
        issues.add(0, None, "INVALID_ENCODING")
    except csv.Error:
        issues.add(0, None, "MALFORMED_CSV")

    if row_count == 0 and not issues.items:
        issues.add(0, None, "NO_ROWS")
    if issues.items:
        canonical_path.unlink(missing_ok=True)  # never leave a half-written canonical file behind
        issues.raise_if_any()
    return ValidatedDataset(
        schema_id=schema.schema_id, row_count=row_count, checksum_sha256=_file_sha256(canonical_path),
        source_sha256=source_sha256, canonical_path=canonical_path, document_ids=tuple(document_ids), chunk_refs=chunk_refs,
    )


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def _check_header(header: List[str], schema: DatasetSchema, issues: _Issues) -> Dict[str, int]:
    positions: Dict[str, int] = {}
    for index, name in enumerate(header):
        if name not in schema.names:
            issues.add(1, name[:64] if name else None, "UNKNOWN_HEADER")
        elif name in positions:
            issues.add(1, name, "DUPLICATE_HEADER")
        else:
            positions[name] = index
    for name in schema.required:
        if name not in positions:
            issues.add(1, name, "MISSING_HEADER")
    return positions


def _check_row(line_no: int, values: Dict[str, str], schema: DatasetSchema, issues: _Issues) -> Dict[str, str]:
    canonical: Dict[str, str] = {}
    for column in schema.columns:
        value = values.get(column.name, "")
        if not value:
            if column.required:
                issues.add(line_no, column.name, "REQUIRED")
            canonical[column.name] = ""
            continue
        if _formula_like(value):
            issues.add(line_no, column.name, "FORMULA_LIKE")
            continue
        if len(value) > column.max_chars:
            issues.add(line_no, column.name, "TOO_LONG")
            continue
        normalised = _check_cell(column, value)
        if normalised is None:
            issues.add(line_no, column.name, "INVALID_FORMAT")
            continue
        canonical[column.name] = normalised
    return canonical


def _check_cell(column: Column, value: str) -> Optional[str]:
    """The canonical cell, or None when it breaks the column's rule."""
    if column.kind == "int":
        try:
            number = int(value)
        except ValueError:
            return None
        return str(number) if number >= 0 else None
    if column.kind == "enum":
        lowered = value.lower()
        return lowered if lowered in column.choices else None
    if column.kind == "id_list":
        ids = [part.strip() for part in value.split(CHUNK_ID_SEPARATOR)]
        ids = [part for part in ids if part]
        if not ids or (column.pattern and any(not column.pattern.match(part) for part in ids)):
            return None
        return CHUNK_ID_SEPARATOR.join(dict.fromkeys(ids))  # de-duplicated, order kept
    if column.pattern and not column.pattern.match(value):
        return None
    if any(ch in value for ch in ("\x00", "\r")):
        return None
    return value


def _formula_like(value: str) -> bool:
    """Cells a spreadsheet would evaluate: formulas, external references, commands behind a sign."""
    if value.startswith(_FORMULA_PREFIXES):
        return True
    if value.startswith(_SIGN_PREFIXES):
        rest = value[1:].lstrip()
        return not (rest[:1].isdigit() or rest[:1] == "." or rest == "")  # "-5 °C" is a value, "-cmd" is not
    return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()
