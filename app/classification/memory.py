"""
app.classification.memory
=========================
CategoryMemory — the classification state of one document, persisted as classification_memory.json.

    general  : the configured taxonomy (name → definition); fixed for the run
    created  : main categories the model introduced, kept apart from `general`; every later
               chunk is shown them, so the model reuses a category instead of re-inventing it
    chunks   : chunk_id → validated assignment, keyed by a content fingerprint
    failures : chunk_id → last error, for chunks that failed every attempt (retried next run)

Rules enforced here, not only in the prompt (`resolve`):
    main_category  — exactly one; general, an existing created one, or (when allowed) a new one
    sub_categories — general categories only, never the main category, deduplicated, in taxonomy order

`resolve` is pure, so a rejected answer can be retried without side effects; `commit`
is the only mutation that registers a created category. A memory written under a different
set of general categories is discarded on load — its assignments no longer mean the same.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app.classification.schema import (
    F_MAIN_CATEGORY,
    F_SUB_CATEGORIES,
    ChunkClassification,
    ClassificationResponse,
    CreatedCategory,
    MemorySnapshot,
)
from app.utils.fs import atomic_write_json
from app.utils.naming import normalize_category_name

__all__ = ["CategoryMemory", "InvalidClassificationError", "Resolution", "chunk_fingerprint"]


class InvalidClassificationError(ValueError):
    """The model's answer breaks a classification rule; the classifier retries it."""


@dataclass(frozen=True)
class Resolution:
    """A validated answer for one chunk, not yet committed to the memory."""

    classification: ChunkClassification
    new_category_description: str = ""             # meaningful only for a new created category
    dropped_sub_categories: Tuple[str, ...] = ()    # names the rules removed, for logging


def chunk_fingerprint(chunk: Mapping[str, Any]) -> str:
    """Stable hash of everything the classifier reads from a MergedChunk."""
    payload = json.dumps(
        [chunk.get("content") or "", chunk.get("chunk_description") or "", list(chunk.get("concept_vocabulary") or [])],
        ensure_ascii=False,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


class CategoryMemory:
    """General + created taxonomy and every chunk's assignment for one document."""

    def __init__(self, general: Mapping[str, str], *, allow_new: bool, snapshot: Optional[MemorySnapshot] = None) -> None:
        self._allow_new = allow_new
        self._state = snapshot or MemorySnapshot()
        self._state.general_categories = dict(general)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: Path, *, general: Mapping[str, str], allow_new: bool, logger: logging.LoggerAdapter) -> "CategoryMemory":
        """The memory at `path`, or an empty one when absent, unreadable, or built on other general categories."""
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return cls(general, allow_new=allow_new)
        try:
            snapshot = MemorySnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            logger.warning("Ignoring unreadable classification memory %s: %s", path, exc)
            return cls(general, allow_new=allow_new)
        if set(snapshot.general_categories) != set(general):
            logger.info(
                "General categories changed (%s → %s): discarding the classification memory",
                sorted(snapshot.general_categories), sorted(general),
            )
            return cls(general, allow_new=allow_new)
        return cls(general, allow_new=allow_new, snapshot=snapshot)

    def save(self, path: Path) -> None:
        atomic_write_json(path, self._state.model_dump())

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    @property
    def general(self) -> Dict[str, str]:
        return dict(self._state.general_categories)

    @property
    def created(self) -> Dict[str, CreatedCategory]:
        return dict(self._state.created_categories)

    @property
    def failures(self) -> Dict[str, str]:
        return dict(self._state.failures)

    def lookup(self, chunk_id: str, fingerprint: str) -> Optional[ChunkClassification]:
        """The stored assignment, unless the chunk changed since it was classified."""
        entry = self._state.chunks.get(chunk_id)
        return entry if entry is not None and entry.fingerprint == fingerprint else None

    def chunk_fields(self, chunk_id: str, fingerprint: str) -> Dict[str, Any]:
        """The keys written onto a chunk: `main_category` is None when it could not be classified."""
        entry = self.lookup(chunk_id, fingerprint)
        return {
            F_MAIN_CATEGORY: entry.main_category if entry else None,
            F_SUB_CATEGORIES: list(entry.sub_categories) if entry else [],
        }

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def retain(self, fingerprints: Mapping[str, str]) -> None:
        """
        Keep only assignments for these chunks with unchanged content, then drop created
        categories no remaining chunk uses — a re-chunked document must not be steered by
        categories invented for text that no longer exists.
        """
        state = self._state
        state.chunks = {cid: e for cid, e in state.chunks.items() if fingerprints.get(cid) == e.fingerprint}
        state.failures = {cid: err for cid, err in state.failures.items() if cid in fingerprints}
        used = {e.main_category for e in state.chunks.values() if e.main_category_source == "created"}
        state.created_categories = {name: c for name, c in state.created_categories.items() if name in used}

    def resolve(self, response: ClassificationResponse, fingerprint: str) -> Resolution:
        """Validate one answer against the rules. Pure; raises InvalidClassificationError."""
        try:
            main = normalize_category_name(response.main_category)
        except ValueError as exc:
            raise InvalidClassificationError(f"unusable main_category {response.main_category!r}") from exc

        general = self._state.general_categories
        description = ""
        if main in general:
            source = "general"
        elif main in self._state.created_categories:
            source = "created"
        elif self._allow_new:
            source = "created"
            description = " ".join(response.new_category_description.split())
        else:
            raise InvalidClassificationError(
                f"main_category {main!r} is not one of {list(general)} and creating categories is disabled"
            )

        subs, dropped = _sub_categories(response.sub_categories, general, main)
        return Resolution(
            classification=ChunkClassification(
                fingerprint=fingerprint,
                main_category=main,
                main_category_source=source,
                sub_categories=subs,
                reasoning=" ".join(response.reasoning.split()),
            ),
            new_category_description=description,
            dropped_sub_categories=tuple(dropped),
        )

    def commit(self, chunk_id: str, resolution: Resolution) -> bool:
        """Store the assignment; returns True when it introduced a new created category."""
        entry = resolution.classification
        self._state.chunks[chunk_id] = entry
        self._state.failures.pop(chunk_id, None)
        if entry.main_category_source != "created" or entry.main_category in self._state.created_categories:
            return False
        self._state.created_categories[entry.main_category] = CreatedCategory(
            description=resolution.new_category_description, first_chunk_id=chunk_id,
        )
        return True

    def record_failure(self, chunk_id: str, error: str) -> None:
        self._state.chunks.pop(chunk_id, None)
        self._state.failures[chunk_id] = error


def _sub_categories(raw: List[str], general: Mapping[str, str], main: str) -> Tuple[List[str], List[str]]:
    """General categories named in `raw`, minus the main one, in taxonomy order; plus what was dropped."""
    wanted, dropped = set(), []
    for name in raw:
        try:
            key = normalize_category_name(name)
        except ValueError:
            dropped.append(str(name))
            continue
        if key in general and key != main:
            wanted.add(key)
        elif key != main:
            dropped.append(key)  # a created / new / unknown category is never a sub category
    return [name for name in general if name in wanted], dropped
