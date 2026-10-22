"""
app.classification.stage
========================
ClassificationStage — pipeline stage 5: label every merged chunk with a main category and
sub categories, then write those labels onto the chunk artifacts.

    input   : merged_chunks.json (the final retrieval units, document order)
    state   : classification_memory.json — general + created categories, per-chunk assignments
    outputs : merged_chunks.json   + `main_category`, `sub_categories` on every chunk
              target_chunks.json   + the same keys on every source record of that chunk

Main path
    load memory ─► retain unchanged chunks ─► pending chunks in order, in batches
        (`classification_batch_size` chunks / `classification_batch_max_chars`, when the classifier
        supports `classify_batch`; else one at a time):
        classify (Gemini, retried) ─► for each chunk in order: resolve ─► commit ─► save memory per batch
        a chunk the batch answer misses or that breaks a rule is classified alone (`classify`)
    ─► annotate merged_chunks.json ─► annotate target_chunks.json

Chunks are classified in document order: each call is shown the categories created so far, and
answers within a batch are committed in order, so a new topic is created once and reused afterwards.
`on_progress(done, total)` is called after every chunk is settled. The memory is saved
after every chunk and keyed by content fingerprint, so a re-run (merge always re-runs)
costs nothing for unchanged chunks and resumes after a crash.

Failure policy: secondary path. A chunk that fails every attempt is recorded under
`failures`, written with `main_category: null` and retried on the next run; the stage and
the ingestion still complete. A missing API key fails at construction (configuration error).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from app.chunking.store import ChunkStore
from app.classification.classifier import ChunkClassifier, prompt_chars
from app.classification.memory import CategoryMemory, InvalidClassificationError, Resolution, chunk_fingerprint
from app.classification.schema import F_MAIN_CATEGORY, F_SUB_CATEGORIES
from app.config.defaults import ChunkerConfig
from app.app_logging.setup import get_logger
from app.postprocessing.merger import MergedChunk, load_merged_chunks, save_merged_chunks

__all__ = ["ClassificationResult", "ClassificationStage"]

ClassifierFactory = Callable[[ChunkerConfig, logging.LoggerAdapter], ChunkClassifier]
ProgressCallback = Callable[[int, int], None]  # (chunks settled, chunks pending)


@dataclass(frozen=True)
class ClassificationResult:
    chunks: List[MergedChunk]       # annotated, document order
    memory_path: Path
    created_categories: List[str]   # in creation order
    classified: int
    failed: int


class ClassificationStage:
    """Runs classification for one document; the Gemini client is only built when a chunk needs it."""

    def __init__(
        self,
        config: ChunkerConfig,
        logger: Optional[logging.LoggerAdapter] = None,
        classifier_factory: Optional[ClassifierFactory] = None,
    ) -> None:
        self._cfg = config
        self._log = logger or get_logger(module="CLASSIFICATION")
        self._classifier_factory = classifier_factory

    def run(
        self, merged_chunks_path: Path, target_chunks_path: Path, memory_path: Path, on_progress: Optional[ProgressCallback] = None,
    ) -> ClassificationResult:
        chunks = load_merged_chunks(merged_chunks_path)
        fingerprints = {chunk["chunk_id"]: chunk_fingerprint(chunk) for chunk in chunks}
        memory = CategoryMemory.load(
            memory_path,
            general=self._cfg.classification_categories,
            allow_new=self._cfg.classification_allow_new_categories,
            logger=self._log,
        )
        memory.retain(fingerprints)

        pending = [c for c in chunks if memory.lookup(c["chunk_id"], fingerprints[c["chunk_id"]]) is None]
        self._log.info("Classification: %d chunk(s), %d pending", len(chunks), len(pending))
        if pending:
            classifier = (self._classifier_factory or ChunkClassifier)(self._cfg, self._log)
            batched = self._cfg.classification_batch_size > 1 and callable(getattr(type(classifier), "classify_batch", None))
            done = 0
            for batch in (self._batches(pending) if batched else ([chunk] for chunk in pending)):
                if len(batch) > 1:
                    self._classify_batch(classifier, memory, batch, fingerprints)
                else:
                    self._classify_one(classifier, memory, batch[0], fingerprints[batch[0]["chunk_id"]])
                memory.save(memory_path)  # after every batch: an interrupted run resumes here
                done += len(batch)
                if on_progress is not None:
                    on_progress(done, len(pending))
        memory.save(memory_path)

        annotated = [
            _with_categories(chunk, memory.chunk_fields(chunk["chunk_id"], fingerprints[chunk["chunk_id"]])) for chunk in chunks
        ]
        save_merged_chunks(merged_chunks_path, annotated)
        ChunkStore(target_chunks_path, self._log).annotate(_fields_by_source_chunk(annotated))

        failed = sum(1 for chunk in annotated if chunk[F_MAIN_CATEGORY] is None)
        created = list(memory.created)
        log = self._log.warning if failed else self._log.info
        log(
            "Classification complete: %d classified, %d unclassified, created categories: %s",
            len(annotated) - failed, failed, created or "none",
        )
        return ClassificationResult(
            chunks=annotated, memory_path=Path(memory_path), created_categories=created,
            classified=len(annotated) - failed, failed=failed,
        )

    def _batches(self, chunks: List[MergedChunk]) -> Iterator[List[MergedChunk]]:
        """Consecutive runs of at most `classification_batch_size` chunks within the character budget."""
        batch: List[MergedChunk] = []
        chars = 0
        for chunk in chunks:
            size = prompt_chars(chunk)
            if batch and (len(batch) >= self._cfg.classification_batch_size or chars + size > self._cfg.classification_batch_max_chars):
                yield batch
                batch, chars = [], 0
            batch.append(chunk)
            chars += size
        if batch:
            yield batch

    def _classify_batch(
        self, classifier: ChunkClassifier, memory: CategoryMemory, batch: List[MergedChunk], fingerprints: Dict[str, str],
    ) -> None:
        """One call for the batch; answers resolved and committed in order; misses and rule breaks classified alone."""
        try:
            answers = classifier.classify_batch(batch, memory)
        except Exception as exc:  # the whole batch failed: every chunk still gets its own attempts
            self._log.warning("Batch of %d chunk(s) failed, classifying them one by one: %s", len(batch), exc)
            answers = {}
        for chunk in batch:
            chunk_id = chunk["chunk_id"]
            answer = answers.get(chunk_id)
            if answer is not None:
                try:
                    self._commit(memory, chunk_id, memory.resolve(answer, fingerprints[chunk_id]))
                    continue
                except InvalidClassificationError as exc:
                    self._log.debug("Chunk %s: batch answer rejected (%s), classifying alone", chunk_id, exc)
            self._classify_one(classifier, memory, chunk, fingerprints[chunk_id])

    def _classify_one(self, classifier: ChunkClassifier, memory: CategoryMemory, chunk: MergedChunk, fingerprint: str) -> None:
        chunk_id = chunk["chunk_id"]
        try:
            resolution = classifier.classify(chunk, memory, fingerprint)
        except Exception as exc:
            memory.record_failure(chunk_id, f"{type(exc).__name__}: {exc}")
            self._log.error("Chunk %s left unclassified: %s", chunk_id, exc)
        else:
            self._commit(memory, chunk_id, resolution)

    def _commit(self, memory: CategoryMemory, chunk_id: str, resolution: Resolution) -> None:
        entry = resolution.classification
        if memory.commit(chunk_id, resolution):
            self._log.info("Created category %r at chunk %s: %s", entry.main_category, chunk_id, resolution.new_category_description)
        if resolution.dropped_sub_categories:
            self._log.debug("Chunk %s: dropped sub categories %s", chunk_id, list(resolution.dropped_sub_categories))
        self._log.debug("Chunk %s → main=%s sub=%s", chunk_id, entry.main_category, entry.sub_categories)


def _with_categories(chunk: MergedChunk, fields: Dict[str, Any]) -> MergedChunk:
    """The chunk with the category keys placed before `metadata`, so the artifact reads top-down."""
    head = {key: value for key, value in chunk.items() if key not in fields and key != "metadata"}
    return {**head, **fields, "metadata": chunk.get("metadata") or {}}


def _fields_by_source_chunk(chunks: List[MergedChunk]) -> Dict[str, Dict[str, Any]]:
    """Every raw chunk id → the categories of the merged chunk it ended up in."""
    out: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        fields = {F_MAIN_CATEGORY: chunk[F_MAIN_CATEGORY], F_SUB_CATEGORIES: chunk[F_SUB_CATEGORIES]}
        for source_id in (chunk.get("metadata") or {}).get("source_chunk_ids") or [chunk["chunk_id"]]:
            out[source_id] = fields
    return out
