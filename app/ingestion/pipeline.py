"""
app.ingestion.pipeline
======================
IngestPipeline — PDF ─► searchable Milvus collection, with every intermediate artifact on disk.

    stage 1 ocr      : Mistral OCR → <doc>.md, pages.json, pdf_images/  (page windows for long PDFs,
                       resumable per window, retried per call); pages.json holds every PDF page —
                       a page OCR skipped is chunked from its image alone
            language : resolve_document_language — the first document of a collection fixes its
                       language; a later one must match it (checked before any paid stage)
    stage 2 render   : PyMuPDF     → page_images/page_{n}.jpeg   (runs alongside stage 1; resumable per page)
    stage 3 chunk    : Gemini      → target_chunks.json, token_usage.json   (`chunking_concurrency` pages
                       at once, resumable per page; a failed page never stops the others)
    stage 4 merge    : pure        → merged_chunks.json
    stage 5 classify : Gemini      → classification_memory.json; main/sub categories
                                     written onto merged + target chunks     (optional)
    stage 6 vocab    : pure/Groq   → vocab_index.json
    stage 7 ground   : Gemini      → groundtruth_dataset.{json,csv}          (optional)
    stage 8 index    : Cohere+Milvus upsert into `collection_name`, in batches; the document's rows
                       from an earlier version that are gone are deleted (only when a collection is given)

Stages 1–3 skip work that is already present and newer than the PDF (older work is redone), so a
kept workspace (debug, or a failed production run) resumes cheaply; stage 5 reuses every
assignment whose chunk is unchanged, stage 7 every page already grounded.

Progress: `on_progress(stage, done, total)` is called as each stage advances — stage names are
`STAGES` (ocr · chunk · merge · classify · vocab · ground · index); the caller weights them.

Failure policy: every stage error propagates — a partially ingested document is never reported
as complete — except a chunk the classifier could not label, which is indexed without categories
(app.classification.stage). A document with no chunk at all raises `NoContentError` after stage 4,
never an empty success. Stage 3 first tries every page: pages that fail every retry are named
by `IncompleteStageError` once the others are stored; an error no page can survive (invalid key,
unknown model) stops it at once. Synchronous and CPU/IO heavy; the API layer runs it on a worker
thread behind an admission gate.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import fitz  # PyMuPDF

from app.chunking.chunker import IntelligentPDFChunker, is_fatal_gemini_error
from app.chunking.schema import ChunkingInput
from app.chunking.store import ChunkStore
from app.chunking.token_monitor import TokenUsageMonitor
from app.classification.stage import ClassificationResult, ClassificationStage
from app.config.defaults import ChunkerConfig
from app.groundtruth.autogrounding import AutoGrounder
from app.ingestion.document_parser import md_to_page_json
from app.ingestion.errors import IncompleteStageError, NoContentError
from app.ingestion.ocr import run_ocr
from app.logging.setup import get_logger, in_context
from app.postprocessing.merger import MergedChunk, merge_file
from app.postprocessing.vocab_index import build_vocab_index
from app.retrieval.documents import merged_chunks_to_documents
from app.retrieval.errors import CollectionLanguageMismatchError
from app.retrieval.language import detect_language
from app.retrieval.profiles import language_profile
from app.retrieval.vector_db import EnhancedCohereMilvusRetriever
from app.utils.pdf_utils import render_pdf_pages, split_content

__all__ = ["STAGES", "IngestPipeline", "IngestResult", "ProgressCallback", "index_merged_chunks", "resolve_document_language"]

STAGES = ("ocr", "chunk", "merge", "classify", "vocab", "ground", "index")
ProgressCallback = Callable[[str, int, int], None]  # (stage, done, total)
_STORE_FLUSH_SECONDS = 5.0  # target_chunks.json written at most this often while stage 3 runs


@dataclass(frozen=True)
class IngestResult:
    """Where everything landed for one document."""

    pdf_name: str
    base_dir: str
    ocr_markdown_path: str
    target_chunks_path: str
    merged_chunks_path: str
    vocab_index_path: str
    classification_path: Optional[str]          # None when stage 5 did not run
    created_categories: Optional[List[str]]     # categories the model added to the general ones
    groundtruth_path: Optional[str]
    collection_name: Optional[str]
    language: str
    chunk_count: int
    page_count: int
    uploaded: bool


@dataclass(frozen=True)
class _Workspace:
    """Artifact layout under `ChunkerConfig.doc_folder(doc_name)`."""

    pdf: Path
    base: Path
    page_images: Path
    ocr_images: Path

    @property
    def markdown(self) -> Path:
        return self.base / f"{self.pdf.stem}.md"

    @property
    def pages_json(self) -> Path:
        return self.base / "pages.json"

    @property
    def target_chunks(self) -> Path:
        return self.base / "target_chunks.json"

    @property
    def token_usage(self) -> Path:
        return self.base / "token_usage.json"

    @property
    def merged_chunks(self) -> Path:
        return self.base / "merged_chunks.json"

    @property
    def vocab_index(self) -> Path:
        return self.base / "vocab_index.json"

    @property
    def classification_memory(self) -> Path:
        return self.base / "classification_memory.json"

    def page_image(self, page_no: int) -> Path:
        return self.page_images / f"page_{page_no}.jpeg"


class IngestPipeline:
    """Runs the eight stages in order for one PDF."""

    def __init__(self, config: Optional[ChunkerConfig] = None, logger: Optional[logging.LoggerAdapter] = None) -> None:
        self._cfg = config or ChunkerConfig()
        self._log = logger or get_logger(module="PIPELINE")

    def run(
        self,
        pdf_path: Path,
        *,
        collection_name: Optional[str] = None,
        autogrounding: bool = False,
        language: Optional[str] = None,
        classification: Optional[bool] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> IngestResult:
        """
        ocr ─► language ─► render ─► chunk ─► merge ─► [classify] ─► vocab ─► [ground] ─► [index].

        Indexed into `collection_name` when one is given (the API always gives one); never into a name
        derived from the PDF. Language: see `resolve_document_language`.
        `classification=None` follows `ChunkerConfig.classification_enabled`.
        """
        pdf = Path(pdf_path).resolve()
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        doc = pdf.stem
        ws = _Workspace(
            pdf=pdf,
            base=self._cfg.doc_folder(doc),
            page_images=self._cfg.page_images_folder(doc),
            ocr_images=self._cfg.ocr_images_folder(doc),
        )
        ws.base.mkdir(parents=True, exist_ok=True)
        self._log.info("Pipeline start: doc=%s workspace=%s", doc, ws.base)
        self._progress = on_progress or (lambda stage, done, total: None)

        should_classify = classification if classification is not None else self._cfg.classification_enabled

        renderer = _Background(self._render_ahead, ws, name="render")  # needs only the PDF: overlaps OCR
        try:
            pages = self._stage_ocr(ws)
        finally:
            renderer.join()  # never leave a thread writing into a workspace that may be removed
        document_language = resolve_document_language(
            "\n".join(pages.values()), language, collection_name, config=self._cfg, logger=self._log,
        )  # before the paid stages: a document the collection cannot take stops here
        self._stage_render(ws, len(pages), renderer)
        self._stage_chunk(ws, pages)
        merged = self._stage_merge(ws)
        if not merged:
            raise NoContentError(doc, len(pages))  # before classify / vocab / ground / index: nothing to label or publish
        classified = self._stage_classify(ws) if should_classify else None
        if classified is not None:
            merged = classified.chunks
        self._stage_vocab(ws)
        groundtruth = self._stage_ground(ws) if autogrounding else None

        if collection_name:
            self._stage_index(merged, collection_name, document_id=doc, language=document_language)
            self._progress("index", 1, 1)
        else:
            self._log.info("Milvus upload skipped: doc=%s", doc)

        self._log.info(
            "Pipeline complete: doc=%s language=%s pages=%d chunks=%d classified=%s uploaded=%s",
            doc, document_language, len(pages), len(merged), should_classify, bool(collection_name),
        )
        return IngestResult(
            pdf_name=doc,
            base_dir=str(ws.base),
            ocr_markdown_path=str(ws.markdown),
            target_chunks_path=str(ws.target_chunks),
            merged_chunks_path=str(ws.merged_chunks),
            vocab_index_path=str(ws.vocab_index),
            classification_path=str(classified.memory_path) if classified else None,
            created_categories=classified.created_categories if classified else None,
            groundtruth_path=str(groundtruth) if groundtruth else None,
            collection_name=collection_name or None,
            language=document_language,
            chunk_count=len(merged),
            page_count=len(pages),
            uploaded=bool(collection_name),
        )

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def _stage_ocr(self, ws: _Workspace) -> Dict[str, str]:
        if _is_fresh(ws.markdown, ws.pdf):
            self._log.info("Stage 1 ocr: reusing %s", ws.markdown)
        else:
            self._log.info("Stage 1 ocr: running OCR")
            run_ocr(ws.pdf, ws.markdown, ws.ocr_images, self._cfg, self._log, on_progress=lambda d, t: self._progress("ocr", d, t))
        self._progress("ocr", 1, 1)
        return md_to_page_json(ws.markdown, ws.pages_json, self._log, page_count=_pdf_page_count(ws.pdf))

    def _render_ahead(self, ws: _Workspace) -> None:
        """Background, during OCR: render with PyMuPDF's page count (skipped when it cannot read the PDF)."""
        count = _pdf_page_count(ws.pdf)
        if count is not None:
            self._render_pages(ws, count)

    def _render_pages(self, ws: _Workspace, page_count: int) -> None:
        """Every page image, keeping fresh ones; nothing to do when all `page_count` are fresh."""
        if all(_is_fresh(ws.page_image(n), ws.pdf) for n in range(1, page_count + 1)):
            return
        stale = any(not _is_fresh(path, ws.pdf) for path in ws.page_images.glob("page_*.jpeg"))
        self._log.info("Stage 2 render: rendering page images (%s)", "all — PDF changed" if stale else "missing only")
        render_pdf_pages(ws.pdf, ws.page_images, only_missing=not stale)

    def _stage_render(self, ws: _Workspace, page_count: int, renderer: "_Background") -> None:
        """Surface the background render's error, or render now if it did not run; then check every page exists."""
        expected = [ws.page_image(n) for n in range(1, page_count + 1)]
        if renderer.error is not None:
            raise renderer.error
        self._render_pages(ws, page_count)
        self._log.info("Stage 2 render: %d page image(s) ready", page_count)
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Rendered images missing for {len(missing)} page(s): {missing[:3]}")

    def _stage_chunk(self, ws: _Workspace, pages: Dict[str, str]) -> None:
        """
        Pending pages, `chunking_concurrency` at a time. A page that fails every retry is recorded and the
        others go on; the stage then raises IncompleteStageError naming those pages (a rerun chunks only
        them). An error no page can survive stops new pages at once (running ones finish and are kept).
        """
        if ws.target_chunks.exists() and not _is_fresh(ws.target_chunks, ws.pdf):
            # Chunks of an earlier version of this PDF: reusing them would index the old content.
            self._log.info("Stage 3 chunk: %s is older than the PDF; chunking every page again", ws.target_chunks.name)
            for stale in (ws.target_chunks, ChunkStore.empty_pages_path_for(ws.target_chunks)):
                stale.unlink(missing_ok=True)
        store = ChunkStore(ws.target_chunks, self._log, flush_interval=_STORE_FLUSH_SECONDS)
        completed = store.completed_pages()
        numbers = _page_numbers(pages)
        pending = [n for n in numbers if n not in completed]
        done = len(numbers) - len(pending)
        self._log.info("Stage 3 chunk: %d page(s), %d pending, %d at a time", len(pages), len(pending), self._cfg.chunking_concurrency)
        self._progress("chunk", done, len(numbers))
        if not pending:
            return

        chunker = IntelligentPDFChunker(
            config=self._cfg,
            store=store,
            token_monitor=TokenUsageMonitor(ws.token_usage, self._cfg.chunking_model, self._log),
            logger=self._log,
        )

        def page_input(page_no: int) -> ChunkingInput:
            previous = pages.get(f"page_{page_no - 1}", "")
            following = pages.get(f"page_{page_no + 1}", "")
            return ChunkingInput(
                page_number=page_no,
                page_text=pages[f"page_{page_no}"],
                page_image_path=ws.page_image(page_no),
                previous_page_text=split_content(previous, "end") if previous else None,
                next_page_text=split_content(following, "start") if following else None,
            )

        failed: Dict[int, str] = {}
        chunk_page = in_context(chunker.chunk_page)
        pool = ThreadPoolExecutor(max_workers=self._cfg.chunking_concurrency, thread_name_prefix="chunk")
        futures: Dict[Future, int] = {pool.submit(chunk_page, page_input(n)): n for n in pending}
        try:
            for future in as_completed(futures):
                page_no = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    if is_fatal_gemini_error(exc):
                        raise
                    failed[page_no] = f"{type(exc).__name__}: {exc}"
                    self._log.error("Stage 3 chunk: page %d failed every attempt; continuing with the others: %s", page_no, exc)
                    continue
                done += 1
                self._progress("chunk", done, len(numbers))
        except BaseException:
            for future in futures:
                future.cancel()  # pages not started yet; running ones finish and are stored for the resume
            raise
        finally:
            pool.shutdown(wait=True)
            store.flush()
        if failed:
            raise IncompleteStageError("chunk", failed, len(numbers))

    def _stage_merge(self, ws: _Workspace) -> List[MergedChunk]:
        self._log.info("Stage 4 merge: fusing cross-page chains")
        merged = merge_file(ws.target_chunks, ws.merged_chunks, self._log)
        self._progress("merge", 1, 1)
        return merged

    def _stage_classify(self, ws: _Workspace) -> ClassificationResult:
        self._log.info("Stage 5 classify: main/sub categories per merged chunk")
        self._progress("classify", 0, 1)
        result = ClassificationStage(self._cfg, self._log).run(
            ws.merged_chunks, ws.target_chunks, ws.classification_memory, on_progress=lambda d, t: self._progress("classify", d, t),
        )
        self._progress("classify", 1, 1)
        return result

    def _stage_vocab(self, ws: _Workspace) -> None:
        self._log.info("Stage 6 vocab: building vocabulary index")
        build_vocab_index(ws.merged_chunks, ws.vocab_index, self._cfg, self._log)
        self._progress("vocab", 1, 1)

    def _stage_ground(self, ws: _Workspace) -> Path:
        self._log.info("Stage 7 ground: generating evaluation Q&A")
        self._progress("ground", 0, 1)
        path = AutoGrounder(self._cfg, self._log).run(ws.merged_chunks, on_progress=lambda d, t: self._progress("ground", d, t))
        self._progress("ground", 1, 1)
        return path

    def _stage_index(self, merged: List[MergedChunk], collection_name: str, *, document_id: str, language: str) -> None:
        index_merged_chunks(
            merged, collection_name, document_id=document_id, language=language, config=self._cfg, logger=self._log,
            on_progress=lambda done, total: self._progress("index", done, total),
        )


def resolve_document_language(
    text: str,
    requested: Optional[str],
    collection_name: Optional[str],
    *,
    config: ChunkerConfig,
    logger: logging.LoggerAdapter,
) -> str:
    """
    The language a document is indexed with — the collection's first document fixes it.

        new collection (or none) : `requested` when given, else detected from `text`
        existing collection      : its language is authoritative; the document's *detected* language must
                                   match it, else CollectionLanguageMismatchError. `requested` is ignored
                                   (a caller cannot force a document into a collection of the other language).

    Raises LegacyCollectionError for a read-only legacy collection, ValueError for an unsupported `requested`.
    """
    requested_language = language_profile(requested, config).language if requested else None  # validates
    detected = detect_language(text, japanese_threshold=config.japanese_detection_threshold)
    collection_language = None
    if collection_name:
        with EnhancedCohereMilvusRetriever(collection_name, config=config, logger=logger) as retriever:
            collection_language = retriever.existing_language()
    if collection_language is None:
        language = requested_language or detected
        logger.info("Language: %s (%s; sets the language of collection %s)", language,
                    "requested" if requested_language else "detected", collection_name or "-")
        return language
    if requested_language and requested_language != collection_language:
        logger.warning("Language: requested %r ignored — collection '%s' is %r", requested_language, collection_name, collection_language)
    if detected != collection_language:
        raise CollectionLanguageMismatchError(collection_name, collection_language, detected)
    logger.info("Language: %s (detected; matches collection '%s')", detected, collection_name)
    return detected


def index_merged_chunks(
    merged: List[MergedChunk],
    collection_name: str,
    *,
    document_id: str,
    language: str,
    config: ChunkerConfig,
    logger: logging.LoggerAdapter,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Stage 8: upsert `merged` as `document_id`'s rows of `collection_name`, removing its stale rows."""
    documents = merged_chunks_to_documents(merged, document_id=document_id)
    for document in documents:
        document.metadata["language"] = language
    logger.info("Stage 8 index: upserting %d document(s) into '%s' (language=%s)", len(documents), collection_name, language)
    if on_progress is not None:
        on_progress(0, len(documents))
    # The retriever owns an executor and live connections — release them per ingestion.
    with EnhancedCohereMilvusRetriever(collection_name, config=config, logger=logger, language=language) as retriever:
        retriever.replace_document(documents, document_id=document_id, batch_size=config.index_batch_size, on_progress=on_progress)


class _Background:
    """Run `target(*args)` on a thread now; `join()` waits; `error` holds what it raised (re-raised by the caller)."""

    def __init__(self, target: Callable[..., None], *args: object, name: str) -> None:
        self.error: Optional[BaseException] = None

        def run() -> None:
            try:
                target(*args)
            except BaseException as exc:  # handed to the main thread, which decides
                self.error = exc

        self._thread = threading.Thread(target=run, name=name, daemon=True)
        self._thread.start()

    def join(self) -> None:
        self._thread.join()


def _pdf_page_count(pdf: Path) -> Optional[int]:
    try:
        with fitz.open(pdf) as document:
            return document.page_count
    except Exception:  # unreadable here: stage 2 renders after OCR with OCR's page count
        return None


def _page_numbers(pages: Dict[str, str]) -> List[int]:
    return sorted(int(key.split("_", 1)[1]) for key in pages)


def _is_fresh(artifact: Path, source: Path) -> bool:
    """True when `artifact` exists, is non-empty, and is at least as new as `source`."""
    return artifact.exists() and artifact.stat().st_size > 0 and artifact.stat().st_mtime >= source.stat().st_mtime
