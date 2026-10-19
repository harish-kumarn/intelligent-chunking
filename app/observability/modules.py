"""
app.observability.modules
=========================
The canonical module identifiers of the service — the only names a trace may carry.

    ModuleName    : one per workflow with its own entry point; the root trace of a run is named after it
                    and tagged `module:<name>`, so an evaluation pipeline can list every run of a module
    SpanName      : the child spans inside those traces (stage / arm / step names)
    ExecutionKind : why a run happened — production traffic, a test, or an evaluation run — carried as
                    the tag `kind:<value>` so trace populations are told apart by metadata, never by name

Everything is an enum: the telemetry facade accepts these types only, so a typo or an ad-hoc string
can never open a new trace group. Adding a workflow means adding a member here (and documenting it
in docs/OBSERVABILITY.md), nothing else.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ExecutionKind", "ModuleName", "SpanName"]


class ModuleName(StrEnum):
    """Workflows with their own entry point (one root trace per run)."""

    DOCUMENT_INGESTION = "document_ingestion"    # PDFChannel.process: the 8-stage pipeline + S3 publication
    S3_REINDEX = "s3_reindex"                    # S3DocumentImporter.import_document: stage 8 on published chunks
    AUTOGROUNDING = "autogrounding"              # S3DocumentGrounder.ground_document: stage 7 alone
    HYBRID_RETRIEVAL = "hybrid_retrieval"        # EnhancedCohereMilvusRetriever.retrieve_async (dense-only runs included)
    QUERY_REFORMULATION = "query_reformulation"  # QueryReformulationService.reformulate
    DATASET_MANAGEMENT = "dataset_management"    # DatasetService: dataset upload / promotion / Langfuse sync


class SpanName(StrEnum):
    """Child spans. `bm25_search` / `dense_search` are arms of a retrieval, never run on their own."""

    # document_ingestion / s3_reindex
    OCR = "ocr"
    RENDER = "render"
    INTELLIGENT_CHUNKING = "intelligent_chunking"
    CHUNK_PAGE = "chunk_page"            # one generation per page (Gemini vision call, retries folded in)
    MERGE = "merge"
    CLASSIFY = "classify"
    VOCAB = "vocab"
    AUTOGROUNDING = "autogrounding"
    MILVUS_INDEX = "milvus_index"
    S3_PUBLISH = "s3_publish"
    # hybrid_retrieval
    DENSE_SEARCH = "dense_search"
    BM25_SEARCH = "bm25_search"
    FUSION = "fusion"
    RERANK = "rerank"
    # dataset_management
    DATASET_VALIDATE = "dataset_validate"
    DATASET_STORE = "dataset_store"
    LANGFUSE_DATASET_SYNC = "langfuse_dataset_sync"


class ExecutionKind(StrEnum):
    """What a run is for. Set per request with the `X-Execution-Kind` header; `production` when absent."""

    PRODUCTION = "production"
    INGESTION_TEST = "ingestion_test"
    RETRIEVAL_TEST = "retrieval_test"
    EVALUATION = "evaluation"
