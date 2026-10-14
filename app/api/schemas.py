"""
app.api.schemas
===============
Request/response models. Validation happens here so bad input is a 422, never a 500.

`CollectionName` is the one collection-name type of the API (bodies, forms and paths alike):
trimmed + lower-cased, then `^[a-z][a-z0-9_]{2,63}$` (app.utils.naming.normalize_collection_name).
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from pydantic import AfterValidator, BaseModel, Field, field_validator

from app.retrieval.config import FusionStrategy
from app.retrieval.filters import normalize_categories
from app.utils.naming import COLLECTION_NAME_RULE, MAX_CATEGORIES, normalize_collection_name

__all__ = [
    "CollectionDeleteResponse", "CollectionName", "CollectionListResponse", "DocumentProcessResponse", "ErrorDetail",
    "ErrorResponse", "HealthResponse", "IngestionJobResponse", "QueryReformulationRequest", "QueryReformulationResponse",
    "ReadinessResponse", "RetrievalQueryRequest", "RetrievalQueryResponse", "S3DocumentImportRequest", "S3DocumentImportResponse",
    "S3DocumentRequest",
    "S3GroundingResponse",
]


CollectionName = Annotated[
    str,
    AfterValidator(normalize_collection_name),
    Field(description=f"Milvus collection: {COLLECTION_NAME_RULE}; case-insensitive (stored lower-case).", examples=["manuals_en"]),
]


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value.strip()


def _category_list(value: Optional[List[str]]) -> Optional[List[str]]:
    """Normalised like the stored categories (`"Trouble Shooting"` → `trouble_shooting`); [] means no filter."""
    if value is None:
        return None
    return list(normalize_categories(value)) or None


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ErrorResponse(BaseModel):
    error: ErrorDetail


class DocumentProcessResponse(BaseModel):
    document_id: str
    status: str
    collection_name: Optional[str] = None
    language: Optional[str] = None  # "ja" / "en": the document's (and its collection's) language
    page_count: int
    chunk_count: int
    uploaded_to_milvus: bool
    classified: bool = False                            # stage 5 ran: chunks carry main/sub categories
    created_categories: Optional[List[str]] = None      # main categories the model added to the general ones
    artifacts: Optional[Dict[str, Optional[str]]] = None  # local paths; debug mode only


class S3DocumentRequest(BaseModel):
    """A document published to S3: `folder_name` is its folder (its `document_id`). `s3_bucket` /
    `AWS_DEFAULT_REGION` default to the server's S3_BUCKET / AWS_DEFAULT_REGION when omitted.
    Body of `POST /v1/documents/from-s3/autogrounding`."""

    folder_name: str
    s3_bucket: Optional[str] = None
    AWS_DEFAULT_REGION: Optional[str] = None

    @field_validator("folder_name")
    @classmethod
    def _single_folder(cls, value: str) -> str:
        value = _not_blank(value).strip("/")
        if not value or "/" in value:
            raise ValueError("must be a single top-level folder name")
        return value


class S3DocumentImportRequest(S3DocumentRequest):
    """`POST /v1/documents/from-s3`."""

    collection_name: CollectionName         # required: never derived from the folder name
    language: Optional[str] = None          # hint for a new collection only; an existing one's language is authoritative


class S3DocumentImportResponse(BaseModel):
    document_id: str
    status: str
    version: str                            # the S3 version (PDF content hash) that was indexed
    source_artifact: str                    # merged_chunks.json, or target_chunks.json (merged by the service)
    collection_name: str
    language: str
    page_count: int
    chunk_count: int
    uploaded_to_milvus: bool
    classified: bool


class S3GroundingResponse(BaseModel):
    """`POST /v1/documents/from-s3/autogrounding`."""

    document_id: str
    status: str
    version: str                            # the S3 version the groundtruth was generated for and saved into
    source_artifact: str
    page_count: int
    question_count: int
    failed_pages: List[int]                 # failed every retry; calling again retries only these
    artifacts: List[str]                    # S3 keys written (groundtruth_dataset.json / .csv)


class IngestionJobResponse(BaseModel):
    """`POST /v1/ingestion-jobs` and `GET /v1/ingestion-jobs/{job_id}`; poll until `status` is final."""

    job_id: str
    document_id: str
    status: str                              # queued | running | succeeded | failed
    stage: Optional[str] = None              # ocr · chunk · merge · classify · vocab · ground · index · publish
    stage_done: int = 0                      # e.g. pages chunked so far
    stage_total: int = 0                     # e.g. pages to chunk
    progress: float = 0.0                    # 0–1 over the stages this job runs, weighted by their usual share
    stages: List[str] = Field(default_factory=list)
    attempt: int = 0                         # runs started so far (a retry or a restart resumes from the workspace)
    max_attempts: int = 1                    # runs the service makes before the job fails
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[DocumentProcessResponse] = None   # when succeeded
    error: Optional[ErrorDetail] = None                # when failed; `retryable` → resubmit resumes the run


class RetrievalQueryRequest(BaseModel):
    collection_name: CollectionName
    query: str
    top_k: Optional[int] = Field(default=None, ge=1, le=200)
    use_hybrid: Optional[bool] = None
    use_reranking: Optional[bool] = None
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    fusion_strategy: Optional[FusionStrategy] = None
    reformulated_query: Optional[str] = None
    # Category filters (AND-ed; values within a list OR-ed) — see app.retrieval.filters.
    main_categories: Optional[List[str]] = Field(default=None, max_length=MAX_CATEGORIES)
    sub_categories: Optional[List[str]] = Field(default=None, max_length=MAX_CATEGORIES)
    categories: Optional[List[str]] = Field(default=None, max_length=MAX_CATEGORIES)

    _query_not_blank = field_validator("query")(_not_blank)
    _valid_categories = field_validator("main_categories", "sub_categories", "categories")(_category_list)


class RetrievalQueryResponse(BaseModel):
    results: List[Dict[str, Any]]


class QueryReformulationRequest(BaseModel):
    """`s3_bucket` / `AWS_DEFAULT_REGION` default to the server's S3_BUCKET / AWS_DEFAULT_REGION when omitted."""

    query: str
    collection_name: CollectionName
    s3_bucket: Optional[str] = None
    AWS_DEFAULT_REGION: Optional[str] = None

    _query_not_blank = field_validator("query")(_not_blank)


class QueryReformulationResponse(BaseModel):
    query: str
    reformulated_query: str


class CollectionListResponse(BaseModel):
    db_name: str
    count: int
    collections: List[str]


class CollectionDeleteResponse(BaseModel):
    db_name: str
    collection_name: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    milvus: str
    accepting_documents: bool
