"""
app.observability.sanitize
==========================
What may leave the process as trace metadata — and nothing else.

    METADATA_KEYS      : the allowlist; `sanitize_metadata` drops any other key, caps strings and lists,
                         and flattens enums, so a span can never carry chunk text, prompts, model output,
                         document content or a secret by accident (there is no key for them)
    classify_error     : an exception → {error_type, error_category, http_status?, error_code?, error_message?}
                         The message is kept only for the service's own client-facing errors (they have a
                         string `code` and are returned to HTTP clients anyway); provider messages are not
    query_fingerprint  : a query → {query_sha256_12, query_chars} — what retrieval traces record by default

Pure functions; imports nothing from the rest of the app.
"""

from __future__ import annotations

import enum
import hashlib
from typing import Any, Dict, List, Mapping, Optional

__all__ = ["ERROR_CATEGORIES", "METADATA_KEYS", "classify_error", "query_fingerprint", "sanitize_metadata", "sanitize_value"]

MAX_STRING = 256       # one metadata value
MAX_LIST_ITEMS = 200   # ids / scores in an output
MAX_LIST_STRING = 160  # one list item

METADATA_KEYS = frozenset({
    # identity
    "module_name", "execution_kind", "execution_id", "dataset_name", "dataset_version", "dataset_schema_version", "case_id",
    "attempt", "request_id", "job_id", "document_id", "collection_name", "content_hash", "version", "pipeline_version",
    "config_fingerprint", "service_version", "nested_module",
    # configuration
    "language", "language_hint", "ocr_technique", "ocr_model", "chunking_model", "classification_model", "autogrounding_model",
    "embedding_model", "reranking_model", "dense_dim", "classification", "autogrounding", "replace", "concurrency",
    "prompt_fingerprint", "batch_size", "model", "temperature",
    "strategy", "fusion_strategy", "fusion_mode", "alpha", "rrf_k", "top_k", "dense_top_k", "sparse_top_k", "rerank_candidates",
    "final_top_k", "filter_present", "use_hybrid", "use_reranking", "reformulated", "analyzer",
    # counts & outcomes
    "page_count", "page_number", "pages_total", "pages_pending", "pages_resumed", "chunk_count", "chunks_in", "chunks_out",
    "fused_chains", "classified", "failed", "failed_pages", "terms", "questions", "question_count", "rows_upserted",
    "rows_deleted", "uploaded_files", "already_existed", "windows", "rendered", "hits", "degraded", "applied", "candidates",
    "result_count", "chunk_ids", "scores", "retry_count", "status", "stage", "source_artifact", "uploaded_to_milvus",
    "duration_ms",
    # datasets
    "row_count", "checksum_sha256", "created", "promoted", "verify_references", "langfuse_sync_status", "item_count",
    "issue_count",
    # queries (never the text unless the capture flag is on — see Telemetry.query_input)
    "query_sha256_12", "query_chars",
    # errors
    "error_type", "error_category", "error_code", "http_status", "error_message",
})

ERROR_CATEGORIES = ("rate_limited", "timeout", "client_error", "not_found", "conflict", "provider_error", "internal")

_RATE_LIMIT_NAMES = ("toomanyrequests", "ratelimit", "resourceexhausted", "quota")
_TIMEOUT_NAMES = ("timeout", "timedout", "deadlineexceeded")
_PROVIDER_NAMES = ("apierror", "clienterror", "botocoreerror", "serviceunavailable", "httperror", "milvusexception", "grpc")


def sanitize_value(value: Any, *, max_string: int = MAX_STRING) -> Any:
    """A JSON-safe scalar or a short list of scalars; nested mappings are dropped (None)."""
    if isinstance(value, enum.Enum):
        value = value.value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return value if len(value) <= max_string else value[: max_string - 1] + "…"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [sanitize_value(item, max_string=MAX_LIST_STRING) for item in list(value)[:MAX_LIST_ITEMS]]
        return [item for item in items if item is None or isinstance(item, (bool, int, float, str))]
    return None


def sanitize_metadata(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Allowlisted keys only, every value sanitised, None values dropped."""
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if key not in METADATA_KEYS:
            continue
        cleaned = sanitize_value(value)
        if cleaned is not None:
            out[key] = cleaned
    return out


def classify_error(exc: BaseException) -> Dict[str, Any]:
    """
    Structured, non-sensitive description of an exception.

    Category by shape, not by import: the service's own errors carry a string `code` (and `http_status`
    for channel errors); Gemini's carry an int `code`; botocore's carry `response["Error"]["Code"]`;
    timeouts and throttling are recognised by class name. Anything else is `internal`.
    """
    name = type(exc).__name__
    lowered = name.lower()
    out: Dict[str, Any] = {"error_type": name}

    own_code = getattr(exc, "code", None)
    http_status = getattr(exc, "http_status", None)
    aws_code = None
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        aws_code = (response.get("Error") or {}).get("Code") if isinstance(response.get("Error"), Mapping) else None

    if isinstance(own_code, str) and own_code:
        # ChannelError / RetrievalError / NoContentError: client-caused, messages written for HTTP clients.
        out["error_code"] = own_code
        out["error_message"] = sanitize_value(str(exc))
        status = http_status if isinstance(http_status, int) else _status_for_code(own_code)
        if status:
            out["http_status"] = status
        out["error_category"] = _category_for_status(status) if status else "client_error"
        return out

    stage = getattr(exc, "stage", None)
    failed = getattr(exc, "failed", None)
    if isinstance(stage, str) and isinstance(failed, Mapping):  # IncompleteStageError: pages, never the provider text
        out.update(error_category="provider_error", stage=stage, failed=len(failed))
        return out

    if isinstance(own_code, int) and own_code:  # google.genai APIError and friends
        out["http_status"] = own_code
        out["error_category"] = _category_for_status(own_code)
        return out

    if aws_code:
        out["error_code"] = str(aws_code)
        out["error_category"] = "not_found" if aws_code in ("NoSuchKey", "404", "NotFound") else "provider_error"
        return out

    if isinstance(exc, TimeoutError) or any(marker in lowered for marker in _TIMEOUT_NAMES):
        out["error_category"] = "timeout"
    elif any(marker in lowered for marker in _RATE_LIMIT_NAMES):
        out["error_category"] = "rate_limited"
    elif any(marker in lowered for marker in _PROVIDER_NAMES):
        out["error_category"] = "provider_error"
    else:
        out["error_category"] = "internal"
    return out


def query_fingerprint(query: str) -> Dict[str, Any]:
    """Stable, non-reversible identity of a query: enough to join traces, never the text."""
    text = (query or "").strip()
    return {"query_sha256_12": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], "query_chars": len(text)}


def _status_for_code(code: str) -> Optional[int]:
    if code in ("COLLECTION_NOT_FOUND", "S3_DOCUMENT_NOT_FOUND", "JOB_NOT_FOUND"):
        return 404
    if code in ("COLLECTION_LANGUAGE_MISMATCH", "LEGACY_COLLECTION", "DOCUMENT_ID_CONFLICT", "DATASET_VERSION_CONFLICT"):
        return 409
    if code in ("NO_CHUNKS", "DATASET_INVALID", "NO_DATASET_SCHEMA"):
        return 422
    return None


def _category_for_status(status: int) -> str:
    if status == 429:
        return "rate_limited"
    if status in (408, 504):
        return "timeout"
    if status == 404:
        return "not_found"
    if status == 409:
        return "conflict"
    if 400 <= status < 500:
        return "client_error"
    return "provider_error"


def summarize_ids(ids: List[Any]) -> List[str]:
    """Up to MAX_LIST_ITEMS ids as strings (for `chunk_ids` outputs)."""
    return [str(item)[:MAX_LIST_STRING] for item in ids[:MAX_LIST_ITEMS]]
