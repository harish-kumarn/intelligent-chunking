"""
app.datasets
============
Ground-truth datasets: the versioned CSV contract, its validator, immutable publication to S3 and the
optional Langfuse mirror. Sits beside `app.ingestion` (used by the API and by S3 auto-grounding); it
imports `storage`, `observability`, `logging` and `utils` only.

    schema        : DatasetSchema per module (`retrieval_v1` for hybrid_retrieval)
    validation    : validate_csv — streaming checks + canonical CSV + checksum
    service       : DatasetService — publish / info / promote
    langfuse_sync : mirror one version into Langfuse Datasets (S3 stays canonical)
"""

from app.datasets.schema import RETRIEVAL_V1, SCHEMAS, DatasetSchema, NoDatasetSchemaError, schema_for
from app.datasets.service import DatasetPublication, DatasetService
from app.datasets.validation import DatasetValidationError, ValidationIssue, validate_csv

__all__ = [
    "RETRIEVAL_V1", "SCHEMAS", "DatasetPublication", "DatasetSchema", "DatasetService", "DatasetValidationError",
    "NoDatasetSchemaError", "ValidationIssue", "schema_for", "validate_csv",
]
