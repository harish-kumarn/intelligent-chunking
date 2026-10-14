"""
app.retrieval.milvus_schema
===========================
The Milvus collection layout — declared here, created here, read back here.

    chunk_id     VARCHAR(512)  primary key, `{document_id}:{chunk_id}`
    document_id  VARCHAR(512)
    page         INT64         first source page, PAGE_NULL when unknown
    text         VARCHAR       analyzer = the collection language's analyzer (app.retrieval.profiles)
    dense        FLOAT_VECTOR  Cohere embedding; index + metric from ChunkerConfig
    sparse       SPARSE_FLOAT  BM25 of `text`, computed by Milvus (BM25 function)
    main_category   VARCHAR       nullable, INVERTED index — classification filter (schema v2)
    sub_categories  ARRAY<VARCHAR> nullable, INVERTED index — classification filter (schema v2)
    $meta        dynamic       every other metadata key (image_paths, reference_path, language, ...)

The collection property `maryuma.manifest` records what the collection was built with
(language, embedding model, vector size). A retriever reads it back and searches with
exactly that analyzer and embedding model, whatever the configuration says now.
Collections without a manifest were created by the langchain-milvus layout before
language profiles; they are English and stay searchable (read-only).

Schema v1 collections (created before classification) have no category columns: their
category keys land in `$meta`, and the filter expressions in app.retrieval.filters match
typed and dynamic fields alike, so both versions filter the same way.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

from pymilvus import DataType, Function, FunctionType, MilvusClient

from app.config.defaults import ChunkerConfig
from app.retrieval.language import SUPPORTED_LANGUAGES, normalize_language
from app.retrieval.profiles import JAPANESE_ANALYZER, LanguageProfile
from app.utils.naming import MAX_CATEGORIES, MAX_CATEGORY_NAME_LEN

__all__ = [
    "CollectionLayout", "CollectionManifest", "F_DENSE", "F_DOCUMENT_ID", "F_MAIN_CATEGORY", "F_PAGE", "F_PK", "F_SPARSE",
    "F_SUB_CATEGORIES", "F_TEXT", "MAX_PK_LEN", "MAX_TEXT_BYTES", "PAGE_NULL", "create_collection", "dense_index_params", "describe_layout",
    "warm_up_japanese_analyzer",
]

F_PK = "chunk_id"
F_DOCUMENT_ID = "document_id"
F_PAGE = "page"
F_TEXT = "text"
F_DENSE = "dense"
F_SPARSE = "sparse"
F_MAIN_CATEGORY = "main_category"
F_SUB_CATEGORIES = "sub_categories"
_F_DYNAMIC = "$meta"

PAGE_NULL = -1            # INT64 sentinel for "no page"
MAX_PK_LEN = 512
MAX_TEXT_BYTES = 60_000   # below the 65 535-byte VARCHAR ceiling, leaves UTF-8 headroom
_MAX_TEXT_FIELD = 65_535

_MANIFEST_PROPERTY = "maryuma.manifest"
_SCHEMA_VERSION = 2       # 2: typed, indexed category columns
_SCALAR_INDEX = {"index_type": "INVERTED"}
_BM25_FUNCTION = "text_bm25"
_SPARSE_INDEX = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"}
_VECTOR_DTYPES = frozenset(
    getattr(DataType, name)
    for name in ("BINARY_VECTOR", "FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR", "SPARSE_FLOAT_VECTOR", "INT8_VECTOR")
    if hasattr(DataType, name)
)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CollectionManifest:
    """What a collection was built with; stored as a collection property."""

    language: str
    embedding_model: str
    dense_dim: int
    schema_version: int = _SCHEMA_VERSION

    def to_property(self) -> Dict[str, str]:
        return {_MANIFEST_PROPERTY: json.dumps(asdict(self), sort_keys=True)}

    @classmethod
    def from_properties(cls, properties: Mapping[str, Any]) -> Optional["CollectionManifest"]:
        """The manifest in `properties`, or None when absent (a legacy collection). Raises on a corrupt one."""
        raw = (properties or {}).get(_MANIFEST_PROPERTY)
        if raw is None:
            return None
        data = json.loads(raw)
        manifest = cls(
            language=normalize_language(data["language"]),
            embedding_model=str(data["embedding_model"]),
            dense_dim=int(data["dense_dim"]),
            schema_version=int(data.get("schema_version", _SCHEMA_VERSION)),
        )
        if manifest.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Collection manifest names unsupported language {manifest.language!r}")
        return manifest


@dataclass(frozen=True)
class CollectionLayout:
    """An existing collection, as a retriever needs it."""

    manifest: Optional[CollectionManifest]  # None: legacy (pre-manifest) collection
    output_fields: List[str]                # every scalar field, plus the dynamic field when enabled
    dense_metric: str

    @property
    def is_legacy(self) -> bool:
        return self.manifest is None


# --------------------------------------------------------------------------- #
# Create / describe
# --------------------------------------------------------------------------- #

def dense_index_params(config: ChunkerConfig) -> Dict[str, Any]:
    """Dense index definition from config — `milvus_index_params` wins over type/metric."""
    params = dict(config.milvus_index_params or {})
    params.setdefault("index_type", config.milvus_index_type)
    params.setdefault("metric_type", config.milvus_metric_type)
    params.setdefault("params", {})
    params["metric_type"] = str(params["metric_type"]).upper()
    return params


def create_collection(
    client: MilvusClient,
    name: str,
    *,
    profile: LanguageProfile,
    dense_dim: int,
    dense_index: Dict[str, Any],
) -> CollectionManifest:
    """Create `name` for `profile`'s language with a `dense_dim`-wide dense field; returns its manifest."""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(F_PK, DataType.VARCHAR, max_length=MAX_PK_LEN, is_primary=True)
    schema.add_field(F_DOCUMENT_ID, DataType.VARCHAR, max_length=MAX_PK_LEN)
    schema.add_field(F_PAGE, DataType.INT64)
    schema.add_field(
        F_TEXT, DataType.VARCHAR, max_length=_MAX_TEXT_FIELD,
        enable_analyzer=True, analyzer_params=profile.analyzer_params(),
    )
    schema.add_field(F_DENSE, DataType.FLOAT_VECTOR, dim=dense_dim)
    schema.add_field(F_SPARSE, DataType.SPARSE_FLOAT_VECTOR)
    # Nullable: documents ingested with classification off simply leave them NULL.
    schema.add_field(F_MAIN_CATEGORY, DataType.VARCHAR, max_length=MAX_CATEGORY_NAME_LEN, nullable=True)
    schema.add_field(
        F_SUB_CATEGORIES, DataType.ARRAY, element_type=DataType.VARCHAR,
        max_capacity=MAX_CATEGORIES, max_length=MAX_CATEGORY_NAME_LEN, nullable=True,
    )
    schema.add_function(Function(
        name=_BM25_FUNCTION, function_type=FunctionType.BM25,
        input_field_names=[F_TEXT], output_field_names=[F_SPARSE],
    ))

    index_params = client.prepare_index_params()
    index_params.add_index(
        F_DENSE, index_type=dense_index["index_type"], metric_type=dense_index["metric_type"],
        params=dense_index.get("params") or {},
    )
    index_params.add_index(F_SPARSE, **_SPARSE_INDEX)
    index_params.add_index(F_MAIN_CATEGORY, **_SCALAR_INDEX)
    index_params.add_index(F_SUB_CATEGORIES, **_SCALAR_INDEX)

    manifest = CollectionManifest(language=profile.language, embedding_model=profile.embedding_model, dense_dim=dense_dim)
    client.create_collection(
        name, schema=schema, index_params=index_params,
        consistency_level="Bounded", properties=manifest.to_property(),
    )
    return manifest


def describe_layout(client: MilvusClient, name: str, *, fallback_metric: str, logger: logging.LoggerAdapter) -> Optional[CollectionLayout]:
    """The layout of `name`, or None when it does not exist."""
    if not client.has_collection(name):
        return None
    described = client.describe_collection(name)
    output_fields = [f["name"] for f in described["fields"] if DataType(f["type"]) not in _VECTOR_DTYPES]
    if described.get("enable_dynamic_field"):
        output_fields.append(_F_DYNAMIC)
    return CollectionLayout(
        manifest=CollectionManifest.from_properties(described.get("properties") or {}),
        output_fields=output_fields,
        dense_metric=_dense_metric(client, name, fallback_metric, logger),
    )


def _dense_metric(client: MilvusClient, name: str, fallback: str, logger: logging.LoggerAdapter) -> str:
    """
    The metric the live dense index was built with. Searching with another one fails
    with "metric type not match", so the index wins over the configuration.
    """
    try:
        names = client.list_indexes(name, field_name=F_DENSE)
        actual = str(client.describe_index(name, names[0]).get("metric_type") or fallback).upper() if names else fallback
    except Exception as exc:
        logger.debug("No dense index to read the metric from on '%s': %s", name, exc)
        return fallback
    if actual != fallback:
        logger.warning("Collection '%s' dense index uses %s but config says %s — searching with %s.", name, actual, fallback, actual)
    return actual


def warm_up_japanese_analyzer(client: MilvusClient) -> None:
    """
    Tokenize a sample with Lindera so Milvus fetches and builds its dictionary now,
    not during the first Japanese ingestion. Raises when Lindera is unavailable (Milvus < 2.6).
    """
    client.run_analyzer(["刈刃取付用ナット"], analyzer_params=dict(JAPANESE_ANALYZER))
