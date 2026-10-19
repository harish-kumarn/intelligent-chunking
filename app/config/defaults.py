"""
app.config.defaults
===================
ChunkerConfig — the single source of truth for pipeline and provider settings.

    api keys      : Gemini, Mistral, Groq, Cohere, Azure (env / .env)
    ocr           : which engine stage 1 runs, its per-engine tuning, page windows for long PDFs, retries
    throughput    : Gemini request timeout, parallel pages, classification batch size
    model tuning  : chunking, auto-grounding, vocabulary, reformulation, embedding (per language)
    classification: stage-5 switch, general category taxonomy, Gemini tuning
    language      : Japanese/English detection threshold, Lindera warm-up
    milvus        : connection, auth, dense index definition
    workspace     : where per-document artifacts are written

Priority (highest → lowest): environment variables → `.env` file → `config.yaml` → field defaults.
The `config.yaml` layer is loaded by `app.config.settings` and supplies the OCR and classification defaults below.
Instances are cheap and immutable in practice: the server builds one at startup and
derives per-request copies with `model_copy(update=...)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.settings import OCR_TECHNIQUES, classification_config, ocr_config, ocr_engine_config
from app.utils.naming import MAX_CATEGORIES, normalize_category_name

__all__ = ["ChunkerConfig"]

_PAGE_IMAGES_DIR = "page_images"  # PyMuPDF renders consumed by the vision chunker
_OCR_IMAGES_DIR = "pdf_images"    # figures the OCR stage extracted from the PDF

# Behaviour-affecting, non-secret fields that make up the configuration fingerprint stamped on traces.
_FINGERPRINT_FIELDS = (
    "ocr_technique", "azure_ocr_model", "azure_ocr_extract_figures", "azure_ocr_figure_dpi", "mistral_ocr_model",
    "ocr_window_pages", "ocr_window_max_bytes",
    "chunking_model", "chunking_temperature",
    "classification_enabled", "classification_allow_new_categories", "classification_categories", "classification_model",
    "classification_temperature", "classification_thinking_level", "classification_batch_size", "classification_batch_max_chars",
    "autogrounding_model", "autogrounding_temperature", "autogrounding_min_direct", "autogrounding_max_direct",
    "autogrounding_min_indirect", "autogrounding_max_indirect",
    "vocab_model", "vocab_temperature", "vocab_semantic_dedup", "requery_model", "requery_temperature",
    "embedding_model", "reranking_model", "japanese_embedding_model", "japanese_reranking_model", "japanese_detection_threshold",
    "milvus_index_type", "milvus_metric_type", "milvus_index_params", "milvus_search_params",
)


class ChunkerConfig(BaseSettings):
    """Pipeline + provider configuration, populated from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,  # `ChunkerConfig(resources_base=...)` works too, not only the alias
    )

    # ------------------------------------------------------------------ #
    # API keys
    # ------------------------------------------------------------------ #
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")
    azure_ocr_endpoint: str = Field(default="", alias="AZURE_OCR_ENDPOINT")
    azure_ocr_key: str = Field(default="", alias="AZURE_OCR_KEY")

    # ------------------------------------------------------------------ #
    # OCR — stage 1 engine switch (defaults come from `config.yaml`)
    # ------------------------------------------------------------------ #
    ocr_technique: str = Field(default_factory=lambda: ocr_config()["technique"], alias="OCR_TECHNIQUE")

    # Azure AI Document Intelligence
    azure_ocr_model: str = Field(default_factory=lambda: ocr_engine_config("azure", "model"), alias="AZURE_OCR_MODEL")
    azure_ocr_extract_figures: bool = Field(
        default_factory=lambda: ocr_engine_config("azure", "extract_figures"), alias="AZURE_OCR_EXTRACT_FIGURES"
    )
    azure_ocr_figure_dpi: int = Field(
        default_factory=lambda: ocr_engine_config("azure", "figure_dpi"), alias="AZURE_OCR_FIGURE_DPI"
    )

    # Mistral OCR
    mistral_ocr_model: str = Field(default_factory=lambda: ocr_engine_config("mistral", "model"), alias="MISTRAL_OCR_MODEL")

    @field_validator("ocr_technique", mode="before")
    @classmethod
    def _known_technique(cls, value: object) -> str:
        """Case/whitespace tolerant — a typo fails here, not three minutes into an upload."""
        technique = str(value).strip().lower()
        if technique not in OCR_TECHNIQUES:
            raise ValueError(f"ocr_technique must be one of {list(OCR_TECHNIQUES)}, got {value!r}")
        return technique

    # Long PDFs are OCR'd in page windows: Mistral caps one request at 1,000 pages / 50 MB, and a
    # window that fails is the only part redone. A window over the byte cap is split in half.
    ocr_window_pages: int = Field(default=100, ge=1)
    ocr_window_max_bytes: int = Field(default=45 * 1024 * 1024, ge=1024 * 1024)
    ocr_concurrency: int = Field(default=2, ge=1, le=8)
    # Each OCR call (whole PDF or one window) is retried on transient errors with exponential back-off.
    ocr_max_retries: int = Field(default=4, ge=1, le=10)
    ocr_retry_delay: float = Field(default=10.0, ge=0)         # seconds; base of the back-off
    ocr_request_timeout: float = Field(default=900.0, gt=0)    # seconds per OCR call (a 100-page window takes minutes)

    # ------------------------------------------------------------------ #
    # Throughput (every Gemini stage)
    # ------------------------------------------------------------------ #
    gemini_request_timeout: float = Field(default=180.0, gt=0)   # seconds per call; a hung call is retried, not waited on
    # Pages chunked (and auto-grounded) at once. Pages are independent (neighbour context is OCR text),
    # so this divides stage-3 time; bounded by the Gemini quota (429s back off and retry).
    chunking_concurrency: int = Field(default=4, ge=1, le=32, alias="CHUNKING_CONCURRENCY")

    # ------------------------------------------------------------------ #
    # Vision chunking (Gemini)
    # ------------------------------------------------------------------ #
    chunking_model: str = "gemini-3.5-flash"
    chunking_max_retries: int = 5
    chunking_retry_delay: float = 2.0    # seconds; base of the exponential back-off
    chunking_timeout: float = 30.0       # seconds; cap on a single back-off wait
    chunking_temperature: float = 1.0    # required by the Gemini 3 series
    chunking_cache_ttl: str = "43200s"   # 12 h: a whole large document fits one cache life (it is also extended between pages)

    # ------------------------------------------------------------------ #
    # Auto-grounding (Gemini)
    # ------------------------------------------------------------------ #
    autogrounding_model: str = "gemini-3.5-flash"
    autogrounding_temperature: float = 0.2
    autogrounding_max_retries: int = 3
    autogrounding_min_direct: int = 5
    autogrounding_max_direct: int = 8
    autogrounding_min_indirect: int = 5
    autogrounding_max_indirect: int = 8

    # ------------------------------------------------------------------ #
    # Chunk classification (Gemini) — switch and taxonomy come from `config.yaml`
    # ------------------------------------------------------------------ #
    classification_enabled: bool = Field(default_factory=lambda: classification_config()["enabled"])
    classification_allow_new_categories: bool = Field(
        default_factory=lambda: classification_config()["allow_new_categories"]
    )
    # General taxonomy, name → definition (definitions are shown to the model; "" is allowed).
    classification_categories: Dict[str, str] = Field(default_factory=lambda: classification_config()["categories"])
    classification_model: str = "gemini-3.5-flash"
    classification_temperature: float = 1.0   # required by the Gemini 3 series
    # Gemini 3 thinking budget; None = model default. "low" halved latency (≈5.2 s → 2.7 s per chunk) with the
    # same main categories on the BCL200S manual; "minimal" was faster but changed main categories.
    classification_thinking_level: Optional[str] = "low"
    classification_max_retries: int = 3       # per chunk; invalid output counts as a failed attempt
    classification_retry_delay: float = 2.0   # seconds; base of the exponential back-off
    # Chunks per Gemini call, still in document order so later batches see categories created earlier;
    # 1 = the original one-call-per-chunk path. A batch also stops at `classification_batch_max_chars`.
    classification_batch_size: int = Field(default=10, ge=1, le=50, alias="CLASSIFICATION_BATCH_SIZE")
    classification_batch_max_chars: int = Field(default=60_000, ge=1_000)

    @field_validator("classification_categories", mode="before")
    @classmethod
    def _valid_categories(cls, value: Any) -> Dict[str, str]:
        """Accept a mapping or a plain list of names; normalise names, keep order, reject duplicates."""
        items = value.items() if isinstance(value, dict) else ((name, "") for name in (value or []))
        categories: Dict[str, str] = {}
        for name, description in items:
            key = normalize_category_name(str(name))
            if key in categories:
                raise ValueError(f"classification_categories lists {key!r} twice")
            categories[key] = " ".join(str(description or "").split())
        if not 0 < len(categories) <= MAX_CATEGORIES:
            raise ValueError(f"classification_categories needs 1–{MAX_CATEGORIES} categories, got {len(categories)}")
        return categories

    # ------------------------------------------------------------------ #
    # Vocabulary index (Groq)
    # ------------------------------------------------------------------ #
    vocab_model: str = "openai/gpt-oss-120b"
    vocab_temperature: float = 0.0
    vocab_semantic_dedup: bool = False   # extra LLM pass merging synonyms/plurals; costs one Groq call

    # ------------------------------------------------------------------ #
    # Query reformulation (Groq)
    # ------------------------------------------------------------------ #
    requery_model: str = "openai/gpt-oss-20b"
    requery_temperature: float = 0.0

    # ------------------------------------------------------------------ #
    # Embedding & reranking (Cohere)
    # ------------------------------------------------------------------ #
    # One pair per collection language (app.retrieval.profiles). A collection keeps the
    # embedding model it was built with (its manifest); changing these affects new collections.
    # English: embed-v4.0, not embed-english-v3.0 — v3's 512-token window silently truncated longer
    # chunks (the target is up to 1,000 tokens; fused chunks run longer). docs/HYPERPARAMETERS.md §1 #3.
    embedding_model: str = "embed-v4.0"                        # English collections
    reranking_model: str = "rerank-english-v3.0"
    # Japanese: chosen on the BCL200S benchmark (301 queries, hybrid + rerank nDCG@10 0.965
    # vs 0.830 for the English models) — see docs/JAPANESE_RETRIEVAL.md.
    japanese_embedding_model: str = "embed-v4.0"               # Japanese collections
    japanese_reranking_model: str = "rerank-v3.5"
    embedding_batch_size: int = 95       # Cohere embed accepts at most 96 texts per call
    # Stage 8 embeds and upserts this many chunks at a time, reporting progress after each batch.
    index_batch_size: int = Field(default=256, ge=1, le=4096)

    # ------------------------------------------------------------------ #
    # Document language (app.retrieval.language)
    # ------------------------------------------------------------------ #
    # A PDF is Japanese when Japanese script is at least this share of its letters.
    japanese_detection_threshold: float = Field(default=0.2, gt=0.0, lt=1.0)
    # Fetch Milvus' Lindera dictionary at service start instead of on the first Japanese ingest.
    warm_up_japanese_analyzer: bool = True

    # ------------------------------------------------------------------ #
    # Milvus
    # ------------------------------------------------------------------ #
    milvus_host: str = Field(default="127.0.0.1", alias="MILVUS_HOST")
    milvus_port: str = Field(default="19530", alias="MILVUS_PORT")
    milvus_db_name: str = "default"
    milvus_timeout: Optional[float] = None
    milvus_secure: bool = False
    milvus_user: Optional[str] = Field(default=None, alias="MILVUS_USER")
    milvus_password: Optional[str] = Field(default=None, alias="MILVUS_PASSWORD")
    milvus_token: Optional[str] = Field(default=None, alias="MILVUS_TOKEN")
    milvus_index_type: str = "FLAT"
    milvus_metric_type: str = "COSINE"
    milvus_index_params: Optional[dict] = None   # full dense index spec; overrides type/metric above
    milvus_search_params: Optional[dict] = None  # extra dense search params (e.g. {"params": {"ef": 64}})

    # ------------------------------------------------------------------ #
    # Workspace
    # ------------------------------------------------------------------ #
    resources_base: str = Field(default="ic_resources", alias="IC_RESOURCES_BASE")

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #

    def fingerprint_fields(self) -> Dict[str, Any]:
        """
        The settings that change what the pipeline produces — models, prompts' tuning, OCR engine, fusion
        defaults — as a plain mapping for `app.observability.versioning.config_fingerprint`. An explicit
        allowlist: keys, hosts, paths, timeouts and retry budgets are left out (they change nothing in the
        output and must never reach a trace).
        """
        return {name: getattr(self, name) for name in _FINGERPRINT_FIELDS}

    # ------------------------------------------------------------------ #
    # Derived paths
    # ------------------------------------------------------------------ #

    def doc_folder(self, doc_name: str) -> Path:
        return Path(self.resources_base) / doc_name

    def page_images_folder(self, doc_name: str) -> Path:
        return self.doc_folder(doc_name) / _PAGE_IMAGES_DIR

    def ocr_images_folder(self, doc_name: str) -> Path:
        return self.doc_folder(doc_name) / _OCR_IMAGES_DIR
