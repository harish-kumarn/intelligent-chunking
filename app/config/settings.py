"""
app.config.settings
===================
ServerSettings — configuration for the HTTP layer only.

    artifact policy : debug (keep a workspace) vs production (per-document workspace, kept for a resume
                      until the document succeeds)
    S3 defaults     : bucket/region used when a request does not name its own
    limits          : upload size, log level, bind address

Provider and pipeline settings live in `ChunkerConfig`; the two never overlap.

This module also owns `config.yaml` — the non-secret, file-based half of the
configuration (the OCR engine switch and chunk classification). It is read once, cached, and
handed to `ChunkerConfig` as *defaults*, so the usual priority still holds:

    environment variable → `.env` → `config.yaml` → field default
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "OCR_TECHNIQUES", "ServerSettings", "classification_config", "load_yaml_config", "ocr_config", "ocr_engine_config",
]

_DEFAULT_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MiB — large scanned manuals fit, abuse does not
_DEFAULT_MAX_PDF_PAGES = 2000                  # a 1,000-page manual fits twice; bounds one job's cost
_JOBS_DIR = "_jobs"        # under the workspace root: one JSON record per ingestion job
_UPLOADS_DIR = "_uploads"  # under the workspace root: uploads streamed to disk before they are admitted

# --------------------------------------------------------------------------- #
# config.yaml
# --------------------------------------------------------------------------- #

_CONFIG_YAML_ENV = "CONFIG_YAML_PATH"
_CONFIG_YAML_DEFAULT = "config.yaml"

OCR_TECHNIQUES = ("azure", "mistral")

#: Used verbatim when `config.yaml` is absent or a key is missing, so the
#: service runs with no file at all and an unknown key can never crash startup.
_OCR_DEFAULTS: Dict[str, Any] = {
    "technique": "mistral",
    "azure": {"model": "prebuilt-layout", "extract_figures": True, "figure_dpi": 200},
    "mistral": {"model": "mistral-ocr-latest"},
}

_CLASSIFICATION_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "allow_new_categories": True,
    "categories": {
        "operation": "Using the product in normal conditions: starting, stopping, controls, settings, operating procedures "
                     "and handling during work.",
        "technical": "Specifications and how the product is built or works: ratings, dimensions, capacities, components, "
                     "materials, parts and diagrams.",
        "safety": "Hazards and their prevention: warnings, cautions, protective equipment, safe-use rules, prohibited actions "
                  "and emergency measures.",
        "maintenance": "Keeping the product in working order: inspection, cleaning, lubrication, adjustment, part replacement, "
                       "service intervals and storage.",
        "troubleshooting": "Diagnosing and fixing faults: symptoms, probable causes, error codes and corrective actions.",
    },
}


def _config_yaml_path() -> Path:
    """`CONFIG_YAML_PATH` if set, else `config.yaml` beside the repository root."""
    override = os.getenv(_CONFIG_YAML_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / _CONFIG_YAML_DEFAULT


@lru_cache(maxsize=1)
def load_yaml_config() -> Dict[str, Any]:
    """Parse `config.yaml` once. A missing or empty file is an empty mapping, never an error."""
    path = _config_yaml_path()
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def ocr_config() -> Dict[str, Any]:
    """The `ocr:` block, with `_OCR_DEFAULTS` filling anything the file leaves out."""
    block = load_yaml_config().get("ocr")
    if not isinstance(block, dict):
        return dict(_OCR_DEFAULTS)
    merged = dict(_OCR_DEFAULTS)
    merged.update({key: value for key, value in block.items() if key not in ("azure", "mistral")})
    for engine in ("azure", "mistral"):
        nested = block.get(engine)
        merged[engine] = {**_OCR_DEFAULTS[engine], **nested} if isinstance(nested, dict) else dict(_OCR_DEFAULTS[engine])
    return merged


def ocr_engine_config(engine: str, key: str) -> Any:
    """One tuning value for one engine, e.g. `ocr_engine_config("azure", "model")`."""
    return ocr_config()[engine].get(key, _OCR_DEFAULTS[engine].get(key))


def classification_config() -> Dict[str, Any]:
    """The `classification:` block, with `_CLASSIFICATION_DEFAULTS` filling anything the file leaves out."""
    block = load_yaml_config().get("classification")
    merged = dict(_CLASSIFICATION_DEFAULTS)
    if isinstance(block, dict):
        merged.update({key: value for key, value in block.items() if key in _CLASSIFICATION_DEFAULTS and value is not None})
    return merged


class ServerSettings(BaseSettings):
    """HTTP-layer configuration, read from environment variables / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "vision-rag"
    service_version: str = "0.1.0"
    environment: str = Field(default="production", alias="APP_ENV")

    artifact_mode: Literal["debug", "production"] = Field(default="debug", alias="ARTIFACT_MODE")
    debug_workspace: str = Field(default="debug", alias="DEBUG_WORKSPACE")
    # Production ingestion workspaces: kept when a run fails so the retry resumes (OCR windows, chunked
    # pages, classifications); removed on success; abandoned ones pruned after the retention. Mount a volume
    # here so a resume also survives a container restart.
    ingest_workspace_dir: str = Field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "visionrag_ingest"), alias="INGEST_WORKSPACE_DIR"
    )
    ingest_workspace_retention_hours: float = Field(default=72.0, gt=0, alias="INGEST_WORKSPACE_RETENTION_HOURS")
    # Finished ingestion jobs stay readable at GET /v1/ingestion-jobs/{id} this long (in memory).
    ingest_job_retention_hours: float = Field(default=24.0, gt=0, alias="INGEST_JOB_RETENTION_HOURS")
    # A job whose run fails with a retryable error is run again (resuming from its workspace) up to this
    # many attempts in total, waiting `ingest_job_retry_delay_seconds` × attempt in between.
    ingest_job_max_attempts: int = Field(default=3, ge=1, le=10, alias="INGEST_JOB_MAX_ATTEMPTS")
    ingest_job_retry_delay_seconds: float = Field(default=60.0, ge=0, alias="INGEST_JOB_RETRY_DELAY_SECONDS")
    # At startup, continue the job a restart interrupted (its record and workspace are on disk).
    ingest_job_auto_resume: bool = Field(default=True, alias="INGEST_JOB_AUTO_RESUME")

    s3_upload_enabled: bool = Field(default=False, alias="S3_UPLOAD_ENABLED")
    s3_bucket: Optional[str] = Field(default=None, alias="S3_BUCKET")
    AWS_DEFAULT_REGION: Optional[str] = Field(default=None, alias="AWS_DEFAULT_REGION")

    max_upload_size_bytes: int = Field(default=_DEFAULT_MAX_UPLOAD_BYTES, alias="MAX_UPLOAD_SIZE_BYTES")
    max_pdf_pages: int = Field(default=_DEFAULT_MAX_PDF_PAGES, ge=1, alias="MAX_PDF_PAGES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    port: int = Field(default=8000, alias="SERVER_PORT")

    @property
    def debug(self) -> bool:
        return self.artifact_mode == "debug"

    @property
    def workspace_root(self) -> Path:
        """Where per-document workspaces live: `DEBUG_WORKSPACE` in debug mode, else `INGEST_WORKSPACE_DIR`."""
        return Path(self.debug_workspace if self.debug else self.ingest_workspace_dir)

    @property
    def jobs_dir(self) -> Path:
        return self.workspace_root / _JOBS_DIR

    @property
    def uploads_dir(self) -> Path:
        return self.workspace_root / _UPLOADS_DIR

    @property
    def log_dir(self) -> str:
        """Directory for the rotating log file (debug environments only)."""
        if self.debug:
            return os.path.join(self.debug_workspace, "logs")
        return os.path.join(tempfile.gettempdir(), "visionrag_logs")
