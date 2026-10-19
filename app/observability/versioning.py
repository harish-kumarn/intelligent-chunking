"""
app.observability.versioning
============================
How a run names the code and configuration that produced it.

    pipeline_version   : `PIPELINE_VERSION` when set, else the service version — the Langfuse `version` of
                         every trace, so an evaluation can group runs by the pipeline that made them
    config_fingerprint : sha256[:12] over an explicit mapping of behaviour-affecting settings plus the
                         prompt files — two runs with the same fingerprint were produced by the same
                         models, prompts and tuning. The mapping is built by the caller from an allowlist
                         (`ChunkerConfig.fingerprint_fields`), never by dumping a settings object, so a
                         secret cannot end up in the digest input.

Pure functions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

__all__ = ["config_fingerprint", "file_fingerprint", "resolve_pipeline_version"]


def resolve_pipeline_version(explicit: Optional[str], service_version: str) -> str:
    value = (explicit or "").strip()
    return value or service_version


def file_fingerprint(path: Path) -> str:
    """sha256[:16] of a file's bytes (prompt files)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def config_fingerprint(fields: Mapping[str, Any], prompt_files: Iterable[Path] = ()) -> str:
    """Deterministic digest of `fields` (sorted keys, JSON) and the named prompt files (by relative name + content)."""
    prompts = {}
    for path in sorted(Path(p) for p in prompt_files):
        prompts[f"{path.parent.parent.name}/{path.parent.name}/{path.name}"] = file_fingerprint(path)
    payload = json.dumps({"fields": dict(fields), "prompts": prompts}, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
