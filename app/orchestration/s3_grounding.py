"""
app.orchestration.s3_grounding
==============================
S3DocumentGrounder — run auto-grounding (stage 7) alone on a document already published to S3, and save
`groundtruth_dataset.json` / `.csv` into the same S3 version, listed in its manifest.

    resolve : load_published_chunks — `current.json` version, its merged chunks (target fallback)
    resume  : the version's existing groundtruth_dataset.json is the starting point, so pages already
              grounded are not generated again (error-tagged pages are retried)
    ground  : AutoGrounder over a temporary copy of the merged chunks
    save    : S3ArtifactStore.add_version_artifacts (manifest rewritten after the upload)

Main path
    ground_document ─► load_published_chunks ─► seed previous output ─► AutoGrounder.run ─► add_version_artifacts

The temporary directory is removed when the call returns; the S3 version is the only place the output lives.

Failure policy: a missing folder / version / chunks raises `S3DocumentNotFoundError` (HTTP 404). A page whose
generation fails every retry is stored error-tagged (AutoGrounder) and reported in `failed_pages`, so a
rerun retries just those. A missing GOOGLE_API_KEY, S3 read or upload errors propagate (500).
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.config.defaults import ChunkerConfig
from app.groundtruth.autogrounding import AutoGrounder
from app.logging.setup import get_logger
from app.orchestration.s3_import import load_published_chunks
from app.postprocessing.merger import save_merged_chunks
from app.storage.s3_artifact_store import S3ArtifactStore
from app.utils.fs import atomic_write_json

__all__ = ["S3DocumentGrounder", "S3GroundingResult"]

_GROUNDTRUTH_JSON = "groundtruth_dataset.json"


@dataclass(frozen=True)
class S3GroundingResult:
    document_id: str
    version: str
    source_artifact: str          # merged_chunks.json, or target_chunks.json when it had to be merged here
    page_count: int               # pages grounded (chunks grouped by first source page)
    question_count: int           # direct + indirect, over every page
    failed_pages: List[int]       # pages that failed every retry; a rerun retries only these
    artifacts: List[str]          # S3 keys written


class S3DocumentGrounder:
    def __init__(self, store: S3ArtifactStore, config: ChunkerConfig, logger: Optional[logging.LoggerAdapter] = None) -> None:
        self._store = store
        self._cfg = config
        self._log = logger or get_logger(module="S3_GROUNDING")

    def ground_document(self, document_id: str) -> S3GroundingResult:
        version, merged, source = load_published_chunks(self._store, document_id, self._log)
        grounder = AutoGrounder(self._cfg, self._log)  # fails fast without GOOGLE_API_KEY
        self._log.info("S3 grounding: document=%s version=%s source=%s chunks=%d", document_id, version, source, len(merged))

        with tempfile.TemporaryDirectory(prefix="s3_grounding_") as workdir:
            merged_path = Path(workdir) / "merged_chunks.json"
            output_path = Path(workdir) / _GROUNDTRUTH_JSON
            save_merged_chunks(merged_path, merged)
            previous = self._store.get_version_resource_json(document_id, version, _GROUNDTRUTH_JSON)
            if isinstance(previous, dict) and previous:
                atomic_write_json(output_path, previous)
                self._log.info("S3 grounding: resuming from the version's existing %s (%d page(s))", _GROUNDTRUTH_JSON, len(previous))

            grounder.run(merged_path, output_path)
            dataset = json.loads(output_path.read_text(encoding="utf-8"))
            keys = self._store.add_version_artifacts(document_id, version, [output_path, output_path.with_suffix(".csv")])

        pages = {page: entry for page, entry in dataset.items() if isinstance(entry, dict)}
        return S3GroundingResult(
            document_id=document_id,
            version=version,
            source_artifact=source,
            page_count=len(pages),
            question_count=sum(len(entry.get(key) or []) for entry in pages.values() for key in ("direct_questions", "indirect_questions")),
            failed_pages=sorted(int(page) for page, entry in pages.items() if entry.get("error") and str(page).isdigit()),
            artifacts=keys,
        )
