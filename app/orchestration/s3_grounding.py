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
    dataset : optionally, the CSV is also published as an immutable `hybrid_retrieval` dataset version
              (`DatasetService.publish`: validated, reference-checked against this document, content-addressed —
              the same generated rows never make a second version)

Main path
    ground_document ─► load_published_chunks ─► seed previous output ─► AutoGrounder.run ─► add_version_artifacts
                    ─► [DatasetService.publish]

One `autogrounding` trace per call. The temporary directory is removed when the call returns; the S3 version
(and the dataset version) are the only places the output lives.

Failure policy: a missing folder / version / chunks raises `S3DocumentNotFoundError` (HTTP 404). A page whose
generation fails every retry is stored error-tagged (AutoGrounder) and reported in `failed_pages`, so a
rerun retries just those. A missing GOOGLE_API_KEY, S3 read or upload errors propagate (500). Dataset
publication errors propagate after the groundtruth artifacts were saved (the S3 version is already updated).
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.config.defaults import ChunkerConfig
from app.datasets.service import DatasetPublication, DatasetService
from app.groundtruth.autogrounding import AutoGrounder
from app.logging.setup import get_logger
from app.observability.modules import ModuleName
from app.observability.telemetry import get_telemetry
from app.orchestration.errors import InvalidDocumentError
from app.orchestration.s3_import import load_published_chunks
from app.postprocessing.merger import save_merged_chunks
from app.storage.s3_artifact_store import S3ArtifactStore
from app.storage.s3_dataset_store import DATASET_NAME_PATTERN, DATASET_NAME_RULE
from app.utils.fs import atomic_write_json

__all__ = ["S3DocumentGrounder", "S3GroundingResult"]

_GROUNDTRUTH_JSON = "groundtruth_dataset.json"
_DATASET_PURPOSE = "auto-generated retrieval ground truth (direct + indirect questions per page)"


@dataclass(frozen=True)
class S3GroundingResult:
    document_id: str
    version: str
    source_artifact: str          # merged_chunks.json, or target_chunks.json when it had to be merged here
    page_count: int               # pages grounded (chunks grouped by first source page)
    question_count: int           # direct + indirect, over every page
    failed_pages: List[int]       # pages that failed every retry; a rerun retries only these
    artifacts: List[str]          # S3 keys written
    dataset: Optional[DatasetPublication] = None   # when the CSV was published as a dataset version


class S3DocumentGrounder:
    def __init__(
        self,
        store: S3ArtifactStore,
        config: ChunkerConfig,
        logger: Optional[logging.LoggerAdapter] = None,
        dataset_service: Optional[DatasetService] = None,
    ) -> None:
        self._store = store
        self._cfg = config
        self._log = logger or get_logger(module="S3_GROUNDING")
        self._datasets = dataset_service

    def ground_document(
        self,
        document_id: str,
        *,
        publish_dataset: bool = False,
        dataset_name: Optional[str] = None,
        promote: bool = False,
        sync_langfuse: bool = False,
    ) -> S3GroundingResult:
        if publish_dataset:
            if self._datasets is None:
                raise InvalidDocumentError("publish_dataset requires a dataset bucket (S3_DATASET_BUCKET / S3_BUCKET)")
            dataset_name = dataset_name or _dataset_name_for(document_id)

        with get_telemetry().trace(
            ModuleName.AUTOGROUNDING, document_id=document_id, autogrounding_model=self._cfg.autogrounding_model,
            dataset_name=dataset_name if publish_dataset else None,
        ) as trace:
            version, merged, source = load_published_chunks(self._store, document_id, self._log)
            trace.set(version=version, source_artifact=source, chunk_count=len(merged))
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

                grounder.run(merged_path, output_path, document_id=document_id)
                dataset = json.loads(output_path.read_text(encoding="utf-8"))
                csv_path = output_path.with_suffix(".csv")
                keys = self._store.add_version_artifacts(document_id, version, [output_path, csv_path])
                publication = None
                if publish_dataset and self._datasets is not None:
                    publication = self._datasets.publish(
                        module=ModuleName.HYBRID_RETRIEVAL, dataset_name=dataset_name or _dataset_name_for(document_id), source=csv_path,
                        purpose=_DATASET_PURPOSE, promote=promote, verify_references=True, sync_langfuse=sync_langfuse,
                        generator={"kind": "autogrounding", "model": self._cfg.autogrounding_model, "document_version": version},
                        source_filename=csv_path.name,
                    )

            pages = {page: entry for page, entry in dataset.items() if isinstance(entry, dict)}
            result = S3GroundingResult(
                document_id=document_id,
                version=version,
                source_artifact=source,
                page_count=len(pages),
                question_count=sum(len(entry.get(key) or []) for entry in pages.values() for key in ("direct_questions", "indirect_questions")),
                failed_pages=sorted(int(page) for page, entry in pages.items() if entry.get("error") and str(page).isdigit()),
                artifacts=keys,
                dataset=publication,
            )
            trace.set_output(
                status="completed", page_count=result.page_count, question_count=result.question_count,
                failed_pages=result.failed_pages, dataset_version=publication.dataset_version if publication else None,
            )
        return result


def _dataset_name_for(document_id: str) -> str:
    """The document id as a dataset name; a 400 when it cannot be one (then `dataset_name` must be given)."""
    candidate = document_id.strip().lower()[:64]
    if not DATASET_NAME_PATTERN.match(candidate):
        raise InvalidDocumentError(f"dataset_name is required: '{document_id}' is not a valid dataset name ({DATASET_NAME_RULE})")
    return candidate
