"""
app.groundtruth.autogrounding
=============================
AutoGrounder — generates direct + indirect Q&A evaluation pairs per page with Gemini.

    input   : merged_chunks.json (chunks grouped by first source page)
    outputs : groundtruth_dataset.json  {page: {direct_questions, indirect_questions}}
              groundtruth_dataset.csv   flat rows — the `retrieval_v1` dataset format (app.datasets.schema):
                                        case_id, document_id, page, type, tag, answer_chunk_id, question, answer

Main path
    run ─► group chunks by page ─► pending pages, `chunking_concurrency` at a time:
           prompt ─► generate (retried) ─► keep answers whose chunk ids exist ─► case ids ─► write JSON
        ─► CSV (page order)

Identity: every question gets `case_id = {document_id}:{sha256(page|type|question)[:12]}` — deterministic,
so a resumed or repeated run names the same question the same way; `answer_chunk_id`s the model invented
(not among the page's chunks) are dropped, and a question left without one is dropped and counted.

Failure policy: a page that fails every retry is recorded with empty question
lists and an `error` field so the run completes, and is generated again by the next run; the
JSON is rewritten (atomically) after every page, so an interrupted run resumes where it stopped. Every call has
`gemini_request_timeout`. Missing API key fails at construction.
Concurrency: pages are independent; a lock guards the results and the file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from google.genai import types as genai_types

from app.config.defaults import ChunkerConfig
from app.logging.setup import get_logger, in_context
from app.postprocessing.merger import MergedChunk, load_merged_chunks
from app.utils.fs import atomic_write_json, atomic_write_text
from app.utils.gemini import gemini_client
from app.utils.json_utils import repair_and_parse_json

__all__ = ["AutoGrounder", "CSV_FIELDS", "case_id_for"]

_PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "auto_grounding_prompt.md").read_text(encoding="utf-8")
_OUTPUT_NAME = "groundtruth_dataset.json"
CSV_FIELDS = ("case_id", "document_id", "page", "type", "tag", "answer_chunk_id", "question", "answer")
_QUESTION_KEYS = (("direct", "direct_questions"), ("indirect", "indirect_questions"))
_CHUNK_ID_SEPARATOR = ";"


def case_id_for(document_id: str, page: Any, q_type: str, question: str) -> str:
    """Deterministic row identity: the same question on the same page of the same document is the same case."""
    digest = hashlib.sha256(f"{page}|{q_type}|{(question or '').strip()}".encode("utf-8")).hexdigest()[:12]
    return f"{document_id}:{digest}"


class AutoGrounder:
    """Per-document Q&A generator; one Gemini client per instance."""

    def __init__(self, config: ChunkerConfig, logger: Optional[logging.LoggerAdapter] = None) -> None:
        self._cfg = config
        self._log = logger or get_logger(module="AUTOGROUNDING")
        if not config.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for auto-grounding.")
        self._client = gemini_client(config.google_api_key, config.gemini_request_timeout)

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def run(
        self,
        merged_chunks_path: Path,
        output_path: Optional[Path] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        *,
        document_id: Optional[str] = None,
    ) -> Path:
        """
        Generate (or resume) groundtruth for every page; returns the JSON path. `document_id` names the
        rows (`case_id`, `document_id` columns); it defaults to the merged-chunks folder name (the
        pipeline's workspace layout).
        """
        merged_chunks_path = Path(merged_chunks_path)
        document_id = document_id or merged_chunks_path.parent.name
        out_path = Path(output_path) if output_path else merged_chunks_path.parent / _OUTPUT_NAME
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pages = _group_by_page(load_merged_chunks(merged_chunks_path))
        results = self._resolve_existing(out_path)
        pending = [page for page in pages if not _is_grounded(results.get(str(page)))]
        self._log.info("Auto-grounding: %d page(s), %d pending", len(pages), len(pending))

        lock = threading.Lock()
        done = 0

        def ground(page_no: int) -> None:
            nonlocal done
            output = self._ground_page(page_no, pages[page_no])
            if not output.get("error"):
                output = _normalise_page(output, page_no, {str(c["chunk_id"]) for c in pages[page_no]}, document_id, self._log)
            with lock:
                results[str(page_no)] = output
                atomic_write_json(out_path, dict(sorted(results.items(), key=lambda item: int(item[0]))))
                done += 1
                if on_progress is not None:
                    on_progress(done, len(pending))

        with ThreadPoolExecutor(max_workers=self._cfg.chunking_concurrency, thread_name_prefix="grounding") as pool:
            for future in [pool.submit(in_context(ground), page_no) for page_no in pending]:
                future.result()  # _ground_page never raises for a model failure; an I/O error propagates

        csv_path = self.to_csv(out_path, document_id=document_id)
        self._log.info("Groundtruth written: %s, %s", out_path, csv_path)
        return out_path

    def to_csv(self, json_path: Path, csv_path: Optional[Path] = None, *, document_id: Optional[str] = None) -> Path:
        """
        Flatten a groundtruth JSON file into one `retrieval_v1` CSV row per question. Entries written before
        case ids existed get theirs computed here (deterministic), so older JSON converts the same way.
        """
        json_path = Path(json_path)
        csv_path = Path(csv_path) if csv_path else json_path.with_suffix(".csv")
        document_id = document_id or json_path.parent.name
        data = json.loads(json_path.read_text(encoding="utf-8"))

        rows = [
            {
                "case_id": qa.get("case_id") or case_id_for(document_id, page, q_type, qa.get("question", "")),
                "document_id": document_id,
                "page": page,
                "type": q_type,
                "tag": qa.get("tag", ""),
                "answer_chunk_id": _CHUNK_ID_SEPARATOR.join(_as_list(qa.get("answer_chunk_id"))),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
            }
            for page, page_data in data.items()
            for q_type, key in _QUESTION_KEYS
            for qa in page_data.get(key, [])
            if isinstance(qa, dict)
        ]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(csv_path, buffer.getvalue())
        return csv_path

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _resolve_existing(self, out_path: Path) -> Dict[str, Any]:
        if not out_path.exists() or out_path.stat().st_size == 0:
            return {}
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            self._log.warning("Ignoring unreadable groundtruth file %s: %s", out_path, exc)
            return {}
        return existing if isinstance(existing, dict) else {}

    def _ground_page(self, page_no: int, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retry up to `autogrounding_max_retries`; on exhaustion return an empty, error-tagged page."""
        prompt = self._render_prompt(chunks)
        retries = self._cfg.autogrounding_max_retries
        for attempt in range(1, retries + 1):
            try:
                output = self._generate(prompt)
            except Exception as exc:
                self._log.warning("Page %d attempt %d/%d failed: %s", page_no, attempt, retries, exc)
                if attempt == retries:
                    return {"page_no": page_no, "direct_questions": [], "indirect_questions": [], "error": str(exc)}
                time.sleep(2 ** (attempt - 1))
                continue
            self._log.info(
                "Page %d grounded: %d direct, %d indirect",
                page_no, len(output.get("direct_questions", [])), len(output.get("indirect_questions", [])),
            )
            return output
        raise RuntimeError(f"_ground_page exhausted retries for page {page_no}")  # unreachable

    def _generate(self, prompt: str) -> Dict[str, Any]:
        response = self._client.models.generate_content(
            model=self._cfg.autogrounding_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=self._cfg.autogrounding_temperature,
                response_mime_type="application/json",
            ),
        )
        return repair_and_parse_json(response.text or "")

    def _render_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        cfg = self._cfg
        return _PROMPT_TEMPLATE.format(
            CHUNKS_JSON=json.dumps(chunks, indent=2),
            MIN_DIRECT=cfg.autogrounding_min_direct,
            MAX_DIRECT=cfg.autogrounding_max_direct,
            MIN_INDIRECT=cfg.autogrounding_min_indirect,
            MAX_INDIRECT=cfg.autogrounding_max_indirect,
        )


def _group_by_page(chunks: List[MergedChunk]) -> Dict[int, List[Dict[str, Any]]]:
    """Group by first source page (page 0 collects chunks without one), sorted by page."""
    pages: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        source_pages = (chunk.get("metadata") or {}).get("source_page_nos") or []
        first = str(source_pages[0]) if source_pages else ""
        pages[int(first) if first.isdigit() else 0].append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("content", ""),
                "chunk_description": chunk.get("chunk_description", ""),
            }
        )
    return dict(sorted(pages.items()))


def _normalise_page(
    output: Dict[str, Any], page_no: int, chunk_ids: set, document_id: str, log: logging.LoggerAdapter,
) -> Dict[str, Any]:
    """Keep only answer chunk ids the page really has; drop questions left without one; assign case ids."""
    dropped_ids = dropped_questions = 0
    normalised: Dict[str, Any] = {key: value for key, value in output.items() if key not in dict(_QUESTION_KEYS).values()}
    for q_type, key in _QUESTION_KEYS:
        kept: List[Dict[str, Any]] = []
        for qa in output.get(key) or []:
            if not isinstance(qa, dict) or not str(qa.get("question") or "").strip():
                dropped_questions += 1
                continue
            ids = [chunk_id for chunk_id in dict.fromkeys(_as_list(qa.get("answer_chunk_id"))) if chunk_id in chunk_ids]
            dropped_ids += len(_as_list(qa.get("answer_chunk_id"))) - len(ids)
            if not ids:
                dropped_questions += 1
                continue
            kept.append({**qa, "answer_chunk_id": ids, "case_id": case_id_for(document_id, page_no, q_type, qa.get("question", ""))})
        normalised[key] = kept
    if dropped_ids or dropped_questions:
        log.warning(
            "Page %d grounding: dropped %d unknown chunk id(s) and %d question(s) without a valid answer chunk",
            page_no, dropped_ids, dropped_questions,
        )
    return normalised


def _is_grounded(entry: Any) -> bool:
    """A stored page result that needs no new attempt (absent or error-tagged ones do)."""
    return isinstance(entry, dict) and not entry.get("error")


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]
