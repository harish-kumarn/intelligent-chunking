"""
app.groundtruth.autogrounding
=============================
(single file: generation, validation and metrics)
AutoGrounder — generates direct + indirect Q&A evaluation pairs per page with Gemini, validates them
against the source chunks, and scores the result.

    input   : merged_chunks.json (chunks grouped by first source page)
    outputs : groundtruth_dataset.json  {page: {direct_questions, indirect_questions}}
                                        (each question carries `evidence` and `content` for the metrics)
              groundtruth_dataset.csv   flat rows — the `retrieval_v1` dataset format (app.datasets.schema):
                                        case_id, document_id, page, type, tag, answer_chunk_id, question, answer
              ground_truth_metrics.json / .csv   HCPR, EVR, DCC, Indirect Overlap, MHR, Tag Consistency

Main path
    run ─► group chunks by page ─► pending pages, `chunking_concurrency` at a time:
           prompt ─► generate ─► validate (ids exist, evidence verbatim, tag matches citations; retried with
           the validation error as feedback) ─► content injected from the source chunks ─► case ids ─► write JSON
        ─► CSV (page order) ─► metrics

Identity: every question gets `case_id = {document_id}:{sha256(page|type|question)[:12]}` — deterministic,
so a resumed or repeated run names the same question the same way.

Failure policy: an error no page can survive (invalid key, unknown model) propagates and stops the stage. A page
that fails every retry (including validation) is recorded with empty question lists and an `error` field so the
run completes, and is generated again by the next run. The JSON is rewritten atomically after every page, so an
interrupted run resumes where it stopped. Metrics are best effort: a failure there is logged, never fatal.

Concurrency: pages are independent; a lock guards the results and the file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from google.genai import types as genai_types

from app.app_logging.setup import get_logger, in_context
from app.chunking.chunker import is_fatal_gemini_error
from app.config.defaults import ChunkerConfig
from app.postprocessing.merger import MergedChunk, load_merged_chunks
from app.utils.fs import atomic_write_json, atomic_write_text
from app.utils.gemini import gemini_client
from app.utils.json_utils import repair_and_parse_json

__all__ = ["AutoGrounder", "CSV_FIELDS", "METRIC_DEFINITIONS", "case_id_for", "compute_metrics", "write_metrics"]

_PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "auto_grounding_prompt.md").read_text(encoding="utf-8")

_OUTPUT_NAME = "groundtruth_dataset.json"
_METRICS_NAME = "ground_truth_metrics.json"

CSV_FIELDS = ("case_id", "document_id", "page", "type", "tag", "answer_chunk_id", "question", "answer")

_QUESTION_KEYS = (("direct", "direct_questions"), ("indirect", "indirect_questions"))
_CHUNK_ID_SEPARATOR = ";"

TAG_SINGLE = "single"
TAG_MULTI_HOP = "multi_hop"

_RETRY_FEEDBACK = (
    "\n\nYour previous output was rejected by validation:\n{error}\n"
    "Return the COMPLETE corrected JSON object. Every QA pair must include an 'evidence' field: a single verbatim "
    "string for 'single' questions, or a list of verbatim strings (one per answer_chunk_id, in the same order) "
    "for 'multi_hop' questions. Do not put chunk_ids inside evidence."
)


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
        rows (`case_id`, `document_id` columns); it defaults to the merged-chunks folder name.
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
            output = self._ground_page(page_no, pages[page_no], document_id)
            with lock:
                results[str(page_no)] = output
                atomic_write_json(out_path, dict(sorted(results.items(), key=lambda item: int(item[0]))))
                done += 1
                if on_progress is not None:
                    on_progress(done, len(pending))

        with ThreadPoolExecutor(max_workers=self._cfg.chunking_concurrency, thread_name_prefix="grounding") as pool:
            futures = [pool.submit(in_context(ground), page_no) for page_no in pending]
            try:
                for future in futures:
                    future.result()  # a fatal Gemini error or an I/O error propagates; model failures do not
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

        csv_path = self.to_csv(out_path, document_id=document_id)
        self._log.info("Groundtruth written: %s, %s", out_path, csv_path)
        self._write_metrics(results, pages, out_path.parent / _METRICS_NAME)
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

    def _ground_page(self, page_no: int, chunks: List[Dict[str, Any]], document_id: str) -> Dict[str, Any]:
        """
        Generate + validate one page, retrying with the validation error as feedback. Fatal Gemini errors
        propagate; on exhaustion the page is an empty, error-tagged fallback (regenerated by the next run).
        """
        retries = self._cfg.autogrounding_max_retries
        base_prompt = prompt = self._render_prompt(chunks)
        chunk_map = {str(c["chunk_id"]): c for c in chunks if c.get("chunk_id") is not None}
        last_error: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                output = _validate_output(self._generate(prompt), chunk_map)
            except Exception as exc:
                if is_fatal_gemini_error(exc):
                    raise
                last_error = exc
                self._log.warning("Page %d attempt %d/%d failed: %s", page_no, attempt, retries, exc)
                if attempt == retries:
                    break
                prompt = base_prompt + _RETRY_FEEDBACK.format(error=exc)
                time.sleep(2 ** (attempt - 1))
                continue
            output = _stamp_case_ids(output, page_no, document_id)
            self._log.info(
                "Page %d grounded: %d direct, %d indirect",
                page_no, len(output["direct_questions"]), len(output["indirect_questions"]),
            )
            return output

        self._log.error("Page %d: all %d attempts failed (%s); saved as an empty page", page_no, retries, last_error)
        return {"page_no": page_no, "direct_questions": [], "indirect_questions": [], "error": str(last_error)}

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
            CHUNKS_JSON=json.dumps(chunks, indent=2, ensure_ascii=False),
            MIN_DIRECT=cfg.autogrounding_min_direct,
            MAX_DIRECT=cfg.autogrounding_max_direct,
            MIN_INDIRECT=cfg.autogrounding_min_indirect,
            MAX_INDIRECT=cfg.autogrounding_max_indirect,
        )

    def _write_metrics(self, results: Dict[str, Any], pages: Dict[int, List[Dict[str, Any]]], path: Path) -> None:
        """Best effort: the DCC universe is every merged chunk (this repo writes no bm25_data.json)."""
        try:
            chunk_text, page_chunk_ids = index_chunks(pages)
            registered = [cid for ids in page_chunk_ids.values() for cid in ids]
            metrics = compute_metrics(
                results=results,
                page_chunk_ids=page_chunk_ids,
                chunk_text=chunk_text,
                registered_ids=registered,
                overlap_threshold=getattr(self._cfg, "autogrounding_overlap_threshold", 0.80),
            )
            write_metrics(metrics, str(path))
        except Exception as exc:
            self._log.warning("Groundtruth metrics skipped: %s: %s", type(exc).__name__, exc)
            return
        for name, value in metrics["overall"]["metrics"].items():
            self._log.info("Groundtruth %-16s %.2f%%", name, value)


# --------------------------------------------------------------------------- #
# Chunks, validation, identity
# --------------------------------------------------------------------------- #

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


def _validate_output(output: Any, chunk_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Raise ValueError unless every question is well formed; inject `evidence` (normalised) and `content`."""
    if not isinstance(output, dict):
        raise ValueError("Model output must be a JSON object")
    for _, key in _QUESTION_KEYS:
        questions = output.get(key)
        if not isinstance(questions, list):
            raise ValueError(f"Missing or invalid section: {key}")
        for qa in questions:
            if not isinstance(qa, dict):
                raise ValueError(f"Each item in {key} must be an object")
            _validate_question(qa, chunk_map)
    return output


def _validate_question(qa: Dict[str, Any], chunk_map: Dict[str, Dict[str, Any]]) -> None:
    question = qa.get("question")
    if not str(question or "").strip():
        raise ValueError("A question item has no question text")

    raw_ids = qa.get("answer_chunk_id")
    if isinstance(raw_ids, (str, int)):
        ids = [str(raw_ids)]
    elif isinstance(raw_ids, list):
        ids = [str(x) for x in raw_ids]
    else:
        raise ValueError(f"answer_chunk_id must be a list or scalar: {question}")
    if not ids:
        raise ValueError(f"Missing answer_chunk_id: {question}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate answer_chunk_id values: {question}")

    evidence = qa.get("evidence")
    if isinstance(evidence, str):
        excerpts = [evidence]
    elif isinstance(evidence, list) and all(isinstance(e, str) for e in evidence):
        excerpts = evidence
    else:
        raise ValueError(f"Evidence must be a string (single) or a list of strings (multi_hop): {question}")
    if len(excerpts) != len(ids):
        raise ValueError(f"Evidence must have exactly one excerpt per answer_chunk_id, in the same order: {question}")

    for chunk_id, excerpt in zip(ids, excerpts):  # matched to their chunks by position
        if chunk_id not in chunk_map:
            raise ValueError(f"Unknown answer chunk ID: {chunk_id}")
        needle = _collapse(excerpt)
        if not needle:
            raise ValueError(f"Empty evidence excerpt for chunk {chunk_id}")
        chunk = chunk_map[chunk_id]
        if needle not in _collapse(chunk.get("text")) and needle not in _collapse(chunk.get("chunk_description")):
            raise ValueError(f"Evidence is not a verbatim excerpt from chunk {chunk_id}: {excerpt!r}")

    tag = qa.get("tag")
    if tag == TAG_SINGLE and len(ids) != 1:
        raise ValueError(f"Single question must reference exactly one chunk: {question}")
    if tag == TAG_MULTI_HOP and len(ids) < 2:
        raise ValueError(f"Multi-hop question must reference at least two chunks: {question}")
    if tag not in (TAG_SINGLE, TAG_MULTI_HOP):
        raise ValueError(f"Tag must be exactly '{TAG_SINGLE}' or '{TAG_MULTI_HOP}', got {tag!r}: {question}")

    qa["answer_chunk_id"] = ids
    qa["evidence"] = _one_or_many(excerpts)
    # Content comes from the source chunks, never from the model.
    qa["content"] = _one_or_many([_chunk_text_or_description(chunk_map[c]) for c in ids])


def _stamp_case_ids(output: Dict[str, Any], page_no: int, document_id: str) -> Dict[str, Any]:
    for q_type, key in _QUESTION_KEYS:
        for qa in output[key]:
            qa["case_id"] = case_id_for(document_id, page_no, q_type, qa.get("question", ""))
    return output


def _is_grounded(entry: Any) -> bool:
    """A stored page result that needs no new attempt (absent or error-tagged ones do)."""
    return isinstance(entry, dict) and not entry.get("error")


def _collapse(text: Any) -> str:
    return " ".join(str(text or "").split())


def _one_or_many(items: List[str]) -> Any:
    return items[0] if len(items) == 1 else items


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]

# =========================================================================== #
# Groundtruth quality metrics (HCPR, EVR, DCC, Indirect Overlap, MHR, Tag Consistency)
# =========================================================================== #

REQUIRED_FIELDS = ("question", "answer", "tag", "answer_chunk_id", "evidence", "content")

QUESTION_SECTIONS = (("direct", "direct_questions"), ("indirect", "indirect_questions"))

METRIC_DEFINITIONS = {
    "HCPR": "% of questions with all required fields populated.",
    "EVR": "Mean containment of evidence inside the cited chunk text (%).",
    "DCC": "% of registered chunks cited by >= 1 question.",
    "Indirect Overlap": "% of indirect questions that overlap another question on the same page.",
    "MHR": "% of questions tagged multi_hop.",
    "Tag Consistency": "% of single/multi_hop questions whose tag matches their citation count.",
}
METRIC_NAMES = tuple(METRIC_DEFINITIONS)

_SUMMED_COUNT_KEYS = (
    "total_questions",
    "complete_questions",
    "multi_hop_questions",
    "tag_checked",
    "tag_inconsistent",
    "indirect_questions",
    "indirect_with_overlap",
)

_METRICS_CSV_COUNT_COLUMNS = (
    "total_questions",
    "complete_questions",
    "multi_hop_questions",
    "tag_inconsistent",
    "registered_chunks",
    "registered_chunks_covered",
)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def _normalize_for_matching(value) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    if value is None:
        return ""
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def _text_similarity(left, right) -> float:
    a, b = _normalize_for_matching(left), _normalize_for_matching(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _containment_ratio(evidence, source) -> float:
    """Fraction of the evidence found in source, in [0, 1]."""
    ev, src = _normalize_for_matching(evidence), _normalize_for_matching(source)
    if not ev or not src:
        return 0.0
    if ev in src:
        return 1.0
    ev_tokens, src_tokens = Counter(ev.split()), Counter(src.split())
    hits = sum(min(n, src_tokens[token]) for token, n in ev_tokens.items())
    return hits / sum(ev_tokens.values())


def _percentage(numerator, denominator, empty=0.0):
    return round(numerator / denominator * 100, 2) if denominator else empty


def _mean_percentage(scores: list) -> float:
    return round(sum(scores) / len(scores) * 100, 2) if scores else 0.0


def _write_json(path, data) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _write_csv(path, fieldnames, rows) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k, "")) for k in fieldnames})


# --------------------------------------------------------------------------- #
# Chunk / question helpers
# --------------------------------------------------------------------------- #

def _chunk_text_or_description(chunk: dict) -> str:
    """Full source text of a chunk; falls back to chunk_description (figures / tables)."""
    text = chunk.get("text") or ""
    if text.strip():
        return text
    return chunk.get("chunk_description") or ""


def index_chunks(pages: dict):
    """({chunk_id: full text}, {page_key: [chunk ids]}) for {page: [chunk dicts]}."""
    chunk_text, page_chunk_ids = {}, {}
    for page_no, chunks in pages.items():
        ids = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id is None:
                continue
            chunk_id = str(chunk_id)
            ids.append(chunk_id)
            chunk_text[chunk_id] = _chunk_text_or_description(chunk)
        page_chunk_ids[str(page_no)] = ids
    return chunk_text, page_chunk_ids


def _as_chunk_ids(value) -> list:
    """Ordered, de-duplicated list of chunk ids (lenient)."""
    if value is None:
        return []
    if isinstance(value, (str, int)):
        value = [value]
    if not isinstance(value, list):
        return []
    seen, ids = set(), []
    for item in value:
        chunk_id = str(item).strip()
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            ids.append(chunk_id)
    return ids


def _evidence_as_list(qa: dict) -> list:
    evidence = qa.get("evidence")
    if isinstance(evidence, list):
        return [str(x) for x in evidence if x is not None and str(x).strip()]
    if isinstance(evidence, str) and evidence.strip():
        return [evidence]
    return []


def _check_tag_vs_citations(qa: dict):
    """(was_checked, issue). multi_hop needs >= 2 chunks; single exactly 1."""
    tag = qa.get("tag")
    n_chunks = len(_as_chunk_ids(qa.get("answer_chunk_id")))
    if tag == TAG_MULTI_HOP:
        return True, (None if n_chunks >= 2 else f"tagged multi_hop but cites {n_chunks} chunk(s); need >= 2")
    if tag == TAG_SINGLE:
        return True, (None if n_chunks == 1 else f"tagged single but cites {n_chunks} chunk(s); need exactly 1")
    return False, None


def _missing_required_fields(qa: dict) -> list:
    return [f for f in REQUIRED_FIELDS if qa.get(f) in (None, "", [])]


def _overlap_comparison_text(qa: dict) -> str:
    parts = [qa.get("question", ""), qa.get("answer", "")] + _evidence_as_list(qa)
    return " ".join(str(p) for p in parts if p)


def _split_page_questions(page_data: dict):
    """(direct, indirect) question lists, ignoring non-dict entries."""
    direct, indirect = (
        [q for q in (page_data.get(section) or []) if isinstance(q, dict)]
        for _, section in QUESTION_SECTIONS
    )
    return direct, indirect


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _best_chunk_match(evidence: str, candidate_ids: list, chunk_text: dict):
    """Best (chunk_id, containment) for an excerpt; with several candidates their concatenation is tried too."""
    best_id, best_score = max(
        ((c, _containment_ratio(evidence, chunk_text[c])) for c in candidate_ids),
        key=lambda pair: pair[1],
    )
    if len(candidate_ids) > 1:
        combined = _containment_ratio(evidence, " ".join(chunk_text[c] for c in candidate_ids))
        if combined > best_score:
            best_id, best_score = "+".join(candidate_ids), combined
    return best_id, best_score


def _score_question_evidence(qa: dict, chunk_text: dict):
    """(mean containment in [0, 1] or None, per-excerpt comparisons)."""
    chunk_ids = _as_chunk_ids(qa.get("answer_chunk_id"))
    excerpts = _evidence_as_list(qa)
    if not chunk_ids or not excerpts:
        return None, []

    one_to_one = len(chunk_ids) == len(excerpts)
    comparisons = []
    for i, excerpt in enumerate(excerpts):
        pool = [chunk_ids[i]] if one_to_one else chunk_ids
        candidates = [c for c in pool if c in chunk_text]
        if not candidates:
            comparisons.append({"evidence": excerpt, "chunk_id": None, "containment": None})
            continue
        best_id, best_score = _best_chunk_match(excerpt, candidates, chunk_text)
        comparisons.append({"evidence": excerpt, "chunk_id": best_id, "containment": round(best_score, 4)})

    scores = [c["containment"] for c in comparisons if c["containment"] is not None]
    return (sum(scores) / len(scores) if scores else None), comparisons


def _find_indirect_overlaps(direct: list, indirect: list, threshold: float):
    """Each indirect question against every other question on the page: (overlapping indices, pairs)."""
    pool = [("direct", i, q) for i, q in enumerate(direct)] + [("indirect", i, q) for i, q in enumerate(indirect)]
    overlapping, pairs = set(), []
    for ind_idx, indirect_qa in enumerate(indirect):
        source_text = _overlap_comparison_text(indirect_qa)
        for group, other_idx, other_qa in pool:
            if group == "indirect" and other_idx == ind_idx:
                continue
            score = _text_similarity(source_text, _overlap_comparison_text(other_qa))
            if score >= threshold:
                overlapping.add(ind_idx)
                pairs.append({
                    "indirect_question": indirect_qa.get("question"),
                    "compared_group": group,
                    "compared_question": other_qa.get("question"),
                    "similarity": round(score, 4),
                })
    return overlapping, pairs


def _compute_metric_values(counts: dict, evr_scores: list, dcc_when_no_chunks) -> dict:
    return {
        "HCPR": _percentage(counts["complete_questions"], counts["total_questions"]),
        "EVR": _mean_percentage(evr_scores),
        "DCC": _percentage(counts["registered_chunks_covered"], counts["registered_chunks"], empty=dcc_when_no_chunks),
        "Indirect Overlap": _percentage(counts["indirect_with_overlap"], counts["indirect_questions"]),
        "MHR": _percentage(counts["multi_hop_questions"], counts["total_questions"]),
        "Tag Consistency": _percentage(
            counts["tag_checked"] - counts["tag_inconsistent"], counts["tag_checked"], empty=100.0,
        ),
    }


def _evaluate_page(page_key, page_data, page_registered, chunk_text, overlap_threshold):
    """One page: (report, EVR scores, chunk usage Counter)."""
    direct, indirect = _split_page_questions(page_data)
    labelled = [("direct", q) for q in direct] + [("indirect", q) for q in indirect]

    complete = multi_hop = tag_checked = tag_ok = 0
    evr_scores, chunk_usage, details, tag_issues = [], Counter(), [], []

    for idx, (group, qa) in enumerate(labelled):
        missing = _missing_required_fields(qa)
        complete += not missing
        multi_hop += qa.get("tag") == TAG_MULTI_HOP

        chunk_ids = _as_chunk_ids(qa.get("answer_chunk_id"))
        chunk_usage.update(chunk_ids)

        was_checked, issue = _check_tag_vs_citations(qa)
        if was_checked:
            tag_checked += 1
            tag_ok += issue is None
            if issue:
                tag_issues.append({
                    "group": group, "question": qa.get("question"),
                    "tag": qa.get("tag"), "answer_chunk_id": chunk_ids, "issue": issue,
                })

        evr, comparisons = _score_question_evidence(qa, chunk_text)
        if evr is not None:
            evr_scores.append(evr)

        details.append({
            "group": group,
            "question": qa.get("question") or f"Question {idx + 1}",
            "tag": qa.get("tag"),
            "answer_chunk_id": chunk_ids,
            "missing_required_fields": missing,
            "tag_consistent": issue is None if was_checked else None,
            "evr": round(evr, 4) if evr is not None else None,
            "evidence_comparisons": comparisons,
        })

    overlapping, overlap_pairs = _find_indirect_overlaps(direct, indirect, overlap_threshold)

    covered = [c for c in page_registered if chunk_usage.get(c, 0) > 0]
    unused = [c for c in page_registered if chunk_usage.get(c, 0) == 0]

    counts = {
        "total_questions": len(labelled),
        "direct_questions": len(direct),
        "indirect_questions": len(indirect),
        "complete_questions": complete,
        "multi_hop_questions": multi_hop,
        "tag_checked": tag_checked,
        "tag_inconsistent": tag_checked - tag_ok,
        "registered_chunks": len(page_registered),
        "registered_chunks_covered": len(covered),
        "indirect_with_overlap": len(overlapping),
    }
    report = {
        "page": str(page_key),
        "error": page_data.get("error"),
        "metrics": _compute_metric_values(counts, evr_scores, dcc_when_no_chunks=None),
        "counts": counts,
        "unused_chunks": unused,
        "tag_consistency_issues": tag_issues,
        "indirect_overlap_pairs": overlap_pairs,
        "questions": details,
    }
    return report, evr_scores, chunk_usage


def compute_metrics(
    results: dict,
    page_chunk_ids: dict,
    chunk_text: dict,
    registered_ids: list,
    overlap_threshold: float = 0.80,
) -> dict:
    """
    results:        groundtruth dataset {page_key: {direct_questions, indirect_questions, ...}}
    page_chunk_ids: {page_key(str): [chunk ids of that page]}
    chunk_text:     {chunk_id: full chunk text}
    registered_ids: the DCC universe (every chunk that was indexed)
    """
    registered = list(dict.fromkeys(registered_ids))
    registered_set = set(registered)

    page_reports, all_evr_scores = {}, []
    usage_all, totals = Counter(), Counter()

    for page_key, page_data in results.items():
        if not isinstance(page_data, dict):
            continue
        page_registered = [c for c in page_chunk_ids.get(str(page_key), []) if c in registered_set]
        report, evr_scores, usage = _evaluate_page(page_key, page_data, page_registered, chunk_text, overlap_threshold)
        page_reports[str(page_key)] = report
        all_evr_scores += evr_scores
        usage_all.update(usage)
        totals.update({k: report["counts"][k] for k in _SUMMED_COUNT_KEYS})

    covered = [c for c in registered if usage_all.get(c, 0) > 0]
    unused = [c for c in registered if usage_all.get(c, 0) == 0]

    overall_counts = {
        "total_pages": len(page_reports),
        "total_questions": totals["total_questions"],
        "complete_questions": totals["complete_questions"],
        "multi_hop_questions": totals["multi_hop_questions"],
        "tag_checked": totals["tag_checked"],
        "tag_inconsistent": totals["tag_inconsistent"],
        "registered_chunks": len(registered),
        "registered_chunks_covered": len(covered),
        "indirect_questions": totals["indirect_questions"],
        "indirect_with_overlap": totals["indirect_with_overlap"],
    }

    return {
        "metric_definitions": METRIC_DEFINITIONS,
        "configuration": {
            "overlap_threshold": overlap_threshold,
            "required_fields": list(REQUIRED_FIELDS),
            "evr_method": "containment",
            "registered_chunk_count": len(registered),
        },
        "overall": {
            "metrics": _compute_metric_values(overall_counts, all_evr_scores, dcc_when_no_chunks=0.0),
            "counts": overall_counts,
            "unused_chunks": unused,
            "chunks_cited_more_than_once": sorted(c for c, n in usage_all.items() if n > 1),
            "unresolved_chunk_ids": sorted(c for c in usage_all if c not in registered_set),
            "chunk_usage": dict(sorted(usage_all.items())),
        },
        "pages": page_reports,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def _page_sort_key(page_key):
    """Numeric pages first (in numeric order), then any other keys alphabetically."""
    return (0, int(page_key)) if str(page_key).isdigit() else (1, str(page_key))


def _metrics_csv_row(scope: str, entry: dict, error: str = "") -> dict:
    row = {"scope": scope, **entry["metrics"]}
    row.update({c: entry["counts"].get(c, "") for c in _METRICS_CSV_COUNT_COLUMNS})
    row["unused_chunks"] = ";".join(entry["unused_chunks"])
    row["error"] = error
    return row


def write_metrics(metrics: dict, json_path: str, csv_path: Optional[str] = None) -> str:
    """Write ground_truth_metrics.json and the flat ground_truth_metrics.csv; returns the CSV path."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(json_path, metrics)

    csv_path = Path(csv_path) if csv_path else json_path.with_suffix(".csv")
    fieldnames = ["scope"] + list(METRIC_NAMES) + list(_METRICS_CSV_COUNT_COLUMNS) + ["unused_chunks", "error"]

    rows = [
        _metrics_csv_row(f"page_{key}", metrics["pages"][key], metrics["pages"][key].get("error") or "")
        for key in sorted(metrics["pages"], key=_page_sort_key)
    ]
    rows.append(_metrics_csv_row("OVERALL", metrics["overall"]))

    _write_csv(csv_path, fieldnames, rows)
    return str(csv_path)