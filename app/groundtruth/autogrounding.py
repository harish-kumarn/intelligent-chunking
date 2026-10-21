"""
visionrag_chunker.groundtruth.autogrounding
---------------------------------------------
Auto-groundtruth Q&A generator + quality metrics for Visual RAG evaluation
(single-file version).

Inputs:
    merged-chunks JSON   -> source chunks used to generate Q&A
    bm25_data.json       -> "local_chunk_ids": the full chunk universe used as
                            the denominator for DCC

Outputs (written next to the groundtruth file):
    groundtruth_dataset.json / groundtruth_dataset.csv
    ground_truth_metrics.json / ground_truth_metrics.csv

Evidence format:
    tag "single"    -> "evidence": "verbatim excerpt"
    tag "multi_hop" -> "evidence": ["excerpt for chunk 1", "excerpt for chunk 2"]
                       (same order as answer_chunk_id)

Content format (full source chunk text, injected from the source chunks):
    tag "single"    -> "content": "full chunk text"
    tag "multi_hop" -> "content": ["chunk 1 text", "chunk 2 text"]

Metrics (per page and OVERALL):
    HCPR             % of questions with all required fields populated.
    EVR              Mean containment of evidence inside the cited chunk text, %.
    DCC              % of registered chunks (local_chunk_ids) cited by >= 1 question.
    Indirect Overlap % of indirect questions that overlap another question on the page.
    MHR              % of questions tagged multi_hop.
    Tag Consistency  % of single/multi_hop questions whose tag matches their
                     citation count (multi_hop >= 2 chunks, single == 1).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# google-genai imported lazily inside AutoGrounder
from visionrag_chunker.config.defaults import ChunkerConfig

# =========================================================================== #
# Constants
# =========================================================================== #

_PROMPT_FILE = Path(__file__).parent / "prompts" / "auto_grounding_prompt.md"

GROUNDTRUTH_JSON = "groundtruth_dataset.json"
METRICS_JSON = "ground_truth_metrics.json"
BM25_DEFAULT_NAME = "bm25_data.json"

TAG_SINGLE = "single"
TAG_MULTI_HOP = "multi_hop"

REQUIRED_FIELDS = (
    "question",
    "answer",
    "tag",
    "answer_chunk_id",
    "evidence",
    "content",
)

# (question type label, key in a page's groundtruth dict)
QUESTION_SECTIONS = (
    ("direct", "direct_questions"),
    ("indirect", "indirect_questions"),
)

METRIC_DEFINITIONS = {
    "HCPR": "% of questions with all required fields populated.",
    "EVR": "Mean containment of evidence inside the cited chunk text (%).",
    "DCC": "% of registered chunks (local_chunk_ids) cited by >= 1 question.",
    "Indirect Overlap": "% of indirect questions that overlap another question on the same page.",
    "MHR": "% of questions tagged multi_hop.",
    "Tag Consistency": "% of single/multi_hop questions whose tag matches their citation count.",
}
METRIC_NAMES = tuple(METRIC_DEFINITIONS)

# Per-page counts that are summed to produce the OVERALL counts.
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

_GROUNDTRUTH_CSV_COLUMNS = (
    "page",
    "type",
    "tag",
    "answer_chunk_id",
    "question",
    "answer",
    "evidence",
    "content",
)

_RETRY_FEEDBACK_TEMPLATE = (
    "\n\nYour previous output was rejected by validation:\n"
    "{error}\n"
    "Return the COMPLETE corrected JSON object. Every QA pair "
    "must include an 'evidence' field: a single verbatim "
    "string for 'single' questions, or a list of verbatim "
    "strings (one per answer_chunk_id, in the same order) "
    "for 'multi_hop' questions. Do not put chunk_ids inside "
    "evidence."
)


# =========================================================================== #
# Generic helpers (text, JSON, CSV, math)
# =========================================================================== #

def _collapse_whitespace(text: str) -> str:
    """Collapse all whitespace so line breaks / double spaces don't matter."""
    return " ".join((text or "").split())


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
    """Mean of scores in [0, 1] expressed as a percentage (0.0 when empty)."""
    return round(sum(scores) / len(scores) * 100, 2) if scores else 0.0


def _one_or_many(items: list):
    """Stored shape: a bare value for one item, otherwise the list."""
    return items[0] if len(items) == 1 else items


def _cell_text(value) -> str:
    """CSV cell: strings as-is, lists/other values as a JSON string."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, data) -> None:
    """Write data to path as UTF-8 JSON (indent=2, non-ASCII preserved)."""
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _write_csv(path, fieldnames, rows) -> None:
    """Write rows (dicts) to CSV; missing / None values become empty cells."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: ("" if row.get(k) is None else row.get(k, "")) for k in fieldnames}
            )


# =========================================================================== #
# Chunk / question helpers
# =========================================================================== #

def _chunk_text_or_description(chunk: dict) -> str:
    """Full source text of a chunk; falls back to chunk_description when the
    chunk has no text (e.g., figures / tables)."""
    text = chunk.get("text") or ""
    if text.strip():
        return text
    return chunk.get("chunk_description") or ""


def _as_chunk_ids(value) -> list:
    """Ordered, de-duplicated list of chunk ids (lenient; used for metrics)."""
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
    """Return (was_checked, issue). multi_hop needs >= 2 chunks; single exactly 1."""
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
    """Return (direct, indirect) question lists, ignoring non-dict entries."""
    direct, indirect = (
        [q for q in (page_data.get(section) or []) if isinstance(q, dict)]
        for _, section in QUESTION_SECTIONS
    )
    return direct, indirect


# =========================================================================== #
# Metrics
# =========================================================================== #

def _find_local_chunk_ids(node):
    """Recursively search a JSON structure for a "local_chunk_ids" list."""
    if isinstance(node, dict):
        if isinstance(node.get("local_chunk_ids"), list):
            return node["local_chunk_ids"]
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None

    for child in children:
        found = _find_local_chunk_ids(child)
        if found is not None:
            return found
    return None


def load_local_chunk_ids(bm25_path: str) -> list:
    """Read "local_chunk_ids" from bm25_data.json (searched recursively)."""
    ids = _find_local_chunk_ids(_read_json(bm25_path))
    if ids is None:
        raise ValueError(f"'local_chunk_ids' not found in {bm25_path}")
    return list(dict.fromkeys(str(x).strip() for x in ids if str(x).strip()))


def _best_chunk_match(evidence: str, candidate_ids: list, chunk_text: dict):
    """Best (chunk_id, containment) for an excerpt among candidate chunks.

    When several candidates exist, the concatenation of all of them is also
    tried (evidence may span chunks); its id is the candidates joined by "+".
    """
    best_id, best_score = max(
        ((c, _containment_ratio(evidence, chunk_text[c])) for c in candidate_ids),
        key=lambda pair: pair[1],
    )
    if len(candidate_ids) > 1:
        combined = _containment_ratio(
            evidence, " ".join(chunk_text[c] for c in candidate_ids)
        )
        if combined > best_score:
            best_id, best_score = "+".join(candidate_ids), combined
    return best_id, best_score


def _score_question_evidence(qa: dict, chunk_text: dict):
    """Return (mean containment in [0, 1] or None, per-excerpt comparisons)."""
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
        comparisons.append(
            {"evidence": excerpt, "chunk_id": best_id, "containment": round(best_score, 4)}
        )

    scores = [c["containment"] for c in comparisons if c["containment"] is not None]
    return (sum(scores) / len(scores) if scores else None), comparisons


def _find_indirect_overlaps(direct: list, indirect: list, threshold: float):
    """Compare each indirect question to every other question on the page.

    Returns (set of indirect indices that overlap something, overlap pairs).
    """
    pool = [("direct", i, q) for i, q in enumerate(direct)] + [
        ("indirect", i, q) for i, q in enumerate(indirect)
    ]
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
    """Turn counts + EVR scores into the six headline metrics (shared by the
    per-page and OVERALL reports)."""
    return {
        "HCPR": _percentage(counts["complete_questions"], counts["total_questions"]),
        "EVR": _mean_percentage(evr_scores),
        "DCC": _percentage(
            counts["registered_chunks_covered"],
            counts["registered_chunks"],
            empty=dcc_when_no_chunks,
        ),
        "Indirect Overlap": _percentage(
            counts["indirect_with_overlap"], counts["indirect_questions"]
        ),
        "MHR": _percentage(counts["multi_hop_questions"], counts["total_questions"]),
        "Tag Consistency": _percentage(
            counts["tag_checked"] - counts["tag_inconsistent"],
            counts["tag_checked"],
            empty=100.0,
        ),
    }


def _evaluate_page(page_key, page_data, page_registered, chunk_text, overlap_threshold):
    """Evaluate one page. Returns (page report, EVR scores, chunk usage Counter)."""
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
    Args:
        results:        groundtruth dataset {page_key: {direct_questions, ...}}
        page_chunk_ids: {page_key(str): [chunk ids shown to the model for that page]}
        chunk_text:     {chunk_id: full chunk text} from the merged chunks
        registered_ids: bm25_data.json "local_chunk_ids" (DCC denominator)
    """
    registered = list(dict.fromkeys(registered_ids))
    registered_set = set(registered)

    page_reports, all_evr_scores = {}, []
    usage_all, totals = Counter(), Counter()

    for page_key, page_data in results.items():
        if not isinstance(page_data, dict):
            continue
        page_registered = [
            c for c in page_chunk_ids.get(str(page_key), []) if c in registered_set
        ]
        report, evr_scores, usage = _evaluate_page(
            page_key, page_data, page_registered, chunk_text, overlap_threshold
        )
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
            "metrics": _compute_metric_values(
                overall_counts, all_evr_scores, dcc_when_no_chunks=0.0
            ),
            "counts": overall_counts,
            "unused_chunks": unused,
            "chunks_cited_more_than_once": sorted(c for c, n in usage_all.items() if n > 1),
            "unresolved_chunk_ids": sorted(c for c in usage_all if c not in registered_set),
            "chunk_usage": dict(sorted(usage_all.items())),
        },
        "pages": page_reports,
    }


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
    """Write ground_truth_metrics.json and the flat ground_truth_metrics.csv."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(json_path, metrics)

    csv_path = Path(csv_path) if csv_path else json_path.with_suffix(".csv")
    fieldnames = (
        ["scope"] + list(METRIC_NAMES) + list(_METRICS_CSV_COUNT_COLUMNS)
        + ["unused_chunks", "error"]
    )

    rows = [
        _metrics_csv_row(f"page_{key}", metrics["pages"][key], metrics["pages"][key].get("error") or "")
        for key in sorted(metrics["pages"], key=_page_sort_key)
    ]
    rows.append(_metrics_csv_row("OVERALL", metrics["overall"]))

    _write_csv(csv_path, fieldnames, rows)
    return str(csv_path)


# =========================================================================== #
# Validation of model output
# =========================================================================== #

def _parse_answer_chunk_ids(qa: dict) -> list:
    """Strictly parse answer_chunk_id into a non-empty list of unique strings."""
    raw = qa.get("answer_chunk_id", [])
    if isinstance(raw, (str, int)):
        ids = [str(raw)]
    elif isinstance(raw, list):
        ids = [str(x) for x in raw]
    else:
        raise ValueError(f"answer_chunk_id must be a list or scalar: {qa.get('question')}")

    if not ids:
        raise ValueError(f"Missing answer_chunk_id: {qa.get('question')}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate answer_chunk_id values: {qa.get('question')}")
    return ids


def _parse_evidence(qa: dict) -> list:
    """Strictly parse evidence: a string (single) or list of strings (multi_hop)."""
    evidence = qa.get("evidence")
    if isinstance(evidence, str):
        return [evidence]
    if isinstance(evidence, list) and all(isinstance(e, str) for e in evidence):
        return evidence
    raise ValueError(
        "Evidence must be a string (single) or a list of "
        f"strings (multi_hop): {qa.get('question')}"
    )


def _require_verbatim_excerpt(chunk_id: str, excerpt: str, chunk: dict) -> None:
    if not excerpt.strip():
        raise ValueError(f"Empty evidence excerpt for chunk {chunk_id}")

    needle = _collapse_whitespace(excerpt)
    source_text = _collapse_whitespace(chunk.get("text") or "")
    description = _collapse_whitespace(chunk.get("chunk_description") or "")
    if needle not in source_text and needle not in description:
        raise ValueError(
            f"Evidence is not a verbatim excerpt from chunk {chunk_id}: {excerpt!r}"
        )


def _require_tag_matches_citation_count(qa: dict, n_chunks: int) -> None:
    tag = qa.get("tag")
    if tag == TAG_SINGLE:
        if n_chunks != 1:
            raise ValueError(
                f"Single question must reference exactly one chunk: {qa.get('question')}"
            )
    elif tag == TAG_MULTI_HOP:
        if n_chunks < 2:
            raise ValueError(
                f"Multi-hop question must reference at least two chunks: {qa.get('question')}"
            )
    else:
        raise ValueError(
            f"Tag must be exactly '{TAG_SINGLE}' or '{TAG_MULTI_HOP}', got {tag!r}: "
            f"{qa.get('question')}"
        )


def _validate_question(qa: dict, chunk_map: dict) -> None:
    """Validate one Q&A pair in place and inject the authoritative `content`."""
    if not isinstance(qa, dict):
        raise ValueError("Each question item must be an object")

    answer_ids = _parse_answer_chunk_ids(qa)
    excerpts = _parse_evidence(qa)

    if len(excerpts) != len(answer_ids):
        raise ValueError(
            "Evidence must have exactly one excerpt per "
            "answer_chunk_id, in the same order. "
            f"Question: {qa.get('question')}"
        )

    # Excerpts are matched to their chunks by position.
    for chunk_id, excerpt in zip(answer_ids, excerpts):
        if chunk_id not in chunk_map:
            raise ValueError(f"Unknown answer chunk ID: {chunk_id}")
        _require_verbatim_excerpt(chunk_id, excerpt, chunk_map[chunk_id])

    qa["evidence"] = _one_or_many(excerpts)
    # Content comes from the source chunks, never from the model.
    qa["content"] = _one_or_many(
        [_chunk_text_or_description(chunk_map[cid]) for cid in answer_ids]
    )

    _require_tag_matches_citation_count(qa, len(answer_ids))


def _index_chunks(pages: dict):
    """Return ({chunk_id: full text}, {page_key: [chunk ids]}) for all pages."""
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


def _extract_page_number(chunk: dict) -> int:
    meta = chunk.get("metadata", {}) or {}
    source_pages = meta.get("source_page_nos", []) if isinstance(meta, dict) else []
    raw_page = chunk.get("page_no") or (source_pages[0] if source_pages else None)
    return int(raw_page) if str(raw_page or "").isdigit() else 0


# =========================================================================== #
# Run statistics
# =========================================================================== #

@dataclass
class _RunStats:
    pages_total: int = 0
    pages_ok: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    page_times: list = field(default_factory=list)  # seconds per processed page
    loop_time: float = 0.0
    csv_time: float = 0.0
    metrics_time: float = 0.0


# =========================================================================== #
# AutoGrounder
# =========================================================================== #

class AutoGrounder:
    """
    Generates direct + indirect Q&A groundtruth pairs for every page in a
    merged-chunks JSON file using Gemini, then computes quality metrics.

    Args:
        config: ChunkerConfig with autogrounding_* fields.
        logger: Optional logger adapter.
    """

    def __init__(
        self,
        config: ChunkerConfig,
        logger: Optional[logging.LoggerAdapter] = None,
    ):
        self._cfg = config
        self._log = logger or logging.getLogger("visionrag_chunker.autogrounding")

        # Lazy-import heavy dependencies only when the grounder is instantiated
        try:
            from google import genai  # noqa: F401
            from google.genai import types as _types
            from tqdm import tqdm as _tqdm
        except ImportError as exc:
            raise ImportError(
                "Missing dependency for AutoGrounder. "
                "Run: pip install 'google-genai>=1.0' tqdm\n"
                f"Original error: {exc}"
            ) from exc

        self._tqdm = _tqdm
        self._types = _types

        if not _PROMPT_FILE.exists():
            raise FileNotFoundError(
                f"Auto-grounding prompt not found: {_PROMPT_FILE}\n"
                "Ensure 'auto_grounding_prompt.md' is in "
                "visionrag_chunker/groundtruth/prompts/"
            )
        self._prompt_template = _PROMPT_FILE.read_text(encoding="utf-8")
        self._log.debug("Prompt template loaded from %s", _PROMPT_FILE)

        self._client = None
        self._api_key = config.google_api_key or os.getenv("GOOGLE_API_KEY", "")
        self._gemini_times: list = []  # seconds per Gemini call (reset each run)

    @property
    def config(self) -> ChunkerConfig:
        return self._cfg

    # ------------------------------------------------------------------ #
    # Gemini access
    # ------------------------------------------------------------------ #

    def _get_client(self):
        if self._client is None:
            from google import genai

            key = self._api_key or os.getenv("GOOGLE_API_KEY", "")
            if not key:
                raise ValueError(
                    "GOOGLE_API_KEY not found — required for AutoGrounder generation."
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def _generate(self, prompt: str) -> dict:
        from visionrag_chunker.utils.json_utils import repair_and_parse_json

        client = self._get_client()
        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=self._cfg.autogrounding_model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    temperature=self._cfg.autogrounding_temperature,
                    response_mime_type="application/json",
                ),
            )
        finally:
            # Count failed API calls too, so totals match real wall-clock spent.
            self._gemini_times.append(time.perf_counter() - start)
        return repair_and_parse_json(response.text)

    # ------------------------------------------------------------------ #
    # Input parsing, prompting, validation
    # ------------------------------------------------------------------ #

    def _parse_pages(self, input_file: str) -> dict:
        """Group merged chunks by page number: {page_no: [chunk dicts]}."""
        pages: dict = defaultdict(list)
        for chunk in _read_json(input_file).values():
            pages[_extract_page_number(chunk)].append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("content") or chunk.get("text", ""),
                    "chunk_description": (
                        chunk.get("chunk_description", "")
                        or chunk.get("description", "")
                    ),
                }
            )
        return dict(sorted(pages.items()))

    def _build_prompt(self, chunks: list) -> str:
        cfg = self._cfg
        return self._prompt_template.format(
            CHUNKS_JSON=json.dumps(chunks, indent=2, ensure_ascii=False),
            MIN_DIRECT=cfg.autogrounding_min_direct,
            MAX_DIRECT=cfg.autogrounding_max_direct,
            MIN_INDIRECT=cfg.autogrounding_min_indirect,
            MAX_INDIRECT=cfg.autogrounding_max_indirect,
        )

    def _validate_qa_output(self, output: dict, chunks: list) -> dict:
        """Validate answer chunk IDs and ensure evidence excerpts occur verbatim
        in their corresponding source chunks (matched by position). Also
        injects the authoritative full chunk content for each answer chunk."""
        if not isinstance(output, dict):
            raise ValueError("Model output must be a JSON object")

        chunk_map = {
            str(chunk.get("chunk_id")): chunk
            for chunk in chunks
            if chunk.get("chunk_id") is not None
        }

        for _, section in QUESTION_SECTIONS:
            if section not in output:
                raise ValueError(f"Missing required section: {section}")
            questions = output[section]
            if not isinstance(questions, list):
                raise ValueError(f"{section} must be a list")
            for qa in questions:
                if not isinstance(qa, dict):
                    raise ValueError(f"Each item in {section} must be an object")
                _validate_question(qa, chunk_map)

        return output

    # ------------------------------------------------------------------ #
    # Per-page generation (with retries)
    # ------------------------------------------------------------------ #

    def _generate_page_groundtruth(self, page_no: int, chunks: list):
        """Generate + validate Q&A for one page, retrying with validation feedback.

        Returns (page result, succeeded). On total failure the result is an
        empty fallback carrying the last error.
        """
        max_retries = self._cfg.autogrounding_max_retries
        base_prompt = self._build_prompt(chunks)
        prompt = base_prompt
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                output = self._validate_qa_output(self._generate(prompt), chunks)
                self._log.info(
                    "Page %s ✅ — %d direct, %d indirect Q&A",
                    page_no,
                    len(output.get("direct_questions", [])),
                    len(output.get("indirect_questions", [])),
                )
                return output, True
            except Exception as exc:
                last_error = exc
                self._log.error(
                    "Page %s | attempt %d/%d failed: %s",
                    page_no, attempt, max_retries, exc,
                )
                if attempt == max_retries:
                    break

                # Show the model what was wrong so the next attempt can fix it.
                prompt = base_prompt + _RETRY_FEEDBACK_TEMPLATE.format(error=exc)
                wait = 2 ** (attempt - 1)
                self._log.warning("Retry in %ds…", wait)
                time.sleep(wait)

        self._log.warning(
            "Page %s ⚠️ — All %d retries failed (%s). Saved empty fallback groundtruth.",
            page_no, max_retries, last_error,
        )
        fallback = {
            "page_no": page_no,
            "direct_questions": [],
            "indirect_questions": [],
            "error": str(last_error),
        }
        return fallback, False

    def _load_existing_results(self, out_path: Path) -> dict:
        """Resume support: load previously written groundtruth, if any."""
        if not (out_path.exists() and out_path.stat().st_size > 0):
            return {}
        try:
            results = _read_json(out_path)
            self._log.info("Resumed: %d pages already processed", len(results))
            return results
        except Exception as exc:
            self._log.warning(
                "Could not load existing groundtruth file %s (%s) — starting fresh",
                out_path, exc,
            )
            return {}

    # ------------------------------------------------------------------ #
    # Public methods
    # ------------------------------------------------------------------ #

    def run(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        bm25_file: Optional[str] = None,
    ) -> str:
        """
        Generate groundtruth Q&A for all pages in input_file, then compute
        metrics against the chunk universe in bm25_data.json.

        Args:
            input_file: Path to the merged-chunks JSON.
            output_file: Optional explicit groundtruth JSON path. Defaults to
                         <input_dir>/groundtruth_dataset.json.
            bm25_file: Path to bm25_data.json (must contain "local_chunk_ids").
                       Defaults to cfg.bm25_data_file, else
                       <input_dir>/bm25_data.json.

        Returns:
            Path to the groundtruth JSON. Also writes groundtruth_dataset.csv,
            ground_truth_metrics.json and ground_truth_metrics.csv alongside it.
        """
        run_start = time.perf_counter()
        self._gemini_times = []

        out_path = (
            Path(output_file) if output_file else Path(input_file).parent / GROUNDTRUTH_JSON
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pages = self._parse_pages(input_file)
        self._log.info("Auto-grounding: %d pages to process", len(pages))

        results = self._load_existing_results(out_path)
        stats = _RunStats(pages_total=len(pages))

        loop_start = time.perf_counter()
        for page_no, chunks in self._tqdm(pages.items(), desc="Generating groundtruth"):
            key = str(page_no)

            # Skip only pages that succeeded; retry pages saved as error fallbacks.
            if key in results and not results[key].get("error"):
                self._log.debug("Skip page %s (already processed)", page_no)
                stats.pages_skipped += 1
                continue

            page_start = time.perf_counter()
            results[key], succeeded = self._generate_page_groundtruth(page_no, chunks)
            _write_json(out_path, results)  # incremental save

            stats.page_times.append(time.perf_counter() - page_start)
            if succeeded:
                stats.pages_ok += 1
            else:
                stats.pages_failed += 1
        stats.loop_time = time.perf_counter() - loop_start

        csv_start = time.perf_counter()
        csv_path = self.to_csv(str(out_path))
        stats.csv_time = time.perf_counter() - csv_start
        self._log.info("Groundtruth CSV written → %s", csv_path)

        metrics_start = time.perf_counter()
        metrics_json, metrics_csv = self.evaluate(
            results=results,
            pages=pages,
            input_file=input_file,
            bm25_file=bm25_file,
            out_dir=out_path.parent,
        )
        stats.metrics_time = time.perf_counter() - metrics_start
        self._log.info("Metrics written → %s | %s", metrics_json, metrics_csv)

        latency = self._compute_latency_summary(stats, total_time=time.perf_counter() - run_start)
        self._log_latency_summary(latency)
        self._attach_latency_to_metrics(metrics_json, latency)

        return str(out_path)

    # ------------------------------------------------------------------ #
    # Latency reporting
    # ------------------------------------------------------------------ #

    def _compute_latency_summary(self, stats: _RunStats, total_time: float) -> dict:
        gemini_times = self._gemini_times
        gemini_total = sum(gemini_times)
        gemini_avg = gemini_total / len(gemini_times) if gemini_times else 0.0

        page_times = stats.page_times
        if page_times:
            page_avg = sum(page_times) / len(page_times)
            page_min, page_max = min(page_times), max(page_times)
        else:
            page_avg = page_min = page_max = 0.0

        # Time inside the page loop not spent waiting on Gemini:
        # retry backoff sleeps, validation, and incremental file writes.
        non_gemini = max(stats.loop_time - gemini_total, 0.0)

        return {
            "total_run_time_s": round(total_time, 2),
            "page_generation_loop_s": round(stats.loop_time, 2),
            "csv_export_s": round(stats.csv_time, 2),
            "metrics_export_s": round(stats.metrics_time, 2),
            "pages": {
                "total": stats.pages_total,
                "ok": stats.pages_ok,
                "failed": stats.pages_failed,
                "skipped": stats.pages_skipped,
            },
            "page_latency_s": {
                "avg": round(page_avg, 2),
                "min": round(page_min, 2),
                "max": round(page_max, 2),
            },
            "gemini": {
                "calls": len(gemini_times),
                "total_s": round(gemini_total, 2),
                "avg_s_per_call": round(gemini_avg, 2),
            },
            "non_gemini_time_s": round(non_gemini, 2),
        }

    def _log_latency_summary(self, latency: dict) -> None:
        pages, page_lat, gemini = latency["pages"], latency["page_latency_s"], latency["gemini"]
        lines = [
            "========== OVERALL LATENCY SUMMARY ==========",
            f"  Total run time        : {latency['total_run_time_s']:.2f}s",
            f"  Page generation loop  : {latency['page_generation_loop_s']:.2f}s",
            f"  CSV export            : {latency['csv_export_s']:.2f}s",
            f"  Metrics export        : {latency['metrics_export_s']:.2f}s",
            f"  Pages (total/ok/failed/skipped): "
            f"{pages['total']} / {pages['ok']} / {pages['failed']} / {pages['skipped']}",
            f"  Page latency (avg/min/max)     : "
            f"{page_lat['avg']:.2f}s / {page_lat['min']:.2f}s / {page_lat['max']:.2f}s",
            f"  Gemini calls          : {gemini['calls']} "
            f"(total {gemini['total_s']:.2f}s, avg {gemini['avg_s_per_call']:.2f}s/call)",
            f"  Non-Gemini time (retry backoff, validation, file I/O): "
            f"{latency['non_gemini_time_s']:.2f}s",
            "=============================================",
        ]
        for line in lines:
            self._log.info(line)

    def _attach_latency_to_metrics(self, metrics_json: str, latency: dict) -> None:
        """Add the latency summary to ground_truth_metrics.json (best effort)."""
        try:
            data = _read_json(metrics_json)
            data["latency"] = latency
            _write_json(metrics_json, data)
        except Exception as exc:  # never fail the run over reporting
            self._log.warning("Could not attach latency to %s (%s)", metrics_json, exc)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def _resolve_bm25_path(self, input_file: str, bm25_file: Optional[str]) -> Path:
        if bm25_file:
            return Path(bm25_file)
        cfg_path = getattr(self._cfg, "bm25_data_file", None)
        if cfg_path:
            return Path(cfg_path)
        return Path(input_file).parent / BM25_DEFAULT_NAME

    def _load_registered_chunk_ids(
        self, pages: dict, input_file: str, bm25_file: Optional[str]
    ) -> list:
        """DCC universe: bm25 "local_chunk_ids", else every merged-chunk id."""
        bm25_path = self._resolve_bm25_path(input_file, bm25_file)
        if bm25_path.is_file():
            registered = load_local_chunk_ids(str(bm25_path))
            self._log.info(
                "DCC universe: %d local_chunk_ids from %s", len(registered), bm25_path
            )
            return registered

        self._log.warning(
            "bm25 file not found (%s) — falling back to merged-chunk IDs "
            "as the DCC universe (DCC will be trivially high).",
            bm25_path,
        )
        return [
            str(chunk["chunk_id"])
            for chunks in pages.values()
            for chunk in chunks
            if chunk.get("chunk_id") is not None
        ]

    def _log_overall_metrics(self, overall: dict) -> None:
        for name, value in overall["metrics"].items():
            self._log.info("Overall %-16s %.2f%%", name, value)
        if overall["unused_chunks"]:
            self._log.warning(
                "%d chunk(s) never used in groundtruth: %s",
                len(overall["unused_chunks"]),
                overall["unused_chunks"],
            )
        if overall["counts"]["tag_inconsistent"]:
            self._log.warning(
                "%d question(s) have a tag that does not match their citation count.",
                overall["counts"]["tag_inconsistent"],
            )

    def evaluate(
        self,
        results: dict,
        pages: dict,
        input_file: str,
        bm25_file: Optional[str] = None,
        out_dir: Optional[Path] = None,
    ) -> tuple:
        """Compute HCPR / EVR / DCC / Indirect Overlap / MHR / Tag Consistency.

        DCC is the share of "local_chunk_ids" (bm25_data.json) cited by at
        least one groundtruth question. Writes ground_truth_metrics.json and
        ground_truth_metrics.csv into out_dir.
        """
        out_dir = Path(out_dir) if out_dir else Path(input_file).parent

        registered = self._load_registered_chunk_ids(pages, input_file, bm25_file)
        chunk_text, page_chunk_ids = _index_chunks(pages)

        metrics = compute_metrics(
            results=results,
            page_chunk_ids=page_chunk_ids,
            chunk_text=chunk_text,
            registered_ids=registered,
            overlap_threshold=getattr(self._cfg, "autogrounding_overlap_threshold", 0.80),
        )

        metrics_json = out_dir / METRICS_JSON
        metrics_csv = write_metrics(metrics, str(metrics_json))

        self._log_overall_metrics(metrics["overall"])
        return str(metrics_json), metrics_csv

    # ------------------------------------------------------------------ #
    # CSV export
    # ------------------------------------------------------------------ #

    def to_csv(self, json_path: str, csv_path: Optional[str] = None) -> str:
        """Convert a groundtruth JSON file to a flat CSV."""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Groundtruth JSON not found: {json_path}")

        data = _read_json(json_path)
        csv_path = csv_path or str(json_path.with_suffix(".csv"))

        rows = []
        for page_key, page_data in data.items():
            for question_type, section in QUESTION_SECTIONS:
                for qa in page_data.get(section, []):
                    chunk_ids = qa.get("answer_chunk_id", [])
                    if isinstance(chunk_ids, (str, int)):
                        chunk_ids = [chunk_ids]

                    rows.append(
                        {
                            "page": page_key,
                            "type": question_type,
                            "tag": qa.get("tag", ""),
                            "answer_chunk_id": ";".join(str(x) for x in chunk_ids),
                            "question": qa.get("question", ""),
                            "answer": qa.get("answer", ""),
                            # Single -> plain text; multi_hop -> JSON array string.
                            "evidence": _cell_text(qa.get("evidence", "")),
                            "content": _cell_text(qa.get("content", "")),
                        }
                    )

        _write_csv(csv_path, _GROUNDTRUTH_CSV_COLUMNS, rows)
        self._log.info("CSV: %d rows → %s", len(rows), csv_path)
        return csv_path