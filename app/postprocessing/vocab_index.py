"""
app.postprocessing.vocab_index
==============================
Vocabulary index: the union of every chunk's `concept_vocabulary`, deduplicated.

    output artifact : vocab_index.json — a flat JSON array of terms, consumed by query reformulation

An optional Groq pass (`ChunkerConfig.vocab_semantic_dedup`) merges plurals,
spelling variants and true synonyms. It degrades to the exact-dedupe list on
any failure — the index is never left empty because an LLM call failed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from app.config.defaults import ChunkerConfig
from app.app_logging.setup import get_logger
from app.postprocessing.merger import MergedChunk, load_merged_chunks
from app.utils.fs import atomic_write_json

__all__ = ["build_vocab_index", "collect_vocabulary"]

_SEMANTIC_DEDUP_SYSTEM = "You are a technical vocabulary expert. Return only a JSON array."
_SEMANTIC_DEDUP_PROMPT = """\
You will be given a list of keywords. Remove semantically identical duplicates — keywords that would serve the same purpose in a search query (exact duplicates, singular/plural forms, spelling variants, abbreviation vs. expansion, true synonyms).

<keywords>
{keywords}
</keywords>

RULES:
1. Two keywords are duplicates ONLY if they are interchangeable in a search query. Related, co-occurring, or broader/narrower terms are NOT duplicates — keep both. When in doubt, keep both.
2. Among duplicates, keep the one that appears earliest in the list.
3. Output keywords must be copied character-for-character from the input.

OUTPUT: Return ONLY a JSON array of the keywords to keep, in original input order. No explanation, no markdown fences.
"""


def collect_vocabulary(chunks: Iterable[MergedChunk]) -> List[str]:
    """Exact, order-preserving dedupe of every term across `chunks` (comma-joined strings are split)."""
    seen, terms = set(), []
    for chunk in chunks:
        raw = chunk.get("concept_vocabulary") or []
        entries = raw.split(",") if isinstance(raw, str) else raw
        for entry in entries:
            term = str(entry).strip()
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def build_vocab_index(
    merged_chunks_path: Path,
    output_path: Path,
    config: ChunkerConfig,
    logger: Optional[logging.LoggerAdapter] = None,
) -> List[str]:
    """load merged chunks ─► collect ─► [semantic dedupe] ─► write JSON array. Returns the terms."""
    log = logger or get_logger(module="VOCAB_INDEX")
    terms = collect_vocabulary(load_merged_chunks(merged_chunks_path))
    log.info("Vocabulary: %d unique terms", len(terms))

    if config.vocab_semantic_dedup and len(terms) > 1:
        terms = _semantic_dedupe(terms, config, log)

    atomic_write_json(output_path, terms)
    log.info("Vocabulary index written: %s (%d terms)", output_path, len(terms))
    return terms


def _semantic_dedupe(terms: Sequence[str], config: ChunkerConfig, log: logging.LoggerAdapter) -> List[str]:
    """Secondary step — degrades to the input list on any failure."""
    from groq import Groq  # local import: only paid for when the feature is on

    if not config.groq_api_key:
        log.warning("vocab_semantic_dedup enabled but GROQ_API_KEY is unset — keeping exact-dedupe list")
        return list(terms)
    numbered = "\n".join(f"{i}. {term}" for i, term in enumerate(terms, start=1))
    try:
        completion = Groq(api_key=config.groq_api_key).chat.completions.create(
            model=config.vocab_model,
            temperature=config.vocab_temperature,
            messages=[
                {"role": "system", "content": _SEMANTIC_DEDUP_SYSTEM},
                {"role": "user", "content": _SEMANTIC_DEDUP_PROMPT.format(keywords=numbered)},
            ],
        )
        kept = json.loads(completion.choices[0].message.content.strip())
    except Exception as exc:
        log.warning("Semantic vocabulary dedupe failed (%s) — keeping exact-dedupe list", exc)
        return list(terms)

    allowed = set(terms)
    result = [term for term in kept if isinstance(term, str) and term in allowed] if isinstance(kept, list) else []
    if not result:
        log.warning("Semantic dedupe returned nothing usable — keeping exact-dedupe list")
        return list(terms)
    log.info("Semantic dedupe: %d → %d terms", len(terms), len(result))
    return result
