"""
app.postprocessing.merger
=========================
Cross-page continuation merger: fuses chunk halves the chunker marked as
continuing across a page break into single retrieval units.

    input  : ChunkRecord list from ChunkStore (document order)
    output : MergedChunk list — the published `merged_chunks.json` contract:
             {chunk_id, content, chunk_description, concept_vocabulary, images,
              metadata{source_chunk_ids, source_page_nos[, merge_reason,
                       ocr_corrections, content_continuation]}}

Main path
    merge_chunks ─► _build_chains (link outgoing → incoming by seam id) ─► _fuse / _standalone

Chain rules: only `outgoing`/`bidirectional` open a chain; the partner is the
`incoming`/`bidirectional` chunk whose `incoming_group_id` equals the current
`outgoing_group_id`. When the model forgot the seam id, the immediately
following `outgoing` chunk is absorbed instead (capped). Merged ids are unique
within the document (a repeated fused id gets `_2`, `_3`, …). Pure functions; the
only I/O is in `merge_file` / `load_merged_chunks` / `save_merged_chunks`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from app.chunking.schema import ChunkRecord
from app.chunking.store import ChunkStore
from app.logging.setup import get_logger
from app.utils.fs import atomic_write_json

__all__ = ["MergedChunk", "load_merged_chunks", "merge_chunks", "merge_file", "save_merged_chunks"]

MergedChunk = Dict[str, Any]

_TEXT_JOINER = "\n"
_DESC_JOINER = " "
_VOCAB_CAP = 7                    # matches the per-chunk vocabulary budget in the system prompt
_MAX_FORCED_ABSORPTIONS = 9       # bound on seam-less outgoing chunks glued together
_MERGE_REASON = "cross_page_continuation"
_OPENERS = frozenset({"outgoing", "bidirectional"})
_RECEIVERS = frozenset({"incoming", "bidirectional"})
_PAGE_FROM_ID = re.compile(r"page_(\d+)")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def merge_chunks(records: Sequence[ChunkRecord], logger: Optional[logging.LoggerAdapter] = None) -> List[MergedChunk]:
    """Fuse continuation chains; standalone chunks are re-shaped to the same contract."""
    log = logger or get_logger(module="MERGER")
    groups = _build_chains(list(records), log)
    merged = _unique_ids([_fuse(item) if isinstance(item, list) else _standalone(item) for item in groups], log)
    log.info(
        "Merged %d → %d chunks (%d chain(s) fused)",
        len(records), len(merged), sum(1 for item in groups if isinstance(item, list)),
    )
    return merged


def merge_file(in_path: Path, out_path: Path, logger: Optional[logging.LoggerAdapter] = None) -> List[MergedChunk]:
    """Read a ChunkStore file, merge, and write `merged_chunks.json` (keys chunk1..N)."""
    merged = merge_chunks(ChunkStore(Path(in_path), logger).load(), logger)
    save_merged_chunks(out_path, merged)
    return merged


def load_merged_chunks(path: Path) -> List[MergedChunk]:
    """Read `merged_chunks.json` back in file order."""
    data: Dict[str, MergedChunk] = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.values())


def save_merged_chunks(path: Path, chunks: Sequence[MergedChunk]) -> None:
    """Write `merged_chunks.json` (keys chunk1..N, document order)."""
    atomic_write_json(path, {f"chunk{i}": chunk for i, chunk in enumerate(chunks, start=1)})


# --------------------------------------------------------------------------- #
# Chain building
# --------------------------------------------------------------------------- #

def _build_chains(chunks: List[ChunkRecord], log: logging.LoggerAdapter) -> List[Union[ChunkRecord, List[ChunkRecord]]]:
    """Group chunks into standalone records and ordered chains (lists of ≥ 2)."""
    # A seam continues into the *first* chunk that claims it (the top of the next page); a later
    # chunk repeating the id is a model slip and stays standalone.
    receivers: Dict[str, ChunkRecord] = {}
    for c in chunks:
        if c.get("continuation_type") in _RECEIVERS and c.get("incoming_group_id"):
            receivers.setdefault(c["incoming_group_id"], c)
    position = {id(c): i for i, c in enumerate(chunks)}
    consumed: set = set()
    result: List[Union[ChunkRecord, List[ChunkRecord]]] = []

    for chunk in chunks:
        if id(chunk) in consumed:
            continue
        if chunk.get("continuation_type") not in _OPENERS:
            result.append(chunk)  # standalone, or an incoming half whose opener never came
            continue

        chain = [chunk]
        consumed.add(id(chunk))
        current, absorptions = chunk, 0
        while True:
            partner = receivers.get(current.get("outgoing_group_id") or "")
            if partner is None or id(partner) in consumed:
                absorbed = _next_unconsumed_outgoing(chunks, position[id(current)] + 1, consumed)
                if absorbed is None or absorptions >= _MAX_FORCED_ABSORPTIONS:
                    break
                log.debug("No seam partner for %s — absorbing %s", current.get("chunk_id"), absorbed.get("chunk_id"))
                chain.append(absorbed)
                consumed.add(id(absorbed))
                current, absorptions = absorbed, absorptions + 1
                continue
            chain.append(partner)
            consumed.add(id(partner))
            if partner.get("continuation_type") not in _OPENERS:
                break
            current = partner

        result.append(chain if len(chain) > 1 else chain[0])
    return result


def _next_unconsumed_outgoing(chunks: List[ChunkRecord], start: int, consumed: set) -> Optional[ChunkRecord]:
    for candidate in chunks[start:]:
        if id(candidate) in consumed:
            continue
        return candidate if candidate.get("continuation_type") == "outgoing" else None
    return None


# --------------------------------------------------------------------------- #
# Shaping — the only two places a MergedChunk is built
# --------------------------------------------------------------------------- #

def _fuse(chain: List[ChunkRecord]) -> MergedChunk:
    first_page, last_page = _page_of(chain[0]["chunk_id"]), _page_of(chain[-1]["chunk_id"])
    chunk_id = f"page_{first_page}_{last_page}_chunk" if first_page and last_page else chain[0]["chunk_id"]
    metadata: Dict[str, Any] = {
        "source_chunk_ids": [c["chunk_id"] for c in chain],
        "source_page_nos": _dedupe([str(c["page_no"]) for c in chain if c.get("page_no") is not None]),
        "merge_reason": _MERGE_REASON,
    }
    _attach_page_metadata(metadata, chain)
    return {
        "chunk_id": chunk_id,
        "content": _TEXT_JOINER.join((c.get("text") or "").strip("\n") for c in chain),
        "chunk_description": _DESC_JOINER.join(d for d in ((c.get("chunk_description") or "").strip() for c in chain) if d),
        "concept_vocabulary": _dedupe_vocab(v for c in chain for v in c.get("index_concept_vocabulary") or [])[:_VOCAB_CAP],
        "images": _dedupe([img for c in chain for img in c.get("images") or []]),
        "metadata": metadata,
    }


def _standalone(chunk: ChunkRecord) -> MergedChunk:
    metadata: Dict[str, Any] = {
        "source_chunk_ids": [chunk["chunk_id"]],
        "source_page_nos": [str(chunk["page_no"])] if chunk.get("page_no") is not None else [],
    }
    _attach_page_metadata(metadata, [chunk])
    return {
        "chunk_id": chunk["chunk_id"],
        "content": chunk.get("text") or "",
        "chunk_description": chunk.get("chunk_description") or "",
        "concept_vocabulary": list(chunk.get("index_concept_vocabulary") or []),
        "images": _dedupe(chunk.get("images") or []),
        "metadata": metadata,
    }


def _unique_ids(chunks: List[MergedChunk], log: logging.LoggerAdapter) -> List[MergedChunk]:
    """
    Two fused chains over the same page span would share `page_{a}_{b}_chunk`; the id keys the
    Milvus row and the classification, so later ones get `_2`, `_3`, … (deterministic: document order).
    """
    seen: Dict[str, int] = {}
    for chunk in chunks:
        base = chunk["chunk_id"]
        count = seen.get(base, 0) + 1
        seen[base] = count
        if count > 1:
            chunk["chunk_id"] = f"{base}_{count}"
            while chunk["chunk_id"] in seen:
                count += 1
                chunk["chunk_id"] = f"{base}_{count}"
            seen[chunk["chunk_id"]] = 1
            log.warning("Merged chunk id %s already used; renamed to %s", base, chunk["chunk_id"])
    return chunks


def _attach_page_metadata(metadata: Dict[str, Any], chain: Sequence[ChunkRecord]) -> None:
    corrections = _dedupe([c for chunk in chain for c in chunk.get("ocr_corrections") or []])
    continuations = _dedupe([chunk["content_continuation"] for chunk in chain if chunk.get("content_continuation")])
    if corrections:
        metadata["ocr_corrections"] = corrections
    if continuations:
        metadata["content_continuation"] = continuations


def _page_of(chunk_id: str) -> Optional[str]:
    match = _PAGE_FROM_ID.match(chunk_id or "")
    return match.group(1) if match else None


def _dedupe(items: Sequence[Any]) -> List[Any]:
    """Order-preserving dedupe that also handles unhashable dicts/lists."""
    seen, out = set(), []
    for item in items:
        key = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _dedupe_vocab(terms: Any) -> List[str]:
    """Case-insensitive, order-preserving dedupe of stripped, non-empty terms."""
    seen, out = set(), []
    for term in terms:
        cleaned = str(term).strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out
