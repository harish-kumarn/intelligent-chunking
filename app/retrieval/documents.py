"""
app.retrieval.documents
=======================
MergedChunk ─► LangChain Document, the shape the retriever indexes.

    page_content : vocabulary + description + text, so all three influence the embedding and BM25
    metadata     : chunk_id, document_id, page, image_paths, reference_path (scalar-column friendly),
                   main_category / sub_categories when stage 5 classified the chunk (filter fields)

Chunk ids are minted per document (`page_1_chunk_1`, …), so the Milvus primary key is
`{document_id}:{chunk_id}`: several documents share a collection without clobbering each
other, and re-ingesting one document still replaces exactly its own rows.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from langchain_core.documents import Document

from app.postprocessing.merger import MergedChunk
from app.retrieval.milvus_schema import F_MAIN_CATEGORY, F_SUB_CATEGORIES

__all__ = ["merged_chunks_to_documents", "scoped_chunk_id"]

_SECTION_SEP = "\n\n"
_ID_SEP = ":"


def scoped_chunk_id(document_id: Optional[str], chunk_id: str) -> str:
    """Collection-unique primary key; bare `chunk_id` when no document id is known."""
    return f"{document_id}{_ID_SEP}{chunk_id}" if document_id else chunk_id


def merged_chunks_to_documents(chunks: Iterable[MergedChunk], document_id: Optional[str] = None) -> List[Document]:
    """
    One Document per merged chunk; chunks without text are still indexed (images/description).
    `document_id` scopes the primary key so documents can share a collection.
    """
    return [_to_document(chunk, document_id) for chunk in chunks]


def _to_document(chunk: MergedChunk, document_id: Optional[str]) -> Document:
    source_pages = (chunk.get("metadata") or {}).get("source_page_nos") or []
    page = _first_int(source_pages)
    vocab = ", ".join(str(v).strip() for v in chunk.get("concept_vocabulary") or [] if str(v).strip())
    description = (chunk.get("chunk_description") or "").strip()

    sections = []
    if vocab:
        sections.append(f"INDEX_CONCEPT_VOCABULARY: {vocab}")
    if description:
        sections.append(f"CHUNK_DESCRIPTION: {description}")
    sections.append(f"CHUNK: {chunk.get('content') or ''}")

    metadata = {
        "chunk_id": scoped_chunk_id(document_id, chunk["chunk_id"]),
        "document_id": document_id or "",
        "page": page,
        "image_paths": [img["path"] if isinstance(img, dict) else str(img) for img in chunk.get("images") or []],
        "reference_path": f"reference_imgs/reference_page_{page}.png" if page is not None else "",
    }
    # Only classified chunks carry categories; a None main category is not stored (NULL in Milvus).
    for key in (F_MAIN_CATEGORY, F_SUB_CATEGORIES):
        if key in chunk:
            metadata[key] = chunk[key]
    return Document(page_content=_SECTION_SEP.join(sections), metadata=metadata)


def _first_int(values: Iterable) -> int | None:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
