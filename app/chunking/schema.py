"""
app.chunking.schema
===================
Data contracts for one page of vision chunking.

    ChunkingInput    : what the chunker needs to process a page
    ChunkingResponse : the structured output Gemini is constrained to emit
    ChunkRecord      : one chunk as persisted in target_chunks.json

Field names here are the field names in the system prompt and in every
downstream artifact — there are no aliases, so nothing needs to guess keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Chunk", "ChunkRecord", "ChunkingInput", "ChunkingResponse", "ContinuationType", "PageMetadata", "canonical_chunk_id"]

ContinuationType = Literal["none", "incoming", "outgoing", "bidirectional"]
ChunkRecord = Dict[str, Any]


# --------------------------------------------------------------------------- #
# Gemini response schema (passed as `response_schema`)
# --------------------------------------------------------------------------- #

class OCRCorrection(BaseModel):
    original: str
    corrected: str
    confidence: float
    context: Optional[str] = None


class ContentContinuation(BaseModel):
    from_previous_page: bool = False
    to_next_page: bool = False


class Boilerplate(BaseModel):
    header: Optional[str] = None
    footer: Optional[str] = None
    page_marker: Optional[str] = None
    other: List[str] = Field(default_factory=list)


class PageMetadata(BaseModel):
    page_number: int
    boilerplate: Optional[Boilerplate] = None
    ocr_corrections: List[OCRCorrection] = Field(default_factory=list)
    content_continuation: ContentContinuation = Field(default_factory=ContentContinuation)


class ImageRef(BaseModel):
    path: str
    association: Literal["semantic", "nearest_fallback"] = "semantic"


class Chunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    text: str = ""
    images: List[ImageRef] = Field(default_factory=list)
    chunk_description: str = ""
    continuation_type: ContinuationType = "none"
    incoming_group_id: Optional[str] = None
    outgoing_group_id: Optional[str] = None
    oversized: bool = False
    index_concept_vocabulary: List[str] = Field(default_factory=list)


class ChunkingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunks: List[Chunk] = Field(default_factory=list)
    metadata: Optional[PageMetadata] = None

    def to_records(self, page_no: int) -> List[ChunkRecord]:
        """
        Flatten into per-chunk records carrying the page-level metadata. `chunk_id` is set to the
        prompt's `page_{N}_chunk_{M}` from the page number and emission order, whatever the model
        wrote: ids key the Milvus rows and the classification, so they must be unique per document.
        Seams are linked by the group ids, never by chunk ids, so this changes no merge.
        """
        meta = self.metadata or PageMetadata(page_number=page_no)
        shared = {
            "page_no": page_no,
            "ocr_corrections": [c.model_dump() for c in meta.ocr_corrections],
            "content_continuation": meta.content_continuation.model_dump(),
        }
        return [
            {**chunk.model_dump(), "chunk_id": canonical_chunk_id(page_no, index), **shared}
            for index, chunk in enumerate(self.chunks, start=1)
        ]


def canonical_chunk_id(page_no: int, index: int) -> str:
    return f"page_{page_no}_chunk_{index}"


# --------------------------------------------------------------------------- #
# Chunker input
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChunkingInput:
    """Everything the chunker needs for one page."""

    page_number: int
    page_text: str
    page_image_path: Path
    previous_page_text: Optional[str] = None  # trailing half of page N-1, boundary context only
    next_page_text: Optional[str] = None      # leading half of page N+1, boundary context only
