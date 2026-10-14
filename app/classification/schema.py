"""
app.classification.schema
=========================
Data contracts for chunk classification.

    ClassificationResponse : the structured output Gemini is constrained to emit for one chunk
    ClassificationBatchResponse : the same for several chunks in one call (each item names its chunk_id)
    ChunkClassification    : one validated assignment, as kept in the memory
    CreatedCategory        : a main category the model introduced for this document
    MemorySnapshot         : the whole memory, as persisted in classification_memory.json

Field names here are the field names in prompts/*.md and in the artifact — there are no
aliases. `main_category` / `sub_categories` are also the keys written onto every chunk
and the Milvus fields the retriever filters on.
"""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BatchClassification", "CategorySource", "ChunkClassification", "ClassificationBatchResponse",
    "ClassificationResponse", "CreatedCategory", "MemorySnapshot",
    "F_MAIN_CATEGORY", "F_SUB_CATEGORIES",
]

F_MAIN_CATEGORY = "main_category"
F_SUB_CATEGORIES = "sub_categories"

CategorySource = Literal["general", "created"]


# --------------------------------------------------------------------------- #
# Gemini response schema (passed as `response_schema`)
# --------------------------------------------------------------------------- #

class ClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reasoning: str = ""                  # emitted first so the model decides before it labels
    main_category: str
    new_category_description: str = ""   # set only when `main_category` is a new category
    sub_categories: List[str] = Field(default_factory=list)


class BatchClassification(ClassificationResponse):
    chunk_id: str


class ClassificationBatchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    classifications: List[BatchClassification] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Persisted memory
# --------------------------------------------------------------------------- #

class ChunkClassification(BaseModel):
    """A validated assignment: the main category is general or created, sub categories are general only."""

    fingerprint: str                     # of the classified content; a changed chunk is classified again
    main_category: str
    main_category_source: CategorySource
    sub_categories: List[str] = Field(default_factory=list)
    reasoning: str = ""


class CreatedCategory(BaseModel):
    description: str = ""
    first_chunk_id: str


class MemorySnapshot(BaseModel):
    """`classification_memory.json`. General and created categories are kept apart on purpose."""

    general_categories: Dict[str, str] = Field(default_factory=dict)
    created_categories: Dict[str, CreatedCategory] = Field(default_factory=dict)
    chunks: Dict[str, ChunkClassification] = Field(default_factory=dict)
    failures: Dict[str, str] = Field(default_factory=dict)
