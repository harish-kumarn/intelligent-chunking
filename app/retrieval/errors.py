"""Client-caused retrieval failures; the API layer maps them to 404 / 409."""

from __future__ import annotations

__all__ = [
    "CollectionLanguageMismatchError", "CollectionNotFoundError", "LegacyCollectionError", "RetrievalError",
]


class RetrievalError(Exception):
    """Non-retryable failure caused by the request, not by Milvus or Cohere."""

    code = "RETRIEVAL_ERROR"


class CollectionNotFoundError(RetrievalError):
    code = "COLLECTION_NOT_FOUND"

    def __init__(self, collection_name: str) -> None:
        super().__init__(f"Collection '{collection_name}' does not exist.")


class CollectionLanguageMismatchError(RetrievalError):
    """A document's language differs from the language its target collection was created with."""

    code = "COLLECTION_LANGUAGE_MISMATCH"

    def __init__(self, collection_name: str, collection_language: str, document_language: str) -> None:
        super().__init__(
            f"Collection '{collection_name}' holds '{collection_language}' documents; this document is "
            f"'{document_language}'. Use a different collection for '{document_language}' documents."
        )


class LegacyCollectionError(RetrievalError):
    """The collection predates language profiles, so its schema cannot take new rows."""

    code = "LEGACY_COLLECTION"

    def __init__(self, collection_name: str) -> None:
        super().__init__(
            f"Collection '{collection_name}' was created by an older version of this service and is read-only. "
            "Ingest into a new collection, or drop this one and re-ingest its documents."
        )
