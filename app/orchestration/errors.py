"""Client-caused channel failures; the API layer answers with each one's `http_status` (400 unless stated)."""

from __future__ import annotations

__all__ = [
    "ChannelError", "DocumentIdConflictError", "InvalidDocumentError", "S3DocumentNotFoundError", "UnsupportedSourceTypeError",
]


class ChannelError(Exception):
    """Non-retryable, client-caused failure."""

    code = "CHANNEL_ERROR"
    http_status = 400


class UnsupportedSourceTypeError(ChannelError):
    code = "UNSUPPORTED_SOURCE_TYPE"


class InvalidDocumentError(ChannelError):
    """Not a PDF, empty, over the size limit, or missing a required parameter."""

    code = "INVALID_DOCUMENT"


class S3DocumentNotFoundError(ChannelError):
    """The S3 folder has no published version, or that version holds no indexable chunks."""

    code = "S3_DOCUMENT_NOT_FOUND"
    http_status = 404


class DocumentIdConflictError(ChannelError):
    """Another PDF (different content) is already published under this document id — a same-named file."""

    code = "DOCUMENT_ID_CONFLICT"
    http_status = 409

    def __init__(self, document_id: str) -> None:
        super().__init__(
            f"A different PDF is already published as document '{document_id}' (same file name, other content). "
            "Rename the file, or send replace=true to replace that document everywhere it is indexed."
        )
