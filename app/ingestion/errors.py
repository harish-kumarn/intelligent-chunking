"""
app.ingestion.errors
====================
Pipeline errors that carry more than a message.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["IncompleteStageError", "NoContentError"]

_LISTED = 20  # items quoted in the message


class IncompleteStageError(RuntimeError):
    """
    A stage processed every item but some failed every attempt. Their results are absent, the
    finished ones are kept, so a rerun (a job retry or a resubmit) processes only `failed`.
    """

    def __init__(self, stage: str, failed: Dict[int, str], total: int) -> None:
        self.stage = stage
        self.failed = dict(sorted(failed.items()))
        self.total = total
        first = next(iter(self.failed.values()), "")
        super().__init__(
            f"Stage {stage}: {len(self.failed)} of {total} page(s) failed after retries: "
            f"{list(self.failed)[:_LISTED]} — a rerun processes only these (first error: {first})"
        )


class NoContentError(Exception):
    """
    The document produced no indexable chunk (blank, or nothing the chunker could use). Not retryable:
    every page was chunked and stored, so a rerun gives the same answer. The API maps it to 422.
    """

    code = "NO_CHUNKS"

    def __init__(self, document: str, page_count: int) -> None:
        super().__init__(f"Document '{document}' produced no indexable content ({page_count} page(s), 0 chunks); nothing was indexed")
