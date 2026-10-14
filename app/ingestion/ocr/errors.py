"""
app.ingestion.ocr.errors
========================
OCR failures that retrying cannot fix — `run_ocr` retries every other error.
"""

from __future__ import annotations

__all__ = ["OCRConfigurationError", "OCRResultError"]


class OCRConfigurationError(RuntimeError):
    """The selected engine is not configured (missing credentials or SDK)."""


class OCRResultError(RuntimeError):
    """The engine answered, but with nothing usable for this document."""
