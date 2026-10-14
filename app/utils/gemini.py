"""
app.utils.gemini
================
`gemini_client` — the one place a google-genai client is built, so every Gemini stage gets a
request timeout. Without one a stalled HTTP call blocks its page (and the single-flight
ingestion gate) forever; with one it fails and the caller's retry loop takes over.
"""

from __future__ import annotations

from google import genai
from google.genai import types as genai_types

__all__ = ["gemini_client"]

_MS_PER_SECOND = 1000


def gemini_client(api_key: str, timeout_seconds: float) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(timeout=int(timeout_seconds * _MS_PER_SECOND)),
    )
