"""
app.api.errors
==============
One shape for every error body: `{"error": {"code", "message", "retryable"}}`.
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

__all__ = ["http_error"]


def http_error(status_code: int, code: str, message: str, *, retryable: bool = False, retry_after: Optional[int] = None) -> HTTPException:
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return HTTPException(
        status_code=status_code,
        detail={"error": {"code": code, "message": message, "retryable": retryable}},
        headers=headers,
    )
