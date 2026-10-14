"""
app.utils.naming
================
Identifier normalisation shared by the API layer, the retriever and chunk classification.

    collection names : the API's one rule — `normalize_collection_name`: trimmed, lower-cased, then
                       `^[a-z][a-z0-9_]{2,63}$`, else rejected (never silently rewritten)
    identifiers      : `sanitize_collection_name` coerces free text (a PDF file stem → document_id) into
                       a Milvus-valid `[A-Za-z0-9_]` name
    category names   : lowercase ASCII snake_case — safe to embed in a Milvus filter expression
"""

from __future__ import annotations

import re

__all__ = [
    "COLLECTION_NAME_RULE", "MAX_CATEGORIES", "MAX_CATEGORY_NAME_LEN", "normalize_category_name", "normalize_collection_name",
    "sanitize_collection_name",
]

_MAX_COLLECTION_NAME_LEN = 255  # Milvus limit
_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9_]")
_UNDERSCORE_RUNS = re.compile(r"_+")

_COLLECTION_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
COLLECTION_NAME_RULE = "3-64 characters: lowercase letters, digits and '_', starting with a letter"

MAX_CATEGORY_NAME_LEN = 64  # well below the Milvus VARCHAR column declared for categories
MAX_CATEGORIES = 32         # cap on general categories, hence on a chunk's sub-categories
_NON_CATEGORY_CHARS = re.compile(r"[^a-z0-9]+")


def normalize_collection_name(name: str) -> str:
    """
    `" Manuals_EN "` → `"manuals_en"`. Every collection name the API accepts passes through here, so
    one collection is always spelled one way. Raises ValueError (naming the rule) for anything that does
    not match after trimming and lower-casing — a name is never rewritten into a different one.
    """
    if not isinstance(name, str):
        raise ValueError("Collection name must be a string.")
    cleaned = name.strip().lower()
    if not _COLLECTION_NAME.match(cleaned):
        raise ValueError(f"Invalid collection name {name!r}: use {COLLECTION_NAME_RULE} (e.g. 'manuals_en').")
    return cleaned


def sanitize_collection_name(name: str) -> str:
    """
    Coerce `name` into a valid Milvus collection name: `[A-Za-z0-9_]`, not
    starting with a digit, at most 255 characters. Raises ValueError when
    nothing usable remains.
    """
    if not isinstance(name, str):
        raise ValueError("Collection name must be a string.")
    cleaned = _UNDERSCORE_RUNS.sub("_", _INVALID_CHARS.sub("_", name.strip())).strip("_")
    if not cleaned:
        raise ValueError("Collection name contains no valid characters.")
    if cleaned[0].isdigit():
        cleaned = f"collection_{cleaned}"
    return cleaned[:_MAX_COLLECTION_NAME_LEN]


def normalize_category_name(name: str) -> str:
    """
    `"Trouble-Shooting "` → `"trouble_shooting"`: lowercase ASCII words joined by `_`, at most
    `MAX_CATEGORY_NAME_LEN` characters. Raises ValueError when nothing usable remains — the
    result only ever contains `[a-z0-9_]`, so it can be quoted into a filter expression as-is.
    """
    if not isinstance(name, str):
        raise ValueError("Category name must be a string.")
    cleaned = _NON_CATEGORY_CHARS.sub("_", name.strip().lower()).strip("_")[:MAX_CATEGORY_NAME_LEN].rstrip("_")
    if not cleaned:
        raise ValueError(f"Category name {name!r} contains no ASCII letters or digits.")
    return cleaned
