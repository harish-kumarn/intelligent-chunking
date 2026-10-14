"""
app.utils.json_utils
====================
Tolerant parsing of LLM JSON output.

Models occasionally wrap JSON in fences, leave raw newlines inside strings,
or emit trailing commas. `repair_and_parse_json` applies a fixed sequence of
increasingly aggressive repairs and raises ValueError only when none yields
a JSON object.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict

__all__ = ["repair_and_parse_json"]

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_MISSING_COMMA_AFTER_CLOSE = re.compile(r"([}\]])\s*([{\"])")
_MISSING_COMMA_AFTER_VALUE = re.compile(r"(\"|\d+|true|false|null)\s*\n\s*(\"|\{|\[)")
_EXCERPT_LEN = 800


def repair_and_parse_json(raw: str) -> Dict[str, Any]:
    """Parse `raw` into a dict, repairing common LLM syntax slips. Raises ValueError."""
    candidate = _outermost_object(_strip_fences(raw.strip()))
    escaped = _escape_control_chars_in_strings(candidate)
    no_trailing_commas = _TRAILING_COMMA.sub(r"\1", escaped)
    commas_inserted = _insert_missing_commas(no_trailing_commas)

    for attempt in (candidate, escaped, no_trailing_commas, commas_inserted):
        parsed = _try_json(attempt)
        if parsed is not None:
            return parsed

    parsed = _try_literal(commas_inserted)  # last resort: Python-literal syntax (single quotes, True/None)
    if parsed is not None:
        return parsed
    raise ValueError(f"Cannot parse response as a JSON object.\n--- Excerpt ---\n{candidate[:_EXCERPT_LEN]}")


# --------------------------------------------------------------------------- #
# Repair steps — each is pure and idempotent
# --------------------------------------------------------------------------- #

def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text)).strip()
    return text


def _outermost_object(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


def _escape_control_chars_in_strings(text: str) -> str:
    """Replace raw newlines/tabs inside JSON string literals with their escapes."""
    out = []
    in_string = escaped = False
    for char in text:
        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\" and in_string:
            out.append(char)
            escaped = True
        elif char == '"':
            out.append(char)
            in_string = not in_string
        elif in_string and char in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[char])
        else:
            out.append(char)
    return "".join(out)


def _insert_missing_commas(text: str) -> str:
    text = _MISSING_COMMA_AFTER_CLOSE.sub(r"\1, \2", text)
    return _MISSING_COMMA_AFTER_VALUE.sub(r"\1,\n\2", text)


def _try_json(text: str) -> Any:
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _try_literal(text: str) -> Any:
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None
