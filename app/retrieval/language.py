"""
app.retrieval.language
======================
Which language a document is written in — decided once per PDF, at ingestion.

A PDF is either Japanese or English as a whole: Japanese manuals contain Japanese
prose with the odd Latin product code (BCL200S), English manuals contain no
Japanese at all. So detection is a script count, not a model:

    japanese = hiragana + katakana + CJK ideographs   (full- and half-width)
    latin    = A–Z / a–z                               (full- and half-width)
    "ja"  when japanese / (japanese + latin) >= threshold
    "en"  otherwise, and for text with no letters at all

The share is taken over *letters*, so a Japanese page full of part numbers,
tables and product codes still reads as Japanese.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["ENGLISH", "JAPANESE", "SUPPORTED_LANGUAGES", "detect_language", "normalize_language"]

JAPANESE = "ja"
ENGLISH = "en"
SUPPORTED_LANGUAGES = frozenset({JAPANESE, ENGLISH})

_JAPANESE_CHARS = re.compile(
    "["
    "぀-ゟ"  # hiragana
    "゠-ヿ"  # katakana
    "ㇰ-ㇿ"  # katakana phonetic extensions
    "㐀-䶿"  # CJK extension A
    "一-鿿"  # CJK unified ideographs
    "豈-﫿"  # CJK compatibility ideographs
    "ｦ-ﾟ"  # half-width katakana
    "]"
)
_LATIN_CHARS = re.compile("[A-Za-zＡ-Ｚａ-ｚ]")  # ASCII + full-width Latin


def normalize_language(code: Optional[str]) -> str:
    """Canonical form of a language code: stripped, lower-case ("JA " -> "ja")."""
    return (code or "").strip().lower()


def detect_language(text: str, *, japanese_threshold: float = 0.2) -> str:
    """"ja" when Japanese script makes up at least `japanese_threshold` of the letters, else "en"."""
    japanese = len(_JAPANESE_CHARS.findall(text or ""))
    latin = len(_LATIN_CHARS.findall(text or ""))
    if japanese + latin == 0:
        return ENGLISH
    return JAPANESE if japanese / (japanese + latin) >= japanese_threshold else ENGLISH
