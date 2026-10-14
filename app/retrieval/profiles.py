"""
app.retrieval.profiles
======================
LanguageProfile — everything about retrieval that depends on a collection's language.

    language          analyzer (BM25 `text` field)                 dense / rerank models (ChunkerConfig)
    "en"              standard + lowercase + English stop words    embedding_model / reranking_model
    "ja"              Lindera IPADIC + lowercase                   japanese_embedding_model / japanese_reranking_model

A collection holds one language (see app.retrieval.language): its analyzer is fixed
by Milvus at creation, and its dense vectors must all come from one embedding model.
Both are recorded in the collection manifest (app.retrieval.milvus_schema), so a
query is always analyzed and embedded the way the collection was indexed.

Lindera needs Milvus 2.6+ (the 2.5.x images ship no Lindera dictionary). Milvus
downloads the IPADIC dictionary (~45 MB) on first use, so the Milvus container needs
outbound network access once; `milvus_schema.warm_up_japanese_analyzer` triggers that
at service start.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from app.config.defaults import ChunkerConfig
from app.retrieval.language import ENGLISH, JAPANESE, SUPPORTED_LANGUAGES, normalize_language

__all__ = ["ENGLISH_ANALYZER", "JAPANESE_ANALYZER", "LanguageProfile", "language_profile"]

# Unchanged from every collection created before language profiles existed.
ENGLISH_ANALYZER: Mapping[str, Any] = MappingProxyType({
    "tokenizer": "standard",
    "filter": ["lowercase", {"type": "stop", "stop_words": ["_english_"]}],
})
# Morphological segmentation (刈刃取付用ナット -> 刈 / 刃 / 取付 / 用 / ナット). `lowercase` keeps
# Latin product codes case-insensitive: BCL200S and bcl200s both become bcl / 200 / s.
JAPANESE_ANALYZER: Mapping[str, Any] = MappingProxyType({
    "tokenizer": {"type": "lindera", "dict_kind": "ipadic"},
    "filter": ["lowercase"],
})


@dataclass(frozen=True)
class LanguageProfile:
    language: str
    analyzer: Mapping[str, Any]
    embedding_model: str
    reranking_model: str

    def analyzer_params(self) -> dict:
        """A plain, mutable copy for pymilvus."""
        return {key: (list(val) if isinstance(val, list) else val) for key, val in self.analyzer.items()}


def language_profile(language: str, config: ChunkerConfig) -> LanguageProfile:
    """The profile for `language` ("ja" / "en") under `config`. Raises ValueError for anything else."""
    code = normalize_language(language)
    if code == JAPANESE:
        return LanguageProfile(JAPANESE, JAPANESE_ANALYZER, config.japanese_embedding_model, config.japanese_reranking_model)
    if code == ENGLISH:
        return LanguageProfile(ENGLISH, ENGLISH_ANALYZER, config.embedding_model, config.reranking_model)
    raise ValueError(f"Unsupported language {language!r}; expected one of {sorted(SUPPORTED_LANGUAGES)}")
