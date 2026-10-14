"""
app.retrieval.filters
=====================
CategoryFilter — narrows a query to chunks with given classification categories.

    main_categories : the chunk's main category is one of these
    sub_categories  : the chunk has at least one of these as a sub category
    categories      : either of the above — "the chunk is about any of these at all"

Values within one list are OR-ed; the lists given are AND-ed. The expression is a Milvus
boolean filter applied to every arm (dense, BM25, server-side hybrid) *before* ranking, so
`top_k` is always filled from matching chunks. It matches typed columns (schema v2) and
dynamic-field keys (schema v1) alike. Values are normalised with `normalize_category_name`,
which leaves only `[a-z0-9_]` — nothing a caller sends can alter the expression's structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from app.retrieval.milvus_schema import F_MAIN_CATEGORY, F_SUB_CATEGORIES
from app.utils.naming import normalize_category_name

__all__ = ["CategoryFilter", "normalize_categories"]


def normalize_categories(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """Normalised, de-duplicated, order-preserving; raises ValueError on an unusable name."""
    return tuple(dict.fromkeys(normalize_category_name(value) for value in values or ()))


@dataclass(frozen=True)
class CategoryFilter:
    main_categories: Tuple[str, ...] = ()
    sub_categories: Tuple[str, ...] = ()
    categories: Tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        main_categories: Optional[Iterable[str]] = None,
        sub_categories: Optional[Iterable[str]] = None,
        categories: Optional[Iterable[str]] = None,
    ) -> Optional["CategoryFilter"]:
        """A filter from raw request values, or None when no value was given. Raises ValueError."""
        built = cls(normalize_categories(main_categories), normalize_categories(sub_categories), normalize_categories(categories))
        return built if built else None

    def __bool__(self) -> bool:
        return bool(self.main_categories or self.sub_categories or self.categories)

    def expression(self) -> str:
        """The Milvus filter expression; empty when the filter is empty."""
        clauses = []
        if self.main_categories:
            clauses.append(_main_in(self.main_categories))
        if self.sub_categories:
            clauses.append(_sub_any(self.sub_categories))
        if self.categories:
            clauses.append(f"({_main_in(self.categories)} or {_sub_any(self.categories)})")
        return " and ".join(clauses)


def _main_in(values: Tuple[str, ...]) -> str:
    return f"{F_MAIN_CATEGORY} in {json.dumps(list(values))}"


def _sub_any(values: Tuple[str, ...]) -> str:
    return f"array_contains_any({F_SUB_CATEGORIES}, {json.dumps(list(values))})"
