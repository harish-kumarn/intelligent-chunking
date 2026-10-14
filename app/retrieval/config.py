"""
visionrag_chunker.retrieval.config
==================================
Single, immutable configuration for the hybrid retrieval pipeline.

`RetrievalConfig` is the only knob surface the retriever reads. It is frozen so
one instance can be shared safely across concurrent requests; per-request
overrides produce a new instance via `with_overrides()`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class FusionStrategy(str, Enum):
    """How the dense and sparse arms are combined into one ranking."""

    RRF = "rrf"            # Reciprocal Rank Fusion — rank-based, score-scale agnostic
    WEIGHTED = "weighted"  # Convex blend of per-arm normalised scores
    DBSF = "dbsf"          # Distribution-Based Score Fusion (z-score normalisation)


class FusionMode(str, Enum):
    """Where fusion is executed."""

    CLIENT = "client"  # Two Milvus searches fused in-process. Full per-arm telemetry.
    SERVER = "server"  # One Milvus `hybrid_search` with a native ranker. No per-arm telemetry.


# Strategies Milvus can execute server-side (RRFRanker / WeightedRanker).
_SERVER_STRATEGIES = frozenset({FusionStrategy.RRF, FusionStrategy.WEIGHTED})


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Hyper-parameters for one retrieval run.

    Depth invariants are enforced on construction so a request can never ask
    for more results than the pipeline searches for:

        final_top_k <= rerank_candidates <= dense_top_k / sparse_top_k
    """

    # Candidate pool depths
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rerank_candidates: int = 20
    final_top_k: int = 3

    # Pipeline switches
    use_hybrid: bool = True
    use_reranking: bool = True

    # Fusion
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    fusion_mode: FusionMode = FusionMode.CLIENT
    alpha: float = 0.5         # dense share of the fused score, in [0, 1]; 0.5 won on the Japanese and English-model benchmark arms
    rrf_k: int = 60            # RRF smoothing constant
    dbsf_sigma: float = 3.0    # DBSF clipping width in standard deviations

    # Milvus / runtime
    drop_ratio_search: float = 0.0  # BM25: fraction of low-weight query terms to drop
    executor_workers: int = 8

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # frozen dataclass — normalise in place once

        set_(self, "fusion_strategy", FusionStrategy(self.fusion_strategy))
        set_(self, "fusion_mode", FusionMode(self.fusion_mode))

        for name in ("dense_top_k", "sparse_top_k", "rerank_candidates", "final_top_k", "rrf_k", "executor_workers"):
            _require_positive_int(name, getattr(self, name))
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.dbsf_sigma <= 0:
            raise ValueError(f"dbsf_sigma must be > 0, got {self.dbsf_sigma}")
        if not 0.0 <= self.drop_ratio_search <= 1.0:
            raise ValueError(f"drop_ratio_search must be in [0, 1], got {self.drop_ratio_search}")

        # Widen upstream depths so the requested final_top_k is always reachable.
        rerank = max(self.rerank_candidates, self.final_top_k)
        set_(self, "rerank_candidates", rerank)
        set_(self, "dense_top_k", max(self.dense_top_k, rerank))
        set_(self, "sparse_top_k", max(self.sparse_top_k, rerank))

    @property
    def effective_fusion_mode(self) -> FusionMode:
        """DBSF has no Milvus-native ranker, so it always runs client-side."""
        if self.fusion_strategy not in _SERVER_STRATEGIES:
            return FusionMode.CLIENT
        return self.fusion_mode

    @property
    def dense_weight(self) -> float:
        return self.alpha

    @property
    def sparse_weight(self) -> float:
        return 1.0 - self.alpha

    def with_overrides(self, **overrides: Any) -> "RetrievalConfig":
        """Return a copy with the non-None overrides applied and re-validated."""
        changes = {key: val for key, val in overrides.items() if val is not None}
        return replace(self, **changes) if changes else self


def _require_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
