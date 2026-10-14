"""
visionrag_chunker.retrieval.fusion
==================================
Client-side fusion of a dense and a sparse (BM25) ranking.

Every strategy takes two ranked hit lists (best first) plus a
`RetrievalConfig` and returns one deduplicated, re-scored list sorted best
first. All functions are pure — no I/O, no shared state — and therefore
trivially unit-testable.

A *hit* is a plain dict with at least:
    chunk_id, text, page, metadata, score
Fusion adds:
    final_score, retrieval_source, and per-arm diagnostics
    (dense_score / dense_rank / sparse_score / sparse_rank when present).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from app.retrieval.config import FusionStrategy, RetrievalConfig

Hit = Dict[str, Any]
_Normaliser = Callable[[Sequence[float]], Callable[[float], float]]


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

def fuse_rrf(dense: List[Hit], sparse: List[Hit], cfg: RetrievalConfig) -> List[Hit]:
    """
    Weighted Reciprocal Rank Fusion.

        score = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)

    Rank-based, so it is immune to the different score scales of cosine
    similarity and BM25. The most robust default for hybrid retrieval.
    """
    k = cfg.rrf_k

    def rrf(rank: int) -> float:
        return 1.0 / (k + rank)

    return _combine(dense, sparse, cfg, rrf, rrf, source="hybrid_rrf")


def fuse_weighted(dense: List[Hit], sparse: List[Hit], cfg: RetrievalConfig) -> List[Hit]:
    """
    Convex blend of min-max normalised scores.

        score = alpha * norm(dense) + (1 - alpha) * norm(sparse)

    Preserves score magnitude within each arm, so a dominant dense match
    stays dominant. Client-side equivalent of Milvus' `WeightedRanker`.
    """
    d_norm = _minmax([h["score"] for h in dense])
    s_norm = _minmax([h["score"] for h in sparse])
    return _combine(dense, sparse, cfg, d_norm, s_norm, source="hybrid_weighted", by_rank=False)


def fuse_dbsf(dense: List[Hit], sparse: List[Hit], cfg: RetrievalConfig) -> List[Hit]:
    """
    Distribution-Based Score Fusion.

    Each arm is normalised against its own score distribution
    (mean ± sigma·std, clipped to [0, 1]) before the alpha blend. More
    outlier-tolerant than min-max when one arm has a single runaway score.
    """
    d_norm = _zscore([h["score"] for h in dense], cfg.dbsf_sigma)
    s_norm = _zscore([h["score"] for h in sparse], cfg.dbsf_sigma)
    return _combine(dense, sparse, cfg, d_norm, s_norm, source="hybrid_dbsf", by_rank=False)


_STRATEGIES: Dict[FusionStrategy, Callable[[List[Hit], List[Hit], RetrievalConfig], List[Hit]]] = {
    FusionStrategy.RRF: fuse_rrf,
    FusionStrategy.WEIGHTED: fuse_weighted,
    FusionStrategy.DBSF: fuse_dbsf,
}


def fuse(dense: List[Hit], sparse: List[Hit], cfg: RetrievalConfig) -> List[Hit]:
    """Dispatch to the strategy named by `cfg.fusion_strategy`."""
    return _STRATEGIES[cfg.fusion_strategy](dense, sparse, cfg)


# --------------------------------------------------------------------------- #
# Shared machinery
# --------------------------------------------------------------------------- #

def _combine(
    dense: List[Hit],
    sparse: List[Hit],
    cfg: RetrievalConfig,
    dense_fn: Callable[[Any], float],
    sparse_fn: Callable[[Any], float],
    *,
    source: str,
    by_rank: bool = True,
) -> List[Hit]:
    """
    Union both arms by chunk_id, score each chunk as
    `w_dense * dense_fn(x) + w_sparse * sparse_fn(x)` where `x` is the
    1-based rank (by_rank=True) or the raw arm score (by_rank=False), and a
    missing arm contributes 0.
    """
    dense_pos = {h["chunk_id"]: i for i, h in enumerate(dense)}
    sparse_pos = {h["chunk_id"]: i for i, h in enumerate(sparse)}

    # Dense order first, then sparse-only chunks — deterministic tie-breaking.
    ordered_ids = list(dense_pos) + [cid for cid in sparse_pos if cid not in dense_pos]

    fused: List[Hit] = []
    for cid in ordered_ids:
        d_idx, s_idx = dense_pos.get(cid), sparse_pos.get(cid)
        base = dense[d_idx] if d_idx is not None else sparse[s_idx]  # type: ignore[index]
        record: Hit = {**base, "metadata": dict(base.get("metadata") or {})}

        score = 0.0
        if d_idx is not None:
            d_score = dense[d_idx]["score"]
            score += cfg.dense_weight * dense_fn(d_idx + 1 if by_rank else d_score)
            record["dense_score"], record["dense_rank"] = d_score, d_idx + 1
        if s_idx is not None:
            s_score = sparse[s_idx]["score"]
            score += cfg.sparse_weight * sparse_fn(s_idx + 1 if by_rank else s_score)
            record["sparse_score"], record["sparse_rank"] = s_score, s_idx + 1

        record.update(score=score, final_score=score, retrieval_source=source)
        fused.append(record)

    fused.sort(key=lambda h: h["score"], reverse=True)
    return fused


def _minmax(scores: Sequence[float]) -> Callable[[float], float]:
    """Min-max scaler fitted on `scores`; a degenerate range maps to 1.0."""
    if not scores:
        return lambda _: 0.0
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span <= 0.0:
        return lambda _: 1.0
    return lambda s: (s - lo) / span


def _zscore(scores: Sequence[float], sigma: float) -> Callable[[float], float]:
    """Scaler mapping mean ± sigma·std onto [0, 1], clipped."""
    if not scores:
        return lambda _: 0.0
    n = len(scores)
    mean = sum(scores) / n
    std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5
    if std <= 0.0:
        return lambda _: 1.0
    lo, span = mean - sigma * std, 2.0 * sigma * std
    return lambda s: min(max((s - lo) / span, 0.0), 1.0)
