"""
app.retrieval.vector_db
=======================
EnhancedCohereMilvusRetriever — language-aware hybrid dense + BM25 retriever on one Milvus collection.

    dense arm   : FLOAT_VECTOR, embedded with the model the collection was built with
    sparse arm  : SPARSE_FLOAT_VECTOR, BM25 computed by Milvus with the collection's analyzer
                  (English: standard; Japanese: Lindera IPADIC — app.retrieval.profiles)
    fusion      : client-side  RRF / weighted / DBSF        (app.retrieval.fusion)
                  server-side  hybrid_search + RRFRanker / WeightedRanker
    rerank      : Cohere reranker of the collection's language, over the top `rerank_candidates`
    filter      : optional CategoryFilter (app.retrieval.filters), applied inside every Milvus search

Language binding
    A collection has one language, fixed when the first document creates it and recorded
    in its manifest (app.retrieval.milvus_schema). The retriever binds to that manifest on
    first contact and keeps it: queries are analyzed by the collection's analyzer (Milvus
    does that for the field) and embedded with the collection's embedding model, so an
    ASCII query such as "BCL200S" is tokenized exactly like the Japanese rows it must match.

Read path
    retrieve_async ─► bind ─► [dense ‖ sparse] (filtered) ─► fuse ─► rerank ─► slice(final_top_k)

Write path
    add_documents    ─► check language ─► prepare rows ─► per batch: embed ─► [create collection] ─► upsert
                     ─► flush + load
    replace_document ─► add_documents ─► delete the document's rows that are no longer among its chunks

Failure policy: the dense arm is authoritative — its errors propagate. The sparse arm
and the reranker degrade gracefully (logged, query still answered). A missing collection,
a language mismatch or a write to a legacy collection raise app.retrieval.errors types.

Every blocking Milvus / Cohere call on the read path runs on a per-instance
ThreadPoolExecutor (each call bound to the caller's context), so `retrieve_async` never blocks the
event loop. Instances are safe to share across concurrent `retrieve_async` calls; `close()` is idempotent.

Tracing: `retrieve_async` is one `hybrid_retrieval` trace (strategy, depths, fusion, models; input = the
query's fingerprint, or its text when `LANGFUSE_CAPTURE_QUERY_TEXT`), with `dense_search` / `bm25_search` /
`fusion` / `rerank` spans; the output lists the returned chunk ids and scores, never their text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker

from app.config.defaults import ChunkerConfig
from app.logging.setup import in_context
from app.observability.modules import ModuleName, SpanName
from app.observability.sanitize import query_fingerprint
from app.observability.telemetry import get_telemetry
from app.retrieval.config import FusionMode, FusionStrategy, RetrievalConfig
from app.retrieval.embeddings import CohereEmbeddings
from app.retrieval.errors import CollectionLanguageMismatchError, CollectionNotFoundError, LegacyCollectionError
from app.retrieval.filters import CategoryFilter
from app.retrieval.fusion import Hit, fuse
from app.retrieval.language import ENGLISH, JAPANESE
from app.retrieval.milvus_schema import (
    F_DENSE,
    F_PAGE,
    F_PK,
    F_SPARSE,
    F_TEXT,
    PAGE_NULL,
    CollectionLayout,
    create_collection,
    dense_index_params,
    describe_layout,
)
from app.retrieval.profiles import LanguageProfile, language_profile
from app.retrieval.reranker import CohereReranker
from app.retrieval.vector_db_writer import DEFAULT_BATCH_SIZE, delete_stale, flush_and_load, prepare_rows
from app.utils.naming import sanitize_collection_name

__all__ = ["EnhancedCohereMilvusRetriever", "Hit", "milvus_connection_args"]

SOURCE_DENSE = "vector"
SOURCE_SPARSE = "bm25"


def milvus_connection_args(config: ChunkerConfig) -> Dict[str, Any]:
    """`MilvusClient(**kwargs)` for `config`: scheme follows `milvus_secure`; token wins over user/password."""
    scheme = "https" if config.milvus_secure else "http"
    args: Dict[str, Any] = {"uri": f"{scheme}://{config.milvus_host}:{config.milvus_port}", "db_name": config.milvus_db_name}
    if config.milvus_token:
        args["token"] = config.milvus_token
    elif config.milvus_user and config.milvus_password:
        args["user"] = config.milvus_user
        args["password"] = config.milvus_password
    if config.milvus_timeout is not None:
        args["timeout"] = config.milvus_timeout
    return args


@dataclass(frozen=True)
class _Binding:
    """What the retriever learned from the live collection; fixed for the retriever's lifetime."""

    language: str
    layout: CollectionLayout
    embeddings: Any
    reranker: CohereReranker


class EnhancedCohereMilvusRetriever:
    """Hybrid dense + Milvus-native-BM25 retriever with Cohere reranking, bound to one collection's language."""

    def __init__(
        self,
        collection_name: str,
        config: Optional[ChunkerConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        logger: Optional[logging.Logger] = None,
        embeddings: Optional[Any] = None,
        language: Optional[str] = None,
    ) -> None:
        """
        Args:
            collection_name:  Milvus collection (sanitised on entry).
            config:           Infrastructure settings (Milvus, Cohere, models).
            retrieval_config: Pipeline defaults; every field can be overridden per call.
            logger:           Optional logger; the app logger is used when omitted.
            embeddings:       Optional LangChain-compatible embeddings to inject (tests /
                              alternative providers). Defaults to Cohere, per collection language.
            language:         Language of the documents this instance will write ("ja" / "en").
                              Required to create the collection; checked against an existing one.
                              Queries never need it — they follow the collection.
        """
        self._cfg = config or ChunkerConfig()
        self._retrieval_config = retrieval_config or RetrievalConfig()
        self.collection_name = sanitize_collection_name(collection_name)
        self._log = _resolve_logger(logger)
        self._write_language = language_profile(language, self._cfg).language if language else None

        self._closed = False
        self._lock = threading.Lock()
        self._api_key = self._cfg.cohere_api_key or os.getenv("COHERE_API_KEY", "")
        self._injected_embeddings = embeddings
        self._embedders: Dict[str, Any] = {}
        self._rerankers: Dict[str, CohereReranker] = {}
        self._dense_index = dense_index_params(self._cfg)
        self._client = MilvusClient(**milvus_connection_args(self._cfg))
        self._binding: Optional[_Binding] = None

        self._executor = ThreadPoolExecutor(
            max_workers=self._retrieval_config.executor_workers,
            thread_name_prefix=f"milvus-{self.collection_name}",
        )
        binding = self._try_bind()
        self._log.info(
            "Retriever ready: collection=%s language=%s fusion=%s/%s depths(dense=%d sparse=%d rerank=%d final=%d)",
            self.collection_name, binding.language if binding else "(collection not created yet)",
            self._retrieval_config.fusion_mode.value, self._retrieval_config.fusion_strategy.value,
            self._retrieval_config.dense_top_k, self._retrieval_config.sparse_top_k,
            self._retrieval_config.rerank_candidates, self._retrieval_config.final_top_k,
        )

    @property
    def language(self) -> Optional[str]:
        """The collection's language, or the language it will be created with; None when neither is known."""
        binding = self._binding
        return binding.language if binding else self._write_language

    @property
    def embedding_model(self) -> Optional[str]:
        """The model the bound collection was embedded with; None before binding or for injected embeddings."""
        binding = self._binding
        return getattr(binding.embeddings, "model", None) if binding else None

    # ------------------------------------------------------------------ #
    # Binding to the live collection
    # ------------------------------------------------------------------ #

    def _try_bind(self) -> Optional[_Binding]:
        """Bind to the collection once it exists; None while it does not."""
        with self._lock:
            if self._binding is None:
                layout = describe_layout(
                    self._client, self.collection_name,
                    fallback_metric=self._dense_index["metric_type"], logger=self._log,
                )
                if layout is not None:
                    self._binding = self._make_binding(layout)
            return self._binding

    def _require_binding(self) -> _Binding:
        binding = self._try_bind()
        if binding is None:
            raise CollectionNotFoundError(self.collection_name)
        return binding

    def _make_binding(self, layout: CollectionLayout) -> _Binding:
        if layout.manifest is None:
            # Created before manifests existed: English analyzer, embedded with the English model.
            profile = language_profile(ENGLISH, self._cfg)
            embedding_model = profile.embedding_model
            self._log.warning(
                "Collection '%s' has no manifest (legacy layout): searching as English with %s. "
                "Re-ingest it into a new collection to use language profiles.",
                self.collection_name, embedding_model,
            )
        else:
            profile = language_profile(layout.manifest.language, self._cfg)
            embedding_model = layout.manifest.embedding_model
            if embedding_model != profile.embedding_model:
                self._log.warning(
                    "Collection '%s' was embedded with %s; config now says %s. Searching with %s — "
                    "re-ingest into a new collection to switch models.",
                    self.collection_name, embedding_model, profile.embedding_model, embedding_model,
                )
        return _Binding(
            language=profile.language,
            layout=layout,
            embeddings=self._embedder(embedding_model),
            reranker=self._reranker(profile.reranking_model),
        )

    def _embedder(self, model: str) -> Any:
        if self._injected_embeddings is not None:
            return self._injected_embeddings
        if model not in self._embedders:
            self._embedders[model] = _LazyCohereEmbeddings(model, self._cfg.embedding_batch_size, self._api_key)
        return self._embedders[model]

    def _reranker(self, model: str) -> CohereReranker:
        if model not in self._rerankers:
            self._rerankers[model] = CohereReranker(api_key=self._api_key, model=model, logger=self._log)
        return self._rerankers[model]

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #

    def existing_language(self) -> Optional[str]:
        """The language the collection was created with (by its first document); None when it does not exist yet."""
        binding = self._try_bind()
        if binding is None:
            return None
        if binding.layout.is_legacy:
            raise LegacyCollectionError(self.collection_name)
        return binding.language

    def check_writable(self) -> Optional[_Binding]:
        """
        Raise unless this instance may write its language into the collection: the binding
        of an existing collection, or None when the first write will create it.
        """
        binding = self._try_bind()
        if binding is None:
            if self._write_language is None:
                raise ValueError(f"Collection '{self.collection_name}' does not exist; a language is required to create it.")
            return None
        if binding.layout.is_legacy:
            raise LegacyCollectionError(self.collection_name)
        if self._write_language and self._write_language != binding.language:
            raise CollectionLanguageMismatchError(self.collection_name, binding.language, self._write_language)
        return binding

    def add_documents(
        self,
        docs: Sequence[Document],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """
        Upsert documents keyed by `metadata["chunk_id"]` (or a stable content hash), `batch_size` at a
        time (embed, then upsert — progress after each batch); creates the collection on the first
        write. Returns the chunk ids written.

        Raises ValueError on duplicate ids within the call, and the app.retrieval.errors
        types when the collection cannot take this language.
        """
        binding = self.check_writable()
        rows = prepare_rows(docs, self._log)
        if not rows:
            self._log.warning("add_documents(): nothing to write (%d empty documents).", len(docs))
            return []

        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            if binding is None:  # the first batch creates the collection with its vector size
                profile = language_profile(self._write_language, self._cfg)
                vectors = self._embedder(profile.embedding_model).embed_documents([row.text for row in batch])
                self._create_collection(profile, dense_dim=len(vectors[0]))
                binding = self._require_binding()
            else:
                vectors = binding.embeddings.embed_documents([row.text for row in batch])
            self._client.upsert(self.collection_name, [row.with_vector(vec) for row, vec in zip(batch, vectors, strict=True)])
            if on_progress is not None:
                on_progress(min(start + batch_size, len(rows)), len(rows))
        flush_and_load(self._client, self.collection_name, self._log)
        self._log.info(
            "Upserted %d documents into '%s' (language=%s, %d skipped as empty).",
            len(rows), self.collection_name, self.language, len(docs) - len(rows),
        )
        return [row.chunk_id for row in rows]

    def replace_document(
        self,
        docs: Sequence[Document],
        *,
        document_id: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """
        Make `document_id`'s rows exactly `docs`: upsert them, then delete the document's rows from an
        earlier version that are not among them. New rows land before old ones go, so the document is
        never missing from search. A version with no indexable chunk removes all of the document's rows.
        Every error propagates (a stale row is a wrong answer).
        """
        written = self.add_documents(docs, batch_size=batch_size, on_progress=on_progress)
        if not written and self._try_bind() is None:
            return written  # nothing written and no collection: there are no rows to remove
        removed = delete_stale(self._client, self.collection_name, document_id, set(written))
        if removed:
            flush_and_load(self._client, self.collection_name, self._log)
            self._log.info("Removed %d stale row(s) of document '%s' from '%s'.", removed, document_id, self.collection_name)
        return written

    def _create_collection(self, profile: LanguageProfile, *, dense_dim: int) -> None:
        try:
            manifest = create_collection(
                self._client, self.collection_name, profile=profile, dense_dim=dense_dim, dense_index=self._dense_index,
            )
        except Exception as exc:
            if profile.language == JAPANESE:
                raise RuntimeError(
                    f"Could not create Japanese collection '{self.collection_name}': the Lindera analyzer needs "
                    f"Milvus 2.6+ and one-time network access to fetch its dictionary ({exc})"
                ) from exc
            raise
        self._log.info(
            "Created collection '%s': language=%s embedding=%s dim=%d",
            self.collection_name, manifest.language, manifest.embedding_model, manifest.dense_dim,
        )
        self._try_bind()

    # ------------------------------------------------------------------ #
    # Read path — public
    # ------------------------------------------------------------------ #

    async def retrieve_async(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
        alpha: Optional[float] = None,
        fusion_strategy: Optional[str] = None,
        reformulated_query: Optional[str] = None,
        category_filter: Optional[CategoryFilter] = None,
    ) -> List[Hit]:
        """
        Hybrid search → fusion → rerank → top-k. Each keyword overrides the
        corresponding `RetrievalConfig` field for this call only; `top_k`
        widens the search depths as needed so it is always honoured.

        When `reformulated_query` is set it is what gets searched and reranked;
        `query` is kept for logging (see `/v1/retrieval/reformulate`).
        `category_filter` restricts every arm to chunks with those categories.
        Raises CollectionNotFoundError when the collection does not exist, and
        ValueError when filtering a legacy collection (it has no category fields).
        """
        cfg = self._retrieval_config.with_overrides(
            final_top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
            alpha=alpha,
            fusion_strategy=fusion_strategy,
        )
        if not (query or "").strip():
            raise ValueError("query must not be empty")
        search_query = (reformulated_query or "").strip() or query.strip()
        reformulated = search_query != query.strip()
        telemetry = get_telemetry()
        strategy = "dense" if not cfg.use_hybrid else f"hybrid_{cfg.effective_fusion_mode.value}"

        started = time.perf_counter()
        with telemetry.trace(
            ModuleName.HYBRID_RETRIEVAL, input=telemetry.query_input(search_query, reformulated=reformulated),
            collection_name=self.collection_name, strategy=strategy, fusion_strategy=cfg.fusion_strategy, fusion_mode=cfg.effective_fusion_mode,
            alpha=cfg.alpha, rrf_k=cfg.rrf_k, top_k=cfg.final_top_k, dense_top_k=cfg.dense_top_k, sparse_top_k=cfg.sparse_top_k,
            rerank_candidates=cfg.rerank_candidates, use_hybrid=cfg.use_hybrid, use_reranking=cfg.use_reranking,
            filter_present=bool(category_filter), reformulated=reformulated,
        ) as trace:
            binding = self._binding or await self._run(self._require_binding)
            trace.set(language=binding.language, embedding_model=self.embedding_model, reranking_model=binding.reranker.model)
            expr = category_filter.expression() if category_filter else ""
            if expr and binding.layout.is_legacy:
                raise ValueError(f"Collection '{self.collection_name}' predates classification; category filters are not supported.")
            fused = await self._search(search_query, cfg, binding, expr)
            candidates = fused[: cfg.rerank_candidates]

            reranked = False
            if cfg.use_reranking and binding.reranker.enabled and len(candidates) > 1:
                with telemetry.span(SpanName.RERANK, model=binding.reranker.model, candidates=len(candidates)) as span:
                    reranked_hits = await binding.reranker.rerank_async(
                        query=search_query, docs=candidates, top_k=cfg.rerank_candidates, run_fn=self._run,
                    )
                    reranked = reranked_hits is not candidates  # the reranker returns its input on failure
                    span.set_output(applied=reranked)
                candidates = reranked_hits

            results = candidates[: cfg.final_top_k]
            trace.set_output(
                chunk_ids=[hit["chunk_id"] for hit in results], scores=[hit.get("final_score") for hit in results],
                result_count=len(results), candidates=len(fused), applied=reranked,
            )
        self._log.info(
            "Retrieved %d/%d hits from '%s' in %dms (language=%s hybrid=%s fusion=%s/%s reranked=%s filter=%s)",
            len(results), len(fused), self.collection_name,
            int((time.perf_counter() - started) * 1000), binding.language,
            cfg.use_hybrid, cfg.effective_fusion_mode.value, cfg.fusion_strategy.value, reranked, expr or "none",
        )
        if reformulated:  # query text stays out of INFO logs (customer data); fingerprints are enough to correlate
            self._log.info(
                "Searched with reformulated query: %s -> %s",
                query_fingerprint(query)["query_sha256_12"], query_fingerprint(search_query)["query_sha256_12"],
            )
            self._log.debug("Reformulated query text: %r -> %r", query, search_query)
        self._log.debug("Final chunk ids: %s", [hit["chunk_id"] for hit in results])
        return results

    def retrieve(self, query: str, **kwargs: Any) -> List[Hit]:
        """Synchronous wrapper for scripts and tests; see `retrieve_async` for arguments."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.retrieve_async(query, **kwargs))
        raise RuntimeError(
            "retrieve() cannot be called from a running event loop; await retrieve_async() instead."
        )

    # ------------------------------------------------------------------ #
    # Read path — internals
    # ------------------------------------------------------------------ #

    async def _search(self, query: str, cfg: RetrievalConfig, binding: _Binding, expr: str) -> List[Hit]:
        """Run the arms (each restricted by `expr`, "" for none) and return one fused, best-first candidate list."""
        if not cfg.use_hybrid:
            return await self._run(self._search_dense, query, cfg, binding, expr)

        if cfg.effective_fusion_mode is FusionMode.SERVER:
            return await self._run(self._search_hybrid_server, query, cfg, binding, expr)

        dense, sparse = await asyncio.gather(
            self._run(self._search_dense, query, cfg, binding, expr),
            self._run(self._search_sparse, query, cfg, binding, expr),
        )
        self._log.debug(
            "Arm hits — dense: %s | sparse: %s",
            [h["chunk_id"] for h in dense], [h["chunk_id"] for h in sparse],
        )
        with get_telemetry().span(SpanName.FUSION, fusion_strategy=cfg.fusion_strategy, fusion_mode="client", alpha=cfg.alpha, rrf_k=cfg.rrf_k) as span:
            fused = fuse(dense, sparse, cfg)
            span.set_output(candidates=len(fused))
        return fused

    async def _run(self, fn: Any, *args: Any) -> Any:
        if self._closed:
            raise RuntimeError("Retriever is closed.")
        # run_in_executor does not copy contextvars: bind the call so the worker keeps the request's log and trace context.
        return await asyncio.get_running_loop().run_in_executor(self._executor, in_context(fn), *args)

    def _dense_search_params(self, binding: _Binding) -> Dict[str, Any]:
        params = dict(self._cfg.milvus_search_params or {})
        params["metric_type"] = binding.layout.dense_metric
        params.setdefault("params", {})
        return params

    @staticmethod
    def _sparse_search_params(cfg: RetrievalConfig) -> Dict[str, Any]:
        # No analyzer choice: Milvus tokenizes the query with the `text` field's own analyzer.
        return {"metric_type": "BM25", "params": {"drop_ratio_search": cfg.drop_ratio_search}}

    def _search_dense(self, query: str, cfg: RetrievalConfig, binding: _Binding, expr: str) -> List[Hit]:
        """Primary arm — errors propagate."""
        with get_telemetry().span(SpanName.DENSE_SEARCH, top_k=cfg.dense_top_k, filter_present=bool(expr)) as span:
            results = self._client.search(
                collection_name=self.collection_name,
                data=[binding.embeddings.embed_query(query)],
                anns_field=F_DENSE,
                search_params=self._dense_search_params(binding),
                limit=cfg.dense_top_k,
                output_fields=binding.layout.output_fields,
                filter=expr,
            )
            hits = [self._to_hit(hit, SOURCE_DENSE, binding) for hit in (results[0] if results else [])]
            span.set_output(hits=len(hits))
        return hits

    def _search_sparse(self, query: str, cfg: RetrievalConfig, binding: _Binding, expr: str) -> List[Hit]:
        """Secondary arm — degrades to [] on failure."""
        with get_telemetry().span(SpanName.BM25_SEARCH, top_k=cfg.sparse_top_k, filter_present=bool(expr), language=binding.language) as span:
            try:
                results = self._client.search(
                    collection_name=self.collection_name,
                    data=[query],
                    anns_field=F_SPARSE,
                    search_params=self._sparse_search_params(cfg),
                    limit=cfg.sparse_top_k,
                    output_fields=binding.layout.output_fields,
                    filter=expr,
                )
            except Exception as exc:
                self._log.error("BM25 search failed on '%s' (%s) — continuing dense-only.", self.collection_name, exc)
                span.record_error(exc)
                span.set_output(hits=0, degraded=True)
                return []
            hits = [self._to_hit(hit, SOURCE_SPARSE, binding) for hit in (results[0] if results else [])]
            span.set_output(hits=len(hits), degraded=False)
        return hits

    def _search_hybrid_server(self, query: str, cfg: RetrievalConfig, binding: _Binding, expr: str) -> List[Hit]:
        """Single round-trip: Milvus runs both arms and fuses with a native ranker."""
        with get_telemetry().span(
            SpanName.FUSION, fusion_strategy=cfg.fusion_strategy, fusion_mode="server", alpha=cfg.alpha, rrf_k=cfg.rrf_k,
            dense_top_k=cfg.dense_top_k, sparse_top_k=cfg.sparse_top_k, filter_present=bool(expr),
        ) as span:
            hits = self._hybrid_server(query, cfg, binding, expr)
            span.set_output(candidates=len(hits))
        return hits

    def _hybrid_server(self, query: str, cfg: RetrievalConfig, binding: _Binding, expr: str) -> List[Hit]:
        requests = [
            AnnSearchRequest(
                data=[binding.embeddings.embed_query(query)],
                anns_field=F_DENSE,
                param=self._dense_search_params(binding),
                limit=cfg.dense_top_k,
                expr=expr or None,
            ),
            AnnSearchRequest(
                data=[query],
                anns_field=F_SPARSE,
                param=self._sparse_search_params(cfg),
                limit=cfg.sparse_top_k,
                expr=expr or None,
            ),
        ]
        ranker = (
            RRFRanker(cfg.rrf_k)
            if cfg.fusion_strategy is FusionStrategy.RRF
            else WeightedRanker(cfg.dense_weight, cfg.sparse_weight)
        )
        results = self._client.hybrid_search(
            collection_name=self.collection_name,
            reqs=requests,
            ranker=ranker,
            limit=cfg.rerank_candidates,
            output_fields=binding.layout.output_fields,
        )
        source = f"hybrid_server_{cfg.fusion_strategy.value}"
        return [self._to_hit(hit, source, binding) for hit in (results[0] if results else [])]

    def _to_hit(self, hit: Dict[str, Any], source: str, binding: _Binding) -> Hit:
        """Normalise a MilvusClient result row into the pipeline's hit shape."""
        entity = dict(hit.get("entity") or {})
        chunk_id = entity.pop(F_PK, None) or hit.get(F_PK) or hit.get("id")
        text = entity.pop(F_TEXT, None) or ""
        page = entity.pop(F_PAGE, None)
        if page == PAGE_NULL:
            page = None

        score = float(hit.get("distance", 0.0))
        if source == SOURCE_DENSE and binding.layout.dense_metric == "L2":
            score = -score  # L2 is a distance; make "higher is better" uniform across arms

        metadata = _decode_legacy_metadata(entity) if binding.layout.is_legacy else entity
        metadata[F_PK] = chunk_id
        metadata[F_PAGE] = page
        return {
            "chunk_id": chunk_id,
            "text": text,
            "page": page,
            "metadata": metadata,
            "score": score,
            "final_score": score,
            "retrieval_source": source,
        }

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)
        try:
            self._client.close()
        except Exception as exc:
            self._log.debug("Milvus client close: %s", exc)
        self._log.info("Retriever closed: collection=%s", self.collection_name)

    def __enter__(self) -> "EnhancedCohereMilvusRetriever":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<EnhancedCohereMilvusRetriever collection={self.collection_name!r} language={self.language} "
            f"fusion={self._retrieval_config.fusion_mode.value}/{self._retrieval_config.fusion_strategy.value}>"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _LazyCohereEmbeddings:
    """CohereEmbeddings built on first use, so binding to a collection never needs the API key."""

    def __init__(self, model: str, batch_size: int, api_key: str) -> None:
        self.model = model
        self._args = (model, batch_size, api_key)
        self._inner: Optional[CohereEmbeddings] = None
        self._lock = threading.Lock()

    def _get(self) -> CohereEmbeddings:
        with self._lock:
            if self._inner is None:
                model, batch_size, api_key = self._args
                self._inner = CohereEmbeddings(model=model, batch_size=batch_size, api_key=api_key)
            return self._inner

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get().embed_query(text)


def _decode_legacy_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy (langchain-milvus) rows stored lists/dicts as JSON strings — restore them."""
    out: Dict[str, Any] = {}
    for key, val in metadata.items():
        if isinstance(val, str) and val[:1] in ("[", "{"):
            try:
                decoded = json.loads(val)
                val = decoded if isinstance(decoded, (list, dict)) else val
            except ValueError:
                pass
        out[key] = val
    return out


def _resolve_logger(logger: Optional[logging.Logger]) -> logging.LoggerAdapter:
    if isinstance(logger, logging.LoggerAdapter):
        return logger
    if logger is not None:
        return logging.LoggerAdapter(logger, {"log_module": "RETRIEVER"})
    from app.logging.setup import get_logger
    return get_logger(module="RETRIEVER")
