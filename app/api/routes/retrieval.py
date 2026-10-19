"""
app.api.routes.retrieval
========================
`POST /v1/retrieval/query`              — hybrid search over one collection, optionally narrowed by category
`POST /v1/retrieval/query-reformulation` — rewrite a query in the collection's vocabulary
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_logger, get_query_reformulation_service, get_retriever_registry, get_settings
from app.api.errors import http_error
from app.api.schemas import (
    ErrorResponse,
    QueryReformulationRequest,
    QueryReformulationResponse,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
)
from app.config.settings import ServerSettings
from app.logging.setup import request_id_var
from app.retrieval.errors import CollectionNotFoundError
from app.retrieval.filters import CategoryFilter
from app.retrieval.reformulation import QueryReformulationService
from app.retrieval.registry import RetrieverRegistry

router = APIRouter(tags=["retrieval"])


@router.post(
    "/v1/retrieval/query",
    response_model=RetrievalQueryResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def query_retrieval(
    payload: RetrievalQueryRequest,
    registry: RetrieverRegistry = Depends(get_retriever_registry),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> RetrievalQueryResponse:
    token = request_id_var.set(request_id_var.get() or uuid.uuid4().hex)  # id from the middleware
    started = time.monotonic()
    try:
        collection = payload.collection_name  # CollectionName: already normalised
        try:
            retriever = registry.get(collection)
            results = await retriever.retrieve_async(
                payload.query,
                top_k=payload.top_k,
                use_hybrid=payload.use_hybrid,
                use_reranking=payload.use_reranking,
                alpha=payload.alpha,
                fusion_strategy=payload.fusion_strategy,
                reformulated_query=payload.reformulated_query,
                category_filter=CategoryFilter.build(payload.main_categories, payload.sub_categories, payload.categories),
            )
        except CollectionNotFoundError as exc:
            registry.evict(collection)  # don't keep a client + thread pool alive for a name that doesn't exist
            raise http_error(404, exc.code, str(exc)) from exc
        except ValueError as exc:
            raise http_error(400, "INVALID_REQUEST", str(exc)) from exc
        except Exception as exc:
            logger.error(
                "retrieval query failed",
                extra={"channel": "retrieval", "stage": "query", "status": "failed", "collection_name": collection,
                       "error_type": type(exc).__name__, "error_message": str(exc)},
                exc_info=True,
            )
            raise http_error(500, "RETRIEVAL_FAILED", "Retrieval query failed.", retryable=True) from exc

        logger.info(
            "retrieval query completed",
            extra={"channel": "retrieval", "stage": "query", "status": "completed", "collection_name": collection,
                   "duration_ms": int((time.monotonic() - started) * 1000), "chunk_count": len(results)},
        )
        return RetrievalQueryResponse(results=results)
    finally:
        request_id_var.reset(token)


@router.post(
    "/v1/retrieval/query-reformulation",
    response_model=QueryReformulationResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Reformulate a query using the collection's published vocabulary index",
    description=(
        "Rewrites `query` with the terminology in the collection's current `vocab_index.json` on S3. "
        "Fail-safe: any failure returns the original query unchanged so retrieval is never blocked."
    ),
)
async def reformulate_query(
    payload: QueryReformulationRequest,
    service: QueryReformulationService = Depends(get_query_reformulation_service),
    settings: ServerSettings = Depends(get_settings),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> QueryReformulationResponse:
    token = request_id_var.set(request_id_var.get() or uuid.uuid4().hex)  # id from the middleware
    started = time.monotonic()
    try:
        collection = payload.collection_name
        bucket = payload.s3_bucket or settings.s3_bucket
        if not bucket:
            raise http_error(400, "S3_BUCKET_REQUIRED", "s3_bucket is required: no S3_BUCKET default is configured.")

        reformulated = await run_in_threadpool(
            service.reformulate, payload.query, collection, s3_bucket=bucket, AWS_DEFAULT_REGION=payload.AWS_DEFAULT_REGION or settings.AWS_DEFAULT_REGION,
        )
        logger.info(
            "query reformulation completed",
            extra={"channel": "retrieval", "stage": "query_reformulation", "status": "completed",
                   "collection_name": collection, "duration_ms": int((time.monotonic() - started) * 1000)},
        )
        return QueryReformulationResponse(query=payload.query, reformulated_query=reformulated)
    finally:
        request_id_var.reset(token)
