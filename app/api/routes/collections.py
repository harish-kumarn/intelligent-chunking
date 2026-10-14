"""
app.api.routes.collections
==========================
`GET    /v1/collections`                    — list the collections in the service's Milvus database
`DELETE /v1/collections/{collection_name}`  — drop one collection, forget its cached retriever and
                                              (best-effort) its S3 registry `_collections/{name}.json`

Collection administration for clients that hold no Milvus connection of their own
(the maryuma agent proxies its admin routes here). Both endpoints work against the
database this service ingests into (`MILVUS_DB_NAME`), never another one.
"""

from __future__ import annotations

import logging
import uuid
from typing import Callable, List, Optional, TypeVar

from fastapi import APIRouter, Depends
from pymilvus import MilvusClient
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_base_config, get_gate, get_logger, get_retriever_registry, get_settings
from app.api.errors import http_error
from app.api.schemas import CollectionDeleteResponse, CollectionListResponse, CollectionName, ErrorResponse
from app.concurrency import ProcessingGate
from app.config.defaults import ChunkerConfig
from app.config.settings import ServerSettings
from app.logging.setup import request_id_var
from app.retrieval.registry import RetrieverRegistry
from app.retrieval.vector_db import milvus_connection_args
from app.storage.s3_artifact_store import S3ArtifactStore

router = APIRouter(tags=["collections"])

_RETRY_AFTER_SECONDS = 5

T = TypeVar("T")


@router.get(
    "/v1/collections",
    response_model=CollectionListResponse,
    responses={500: {"model": ErrorResponse}},
)
async def list_collections(
    base_config: ChunkerConfig = Depends(get_base_config),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> CollectionListResponse:
    token = request_id_var.set(uuid.uuid4().hex)
    try:
        try:
            names = await run_in_threadpool(_list_collections, base_config)
        except Exception as exc:
            logger.error(
                "collection listing failed",
                extra={"channel": "collections", "stage": "list", "status": "failed",
                       "error_type": type(exc).__name__, "error_message": str(exc)},
                exc_info=True,
            )
            raise http_error(500, "MILVUS_UNAVAILABLE", "Could not list collections.", retryable=True) from exc
        return CollectionListResponse(db_name=base_config.milvus_db_name, count=len(names), collections=sorted(names))
    finally:
        request_id_var.reset(token)


@router.delete(
    "/v1/collections/{collection_name}",
    response_model=CollectionDeleteResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def delete_collection(
    collection_name: CollectionName,
    base_config: ChunkerConfig = Depends(get_base_config),
    gate: ProcessingGate = Depends(get_gate),
    registry: RetrieverRegistry = Depends(get_retriever_registry),
    settings: ServerSettings = Depends(get_settings),
    logger: logging.LoggerAdapter = Depends(get_logger),
) -> CollectionDeleteResponse:
    token = request_id_var.set(uuid.uuid4().hex)
    try:
        collection = collection_name
        # Hold the ingestion gate for the drop: a document must not start landing in a
        # collection that is being removed, and a running ingestion must not lose its target.
        if not gate.try_acquire():
            raise http_error(429, "SERVER_BUSY", "Another document is processing; retry shortly.", retryable=True, retry_after=_RETRY_AFTER_SECONDS)
        try:
            dropped = await run_in_threadpool(_drop_collection, base_config, collection)
        except Exception as exc:
            logger.error(
                "collection drop failed",
                extra={"channel": "collections", "stage": "drop", "status": "failed", "collection_name": collection,
                       "error_type": type(exc).__name__, "error_message": str(exc)},
                exc_info=True,
            )
            raise http_error(500, "MILVUS_UNAVAILABLE", f"Could not drop collection '{collection}'.", retryable=True) from exc
        finally:
            gate.release()

        if not dropped:
            raise http_error(404, "COLLECTION_NOT_FOUND", f"Collection '{collection}' does not exist.")

        registry.evict(collection)  # its retriever held a client bound to the dropped collection
        await run_in_threadpool(_forget_registry, settings, collection, logger)
        logger.info(
            "collection dropped",
            extra={"channel": "collections", "stage": "drop", "status": "completed", "collection_name": collection},
        )
        return CollectionDeleteResponse(db_name=base_config.milvus_db_name, collection_name=collection, deleted=True)
    finally:
        request_id_var.reset(token)


def _forget_registry(settings: ServerSettings, collection: str, logger: logging.LoggerAdapter) -> None:
    """Best-effort: the registry only feeds query reformulation, so a failure never fails the drop."""
    bucket: Optional[str] = settings.s3_bucket
    if not bucket:
        return
    try:
        S3ArtifactStore(bucket=bucket, region=settings.AWS_DEFAULT_REGION).delete_collection_registry(collection)
    except Exception as exc:
        logger.warning("Collection registry not removed: collection=%s: %s", collection, exc)


def _with_client(config: ChunkerConfig, operation: Callable[[MilvusClient], T]) -> T:
    """Short-lived client with the same connection settings the retriever uses."""
    client = MilvusClient(**milvus_connection_args(config))
    try:
        return operation(client)
    finally:
        client.close()


def _list_collections(config: ChunkerConfig) -> List[str]:
    return _with_client(config, lambda client: list(client.list_collections()))


def _drop_collection(config: ChunkerConfig, name: str) -> bool:
    """True when the collection existed and was dropped; False when there was nothing to drop."""

    def drop(client: MilvusClient) -> bool:
        if not client.has_collection(name):
            return False
        client.drop_collection(name)
        return True

    return _with_client(config, drop)
