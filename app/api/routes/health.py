"""
app.api.routes.health
=====================
`GET /v1/health` — liveness.  `GET /v1/ready` — Milvus reachability + gate state (503 when not ready).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pymilvus import MilvusClient
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import get_base_config, get_gate
from app.api.schemas import HealthResponse, ReadinessResponse
from app.concurrency import ProcessingGate
from app.config.defaults import ChunkerConfig
from app.retrieval.vector_db import milvus_connection_args

router = APIRouter(tags=["health"])

_PROBE_TIMEOUT_SECONDS = 3.0


@router.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/v1/ready", response_model=ReadinessResponse, responses={503: {"model": ReadinessResponse}})
async def ready(
    base_config: ChunkerConfig = Depends(get_base_config),
    gate: ProcessingGate = Depends(get_gate),
) -> ReadinessResponse:
    milvus_ok = await run_in_threadpool(_milvus_reachable, base_config)
    body = ReadinessResponse(
        status="ready" if milvus_ok else "not_ready",
        milvus="reachable" if milvus_ok else "unreachable",
        accepting_documents=not gate.busy,
    )
    if not milvus_ok:
        raise HTTPException(status_code=503, detail=body.model_dump())
    return body


def _milvus_reachable(config: ChunkerConfig) -> bool:
    """Best-effort probe with the same connection settings the retriever uses."""
    args = {**milvus_connection_args(config), "timeout": _PROBE_TIMEOUT_SECONDS}
    try:
        client = MilvusClient(**args)
        try:
            client.list_collections()
        finally:
            client.close()
    except Exception:  # any failure means "not reachable"; the reason is not actionable here
        return False
    return True
