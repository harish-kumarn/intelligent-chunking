"""The single API surface: health, documents (upload or from S3), ingestion jobs, retrieval, collections, datasets."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.collections import router as collections_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.ingestion_jobs import router as ingestion_jobs_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.s3_documents import router as s3_documents_router

router = APIRouter()
router.include_router(health_router)
router.include_router(documents_router)
router.include_router(s3_documents_router)
router.include_router(ingestion_jobs_router)
router.include_router(retrieval_router)
router.include_router(collections_router)
router.include_router(datasets_router)

__all__ = ["router"]
