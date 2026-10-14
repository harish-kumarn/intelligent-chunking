"""
app.main
========
FastAPI application factory and process lifecycle.

    startup  : logging → settings → ChunkerConfig → orchestrator (+PDF channel) → gate →
               ingestion job manager (shares the gate; recorded jobs loaded, an interrupted one continued) →
               retriever registry → reformulation service, all stored on `app.state`;
               abandoned staged uploads removed; Lindera warm-up in the background (Japanese BM25 dictionary)
    shutdown : stop job retry waits, close every retriever (thread pools, Milvus/Cohere clients)

Run with `uvicorn app.main:app` and exactly one worker (see Dockerfile).
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pymilvus import MilvusClient

from app.api.routes import router
from app.api.uploads import clear_staged_uploads
from app.concurrency import ProcessingGate
from app.config.defaults import ChunkerConfig
from app.config.settings import ServerSettings
from app.logging.setup import configure_logging, get_logger
from app.orchestration.channels.pdf.channel import PDFChannel
from app.orchestration.jobs import IngestionJobManager, JobStore
from app.orchestration.orchestrator import ChannelOrchestrator
from app.retrieval.milvus_schema import warm_up_japanese_analyzer
from app.retrieval.reformulation import QueryReformulationService
from app.retrieval.registry import RetrieverRegistry
from app.retrieval.vector_db import milvus_connection_args

__all__ = ["app", "create_app"]


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    configure_logging(settings.service_name, settings.environment, level=settings.log_level, log_dir=settings.log_dir)
    log = get_logger(module="API")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        base_config = ChunkerConfig()
        orchestrator = ChannelOrchestrator()
        orchestrator.register(PDFChannel(settings, base_config))

        application.state.settings = settings
        application.state.base_config = base_config
        application.state.orchestrator = orchestrator
        application.state.gate = ProcessingGate()
        clear_staged_uploads(settings.uploads_dir)
        application.state.ingestion_jobs = jobs = IngestionJobManager(
            application.state.gate,
            store=JobStore(settings.jobs_dir, log),
            retention_hours=settings.ingest_job_retention_hours,
            max_attempts=settings.ingest_job_max_attempts,
            retry_delay_seconds=settings.ingest_job_retry_delay_seconds,
            s3_publish=settings.s3_upload_enabled,
            classify_default=base_config.classification_enabled,
            logger=log,
        )
        jobs.recover(orchestrator, auto_resume=settings.ingest_job_auto_resume)
        application.state.retriever_registry = RetrieverRegistry(base_config, logger=log)
        application.state.query_reformulation_service = QueryReformulationService(base_config)
        application.state.logger = log
        log.info(
            "Service started: name=%s version=%s env=%s artifact_mode=%s s3_upload=%s",
            settings.service_name, settings.service_version, settings.environment, settings.artifact_mode, settings.s3_upload_enabled,
        )
        if base_config.warm_up_japanese_analyzer:
            threading.Thread(target=_warm_up_japanese_analyzer, args=(base_config, log), name="lindera-warm-up", daemon=True).start()
        try:
            yield
        finally:
            jobs.stop()
            application.state.retriever_registry.close_all()
            log.info("Service stopped: name=%s", settings.service_name)

    application = FastAPI(title=settings.service_name, version=settings.service_version, lifespan=lifespan)
    application.include_router(router)
    return application


def _warm_up_japanese_analyzer(config: ChunkerConfig, log: logging.LoggerAdapter) -> None:
    """Best effort: a failure only means the first Japanese ingestion pays (or reports) it instead."""
    try:
        client = MilvusClient(**milvus_connection_args(config))
        try:
            warm_up_japanese_analyzer(client)
        finally:
            client.close()
    except Exception as exc:
        log.error(
            "Japanese analyzer (Lindera) unavailable: %s — Japanese documents cannot be indexed until Milvus 2.6+ "
            "can load the IPADIC dictionary (needs one-time network access from the Milvus container).", exc,
        )
        return
    log.info("Japanese analyzer (Lindera IPADIC) ready")


app = create_app()
