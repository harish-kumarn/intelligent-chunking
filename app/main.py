"""
app.main
========
FastAPI application factory and process lifecycle.

    startup  : logging → settings → ChunkerConfig → telemetry (Langfuse when LANGFUSE_ENABLED; stamped with the
               pipeline version and a fingerprint of the behaviour-affecting configuration + prompt files) →
               orchestrator (+PDF channel) → gate →
               ingestion job manager (shares the gate; recorded jobs loaded, an interrupted one continued) →
               retriever registry → reformulation service, all stored on `app.state`;
               abandoned staged uploads removed; Lindera warm-up in the background (Japanese BM25 dictionary)
    requests : RequestContextMiddleware gives each request its id and execution context (X-Execution-* headers)
               and returns X-Request-Id / X-Trace-Id
    shutdown : stop job retry waits, close every retriever (thread pools, Milvus/Cohere clients), flush telemetry

Run with `uvicorn app.main:app` and exactly one worker (see Dockerfile).
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from pymilvus import MilvusClient

from app.api.middleware import RequestContextMiddleware
from app.api.routes import router
from app.api.uploads import clear_staged_uploads
from app.concurrency import ProcessingGate
from app.config.defaults import ChunkerConfig
from app.config.settings import ServerSettings
from app.app_logging.setup import configure_logging, get_logger
from app.observability.telemetry import Telemetry, configure_telemetry, set_telemetry
from app.observability.versioning import config_fingerprint
from app.orchestration.channels.pdf.channel import PDFChannel
from app.orchestration.jobs import IngestionJobManager, JobStore
from app.orchestration.orchestrator import ChannelOrchestrator
from app.retrieval.config import RetrievalConfig
from app.retrieval.milvus_schema import warm_up_japanese_analyzer
from app.retrieval.reformulation import QueryReformulationService
from app.retrieval.registry import RetrieverRegistry
from app.retrieval.vector_db import milvus_connection_args

__all__ = ["app", "create_app"]

_PROMPT_FILES = sorted(Path(__file__).parent.glob("*/prompts/*.md"))  # part of the configuration fingerprint


def create_app(settings: ServerSettings | None = None, *, telemetry: Telemetry | None = None) -> FastAPI:
    """`telemetry`: a ready instance to install instead of configuring one from `settings` (tests)."""
    settings = settings or ServerSettings()
    configure_logging(settings.service_name, settings.environment, level=settings.log_level, log_dir=settings.log_dir)
    log = get_logger(module="API")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        base_config = ChunkerConfig()
        fingerprint = config_fingerprint({**base_config.fingerprint_fields(), "retrieval": asdict(RetrievalConfig())}, _PROMPT_FILES)
        tracer = set_telemetry(telemetry) if telemetry is not None else configure_telemetry(
            enabled=settings.langfuse_enabled,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value() if settings.langfuse_secret_key else None,
            host=settings.langfuse_host,
            timeout_seconds=settings.langfuse_timeout_seconds,
            sample_rate=settings.langfuse_sample_rate,
            environment=settings.environment,
            release=settings.service_version,
            pipeline_version=settings.effective_pipeline_version,
            config_fingerprint=fingerprint,
            capture_query_text=settings.langfuse_capture_query_text,
        )
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
            "Service started: name=%s version=%s env=%s artifact_mode=%s s3_upload=%s tracing=%s pipeline_version=%s config_fingerprint=%s",
            settings.service_name, settings.service_version, settings.environment, settings.artifact_mode, settings.s3_upload_enabled,
            "langfuse" if tracer.enabled else "off", tracer.pipeline_version, tracer.config_fingerprint,
        )
        if base_config.warm_up_japanese_analyzer:
            threading.Thread(target=_warm_up_japanese_analyzer, args=(base_config, log), name="lindera-warm-up", daemon=True).start()
        try:
            yield
        finally:
            jobs.stop()
            application.state.retriever_registry.close_all()
            tracer.shutdown()  # flushes pending spans, bounded by LANGFUSE_TIMEOUT
            log.info("Service stopped: name=%s", settings.service_name)

    application = FastAPI(title=settings.service_name, version=settings.service_version, lifespan=lifespan)
    application.add_middleware(RequestContextMiddleware)
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
