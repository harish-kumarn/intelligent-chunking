"""
app.api.dependencies
====================
`Depends`-compatible accessors for the process-scoped objects `app.main` places on `app.state`.
Route modules reach shared state only through these.
"""

from __future__ import annotations

import logging

from fastapi import Request

from app.concurrency import ProcessingGate
from app.config.defaults import ChunkerConfig
from app.config.settings import ServerSettings
from app.orchestration.jobs import IngestionJobManager
from app.orchestration.orchestrator import ChannelOrchestrator
from app.retrieval.reformulation import QueryReformulationService
from app.retrieval.registry import RetrieverRegistry

__all__ = [
    "get_base_config", "get_gate", "get_ingestion_jobs", "get_logger", "get_orchestrator",
    "get_query_reformulation_service", "get_retriever_registry", "get_settings",
]


def get_settings(request: Request) -> ServerSettings:
    return request.app.state.settings


def get_base_config(request: Request) -> ChunkerConfig:
    return request.app.state.base_config


def get_orchestrator(request: Request) -> ChannelOrchestrator:
    return request.app.state.orchestrator


def get_gate(request: Request) -> ProcessingGate:
    return request.app.state.gate


def get_ingestion_jobs(request: Request) -> IngestionJobManager:
    return request.app.state.ingestion_jobs


def get_retriever_registry(request: Request) -> RetrieverRegistry:
    return request.app.state.retriever_registry


def get_query_reformulation_service(request: Request) -> QueryReformulationService:
    return request.app.state.query_reformulation_service


def get_logger(request: Request) -> logging.LoggerAdapter:
    return request.app.state.logger
