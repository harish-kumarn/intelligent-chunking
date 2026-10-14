"""
app.classification.classifier
=============================
ChunkClassifier — Gemini classification of merged chunks, validated against the CategoryMemory.

    system prompt : prompts/system_prompt.md (the rules)
    user prompt   : general + created categories so far + the chunk, from prompts/user_prompt.md
                    (batch: several chunks, from prompts/user_prompt_batch.md)
    output        : `ClassificationResponse` (schema-constrained JSON) ─► `CategoryMemory.resolve`
                    (batch: `ClassificationBatchResponse`, raw answers by chunk_id; the stage resolves them)

Main path
    classify ─► render prompt ─► generate ─► parse ─► resolve (rules) ─► Resolution
                                 └─ any failure, including a rule violation, is retried with back-off

Failure policy: the last error propagates after `classification_max_retries` attempts — the
stage decides what a failed chunk means (a failed batch falls back to `classify` per chunk). Every
call has `gemini_request_timeout`. A missing API key fails at construction.
Not thread-safe; chunks (and batches) are classified in sequence so each sees the categories created before it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from google.genai import types as genai_types

from app.classification.memory import CategoryMemory, Resolution
from app.classification.schema import ClassificationBatchResponse, ClassificationResponse
from app.config.defaults import ChunkerConfig
from app.logging.setup import get_logger
from app.utils.gemini import gemini_client
from app.utils.json_utils import repair_and_parse_json

__all__ = ["ChunkClassifier", "prompt_chars", "render_batch_prompt", "render_user_prompt"]

_PROMPT_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT = (_PROMPT_DIR / "system_prompt.md").read_text(encoding="utf-8")
_USER_PROMPT = (_PROMPT_DIR / "user_prompt.md").read_text(encoding="utf-8")
_BATCH_PROMPT = (_PROMPT_DIR / "user_prompt_batch.md").read_text(encoding="utf-8")
_MAX_TEXT_CHARS = 12_000   # long fused chunks are classified on their opening; bounds cost per call
_TRUNCATED = "\n[… truncated]"
_NONE = "(none)"
_MAX_WAIT = 30.0           # seconds; cap on a single back-off wait


class ChunkClassifier:
    """Per-document classifier; one Gemini client per instance."""

    def __init__(self, config: ChunkerConfig, logger: Optional[logging.LoggerAdapter] = None) -> None:
        self._cfg = config
        self._log = logger or get_logger(module="CLASSIFICATION")
        if not config.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for chunk classification.")
        self._client = gemini_client(config.google_api_key, config.gemini_request_timeout)

    def classify(self, chunk: Mapping[str, Any], memory: CategoryMemory, fingerprint: str) -> Resolution:
        """prompt ─► generate ─► resolve, retried. Returns a validated, uncommitted Resolution."""
        prompt = render_user_prompt(chunk, memory)
        chunk_id = chunk.get("chunk_id")
        retries = self._cfg.classification_max_retries
        for attempt in range(1, retries + 1):
            try:
                return memory.resolve(self._generate(prompt), fingerprint)
            except Exception as exc:
                self._log.warning("Chunk %s classification attempt %d/%d failed: %s", chunk_id, attempt, retries, exc)
                if attempt == retries:
                    raise
                time.sleep(min(self._cfg.classification_retry_delay * (2 ** (attempt - 1)), _MAX_WAIT))
        raise RuntimeError(f"classify exhausted retries for chunk {chunk_id}")  # unreachable

    def classify_batch(self, chunks: Sequence[Mapping[str, Any]], memory: CategoryMemory) -> Dict[str, ClassificationResponse]:
        """
        One call for several chunks → raw answers by chunk_id (unresolved: the caller resolves and commits
        them in order). Chunks the model skipped are absent. Transport / parse errors are retried; the last propagates.
        """
        prompt = render_batch_prompt(chunks, memory)
        wanted = {str(chunk.get("chunk_id")) for chunk in chunks}
        retries = self._cfg.classification_max_retries
        for attempt in range(1, retries + 1):
            try:
                response = ClassificationBatchResponse.model_validate(self._call(prompt, ClassificationBatchResponse))
            except Exception as exc:
                self._log.warning("Batch of %d chunk(s) attempt %d/%d failed: %s", len(chunks), attempt, retries, exc)
                if attempt == retries:
                    raise
                time.sleep(min(self._cfg.classification_retry_delay * (2 ** (attempt - 1)), _MAX_WAIT))
                continue
            answers: Dict[str, ClassificationResponse] = {}
            for item in response.classifications:
                if item.chunk_id in wanted and item.chunk_id not in answers:
                    answers[item.chunk_id] = ClassificationResponse.model_validate(item.model_dump(exclude={"chunk_id"}))
            return answers
        raise RuntimeError("classify_batch exhausted retries")  # unreachable

    def _generate(self, prompt: str) -> ClassificationResponse:
        return ClassificationResponse.model_validate(self._call(prompt, ClassificationResponse))

    def _call(self, prompt: str, schema: type) -> Any:
        result = self._client.models.generate_content(
            model=self._cfg.classification_model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=self._cfg.classification_temperature,
                thinking_config=_thinking(self._cfg.classification_thinking_level),
            ),
        )
        return repair_and_parse_json(result.text or "")


def _thinking(level: Optional[str]) -> Optional[genai_types.ThinkingConfig]:
    return genai_types.ThinkingConfig(thinking_level=level) if level else None


def render_user_prompt(chunk: Mapping[str, Any], memory: CategoryMemory) -> str:
    description, vocabulary, text = _chunk_parts(chunk)
    return _USER_PROMPT.format(
        general_categories=_listing(memory.general),
        created_categories=_listing({name: c.description for name, c in memory.created.items()}),
        chunk_description=description,
        concept_vocabulary=vocabulary,
        chunk_text=text,
    )


def render_batch_prompt(chunks: Sequence[Mapping[str, Any]], memory: CategoryMemory) -> str:
    sections = []
    for chunk in chunks:
        description, vocabulary, text = _chunk_parts(chunk)
        sections.append(
            f"### chunk_id: {chunk.get('chunk_id')}\n\nDescription: {description}\n\n"
            f"Concept vocabulary: {vocabulary}\n\nText:\n<<<\n{text}\n>>>"
        )
    return _BATCH_PROMPT.format(
        general_categories=_listing(memory.general),
        created_categories=_listing({name: c.description for name, c in memory.created.items()}),
        chunks="\n\n".join(sections),
        count=len(chunks),
    )


def prompt_chars(chunk: Mapping[str, Any]) -> int:
    """Characters a chunk adds to a batch prompt (the batching budget)."""
    return sum(len(part) for part in _chunk_parts(chunk))


def _chunk_parts(chunk: Mapping[str, Any]) -> tuple[str, str, str]:
    text = (chunk.get("content") or "").strip()
    if len(text) > _MAX_TEXT_CHARS:
        text = text[:_MAX_TEXT_CHARS] + _TRUNCATED
    vocabulary = ", ".join(str(term).strip() for term in chunk.get("concept_vocabulary") or [] if str(term).strip())
    return (chunk.get("chunk_description") or "").strip() or _NONE, vocabulary or _NONE, text or _NONE


def _listing(categories: Mapping[str, str]) -> str:
    if not categories:
        return _NONE
    return "\n".join(f"- `{name}`: {description}" if description else f"- `{name}`" for name, description in categories.items())
