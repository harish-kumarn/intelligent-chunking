"""
app.chunking.chunker
====================
IntelligentPDFChunker — Gemini-vision semantic chunking of one OCR'd page at a time.

    system prompt : Gemini context cache, shared across documents and kept alive (see Cache lifecycle)
    user prompt   : page text + neighbour context, rendered from prompts/user_prompt.md
    page image    : JPEG bytes attached to every call for OCR verification
    output        : `ChunkingResponse` (schema-constrained JSON), persisted via ChunkStore

Main path
    generate_page ─► build prompt ─► generate (retry w/ back-off) ─► parse+validate ─► record tokens
    chunk_page    ─► generate_page ─► store

Each page is one `chunk_page` generation in the caller's trace (model, token usage summed over attempts,
retry count, chunk count — never the prompt, the image or the output).

Cache lifecycle (a long document outlives any fixed TTL: ~20 s per page, so ~170 pages per hour)
    acquire : reuse a live cache whose display name carries the hash of model + system prompt (so a
              changed prompt never reuses a stale cache), else create one
    refresh : before each page, extend the TTL when less than `_CACHE_REFRESH_MARGIN` remains
    recover : a refresh or a page call that finds the cache gone (expired / deleted: 403 or 404
              "CachedContent") creates a new cache and retries that page at once, without back-off

Failure policy: transient Gemini errors (5xx, 408, 429, timeouts) and unparsable output are retried
up to `chunking_max_retries`; the last error propagates. A 4xx answer is not retried — the same
request would fail again — and `is_fatal_gemini_error` tells the pipeline when no page can succeed
(bad key, no permission, unknown model), so it stops instead of failing every page in turn. Listing caches is secondary (any error → create a
new cache); creating one is primary. Every Gemini call has `gemini_request_timeout`. A missing API
key fails at construction.
Concurrency: one chunker per document, thread-safe — stage 3 calls `chunk_page` from
`chunking_concurrency` threads. A lock guards the cache state; when several pages find the cache
gone at once, only the first creates a new one (the others see the name has changed).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.chunking.schema import ChunkingInput, ChunkingResponse
from app.chunking.store import ChunkStore
from app.chunking.token_monitor import TokenUsageMonitor
from app.config.defaults import ChunkerConfig
from app.app_logging.setup import get_logger
from app.observability.modules import SpanName
from app.observability.telemetry import get_telemetry
from app.utils.gemini import gemini_client
from app.utils.json_utils import repair_and_parse_json

__all__ = ["IntelligentPDFChunker", "is_fatal_gemini_error"]

_PROMPT_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT = (_PROMPT_DIR / "system_prompt.md").read_text(encoding="utf-8")
_USER_PROMPT = (_PROMPT_DIR / "user_prompt.md").read_text(encoding="utf-8")
_CACHE_DISPLAY_PREFIX = "visionrag_chunker_"
_CACHE_REFRESH_MARGIN = timedelta(minutes=10)  # > the slowest page incl. its retries, so no call meets an expired cache
_CACHE_GONE_CODES = frozenset({403, 404})  # Gemini answers 403 PERMISSION_DENIED for an expired cache
_CACHE_LIST_PAGE_SIZE = 100
_JITTER = 0.1  # 10 % added to every back-off wait so retries from parallel runs de-synchronise
_RETRYABLE_CLIENT_CODES = frozenset({408, 429, 499})  # timeout, quota, cancelled: worth another attempt
_FATAL_CODES = frozenset({401, 403, 404})             # auth / permission / unknown model (403 cache errors excepted)
_NO_PREVIOUS = "No previous page — first page."
_NO_NEXT = "No next page — last page."


class IntelligentPDFChunker:
    """Per-document chunking engine; one instance amortises the system-prompt cache across pages."""

    def __init__(
        self,
        config: ChunkerConfig,
        store: ChunkStore,
        token_monitor: TokenUsageMonitor,
        logger: Optional[logging.LoggerAdapter] = None,
    ) -> None:
        """
        Args:
            config:        Model, retry, and cache settings; `google_api_key` is required.
            store:         Destination for each page's chunk records.
            token_monitor: Receives per-attempt token usage.
            logger:        Optional logger; the app logger is used when omitted.
        """
        self._cfg = config
        self._store = store
        self._tokens = token_monitor
        self._log = logger or get_logger(module="CHUNKING")

        if not config.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for vision chunking.")
        self._client = gemini_client(config.google_api_key, config.gemini_request_timeout)
        self._cache_lock = threading.Lock()
        self._cache_display_name = _cache_display_name(config.chunking_model)
        self._cache_ttl = _ttl_seconds(config.chunking_cache_ttl)
        self._cache_name = ""
        self._cache_expires = datetime.min.replace(tzinfo=timezone.utc)
        self._acquire_cache()
        self._log.info("Chunker ready: model=%s cache=%s", config.chunking_model, self._cache_name)

    @property
    def prompt_fingerprint(self) -> str:
        """Hash of model + system prompt (the cache display-name suffix): names the prompt a trace was produced with."""
        return self._cache_display_name[len(_CACHE_DISPLAY_PREFIX):]

    # ------------------------------------------------------------------ #
    # System-prompt cache
    # ------------------------------------------------------------------ #

    def _acquire_cache(self) -> None:
        """Reuse a live cache of this model + prompt (TTL extended), else create one."""
        existing = self._find_cache()
        if existing is not None:
            self._use(existing)
            self._log.info("System-prompt cache reused: cache=%s expires=%s", self._cache_name, self._cache_expires)
            self._refresh_cache(force=True)
            return
        self._create_cache()

    def _find_cache(self) -> Optional[Any]:
        """A cache with our display name and model that outlives the refresh margin; secondary — None on any error."""
        horizon = _utcnow() + _CACHE_REFRESH_MARGIN
        try:
            for cache in self._client.caches.list(config={"page_size": _CACHE_LIST_PAGE_SIZE}):
                expires = getattr(cache, "expire_time", None)
                if (
                    getattr(cache, "display_name", None) == self._cache_display_name
                    and str(getattr(cache, "model", "")).endswith(self._cfg.chunking_model)
                    and isinstance(expires, datetime)
                    and expires > horizon
                ):
                    return cache
        except Exception as exc:  # listing only saves a create; never fail a document over it
            self._log.warning("System-prompt cache lookup failed, creating a new one: %s", exc)
        return None

    def _create_cache(self) -> None:
        """Primary — a failure here fails the document, as before."""
        cache = self._client.caches.create(
            model=self._cfg.chunking_model,
            config=genai_types.CreateCachedContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                ttl=self._cfg.chunking_cache_ttl,
                display_name=self._cache_display_name,
            ),
        )
        self._use(cache)
        self._log.info("System-prompt cache created: cache=%s expires=%s", self._cache_name, self._cache_expires)

    def _refresh_cache(self, *, force: bool = False) -> None:
        """Extend the TTL when it is close to expiry; recreate the cache when it is already gone."""
        with self._cache_lock:
            if force or self._cache_expires - _utcnow() <= _CACHE_REFRESH_MARGIN:
                self._extend_cache()

    def _extend_cache(self) -> None:
        """Lock held."""
        try:
            cache = self._client.caches.update(
                name=self._cache_name,
                config=genai_types.UpdateCachedContentConfig(ttl=self._cfg.chunking_cache_ttl),
            )
        except genai_errors.APIError as exc:
            if exc.code not in _CACHE_GONE_CODES:
                raise
            self._log.warning("System-prompt cache gone (%s), creating a new one: cache=%s", exc.code, self._cache_name)
            self._create_cache()
            return
        self._use(cache)
        self._log.info("System-prompt cache extended: cache=%s expires=%s", self._cache_name, self._cache_expires)

    def _use(self, cache: Any) -> None:
        """Track the cache's name and expiry (the API's when given, else now + TTL)."""
        expires = getattr(cache, "expire_time", None)
        self._cache_name = cache.name
        self._cache_expires = expires if isinstance(expires, datetime) else _utcnow() + self._cache_ttl

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def chunk_page(self, page: ChunkingInput) -> ChunkingResponse:
        """generate_page ─► persist. Returns the parsed response."""
        response, attempt = self._generate_page(page)
        added = self._store.append_page(response, page.page_number)
        self._log.info("Page %d chunked: %d chunk(s) on attempt %d", page.page_number, added, attempt)
        return response

    def generate_page(self, page: ChunkingInput) -> ChunkingResponse:
        """prompt ─► generate (retried) ─► validate, without persisting."""
        return self._generate_page(page)[0]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _generate_page(self, page: ChunkingInput) -> tuple[ChunkingResponse, int]:
        prompt = _render_user_prompt(page)
        image_bytes = page.page_image_path.read_bytes()
        retries = self._cfg.chunking_max_retries

        # One generation per page, retries folded in: usage summed over attempts, `retry_count` recorded.
        with get_telemetry().generation(
            SpanName.CHUNK_PAGE, model=self._cfg.chunking_model, page_number=page.page_number, prompt_fingerprint=self.prompt_fingerprint,
        ) as generation:
            usage_total: dict[str, int] = {}
            for attempt in range(1, retries + 1):
                cache_name = self._cache_name
                try:
                    self._refresh_cache()
                    cache_name = self._cache_name
                    response, usage = self._generate(prompt, image_bytes, page.page_number, attempt, cache_name)
                except Exception as exc:
                    self._log.warning("Page %d attempt %d/%d failed: %s", page.page_number, attempt, retries, exc)
                    if attempt == retries or not _is_retryable(exc):
                        generation.set(retry_count=attempt - 1)
                        raise
                    if _cache_gone(exc):  # expired or deleted under us: a new cache, then retry at once
                        self._replace_cache(cache_name)
                        continue
                    time.sleep(self._backoff(attempt))
                    continue
                for key, value in usage.items():
                    usage_total[key] = usage_total.get(key, 0) + value
                generation.set(retry_count=attempt - 1, chunk_count=len(response.chunks))
                generation.set_usage(**usage_total)
                return response, attempt

        raise RuntimeError(f"chunk_page exhausted retries for page {page.page_number}")  # unreachable

    def _replace_cache(self, gone: str) -> None:
        """Create a new cache unless another page already replaced `gone`."""
        with self._cache_lock:
            if self._cache_name == gone:
                self._create_cache()

    def _generate(self, prompt: str, image_bytes: bytes, page_number: int, attempt: int, cache_name: str) -> tuple[ChunkingResponse, dict[str, int]]:
        """One Gemini call — errors propagate to the retry loop. Returns the response and its token usage."""
        result = self._client.models.generate_content(
            model=self._cfg.chunking_model,
            contents=[
                prompt,
                "**Page Image:**",
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ChunkingResponse,
                cached_content=cache_name,
                temperature=self._cfg.chunking_temperature,
            ),
        )
        usage = self._record_usage(result.usage_metadata, page_number, attempt)
        return ChunkingResponse.model_validate(repair_and_parse_json(result.text or "")), usage

    def _record_usage(self, usage: Any, page_number: int, attempt: int) -> dict[str, int]:
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        candidate_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
        thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        self._tokens.record(
            page_number=page_number,
            attempt=attempt,
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            cached_tokens=cached_tokens,
            thinking_tokens=thinking_tokens,
        )
        # Langfuse usage_details: `input` / `output` / `total` are its standard keys, the rest are detail buckets.
        return {
            "input": prompt_tokens, "output": candidate_tokens + thinking_tokens, "total": prompt_tokens + candidate_tokens + thinking_tokens,
            "input_cached_tokens": cached_tokens, "output_reasoning_tokens": thinking_tokens,
        }

    def _backoff(self, attempt: int) -> float:
        wait = min(self._cfg.chunking_retry_delay * (2 ** (attempt - 1)), self._cfg.chunking_timeout)
        return wait * (1 + _JITTER)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _cache_display_name(model: str) -> str:
    """Stable per model + system prompt, so any process reuses a matching cache and a prompt change never does."""
    digest = hashlib.sha256(f"{model}\n{_SYSTEM_PROMPT}".encode("utf-8")).hexdigest()[:16]
    return f"{_CACHE_DISPLAY_PREFIX}{digest}"


def _ttl_seconds(ttl: str) -> timedelta:
    """Gemini duration text ("3600s") → timedelta."""
    return timedelta(seconds=float(str(ttl).strip().rstrip("s")))


def _cache_gone(exc: Exception) -> bool:
    """Gemini's answer to a call that references an expired or deleted context cache."""
    return (
        isinstance(exc, genai_errors.APIError)
        and exc.code in _CACHE_GONE_CODES
        and "cachedcontent" in str(exc).lower().replace(" ", "")
    )


def is_fatal_gemini_error(exc: BaseException) -> bool:
    """An error every page would hit: invalid key, no permission, unknown model."""
    if not isinstance(exc, genai_errors.APIError) or _cache_gone(exc):
        return False
    return exc.code in _FATAL_CODES or (exc.code == 400 and "api key" in str(exc).lower())


def _is_retryable(exc: BaseException) -> bool:
    """Everything but a 4xx answer (other than timeout / quota / cancelled) — and a gone cache is always retried."""
    if not isinstance(exc, genai_errors.APIError) or _cache_gone(exc):
        return True
    code = exc.code or 0
    return not 400 <= code < 500 or code in _RETRYABLE_CLIENT_CODES


def _render_user_prompt(page: ChunkingInput) -> str:
    return _USER_PROMPT.format(
        page_number=page.page_number,
        previous_page_number=page.page_number - 1,
        next_page_number=page.page_number + 1,
        page_text=page.page_text,
        previous_page_text=page.previous_page_text or _NO_PREVIOUS,
        next_page_text=page.next_page_text or _NO_NEXT,
    )
