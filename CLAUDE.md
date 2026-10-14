# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single FastAPI service ("Vision RAG"): PDF → OCR (Azure DI or Mistral) → Gemini vision chunking per page →
cross-page merge → Cohere embeddings + Milvus server-side BM25 → hybrid retrieval with fusion + Cohere rerank.
Detailed docs live in `docs/` (ARCHITECTURE, API, CONFIGURATION, ARTIFACTS, CLASSIFICATION, DEPLOYMENT,
INGESTION_RELIABILITY, JAPANESE_RETRIEVAL, HYPERPARAMETERS) — read the relevant one before changing behaviour in that
area, and keep it in sync.

## Commands

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload              # needs .env (see .env.example) and Milvus 2.6+ reachable
docker compose up -d --build               # service on :8001 + standalone Milvus on :19530
docker compose up -d milvus                # Milvus only, for running uvicorn on the host

pytest -q                                  # offline; every provider is faked
pytest tests/test_chunking.py::test_name   # single test
MILVUS_INTEGRATION_URI=http://127.0.0.1:19530 pytest tests/test_vector_db_milvus.py  # live Milvus; creates/drops it_* collections only
ruff check app tests                       # line length 130, rules E,F,I,B
```

`tests/conftest.py` forces `APP_ENV=test`, `ARTIFACT_MODE=production`, `WARM_UP_JAPANESE_ANALYZER=false`, throwaway
`INGEST_WORKSPACE_DIR` / `DEBUG_WORKSPACE` (a job record left in a real one would be resumed by a test app), zero retry
delays, and
the `chunker_config` fixture builds `ChunkerConfig(_env_file=None, ...)` so tests never read the real `.env`.
Note `tests/` and `evaluation/` are listed in `.gitignore` (tests are nonetheless tracked).

The `evaluation/` hyper-parameter harness (not part of the service) is run as modules, e.g.
`python -m evaluation.hp.experiments.e00_sanity`; it needs real API keys and Milvus on `127.0.0.1:19530`, and only
touches collections prefixed `hp_`. See `evaluation/README.md`.

## Architecture

**Lifecycle.** `app/main.py:create_app` lifespan builds every process-scoped object once (`ChunkerConfig`,
`ChannelOrchestrator` + `PDFChannel`, `ProcessingGate`, `IngestionJobManager` (loads job records, may resume one),
`RetrieverRegistry`, `QueryReformulationService`) onto `app.state`; routes reach them only via `app/api/dependencies.py`. Dependencies point downward only:
`api → orchestration → ingestion → {chunking, postprocessing, classification, groundtruth, retrieval} → storage`;
`config`, `logging`, `utils` import nothing from `app`.

**Ingestion** (`POST /v1/documents`, or async `POST /v1/ingestion-jobs` + poll — the path for long PDFs): one document
per process — `app/concurrency.py`'s in-process gate returns 429 when busy, which is why the container runs exactly
**one uvicorn worker**. Uploads are streamed to `<workspace root>/_uploads/` (`app/api/uploads.py`), never held in
memory. `PDFChannel` validates (opens the PDF: encrypted / page limit → 400), admits the file into
`<workspace root>/<document_id>/` (content-hash marker; other content starts clean) and owns S3 publication; production
removes the workspace on success and keeps it on failure, debug keeps it. `IngestionJobManager`
(`app/orchestration/jobs.py`) records jobs under `<workspace root>/_jobs/`, re-runs retryable failures
(`INGEST_JOB_MAX_ATTEMPTS`, each run resuming from the workspace) and continues an interrupted job at startup.
`IngestPipeline` (`app/ingestion/pipeline.py`) runs 8 stages: OCR → render pages → Gemini chunk (per page,
resumable via `ChunkStore.completed_pages()`) → merge → [classify] → vocab → [autogrounding] → [Milvus index].
Stages 1–3 reuse outputs only when newer than the PDF (stage 3 discards older chunks). Every resumable artifact is written with
`app/utils/fs.py` (atomic) — never write one in place. Reliability design: `docs/INGESTION_RELIABILITY.md`.

**Re-index from S3** (`POST /v1/documents/from-s3`, `app/orchestration/s3_import.py`): reads a published folder's
`current.json` version and indexes its `merged_chunks.json` (fallback: `target_chunks.json` merged locally) via
`index_merged_chunks` (stage 8) — no OCR / Gemini; same gate as ingestion. `POST /v1/documents/from-s3/autogrounding`
(`app/orchestration/s3_grounding.py`) runs stage 7 alone on the same chunks and adds `groundtruth_dataset.{json,csv}` to
that S3 version and its manifest (resuming from any groundtruth already there).

**Chunking contract.** Gemini output is schema-constrained to `ChunkingResponse` (`app/chunking/schema.py`); the
same field names are used in the prompts (`app/chunking/prompts/*.md`), `ChunkStore`, the merger and the
retriever — never introduce key-guessing or aliases. Chunks carry `continuation_type` and seam ids
`page_{N}_to_{N+1}` that `app/postprocessing/merger.py` uses to fuse ideas split across pages. `chunk_id`s are
assigned in `ChunkingResponse.to_records` (`page_{N}_chunk_{M}`), never trusted from the model, and merged ids are
made unique — they key Milvus rows and classifications. Artifact formats
are specified in `docs/ARTIFACTS.md`.

**Classification** (stage 5, `app/classification/`, switch `classification.enabled` in `config.yaml`): each merged
chunk gets one `main_category` (a general category, or one the model created — main only) and general-only
`sub_categories`, classified in document-order batches (`classification_batch_size`) so later chunks reuse created categories. `CategoryMemory`
(`classification_memory.json`) keeps general and created categories apart and enforces the rules in code; assignments
are keyed by content fingerprint, so re-runs are free. The keys are written onto merged and target chunks and become
Milvus filter fields (typed + INVERTED-indexed on schema-v2 collections, dynamic field on v1); `app/retrieval/filters.py`
builds the expression. Details: `docs/CLASSIFICATION.md`.

**Retrieval** (`POST /v1/retrieval/query`): `RetrieverRegistry` holds one `EnhancedCohereMilvusRetriever` per
collection for the process lifetime; each has its own thread pool so Milvus/Cohere calls never block the event
loop. `RetrievalConfig` (`app/retrieval/config.py`) is a frozen dataclass; per-request changes go through
`with_overrides()`. Dense and BM25 arms run concurrently and are fused client-side (`app/retrieval/fusion.py`:
rrf / weighted / dbsf) or server-side via Milvus rankers, then optionally reranked.

**Collection names.** Required on every ingest route and never derived from a PDF / folder name. One rule
everywhere: `CollectionName` (`app/api/schemas.py`) → `app/utils/naming.py:normalize_collection_name` (trim, lower-case,
`^[a-z][a-z0-9_]{2,63}$`, else 422 — never silently rewritten). `sanitize_collection_name` is only for deriving a
`document_id` from a file stem. Use `Annotated[CollectionName, Form()]` for form fields — `CollectionName = Form(...)`
silently skips the validator. Each successful ingest registers the document in S3 `_collections/{collection}.json`
(best-effort); query reformulation merges the vocab of every registered document.

**Languages.** Each Milvus collection holds one language (ja or en), fixed by its **first** document.
`app/ingestion/pipeline.py:resolve_document_language` is the one policy (pipeline and S3 import): new collection →
`language` hint or detection (`app/retrieval/language.py`); existing collection → detected language must match, else
409, and the hint is ignored. `app/retrieval/profiles.py` maps language → BM25 analyzer (English standard /
Japanese Lindera IPADIC, needs Milvus 2.6+) and Cohere embed/rerank models. The collection property
`maryuma.manifest` records `{language, embedding_model, dense_dim}` and the retriever binds to it; schema is in
`app/retrieval/milvus_schema.py`. Primary key is `{document_id}:{chunk_id}`; re-ingesting a document (`replace_document`) upserts
its chunks, then deletes its rows that are gone — other documents are untouched.

**Failure policy** (every module docstring states which applies): primary paths raise (OCR, chunk, merge, dense
arm, Milvus upload, S3 upload) — transient provider errors are retried first, and stage 3 attempts every page before
raising `IncompleteStageError` with the failed page numbers; secondary paths degrade and log (BM25 arm → dense-only, reranker → fused order,
reformulation → original query, classification of a chunk → `main_category: null`, S3 retention, collection registry).
Client-caused `ChannelError`s carry their HTTP status (`http_status`: 400, 404 `S3_DOCUMENT_NOT_FOUND`, 409
`DOCUMENT_ID_CONFLICT` — a same-named different PDF, overridable with `replace=true`); a PDF with no chunk is
`NoContentError` (422 `NO_CHUNKS`). Preserve this split when editing.

## Configuration

Two settings objects, never overlapping: `ChunkerConfig` (`app/config/defaults.py` — providers, models, Milvus,
pipeline) and `ServerSettings` (`app/config/settings.py` — HTTP layer, `ARTIFACT_MODE`, S3 defaults, limits).
Priority: env var → `.env` → `config.yaml` → field default. `config.yaml` holds non-secret switches only
(`ocr.technique: azure|mistral`, default mistral; the `classification:` block); secrets stay in `.env`. Full reference:
`docs/CONFIGURATION.md`.
