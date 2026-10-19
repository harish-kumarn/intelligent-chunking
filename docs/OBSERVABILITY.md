# Observability (Langfuse)

Every workflow run is one Langfuse trace, named after its **module** and tagged with why it ran, so an
independent evaluation pipeline can list the runs of a module, join them to ground-truth rows and score them —
without the service ever calling that pipeline. The code lives in `app/observability/` (a leaf package beside
`app/logging/`); the SDK is `langfuse==5.0.2` (OpenTelemetry-based).

Tracing is **off by default** (`LANGFUSE_ENABLED=false`): the facade is then a no-op and nothing in the service
changes. It is a *secondary path* in the sense of [ARCHITECTURE.md §7](ARCHITECTURE.md#7-failure-policy): an SDK
error is logged (WARNING at most once a minute per operation) and the run continues; exports happen on the SDK's
background thread, bounded by `LANGFUSE_TIMEOUT`.

## 1. Module registry

`app/observability/modules.py` is the only source of trace names. The facade accepts the enums only — a string
cannot open a new trace group.

| `module_name` (root trace) | Entry point | One trace per |
|---|---|---|
| `document_ingestion` | `PDFChannel.process` | ingest attempt (`POST /v1/documents`, each attempt of a `/v1/ingestion-jobs` job) |
| `s3_reindex` | `S3DocumentImporter.import_document` | `POST /v1/documents/from-s3` |
| `autogrounding` | `S3DocumentGrounder.ground_document` | `POST /v1/documents/from-s3/autogrounding` |
| `hybrid_retrieval` | `EnhancedCohereMilvusRetriever.retrieve_async` | `POST /v1/retrieval/query` (dense-only runs included; `strategy` says which) |
| `query_reformulation` | `QueryReformulationService.reformulate` | `POST /v1/retrieval/query-reformulation` |
| `dataset_management` | `DatasetService` | dataset upload / promotion ([DATASETS.md](DATASETS.md)) |

Child spans (`SpanName`): `ocr`, `render`, `intelligent_chunking` (with one `chunk_page` **generation** per page),
`merge`, `classify`, `vocab`, `autogrounding`, `milvus_index`, `s3_publish`; `dense_search`, `bm25_search`,
`fusion`, `rerank`; `dataset_validate`, `dataset_store`, `langfuse_dataset_sync`. BM25 and dense search are arms
of a retrieval, never modules. A module run started inside another (a dataset published by auto-grounding) is a
child span of the outer trace, with `nested_module` in its metadata — one trace per request.

## 2. Execution context (request headers)

| Header | Values | Where it lands |
|---|---|---|
| `X-Execution-Kind` | `production` (default) · `ingestion_test` · `retrieval_test` · `evaluation` | tag `kind:<value>`, metadata `execution_kind` |
| `X-Execution-Id` | `^[A-Za-z0-9._:@-]{1,128}$` | Langfuse **session id** (groups the traces of one run); default: the request id, or the job id for an ingestion job |
| `X-Dataset-Name`, `X-Dataset-Version` | same rule | tag `dataset:<name>@<version>`, metadata |
| `X-Case-Id` | same rule | metadata `case_id` — the ground-truth row this request replays |

An invalid value is **422 `INVALID_EXECUTION_HEADER`** before any work. Every response carries `X-Request-Id`;
a request that opened a trace also carries `X-Trace-Id` (the Langfuse trace id). Ingestion jobs record the
context with the job (`_jobs/<id>.json`), restore it in the job thread and expose `execution_id` and `trace_ids`
(one per attempt) on `GET /v1/ingestion-jobs/{id}`.

Context (request id, execution context, the current span) travels on contextvars: thread pools are bound with
`app.logging.setup.in_context`, and the retriever's executor calls are wrapped the same way, so every span has
its parent whatever thread it ran on.

## 3. What a trace carries

Trace-level attributes (set with `propagate_attributes`, so every child span has them):

| Attribute | Value |
|---|---|
| name | the module name |
| tags | `module:<m>`, `kind:<k>`, `dataset:<name>@<version>` when given |
| session id | execution id |
| version | `PIPELINE_VERSION` (default: the service version) |
| release / environment | service version / `APP_ENV` (sanitised to Langfuse's form) |
| metadata | `module_name`, `execution_kind`, `execution_id`, `dataset_name`, `dataset_version`, `case_id`, `attempt`, `request_id`, `document_id`, `collection_name`, `content_hash`, `pipeline_version`, `config_fingerprint` |

`config_fingerprint` is sha256[:12] over an explicit allowlist of behaviour-affecting settings
(`ChunkerConfig.fingerprint_fields()`: OCR engine and models, chunking / classification / auto-grounding models
and tuning, embedding and rerank models, Milvus index defaults, the default `RetrievalConfig`) plus every
`app/*/prompts/*.md` file. Two runs with the same fingerprint were produced by the same models, prompts and
tuning. Keys, hosts, paths, timeouts are not part of it.

Per-module metadata and outputs:

```
document_ingestion  meta: language_hint, classification, autogrounding, replace, ocr_technique, chunking_model
  ├ ocr {ocr_technique} → {page_count}          ├ render {page_count}
  ├ intelligent_chunking {chunking_model, prompt_fingerprint, concurrency, pages_total, pages_pending, pages_resumed} → {page_count, failed}
  │   └ chunk_page ×N  generation {model, page_number, prompt_fingerprint, retry_count, chunk_count}
  │                    usage_details {input, output, total, input_cached_tokens, output_reasoning_tokens} (summed over retries)
  ├ merge → {chunks_out}   ├ classify {classification_model, batch_size} → {classified, failed}   ├ vocab
  ├ autogrounding {autogrounding_model}
  ├ milvus_index {collection_name, language, chunk_count, batch_size} → {rows_upserted, embedding_model}
  └ s3_publish {version} → {uploaded_files, already_existed}
  output: {status, page_count, chunk_count, language, uploaded_to_milvus, classified}

hybrid_retrieval  meta: collection_name, strategy (dense | hybrid_client | hybrid_server), fusion_strategy, fusion_mode, alpha, rrf_k,
                  top_k, dense_top_k, sparse_top_k, rerank_candidates, use_hybrid, use_reranking, filter_present, reformulated,
                  language, embedding_model, reranking_model
  input: {query_sha256_12, query_chars, reformulated}   — or {query: "…"} with LANGFUSE_CAPTURE_QUERY_TEXT=true
  ├ dense_search {top_k, filter_present} → {hits}
  ├ bm25_search {top_k, filter_present, language} → {hits, degraded}   (ERROR level when the arm failed)
  ├ fusion {fusion_strategy, fusion_mode, alpha, rrf_k} → {candidates}
  └ rerank {model, candidates} → {applied}
  output: {chunk_ids: [scoped ids "{document_id}:{chunk_id}"], scores, result_count, candidates, applied}

s3_reindex        meta: document_id, collection_name, language_hint, version, source_artifact, chunk_count; output: {status, language, chunk_count, page_count, classified}
autogrounding     meta: document_id, autogrounding_model, version, source_artifact, chunk_count[, dataset_name]; output: {page_count, question_count, failed_pages[, dataset_version]}
query_reformulation meta: collection_name, model; input: query fingerprint; output: {reformulated, status, query_sha256_12, query_chars}
dataset_management  meta: dataset_name, collection_name, dataset_schema_version, verify_references; output: {dataset_version, created, row_count, promoted, langfuse_sync_status}
```

Errors: a failed span gets level `ERROR`, `status_message = "<error_category>: <ErrorType>"` and metadata
`error_type`, `error_category` (`rate_limited | timeout | client_error | not_found | conflict | provider_error | internal`),
plus `http_status` / `error_code` where known. The message text is recorded **only** for the service's own
client-facing errors (`ChannelError`, `RetrievalError`, `NoContentError` — the same text the HTTP client gets);
provider messages are not. OpenTelemetry's raw `record_exception` never runs.

### What is never recorded

Chunk text, prompts, model output, page images, document content, query text (unless the explicit capture flag
is on), API keys, bucket names, file paths. `app/observability/sanitize.py` enforces it: metadata keys must be on
`METADATA_KEYS` (anything else is dropped), strings are capped at 256 characters, lists at 200 items. There is no
key for text, so a span cannot carry it by accident.

Only spans created by the facade are exported (`should_export_span=is_langfuse_span`). The SDK's default also
exports third-party GenAI instrumentation — the Mistral SDK instruments its own OCR and file calls
(`ocr_v1_ocr_post`, `files_api_routes_*`), which would bypass the allowlist above. Found and closed during the
first live run; covered by `tests/test_telemetry.py::test_third_party_instrumentation_is_never_exported`.

Example root metadata of an evaluation replay:

```json
{"module_name": "hybrid_retrieval", "execution_kind": "evaluation", "execution_id": "eval-run-2026-10-19",
 "dataset_name": "operator_manual", "dataset_version": "v-5a0f2e7e645e089d", "case_id": "operator_manual:3f1c9a0b2e7d",
 "attempt": 1, "request_id": "bc80f556d3bf44779f160ee580037904", "collection_name": "manuals_en",
 "pipeline_version": "0.1.0", "config_fingerprint": "59e4c2f6992c"}
```

## 4. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LANGFUSE_ENABLED` | `false` | `true` requires both keys, else the service fails at startup |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | project keys (`.env` / environment; never `config.yaml`). The secret is a `SecretStr`: it never appears in logs or `repr` |
| `LANGFUSE_HOST` (or `LANGFUSE_BASE_URL`) | SDK default (Langfuse Cloud) | self-hosted / EU base URL, e.g. `http://localhost:3000` for the local Docker stack |
| `LANGFUSE_TIMEOUT` | `5` s | per export batch; also bounds the flush at shutdown |
| `LANGFUSE_SAMPLE_RATE` | `1.0` | fraction of traces kept |
| `LANGFUSE_CAPTURE_QUERY_TEXT` | `false` | record query text as trace input (non-production evaluation environments) |
| `LANGFUSE_DATASET_SYNC_MAX_ITEMS` | `5000` | cap for the Langfuse Datasets mirror ([DATASETS.md](DATASETS.md)) |
| `PIPELINE_VERSION` | service version | the `version` of every trace |

The SDK's own `LANGFUSE_FLUSH_AT` / `LANGFUSE_FLUSH_INTERVAL` environment variables still apply to batching.

## 5. Verified SDK API (langfuse 5.0.2)

The facade (`app/observability/telemetry.py`) uses exactly: `Langfuse(public_key, secret_key, base_url, timeout,
sample_rate, environment, release, tracing_enabled, should_export_span, span_exporter, tracer_provider)`,
`start_as_current_observation(name, as_type="span" | "generation", input, metadata, model)`,
`observation.update(metadata, output, usage_details, level, status_message)`, `propagate_attributes(session_id,
version, tags, metadata, trace_name)`, `flush()`, `shutdown()`; the current trace id is read from OpenTelemetry's
current span. The dataset mirror uses `create_dataset(name, description, metadata)`,
`create_dataset_item(dataset_name, id, input, expected_output, metadata, status)`, `get_dataset(name)`.
`span_exporter` / `tracer_provider` injection is what the tests use (in-memory exporter, no network).
Re-verify this list before moving the pin.

## 6. How the evaluation pipeline reads traces

1. Pick the dataset version from S3 ([DATASETS.md §6](DATASETS.md#6-how-the-evaluation-pipeline-reads-a-dataset)).
2. Replay each row against `POST /v1/retrieval/query` with `X-Execution-Kind: evaluation`,
   `X-Execution-Id: <run id>`, `X-Dataset-Name`, `X-Dataset-Version`, `X-Case-Id: <case_id>`; keep `X-Trace-Id`.
3. Query the Langfuse public API. On a **Langfuse v4 server** (the local Docker stack, Langfuse Cloud's current
   API) `GET /api/public/traces` answers 404; the real-time read path is **`GET /api/public/v2/observations`**:
   `name=hybrid_retrieval`, `isRootObservation=true`, `sessionId=<run id>` (or `version=<pipeline_version>`,
   `fromStartTime` / `toStartTime`), `fields=basic,io,metadata,trace_context`, `limit≤1000` + `cursor` paging.
   Filter on `tags` (returned by `trace_context`: `module:hybrid_retrieval`, `kind:evaluation`) and join on
   `metadata.case_id`. `output` is a JSON string whose `chunk_ids` are the scoped Milvus keys
   `{document_id}:{answer_chunk_id}` of the dataset row. A whole trace tree: `traceId=<id>`. (Verified against
   the local v4 stack: 33 replayed cases → 33 root observations, each with its `case_id`.)
4. Write scores with the Langfuse scores API against each `traceId`. The service computes no scores.

Trace populations are told apart by **explicit tags**, never by name matching: `kind:production` (live traffic),
`kind:ingestion_test`, `kind:retrieval_test`, `kind:evaluation`. Langfuse API keys are project-scoped (there are
no read-only keys): give the evaluation pipeline its own project keys and treat them as write-capable.
