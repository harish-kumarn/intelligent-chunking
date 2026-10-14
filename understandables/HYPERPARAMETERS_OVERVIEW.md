# Hyper-parameters, Models & Settings — at a glance

Every model, model parameter, hyper-parameter, limit and tunable constant used by the Vision RAG service (`app/`).
This page is a reference only; the service never reads it. The live value is always in the **Source** column.

> **Snapshot:** 2026-10-13 · branch `feat/stage-memory-state`
> **Override priority:** environment variable → `.env` → `config.yaml` → code default.
> Values in `_UPPER_CASE` constants can only be changed in code.
> 💡 = docs/HYPERPARAMETERS.md recommends a different value.

**Contents**
1. [Pipeline map](#1-pipeline-map)
2. [Models](#2-models)
3. [Ingestion](#3-ingestion)
4. [Retrieval](#4-retrieval)
5. [Cohere retry policy](#5-cohere-retry-policy)
6. [Milvus](#6-milvus)
7. [Server, jobs & logging](#7-server-jobs--logging)
8. [S3 artifact store](#8-s3-artifact-store)
9. [Secrets](#9-secrets)
10. [Dependency versions](#10-dependency-versions)

---

## 1. Pipeline map

```
                ┌───────────────────────────── INGESTION (POST /v1/documents · /v1/ingestion-jobs) ─────────────────────────────┐
  PDF ──► 1 OCR ──► 2 Render ──► 3 Chunk ──► 4 Merge ──► 5 Classify ──► 6 Vocab ──► 7 Auto-ground ──► 8 Index ──► Milvus
        Mistral /   PyMuPDF     Gemini       cross-page   Gemini         Groq         Gemini            Cohere embed
        Azure       200 DPI     per page     seams        (optional)     (optional)   (optional)        + BM25

                ┌───────────────────────────── RETRIEVAL (POST /v1/retrieval/query) ───────────────────────────────────────────┐
  query ──► Cohere embed ──► dense top 30 ─┐
                                           ├─► fusion (RRF k=60, α=0.5) ──► top 20 ──► Cohere rerank ──► top 3
            Milvus BM25  ──► sparse top 30 ┘
```

---

## 2. Models

### 2.1 Every model the service calls

| Role | Model | Provider | Pipeline step | Config key | Override |
|---|---|---|---|---|---|
| OCR (default) | `mistral-ocr-latest` 💡 pin a dated version | Mistral | Stage 1 | `mistral_ocr_model` | `MISTRAL_OCR_MODEL` · `ocr.mistral.model` |
| OCR (alternative) | `prebuilt-layout` | Azure Document Intelligence | Stage 1 | `azure_ocr_model` | `AZURE_OCR_MODEL` · `ocr.azure.model` |
| Vision chunking | `gemini-3.5-flash` | Google Gemini | Stage 3 | `chunking_model` | code |
| Classification | `gemini-3.5-flash` | Google Gemini | Stage 5 | `classification_model` | code |
| Auto-grounding (Q&A) | `gemini-3.5-flash` | Google Gemini | Stage 7 | `autogrounding_model` | code |
| Vocabulary dedupe | `openai/gpt-oss-120b` | Groq | Stage 6 (off) | `vocab_model` | code |
| Query reformulation | `openai/gpt-oss-20b` | Groq | `/v1/retrieval/query-reformulation` | `requery_model` | code |
| Embedding — English | `embed-english-v3.0` (1024-d) 💡 `embed-v4.0` | Cohere | Stage 8 + query | `embedding_model` | `EMBEDDING_MODEL` |
| Embedding — Japanese | `embed-v4.0` (1536-d) | Cohere | Stage 8 + query | `japanese_embedding_model` | `JAPANESE_EMBEDDING_MODEL` |
| Rerank — English | `rerank-english-v3.0` | Cohere | query | `reranking_model` | `RERANKING_MODEL` |
| Rerank — Japanese | `rerank-v3.5` | Cohere | query | `japanese_reranking_model` | `JAPANESE_RERANKING_MODEL` |

> A collection keeps the embedding model it was built with (recorded in its `maryuma.manifest`); changing the model only affects new collections.

### 2.2 Model call parameters

| Model call | Temperature | Thinking | Output format | Timeout | Other |
|---|---|---|---|---|---|
| Gemini — chunking | **1.0** (Gemini 3 requirement) | model default (HIGH) 💡 expose; LOW for bulk | JSON, schema `ChunkingResponse` | 180 s | system prompt in a context cache (TTL `43200s`, refreshed between pages); page image as JPEG; `media_resolution` default 💡 MEDIUM |
| Gemini — classification | **1.0** | **low** | JSON | 180 s | "low" halved latency (5.2 s → 2.7 s/chunk) with identical categories |
| Gemini — auto-grounding | **0.2** | default | JSON | 180 s | — |
| Groq — vocabulary dedupe | **0.0** | — | JSON array | SDK default | runs only if `vocab_semantic_dedup = true` |
| Groq — reformulation | **0.0** | — | plain text | SDK default | benchmark: rewriting *hurts* reranked retrieval (−0.022 nDCG) |
| Cohere — embed | — | — | float vectors | SDK default 300 s 💡 10 s | `input_type` = `search_document` / `search_query`; batch **95** (API limit 96) |
| Cohere — rerank | — | — | scores | SDK default 300 s 💡 5 s | `top_n = min(rerank_candidates, #candidates)` |
| Mistral OCR | — | — | markdown + images | 900 s | `include_image_base64 = true`; signed URL expiry 5 h 💡 1 h |
| Azure OCR | — | — | markdown | 900 s | `output_content_format = MARKDOWN` |

### 2.3 Keyword-search (BM25) analyzers — `app/retrieval/profiles.py`

| Language | Tokenizer | Filters | Notes |
|---|---|---|---|
| English (`en`) | `standard` | `lowercase`, English stop words | stemmer tested: no significant gain |
| Japanese (`ja`) | Lindera `ipadic` | `lowercase` | needs Milvus 2.6+; ~45 MB dictionary downloaded on first use. `ipadic-neologd` and `unidic` tested worse |

> The analyzer is fixed when a collection is created.

---

## 3. Ingestion

### Stage 1 — OCR (`app/ingestion/ocr/`)

| Parameter | Value | Range | Override | Source | Notes |
|---|---|---|---|---|---|
| Engine | **mistral** | `azure` / `mistral` | `OCR_TECHNIQUE` · `ocr.technique` | `settings.py` `_OCR_DEFAULTS` | |
| Azure: extract figures | **true** | — | `AZURE_OCR_EXTRACT_FIGURES` | config.yaml | crops figures into `pdf_images/` |
| Azure: figure DPI | **200** | — | `AZURE_OCR_FIGURE_DPI` | config.yaml | |
| Azure: figure padding | **4.0 pt** | — | code | `azure.py` `_FIGURE_PADDING_PT` | |
| Window size | **100 pages** | ≥ 1 | env | `ocr_window_pages` | long PDFs OCR'd in windows; only a failed window is redone |
| Window max size | **45 MiB** | ≥ 1 MiB | env | `ocr_window_max_bytes` | Mistral caps 1,000 pages / 50 MB; oversized window split in half |
| Concurrency | **2** windows | 1–8 | env | `ocr_concurrency` | |
| Max retries | **4** | 1–10 | env | `ocr_max_retries` | |
| Retry base delay | **10 s** | ≥ 0 | env | `ocr_retry_delay` | back-off = delay × 2^(attempt−1) |
| Max back-off | **300 s** | — | code | `ocr/__init__.py` `_MAX_BACKOFF_SECONDS` | |
| Retryable HTTP | **408, 429** + 5xx | — | code | `_RETRYABLE_CLIENT_STATUSES` | |
| Request timeout | **900 s** | > 0 | env | `ocr_request_timeout` | per call (whole PDF or one window) |
| Signed URL expiry | **5 h** 💡 1 h | — | code | `mistral.py` `_SIGNED_URL_EXPIRY_HOURS` | |

### Stage 2 — Page render (`app/utils/pdf_utils.py`)

| Parameter | Value | Source | Notes |
|---|---|---|---|
| Render DPI | **200** | `_RENDER_DPI` | A4 ≈ 1654×2339 JPEG per page |
| Neighbour-page context | **half** of previous / next page OCR text | `split_content` | split on a paragraph boundary |

### Stage 3 — Vision chunking (`app/chunking/chunker.py`)

| Parameter | Value | Range | Override | Source | Notes |
|---|---|---|---|---|---|
| Pages in parallel | **4** | 1–32 | `CHUNKING_CONCURRENCY` | `chunking_concurrency` | also used by auto-grounding |
| Gemini request timeout | **180 s** | > 0 | env | `gemini_request_timeout` | all Gemini stages |
| Max retries | **5** | — | env | `chunking_max_retries` | |
| Retry base delay | **2.0 s** | — | env | `chunking_retry_delay` | |
| Max back-off | **30 s** | — | env | `chunking_timeout` | caps one wait — *not* a request timeout |
| Jitter | **10 %** | — | code | `_JITTER` | |
| Context-cache TTL | **43200s** (12 h) | — | env | `chunking_cache_ttl` | extended between pages; recreated if expired |
| Cache refresh margin | **10 min** | — | code | `_CACHE_REFRESH_MARGIN` | |
| Retryable HTTP | **408, 429, 499** + 5xx | — | code | `_RETRYABLE_CLIENT_CODES` | 403/404 on expired cache → new cache, retry now |
| Fatal HTTP | **401, 403, 404** | — | code | `_FATAL_CODES` | |
| Target chunk size | **400–800 tokens**, soft max 1,000 | — | prompt | `prompts/system_prompt.md` | larger only for one unsplittable idea (`oversized: true`) |
| Vocabulary per chunk | **3–7 terms** | — | prompt | `prompts/system_prompt.md` | `index_concept_vocabulary` |
| Store flush interval | **5 s** | — | code | `pipeline.py` `_STORE_FLUSH_SECONDS` | |

### Stage 4 — Cross-page merge (`app/postprocessing/merger.py`)

| Parameter | Value | Source | Notes |
|---|---|---|---|
| Vocabulary cap per merged chunk | **7** | `_VOCAB_CAP` | matches prompt budget |
| Max forced absorptions | **9** 💡 2 | `_MAX_FORCED_ABSORPTIONS` | bound on seam-less chunks glued together |
| Openers / receivers | `outgoing`,`bidirectional` / `incoming`,`bidirectional` | `_OPENERS` / `_RECEIVERS` | |

### Stage 5 — Classification (`app/classification/`)

| Parameter | Value | Range | Override | Source | Notes |
|---|---|---|---|---|---|
| Enabled | **true** | — | `CLASSIFICATION_ENABLED` · `classification.enabled` · API form field | config.yaml | |
| Allow new categories | **true** | — | `CLASSIFICATION_ALLOW_NEW_CATEGORIES` | config.yaml | new *main* categories only |
| Batch size | **10** chunks | 1–50 | `CLASSIFICATION_BATCH_SIZE` | `classification_batch_size` | document order |
| Batch max chars | **60,000** | ≥ 1,000 | env | `classification_batch_max_chars` | |
| Max text per chunk | **12,000** chars | — | code | `classifier.py` `_MAX_TEXT_CHARS` | longer chunks classified on their opening |
| Max retries | **3** | — | env | `classification_max_retries` | |
| Retry base delay | **2.0 s** | — | env | `classification_retry_delay` | |
| Max back-off | **30 s** | — | code | `_MAX_WAIT` | |
| Max categories | **32** | — | code | `naming.py` `MAX_CATEGORIES` | |
| Category name length | **64** chars | — | code | `MAX_CATEGORY_NAME_LEN` | |

**General categories** (`CLASSIFICATION_CATEGORIES` · `classification.categories`)

| Category | Definition shown to the model |
|---|---|
| `operation` | Using the product in normal conditions: starting, stopping, controls, settings, operating procedures and handling during work. |
| `technical` | Specifications and how the product is built or works: ratings, dimensions, capacities, components, materials, parts and diagrams. |
| `safety` | Hazards and their prevention: warnings, cautions, protective equipment, safe-use rules, prohibited actions and emergency measures. |
| `maintenance` | Keeping the product in working order: inspection, cleaning, lubrication, adjustment, part replacement, service intervals and storage. |
| `troubleshooting` | Diagnosing and fixing faults: symptoms, probable causes, error codes and corrective actions. |

### Stage 6 — Vocabulary index (`app/postprocessing/vocab_index.py`)

| Parameter | Value | Source | Notes |
|---|---|---|---|
| Semantic dedupe | **false** | `vocab_semantic_dedup` | extra Groq pass; falls back to exact dedupe on failure |

### Stage 7 — Auto-grounding (`app/groundtruth/autogrounding.py`)

| Parameter | Value | Source | Notes |
|---|---|---|---|
| Enabled | **false** | API form field `autogrounding` | |
| Max retries | **3** | `autogrounding_max_retries` | back-off 2^(attempt−1) s |
| Direct questions / page | **5–8** 💡 3–5 | `autogrounding_min/max_direct` | |
| Indirect questions / page | **5–8** 💡 3–5 | `autogrounding_min/max_indirect` | |

### Stage 8 — Index into Milvus

| Parameter | Value | Range | Source | Notes |
|---|---|---|---|---|
| Upload to vector DB | **only if `collection_name` given** | — | API form field `upload_vector_db` | |
| Index batch size | **256** chunks | 1–4096 | `index_batch_size` | embed + upsert per batch |
| Writer batch size | **256** | — | `vector_db_writer.py` `DEFAULT_BATCH_SIZE` | |
| Key scan / delete batch | **1,000** | — | `_SCAN_BATCH` | |
| Embedding batch | **95** | — | `embedding_batch_size` | Cohere limit 96 |
| Indexed text | vocabulary + description + content | — | `app/retrieval/documents.py` | removing any part measured worse |

### Language detection (`app/retrieval/language.py`)

| Parameter | Value | Range | Override | Notes |
|---|---|---|---|---|
| Japanese threshold | **0.2** | (0, 1) | `JAPANESE_DETECTION_THRESHOLD` | PDF is `ja` when Japanese script ≥ 20 % of letters |
| Warm up Japanese analyzer | **true** | — | `WARM_UP_JAPANESE_ANALYZER` | fetch Lindera dictionary at startup |
| Force language | — | `ja` / `en` | API form field `language` | |

---

## 4. Retrieval

`RetrievalConfig` — `app/retrieval/config.py` (frozen; per-request copies via `with_overrides()`).
Enforced rule: **final_top_k ≤ rerank_candidates ≤ dense_top_k / sparse_top_k** (upstream depths are widened automatically).

| Parameter | Value | Range | API field | Notes |
|---|---|---|---|---|
| `dense_top_k` | **30** | > 0 | — | depth 50 measured worse at scale |
| `sparse_top_k` | **30** | > 0 | — | BM25 candidates |
| `rerank_candidates` | **20** | > 0 | — | consider 30 past ~1,000 chunks |
| `final_top_k` | **3** 💡 5 | > 0 | `top_k` (1–200 💡 ≤ 50) | |
| `use_hybrid` | **true** | — | `use_hybrid` | dense + BM25 |
| `use_reranking` | **true** | — | `use_reranking` | biggest quality gain (JA 0.885 → 0.970 nDCG@10) |
| `fusion_strategy` | **rrf** | `rrf` / `weighted` / `dbsf` | `fusion_strategy` | |
| `fusion_mode` | **client** | `client` / `server` | — | server = Milvus `hybrid_search`; `dbsf` always client |
| `alpha` | **0.5** | 0–1 | `alpha` | dense share; sparse = 1 − α |
| `rrf_k` | **60** | > 0 | — | score = w_d/(k+rank_d) + w_s/(k+rank_s) |
| `dbsf_sigma` | **3.0** | > 0 | — | z-score clip width |
| `drop_ratio_search` | **0.0** | 0–1 | — | BM25: share of low-weight query terms dropped |
| `executor_workers` | **8** | > 0 | — | thread pool per collection |
| Category filters | — | ≤ 32 values each | `main_categories`, `sub_categories`, `categories` | AND across lists, OR within |
| Reformulated query | — | — | `reformulated_query` | replaces the search text |

💡 **Recommended additions:** a `document_ids` filter (JA nDCG 0.776 → 0.968), optional `min_rerank_score`, and a fallback fusion (`weighted`, α 0.6) when the reranker fails.

---

## 5. Cohere retry policy

`app/retrieval/cohere_retry.py` — shared by embed and rerank.

| Parameter | Value | Notes |
|---|---|---|
| Max attempts | **7** 💡 query path: 2 | |
| Back-off | **min(2^attempt, 60) s** 💡 query path: ≤ 1 s | |
| Retryable errors | `TooManyRequestsError`, `ServiceUnavailableError` | timeouts are *not* retried |
| When retries run out | embed → **raises** · rerank → **keeps fused order** | |

---

## 6. Milvus

### Connection

| Parameter | Value | Override | Notes |
|---|---|---|---|
| Server version | **v2.6.24** | docker-compose | 2.6+ needed for Lindera |
| Host / port | **127.0.0.1 : 19530** | `MILVUS_HOST`, `MILVUS_PORT` | compose uses `milvus` |
| Database | **default** | env | |
| TLS | **false** | `MILVUS_SECURE` | |
| Auth | none | `MILVUS_TOKEN` or `MILVUS_USER` / `MILVUS_PASSWORD` | |
| Call timeout | **none** 💡 10 s | env `MILVUS_TIMEOUT` | |
| Health-probe timeout | **3 s** | code | `health.py` |

### Indexes

| Index | Type | Metric | Parameters | Notes |
|---|---|---|---|---|
| Dense (`dense`) | **FLAT** | **COSINE** | none (`milvus_index_params` / `milvus_search_params` override) | 💡 HNSW `{M:16, efConstruction:200}`, `ef` 64–128 past ~0.5–1 M vectors |
| Sparse (`sparse`) | **SPARSE_INVERTED_INDEX** | **BM25** | `k1 = 1.2`, `b = 0.75` (Milvus defaults) | 💡 `b = 0.5` for new Japanese collections |
| Categories | **INVERTED** | — | — | `main_category`, `sub_categories` |

### Collection schema (v2)

| Field | Type | Notes |
|---|---|---|
| `chunk_id` | VARCHAR(512), primary key | `{document_id}:{chunk_id}` |
| `document_id` | VARCHAR(512) | |
| `page` | INT64 | −1 = no page |
| `text` | VARCHAR(65535) + analyzer | stored text capped at 60,000 bytes |
| `dense` | FLOAT_VECTOR(dim of embedding model) | |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 function over `text` |
| `main_category` | VARCHAR(64), nullable | |
| `sub_categories` | ARRAY<VARCHAR(64)>, max 32, nullable | |

Consistency level **Bounded** · dynamic fields **on** · manifest property `maryuma.manifest` = `{language, embedding_model, dense_dim}` · collection name ≤ 255 chars.

---

## 7. Server, jobs & logging

| Parameter | Value | Override | Notes |
|---|---|---|---|
| Host / port | **0.0.0.0 : 8000** | `SERVER_HOST`, `SERVER_PORT` | compose publishes 8001 |
| Uvicorn workers | **1** | Dockerfile | required: one document at a time |
| Environment | **production** | `APP_ENV` | development/debug/local → console + file logs |
| Artifact mode | **debug** | `ARTIFACT_MODE` | `debug` keeps workspaces; `production` removes on success |
| Debug workspace | `debug` | `DEBUG_WORKSPACE` | |
| Ingest workspace | `<tmp>/visionrag_ingest` | `INGEST_WORKSPACE_DIR` | |
| Workspace retention | **72 h** | `INGEST_WORKSPACE_RETENTION_HOURS` | |
| Max upload size | **200 MiB** | `MAX_UPLOAD_SIZE_BYTES` | |
| Max PDF pages | **2,000** | `MAX_PDF_PAGES` | |
| Upload read chunk | **1 MiB** | code | |
| `Retry-After` (documents, 429) | **5 s** 💡 60 s | code | |
| `Retry-After` (collections, 429) | **5 s** | code | |
| `Retry-After` (ingestion jobs, 429) | **30 s** | code | |
| Job max attempts | **3** | `INGEST_JOB_MAX_ATTEMPTS` (1–10) | each run resumes from the workspace |
| Job retry delay | **60 s × attempt** | `INGEST_JOB_RETRY_DELAY_SECONDS` | |
| Job auto-resume | **true** | `INGEST_JOB_AUTO_RESUME` | after restart |
| Job retention | **24 h** | `INGEST_JOB_RETENTION_HOURS` | |
| Job progress persist | **every 5 s** | code | |
| Log level | **INFO** | `LOG_LEVEL` | |
| Log rotation | **5 MiB × 5 files** | code | |

**Job progress weights per stage** (only ratios matter)

| ocr | chunk | merge | classify | vocab | ground | index | publish |
|---|---|---|---|---|---|---|---|
| 10 | **55** | 1 | 15 | 2 | 10 | 5 | 2 |

---

## 8. S3 artifact store

`app/storage/s3_artifact_store.py`

| Parameter | Value | Override | Notes |
|---|---|---|---|
| Upload enabled | **false** | `S3_UPLOAD_ENABLED` | |
| Bucket / region | none | `S3_BUCKET`, `AWS_DEFAULT_REGION`, API fields | |
| Credentials | boto3 default chain | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` | |
| Versions kept | **2** | code | current + previous |
| Max concurrency | **8** | code | |
| Delete batch | **1,000** | code | S3 hard limit |
| Hash read size | **1 MiB** | code | |

---

## 9. Secrets

Names only — values live in `.env` (template: `.env.example`).

| Variable | Used for | Required |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini: chunking, classification, auto-grounding | ✅ |
| `MISTRAL_API_KEY` | Mistral OCR | when OCR = mistral |
| `AZURE_OCR_ENDPOINT`, `AZURE_OCR_KEY` | Azure OCR | when OCR = azure |
| `COHERE_API_KEY` | embeddings + rerank | ✅ for indexing / retrieval |
| `GROQ_API_KEY` | reformulation, vocabulary dedupe | optional |
| `MILVUS_TOKEN` / `MILVUS_USER` / `MILVUS_PASSWORD` | Milvus auth | if enabled |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3 | optional |

---

## 10. Dependency versions

| Package | Version | Purpose |
|---|---|---|
| Python | 3.12 | runtime |
| fastapi | ≥0.111, <0.142 | HTTP |
| uvicorn | ≥0.29, <0.53 | server |
| pydantic / pydantic-settings | ≥2.7 / ≥2.2, <3 | config |
| google-genai | ≥1.75, <2 | Gemini |
| mistralai | ≥1.12, <2 | OCR |
| azure-ai-documentintelligence | ≥1.0, <2 | OCR |
| cohere | ≥5.21, <6 | embed + rerank |
| groq | ≥0.37, <1 | reformulation, vocab |
| boto3 | ≥1.34, <2 | S3 |
| PyMuPDF | ≥1.24 | page render |
| pymilvus | ≥2.6, <2.7 | vector store |
| langchain-core | ≥0.3, <1 | `Document` type |
