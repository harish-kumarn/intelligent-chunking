# API reference

Base path `/v1`. All error bodies share one shape:

```json
{"detail": {"error": {"code": "SERVER_BUSY", "message": "…", "retryable": true}}}
```

Pydantic validation failures return FastAPI's standard 422 body.

## Request identity and tracing headers (every route)

Every response carries `X-Request-Id`. Optional request headers declare why a call is made
([OBSERVABILITY.md](OBSERVABILITY.md#2-execution-context-request-headers)): `X-Execution-Kind`
(`production` · `ingestion_test` · `retrieval_test` · `evaluation`), `X-Execution-Id`, `X-Dataset-Name`,
`X-Dataset-Version`, `X-Case-Id` (ids: `^[A-Za-z0-9._:@-]{1,128}$`; an invalid value is **422
`INVALID_EXECUTION_HEADER`**). When Langfuse tracing is on, a request that opened a trace returns `X-Trace-Id`.

## Collection names and languages (every route)

**`collection_name` is always given explicitly** — it is required on every ingest route and is never derived
from a file or folder name. Every route (form field, JSON body or path) validates it with one rule,
`CollectionName` (`app/utils/naming.py:normalize_collection_name`):

- trimmed and lower-cased, so `Manuals_EN` and `manuals_en` are the same collection;
- then it must match `^[a-z][a-z0-9_]{2,63}$` — 3–64 characters, lowercase letters, digits and `_`, starting
  with a letter. Anything else (`my col`, `1abc`, `a-b`, `ab`) is a **422** naming the rule; a name is never
  silently rewritten into a different one.

**The first document fixes a collection's language.** A collection holds one language (`ja` / `en`), recorded
when its first document creates it. For every later document the language is *detected* from its text and
must equal the collection's, else **409 `COLLECTION_LANGUAGE_MISMATCH`** (checked after OCR, before any Gemini
spend). The `language` field is only a hint for the first document of a new collection; for an existing
collection it is ignored (logged), so a caller cannot force a document into a collection of the other language.

---

## `POST /v1/ingestion-jobs` · `GET /v1/ingestion-jobs/{job_id}`

The asynchronous form of `POST /v1/documents`, for documents that outlive any HTTP timeout (a 1,000-page
manual takes hours). Same form fields as below.

| | Response |
|---|---|
| `POST` | **202** + the job, `Location: /v1/ingestion-jobs/{job_id}`; **200** + the existing job when the same document *and content* already has an active job (a retried submit attaches). 400 invalid upload (incl. over `MAX_UPLOAD_SIZE_BYTES`, unreadable, password-protected, more than `MAX_PDF_PAGES` pages) / language, 422 missing or invalid `collection_name`, 409 `DOCUMENT_BUSY` (the document is being ingested from another file), 429 `SERVER_BUSY` (`Retry-After: 30`; shares the single-flight gate with `/v1/documents`) |
| `GET` | the job; 404 `JOB_NOT_FOUND` — unknown, or finished more than `INGEST_JOB_RETENTION_HOURS` ago. Jobs are recorded on disk, so they survive a restart |

Job: `{job_id, document_id, status: queued|running|succeeded|failed, stage, stage_done, stage_total, progress, stages,
attempt, max_attempts, created_at, started_at, finished_at, result, error, execution_id, trace_ids}`. `execution_id`
is the Langfuse session of the job's traces (`X-Execution-Id` at submit time, else the job id); `trace_ids` holds one
trace id per attempt (empty when tracing is off). `stage` is one of `ocr` (windows) ·
`chunk` (pages) · `merge` · `classify` (chunks) · `vocab` · `ground` (pages) · `index` (chunks) · `publish`;
`progress` (0–1) weights the stages this job runs and never goes backwards. `result` is the `/v1/documents`
response; `error` is `{code, message, retryable}` — a job's failure is never an HTTP status.

**Retries are automatic.** A run that fails with a retryable error is run again (`attempt` counts runs, up to
`max_attempts` = `INGEST_JOB_MAX_ATTEMPTS`), each one resuming from the kept workspace: finished OCR windows,
chunked pages, classifications and grounded pages are reused, so e.g. only the pages that failed are chunked again.
A restart continues the running job by itself (`INGEST_JOB_AUTO_RESUME`). A job that still ends with
`retryable: true` — attempts spent, or `INTERRUPTED` when it could not be continued — is safe to resubmit as is:
the run resumes from the same workspace. `retryable: false` means the request itself is wrong (e.g. the
collection holds another language, `DOCUMENT_ID_CONFLICT` — resubmit with `replace=true` — or `NO_CHUNKS`).
A fatal Gemini error (invalid key, no permission, unknown model) fails at once without service retries; it
stays `retryable: true`, so resubmitting after fixing the configuration resumes. The form takes the same
fields as `/v1/documents`, including `replace`. For stage 3 the message names the pages that failed every retry.
See [INGESTION_RELIABILITY.md](INGESTION_RELIABILITY.md).

## `POST /v1/documents`

Multipart upload. Runs the full ingestion pipeline **synchronously** — the connection stays open for the
duration (minutes for large PDFs; prefer `/v1/ingestion-jobs` above ~100 pages). One document per process at a time.

| Form field | Type | Default | Meaning |
|---|---|---|---|
| `file` | file | required | a `.pdf`, streamed to disk: ≤ `MAX_UPLOAD_SIZE_BYTES`, `%PDF-` signature, readable by PyMuPDF, not password-protected, 1–`MAX_PDF_PAGES` pages — all checked before any paid stage |
| `collection_name` | str | **required** | Milvus collection to index into — see [collection names](#collection-names-and-languages-every-route) |
| `autogrounding` | bool | `false` | also generate the Q&A evaluation set |
| `language` | `ja` \| `en` \| `auto` | `auto` | hint for the first document of a **new** collection; ignored for an existing one, whose language is authoritative |
| `classification` | bool | `CLASSIFICATION_ENABLED` | run stage 5 (main/sub category per chunk) — see [CLASSIFICATION.md](CLASSIFICATION.md) |
| `s3_bucket` | str | `S3_BUCKET` | where artifacts are published when `S3_UPLOAD_ENABLED=true` |
| `AWS_DEFAULT_REGION` | str | `AWS_DEFAULT_REGION` | |
| `replace` | bool | `false` | allow replacing a *different* PDF already published under the same `document_id` (same file name) — see below |

**Document identity.** `document_id` is the sanitised file stem. With `S3_UPLOAD_ENABLED=true`, S3's
`{document_id}/current.json` records which content holds that id: uploading a *different* PDF with the same
file name is **409 `DOCUMENT_ID_CONFLICT`** (before any paid stage) unless `replace=true`, which replaces that
document's rows and S3 artifacts. Re-uploading the same PDF is always allowed (a re-ingest). Without S3 the
service keeps no record of earlier content, and a same-named PDF replaces the earlier one.

**200**

```json
{
  "document_id": "operator_manual",
  "status": "completed",
  "collection_name": "manuals",
  "language": "en",
  "page_count": 132,
  "chunk_count": 418,
  "uploaded_to_milvus": true,
  "classified": true,
  "created_categories": ["warranty"],
  "artifacts": { "ocr_markdown_path": "…", "target_chunks_path": "…", "merged_chunks_path": "…",
                 "vocab_index_path": "…", "classification_path": "…", "groundtruth_path": null }
}
```

`artifacts` is present only in `ARTIFACT_MODE=debug`. `document_id` is the sanitised file stem and is the
S3 document key. `created_categories` lists the main categories the model added to the general ones for
this document (`[]` when none, `null` when classification did not run).

| Status | Code | When |
|---|---|---|
| 400 | `INVALID_DOCUMENT` | not a PDF, empty, over size, or `s3_bucket` missing while S3 upload is enabled |
| 400 | `INVALID_LANGUAGE` | `language` is not `ja`, `en` or `auto` |
| 409 | `COLLECTION_LANGUAGE_MISMATCH` | the document is detected as the other language than the collection's (checked after OCR, before chunking) |
| 409 | `DOCUMENT_ID_CONFLICT` | a different PDF is already published under this file name; rename it or send `replace=true` |
| 422 | `NO_CHUNKS` | the PDF produced no indexable chunk (blank / nothing to chunk); nothing is indexed or published |
| 422 | — | `collection_name` missing or not matching the rule |
| 409 | `LEGACY_COLLECTION` | the collection predates language profiles and is read-only — ingest into a new one |
| 429 | `SERVER_BUSY` | another document is processing (`Retry-After: 5`) |
| 500 | `PROCESSING_FAILED` | any pipeline / Milvus / S3 error (retryable) |

```bash
curl -F file=@manual.pdf -F collection_name=manuals -F autogrounding=true localhost:8000/v1/documents
```

---

## `POST /v1/documents/from-s3`

Index a document that an earlier ingestion already published to S3, **without** OCR, Gemini chunking or
any other LLM stage: only Cohere embedding and the Milvus upsert run (`app/orchestration/s3_import.py`).
Synchronous; shares the single-flight gate with `/v1/documents`. Reads S3 only — nothing is written back.

The folder must follow the [S3 layout](ARTIFACTS.md#s3-layout-s3_upload_enabledtrue): `{folder_name}/current.json`
names the live version, whose `manifest.json` must exist. From that version, `merged_chunks.json` is indexed
as published (categories included); when it is missing or empty, `target_chunks.json` is merged by the
service (pure code; the result carries no categories). Re-indexing replaces exactly this document's rows
in the collection, like any ingestion.

JSON body:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `folder_name` | str | required | the top-level S3 folder, i.e. the document's `document_id` (no `/` inside) |
| `collection_name` | str | **required** | Milvus collection to index into — never the folder name |
| `language` | `ja` \| `en` \| `auto` | `auto` | hint for a new collection only; otherwise detected from the chunk text and checked against the collection |
| `s3_bucket` | str | `S3_BUCKET` | bucket to read from (independent of `S3_UPLOAD_ENABLED`) |
| `AWS_DEFAULT_REGION` | str | `AWS_DEFAULT_REGION` | |

**200**

```json
{
  "document_id": "operator_manual",
  "status": "completed",
  "version": "9f2c…",
  "source_artifact": "merged_chunks.json",
  "collection_name": "manuals_en",
  "language": "en",
  "page_count": 132,
  "chunk_count": 418,
  "uploaded_to_milvus": true,
  "classified": true
}
```

`version` is the S3 version (PDF content hash) that was indexed. `page_count` counts the version's
`page_images/` in its manifest (the distinct source pages of the chunks when there are none).

| Status | Code | When |
|---|---|---|
| 400 | `S3_BUCKET_REQUIRED` | no `s3_bucket` and no `S3_BUCKET` default |
| 400 | `INVALID_LANGUAGE` | as for `/v1/documents` |
| 404 | `S3_DOCUMENT_NOT_FOUND` | no `current.json`, the version has no manifest, or neither chunk file holds chunks |
| 409 | `COLLECTION_LANGUAGE_MISMATCH` / `LEGACY_COLLECTION` | as for `/v1/documents` |
| 422 | — | `folder_name` blank or nested; `collection_name` missing or invalid |
| 429 | `SERVER_BUSY` | another document is processing (`Retry-After: 5`) |
| 500 | `PROCESSING_FAILED` | S3 / Cohere / Milvus error (retryable) |

```bash
curl -H 'Content-Type: application/json' -d '{"folder_name": "operator_manual", "collection_name": "manuals_en"}' \
     localhost:8000/v1/documents/from-s3
```

---

## `POST /v1/documents/from-s3/autogrounding`

Run auto-grounding (stage 7) **alone** on a document already published to S3, and save
`groundtruth_dataset.json` / `groundtruth_dataset.csv` into the same S3 version
(`{folder_name}/versions/{current hash}/`), adding them to its `manifest.json`
(`app/orchestration/s3_grounding.py`). Synchronous: one Gemini call per page, so a long document holds the
connection for minutes. Shares the single-flight gate with `/v1/documents`. Needs `GOOGLE_API_KEY`.

The input is the same as for `/v1/documents/from-s3` and is resolved the same way (`merged_chunks.json`, with
`target_chunks.json` merged as a fallback). Its body needs `folder_name` (+ optional `s3_bucket` /
`AWS_DEFAULT_REGION`). If the version
already has a `groundtruth_dataset.json`, it is the starting point: pages already grounded are kept, and
pages that failed (`error`) or are new are generated. Calling again therefore retries only `failed_pages`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `publish_dataset` | bool | `false` | also register the CSV as an immutable `hybrid_retrieval` dataset version ([DATASETS.md §5](DATASETS.md#5-auto-grounding--dataset)) |
| `dataset_name` | str | the document id, lower-cased | `^[a-z][a-z0-9_-]{2,63}$`; required when the document id cannot be one (400) |
| `promote` | bool | `false` | make that version the dataset's current one |
| `sync_langfuse` | bool | `false` | mirror it into Langfuse Datasets |

**200**

```json
{
  "document_id": "operator_manual",
  "status": "completed",
  "version": "9f2c…",
  "source_artifact": "merged_chunks.json",
  "page_count": 132,
  "question_count": 1180,
  "failed_pages": [57],
  "artifacts": ["operator_manual/versions/9f2c…/groundtruth_dataset.json",
                "operator_manual/versions/9f2c…/groundtruth_dataset.csv"],
  "dataset": {"module_name": "hybrid_retrieval", "dataset_name": "operator_manual", "dataset_version": "v-5a0f2e7e645e089d",
              "created": true, "row_count": 1180, "checksum_sha256": "…", "s3_keys": ["…/dataset.csv", "…/manifest.json"],
              "promoted": false, "langfuse_sync": null}
}
```

`dataset` is present only with `publish_dataset=true`. Errors are as for `/v1/documents/from-s3` (400 `S3_BUCKET_REQUIRED`,
404 `S3_DOCUMENT_NOT_FOUND`, 422, 429 `SERVER_BUSY`, 500 `PROCESSING_FAILED`: a Gemini key problem, or an S3 read/upload
failure, in which case the manifest is left unchanged), plus the dataset errors (422 `DATASET_INVALID`, 409
`DATASET_VERSION_CONFLICT`, 502 `DATASET_STORE_UNAVAILABLE`) — raised after the groundtruth artifacts were saved.

---

## Datasets — `POST /v1/datasets` · `GET /v1/datasets/{module_name}/{dataset_name}[/versions/{version}]` · `POST …/current`

Versioned ground-truth CSVs in S3 for the evaluation pipeline. Full contract, layout, manifest and error codes:
[DATASETS.md](DATASETS.md). Upload (multipart): `file`, `module_name` (one of the canonical modules —
`hybrid_retrieval` is the only one with a schema today), `dataset_name`, `purpose?`, `label?`, `collection_name?`,
`promote=false`, `verify_references=true`, `sync_langfuse=false`, `s3_bucket?`, `AWS_DEFAULT_REGION?` → **201** with
`{module_name, dataset_name, dataset_version, created, row_count, checksum_sha256, s3_keys, promoted, langfuse_sync, manifest}`,
**200** when identical content was already stored (`created: false`).

```bash
curl -F file=@operator_manual_qa.csv -F module_name=hybrid_retrieval -F dataset_name=operator_manual_qa -F promote=true \
     localhost:8000/v1/datasets
curl localhost:8000/v1/datasets/hybrid_retrieval/operator_manual_qa
curl -H 'Content-Type: application/json' -d '{"dataset_version": "v-5a0f2e7e645e089d"}' \
     localhost:8000/v1/datasets/hybrid_retrieval/operator_manual_qa/current
```

```bash
curl -H 'Content-Type: application/json' -d '{"folder_name": "operator_manual"}' \
     localhost:8000/v1/documents/from-s3/autogrounding
```

---

## `POST /v1/retrieval/query`

```json
{
  "collection_name": "manuals",
  "query": "how do I cold start the engine",
  "top_k": 5,
  "use_hybrid": true,
  "use_reranking": true,
  "alpha": 0.5,
  "fusion_strategy": "rrf",
  "reformulated_query": null,
  "main_categories": ["maintenance"],
  "sub_categories": null,
  "categories": null
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `collection_name` | str | required | same rule as on ingest (`Manuals_EN` → `manuals_en`; invalid → 422) |
| `query` | str | required | non-blank; logged/returned as given |
| `top_k` | int 1–200 | `RetrievalConfig.final_top_k` (3) | search depths widen automatically so `top_k` is always reachable |
| `use_hybrid` | bool | `true` | `false` → dense only |
| `use_reranking` | bool | `true` | ignored when no Cohere key |
| `alpha` | float 0–1 | `0.5` | dense share of the fused score |
| `fusion_strategy` | `rrf` \| `weighted` \| `dbsf` | `rrf` | see ARCHITECTURE §4 |
| `reformulated_query` | str | null | if set, this is what is searched; `query` is kept for logs |
| `main_categories` | list[str] ≤ 32 | null | only chunks whose main category is one of these |
| `sub_categories` | list[str] ≤ 32 | null | only chunks with at least one of these sub categories |
| `categories` | list[str] ≤ 32 | null | only chunks with one of these as main **or** sub category |

Category filters are AND-ed with each other; values within one list are OR-ed. Names are normalised like
the stored ones (`"Trouble Shooting"` → `trouble_shooting`); a name with no ASCII letters or digits is a 422.
The filter runs inside every Milvus search, before fusion and reranking, so `top_k` is filled from matching
chunks only. Unclassified chunks never match a filter. Filtering a pre-manifest (legacy) collection is a
400 `INVALID_REQUEST`.

**200** — `{"results": [Hit, …]}` best-first; the hit shape is documented in ARCHITECTURE §4.

| Status | Code | When |
|---|---|---|
| 400 | `INVALID_REQUEST` | value rejected by `RetrievalConfig` |
| 404 | `COLLECTION_NOT_FOUND` | no such collection |
| 500 | `RETRIEVAL_FAILED` | embedding or dense search failed (BM25 and rerank failures degrade instead) |

---

## `POST /v1/retrieval/query-reformulation`

Rewrites `query` using the terminology of **every document in the collection**: the S3 collection registry
`_collections/{collection_name}.json` (written on each ingest — see [ARTIFACTS.md](ARTIFACTS.md#s3-layout-s3_upload_enabledtrue))
lists the documents, and each one's current `vocab_index.json` is merged. A collection without a registry
(ingested before it existed) is read as the single document folder named like the collection. The combined
vocabulary is cached for 60 s, so a newly ingested document is picked up within a minute.
Fail-safe: any missing piece (no Groq key, no registry or published version, no vocabulary, Groq error) returns the
original query unchanged; one unreadable document is skipped, not fatal.

```json
{"collection_name": "manuals", "query": "engine won't start when cold", "s3_bucket": null, "AWS_DEFAULT_REGION": null}
```

`s3_bucket` / `AWS_DEFAULT_REGION` default to `S3_BUCKET` / `AWS_DEFAULT_REGION`; a 400 `S3_BUCKET_REQUIRED` is returned when
neither is available.

**200** — `{"query": "…", "reformulated_query": "…"}`

Typical use: call this first, then pass the result as `reformulated_query` to `/v1/retrieval/query`.

---

## `GET /v1/collections`

Lists the collections in the database this service ingests into (`MILVUS_DB_NAME`).

**200**

```json
{"db_name": "default", "count": 2, "collections": ["manuals", "reports"]}
```

**500 `MILVUS_UNAVAILABLE`** (retryable) when Milvus cannot be reached.

## `DELETE /v1/collections/{collection_name}`

Drops one collection. The name follows the same rule as everywhere else (`My_Col` → `my_col`; invalid → 422).
The service also forgets its cached retriever for that collection, so a later ingestion under the
same name starts clean, and (best-effort, when `S3_BUCKET` is set) deletes its S3 registry
`_collections/{name}.json`. Document artifacts on S3 are **not** touched.

**200** — `{"db_name": "default", "collection_name": "my_col", "deleted": true}`

| Status | `code` | When |
|---|---|---|
| 422 | — | the name does not match the rule |
| 404 | `COLLECTION_NOT_FOUND` | no such collection |
| 429 | `SERVER_BUSY` (retryable, `Retry-After: 5`) | an ingestion is running — a drop never races a write |
| 500 | `MILVUS_UNAVAILABLE` (retryable) | Milvus cannot be reached |

---

## `GET /v1/health` → `{"status": "ok"}`

Liveness only.

## `GET /v1/ready`

```json
{"status": "ready", "milvus": "reachable", "accepting_documents": true}
```

503 with the same body when Milvus is unreachable. `accepting_documents=false` means an ingestion is in
progress — queries are still served.
