# Ground-truth datasets

Versioned CSV datasets an independent evaluation pipeline scores the service against. **S3 is the canonical
store**; Langfuse Datasets is an optional mirror. Code: `app/datasets/` (contract, validation, service),
`app/storage/s3_dataset_store.py` (immutable versions), `app/api/routes/datasets.py`.

Why S3 and not Langfuse as the source of truth: the producers already write to S3 (`S3DocumentGrounder`,
stage 7), the chunk ids a dataset references are read from S3, versions can be made immutable and
checksummed, and the evaluation pipeline can read them with a prefix-scoped read-only IAM role. Langfuse
versions dataset items by timestamp, items stay mutable, retention follows the plan, and its API keys can
always write.

## 1. Layout

```
_datasets/{module_name}/{dataset_name}/
├── current.json                                  {"module_name", "dataset_name", "dataset_version", "checksum_sha256", "promoted_at"}
│                                                 ← the only mutable object; moved by `promote` only, never implicitly
└── versions/{dataset_version}/
    ├── dataset.csv                               canonical bytes — what checksum_sha256 covers
    ├── manifest.json                             written LAST with IfNoneMatch="*"; its presence proves a complete version
    └── langfuse_sync.json                        sidecar when a Langfuse mirror was requested (bookkeeping, not dataset content)
```

`_datasets/` follows `_collections/`: document ids never start with `_`, so it never clashes with a document
folder. `dataset_version = "v-" + sha256(canonical csv)[:16]`: **content-addressed** — the same rows are always the
same version (a repeated upload is a no-op, `created: false`), different rows are always a different version.
`dataset.csv` and `manifest.json` are written with S3 conditional puts (`IfNoneMatch="*"`), so an existing version
is never overwritten: identical content → "already there", different content under the same version → 409
`DATASET_VERSION_CONFLICT`. Recommended bucket policy: deny `s3:DeleteObject` on `_datasets/*/versions/*`.

## 2. Manifest

```json
{
  "module_name": "hybrid_retrieval", "dataset_name": "operator_manual", "dataset_version": "v-5a0f2e7e645e089d",
  "dataset_schema_version": "retrieval_v1",
  "purpose": "auto-generated retrieval ground truth (direct + indirect questions per page)", "label": null,
  "created_at": "2026-10-19T11:40:02.113422Z", "row_count": 1180,
  "checksum_sha256": "5a0f2e7e645e089d…", "source_sha256": "…", "source_filename_sha256": "…",
  "source_documents": [{"document_id": "Operator_Manual", "version": "9f2c…"}], "references_verified": true,
  "collection_name": null, "columns": ["case_id", "document_id", "page", "type", "tag", "answer_chunk_id", "question", "answer", "language"],
  "pipeline_version": "0.1.0", "service_version": "0.1.0",
  "generator": {"kind": "autogrounding", "model": "gemini-3.5-flash", "document_version": "9f2c…"}
}
```

`source_documents` names the published S3 version of every document the rows reference at validation time, so
a later re-ingestion cannot silently change what a dataset meant.

## 3. CSV contract — `retrieval_v1` (module `hybrid_retrieval`)

The auto-grounding format ([ARTIFACTS.md](ARTIFACTS.md#groundtruth_datasetjson)) plus two leading identity columns.
`module_name` and `dataset_version` are not row columns: the path and the manifest carry them once.

| Column | Required | Rule |
|---|---|---|
| `case_id` | ✔ | `^[A-Za-z0-9._:@-]{1,128}$`, unique in the file; sent back as `X-Case-Id` when the row is replayed |
| `document_id` | ✔ | `^[A-Za-z][A-Za-z0-9_]{0,254}$` — the published document (S3 folder, Milvus `document_id`) |
| `page` | | integer ≥ 0 (first source page; 0 = unknown) |
| `type` | | `direct` \| `indirect` |
| `tag` | | `single` \| `multi_hop` |
| `answer_chunk_id` | ✔ | `;`-separated merged chunk ids, each `^page_\d+(_\d+)?_chunk(_\d+){0,2}$` (unscoped; the Milvus key is `{document_id}:{answer_chunk_id}`) |
| `question` | ✔ | ≤ 4000 characters |
| `answer` | | ≤ 2000 characters |
| `language` | | `ja` \| `en` |

**Validation** (`app/datasets/validation.py`, streaming, memory bounded by the key set): exact header set (any
order; optional columns may be absent), per-column rules, duplicate `case_id`, spreadsheet-executable cells
(`=…`, `@…`, `+cmd`, `-cmd`), `MAX_DATASET_ROWS`, UTF-8 (BOM tolerated). With `verify_references` (default) every
`document_id` must be published and every `answer_chunk_id` must be one of its merged chunks (at most 50
distinct documents per upload). Failures are **422 `DATASET_INVALID`** with up to 50 `issues: [{row, column,
code}]` — row numbers and codes only, never a cell value. Codes: `EMPTY_FILE`, `NO_ROWS`, `MISSING_HEADER`,
`UNKNOWN_HEADER`, `DUPLICATE_HEADER`, `ROW_LENGTH_MISMATCH`, `REQUIRED`, `INVALID_FORMAT`, `TOO_LONG`,
`DUPLICATE_CASE_ID`, `FORMULA_LIKE`, `TOO_MANY_ROWS`, `INVALID_ENCODING`, `MALFORMED_CSV`, `UNKNOWN_DOCUMENT`,
`UNKNOWN_CHUNK_ID`, `TOO_MANY_DOCUMENTS`, `REFERENCE_CHECK_UNAVAILABLE`.

**Canonical form** (what is stored and hashed): every schema column in schema order, cells trimmed, enums
lower-cased, id lists de-duplicated, UTF-8 without BOM, LF line endings. Presentation differences (column
order, BOM, CRLF, surrounding spaces) therefore never create a new version.

Other modules have no schema yet (`SCHEMAS` in `app/datasets/schema.py` is the registry); uploading for one is
422 `NO_DATASET_SCHEMA`.

## 4. API

| | |
|---|---|
| `POST /v1/datasets` (multipart) | `file`, `module_name` (a `ModuleName`), `dataset_name` (`^[a-z][a-z0-9_-]{2,63}$`), `purpose?`, `label?`, `collection_name?`, `promote=false`, `verify_references=true`, `sync_langfuse=false`, `s3_bucket?`, `AWS_DEFAULT_REGION?` → **201** created / **200** identical content already stored |
| `GET /v1/datasets/{module}/{name}` | `{module_name, dataset_name, current, versions[]}` (manifests, newest first) |
| `GET /v1/datasets/{module}/{name}/versions/{version}` | `{manifest, s3_keys}` |
| `POST /v1/datasets/{module}/{name}/current` | `{"dataset_version": "v-…"}` → promote |

Response of an upload: `{module_name, dataset_name, dataset_version, created, row_count, checksum_sha256,
s3_keys, promoted, langfuse_sync, manifest}`. Errors: 413 `DATASET_TOO_LARGE` (`MAX_DATASET_UPLOAD_BYTES`), 422
`DATASET_INVALID` / `NO_DATASET_SCHEMA`, 409 `DATASET_VERSION_CONFLICT`, 404 `DATASET_NOT_FOUND`, 502
`DATASET_STORE_UNAVAILABLE` (retryable), 400 `S3_BUCKET_REQUIRED`. The dataset bucket is `s3_bucket`, else
`S3_DATASET_BUCKET`, else `S3_BUCKET`; references are checked against the artifact bucket (`S3_BUCKET`). These
calls do not take the ingestion gate.

```bash
curl -F file=@operator_manual_qa.csv -F module_name=hybrid_retrieval -F dataset_name=operator_manual_qa \
     -F promote=true localhost:8000/v1/datasets
```

Uploads are traced as `dataset_management` (`dataset_validate` → `dataset_store` → `langfuse_dataset_sync`
spans; counts and ids only).

## 5. Auto-grounding → dataset

`groundtruth_dataset.csv` is a valid `retrieval_v1` file: every question gets a deterministic
`case_id = {document_id}:{sha256(page|type|question)[:12]}`, `answer_chunk_id`s the model invented (not among the
page's chunks) are dropped and a question left without one is dropped (counted in the log).
`POST /v1/documents/from-s3/autogrounding` with `publish_dataset=true` (+ `dataset_name` — default: the document id
lower-cased — `promote`, `sync_langfuse`) registers the CSV as a dataset version with
`generator.kind = "autogrounding"`; the response carries `dataset`. Being content-addressed, re-running the
grounding on an unchanged document yields `created: false`.

## 6. Langfuse mirror (optional)

`sync_langfuse=true` mirrors the version into the Langfuse dataset `{module_name}/{dataset_name}`: dataset
metadata records `s3_bucket`, `s3_key`, `dataset_version`, `checksum_sha256`, `dataset_schema_version`; each
row is an item with a **stable id** `sha256("{module}/{dataset}|{case_id}")[:32]` (so a new version upserts in
place), `input = {question}`, `expected_output = {answer_chunk_ids, answer}`, metadata `{case_id, document_id,
page, type, tag, language, dataset_version, checksum_sha256}`. Items missing from the synced version are
archived, so "latest" in Langfuse equals exactly one S3 version; `synced_at` (the Langfuse version timestamp,
`get_dataset(name, version=synced_at)`) is written to `langfuse_sync.json`. Datasets above
`LANGFUSE_DATASET_SYNC_MAX_ITEMS` rows are skipped (one API call per row). A mirror failure is reported in the
response (`langfuse_sync.status = "failed"`) — the S3 version is already published.

## 7. How the evaluation pipeline reads a dataset

```python
import boto3, csv, hashlib, io, json
s3, bucket = boto3.client("s3"), "<artifact bucket>"
base = "_datasets/hybrid_retrieval/operator_manual"
current = json.load(s3.get_object(Bucket=bucket, Key=f"{base}/current.json")["Body"])   # or pin a version
version = current["dataset_version"]
manifest = json.load(s3.get_object(Bucket=bucket, Key=f"{base}/versions/{version}/manifest.json")["Body"])
data = s3.get_object(Bucket=bucket, Key=f"{base}/versions/{version}/dataset.csv")["Body"].read()
assert hashlib.sha256(data).hexdigest() == manifest["checksum_sha256"] == current["checksum_sha256"]
rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
# replay: POST /v1/retrieval/query with X-Execution-Kind: evaluation, X-Execution-Id, X-Dataset-Name,
#         X-Dataset-Version: version, X-Case-Id: row["case_id"]; expected Milvus keys: f"{row['document_id']}:{id}"
```

No IC → evaluation call exists and none is needed: the pipeline pulls the dataset from S3 and the traces from
Langfuse ([OBSERVABILITY.md §6](OBSERVABILITY.md#6-how-the-evaluation-pipeline-reads-traces)).

## 8. IAM

| Role | Permissions |
|---|---|
| service | existing artifact permissions ([DEPLOYMENT.md](DEPLOYMENT.md#s3-permissions)) plus `s3:PutObject`, `s3:GetObject` on `arn:…:<bucket>/_datasets/*` and `s3:ListBucket` with `s3:prefix` = `_datasets/*`; recommended **Deny** `s3:DeleteObject` on `<bucket>/_datasets/*/versions/*` |
| evaluation pipeline | `s3:GetObject` on `<bucket>/_datasets/*`, `s3:ListBucket` with `s3:prefix` = `_datasets/*` — nothing else |

## 9. Limits

`MAX_DATASET_UPLOAD_BYTES` (25 MiB, enforced while streaming), `MAX_DATASET_ROWS` (50 000), 50 distinct
documents per reference check, 50 issues per error body, `LANGFUSE_DATASET_SYNC_MAX_ITEMS` (5 000).
