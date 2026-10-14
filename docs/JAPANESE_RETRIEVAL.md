# Japanese retrieval — language profiles

The service indexes and searches Japanese PDFs as well as English ones. Each collection holds
**one language**, detected from the PDF, and everything that depends on language — the BM25
analyzer, the embedding model and the reranker — follows the collection.

| | English collection | Japanese collection |
|---|---|---|
| BM25 analyzer (`text` field) | `standard` + lowercase + English stop words | Lindera IPADIC + lowercase |
| Embedding model | `EMBEDDING_MODEL` (`embed-v4.0`; was `embed-english-v3.0`, whose 512-token window truncated long chunks) | `JAPANESE_EMBEDDING_MODEL` (`embed-v4.0`) |
| Reranker | `RERANKING_MODEL` (`rerank-english-v3.0`) | `JAPANESE_RERANKING_MODEL` (`rerank-v3.5`) |
| Milvus server | 2.5+ | **2.6+** (Lindera is not in the 2.5.x images) |

## 1. How it works

```
POST /v1/documents (file, collection_name, language?)
  OCR ─► resolve_document_language                                  app/ingestion/pipeline.py
           new collection      : language = form field, or detect_language(OCR text)
           existing collection : detect_language(OCR text) must equal the collection's language
                                 (form field ignored)               409 before any Gemini spend
  render ─► chunk ─► merge ─► vocab ─► [ground]
  index ─► EnhancedCohereMilvusRetriever(collection, language=…)
            first write: create the collection for the language's profile  app/retrieval/milvus_schema.py
              text   VARCHAR  analyzer_params = profile analyzer
              dense  FLOAT_VECTOR(dim of the embedding model)
              sparse BM25(text), computed by Milvus
              property maryuma.manifest = {language, embedding_model, dense_dim}
            upsert rows (embedded with the collection's model) ─► flush ─► load

POST /v1/retrieval/query (collection_name, query)
  RetrieverRegistry.get(collection) ─► retriever binds to the manifest once and keeps it
  dense  : query embedded with the manifest's embedding model
  sparse : raw query; Milvus tokenizes it with the field's analyzer
  fuse (client RRF/weighted/DBSF, or server hybrid_search) ─► rerank with the language's reranker
```

**Language detection** (`detect_language`) counts scripts over the letters of the OCR text:
Japanese (hiragana, katakana, kanji, half-width katakana) against Latin (ASCII and full-width).
A PDF is `ja` when Japanese makes up at least `JAPANESE_DETECTION_THRESHOLD` (0.2) of its letters.
Product codes such as `BCL200S` inside Japanese prose do not move the decision; an English
manual has no Japanese script at all. The `language` form field (`ja` / `en`) overrides detection only
for the first document of a new collection — that document fixes the collection's language.

**Why one language per collection.** Dense vectors are only comparable when one model produced
them all, and Milvus fixes a field's analyzer when the collection is created — by its first
document. Ingesting a later document whose *detected* language differs from its collection's returns
**409 `COLLECTION_LANGUAGE_MISMATCH`**, right after OCR and before chunking; the `language` field cannot
override that. `POST /v1/documents/from-s3` applies the same rule to the published chunk text.

**Why queries need no language.** The retriever takes the analyzer and embedding model from the
collection, never from the query. An ASCII query such as `BCL200S` against a Japanese
collection is tokenized by Lindera (`bcl / 200 / s`), exactly like the rows it must match.

**The manifest is authoritative.** A collection keeps searching with the embedding model it
was built with, even after `JAPANESE_EMBEDDING_MODEL` changes (a warning is logged). To move a
collection to a new model, re-ingest it into a new collection. Rerankers do not touch the index,
so they always follow the current configuration.

**Legacy collections** (created before this change by the langchain-milvus layout) have no
manifest. They stay searchable as English collections, with the English models, and are
read-only: ingesting into one returns **409 `LEGACY_COLLECTION`**.

## 2. Benchmark

The corpus is the BCL200S/BCL200SL brush cutter manual (33 pages, 55 chunks), with 301 scored
Japanese queries across 16 categories. The dataset is the one on branch
`feat/language-based-tokenzier` (`evaluation/dataset/`). The run used the production
`EnhancedCohereMilvusRetriever` on Milvus 2.6.24 on 2026-09-30.

The full production path is hybrid + rerank; the α column is the dense weight in RRF.

| Setup | BM25 only | Dense only | Hybrid, α 0.5 | **Hybrid + rerank, α 0.6** | **Hybrid + rerank, α 0.5** |
|---|---|---|---|---|---|
| A. `develop` before: standard analyzer, English models | 0.271 | 0.503 | 0.597 | 0.806 | 0.830 |
| B. Lindera + English models | 0.847 | 0.503 | 0.726 | 0.858 | 0.917 |
| C. Lindera + `embed-multilingual-v3.0` + `rerank-multilingual-v3.0` | 0.847 | 0.797 | 0.866 | 0.916 | 0.923 |
| D. Lindera + `embed-multilingual-v3.0` + `rerank-v3.5` | 0.847 | 0.797 | 0.866 | 0.958 | 0.965 |
| **E. Lindera + `embed-v4.0` + `rerank-v3.5` (default)** | 0.847 | **0.843** | **0.880** | 0.959 | **0.965** |

All figures are nDCG@10. Recall@10 for E at α 0.5 is 0.984, and MRR is 0.972. Median latency
with rerank is about 0.5 s, almost all of it the Cohere rerank call.

- **The analyzer is the largest single gain.** BM25 goes from 0.27 to 0.85 (A→B).
- **Multilingual models are the second.** Dense search goes from 0.50 to 0.84 (B→E).
- **`rerank-v3.5` beats `rerank-multilingual-v3.0`** by 0.04 (C→D).
- **D and E tie on the full path.** E is the default because its dense arm is stronger, which
  matters when the reranker is disabled or unavailable.
- **α = 0.5 beat 0.6 in every reranked setup.** At 0.6, a chunk that only BM25 finds ranks
  below every dense hit, so it never reaches the rerank window. The `RetrievalConfig.alpha`
  default is now 0.5.
- **English collections keep the English models.** There is no English benchmark here, so their
  defaults are unchanged.

## 3. Operations

### Milvus 2.6 and the Lindera dictionary

- `docker-compose.yml` runs `milvusdb/milvus:v2.6.24` with `DEPLOY_MODE: STANDALONE` (2.6
  refuses embedded etcd without it) on the volume `milvus-data-v26`.
- On first use, Milvus downloads the IPADIC dictionary (about 45 MB) into
  `/var/lib/milvus/analyzer/lindera`, inside the data volume. The container needs outbound
  network access **once**. Offline hosts must pre-seed that directory from a machine that has
  already built it.
- At startup the service warms Lindera up in the background (`WARM_UP_JAPANESE_ANALYZER=true`).
  When that fails (Milvus < 2.6, no network), it logs an ERROR. English keeps working;
  Japanese ingestion fails with "the Lindera analyzer needs Milvus 2.6+".

### Moving from Milvus 2.5.x

**2.6 runs on a fresh volume (`milvus-data-v26`); the 2.5 volume is kept, not upgraded.**
On 2026-09-30, 2.6.24 started on the existing 2.5.12 dev volume panicked on every start with
`panic: etcdserver: leader changed`. Its embedded etcd (122 MB of history) could not elect a
leader in time: etcd logged `apply request took too long … 1.88s, expected 100ms`. The
following did **not** help, each tried on a copy of the volume:

- going through 2.5.18 first (2.5.12 → 2.5.18 → 2.6.24);
- raising etcd's `heartbeat-interval` to 500 ms and `election-timeout` to 5 s.

2.6.24 on an empty volume starts in seconds. Every collection is rebuilt from its PDF or its
`merged_chunks.json` anyway (the analyzer is fixed at creation), so a fresh volume loses nothing.

```bash
# 1. back up the 2.5 volume (optional but cheap; it is never modified by 2.6)
docker compose stop vision-rag milvus
mkdir -p backup && docker run --rm -v maryuma-intelligent-chunking_milvus-data:/data:ro \
  -v "$PWD/backup:/backup" alpine tar czf /backup/milvus-data-$(date +%F)-v2.5.tgz -C /data .

# 2. start 2.6 on its own volume (docker-compose.yml already says so)
docker compose up -d --build

# 3. re-ingest each document (next section), then verify with a real query
curl -s -X POST localhost:8001/v1/retrieval/query -H 'content-type: application/json' \
     -d '{"collection_name":"<name>","query":"test","top_k":1}'
```

**Rollback:** in `docker-compose.yml`, set the image back to `v2.5.12` and the volume back to
`milvus-data`, then run `docker compose up -d`. The 2.5 volume is exactly as it was.

For a deployment whose data cannot be re-derived, rehearse the in-place upgrade on a **copy** of
the volume first, and keep the backup: a 2.5.x server cannot start on a volume that 2.6 has
initialized (`panic: dirty recovery info in metastore`).

### Re-ingesting existing collections

Existing collections keep working as legacy (English, read-only). To give a Japanese manual the
Japanese profile, ingest it into a **new** collection name, compare, then switch clients:

```bash
curl -F file=@maryuma-bcl.pdf -F collection_name=maryuma_bcl_ja localhost:8001/v1/documents
# response: {"language": "ja", "collection_name": "maryuma_bcl_ja", ...}
# query both collections with the same questions, switch clients, then drop the old one:
curl -X DELETE localhost:8001/v1/collections/maryuma_bcl
```

Re-ingestion re-runs OCR, Gemini chunking and Cohere embedding for each PDF.

## 4. Tests

```bash
pytest tests                                             # offline, no network
MILVUS_INTEGRATION_URI=http://127.0.0.1:19540 \
  pytest tests/test_vector_db_milvus.py                  # live, against a throwaway Milvus 2.6+
```

The live tests create and drop their own `it_*` collections. Never point them at a Milvus that
holds real data. On Milvus 2.5.x, the English and legacy tests pass and the Japanese tests fail
with the Milvus 2.6 requirement message; this is the intended behavior.

## 5. Adding a language

1. Add its code to `SUPPORTED_LANGUAGES` and a detection rule in `app/retrieval/language.py`.
2. Add an analyzer and a branch to `language_profile` in `app/retrieval/profiles.py`, plus its
   model settings in `ChunkerConfig`.
3. Benchmark it before changing any default.
