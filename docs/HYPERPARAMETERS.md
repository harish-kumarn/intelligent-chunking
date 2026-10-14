# Hyper-parameters — current defaults and recommended values

Every tunable value in the service is listed here: where it lives, its current default, the value we recommend, and the evidence for that value. The evidence comes from a benchmark built for this study (`evaluation/`, reproducible with the commands in [evaluation/README.md](../evaluation/README.md)), the service's own logs (`debug/logs`), and the chunking artifacts in `debug/`.

**How to read the Evidence column**

| Tag | Meaning |
|---|---|
| **M+** | Measured, and the difference is statistically significant: paired test, p < 0.05, 95% bootstrap confidence interval excludes 0 |
| **M=** | Measured, and there is no significant difference. The value is kept for cost, latency or robustness |
| **R** | Reasoned from measured facts (timings, sizes, logs) or from the provider's limits. There was no A/B test |

Quality is nDCG@10 over graded relevance labels unless a row says otherwise. `s@k` means "a correct chunk is in the top k". Every Δ is paired over the same queries: Δ [95% CI], p, and W/L = queries that won / lost.

---

## 1. Changes we recommend

Ordered by impact. Everything not listed here is "keep" — see §3.

| # | Parameter | Now | Recommended | Why (headline evidence) |
|---|---|---|---|---|
| 1 | **Document filter on `/v1/retrieval/query`** (new: `document_ids`, a Milvus `filter` on the existing `document_id` field) | none | Add it. Clients send the manual or product the user is asking about. | In a multi-manual collection, 95–100% of top-1 errors are chunks from **another manual**, which answers the same generic question for a different product. Filtering restores single-manual quality: JA nDCG **0.776 → 0.968** (+0.192 [+0.149, +0.236], p<0.001, W70/L0); EN **0.660 → 0.989** (+0.328, p<0.001, W34/L0). **M+** |
| 2 | `RetrievalConfig.final_top_k` / API default `top_k` | 3 | **5** | Multi-answer recall rises from 0.914 to **0.974** (JA, one manual). At 532 chunks, s@k goes 0.803 → **0.848**. The cost is 2 more chunks (~800 tokens) in the LLM context. Rerank latency does not change, because the window stays at 20. **M+** |
| 3 | `embedding_model` (new **English** collections) | `embed-english-v3.0` | **`embed-v4.0`** | At 410 chunks, dense arm +0.145 [+0.066, +0.228] (p=0.001) and no-rerank hybrid +0.083 (p=0.029). The full reranked path is unchanged (+0.008, n.s.). The old model has a 512-token window: English distractor chunks already reach 551 tokens, and the chunk-size target allows up to 1,000. Query-embed latency +24 ms. **M+** |
| 4 | Cohere client timeout (new; `cohere.Client(timeout=…)`) | SDK default **300 s** | **rerank 5 s, embed 10 s** | Reproduced during this study: a rerank call hung until `httpx.ReadTimeout`. `ReadTimeout` is not in `RETRYABLE_EXCEPTIONS`, so today one query can block for 300 s. Measured rerank p95 is ≤ 1.12 s even at 100 documents. The logs already show a 15.5 s query. **R** (from measurements) |
| 5 | Cohere retry policy on the **query path** (`cohere_retry.MAX_ATTEMPTS`, `_MAX_BACKOFF_SECONDS`) | 7 attempts, backoff up to 60 s (worst case 122 s of sleeps) | **Query path: 2 attempts, ≤ 1 s backoff.** Ingestion keeps 7 / 60 s. | The p95 target is 1.5 s and measured end-to-end p95 is 0.89 s, so any retry sleep over ~0.5 s breaks the target. On failure the reranker already degrades to fused order (recommendation 7 makes that order better). **R** |
| 6 | BM25 `bm25_b` (Milvus sparse index parameter; hard-coded default today) | 0.75 | **0.5 for new Japanese collections**; keep 0.75 for English | JA at 532 chunks: BM25 +0.015 [+0.005, +0.027] (p=0.005); no-rerank hybrid +0.014 (p<0.001, W19/L3). Neutral on one manual and on the full reranked path (+0.004, n.s.). English: no significant effect. **M+** |
| 7 | Fusion used **when the reranker fails** (new: `fallback_fusion_strategy` / `fallback_alpha`) | same as the main path (RRF α 0.5) | **`weighted`, α 0.6** — re-fuse the in-memory arms only when rerank is unavailable | JA without rerank: +0.051 (p<0.001) on one manual, +0.044 (p<0.001) at 532 chunks. English is neutral. The main path must stay RRF: `weighted` is −0.010 (p=0.02) at scale *with* rerank. **M+** |
| 8 | `milvus_timeout` | `None` (no deadline) | **10 s** | Milvus searches take 3.5–4.5 ms p50 and ≤ 10 ms p95. The logs show `ConnectionNotExistException` and DNS failures after Milvus restarts. A deadline turns a hang into a fast 500 that clients can retry. **R** |
| 9 | Gemini `media_resolution` for page images (new; not set today) | model default | **`MEDIA_RESOLUTION_MEDIUM`** | −9% input tokens and −15% thinking tokens per page. No quality change: JA +0.000 (p=0.997), EN +0.002 (p=0.51). Text fidelity 100%. **M=** (same quality, cheaper) |
| 10 | `chunking_cache_ttl` | `3600s`, fixed | **`max(3600, pages × 45) s`** at ingestion start (or refresh the cache between pages) | Measured: 17–22 s per page p50, 35 s max. The cache is created once per document, so a PDF over roughly 170 pages outlives it, and every later page call then references an expired cache and fails. **R** | **Done 2026-10-11:** the chunker refreshes the TTL between pages and recreates an expired cache, so the TTL stays 3600 s.
| 11 | `merger._MAX_FORCED_ABSORPTIONS` | 9 | **2** | In every observed run the forced-absorption path never fired (longest chain 2, no seam-less openers in 242 chunks). If it misfires, 10 chunks (~8k tokens) get glued together — bigger than any embedding window. **R** |
| 12 | Query reformulation (`/v1/retrieval/query-reformulation`, `reformulated_query`) | available; clients may use it | **Don't send rewritten queries into the reranked pipeline.** Use the document filter (1) instead. If the route stays, keep `gpt-oss-20b` at T 0. | Rewriting **hurts** on one manual: −0.022 (p=0.001, W7/L20) with 20b at T 0, −0.032 at T 0.3, −0.056 (p<0.001) with 120b. It also adds 460–770 ms. Its only positive result (EN at scale, +0.048, n.s.) comes from inserting the product name — a crude document filter. **M+** |
| 13 | `min_rerank_score` (new; optional, **off by default**) | none — `top_k` chunks are always returned | Off by default. If enabled: `rerank-v3.5` **0.15**, `rerank-v4.0-fast` **0.48**, `rerank-v4.0-pro` **0.63**, `rerank-english-v3.0` **0.0002** (fragile — its scores cluster at 0 and 1). Each value is the lower of the Japanese and English cut-offs, so both keep ≥ 98% of correct answers. | Keeps ≥ 98% of correct top-1 answers. It rejects **100%** of unanswerable English questions but only **47%** of Japanese ones: in-domain questions still score high against the table of contents, which names the missing sections. Based on 23 unanswerable queries — validate on real traffic before relying on it. **M** (small n) |
| 14 | `RetrievalQueryRequest.top_k` upper bound | `le=200` | **`le=50`** | `top_k` widens the rerank window. A `top_k=200` request reranks 200 documents: 2 Cohere search units, and 600 ms p50 / 1.1 s p95 already at 100 documents. **R** (from measurements) |
| 15 | `_RETRY_AFTER_SECONDS` (HTTP 429 from the ingestion gate) | 5 | **60** | One 33-page manual takes ~10 minutes to chunk (17–22 s per page). Retrying every 5 s means ~120 rejected requests per document. **R** |
| 16 | Gemini `thinking_level` for chunking (new; default HIGH) | HIGH (implicit) | **Expose it; keep HIGH by default. Use LOW for bulk or backfill ingestion.** | LOW is **2.4× faster** (7.3 s vs 17.3 s per page) and uses **−83% thinking tokens** (624 vs 3,656 per page), with text fidelity still 100%. Quality: EN +0.004 and JA at scale −0.001 (both n.s.), but JA on one manual is **−0.007 [−0.014, −0.001], p=0.066**. That is a borderline loss, so this is a cost/quality decision for you. Testing MEDIUM is the next step. **M** |

---

## 2. What the benchmark is

- **Corpus.** The three chunked PDFs in `debug/` are frozen in `evaluation/dataset/corpus/`:
  - BCL200S brush-cutter manual (Japanese; 33 pages, 52 chunks)
  - Maruyama water-pump manual `auger_v1` (English; 5 pages, 10 chunks)
  - `bcl_cutter_sample` (pages 1–14 of the same Japanese manual, chunked in a separate run)
- **Queries.** 256 queries, written by hand from the chunks (`evaluation/dataset/queries.yaml`):
  - 178 Japanese and 55 English answerable queries, plus 23 unanswerable or out-of-domain ones
  - 14 categories: exact term, natural question, paraphrase, number/spec, procedure, part number, spelling variant (キャブレター/キャブレタ), cross-page, multi-answer, cross-lingual, navigation, the 3 real queries found in the logs, unanswerable, out-of-domain
- **Labels are tied to evidence text, not chunk ids.** A chunk is grade 2 if it contains a verbatim answer span and grade 1 if it contains a supporting span, after NFKC normalisation with whitespace and case ignored. Because of this, any re-chunking of the same PDF is graded by the same labels.
- **Label audit.** A pooled audit (top-5 from `rerank-v4.0-pro` plus top-5 from production) was judged by Gemini, which is a different model family from the Cohere rankers. 56 flags were reviewed by hand: 6 were added as full answers, ~30 as partial, and the rest rejected (`evaluation/dataset/audit_additions.yaml`).
- **Scale test.** Collections at production scale were simulated by adding 480 Japanese and 400 English chunks from 12 and 10 *other* outdoor-power-equipment manuals (two of them other brush cutters), written by Gemini in the same chunk format. Each has its own model codes and numbers. Labels stay scoped to the manual the query is about.
- **Faithful replay.**
  - Rankings use the production code (`app.retrieval.fusion.fuse`, `RetrievalConfig`) over cached full-depth arms.
  - The dense arm is exact cosine, which is what Milvus FLAT/COSINE computes.
  - The sparse arm is the live Milvus 2.6 BM25 function with the production analyzer, in throwaway `hp_*` collections.
  - Cohere rerank scores were confirmed independent of the rest of the batch (max drift 2.7e-4), so every candidate window can be replayed offline.
  - Checked against the live `EnhancedCohereMilvusRetriever`: the final top-3 matches on 254 of 256 queries. The 2 differences come from embed-v4.0 returning slightly different vectors on each call (~1e-3 cosine), which production also has.
- **Statistics.** Paired bootstrap (95% CI) and sign-flip permutation p-values over queries. A same-config re-run of the chunker moves nDCG by only ±0.003, so chunking differences above that are real.

**Limitations**

- Two real manuals. The English one is small (10 chunks, 55 queries), and every configuration scores near the ceiling on it, which is why the English decisions lean on the 410-chunk scale set.
- The distractor manuals are synthetic. They are realistic hard negatives, but they are not real manuals.
- Latency was measured from the development host against the Cohere API: 12–40 repetitions per cell.
- OCR engines and auto-grounding could not be A/B tested without OCR ground truth or a use of the generated Q&A. Their rows are **R**.

---

## 3. Full inventory

### 3.1 Query time — `RetrievalConfig` (`app/retrieval/config.py`) and the query API

| Parameter | Default | Recommended | Evidence |
|---|---|---|---|
| `dense_top_k` | 30 | **30** (keep) | At 532 chunks, depth 30 with a window of 20 is the optimum. Deeper hurts: depth 50 → −0.022 (p=0.013) JA and −0.020 (p=0.013) EN, because deeper arm lists push more noise into the fused window. Milvus FLAT search takes 4.5 ms p50. **M=** |
| `sparse_top_k` | 30 | **30** (keep) | Same sweep; both arms were varied together. **M=** |
| `rerank_candidates` | 20 | **20** (keep); **30** if collections grow past ~1,000 chunks | JA 532 chunks: window 10 → −0.039 (p=0.004), 15 → −0.010 (n.s.), 30 → ±0.000. Window recall (a correct chunk reaches the reranker) is 0.955 at 20 and 0.972 at 30. rerank-v3.5 latency: 406 ms at 20, 457 ms at 30. End-to-end p95: 887 ms at 20, 1,001 ms at 30. **M=** |
| `final_top_k` | 3 | **5** | See §1 #2. **M+** |
| `use_hybrid` | true | **true** (keep) | Neutral with rerank (JA hybrid+rerank 0.970 vs dense+rerank 0.968). Costs 3.5 ms. It keeps exact-term and part-number matching when rerank is down. **M=** |
| `use_reranking` | true | **true** (keep) | The largest single contributor: JA 0.885 → 0.970, EN 0.934 → 0.989. Costs ~410 ms, about half of end-to-end latency. **M+** |
| `fusion_strategy` | rrf | **rrf** (keep) | With rerank, every strategy is within ±0.001 on one manual. At scale, `weighted` is −0.010 (p=0.02) for JA. The no-rerank case is §1 #7. **M=** |
| `fusion_mode` | client | **client** (keep) | Server-side fusion is just as fast (365 vs 359 ms p50, both including the query embed) and loses per-arm telemetry. **M=** |
| `alpha` | 0.5 | **0.5** (keep) | α 0.6 is −0.025 (p=0.02) for JA and −0.045 (p=0.023) for EN at scale. α 0.3–0.4 is n.s. **M=** |
| `rrf_k` | 60 | **60** (keep) | Flat from 1 to 100 (all Δ ≤ 0.001, n.s.) on every set. **M=** |
| `dbsf_sigma` | 3.0 | **3.0** (keep) | Only used by `dbsf`, which never beat RRF with rerank. **M=** |
| `drop_ratio_search` | 0.0 | **0.0** (keep) | 0.1–0.3 changes no ranking: queries are short, so no low-weight terms get dropped. **M=** |
| `executor_workers` | 8 | **8** (keep) | Per collection. Each query holds at most 2 worker threads at once (dense ‖ sparse, then rerank), so 8 serves 4 concurrent queries per collection before queueing. Raise it only if per-collection concurrency exceeds that. **R** |
| API `top_k` bound | `le=200` | **`le=50`** | §1 #14. **R** |
| `document_ids` filter | — | **add** | §1 #1. **M+** |
| `min_rerank_score` | — | **add, off by default** | §1 #13. **M** |
| Fallback fusion on rerank failure | — | **add: weighted α 0.6** | §1 #7. **M+** |

### 3.2 Models and language profiles (`ChunkerConfig`, `app/retrieval/profiles.py`)

| Parameter | Default | Recommended | Evidence |
|---|---|---|---|
| `embedding_model` (EN) | ~~embed-english-v3.0~~ embed-v4.0 (adopted) | **embed-v4.0** | §1 #3. **M+** |
| `reranking_model` (EN) | rerank-english-v3.0 | **rerank-english-v3.0** (keep) | No significant difference: v4.0-fast/pro 1.000 vs 0.991 (+0.009, p=0.5); v3.5 −0.015 (n.s.). If `min_rerank_score` is adopted, consider `rerank-v4.0-fast`: english-v3.0 scores are bimodal, and 2% of correct answers score ≈ 0.000. **M=** |
| `japanese_embedding_model` | embed-v4.0 | **embed-v4.0** (keep) | Dense arm 0.884 vs 0.820 (multilingual-v3.0) and 0.749 (multilingual-light). It matters when rerank is down. **M+** |
| `japanese_reranking_model` | rerank-v3.5 | **rerank-v3.5** (keep) | v4.0-pro +0.008 (p=0.15, n.s.), but +170 ms at 20 docs (576 vs 406 ms). v4.0-fast −0.004 (n.s.). multilingual-v3.0 −0.008 (p≈0.05). **M=** |
| `embedding_batch_size` | 95 | **95** (keep) | The Cohere embed limit is 96 texts per call. **R** |
| embed-v4.0 output dimension (implicit) | 1536 | **1536** (keep) | The full path is unchanged at 1024 and 512. The dense arm at scale drops: −0.014 (p=0.022) at 1024 and −0.029 / −0.091 (p≤0.003) at 512, which hurts outage behaviour. Storage (6 KB per vector) is irrelevant at this scale. **M+** |
| Indexed text = vocabulary + description + content (`app/retrieval/documents.py`) | all three | **all three** (keep) | Content only: −0.028 (p=0.013) full path, JA at scale. Without the description: −0.037 (p=0.037) EN at scale. **M+** |
| `japanese_detection_threshold` | 0.2 | **0.2** (keep) | Japanese PDFs score 0.93 and the English one 0.000. A Japanese PDF would flip only above ~76% English content. Individual pages go as low as 0.08 (parts diagram), which confirms detection should be per document, as it is now. **R** (measured shares) |
| `warm_up_japanese_analyzer` | true | **true** (keep) | Lindera downloads and builds its dictionary on first use (ipadic 45 MB; neologd took 148 s to build during this study). **R** |
| Japanese BM25 analyzer | Lindera ipadic + lowercase | **keep** | `ipadic-neologd` is worse: BM25 −0.017 (p=0.008), and −0.038 (p<0.001) at scale. It also adds 715 MB. POS stop-tags: +0.008 to +0.011 BM25, n.s. `unidic` splits product codes (BCL200SL → b/c/l/2/0/0/s/l). Lindera's `japanese_stop_tags` needs full IPADIC POS paths (`助詞,連体化`, not `助詞`). **M=** |
| English BM25 analyzer | standard + lowercase + English stop words | **keep** | English stemmer: +0.018 / +0.034 BM25, n.s., full path −0.009 (n.s.). Not worth a collection migration (the analyzer is fixed when a collection is created). **M=** |
| BM25 `bm25_k1` (index parameter, not exposed) | 1.2 | **1.2** (keep) | k1 0.6 gives +0.013 no-rerank hybrid at JA scale (p=0.002), but it's n.s. on one manual and on the full path. Not enough to justify a change. **M=** |
| BM25 `bm25_b` (index parameter, not exposed) | 0.75 | **0.5 for Japanese** | §1 #6. **M+** |

### 3.3 Milvus and the write path

| Parameter | Default | Recommended | Evidence |
|---|---|---|---|
| `milvus_index_type` / `milvus_metric_type` | FLAT / COSINE | **FLAT / COSINE** up to ~0.5–1 M vectors. Beyond that, or if dense p95 > 50 ms: **HNSW** `{M: 16, efConstruction: 200}` with `ef: 64–128` | Dense search takes 4.5 ms p50 at 532 vectors; FLAT is exact and cost grows linearly with collection size. Metric COSINE matches Cohere's normalised embeddings. **R** (measured) |
| `milvus_index_params` / `milvus_search_params` | None | **None** (FLAT has none) | **R** |
| `milvus_timeout` | None | **10 s** | §1 #8. **R** |
| `milvus_secure`, `milvus_db_name`, credentials | deployment | deployment | Not tunable for quality. |
| `consistency_level` (`milvus_schema.create_collection`) | Bounded | **Bounded** (keep) | Upserts are followed by `flush` + `load`, so new rows are searchable right away. Strong would only add query latency. **R** |
| `vector_db_writer.DEFAULT_BATCH_SIZE` | 256 | **256** (keep) | A manual is 10–60 chunks; one batch. **R** |
| `MAX_TEXT_BYTES` / `MAX_PK_LEN` | 60,000 / 512 | **keep** | The largest chunk is 2,001 characters. **R** |
| Cohere timeouts / retries | 300 s / 7 attempts | **§1 #4 and #5** | **R** |

### 3.4 Ingestion — vision chunking (`ChunkerConfig`, `app/chunking/`)

| Parameter | Default | Recommended | Evidence |
|---|---|---|---|
| `chunking_model` | gemini-3.5-flash | **keep** | No alternative was tested. **—** |
| Chunk size target (system prompt) | 400–800 tokens, soft ceiling 1,000 | **keep** | Small (150–350): EN −0.024 (p=0.020), JA n.s. Large (800–1,500): n.s. everywhere. The model mostly follows meaning, not the target (median chunk 411 → 413 characters for "large"). **M=** |
| `chunking_temperature` | 1.0 | **1.0** (keep) | Gemini 3 guidance. A same-config re-run moves nDCG by only ±0.003, so the non-determinism doesn't matter. **M=** |
| `thinking_level` (not exposed) | HIGH | **expose; HIGH by default, LOW for bulk** | §1 #16. **M** |
| `media_resolution` (not exposed) | default | **MEDIUM** | §1 #9. **M=** |
| `chunking_max_retries` | 5 | **5** (keep) | 0 retries across 242 page calls (the original runs plus 5 variants). **R** |
| `chunking_retry_delay` | 2.0 s | **2.0 s** (keep) | **R** |
| `chunking_timeout` | 30.0 s | **keep the value; rename it `chunking_max_backoff`.** Separately, add a **120 s request deadline** to each Gemini call. | Despite its name, this value only caps the back-off between retries. No request timeout exists today. Page calls take 35 s at most. **R** |
| `chunking_cache_ttl` | 43200 s (12 h) | **keep** — and refreshed between pages since 2026-10-11 | §1 #10. **Done** |
| `_RENDER_DPI` (`app/utils/pdf_utils.py`) | 200 | **200** (keep) | A4 renders at 1654×2339 (514 KB median). Gemini's image token cost is set by `media_resolution`, not DPI. **R** |
| Neighbour-page context (`split_content`, half a page) | ½ of each neighbouring page | **keep** | Median page is 842 characters, so ~420 characters of context. Continuation detection needs the seam, not the whole page. **R** |
| `merger._VOCAB_CAP` | 7 | **7** (keep) | Matches the prompt's 3–7 range; the observed maximum is 7. **R** |
| `merger._MAX_FORCED_ABSORPTIONS` | 9 | **2** | §1 #11. **R** |
| `chunker._JITTER` | 0.1 | **keep** | **R** |

### 3.5 OCR (`config.yaml`, `app/ingestion/ocr/`)

| Parameter | Default | Recommended | Evidence |
|---|---|---|---|
| `ocr.technique` | mistral | **keep** | No OCR ground truth was available, so this was not A/B tested. Chunking fidelity on the Mistral OCR was 100%. **—** |
| `ocr.mistral.model` | mistral-ocr-latest | **pin a dated model version** | `-latest` can change silently between ingestions of the same collection, which breaks the reproducibility the collection manifest is meant to guarantee. **R** |
| `ocr.azure.model` | prebuilt-layout | **keep** | The only Azure model that returns markdown tables and headings, which the chunker relies on. **R** |
| `ocr.azure.extract_figures` / `figure_dpi` | true / 200 | **keep** | Matches the Mistral path's `pdf_images/` and the page render DPI. **R** |
| `_SIGNED_URL_EXPIRY_HOURS` (Mistral) | 5 | **1** | The OCR call is synchronous and uses the URL within seconds; a shorter-lived URL to the PDF is safer. **R** |
| `_FIGURE_PADDING_PT` (Azure) | 4 | **keep** | **R** |

### 3.6 Vocabulary, reformulation, auto-grounding

| Parameter | Default | Recommended | Evidence |
|---|---|---|---|
| `requery_model` | openai/gpt-oss-20b | **keep** if the route stays (§1 #12) | 120b is worse: −0.056 (p<0.001) and +270 ms. **M+** |
| `requery_temperature` | 0.0 | **0.0** (keep) | T 0.3 is −0.032 vs original (p<0.001), worse than T 0. **M+** |
| `vocab_model` / `vocab_temperature` / `vocab_semantic_dedup` | gpt-oss-120b / 0.0 / false | **keep, dedup off** | The vocabulary index only feeds reformulation, which hurts retrieval (§1 #12). An extra paid LLM pass isn't justified. **R** |
| `autogrounding_model` / `temperature` / `max_retries` | gemini-3.5-flash / 0.2 / 3 | **keep** | **R** |
| `autogrounding_min/max_direct`, `min/max_indirect` | 5–8 each, per page | **3–5 each** | A page averages 1.6 chunks, so 10–16 questions per page mostly paraphrase each other. Fewer, more distinct questions give a cleaner evaluation set. The auto-generated set was not used in this study. **R** |

### 3.7 Server and operations (`ServerSettings`, routes)

| Parameter | Default | Recommended | Evidence |
|---|---|---|---|
| `MAX_UPLOAD_SIZE_BYTES` | 200 MiB | **keep, and add a page limit (~300 pages)** | Ingestion is synchronous at 17–22 s per page. A 200 MiB scan can be thousands of pages (hours in one HTTP request, and past the Gemini cache TTL). Page count, not bytes, is the real limit. **R** |
| `_RETRY_AFTER_SECONDS` | 5 | **60** | §1 #15. **R** |
| S3 `_DEFAULT_RETENTION_COUNT` / `_DEFAULT_MAX_CONCURRENCY` | 2 / 8 | **keep** | **R** |
| Log rotation (`_LOG_FILE_MAX_BYTES`, backups) | 5 MiB × 5 | **keep** | 743 log lines in 5 days. **R** |
| `LOG_LEVEL` | INFO | **keep** | **R** |

---

## 4. Problems found along the way

These aren't hyper-parameters, but they showed up in the data.

1. **Swagger placeholder searched as the query.** `debug/logs` shows `Searched with reformulated query: '…' -> 'string'` three times: the Swagger UI example value was sent as `reformulated_query`. Give the field a `None` example, or reject the literal `"string"`.
2. **Unfiltered search over several manuals answers from the wrong product.** Newer rerankers make this *worse* (EN `rerank-v4.0-pro` −0.088, p=0.005), because they correctly judge another manual's answer as relevant. Only the document filter (§1 #1) fixes it.
3. **3 of 5 "outgoing" continuation markers in the BCL200 run had no matching incoming chunk**, so those halves were never merged. This is a chunking-prompt consistency issue, not a parameter.
4. **The reranker only retries rate-limit and availability errors.** `httpx.ReadTimeout` and SSL EOF errors (both seen in the logs) fall through to "keep fused order". That fallback is correct, but only if a timeout bounds the call (§1 #4).
