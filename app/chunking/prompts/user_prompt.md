STRICT INSTRUCTION: Answer ONLY using the facts explicitly visible in the provided image, USER INPUT and cached content. Do NOT use outside pre-trained knowledge or assume unlisted specifications.

USER INPUT:
<input_data>

<current_page page_number="{page_number}">
{page_text}
</current_page>

<previous_page page_number="{previous_page_number}">
{previous_page_text}
</previous_page>

<next_page page_number="{next_page_number}">
{next_page_text}
</next_page>

</input_data>

<execution_checklist>
Reason privately through these in order, then emit the JSON once.
1. **Boilerplate** — identify by position + repetition; remove from working text; record in metadata (Stage 0).
2. **OCR** — apply ≥0.70 corrections to the working copy; record each with context (Stage 1).
3. **Continuity detection** — inspect both page seams; decide incoming/outgoing/bidirectional using the decision table (Stage 3).
4. **Chunk** — strip image-markdown (remember positions); two-pass; anti-fragmentation strict; size a guide; mark continuation types and group ids per the Stage 3 table (Stage 2, 3).
5. **Images** — full cross-match per image via the matching chain; assign by tier within specificity bar and Tier-3 cap; fallbacks only if nothing resolves; every path assigned at least once with its per-image association label (Stage 4).
6. **Meta** — per chunk: description (≤20 words) + vocabulary (3–7); page level: boilerplate, OCR corrections, continuation flags (Stage 5, schema).
7. **Verify before emitting:**
- Valid JSON: no raw newlines inside strings (use `\n`), backslashes doubled, no trailing commas.
- No chunk `text` contains previous-page, next-page, or boilerplate characters.
- Only the first chunk may be incoming/bidirectional; only the last may be outgoing/bidirectional; group ids follow the `page_{page_number}_to_{next_page_number}` format.
- `content_continuation` flags agree with the first/last chunks' continuation types.
- Every image path on the page appears in at least one chunk; every emitted path exists on the page character-for-character.
- Every chunk has all fields: `chunk_id`, `text`, `images`, `chunk_description`, `continuation_type`, `incoming_group_id`, `outgoing_group_id`, `oversized`, `index_concept_vocabulary`.
</execution_checklist>

<critical_rules>
NEVER include previous-page or next-page text in any chunk's `text`.
NEVER include boilerplate in any chunk's `text`.
NEVER paraphrase, summarize, reorder, or invent content.
NEVER fracture a unified idea for size or structure; mark `oversized` instead.
NEVER invent, rename, or modify image paths.
NEVER apply an OCR correction below 0.70 confidence.
ALWAYS assign every image path on the page to at least one chunk.
ALWAYS keep `content_continuation` flags consistent with the first/last chunks' continuation types.
ALWAYS emit strict-parseable JSON only — no preamble, no fences, no trailing text.
</critical_rules>

Begin your JSON response with:
{{
