# Knowledge

Stable project knowledge, label policy notes, pitfalls, and review lessons.

## Labeling Policy

The PageVerdict student should answer three questions:

1. Is the page mainly English?
2. Is the page related to Tuebingen, Germany?
3. Is the page useful for a general English Tuebingen search engine?

Only pages that satisfy all three should become positive.

## Research Drift

Research-only pages are negative even when they are English and Tuebingen
related.

Examples:

- papers
- publication lists
- datasets
- lab news
- narrow technical project pages
- individual researcher profiles mainly useful to specialists

University or research institution overview pages can be positive when they
are useful to general users, students, visitors, patients, applicants, or
people looking for practical information.

## Teacher Quality

Do not run thousands of labels before checking a 100-page pilot.

Priority cases for manual review:

- `gray`
- low confidence
- research pages labeled positive
- non-English pages labeled positive
- pages with invalid or repaired JSON

## Local Teacher

Default model:

```text
qwen2.5:7b
```

Installed via Ollama. Verified locally with JSON mode on 2026-07-09.

## Pilot 0001

Date: 2026-07-09

Input:

- 10 starter queries
- 20 Serper results per query
- 168 unique SERP URLs
- 100 URL fetch limit

Output:

- 91 page snapshots
- 73 fetchable HTML pages
- 72 teacher labels
- label distribution: 59 positive, 12 negative, 1 gray
- review batch: `data/review_batches/batch-0001.jsonl`

Notes:

- The first 10 starter queries are mostly core-positive queries, so the
  positive-heavy label distribution is expected.
- `Reddit - Please wait for verification` was labeled positive and should be
  checked manually; blocked/interstitial pages need review attention.
- No pages were marked `research_only` in this pilot because the first 10
  queries did not include the research-drift section.
- The original raw labels did not store exact duration metrics. Sequential
  `labeled_at` deltas estimate about 7.77s median per page with `qwen2.5:7b`
  and a 2000-character visible-text excerpt.

## Benchmark 0001

Date: 2026-07-09

Reference: `data/labels.reviewed.jsonl`, 72 reviewed pilot labels.

Results:

- `qwen2.5:7b`, 2000 chars, original raw run: 94.4% agreement, 4 mismatches,
  about 7.77s median duration estimated from label timestamps.
- `qwen2.5:7b`, 1200 chars, fresh timed run: 91.7% agreement, 6 mismatches,
  7.70s median duration.
- `qwen2.5:3b`, 1200 chars, fresh timed run: 76.4% agreement, 17 mismatches,
  4.03s median duration.

Conclusion:

- Keep `qwen2.5:7b` as the default teacher.
- Keep 2000 visible-text characters as the default input cap.
- Use `qwen2.5:3b` only for rough triage unless a later prompt/model benchmark
  closes the quality gap.

## Query List V2

Date: 2026-07-09

Current query file: `data/queries.txt`

Coverage:

- 74 total queries.
- 30 core-positive Tuebingen queries.
- 20 research-drift trap queries.
- 24 non-English, nearby-city, or general-negative queries.
- 1480 requested SERP results at 20 results per query.
- 148 Serper calls for a full run, because the pipeline fetches 10 results per
  Serper call.
- 133 remaining Serper calls when resuming from the existing pilot SERP file.

The next dataset should use the full V2 query list before training a student
model, because Pilot 0001 is too positive-heavy.

## Resume Behavior

Search, fetch, and label now append JSONL rows incrementally.

Resume commands:

```text
PYTHONPATH=src python -m auto_labeling.cli search --resume
PYTHONPATH=src python -m auto_labeling.cli fetch --resume --limit N
PYTHONPATH=src python -m auto_labeling.cli label --resume --limit N
```

If a process stops, rerun the same command with `--resume`. Search resumes by
query rank, fetch resumes by normalized URL, and label resumes by text hash.

## Memory And Retrieval

Use Markdown as the source of truth first.

If Markdown becomes too large, add retrieval:

```text
docs/*.md -> chunks -> embeddings -> local vector index -> retrieved context
```

This is RAG. It does not quantize knowledge into the LLM weights.
