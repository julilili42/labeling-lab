# Auto Labeling Specification

## Goal

Build an independent offline pipeline that creates training data for a
PageVerdict model.

The model must answer:

1. Is the page mainly English?
2. Is the page related to Tuebingen, Germany?
3. Is the page useful for a general English Tuebingen search engine?

The pipeline uses search-engine results as discovery, fetches page snapshots,
labels them with a teacher model, supports human review in batches, and trains
a small fast student classifier for later crawler use.

This project is independent from the existing search-engine repository.

## Non-Goals

- No live LLM calls inside a crawler.
- No full web crawler in the first version.
- No database in the first version.
- No copied full labeling frontend in the first version.
- No three-class production classifier.
- No automatic trust in teacher labels without review/evaluation.

## Core Idea

Use a teacher-student loop:

```text
queries.txt
  -> Serper search results
  -> URL dedupe
  -> fetch page snapshots
  -> teacher labels via local LLM/Ollama or cloud model
  -> review batches
  -> train binary PageVerdict student model
  -> later: use student during crawling
  -> selectively teacher-label new uncertain crawl pages
```

The teacher may output `positive`, `negative`, or `gray`.

Only `positive` and `negative` are used for training. `gray` means abstain.

## Label Definition

### Positive

A page is positive if all are true:

- mainly English
- related to Tuebingen, Germany
- useful for a general English Tuebingen search engine

Examples:

- city overview
- tourism, attractions, museums, events
- transport, accommodation, restaurants, practical local information
- student life, university overview, degree/admission overview
- hospital/clinic overview pages useful to general users
- jobs, public services, official institution overview pages

### Negative

A page is negative if any are true:

- not mainly English
- not related to Tuebingen, Germany
- narrow academic or research-only content
- publication page, paper, dataset, conference paper, lab result
- individual researcher profile mainly about specialist work
- technical project page useful only to specialists
- generic Germany page without meaningful Tuebingen focus
- spam, login, search, cookie, media file, or broken page

### Gray

Use gray if the page is mixed or ambiguous:

- department/institute landing page with unclear general usefulness
- research institution page that also contains practical visitor/general info
- page has too little extracted text
- teacher confidence is low
- JSON output is invalid after one retry

Gray labels are not training labels.

## Data Files

Use JSONL. One JSON object per line.

```text
data/
  queries.txt
  serp_results.jsonl
  page_snapshots.jsonl
  teacher_labels.raw.jsonl
  review_batches/
    batch-0001.jsonl
    batch-0002.jsonl
  labels.reviewed.jsonl
  eval_holdout.jsonl
  models/
    page_verdict.joblib
    metrics.json
docs/
  DECISIONS.md
  KNOWLEDGE.md
```

## Project Knowledge

Keep project knowledge in Markdown first.

Required files:

- `docs/DECISIONS.md`: every meaningful implementation decision.
- `docs/KNOWLEDGE.md`: stable domain knowledge, label policy notes, pitfalls,
  and lessons learned from review batches.

Use Architecture Decision Records (ADRs) for decisions. Keep each entry short:

```text
## ADR-0001: Use JSONL before a database

Date: 2026-07-09
Status: accepted

Context:
We need an inspectable offline labeling pipeline.

Decision:
Use JSONL files for v1.

Consequences:
Easy to diff and review. May need SQLite when data volume or multi-user review
becomes painful.
```

Do not try to store "LLM knowledge" inside model weights. For this project,
"knowledge" means explicit, version-controlled Markdown plus generated data.

If Markdown gets too large to browse manually, add a simple local retrieval
index later:

```text
docs/*.md -> chunks -> embeddings -> local vector index
```

That is retrieval-augmented generation (RAG), not model quantization.
Quantization is mainly for compressing model weights so a local LLM can run on
limited hardware.

## Query File

`data/queries.txt` contains one query per line.

Allow optional result count override:

```text
Tuebingen tourism | 50
University of Tuebingen | 50
Tuebingen research group | 20
Tuebingen publications | 20
Reutlingen tourism | 20
Heidelberg university research group | 20
```

Default result count: `20`.

Hard maximum per query: `100`.

Start with 100-200 queries, not thousands.

## Search Collection

Use Serper Search API for discovery.

Input:

- query
- result count

Output record:

```json
{
  "query": "Tuebingen tourism",
  "rank": 1,
  "title": "Visit Tuebingen",
  "url": "https://example.com/tuebingen",
  "display_url": "example.com",
  "snippet": "Things to do in Tuebingen...",
  "source": "serper",
  "fetched_at": "2026-07-09T12:00:00Z"
}
```

Implementation notes:

- Serper commonly returns 10 organic results per page.
- Top 20 means two Serper pages/calls per query.
- Preserve every query that discovered a URL.
- Dedupe URLs before page fetching.

## URL Dedupe

Normalize URLs before dedupe:

- lowercase scheme and host
- remove `www.`
- remove fragments
- remove known tracking params: `utm_*`, `fbclid`, `gclid`
- trim trailing slash except root

Keep:

- normalized URL
- original URLs
- discovery queries
- best/min rank

## Page Fetching

Fetch only deduped SERP URLs in v1.

Do not crawl links from fetched pages in v1.

Snapshot fields:

```json
{
  "id": "sha256(content identity)",
  "url": "https://example.com/tuebingen",
  "normalized_url": "https://example.com/tuebingen",
  "host": "example.com",
  "status_code": 200,
  "content_type": "text/html; charset=utf-8",
  "title": "Visit Tuebingen",
  "description": "Official tourism information...",
  "h1": "Visit Tuebingen",
  "h2": ["Things to do", "Getting there"],
  "text": "Visible extracted page text...",
  "text_hash": "sha256(normalized visible text)",
  "discovery_queries": ["Tuebingen tourism"],
  "fetched_at": "2026-07-09T12:00:00Z"
}
```

Fetching rules:

- timeout: 15 seconds
- max response body: 2 MB
- HTML only
- skip PDFs/media in v1
- respect obvious non-HTML content types
- save failed fetches with reason, but do not label them

## Text Extraction

Extract visible text only.

For teacher input, cap text:

- title
- URL
- description
- h1
- h2 list
- first 2000 chars of visible text by default

Do not send full HTML to the teacher.

## Text Dedupe

Before teacher labeling:

- dedupe by `text_hash`
- keep all URLs/discovery queries pointing to same text
- label each unique text once

## Teacher Labeling

Teacher can be:

- local Ollama model
- cloud model
- mock teacher for testing

Teacher command must support:

```text
--limit N
--batch-size N
--dry-run
--resume
--prompt-version VERSION
--max-input-chars N
--teacher ollama|openai|mock
--model MODEL_NAME
```

Every teacher call must be cached by:

```text
prompt_version + model_name + max_input_chars + snapshot_text_hash
```

No cache hit may call the model again.

## Teacher Prompt

System:

```text
You label web pages for an English search engine about Tuebingen, Germany.

Return only valid JSON. Do not include markdown.

Definitions:
- positive: mainly English, related to Tuebingen, and useful for a general
  English Tuebingen search engine.
- negative: non-English, not Tuebingen-related, or narrow specialist content.
- gray: mixed, ambiguous, too little evidence, or low confidence.

Research-only pages are negative. This includes papers, publication lists,
lab news, datasets, technical project pages, clinical trial details, and
individual researcher profiles mainly useful to specialists.

University or research institution overview pages can be positive only if
they are useful to general users, students, visitors, patients, applicants,
or people looking for practical information.
```

User:

```text
Label this page.

URL:
{url}

Title:
{title}

Description:
{description}

H1:
{h1}

H2:
{h2}

Visible text excerpt:
{text_excerpt}

Return JSON with exactly these fields:
{
  "english": boolean,
  "tuebingen_related": boolean,
  "general_search_useful": boolean,
  "research_only": boolean,
  "label": "positive" | "negative" | "gray",
  "confidence": number,
  "reason": string
}
```

Post-processing:

- invalid JSON: retry once
- still invalid: label `gray`
- confidence below `0.70`: force `gray`
- `english=false`: force `negative`
- `tuebingen_related=false`: force `negative`
- `research_only=true` and `general_search_useful=false`: force `negative`

## Review Batches

Create review batches of 500 labels.

Batch file contains teacher output plus enough context for review.

Sort review priority:

1. invalid/retried outputs
2. gray
3. low confidence
4. research_only=false but URL/text contains research terms
5. research_only=true but teacher label is positive
6. random sample of positives
7. random sample of negatives

Research terms:

```text
research
publication
publications
paper
papers
conference
dataset
laboratory
lab
group
project
clinical trial
study
studies
doi
arxiv
```

Review output may override:

- label
- reason
- notes

## Review UI

Use a minimal local file-based review UI.

Do not copy the full search-engine labeling frontend in v1 because it carries
SQLite, link-labeling, and crawler-specific complexity.

The review UI should:

- load one review-batch JSONL file
- show page URL, title, snippet/text excerpt, teacher label, confidence, reason
- offer buttons for `positive`, `negative`, `gray`
- save overrides to `data/labels.reviewed.jsonl`
- keep the raw teacher label in the reviewed record
- avoid any database

## Manual Holdout

Maintain `data/eval_holdout.jsonl`.

Rules:

- 200-300 manually checked pages minimum
- never generated by teacher
- never used for training
- host-diverse
- includes known hard cases:
  - English narrow research pages
  - English general Tuebingen pages
  - German Tuebingen pages
  - other German cities
  - university overview pages
  - institute pages

This holdout is the only trusted evaluation source.

## Training

Train binary PageVerdict student.

Input features:

- title
- URL
- display URL / host
- description
- h1/h2
- text excerpt

Baseline model:

- TF-IDF word ngrams
- TF-IDF char ngrams
- Logistic Regression

Training labels:

- positive => positive
- negative => negative
- gray => ignored

Evaluation:

- random split is not enough
- use host-grouped split
- report metrics on manual holdout

Required metrics:

```json
{
  "accuracy": 0.0,
  "positive_precision": 0.0,
  "positive_recall": 0.0,
  "negative_precision": 0.0,
  "negative_recall": 0.0,
  "roc_auc": 0.0,
  "pr_auc": 0.0,
  "confusion_matrix": []
}
```

Priority:

- high negative recall for research-only pages
- acceptable positive recall for general Tuebingen pages

## Iteration After First Crawl

After PageVerdict v1 exists:

1. Run a focused crawl using the student model.
2. Save new page snapshots.
3. Select pages for teacher labeling:
   - student score in uncertainty band, e.g. 0.35-0.75
   - new hosts
   - research terms present
   - random control sample
4. Teacher-label only selected pages.
5. Review batch.
6. Retrain PageVerdict v2.

Do not teacher-label every crawled page.

## Cost and Runtime Guards

All expensive commands must support:

```text
--limit
--dry-run
--resume
```

Cloud teacher additionally supports:

```text
--max-cost-eur
```

Local teacher additionally supports:

```text
--max-runtime-minutes
```

Default behavior should be safe:

- default `--limit 100`
- no unbounded API loops
- print estimated calls before running
- require explicit `--yes` for runs above 500 teacher calls

## CLI Shape

Proposed commands:

```bash
auto-label init
auto-label search --queries data/queries.txt --out data/serp_results.jsonl
auto-label fetch --in data/serp_results.jsonl --out data/page_snapshots.jsonl
auto-label label --teacher ollama --model qwen2.5:7b --limit 100
auto-label make-review-batch --batch-size 500
auto-label train --labels data/labels.reviewed.jsonl
auto-label evaluate --model data/models/page_verdict.joblib --holdout data/eval_holdout.jsonl
```

## Implementation Order

1. Create project skeleton.
2. Create `docs/DECISIONS.md` and `docs/KNOWLEDGE.md`.
3. Add JSONL helpers.
4. Add query parser.
5. Add Serper search collector with dry-run.
6. Add URL normalization and dedupe.
7. Add fetcher and text extractor.
8. Add teacher interface with mock teacher first.
9. Add Ollama teacher.
10. Add label post-processing.
11. Add review-batch writer.
12. Add minimal file-based review UI.
13. Add training baseline.
14. Add evaluation on manual holdout.

Whenever an implementation choice is made, append a short ADR to
`docs/DECISIONS.md` before or in the same change.

## First Pilot

Use:

```text
50 queries
top 20 results
limit 100 teacher labels
manual inspect 100
fix prompt
limit 500
manual inspect priority cases
then run 3000-page label set
```

Do not run 3000 labels before the first 100 have been inspected.

## Acceptance Criteria

The project is ready for first real use when:

- search collection can produce SERP JSONL
- fetch step can produce page snapshots
- teacher step can label 100 pages and resume safely
- invalid teacher outputs do not enter training as positive/negative
- review batches are written
- training ignores gray
- evaluation reports host-split and holdout metrics
- all expensive commands have `--limit`, `--dry-run`, and `--resume`

## Implementation Prompt For Codex

```text
Build the independent "Auto Labeling" project from SPEC.md.

Use the simplest working Python implementation. Prefer JSONL files over a
database. Do not add a web UI. Do not integrate with the existing search-engine
repo. Implement the CLI commands listed in the spec.

Record every meaningful implementation decision in docs/DECISIONS.md as a
short ADR. Keep durable project knowledge and label-policy lessons in
docs/KNOWLEDGE.md.

Start with:
1. project skeleton with pyproject.toml
2. docs/DECISIONS.md and docs/KNOWLEDGE.md
3. JSONL helpers
4. query parsing
5. Serper search collection
6. URL dedupe
7. page fetching and text extraction
8. mock teacher
9. review batch writer
10. minimal file-based review UI

Only after those work, add Ollama teacher and training.

Every command that can call an external service or model must support
--limit, --dry-run, and --resume. Default limit is 100. Never run an unbounded
labeling job.

Search, fetch, and label must append successful JSONL rows incrementally so an
interrupted run can continue with `--resume` without losing successful external
calls.

Default local teacher model:

```text
qwen2.5:7b
```

It fits the local 16 GB Mac target better than larger models and supports the
JSON classification task well enough for the first pilot.

Default teacher input cap:

```text
2000 visible-text characters
```

The 72-page benchmark showed no meaningful median-speed win from lowering the
cap to 1200, but agreement with reviewed labels dropped from 94.4% to 91.7%.
Keep 2000 as the default and lower `--max-input-chars` only for throwaway
speed tests.

Keep the code boring. No database, no async framework, no web app, no crawler
link expansion in v1.
```
