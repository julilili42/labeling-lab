# Decisions

This file records meaningful implementation decisions as short ADRs.

## ADR-0001: Keep Auto Labeling independent

Date: 2026-07-09
Status: accepted

Context:
The existing search-engine project already has crawler, labeling, and model
code. The new pipeline should be easy to reason about without inheriting that
architecture.

Decision:
Build Auto Labeling as a separate project.

Consequences:
The pipeline can move faster and stay simpler. Integration with the existing
crawler/model code happens later via exported model artifacts or data files.

## ADR-0002: Use Markdown as project memory

Date: 2026-07-09
Status: accepted

Context:
Implementation decisions and label-policy lessons must survive context-window
loss and be reviewable by humans.

Decision:
Store decisions in `docs/DECISIONS.md` and stable project knowledge in
`docs/KNOWLEDGE.md`.

Consequences:
Knowledge is versionable, diffable, and cheap. If the docs become too large,
add a local retrieval index over the Markdown files later.

## ADR-0003: Use JSONL before a database

Date: 2026-07-09
Status: accepted

Context:
The first version needs inspectable offline data flow, not concurrent editing
or complex querying.

Decision:
Use JSONL files for SERP results, page snapshots, teacher labels, review
batches, and reviewed labels.

Consequences:
Files are easy to inspect, diff, and regenerate. SQLite can be added when data
volume, ad-hoc querying, or multi-user review makes JSONL painful.

## ADR-0004: Treat the LLM as a teacher, not runtime logic

Date: 2026-07-09
Status: accepted

Context:
Live LLM calls inside crawling would be slow, expensive, and hard to reproduce.

Decision:
Use a teacher model only offline to generate labels. Train a small PageVerdict
student model for later runtime use.

Consequences:
Runtime stays fast and cheap. Teacher quality still needs review and holdout
evaluation.

## ADR-0005: Use gray as abstain

Date: 2026-07-09
Status: accepted

Context:
Ambiguous labels are worse than missing labels because they teach the student
model unstable policy.

Decision:
Teacher label `gray` means abstain and is ignored during binary training.

Consequences:
Training data is smaller but cleaner. Borderline cases can still be reviewed
manually or revisited with a better prompt.

## ADR-0006: Build a minimal file-based review UI

Date: 2026-07-09
Status: accepted

Context:
The existing search-engine labeling UI has useful interaction ideas, but it is
tied to SQLite, SERP labeling, crawler candidates, and link review.

Decision:
Build a small local review UI over JSONL review batches instead of copying the
full existing frontend.

Consequences:
The review workflow stays focused on checking teacher labels. If review volume
or ergonomics become painful, the UI can grow or switch to SQLite later.

## ADR-0007: Implement in milestones

Date: 2026-07-09
Status: accepted

Context:
The pipeline touches external search, page fetching, local LLM calls, review,
and training. A one-shot implementation would be harder to test.

Decision:
Implement and test small milestones: data plumbing first, teacher/review next,
training later.

Consequences:
The first usable version may not include every planned command, but each step
has a runnable check before the next layer is added.

## ADR-0008: Train a drop-in PageVerdict-style artifact

Date: 2026-07-09
Status: accepted

Context:
The existing search-engine project already uses a joblib bundle containing a
scikit-learn model over `title`, `url`, `display_url`, and `snippet` text.

Decision:
Train the Auto Labeling student with the same text feature shape and save a
similar joblib bundle.

Consequences:
The generated model can be tested in the existing crawler with minimal glue.
The feature set stays intentionally small until evaluation shows it is not
enough.

## ADR-0009: Support Python 3.10+

Date: 2026-07-09
Status: accepted

Context:
The local shell used for verification runs Python 3.10.

Decision:
Set the project requirement to Python 3.10+.

Consequences:
The code avoids Python 3.11-only features and runs in the current local
environment.

## ADR-0010: Use qwen2.5:7b as the default local teacher

Date: 2026-07-09
Status: accepted

Context:
The target laptop has 16 GB memory. The teacher needs reliable JSON
classification, not a large reasoning model.

Decision:
Use Ollama `qwen2.5:7b` as the default local teacher model.

Consequences:
The model is small enough for local use and large enough for the first
classification pilot. If review shows poor labels, compare one stronger model
on a 100-page sample before changing the whole pipeline.

## ADR-0011: Load SERPER_API_KEY from `.env` as a fallback

Date: 2026-07-09
Status: accepted

Context:
The Codex execution shell may not inherit environment variables set in another
terminal session.

Decision:
Read `SERPER_API_KEY` from the process environment first, then from a
project-local `.env` file.

Consequences:
The user can keep secrets local without passing them through chat. `.env` is
ignored by git.

## ADR-0012: Lower default teacher excerpt to 1200 characters

Date: 2026-07-09
Status: superseded by ADR-0016

Context:
The first 72-page pilot quality looked good, but local Ollama labeling took
roughly 5-6 seconds per page with 2000 visible-text characters.

Decision:
Lower the default teacher text excerpt to 1200 characters and record per-label
teacher timing metrics.

Consequences:
Labeling should become faster. If review quality drops, individual runs can
raise `--max-input-chars` without changing the pipeline.

## ADR-0013: Include input cap in teacher cache keys

Date: 2026-07-09
Status: accepted

Context:
Teacher output can change when the visible-text excerpt length changes.

Decision:
Include `max_input_chars` in the teacher cache key and store it in each label
record.

Consequences:
Benchmarks for 2000-character and 1200-character prompts cannot accidentally
reuse each other's labels. Cache reuse is slightly less broad but correct.

## ADR-0014: Keep qwen2.5:7b as default after 3B benchmark

Date: 2026-07-09
Status: accepted

Context:
`qwen2.5:3b` was tested on the 72 reviewed pilot pages with the same
1200-character input cap. It was faster, but agreement with reviewed labels
dropped from 91.7% for `qwen2.5:7b` to 76.4%. Median runtime dropped from
7.70 seconds to 4.03 seconds.

Decision:
Keep `qwen2.5:7b` as the default local teacher for now.

Consequences:
Labeling remains slower, but teacher quality is better. Use `qwen2.5:3b` only
for rough prefiltering or if a later prompt/model test closes the quality gap.

## ADR-0015: Normalize teacher confidence to 0-1

Date: 2026-07-09
Status: accepted

Context:
The 3B model sometimes returned confidence values like `8.0` when the schema
expects a 0-1 number.

Decision:
Clamp confidence to 0-1 and treat values in 1-10 as a 10-point scale.

Consequences:
Review sorting and low-confidence filtering stay stable even when a local
teacher ignores the exact numeric scale.

## ADR-0016: Keep 2000 characters as the default teacher excerpt

Date: 2026-07-09
Status: accepted

Context:
The 72-page benchmark compared the original `qwen2.5:7b` labels with a
2000-character excerpt against a fresh `qwen2.5:7b` run with a 1200-character
excerpt. The 1200-character run did not materially improve median runtime
(about 7.70 seconds), while agreement with reviewed labels dropped from 94.4%
to 91.7%.

Decision:
Use 2000 visible-text characters as the default teacher input cap.

Consequences:
Teacher calls stay somewhat heavier, but label quality is better on the pilot
set. Speed work should first reduce the number of LLM calls or run batches in
the background, not shrink the prompt by default.

## ADR-0017: Append pipeline outputs incrementally

Date: 2026-07-09
Status: accepted

Context:
Search, fetch, and teacher labeling may run for hours. Network failures or a
stopped process should not lose successful Serper calls, fetched pages, or
completed labels.

Decision:
Append JSONL output incrementally during `search`, `fetch`, and `label`.
Continue with `--resume` by reading existing rows and skipping completed
queries, URLs, or text hashes.

Consequences:
The files may contain raw SERP duplicates, but `fetch` already dedupes by URL.
This keeps the pipeline simple and credit-safe without adding a database.
