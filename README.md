# Labeling Lab

> Offline data collection, review, training, and evaluation for PageVerdict and
> LinkVerdict.

## Components

- `auto-label` — collects search results, fetches page snapshots, and creates
  versioned teacher labels
- `labeling-lab` — serves the local manual-review UI
- `train-models` — builds leakage-safe datasets, trains both verdict models,
  and writes the central benchmark
- [`training.toml`](training.toml) — freezes inputs, prompt versions, random
  seed, and release location

## Quickstart

Install dependencies:

```bash
uv sync
```

Train and evaluate the configured release:

```bash
uv run train-models run
```

### Manual review

```bash
uv run labeling-lab
```

Open <http://127.0.0.1:8010>. The UI reviews page candidates, link candidates,
and JSONL teacher batches. Ratings are appended to
`data/labels.reviewed.jsonl`, so interrupted batches can be resumed.

### Data collection

Set `SERPER_API_KEY` in the environment or `.env`, then run the required steps:

```bash
uv run auto-label search --resume
uv run auto-label fetch --resume
uv run auto-label label --resume
```

Teacher labeling uses local Ollama with `qwen2.5:7b` by default. See
`uv run auto-label --help` for link labeling and file options.

## Release

`train-models run` creates the configured release under `data/release/`:

```text
datasets/{page,link}/{train,validation,test}.jsonl
models/{page,link}_verdict.joblib
manifest.json
training.json
benchmark/{report.json,metrics.csv,predictions.csv,thresholds.csv,calibration.csv}
```

The `.joblib` files implement the search-engine runtime feature contract. Copy
them only after reviewing the benchmark.

## Commands

```bash
uv run train-models prepare   # validate, deduplicate, and freeze splits
uv run train-models train     # train without reading the test split
uv run train-models benchmark # evaluate both frozen models on test once
uv run train-models run       # all three steps
```

Method, leakage controls, and benchmark semantics are documented in
[`SPEC.md`](SPEC.md). Stable decisions and label policy live in
[`docs/DECISIONS.md`](docs/DECISIONS.md) and
[`docs/KNOWLEDGE.md`](docs/KNOWLEDGE.md).
