# Labeling Lab

Offline collection, review, training, and evaluation for PageVerdict and
LinkVerdict.

## Method

PageVerdict combines Qwen-generated weak labels using the frozen
[`page-v4` prompt](src/auto_labeling/prompts/page-v4.txt) with human reviews;
LinkVerdict uses binary human judgments only. Ambiguous (`gray`), failed, and
conflicting labels are excluded, and reviewed PageVerdict labels override the
teacher.

Exact duplicate URLs and contents are removed before train, validation, and
test data are split by destination host. Regularization is selected with
grouped cross-validation on training data, the decision threshold is selected
on validation data, and the frozen test split is used only for evaluation.
Human-reviewed hard negatives marked `training_only` can enter training but
never validation or test.

## Final artifacts

[`data/final/`](data/final/) contains the deployed models and the manually
labeled evaluation evidence used in the report. Raw pages and historical
experiments are intentionally omitted.

## Report references

Tables:

- Table 2 – Verdict benchmarks:
  [PageVerdict metrics](data/final/evaluation/page/metrics.json),
  [human labels](data/final/evaluation/page/labels.jsonl), and
  [LinkVerdict metrics](data/final/evaluation/link/metrics.csv)

Figures:

- Figure 1 – Verdict training:
  [training configuration](training.toml),
  [PageVerdict model](data/final/models/page_verdict.joblib), and
  [LinkVerdict model](data/final/models/link_verdict.joblib)

## Use

```bash
uv sync
uv run labeling-lab
```

Open <http://127.0.0.1:8010> to review page, link, or JSONL candidates.

To collect new data, set `SERPER_API_KEY` and run:

```bash
uv run auto-label search --resume
uv run auto-label fetch --resume
uv run auto-label label --resume
```

PageVerdict teacher labels use local Ollama with `qwen2.5:7b`. Configure local training
inputs in [`training.toml`](training.toml), then run:

```bash
uv run train-models run
```
