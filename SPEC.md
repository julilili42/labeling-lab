# Training methodology

## Objective

Train two binary student models:

- PageVerdict decides whether a fetched page belongs in the English Tuebingen index.
- LinkVerdict estimates before fetching whether a discovered link leads to such a page.

The release is reproducible from `training.toml`. Training data, models, and
benchmark outputs are content-hashed in `data/release/manifest.json`.

## Label policy

Every dataset uses exactly one frozen prompt version and accepts only
`positive` or `negative`. `gray`, failed labels, other prompt versions, and
records without their required snapshot are excluded. Teacher, model, prompt
version, and source file remain in the canonical dataset.

Teacher labels are weak supervision, not human ground truth. A release may be
used as a teacher-matching baseline. Claims about real search quality require a
separate human-reviewed benchmark sampled before inspecting model errors.

The current LinkVerdict release uses only binary human reviews. Earlier teacher
runs remain archived but do not enter training; repeated release audits showed
that their agreement with another prompt did not predict human search-quality
judgements.

Human-reviewed PageVerdict hard negatives are marked `training_only`. They are
added only to train and never change the frozen validation or test partitions.

## Leakage control

Preparation happens before splitting:

1. Join page labels to fetched snapshots.
2. Reconstruct the exact runtime feature representation.
3. Remove duplicate URL/context identities.
4. Remove exact duplicate page contents.
5. Drop every identity or content hash with conflicting labels.
6. Split with `StratifiedGroupKFold`, grouped by destination host.
7. Add training-only examples whose hosts do not occur in validation or test.

The resulting train, validation, and test hosts are disjoint. The approximate
split is 64/16/20. The split and every input hash are materialized so later
runs cannot silently move examples between partitions.

Grouping prevents templates and repeated pages from one website appearing in
both training and evaluation. This follows the leakage controls advocated by
Kapoor and Narayanan, *Leakage and the Reproducibility Crisis in ML-based
Science* (2023), https://doi.org/10.1016/j.patter.2023.100804.

## Features and model

Both tasks use the same intentionally small baseline family:

- word TF-IDF with 1-2 grams;
- character TF-IDF with 3-5 grams;
- class-balanced logistic regression.

These models are fast, inspectable, deterministic, and match the online
search-engine feature contract. Page training uses metadata plus the first
3,000 visible-text characters exactly as runtime does. It never substitutes a
body excerpt for the metadata snippet.

The regularization value is selected by five-fold stratified group
cross-validation using average precision on the training partition only. The
selected model is fitted on train, and the binary threshold maximising F1 is
chosen on validation. The release artifact remains the exact train-only model
used for threshold selection, so the sealed test evaluates an unchanged model.
The test split is not read by training.

An operational threshold may instead be frozen from a separately reviewed
development set. Its value, selection rule, and development-file hash must be
declared in `training.toml`; a new untouched holdout is then required.

Keeping offline and online representations identical addresses training-serving
skew, one of the failure modes described by Sculley et al., *Hidden Technical
Debt in Machine Learning Systems* (2015),
https://proceedings.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.

## Benchmark contract

`train-models benchmark` evaluates only the frozen test split and writes:

- `report.json`: release hashes, support, groups, confusion matrices, and metrics;
- `metrics.csv`: tidy scalar metrics with 95% confidence intervals for report tables;
- `predictions.csv`: one row per test example for error analysis;
- `thresholds.csv`: precision/recall/F1 for thresholds 0.00-1.00;
- `calibration.csv`: equal-frequency probability calibration bins.

Reported metrics are accuracy, balanced accuracy, precision, recall, F1,
Matthews correlation, ROC-AUC, average precision, Brier score, and log loss.
Precision-recall results are primary because positives are the minority class;
see Davis and Goadrich, *The Relationship Between Precision-Recall and ROC
Curves* (2006), https://doi.org/10.1145/1143844.1143874.

Confidence intervals use 1,000 bootstrap samples with host, not page, as the
resampling unit. This preserves the dependence between pages from one site.

The benchmark must not be used repeatedly to tune features, hyperparameters,
or thresholds. Such a change requires a new release dataset and a newly frozen
test set.
